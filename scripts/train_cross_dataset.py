"""Cross-dataset SOH/RUL experiments: NASA + CALCE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.calce_loader import load_calce_cell, list_calce_cells
from src.data.nasa_loader import build_sliding_windows, load_battery_soh, chronological_split
from src.data.rul_dataset import build_rul_windows
from src.models.rul_models import build_rul_model
from src.models.tcn import TCNRegressor
from src.utils.metrics import regression_metrics


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_soh_tcn(
    x_train, y_train, n_features: int, epochs: int, batch_size: int, lr: float, device
) -> TCNRegressor:
    model = TCNRegressor(n_features=n_features).to(device)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def eval_soh(model, x_test, y_test, device) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    return regression_metrics(y_test, pred)


def run_calce_in_dataset(calce_path: Path, window: int, epochs: int, device) -> list[dict]:
    results = []
    for cell in list_calce_cells(calce_path):
        _, _, soh = load_calce_cell(calce_path, cell)
        x, y = build_sliding_windows(soh, window)
        x_tr, x_te, y_tr, y_te = chronological_split(x, y, 0.8)
        model = train_soh_tcn(x_tr, y_tr, 1, epochs, 32, 1e-3, device)
        m = eval_soh(model, x_te, y_te, device)
        full_pred = []
        model.eval()
        with torch.no_grad():
            for t in range(window, len(soh)):
                win = torch.tensor(soh[t - window : t].reshape(1, window, 1), dtype=torch.float32, device=device)
                full_pred.append(model(win).item())
        full_m = regression_metrics(soh[window:], np.array(full_pred))
        results.append(
            {
                "dataset": "CALCE",
                "cell": cell,
                "mode": "in_dataset",
                "test_rmse": m["rmse"],
                "test_mae": m["mae"],
                "full_rmse": full_m["rmse"],
                "full_r2": full_m["r2"],
            }
        )
        print(f"  CALCE {cell} test RMSE={m['rmse']:.4f} full R2={full_m['r2']:.4f}")
    return results


def run_cross_transfer(
    nasa_dir: Path,
    train_batteries: list[str],
    calce_path: Path,
    window: int,
    epochs: int,
    device,
) -> list[dict]:
    """Train SOH TCN on NASA cells, zero-shot test on CALCE."""
    xs, ys = [], []
    for bid in train_batteries:
        _, _, soh = load_battery_soh(nasa_dir / f"{bid}.mat", bid)
        x, y = build_sliding_windows(soh, window)
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    model = train_soh_tcn(x_all, y_all, 1, epochs, 32, 1e-3, device)

    results = []
    for cell in list_calce_cells(calce_path):
        _, _, soh = load_calce_cell(calce_path, cell)
        if len(soh) <= window:
            continue
        preds = []
        with torch.no_grad():
            model.eval()
            for t in range(window, len(soh)):
                win = torch.tensor(soh[t - window : t].reshape(1, window, 1), dtype=torch.float32, device=device)
                preds.append(model(win).item())
        m = regression_metrics(soh[window:], np.array(preds))
        results.append(
            {
                "dataset": "CALCE",
                "cell": cell,
                "mode": "transfer_from_NASA",
                "train_nasa": train_batteries,
                "full_rmse": m["rmse"],
                "full_mae": m["mae"],
                "full_r2": m["r2"],
            }
        )
        print(f"  Transfer->CALCE {cell} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
    return results


def run_calce_rul(calce_path: Path, window: int, horizon: int, epochs: int, device) -> list[dict]:
    results = []
    for model_name in ["tcn", "tcn-attn"]:
        for cell in list_calce_cells(calce_path):
            _, _, soh = load_calce_cell(calce_path, cell)
            try:
                x, y_soh, y_rul = build_rul_windows(soh, window, horizon, eol_threshold=0.7)
            except ValueError:
                continue
            x_tr, x_te, y_tr, y_te = chronological_split(x, y_soh, 0.8)
            n = len(x_tr)
            y_rul_tr, y_rul_te = y_rul[:n], y_rul[n:]
            train_ds = TensorDataset(
                torch.tensor(x_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
                torch.tensor(y_rul_tr, dtype=torch.float32),
            )
            loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            model = build_rul_model(model_name, horizon=horizon).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(epochs):
                model.train()
                for xb, yb_soh, yb_rul in loader:
                    xb = xb.to(device)
                    yb_soh, yb_rul = yb_soh.to(device), yb_rul.to(device)
                    opt.zero_grad()
                    pred_soh, pred_rul = model(xb, yb_soh, 0.5)
                    loss = nn.MSELoss()(pred_soh, yb_soh) + 0.5 * nn.MSELoss()(pred_rul, yb_rul)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                pred_rul = []
                for i in range(len(x_te)):
                    xb = torch.tensor(x_te[i : i + 1], dtype=torch.float32, device=device)
                    _, pr = model(xb, None, 0.0)
                    pred_rul.append(pr.item())
            m = regression_metrics(y_rul_te, np.array(pred_rul))
            results.append(
                {
                    "dataset": "CALCE",
                    "cell": cell,
                    "model": model_name,
                    "rul_rmse": m["rmse"],
                    "rul_mae": m["mae"],
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nasa-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--calce-path", type=Path, default=ROOT / "data" / "calce" / "CALCE.npy")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cross_dataset")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs-soh", type=int, default=200)
    parser.add_argument("--epochs-rul", type=int, default=150)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_res = []
    print("=== CALCE in-dataset SOH ===")
    all_res.extend(run_calce_in_dataset(args.calce_path, args.window, args.epochs_soh, device))

    print("=== NASA->CALCE transfer SOH ===")
    all_res.extend(
        run_cross_transfer(
            args.nasa_dir, ["B0005", "B0006", "B0007"], args.calce_path, args.window, args.epochs_soh, device
        )
    )

    print("=== CALCE RUL (TCN vs TCN-Attn) ===")
    rul_res = run_calce_rul(args.calce_path, args.window, args.horizon, args.epochs_rul, device)
    all_res.extend(rul_res)

    with open(args.out_dir / "cross_dataset_results.json", "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved {args.out_dir / 'cross_dataset_results.json'}")


if __name__ == "__main__":
    main()
