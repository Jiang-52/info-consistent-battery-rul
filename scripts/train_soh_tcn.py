"""Train TCN SOH baseline on NASA battery data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nasa_loader import (
    build_sliding_windows,
    load_battery_soh,
    chronological_split,
)
from src.models.tcn import SimpleTCN, TCNRegressor


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_sliding_window(
    battery_id: str,
    data_dir: Path,
    window_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """TCN: past W-cycle SOH window -> next-cycle SOH (no label leakage)."""
    mat_path = data_dir / f"{battery_id}.mat"
    cycles, capacities, soh = load_battery_soh(mat_path, battery_id)
    x_all, y_all = build_sliding_windows(soh, window_size)
    x_train, x_test, y_train, y_test = chronological_split(x_all, y_all, 0.8)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TCNRegressor(n_features=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={epoch_loss:.6f}")

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)
            targets.append(yb.numpy().reshape(-1))
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    metrics = evaluate(y_true, y_pred)

    # Full-sequence one-step predictions for visualization (teacher-forced history)
    full_preds = []
    with torch.no_grad():
        for t in range(window_size, len(soh)):
            win = torch.tensor(
                soh[t - window_size : t].reshape(1, window_size, 1),
                dtype=torch.float32,
            ).to(device)
            full_preds.append(model(win).item())
    pred_curve = np.full(len(soh), np.nan, dtype=np.float32)
    pred_curve[window_size:] = np.array(full_preds, dtype=np.float32)
    full_metrics = evaluate(soh[window_size:], pred_curve[window_size:])

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"tcn_{battery_id.lower()}.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss_history)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(soh, "k-", label="Actual SOH", linewidth=2)
    axes[1].plot(pred_curve, "r--", label="TCN Prediction", linewidth=1.5)
    split_at = window_size + int(len(x_all) * 0.8)
    axes[1].axvline(split_at, color="gray", linestyle=":", label="Train/Test")
    axes[1].set_title(f"{battery_id} SOH — TCN (window={window_size})")
    axes[1].set_xlabel("Discharge cycle index")
    axes[1].set_ylabel("SOH (normalized capacity)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"soh_{battery_id.lower()}.png", dpi=150)
    plt.close()

    result = {
        "battery": battery_id,
        "mode": "sliding_window_forecast",
        "window_size": window_size,
        "epochs": epochs,
        "test_samples": int(len(y_test)),
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "full_rmse": full_metrics["rmse"],
        "full_mae": full_metrics["mae"],
        "full_r2": full_metrics["r2"],
        # keep flat keys for summary compatibility
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
    }
    with open(out_dir / f"metrics_{battery_id.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def train_simple_curve(
    battery_id: str,
    data_dir: Path,
    epochs: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """
    Simple TCN from Bansilal01 — predicts full SOH curve from cycle index.
    Fast sanity check; metrics computed on full sequence (includes train region).
    """
    mat_path = data_dir / f"{battery_id}.mat"
    cycles, _, soh = load_battery_soh(mat_path, battery_id)
    x = cycles.astype(np.float32)
    x_norm = x / x.max()
    y = soh

    n_train = int(len(x) * 0.8)
    x_train = x_norm[:n_train]
    y_train = y[:n_train]

    model = SimpleTCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(x_train.reshape(1, 1, -1), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train.reshape(1, -1, 1), dtype=torch.float32).to(device)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor)
        loss = criterion(out.view_as(y_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 500 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        x_all = torch.tensor(x_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        y_pred = model(x_all).cpu().numpy().squeeze()

    # Test on last 20% chronologically
    y_true_test = y[n_train:]
    y_pred_test = y_pred[n_train:]
    metrics = evaluate(y_true_test, y_pred_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y, "k-", label="Actual SOH", linewidth=2)
    ax.plot(y_pred, "g--", label="TCN Prediction")
    ax.axvline(n_train, color="gray", linestyle=":", label="Train/Test split")
    ax.set_title(f"{battery_id} SOH — Simple TCN (ref baseline)")
    ax.set_xlabel("Discharge cycle index")
    ax.set_ylabel("SOH")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"soh_simple_{battery_id.lower()}.png", dpi=150)
    plt.close()

    return {
        "battery": battery_id,
        "mode": "simple_curve",
        "epochs": epochs,
        "test_samples": int(len(y_true_test)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN SOH baseline training")
    parser.add_argument("--battery", default="B0005", help="Battery ID, e.g. B0005")
    parser.add_argument("--all", action="store_true", help="Run B0005/B0006/B0007")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--mode",
        choices=["window", "simple"],
        default="window",
        help="window=50-cycle sliding (recommended); simple=ref repo curve fit",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batteries = ["B0005", "B0006", "B0007"] if args.all else [args.battery.upper()]
    all_results = []

    for bid in batteries:
        print(f"\n=== Training {bid} ({args.mode}) ===")
        if args.mode == "window":
            res = train_sliding_window(
                bid,
                args.data_dir,
                args.window_size,
                args.epochs,
                args.batch_size,
                args.lr,
                device,
                args.out_dir,
            )
        else:
            res = train_simple_curve(
                bid, args.data_dir, min(args.epochs, 3000), args.lr, device, args.out_dir
            )
        all_results.append(res)
        print(
            f"  Test  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  R2={res['r2']:.4f}"
        )
        if "full_r2" in res:
            print(
                f"  Full  RMSE={res['full_rmse']:.4f}  MAE={res['full_mae']:.4f}  R2={res['full_r2']:.4f}"
            )

    summary_path = args.out_dir / "metrics_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
