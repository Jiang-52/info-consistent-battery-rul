#!/usr/bin/env python3
"""
cycle_norm ablation for pre-submission methodology closure.

Compare Full 6-feature inputs vs Full without cycle_norm (cols 0..4)
for MLP and Patch-TF on B0005/B0007, seeds {42,43,44}, matched protocol.

Outputs: outputs/ei_v3_cycle_norm/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import set_seed, _eval_rul_rmse
from src.data.nasa_loader import chronological_split
from src.data.rul_dataset import load_battery_rul_data
from src.models.rul_models import build_rul_model
from src.models.uncertainty import physics_regularizers
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
FEATURE_SETS = {
    "all": [0, 1, 2, 3, 4, 5],
    "no_cycle_norm": [0, 1, 2, 3, 4],
}
MODELS = ["mlp", "patch-tf"]


def _agg(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n_runs"] = len(items)
        for m in ("rul_rmse", "rul_mae", "rul_r2", "best_val_rul_rmse"):
            vals = [float(it[m]) for it in items if m in it and it[m] is not None]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        out.append(row)
    return out


def train_sliced(
    model_name: str,
    battery: str,
    cols: list[int],
    feat_name: str,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    window: int,
    horizon: int,
    device: torch.device,
    patch_size: int = 5,
) -> dict:
    set_seed(seed)
    data = load_battery_rul_data(
        data_dir / f"{battery}.mat", battery, window, horizon, feature_mode="multifeat"
    )
    x = data["x"][:, :, cols].astype(np.float32)
    y_soh, y_rul = data["y_soh"], data["y_rul"]
    x_tr_full, x_te, y_soh_tr_full, y_soh_te = chronological_split(x, y_soh, 0.8)
    y_rul_tr_full = y_rul[: len(x_tr_full)]
    y_rul_te = y_rul[len(x_tr_full) :]
    n = len(x_tr_full)
    n_val = max(1, int(round(n * 0.15)))
    n_tr = max(1, n - n_val)
    x_tr, x_val = x_tr_full[:n_tr], x_tr_full[n_tr:]
    y_soh_tr, y_soh_val = y_soh_tr_full[:n_tr], y_soh_tr_full[n_tr:]
    y_rul_tr, y_rul_val = y_rul_tr_full[:n_tr], y_rul_tr_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(y_soh_tr), torch.tensor(y_rul_tr)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_soh_val), torch.tensor(y_rul_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_te), torch.tensor(y_soh_te), torch.tensor(y_rul_te)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_rul_model(
        model_name,
        horizon=horizon,
        n_features=len(cols),
        window_size=window,
        patch_size=patch_size,
        d_model=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            ps, pr = model(xb, yb_soh, 0.0)
            loss = soh_fn(ps, yb_soh) + 0.5 * rul_fn(pr, yb_rul) + physics_regularizers(ps)
            loss.backward()
            opt.step()
        val = _eval_rul_rmse(model, val_loader, device)
        if val < best_val:
            best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"  [{model_name}|{battery}|{feat_name}|s{seed}] "
                f"epoch {epoch}/{epochs} val_rul_rmse={val:.4f} best@{best_epoch}"
            )
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, _, yb_rul in test_loader:
            _, pr = model(xb.to(device), None, 0.0)
            preds.append(pr.cpu().numpy())
            trues.append(yb_rul.numpy())
    pred = np.concatenate(preds).reshape(-1)
    true = np.concatenate(trues).reshape(-1)
    m = regression_metrics(true, pred)
    return {
        "experiment": "cycle_norm_ablation",
        "model": model_name,
        "battery": battery,
        "feature_set": feat_name,
        "feature_cols": cols,
        "n_features": len(cols),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "rul_rmse": float(m["rmse"]),
        "rul_mae": float(m["mae"]),
        "rul_r2": float(m["r2"]),
        "n_test": int(len(true)),
    }


def write_report(rows: list[dict], out_dir: Path) -> None:
    agg = _agg(rows, ("experiment", "model", "battery", "feature_set"))
    (out_dir / "cycle_norm_ablation_agg.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    lines = ["# cycle_norm ablation\n\n", "| Model | Features | B0005 | B0007 |\n", "|---|---|---:|---:|\n"]
    # pivot for markdown
    pivot: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in agg:
        key = (r["model"], r["feature_set"])
        cell = f"{r['rul_rmse']:.2f}±{r['rul_rmse_std']:.2f}"
        pivot[key][r["battery"]] = cell
    for model in MODELS:
        for feat in ("all", "no_cycle_norm"):
            cells = pivot.get((model, feat), {})
            lines.append(
                f"| {model} | {feat} | {cells.get('B0005', '-')} | {cells.get('B0007', '-')} |\n"
            )
    (out_dir / "cycle_norm_ablation_report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="mlp,patch-tf",
        help="comma list: mlp,patch-tf",
    )
    parser.add_argument(
        "--skip-all",
        action="store_true",
        help="only train no_cycle_norm (reuse published full numbers in writeup)",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 5)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    feat_items = list(FEATURE_SETS.items())
    if args.skip_all:
        feat_items = [(k, v) for k, v in feat_items if k != "all"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "nasa"
    out_dir = ROOT / "outputs" / "ei_v3_cycle_norm"
    out_dir.mkdir(parents=True, exist_ok=True)
    batteries = ["B0005", "B0007"] if not args.smoke else ["B0005"]
    seeds = SEEDS if not args.smoke else [42]

    print(f"device={device} epochs={args.epochs} models={models}")
    rows: list[dict] = []
    for model_name in models:
        for bat in batteries:
            for feat_name, cols in feat_items:
                for seed in seeds:
                    print(f"=== {model_name} {bat} {feat_name} seed={seed} ===")
                    res = train_sliced(
                        model_name,
                        bat,
                        cols,
                        feat_name,
                        data_dir,
                        args.epochs,
                        args.batch_size,
                        args.lr,
                        seed,
                        args.window,
                        args.horizon,
                        device,
                    )
                    rows.append(res)
                    (out_dir / "cycle_norm_ablation_raw.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )

    write_report(rows, out_dir)
    print("done →", out_dir)


if __name__ == "__main__":
    main()
