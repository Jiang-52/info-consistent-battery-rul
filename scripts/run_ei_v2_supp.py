"""Supplementary EI experiments: fair multi-feat baselines + NASA LOCO."""

from __future__ import annotations

import argparse
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

from scripts.train_rul import set_seed, train_one_model
from src.data.rul_dataset import load_battery_rul_data
from src.models.rul_models import build_rul_model
from src.models.uncertainty import physics_regularizers
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
BATTERIES = ["B0005", "B0007"]


def _clean(res: dict) -> dict:
    return {k: v for k, v in res.items() if not str(k).startswith("_")}


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["model"], r["battery"], r.get("feature_mode"), r.get("holdout"))
        groups[key].append(r)
    out = []
    for key, items in sorted(groups.items()):
        row = {
            "experiment": key[0],
            "model": key[1],
            "battery": key[2],
            "feature_mode": key[3],
            "holdout": key[4],
            "n_seeds": len(items),
        }
        for m in ("rul_rmse", "rul_mae", "soh_rmse", "soh_mae"):
            vals = [float(it[m]) for it in items]
            row[m] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
        out.append(row)
    return out


def train_loco(
    holdout: str,
    train_cells: list[str],
    data_dir: Path,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    feature_mode: str = "multifeat",
) -> dict:
    """Train Patch-TF on train_cells, evaluate on holdout cell (chrono 80/20 within holdout)."""
    import copy

    set_seed(seed)
    xs, ys, yr = [], [], []
    for bid in train_cells:
        d = load_battery_rul_data(
            data_dir / f"{bid}.mat", bid, window, horizon, feature_mode=feature_mode
        )
        # use first 80% of each source cell for LOCO training (chrono)
        n = int(len(d["x"]) * 0.8)
        xs.append(d["x"][:n])
        ys.append(d["y_soh"][:n])
        yr.append(d["y_rul"][:n])
    x_tr = np.concatenate(xs, axis=0)
    y_soh_tr = np.concatenate(ys, axis=0)
    y_rul_tr = np.concatenate(yr, axis=0)

    d_ho = load_battery_rul_data(
        data_dir / f"{holdout}.mat", holdout, window, horizon, feature_mode=feature_mode
    )
    n_ho = int(len(d_ho["x"]) * 0.8)
    # evaluate on holdout test segment (last 20%)
    x_te = d_ho["x"][n_ho:]
    y_soh_te = d_ho["y_soh"][n_ho:]
    y_rul_te = d_ho["y_rul"][n_ho:]
    # val from end of pooled train
    n_val = max(1, int(round(len(x_tr) * 0.15)))
    x_train, x_val = x_tr[:-n_val], x_tr[-n_val:]
    y_soh_train, y_soh_val = y_soh_tr[:-n_val], y_soh_tr[-n_val:]
    y_rul_train, y_rul_val = y_rul_tr[:-n_val], y_rul_tr[-n_val:]

    n_features = x_train.shape[-1]
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_soh_train, dtype=torch.float32),
            torch.tensor(y_rul_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_soh_val, dtype=torch.float32),
            torch.tensor(y_rul_val, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_te, dtype=torch.float32),
            torch.tensor(y_soh_te, dtype=torch.float32),
            torch.tensor(y_rul_te, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_rul_model("patch-tf", horizon=horizon, n_features=n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_fn = nn.MSELoss()
    rul_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            pred_soh, pred_rul = model(xb, yb_soh, 0.0)
            loss = soh_fn(pred_soh, yb_soh) + 0.5 * rul_fn(pred_rul, yb_rul)
            loss = loss + physics_regularizers(pred_soh)
            loss.backward()
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, _, yb_rul in val_loader:
                _, pr = model(xb.to(device), None, 0.0)
                preds.append(pr.cpu().numpy())
                trues.append(yb_rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(trues)) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 40 == 0:
            print(f"  [loco holdout={holdout}|s{seed}] epoch {epoch}/{epochs} val={val_rmse:.4f} best@{best_epoch}")

    model.load_state_dict(best_state)
    model.eval()
    soh_p, soh_t, rul_p, rul_t = [], [], [], []
    with torch.no_grad():
        for xb, yb_soh, yb_rul in test_loader:
            ps, pr = model(xb.to(device), None, 0.0)
            soh_p.append(ps.cpu().numpy())
            soh_t.append(yb_soh.numpy())
            rul_p.append(pr.cpu().numpy())
            rul_t.append(yb_rul.numpy())
    soh_m = regression_metrics(np.concatenate(soh_t).reshape(-1), np.concatenate(soh_p).reshape(-1))
    rul_m = regression_metrics(np.concatenate(rul_t).reshape(-1), np.concatenate(rul_p).reshape(-1))
    return {
        "experiment": "loco",
        "model": "patch-tf",
        "battery": holdout,
        "holdout": holdout,
        "train_cells": train_cells,
        "feature_mode": feature_mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "rul_rmse": rul_m["rmse"],
        "rul_mae": rul_m["mae"],
        "soh_rmse": soh_m["rmse"],
        "soh_mae": soh_m["mae"],
        "soh_r2": soh_m["r2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "ei_v2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 3
        seeds = [42]
    else:
        seeds = SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    ckpt = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    raw: list[dict] = []

    # Fair multi-feat baselines
    for seed in seeds:
        for bat in BATTERIES:
            for model in ("tcn", "tcn-attn", "lstm"):
                print(f"\n=== [fair_multifeat] {model} {bat} seed={seed} ===")
                res = train_one_model(
                    model_name=model,
                    battery_id=bat,
                    data_dir=args.data_dir,
                    out_dir=ckpt,
                    window_size=args.window_size,
                    horizon=args.horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    tf_ratio=0.5,
                    device=device,
                    seed=seed,
                    feature_mode="multifeat",
                    use_physics=False,
                    save_artifacts=True,
                )
                res["experiment"] = "fair_multifeat"
                raw.append(_clean(res))

    # LOCO among B0005/B0006/B0007 for Patch-TF
    all_cells = ["B0005", "B0006", "B0007"]
    for seed in seeds:
        for holdout in all_cells:
            train_cells = [c for c in all_cells if c != holdout]
            print(f"\n=== [loco] holdout={holdout} train={train_cells} seed={seed} ===")
            res = train_loco(
                holdout,
                train_cells,
                args.data_dir,
                args.window_size,
                args.horizon,
                args.epochs,
                args.batch_size,
                args.lr,
                seed,
                device,
            )
            raw.append(res)

    # merge with existing raw
    prev_path = out_dir / "ei_v2_raw.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        # drop old fair_multifeat/loco if re-running
        prev = [r for r in prev if r.get("experiment") not in {"fair_multifeat", "loco"}]
        merged = prev + raw
    else:
        merged = raw

    agg = aggregate([r for r in merged if r.get("experiment") in {"fair_multifeat", "loco"}])
    (out_dir / "ei_v2_supp_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_supp_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_raw.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    lines = ["# EI v2 Supplementary Experiments", "", "## Fair multi-feat baselines (RUL RMSE)", ""]
    for bat in BATTERIES:
        lines.append(f"### {bat}")
        for model in ("patch-tf", "tcn", "tcn-attn", "lstm"):
            if model == "patch-tf":
                # from main
                main_agg = [
                    r
                    for r in json.loads((out_dir / "ei_v2_agg.json").read_text(encoding="utf-8"))
                    if r["experiment"] == "main"
                    and r["battery"] == bat
                    and r["model"] == "patch-tf"
                    and r["feature_mode"] == "multifeat"
                ]
                if main_agg:
                    r = main_agg[0]
                    lines.append(f"- Patch-TF: {r['rul_rmse']:.2f}±{r['rul_rmse_std']:.2f}")
            else:
                row = next((a for a in agg if a["experiment"] == "fair_multifeat" and a["battery"] == bat and a["model"] == model), None)
                if row:
                    lines.append(f"- {model}: {row['rul_rmse']:.2f}±{row['rul_rmse_std']:.2f}")
        lines.append("")
    lines.append("## LOCO (Patch-TF multi-feat)")
    lines.append("")
    for holdout in all_cells:
        row = next((a for a in agg if a["experiment"] == "loco" and a["holdout"] == holdout), None)
        if row:
            lines.append(f"- Holdout {holdout}: {row['rul_rmse']:.2f}±{row['rul_rmse_std']:.2f}")
    (out_dir / "ei_v2_supp_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved supplementary results to", out_dir)


if __name__ == "__main__":
    main()
