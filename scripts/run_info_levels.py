#!/usr/bin/env python3
"""
Information-level (L0/L1/L2) RUL re-evaluation.

L2 — oracle: cycle_channel = t / T_cell^target  (original paper)
L1 — population prior: t / median(T of sibling NASA cells), never target
L0 — prediction-time: t / max(T of sibling cells) as train-only reference
      (absolute time scale; does not encode target lifetime)

Matched protocol: chronological 80/20, seeds {42,43,44}, W=50, H=5,
MLP + Patch-TF (+ optional LSTM), persistence/linear classical.

Outputs: outputs/info_levels/
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
from src.baselines.classical import evaluate_classical
from src.data.cycle_features import INFO_L0, INFO_L1, INFO_L2, cell_discharge_length
from src.data.nasa_loader import chronological_split
from src.data.rul_dataset import load_battery_rul_data
from src.models.rul_models import build_rul_model
from src.models.uncertainty import physics_regularizers
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
ALL_CELLS = ["B0005", "B0006", "B0007"]
LEVELS = [INFO_L2, INFO_L1, INFO_L0]


def sibling_scales(data_dir: Path, target: str) -> dict[str, float]:
    """Leave-target-out lengths (legacy denom = last discharge index)."""
    lengths = {}
    for bat in ALL_CELLS:
        if bat == target:
            continue
        n = cell_discharge_length(data_dir / f"{bat}.mat", bat)
        lengths[bat] = float(n - 1.0)
    vals = np.asarray(list(lengths.values()), dtype=np.float64)
    return {
        "siblings": lengths,
        "t_med": float(np.median(vals)),
        "t_max": float(np.max(vals)),
        "t_min": float(np.min(vals)),
        # Fixed BMS-style clock: no cell lifetime in the denominator (strict L0).
        "t_clock": 1000.0,
    }


def t_scale_for(level: str, scales: dict, target_last_idx: float) -> float | None:
    level = level.upper()
    if level == INFO_L2:
        return float(target_last_idx)
    if level == INFO_L1:
        return float(scales["t_med"])
    if level == INFO_L0:
        return float(scales["t_clock"])
    raise ValueError(level)


def provenance_matrix() -> list[dict]:
    return [
        {
            "feature": "cap_norm (Q_t/Q_1)",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L0+",
            "note": "cycle-level after discharge t",
        },
        {
            "feature": "mean_V / mean_I / mean_T / std_V",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L0+",
            "note": "history ≤ t",
        },
        {
            "feature": "raw cycle index t",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L0",
            "note": "BMS cycle counter",
        },
        {
            "feature": "t / 1000 (fixed clock)",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L0",
            "note": "BMS-style absolute cycle scale; no cell lifetime",
        },
        {
            "feature": "t / median(T_sibling)",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L1",
            "note": "population prior; leave-target-out (on NASA B0005–7 may ≈ L2)",
        },
        {
            "feature": "t / T_cell^target (cycle_norm)",
            "prediction_time_available": False,
            "target_future_dependent": True,
            "level": "L2",
            "note": "oracle relative life progress",
        },
        {
            "feature": "RUL label = T_EOL - t",
            "prediction_time_available": False,
            "target_future_dependent": True,
            "level": "label",
            "note": "labels may use future; inputs must not",
        },
        {
            "feature": "scaler",
            "prediction_time_available": True,
            "target_future_dependent": False,
            "level": "L0+",
            "note": "features built per cell; no joint all-cell scaler",
        },
    ]


def train_one(
    model_name: str,
    battery: str,
    info_level: str,
    t_scale: float | None,
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
        data_dir / f"{battery}.mat",
        battery,
        window,
        horizon,
        feature_mode="multifeat",
        info_level=info_level,
        t_scale=t_scale,
    )
    x, y_soh, y_rul = data["x"], data["y_soh"], data["y_rul"]
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
        n_features=int(x.shape[-1]),
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
        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(
                f"  [{model_name}|{battery}|{info_level}|s{seed}] "
                f"ep {epoch}/{epochs} val={val:.3f} best@{best_epoch}"
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
        "experiment": "info_levels",
        "model": model_name,
        "battery": battery,
        "info_level": info_level,
        "t_scale": None if t_scale is None else float(t_scale),
        "target_last_idx": float(data["target_last_idx"]),
        "seed": seed,
        "window": window,
        "horizon": horizon,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "rul_rmse": float(m["rmse"]),
        "rul_mae": float(m["mae"]),
        "rul_r2": float(m["r2"]),
        "n_test": int(len(true)),
    }


def run_classical_for_level(
    battery: str,
    info_level: str,
    t_scale: float | None,
    data_dir: Path,
    window: int,
    horizon: int,
) -> list[dict]:
    data = load_battery_rul_data(
        data_dir / f"{battery}.mat",
        battery,
        window,
        horizon,
        feature_mode="multifeat",
        info_level=info_level,
        t_scale=t_scale,
    )
    x, y_rul = data["x"], data["y_rul"]
    n = int(len(x) * 0.8)
    x_te, y_te = x[n:], y_rul[n:]
    rows = []
    for method in ("persistence", "linear"):
        m = evaluate_classical(x_te, y_te, method=method)
        m.update(
            {
                "experiment": "info_levels_classical",
                "battery": battery,
                "info_level": info_level,
                "t_scale": None if t_scale is None else float(t_scale),
                "seed": 0,
                "window": window,
                "horizon": horizon,
            }
        )
        rows.append(m)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    keys = ("experiment", "model", "battery", "info_level")
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


def write_report(rows: list[dict], classical: list[dict], out_dir: Path) -> None:
    agg = aggregate(rows)
    (out_dir / "info_levels_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "info_levels_classical.json").write_text(
        json.dumps(classical, indent=2), encoding="utf-8"
    )
    lines = [
        "# Information-level RUL RMSE (cycles)\n\n",
        "Protocol: chronological 80/20, W=50, H=5, 3 seeds, matched dual heads.\n\n",
        "## Learned models (mean±std)\n\n",
        "| Model | Battery | L2 (oracle) | L1 (pop. prior) | L0 (pred.-time) |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    pivot: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in agg:
        cell = f"{r['rul_rmse']:.2f}±{r.get('rul_rmse_std', 0):.2f}"
        pivot[(r["model"], r["battery"])][r["info_level"]] = cell
    for model in sorted({r["model"] for r in agg}):
        for bat in sorted({r["battery"] for r in agg}):
            cells = pivot.get((model, bat), {})
            lines.append(
                f"| {model} | {bat} | {cells.get('L2', '-')} | "
                f"{cells.get('L1', '-')} | {cells.get('L0', '-')} |\n"
            )
    lines.append("\n## Classical (seed-invariant; same across levels if SOH-only)\n\n")
    lines.append("| Method | Battery | Level | RUL RMSE |\n|---|---|---|---:|\n")
    for r in classical:
        lines.append(
            f"| {r['model']} | {r['battery']} | {r['info_level']} | {r['rul_rmse']:.2f} |\n"
        )
    (out_dir / "info_levels_report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models", type=str, default="mlp,patch-tf")
    p.add_argument("--batteries", type=str, default="B0005,B0007")
    p.add_argument("--levels", type=str, default="L2,L1,L0")
    args = p.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 5)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batteries = [b.strip() for b in args.batteries.split(",") if b.strip()]
    levels = [x.strip().upper() for x in args.levels.split(",") if x.strip()]
    seeds = SEEDS if not args.smoke else [42]
    if args.smoke:
        batteries = batteries[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "nasa"
    out_dir = ROOT / "outputs" / "info_levels"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "provenance_matrix.json").write_text(
        json.dumps(provenance_matrix(), indent=2), encoding="utf-8"
    )
    scale_log = {bat: sibling_scales(data_dir, bat) for bat in batteries}
    (out_dir / "sibling_scales.json").write_text(
        json.dumps(scale_log, indent=2), encoding="utf-8"
    )
    print("sibling scales:", json.dumps(scale_log, indent=2))
    print(f"device={device} epochs={args.epochs} models={models} levels={levels}")

    rows: list[dict] = []
    classical: list[dict] = []
    for bat in batteries:
        scales = scale_log[bat]
        # probe target last idx once
        probe = load_battery_rul_data(
            data_dir / f"{bat}.mat", bat, args.window, args.horizon, feature_mode="multifeat"
        )
        target_last = float(probe["target_last_idx"])
        for level in levels:
            ts = t_scale_for(level, scales, target_last)
            classical.extend(
                run_classical_for_level(
                    bat, level, ts if level != INFO_L2 else None, data_dir, args.window, args.horizon
                )
            )
            for model_name in models:
                for seed in seeds:
                    print(f"=== {model_name} {bat} {level} seed={seed} t_scale={ts} ===")
                    res = train_one(
                        model_name,
                        bat,
                        level,
                        ts if level != INFO_L2 else None,
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
                    (out_dir / "info_levels_raw.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )

    write_report(rows, classical, out_dir)
    print("done →", out_dir)


if __name__ == "__main__":
    main()
