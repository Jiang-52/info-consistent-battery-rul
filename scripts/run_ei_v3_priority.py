"""
EI v3 priority experiments — evidence chain for non-OA EI submission.

Priority order (matched protocol first):
  P1 matched_direct   — same input/split/budget; direct dual-head baselines
  P2 classical        — persistence + linear extrapolation
  P3 arch_ablation    — MLP / Vanilla-TF / Patch-TF (+ direct TCN/LSTM/GRU)
  P4 early_life       — RMSE vs observed life fraction (LOCO + within-cell)
  P5 wp_grid          — W × P sensitivity (Patch-TF)
  P6 horizon          — H ∈ {1,3,5,10,20} SOH multi-step
  P7 loco_models      — LOCO for Patch-TF / Vanilla-TF / LSTM-direct / TCN-direct
  P8 uq               — PICP / PINAW / calibration (+ seed ensemble)
  P9 soh_stage        — RUL error by SOH stage

Usage:
  py scripts/run_ei_v3_priority.py --smoke
  py scripts/run_ei_v3_priority.py --epochs 120
  py scripts/run_ei_v3_priority.py --only P1,P2,P3 --epochs 120
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import set_seed, train_one_model, _eval_rul_rmse
from src.baselines.classical import evaluate_classical
from src.data.nasa_loader import chronological_split
from src.data.rul_dataset import (
    filter_by_life_fraction,
    load_battery_rul_data,
    soh_stage_bins,
)
from src.models.rul_models import build_rul_model, count_parameters
from src.models.uncertainty import (
    calibration_curve,
    mc_dropout_predict,
    mpiw,
    physics_regularizers,
    picp,
    pinaw,
)
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
BATTERIES = ["B0005", "B0007"]
NASA_CELLS = ["B0005", "B0006", "B0007"]
LIFE_FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MATCHED_MODELS = [
    "patch-tf",
    "vanilla-tf",
    "mlp",
    "gru",
    "lstm-direct",
    "tcn-direct",
    "lstm",  # seq2seq (decoder mismatch control)
    "tcn",  # seq2seq control
]


def _clean(res: dict) -> dict:
    return {k: v for k, v in res.items() if not str(k).startswith("_")}


def aggregate(rows: list[dict], keys: tuple[str, ...] | None = None) -> list[dict]:
    if keys is None:
        keys = (
            "experiment",
            "model",
            "battery",
            "feature_mode",
            "use_physics",
            "window_size",
            "horizon",
            "patch_size",
            "life_frac",
            "holdout",
            "soh_stage",
        )
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in keys)
        groups[key].append(r)
    out = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n_seeds"] = len(items)
        for m in (
            "rul_rmse",
            "rul_mae",
            "rul_r2",
            "soh_rmse",
            "soh_mae",
            "n_params",
            "picp_rul",
            "pinaw_rul",
            "mpiw_rul",
            "picp_soh",
            "pinaw_soh",
        ):
            vals = [float(it[m]) for it in items if m in it and it[m] is not None]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        out.append(row)
    return out


def _tf_ratio(model_name: str) -> float:
    # Direct-head models ignore TF; seq2seq keep 0.5
    if model_name in {"lstm", "tcn", "tcn-attn"}:
        return 0.5
    return 0.0


def run_matched(args, device, data_dir, out_dir, rows: list) -> None:
    """P1/P3: matched multifeat, no physics, identical protocol."""
    print("\n=== P1/P3 matched direct-head + controls ===")
    models = MATCHED_MODELS if not args.smoke else ["patch-tf", "vanilla-tf", "mlp", "lstm-direct"]
    seeds = SEEDS if not args.smoke else [42]
    batteries = BATTERIES if not args.smoke else ["B0005"]
    for bat in batteries:
        for model in models:
            for seed in seeds:
                print(f" matched {model} {bat} seed={seed}")
                res = train_one_model(
                    model_name=model,
                    battery_id=bat,
                    data_dir=data_dir,
                    out_dir=out_dir / "checkpoints",
                    window_size=args.window,
                    horizon=args.horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    tf_ratio=_tf_ratio(model),
                    device=device,
                    seed=seed,
                    feature_mode="multifeat",
                    use_physics=True,
                    patch_size=args.patch,
                    save_artifacts=True,
                )
                res["experiment"] = "matched"
                rows.append(_clean(res))


def run_classical(args, data_dir, rows: list) -> None:
    print("\n=== P2 classical baselines ===")
    for bat in BATTERIES:
        data = load_battery_rul_data(
            data_dir / f"{bat}.mat", bat, args.window, args.horizon, feature_mode="multifeat"
        )
        x, y_rul = data["x"], data["y_rul"]
        n = int(len(x) * 0.8)
        x_te, y_te = x[n:], y_rul[n:]
        for method in ("persistence", "linear"):
            m = evaluate_classical(x_te, y_te, method=method)
            m.update(
                {
                    "experiment": "classical",
                    "battery": bat,
                    "feature_mode": "multifeat",
                    "use_physics": True,
                    "window_size": args.window,
                    "horizon": args.horizon,
                    "seed": 0,
                    "soh_rmse": None,
                    "soh_mae": None,
                }
            )
            rows.append(m)
            print(f"  {method} {bat} RUL RMSE={m['rul_rmse']:.3f}")


def run_wp_grid(args, device, data_dir, out_dir, rows: list) -> None:
    print("\n=== P5 W×P sensitivity (Patch-TF, B0005) ===")
    windows = [20, 30, 50, 70, 100] if not args.smoke else [30, 50]
    patches = [2, 5, 10, 20] if not args.smoke else [5, 10]
    seeds = [42]  # cost control; optional expand
    for w in windows:
        for p in patches:
            if w % p != 0:
                continue
            for seed in seeds:
                print(f"  W={w} P={p} seed={seed}")
                res = train_one_model(
                    model_name="patch-tf",
                    battery_id="B0005",
                    data_dir=data_dir,
                    out_dir=out_dir / "checkpoints",
                    window_size=w,
                    horizon=args.horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    tf_ratio=0.0,
                    device=device,
                    seed=seed,
                    feature_mode="multifeat",
                    use_physics=True,
                    patch_size=p,
                    save_artifacts=False,
                )
                res["experiment"] = "wp_grid"
                rows.append(_clean(res))


def run_horizon(args, device, data_dir, out_dir, rows: list) -> None:
    print("\n=== P6 horizon sensitivity ===")
    horizons = [1, 3, 5, 10, 20] if not args.smoke else [1, 5]
    models = ["patch-tf", "lstm-direct", "tcn-direct"] if not args.smoke else ["patch-tf", "lstm-direct"]
    for h in horizons:
        for model in models:
            print(f"  H={h} {model}")
            res = train_one_model(
                model_name=model,
                battery_id="B0005",
                data_dir=data_dir,
                out_dir=out_dir / "checkpoints",
                window_size=args.window,
                horizon=h,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                tf_ratio=0.0,
                device=device,
                seed=42,
                feature_mode="multifeat",
                use_physics=True,
                patch_size=args.patch,
                save_artifacts=False,
            )
            res["experiment"] = "horizon"
            rows.append(_clean(res))


def _train_loco_model(
    model_name: str,
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
    patch_size: int = 5,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    xs, ys, yr = [], [], []
    for bid in train_cells:
        d = load_battery_rul_data(
            data_dir / f"{bid}.mat", bid, window, horizon, feature_mode=feature_mode
        )
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
    x_te, y_soh_te, y_rul_te = d_ho["x"][n_ho:], d_ho["y_soh"][n_ho:], d_ho["y_rul"][n_ho:]
    n_val = max(1, int(round(len(x_tr) * 0.15)))
    x_train, x_val = x_tr[:-n_val], x_tr[-n_val:]
    y_soh_train, y_soh_val = y_soh_tr[:-n_val], y_soh_tr[-n_val:]
    y_rul_train, y_rul_val = y_rul_tr[:-n_val], y_rul_tr[-n_val:]

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
    model = build_rul_model(
        model_name,
        horizon=horizon,
        n_features=x_train.shape[-1],
        patch_size=patch_size,
        window_size=window,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            pred_soh, pred_rul = model(xb, yb_soh, _tf_ratio(model_name))
            loss = soh_fn(pred_soh, yb_soh) + 0.5 * rul_fn(pred_rul, yb_rul)
            loss = loss + physics_regularizers(pred_soh)
            loss.backward()
            opt.step()
        val_rmse = _eval_rul_rmse(model, val_loader, device)
        if val_rmse < best_val:
            best_val, best_state, best_epoch = val_rmse, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    soh_preds, soh_true, rul_preds, rul_true = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x_te), batch_size):
            xb = torch.tensor(x_te[i : i + batch_size], dtype=torch.float32, device=device)
            ps, pr = model(xb, None, 0.0)
            soh_preds.append(ps.cpu().numpy())
            soh_true.append(y_soh_te[i : i + batch_size])
            rul_preds.append(pr.cpu().numpy())
            rul_true.append(y_rul_te[i : i + batch_size])
    soh_m = regression_metrics(
        np.concatenate(soh_true).reshape(-1), np.concatenate(soh_preds).reshape(-1)
    )
    rul_m = regression_metrics(
        np.concatenate(rul_true).reshape(-1), np.concatenate(rul_preds).reshape(-1)
    )
    meta = {
        "experiment": "loco_models",
        "model": model_name,
        "battery": holdout,
        "holdout": holdout,
        "train_cells": train_cells,
        "feature_mode": feature_mode,
        "use_physics": True,
        "window_size": window,
        "horizon": horizon,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "n_params": count_parameters(model),
        "rul_rmse": float(rul_m["rmse"]),
        "rul_mae": float(rul_m["mae"]),
        "rul_r2": float(rul_m["r2"]),
        "soh_rmse": float(soh_m["rmse"]),
        "soh_mae": float(soh_m["mae"]),
        "soh_r2": float(soh_m["r2"]),
    }
    return model, meta


def run_loco_models(args, device, data_dir, rows: list) -> None:
    print("\n=== P7 LOCO multi-model ===")
    models = (
        ["patch-tf", "vanilla-tf", "lstm-direct", "tcn-direct"]
        if not args.smoke
        else ["patch-tf", "lstm-direct"]
    )
    seeds = SEEDS if not args.smoke else [42]
    holdouts = NASA_CELLS if not args.smoke else ["B0005"]
    for holdout in holdouts:
        train_cells = [c for c in NASA_CELLS if c != holdout]
        for model in models:
            for seed in seeds:
                print(f"  LOCO {model} holdout={holdout} seed={seed}")
                _, meta = _train_loco_model(
                    model,
                    holdout,
                    train_cells,
                    data_dir,
                    args.window,
                    args.horizon,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    seed,
                    device,
                    patch_size=args.patch,
                )
                rows.append(meta)


def run_early_life(args, device, data_dir, out_dir, rows: list) -> None:
    """
    P4: RMSE vs observed life fraction.
    Primary: LOCO-trained models evaluated on holdout windows in each life bin.
    """
    print("\n=== P4 early-life RUL (LOCO eval by life fraction) ===")
    models = (
        ["patch-tf", "lstm-direct", "tcn-direct"] if not args.smoke else ["patch-tf", "lstm-direct"]
    )
    seeds = SEEDS if not args.smoke else [42]
    holdouts = BATTERIES if not args.smoke else ["B0005"]
    half = 0.05
    for holdout in holdouts:
        train_cells = [c for c in NASA_CELLS if c != holdout]
        for model_name in models:
            for seed in seeds:
                print(f"  early-life train {model_name}→{holdout} seed={seed}")
                model, _ = _train_loco_model(
                    model_name,
                    holdout,
                    train_cells,
                    data_dir,
                    args.window,
                    args.horizon,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    seed,
                    device,
                    patch_size=args.patch,
                )
                d = load_battery_rul_data(
                    data_dir / f"{holdout}.mat",
                    holdout,
                    args.window,
                    args.horizon,
                    feature_mode="multifeat",
                )
                for frac in LIFE_FRACS:
                    x_f, _, y_rul_f, _ = filter_by_life_fraction(
                        d["x"], d["y_soh"], d["y_rul"], d["life_fracs"], frac, half
                    )
                    if len(x_f) < 2:
                        continue
                    with torch.no_grad():
                        xb = torch.tensor(x_f, dtype=torch.float32, device=device)
                        _, pr = model(xb, None, 0.0)
                        pred = pr.cpu().numpy().reshape(-1)
                    m = regression_metrics(y_rul_f.reshape(-1), pred)
                    rows.append(
                        {
                            "experiment": "early_life",
                            "model": model_name,
                            "battery": holdout,
                            "holdout": holdout,
                            "life_frac": frac,
                            "feature_mode": "multifeat",
                            "use_physics": True,
                            "window_size": args.window,
                            "horizon": args.horizon,
                            "seed": seed,
                            "n_windows": int(len(x_f)),
                            "rul_rmse": float(m["rmse"]),
                            "rul_mae": float(m["mae"]),
                            "rul_r2": float(m["r2"]),
                        }
                    )
                    print(f"    frac={frac:.0%} n={len(x_f)} RMSE={m['rmse']:.3f}")


def _load_patch_ckpt(
    ckpt_dir: Path,
    bat: str,
    window: int,
    horizon: int,
    seed: int,
    n_features: int,
    patch: int,
    device: torch.device,
) -> nn.Module | None:
    tag = f"rul_patch-tf_{bat.lower()}_w{window}_h{horizon}_s{seed}_multifeat"
    cands = sorted(ckpt_dir.glob(f"{tag}*.pt"))
    if not cands:
        return None
    model = build_rul_model(
        "patch-tf",
        horizon=horizon,
        n_features=n_features,
        patch_size=patch,
        window_size=window,
    ).to(device)
    model.load_state_dict(torch.load(cands[-1], map_location=device, weights_only=True))
    return model


def run_uq(args, device, data_dir, out_dir, rows: list, uq_rows: list) -> None:
    print("\n=== P8 UQ PICP/PINAW/calibration + ensemble ===")
    ckpt_dir = out_dir / "checkpoints"
    n_mc = 10 if args.smoke else 30
    for bat in BATTERIES if not args.smoke else ["B0005"]:
        data = load_battery_rul_data(
            data_dir / f"{bat}.mat", bat, args.window, args.horizon, feature_mode="multifeat"
        )
        n = int(len(data["x"]) * 0.8)
        x_te = data["x"][n:]
        y_soh_te = data["y_soh"][n:]
        y_rul_te = data["y_rul"][n:]
        model = _load_patch_ckpt(
            ckpt_dir, bat, args.window, args.horizon, 42, data["n_features"], args.patch, device
        )
        if model is None:
            train_one_model(
                "patch-tf",
                bat,
                data_dir,
                ckpt_dir,
                args.window,
                args.horizon,
                args.epochs,
                args.batch_size,
                args.lr,
                0.0,
                device,
                seed=42,
                feature_mode="multifeat",
                use_physics=True,
                patch_size=args.patch,
            )
            model = _load_patch_ckpt(
                ckpt_dir, bat, args.window, args.horizon, 42, data["n_features"], args.patch, device
            )
        if model is None:
            print(f"  skip UQ {bat}: no checkpoint")
            continue

        xb = torch.tensor(x_te, dtype=torch.float32, device=device)
        mc = mc_dropout_predict(model, xb, n_samples=n_mc, device=device)
        z95 = 1.96
        uq = {
            "experiment": "uq_mc",
            "model": "patch-tf",
            "battery": bat,
            "n_mc": n_mc,
            "picp_rul": picp(y_rul_te, mc["rul_mean"], mc["rul_std"], z=z95),
            "pinaw_rul": pinaw(y_rul_te, mc["rul_mean"], mc["rul_std"], z=z95),
            "mpiw_rul": mpiw(mc["rul_mean"], mc["rul_std"], z=z95),
            "picp_soh": picp(y_soh_te, mc["soh_mean"], mc["soh_std"], z=z95),
            "pinaw_soh": pinaw(y_soh_te, mc["soh_mean"], mc["soh_std"], z=z95),
            "calibration_rul": calibration_curve(y_rul_te, mc["rul_mean"], mc["rul_std"]),
            "calibration_soh": calibration_curve(
                y_soh_te.reshape(-1), mc["soh_mean"].reshape(-1), mc["soh_std"].reshape(-1)
            ),
        }
        uq_rows.append(uq)
        rows.append(
            {
                "experiment": "uq_mc",
                "model": "patch-tf",
                "battery": bat,
                "feature_mode": "multifeat",
                "seed": 42,
                "rul_rmse": float(
                    regression_metrics(y_rul_te.reshape(-1), mc["rul_mean"].reshape(-1))["rmse"]
                ),
                "picp_rul": uq["picp_rul"],
                "pinaw_rul": uq["pinaw_rul"],
                "mpiw_rul": uq["mpiw_rul"],
                "picp_soh": uq["picp_soh"],
                "pinaw_soh": uq["pinaw_soh"],
            }
        )
        print(
            f"  MC {bat}: PICP95_rul={uq['picp_rul']:.3f} PINAW={uq['pinaw_rul']:.3f} "
            f"MPIW={uq['mpiw_rul']:.3f}"
        )

        preds_rul, preds_soh = [], []
        for seed in (SEEDS if not args.smoke else [42]):
            m = _load_patch_ckpt(
                ckpt_dir,
                bat,
                args.window,
                args.horizon,
                seed,
                data["n_features"],
                args.patch,
                device,
            )
            if m is None:
                continue
            m.eval()
            with torch.no_grad():
                ps, pr = m(xb, None, 0.0)
            preds_soh.append(ps.cpu().numpy())
            preds_rul.append(pr.cpu().numpy())
        if len(preds_rul) >= 2:
            rul_arr = np.stack(preds_rul, axis=0)
            soh_arr = np.stack(preds_soh, axis=0)
            ens = {
                "experiment": "uq_ensemble",
                "model": "patch-tf-ensemble",
                "battery": bat,
                "n_members": len(preds_rul),
                "picp_rul": picp(y_rul_te, rul_arr.mean(0), rul_arr.std(0), z=z95),
                "pinaw_rul": pinaw(y_rul_te, rul_arr.mean(0), rul_arr.std(0), z=z95),
                "mpiw_rul": mpiw(rul_arr.mean(0), rul_arr.std(0), z=z95),
                "picp_soh": picp(y_soh_te, soh_arr.mean(0), soh_arr.std(0), z=z95),
                "pinaw_soh": pinaw(y_soh_te, soh_arr.mean(0), soh_arr.std(0), z=z95),
                "calibration_rul": calibration_curve(
                    y_rul_te, rul_arr.mean(0), rul_arr.std(0)
                ),
            }
            uq_rows.append(ens)
            rows.append(
                {
                    "experiment": "uq_ensemble",
                    "model": "patch-tf-ensemble",
                    "battery": bat,
                    "feature_mode": "multifeat",
                    "seed": -1,
                    "picp_rul": ens["picp_rul"],
                    "pinaw_rul": ens["pinaw_rul"],
                    "mpiw_rul": ens["mpiw_rul"],
                    "picp_soh": ens["picp_soh"],
                    "pinaw_soh": ens["pinaw_soh"],
                }
            )
            print(
                f"  Ens {bat}: PICP95_rul={ens['picp_rul']:.3f} PINAW={ens['pinaw_rul']:.3f}"
            )


def run_soh_stage(args, device, data_dir, out_dir, rows: list) -> None:
    print("\n=== P9 RUL error by SOH stage ===")
    ckpt_dir = out_dir / "checkpoints"
    for bat in BATTERIES if not args.smoke else ["B0005"]:
        data = load_battery_rul_data(
            data_dir / f"{bat}.mat", bat, args.window, args.horizon, feature_mode="multifeat"
        )
        n = int(len(data["x"]) * 0.8)
        x_te = data["x"][n:]
        y_rul_te = data["y_rul"][n:]
        soh_te = data["soh_at_pred"][n:]
        model = _load_patch_ckpt(
            ckpt_dir, bat, args.window, args.horizon, 42, data["n_features"], args.patch, device
        )
        if model is None:
            print(f"  skip {bat}: no ckpt")
            continue
        model.eval()
        with torch.no_grad():
            _, pr = model(torch.tensor(x_te, dtype=torch.float32, device=device), None, 0.0)
            pred = pr.cpu().numpy().reshape(-1)
        stages = soh_stage_bins(soh_te)
        for stage in [">0.95", "0.90-0.95", "0.85-0.90", "0.80-0.85", "<=0.80"]:
            mask = stages == stage
            if mask.sum() < 2:
                continue
            m = regression_metrics(y_rul_te[mask], pred[mask])
            rows.append(
                {
                    "experiment": "soh_stage",
                    "model": "patch-tf",
                    "battery": bat,
                    "soh_stage": stage,
                    "n_windows": int(mask.sum()),
                    "rul_rmse": float(m["rmse"]),
                    "rul_mae": float(m["mae"]),
                    "seed": 42,
                    "feature_mode": "multifeat",
                }
            )
            print(f"  {bat} {stage}: n={mask.sum()} RMSE={m['rmse']:.3f}")


def plot_early_life(agg: list[dict], out_dir: Path) -> None:
    rows = [r for r in agg if r.get("experiment") == "early_life"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, bat in zip(axes, BATTERIES):
        for model, style in (
            ("patch-tf", "o-"),
            ("lstm-direct", "s--"),
            ("tcn-direct", "^:"),
        ):
            pts = sorted(
                [r for r in rows if r.get("battery") == bat and r.get("model") == model],
                key=lambda r: r.get("life_frac") or 0,
            )
            if not pts:
                continue
            xs = [100 * float(r["life_frac"]) for r in pts]
            ys = [float(r["rul_rmse"]) for r in pts]
            ax.plot(xs, ys, style, label=model)
        ax.set_title(bat)
        ax.set_xlabel("Observed life (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RUL RMSE (cycles)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Early-life RUL prediction (LOCO)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_early_life_rul.png", dpi=150)
    plt.close(fig)


def plot_wp_heatmap(raw: list[dict], out_dir: Path) -> None:
    rows = [r for r in raw if r.get("experiment") == "wp_grid"]
    if not rows:
        return
    ws = sorted({int(r["window_size"]) for r in rows})
    ps = sorted({int(r["patch_size"]) for r in rows})
    mat = np.full((len(ws), len(ps)), np.nan)
    for r in rows:
        i, j = ws.index(int(r["window_size"])), ps.index(int(r["patch_size"]))
        mat[i, j] = float(r["rul_rmse"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps)
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels(ws)
    ax.set_xlabel("Patch size P")
    ax.set_ylabel("Window W")
    ax.set_title("B0005 Patch-TF RUL RMSE (W×P)")
    for i in range(len(ws)):
        for j in range(len(ps)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_wp_heatmap.png", dpi=150)
    plt.close(fig)


def plot_calibration(uq_rows: list[dict], out_dir: Path) -> None:
    mc = [r for r in uq_rows if r.get("experiment") == "uq_mc" and "calibration_rul" in r]
    if not mc:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for r in mc:
        cal = r["calibration_rul"]
        xs = [c["nominal"] for c in cal]
        ys = [c["empirical"] for c in cal]
        ax.plot(xs, ys, "o-", label=r["battery"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("RUL interval calibration (MC-Dropout)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_uq_calibration.png", dpi=150)
    plt.close(fig)


def write_report(agg: list[dict], uq_rows: list[dict], out_dir: Path) -> None:
    lines = ["# EI v3 priority results\n"]
    for exp in (
        "classical",
        "matched",
        "wp_grid",
        "horizon",
        "early_life",
        "loco_models",
        "uq_mc",
        "uq_ensemble",
        "soh_stage",
    ):
        block = [r for r in agg if r.get("experiment") == exp]
        if not block and exp not in {"uq_mc", "uq_ensemble"}:
            continue
        lines.append(f"\n## {exp}\n")
        if exp.startswith("uq"):
            for r in uq_rows:
                if r.get("experiment") == exp:
                    lines.append(
                        f"- {r.get('model')} {r.get('battery')}: "
                        f"PICP95_rul={r.get('picp_rul'):.3f} "
                        f"PINAW={r.get('pinaw_rul'):.3f} MPIW={r.get('mpiw_rul'):.3f}\n"
                    )
            continue
        for r in block:
            extra = ""
            if r.get("life_frac") is not None:
                extra += f" life={100*float(r['life_frac']):.0f}%"
            if r.get("holdout"):
                extra += f" holdout={r['holdout']}"
            if r.get("soh_stage"):
                extra += f" stage={r['soh_stage']}"
            if r.get("window_size") and exp == "wp_grid":
                extra += f" W={r['window_size']} P={r.get('patch_size')}"
            if r.get("horizon") and exp == "horizon":
                extra += f" H={r['horizon']}"
            rmse = r.get("rul_rmse")
            std = r.get("rul_rmse_std")
            if rmse is None:
                continue
            if std is not None:
                lines.append(
                    f"- {r.get('model')} {r.get('battery')}{extra}: "
                    f"{rmse:.3f}±{std:.3f}\n"
                )
            else:
                lines.append(f"- {r.get('model')} {r.get('battery')}{extra}: {rmse:.3f}\n")
    (out_dir / "ei_v3_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--patch", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list: P1,P2,P3,P4,P5,P6,P7,P8,P9 or matched,classical,...",
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "nasa"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "ei_v3"))
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 5)

    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    alias = {
        "p1": "matched",
        "p2": "classical",
        "p3": "matched",
        "p4": "early_life",
        "p5": "wp_grid",
        "p6": "horizon",
        "p7": "loco_models",
        "p8": "uq",
        "p9": "soh_stage",
    }
    selected = {alias.get(x, x) for x in only} if only else set()

    def want(name: str) -> bool:
        return not selected or name in selected

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} epochs={args.epochs} out={out_dir}")

    rows: list[dict] = []
    uq_rows: list[dict] = []

    if want("classical"):
        run_classical(args, data_dir, rows)
    if want("matched"):
        run_matched(args, device, data_dir, out_dir, rows)
    if want("wp_grid"):
        run_wp_grid(args, device, data_dir, out_dir, rows)
    if want("horizon"):
        run_horizon(args, device, data_dir, out_dir, rows)
    if want("loco_models"):
        run_loco_models(args, device, data_dir, rows)
    if want("early_life"):
        run_early_life(args, device, data_dir, out_dir, rows)
    if want("uq"):
        run_uq(args, device, data_dir, out_dir, rows, uq_rows)
    if want("soh_stage"):
        run_soh_stage(args, device, data_dir, out_dir, rows)

    agg = aggregate(rows)
    (out_dir / "ei_v3_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "ei_v3_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "ei_v3_uq.json").write_text(json.dumps(uq_rows, indent=2), encoding="utf-8")
    plot_early_life(agg, out_dir)
    plot_wp_heatmap(rows, out_dir)
    plot_calibration(uq_rows, out_dir)
    write_report(agg, uq_rows, out_dir)
    print(f"\nDone. Wrote {out_dir}")


if __name__ == "__main__":
    main()
