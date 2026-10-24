#!/usr/bin/env python3
"""
Priority-1 revision controls for the information-consistent RUL paper.

1) Cycle-channel scale control (other 5 channels fixed): t, t/T_prior, t/T_target, t/1000
2) L1P SOH-matching distance diagnostics
3) Protocol A train/test cycle-channel ranges
4) L2+SOH channel control (append SOH_t to L2)
5) Dual-head vs SOH-only (short epochs)

Outputs: outputs/revision_controls/
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import set_seed, _eval_rul_rmse
from src.data.cycle_features import (
    INFO_L0,
    INFO_L1,
    INFO_L2,
    apply_cycle_info_level,
    extract_discharge_cycle_features,
    soh_matched_rul_prior,
)
from src.data.nasa_loader import chronological_split, load_battery_soh
from src.data.rul_dataset import (
    build_multifeat_rul_windows,
    compute_rul,
    eol_cycle_index,
)
from src.models.rul_models import build_rul_model
from src.models.uncertainty import physics_regularizers
from src.utils.metrics import regression_metrics

EOL = 0.7
NASA = ["B0005", "B0006", "B0007"]
SEEDS = [42, 43, 44]


def load_soh_raw(data_dir: Path, bat: str):
    _, _, soh = load_battery_soh(data_dir / f"{bat}.mat", bat)
    raw = extract_discharge_cycle_features(data_dir / f"{bat}.mat", bat, normalize_cycle=False)
    n = min(len(raw), len(soh))
    return raw[:n].astype(np.float32), np.asarray(soh[:n], dtype=np.float32)


def sibling_eol(data_dir: Path, target: str) -> float:
    vals = []
    for bat in NASA:
        if bat == target:
            continue
        _, soh = load_soh_raw(data_dir, bat)
        vals.append(float(eol_cycle_index(soh, EOL)))
    return float(np.median(vals))


def make_windows(raw, soh, level: str, t_prior: float, window=50, horizon=5, extra_soh=False):
    last = float(raw[-1, 5])
    t_target = float(eol_cycle_index(soh, EOL))
    if level == "L0":
        feats = apply_cycle_info_level(raw, INFO_L0, t_scale=1.0)  # raw t
    elif level == "L0_clock":
        feats = apply_cycle_info_level(raw, INFO_L0, t_scale=1000.0)
    elif level == "L1":
        feats = apply_cycle_info_level(raw, INFO_L1, t_scale=t_prior)
    elif level == "L2":
        feats = apply_cycle_info_level(raw, INFO_L2, t_scale=t_target)
    else:
        raise ValueError(level)
    if extra_soh:
        # append SOH as 7th channel (L2+SOH control)
        feats = np.concatenate([feats, soh.reshape(-1, 1)], axis=1)
    x, y_soh, y_rul = build_multifeat_rul_windows(feats, soh, window, horizon, EOL)
    # drop RUL=0
    mask = y_rul > 0
    return x[mask], y_soh[mask], y_rul[mask], feats, t_target, t_prior


def channel_ranges(x: np.ndarray, train_n: int) -> dict:
    c_tr = x[:train_n, :, 5].reshape(-1)
    c_te = x[train_n:, :, 5].reshape(-1)
    return {
        "train_min": float(c_tr.min()),
        "train_max": float(c_tr.max()),
        "train_mean": float(c_tr.mean()),
        "test_min": float(c_te.min()),
        "test_max": float(c_te.max()),
        "test_mean": float(c_te.mean()),
        "frac_test_above_train_max": float(np.mean(c_te > c_tr.max())),
    }


def train_eval(
    x, y_soh, y_rul, model_name, seed, epochs, device, soh_only=False, dual_head=True
):
    set_seed(seed)
    n = int(len(x) * 0.8)
    x_tr_full, x_te = x[:n], x[n:]
    y_soh_tr_full, y_soh_te = y_soh[:n], y_soh[n:]
    y_rul_tr_full, y_rul_te = y_rul[:n], y_rul[n:]
    n_val = max(1, int(round(len(x_tr_full) * 0.15)))
    n_tr = max(1, len(x_tr_full) - n_val)
    x_tr, x_val = x_tr_full[:n_tr], x_tr_full[n_tr:]
    y_soh_tr, y_soh_val = y_soh_tr_full[:n_tr], y_soh_tr_full[n_tr:]
    y_rul_tr, y_rul_val = y_rul_tr_full[:n_tr], y_rul_tr_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(y_soh_tr), torch.tensor(y_rul_tr)),
        batch_size=16,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_soh_val), torch.tensor(y_rul_val)),
        batch_size=16,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_te), torch.tensor(y_soh_te), torch.tensor(y_rul_te)),
        batch_size=16,
        shuffle=False,
    )
    model = build_rul_model(
        model_name,
        horizon=5,
        n_features=int(x.shape[-1]),
        window_size=50,
        patch_size=5,
        d_model=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            ps, pr = model(xb, yb_soh, 0.0)
            if soh_only:
                loss = soh_fn(ps, yb_soh) + physics_regularizers(ps)
            elif not dual_head:
                loss = 0.5 * rul_fn(pr, yb_rul)
            else:
                loss = soh_fn(ps, yb_soh) + 0.5 * rul_fn(pr, yb_rul) + physics_regularizers(ps)
            loss.backward()
            opt.step()
        if soh_only:
            # early-stop on SOH RMSE
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb_soh, _ in val_loader:
                    ps, _ = model(xb.to(device), None, 0.0)
                    preds.append(ps.cpu().numpy().reshape(-1))
                    trues.append(yb_soh.numpy().reshape(-1))
            val = float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(trues)) ** 2)))
        else:
            val = _eval_rul_rmse(model, val_loader, device)
        if val < best_val:
            best_val, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    preds_r, trues_r, preds_s, trues_s = [], [], [], []
    with torch.no_grad():
        for xb, yb_soh, yb_rul in test_loader:
            ps, pr = model(xb.to(device), None, 0.0)
            preds_r.append(pr.cpu().numpy().reshape(-1))
            trues_r.append(yb_rul.numpy().reshape(-1))
            preds_s.append(ps.cpu().numpy().reshape(-1))
            trues_s.append(yb_soh.numpy().reshape(-1))
    pr = np.concatenate(preds_r)
    tr = np.concatenate(trues_r)
    ps = np.concatenate(preds_s)
    ts = np.concatenate(trues_s)
    return {
        "rul_rmse": float(regression_metrics(tr, pr)["rmse"]),
        "soh_rmse": float(regression_metrics(ts, ps)["rmse"]),
        "best_val": float(best_val),
        "n_test": int(len(tr)),
        "preds_rul": pr,
        "trues_rul": tr,
    }


def l1p_matching_stats(data_dir: Path) -> dict:
    out = {}
    for target in NASA:
        raw_t, soh_t = load_soh_raw(data_dir, target)
        sibs = []
        for bat in NASA:
            if bat == target:
                continue
            _, s = load_soh_raw(data_dir, bat)
            sibs.append(s)
        # distances at each t
        dists = []
        for i, s_val in enumerate(soh_t):
            d_sib = []
            for ss in sibs:
                j = int(np.argmin(np.abs(ss - s_val)))
                d_sib.append(float(abs(ss[j] - s_val)))
            dists.append(float(np.median(d_sib)))
        dists = np.asarray(dists, dtype=np.float64)
        prior = soh_matched_rul_prior(soh_t, sibs, EOL)
        true_rul = compute_rul(soh_t, EOL)
        abs_err = np.abs(prior - true_rul)
        # SOH bins
        bins = [(0.95, 1.01), (0.85, 0.95), (0.75, 0.85), (0.0, 0.75)]
        by_bin = {}
        for lo, hi in bins:
            m = (soh_t >= lo) & (soh_t < hi)
            if not np.any(m):
                continue
            by_bin[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(m.sum()),
                "mean_dist": float(dists[m].mean()),
                "mean_abs_rul_err": float(abs_err[m].mean()),
            }
        out[target] = {
            "mean_dist": float(dists.mean()),
            "median_dist": float(np.median(dists)),
            "p90_dist": float(np.quantile(dists, 0.9)),
            "corr_dist_abs_rul_err": float(np.corrcoef(dists, abs_err)[0, 1]),
            "mean_abs_rul_err": float(abs_err.mean()),
            "by_soh_bin": by_bin,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 5
        seeds = [42]
        bats = ["B0005"]
    else:
        seeds = SEEDS
        bats = NASA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "nasa"
    out_dir = ROOT / "outputs" / "revision_controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- L1P matching ---
    match = l1p_matching_stats(data_dir)
    (out_dir / "l1p_matching.json").write_text(json.dumps(match, indent=2), encoding="utf-8")
    print("L1P matching:", json.dumps(match, indent=2))

    # --- Protocol A channel ranges ---
    ranges = {}
    for bat in NASA:
        raw, soh = load_soh_raw(data_dir, bat)
        t_prior = sibling_eol(data_dir, bat)
        for level in ["L0", "L1", "L2", "L0_clock"]:
            x, y_soh, y_rul, feats, t_tgt, tp = make_windows(raw, soh, level, t_prior)
            n = int(len(x) * 0.8)
            ranges.setdefault(bat, {})[level] = {
                **channel_ranges(x, n),
                "t_prior": tp,
                "t_target": t_tgt,
                "n_windows": int(len(x)),
            }
    (out_dir / "protocol_a_channel_ranges.json").write_text(
        json.dumps(ranges, indent=2), encoding="utf-8"
    )

    # --- Scale control + L2+SOH + prediction deltas ---
    rows = []
    pred_bank: dict[tuple, np.ndarray] = {}
    levels = ["L0", "L1", "L2", "L0_clock"]
    for bat in bats:
        raw, soh = load_soh_raw(data_dir, bat)
        t_prior = sibling_eol(data_dir, bat)
        for level in levels:
            x, y_soh, y_rul, _, _, _ = make_windows(raw, soh, level, t_prior)
            for seed in seeds:
                print(f"=== scale {bat} {level} seed={seed} ===")
                res = train_eval(x, y_soh, y_rul, "mlp", seed, args.epochs, device)
                row = {
                    "experiment": "scale_control",
                    "battery": bat,
                    "level": level,
                    "seed": seed,
                    "rul_rmse": res["rul_rmse"],
                    "soh_rmse": res["soh_rmse"],
                    "n_test": res["n_test"],
                }
                rows.append(row)
                pred_bank[(bat, level, seed)] = res["preds_rul"]
                (out_dir / "scale_control_raw.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8"
                )
        # L2+SOH
        x, y_soh, y_rul, _, _, _ = make_windows(raw, soh, "L2", t_prior, extra_soh=True)
        for seed in seeds:
            print(f"=== L2+SOH {bat} seed={seed} ===")
            res = train_eval(x, y_soh, y_rul, "mlp", seed, args.epochs, device)
            rows.append(
                {
                    "experiment": "l2_plus_soh",
                    "battery": bat,
                    "level": "L2+SOH",
                    "seed": seed,
                    "rul_rmse": res["rul_rmse"],
                    "soh_rmse": res["soh_rmse"],
                    "n_test": res["n_test"],
                }
            )
        # SOH-only vs dual-head at L2
        x, y_soh, y_rul, _, _, _ = make_windows(raw, soh, "L2", t_prior)
        for seed in seeds:
            print(f"=== SOH-only {bat} seed={seed} ===")
            r1 = train_eval(x, y_soh, y_rul, "mlp", seed, args.epochs, device, soh_only=True)
            rows.append(
                {
                    "experiment": "soh_only",
                    "battery": bat,
                    "level": "L2",
                    "seed": seed,
                    "rul_rmse": r1["rul_rmse"],
                    "soh_rmse": r1["soh_rmse"],
                    "n_test": r1["n_test"],
                }
            )
            print(f"=== dual-head L2 {bat} seed={seed} ===")
            r2 = train_eval(x, y_soh, y_rul, "mlp", seed, args.epochs, device, soh_only=False)
            rows.append(
                {
                    "experiment": "dual_head",
                    "battery": bat,
                    "level": "L2",
                    "seed": seed,
                    "rul_rmse": r2["rul_rmse"],
                    "soh_rmse": r2["soh_rmse"],
                    "n_test": r2["n_test"],
                }
            )

    # prediction differences L1 vs L2
    deltas = {}
    for bat in bats:
        for seed in seeds:
            a = pred_bank.get((bat, "L1", seed))
            b = pred_bank.get((bat, "L2", seed))
            if a is None or b is None:
                continue
            n = min(len(a), len(b))
            deltas[f"{bat}_s{seed}"] = {
                "mean_abs_delta": float(np.mean(np.abs(a[:n] - b[:n]))),
                "rmse_delta": float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))),
                "corr": float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 else None,
            }
    (out_dir / "l1_l2_pred_delta.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    (out_dir / "scale_control_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # aggregate means
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["battery"], r["level"])
        groups[key].append(r["rul_rmse"] if r["experiment"] != "soh_only" else r["soh_rmse"])
    agg = []
    for key, vals in sorted(groups.items()):
        metric = "soh_rmse" if key[0] == "soh_only" else "rul_rmse"
        # for soh_only also keep soh; for dual keep both means of reported field
        vals2 = [
            r["soh_rmse"] if key[0] in ("soh_only", "dual_head") else r["rul_rmse"]
            for r in rows
            if (r["experiment"], r["battery"], r["level"]) == key
        ]
        if key[0] in ("soh_only", "dual_head"):
            soh_vals = [
                r["soh_rmse"]
                for r in rows
                if (r["experiment"], r["battery"], r["level"]) == key
            ]
            rul_vals = [
                r["rul_rmse"]
                for r in rows
                if (r["experiment"], r["battery"], r["level"]) == key
            ]
            agg.append(
                {
                    "experiment": key[0],
                    "battery": key[1],
                    "level": key[2],
                    "soh_rmse_mean": float(np.mean(soh_vals)),
                    "soh_rmse_std": float(np.std(soh_vals)),
                    "rul_rmse_mean": float(np.mean(rul_vals)),
                    "rul_rmse_std": float(np.std(rul_vals)),
                    "n": len(soh_vals),
                }
            )
        else:
            agg.append(
                {
                    "experiment": key[0],
                    "battery": key[1],
                    "level": key[2],
                    "rul_rmse_mean": float(np.mean(vals)),
                    "rul_rmse_std": float(np.std(vals)),
                    "n": len(vals),
                }
            )
    (out_dir / "revision_controls_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # markdown report
    lines = ["# Revision controls\n\n", "## L1P matching distance\n\n"]
    for bat, d in match.items():
        lines.append(
            f"- {bat}: mean dist={d['mean_dist']:.4f}, median={d['median_dist']:.4f}, "
            f"p90={d['p90_dist']:.4f}, corr(dist,|RUL err|)={d['corr_dist_abs_rul_err']:.3f}\n"
        )
    lines.append("\n## Scale-control RUL RMSE (MLP)\n\n")
    lines.append("| Battery | L0 (raw t) | L1 | L2 | L0 clock | L2+SOH |\n|---|---:|---:|---:|---:|---:|\n")
    for bat in bats:
        def g(exp, level):
            for a in agg:
                if a["battery"] == bat and a["level"] == level and a["experiment"] == exp:
                    return f"{a['rul_rmse_mean']:.2f}±{a['rul_rmse_std']:.2f}"
            return "-"

        lines.append(
            f"| {bat} | {g('scale_control','L0')} | {g('scale_control','L1')} | "
            f"{g('scale_control','L2')} | {g('scale_control','L0_clock')} | "
            f"{g('l2_plus_soh','L2+SOH')} |\n"
        )
    lines.append("\n## SOH-only vs dual-head (SOH RMSE at L2)\n\n")
    lines.append("| Battery | SOH-only | Dual-head |\n|---|---:|---:|\n")
    for bat in bats:
        so = next((a for a in agg if a["battery"] == bat and a["experiment"] == "soh_only"), None)
        dh = next((a for a in agg if a["battery"] == bat and a["experiment"] == "dual_head"), None)
        lines.append(
            f"| {bat} | "
            f"{so['soh_rmse_mean']:.3f}±{so['soh_rmse_std']:.3f} | "
            f"{dh['soh_rmse_mean']:.3f}±{dh['soh_rmse_std']:.3f} |\n"
            if so and dh
            else f"| {bat} | - | - |\n"
        )
    lines.append("\n## L1 vs L2 prediction delta\n\n")
    for k, v in deltas.items():
        lines.append(f"- {k}: mean|Δ|={v['mean_abs_delta']:.3f}, rmse Δ={v['rmse_delta']:.3f}, corr={v['corr']}\n")
    (out_dir / "revision_controls_report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print("done →", out_dir)


if __name__ == "__main__":
    main()
