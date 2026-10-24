#!/usr/bin/env python3
"""L1P vs L2 window diagnostics and dual-head SOH loss check.

Writes to outputs/l1p_mechanism/:
  windows.json   predictions, channel, SOH, matching distance
  summary.json   error by SOH / channel bins
  soh_head.json  grad-norm ratio and RUL/T loss control
  report.md
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

from scripts.run_info_levels_v2 import (  # noqa: E402
    EOL_TH,
    load_nasa_windows,
    restrict_pre_eol,
    sibling_t_eol,
    nasa_lifetime_table,
    t_scale_for,
)
from scripts.train_rul import set_seed, _eval_rul_rmse  # noqa: E402
from src.models.rul_models import build_rul_model  # noqa: E402
from src.models.uncertainty import physics_regularizers  # noqa: E402
from src.utils.metrics import regression_metrics  # noqa: E402

SEEDS = [42, 43, 44]
CELLS = ["B0005", "B0006", "B0007"]
DATA = ROOT / "data" / "nasa"
OUT = ROOT / "outputs" / "l1p_mechanism"


def cell_data(cell: str, level: str, life: dict) -> dict:
    scales = sibling_t_eol(life, cell)
    ts = t_scale_for(level, scales)
    d0 = load_nasa_windows(DATA, cell, 50, 5, level, ts)
    soh = np.asarray(d0["soh"], dtype=np.float64)
    n_windows = len(soh) - 50 - 5 + 1
    t_pred = np.arange(50 - 1, 50 - 1 + n_windows)
    keep = np.asarray(d0["y_rul"]).reshape(-1) > 0
    t_pred = t_pred[keep]
    d = {
        "x": np.asarray(d0["x"])[keep],
        "y_soh": np.asarray(d0["y_soh"])[keep],
        "y_rul": np.asarray(d0["y_rul"])[keep],
    }
    # channel value at the end of each window
    c = np.asarray(d["x"])[:, -1, 5]
    # sibling SOH match distance (median over leave-target siblings)
    sibs = []
    for other in CELLS:
        if other == cell:
            continue
        od = load_nasa_windows(DATA, other, 50, 5, "L0", 1.0)
        sibs.append(np.asarray(od["soh"], dtype=np.float64))
    dist = np.empty(len(t_pred))
    for i, j in enumerate(t_pred):
        ds = [float(np.min(np.abs(s - soh[j]))) for s in sibs]
        dist[i] = float(np.median(ds))
    n_train = int(len(d["x"]) * 0.8)
    c_train_max = float(np.max(np.asarray(d["x"])[:n_train, :, 5]))
    return {
        "x": d["x"], "y_soh": d["y_soh"], "y_rul": d["y_rul"],
        "soh": soh, "t_pred": t_pred, "c": c, "match_dist": dist,
        "c_train_max": c_train_max,
        "t_target": float(scales["t_target"]), "t_prior": float(scales["t_med"]),
    }


def train_mlp(data: dict, seed: int, epochs: int, mode: str = "paper", rul_scale: float = 1.0):
    """mode: paper | normalized | soh_only. rul_scale divides the RUL target."""
    set_seed(seed)
    x, y_soh, y_rul = data["x"], data["y_soh"], data["y_rul"]
    n = int(len(x) * 0.8)
    x_tr_full, x_te = x[:n], x[n:]
    ys_tr_full, ys_te = y_soh[:n], y_soh[n:]
    yr_tr_full, yr_te = y_rul[:n], y_rul[n:]
    n_val = max(1, int(round(len(x_tr_full) * 0.15)))
    n_tr = max(1, len(x_tr_full) - n_val)
    x_tr, x_val = x_tr_full[:n_tr], x_tr_full[n_tr:]
    ys_tr, ys_val = ys_tr_full[:n_tr], ys_tr_full[n_tr:]
    yr_tr, yr_val = yr_tr_full[:n_tr], yr_tr_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(ys_tr), torch.tensor(yr_tr)),
        batch_size=16, shuffle=True, generator=g)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(ys_val), torch.tensor(yr_val)),
        batch_size=16, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_te), torch.tensor(ys_te), torch.tensor(yr_te)),
        batch_size=16, shuffle=False)

    model = build_rul_model("mlp", horizon=5, n_features=int(x.shape[-1]),
                            window_size=50, d_model=64, num_layers=2, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    grad_ratio = []
    for epoch in range(1, epochs + 1):
        model.train()
        for bi, (xb, ysb, yrb) in enumerate(train_loader):
            opt.zero_grad()
            ps, pr = model(xb, ysb, 0.0)
            if mode == "soh_only":
                loss = soh_fn(ps, ysb) + physics_regularizers(ps)
                loss.backward()
            else:
                l_soh = soh_fn(ps, ysb) + physics_regularizers(ps)
                l_rul = 0.5 * rul_fn(pr, yrb / rul_scale)
                if epoch <= 3:
                    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
                    gs = torch.autograd.grad(l_soh, backbone, retain_graph=True, allow_unused=True)
                    gr = torch.autograd.grad(l_rul, backbone, retain_graph=True, allow_unused=True)
                    ns = float(torch.sqrt(sum((t ** 2).sum() for t in gs if t is not None)))
                    nr = float(torch.sqrt(sum((t ** 2).sum() for t in gr if t is not None)))
                    grad_ratio.append({"epoch": epoch, "batch": bi, "gnorm_soh": ns,
                                       "gnorm_rul": nr, "ratio_rul_over_soh": nr / max(ns, 1e-12)})
                loss = l_soh + l_rul
                loss.backward()
            opt.step()
        if mode == "soh_only":
            model.eval()
            ps_, ts_ = [], []
            with torch.no_grad():
                for xb, ysb, _ in val_loader:
                    p, _ = model(xb, None, 0.0)
                    ps_.append(p.numpy().reshape(-1)); ts_.append(ysb.numpy().reshape(-1))
            val = float(np.sqrt(np.mean((np.concatenate(ps_) - np.concatenate(ts_)) ** 2)))
        else:
            val = _eval_rul_rmse_scale(model, val_loader, rul_scale)
        if val < best_val:
            best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    p_r, t_r, p_s, t_s = [], [], [], []
    with torch.no_grad():
        for xb, ysb, yrb in test_loader:
            ps, pr = model(xb, None, 0.0)
            p_r.append(pr.numpy().reshape(-1) * rul_scale)
            t_r.append(yrb.numpy().reshape(-1))
            p_s.append(ps.numpy().reshape(-1)); t_s.append(ysb.numpy().reshape(-1))
    pr = np.concatenate(p_r); tr = np.concatenate(t_r)
    ps = np.concatenate(p_s); ts = np.concatenate(t_s)
    return {
        "rul_rmse": float(regression_metrics(tr, pr)["rmse"]),
        "soh_rmse": float(regression_metrics(ts, ps)["rmse"]),
        "best_epoch": best_epoch,
        "pred_rul": pr, "true_rul": tr, "pred_soh": ps, "true_soh": ts,
        "grad_ratio": grad_ratio,
    }


def _eval_rul_rmse_scale(model, loader, scale):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, _, yrb in loader:
            _, pr = model(xb, None, 0.0)
            preds.append(pr.numpy().reshape(-1) * scale)
            trues.append(yrb.numpy().reshape(-1))
    return float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(trues)) ** 2)))


def summarize(records: list[dict]) -> dict:
    out = {}
    for cell in sorted({r["cell"] for r in records}):
        for level in ("L2", "L1P"):
            rows = [r for r in records if r["cell"] == cell and r["level"] == level]
            if not rows:
                continue
            # pool the three seeds; window identity = (t, seed)
            true = np.array([r["true"] for r in rows]); pred = np.array([r["pred"] for r in rows])
            err = pred - true
            soh = np.array([r["soh"] for r in rows]); c = np.array([r["c"] for r in rows])
            dist = np.array([r["dist"] for r in rows])
            c_tr = np.array([r["c_train_max"] for r in rows])
            rec = {
                "n": int(len(rows)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "bias_mean": float(np.mean(err)),
                "bias_median": float(np.median(err)),
                "corr_c_abs_err": float(np.corrcoef(c, np.abs(err))[0, 1]),
                "by_c_bin": {},
                "by_rul_bin": {},
            }
            c_bins = [(0.0, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 1.01)] if level == "L2" else [(0, 10), (10, 20), (20, 40), (40, 10000)]
            for lo, hi in c_bins:
                m = (c >= lo) & (c < hi)
                if not np.any(m):
                    continue
                key = f"{lo:.2f}-{hi:.2f}" if level == "L2" else f"{lo}-{hi}"
                rec["by_c_bin"][key] = {
                    "n": int(m.sum()),
                    "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                    "bias": float(np.mean(err[m])),
                    "mae": float(np.mean(np.abs(err[m]))),
                }
            for lo, hi in [(0, 10), (10, 30), (30, 80), (80, 10000)]:
                m = (true >= lo) & (true < hi)
                if not np.any(m):
                    continue
                rec["by_rul_bin"][f"{lo}-{hi}"] = {
                    "n": int(m.sum()),
                    "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                    "bias": float(np.mean(err[m])),
                }
            if level == "L1P":
                rec["match_dist_mean"] = float(dist.mean())
                rec["corr_dist_abs_err"] = float(np.corrcoef(dist, np.abs(err))[0, 1])
                rec["prior_rmse"] = float(np.sqrt(np.mean((c - true) ** 2)))
                rec["prior_bias"] = float(np.mean(c - true))
            else:
                tr_max = float(np.mean(c_tr))
                for tag, m in [("inside_support", c <= tr_max), ("beyond_support", c > tr_max)]:
                    if np.any(m):
                        rec[tag] = {
                            "n": int(m.sum()),
                            "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                            "bias": float(np.mean(err[m])),
                            "mae": float(np.mean(np.abs(err[m]))),
                        }
            out[f"{cell}:{level}"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--skip-soh", action="store_true")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    life = nasa_lifetime_table(DATA)
    records, summary_rows, soh_rows = [], [], []
    for cell in cells:
        for level in ("L2", "L1P"):
            d = cell_data(cell, level, life)
            for seed in seeds:
                print(f"=== {cell} {level} seed={seed} ===", flush=True)
                res = train_mlp(d, seed, args.epochs)
                n = len(res["pred_rul"])
                n_te = n
                t_pred = d["t_pred"][-n_te:]
                for i in range(n_te):
                    records.append({
                        "cell": cell, "level": level, "seed": seed,
                        "t": int(t_pred[i]), "soh": float(d["soh"][t_pred[i]]),
                        "c": float(d["c"][-n_te + i]),
                        "c_train_max": float(d["c_train_max"]),
                        "dist": float(d["match_dist"][-n_te + i]),
                        "true": float(res["true_rul"][i]),
                        "pred": float(res["pred_rul"][i]),
                    })
                summary_rows.append({"cell": cell, "level": level, "seed": seed,
                                     "rul_rmse": res["rul_rmse"], "soh_rmse": res["soh_rmse"],
                                     "best_epoch": res["best_epoch"]})
        if not args.skip_soh:
            d = cell_data(cell, "L2", life)
            for seed in seeds:
                print(f"=== {cell} dual-head paper seed={seed} ===", flush=True)
                r1 = train_mlp(d, seed, args.epochs, mode="paper")
                print(f"=== {cell} dual-head normalised seed={seed} ===", flush=True)
                r2 = train_mlp(d, seed, args.epochs, mode="normalized", rul_scale=d["t_target"])
                print(f"=== {cell} soh-only seed={seed} ===", flush=True)
                r3 = train_mlp(d, seed, args.epochs, mode="soh_only")
                soh_rows.append({
                    "cell": cell, "seed": seed,
                    "paper": {"rul_rmse": r1["rul_rmse"], "soh_rmse": r1["soh_rmse"]},
                    "normalized": {"rul_rmse": r2["rul_rmse"], "soh_rmse": r2["soh_rmse"]},
                    "soh_only": {"soh_rmse": r3["soh_rmse"]},
                    "grad_ratio": r1["grad_ratio"],
                })

    (OUT / "windows.json").write_text(json.dumps(records), encoding="utf-8")
    summary = {"by_cell": summarize(records),
               "seed_runs": summary_rows,
               "settings": {"epochs": args.epochs, "cells": cells, "seeds": seeds,
                            "window": 50, "horizon": 5}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if soh_rows:
        (OUT / "soh_head.json").write_text(json.dumps(soh_rows, indent=2), encoding="utf-8")

    lines = ["# L1P mechanism diagnostics\n\n"]
    lines.append("## Error structure (test windows, pooled seeds)\n\n")
    lines.append("| Cell:level | n | RMSE | MAE | bias | corr(c,|e|) |\n|---|---:|---:|---:|---:|---:|\n")
    for k, v in summary["by_cell"].items():
        lines.append(f"| {k} | {v['n']} | {v['rmse']:.2f} | {v['mae']:.2f} | {v['bias_mean']:+.2f} | {v['corr_c_abs_err']:+.2f} |\n")
    lines.append("\n## By cycle-channel bin\n\n")
    for k, v in summary["by_cell"].items():
        lines.append(f"### {k}\n\n| c bin | n | RMSE | bias |\n|---|---:|---:|---:|\n")
        for b, d in v["by_c_bin"].items():
            lines.append(f"| {b} | {d['n']} | {d['rmse']:.2f} | {d['bias']:+.2f} |\n")
        lines.append("\n")
    lines.append("## By true-RUL bin\n\n")
    for k, v in summary["by_cell"].items():
        lines.append(f"### {k}\n\n| RUL bin | n | RMSE | bias |\n|---|---:|---:|---:|\n")
        for b, d in v["by_rul_bin"].items():
            lines.append(f"| {b} | {d['n']} | {d['rmse']:.2f} | {d['bias']:+.2f} |\n")
        lines.append("\n")
    if soh_rows:
        lines.append("\n## SOH head control\n\n| Cell | paper SOH | normalised SOH | SOH-only | paper RUL | normalised RUL | mean g_rul/g_soh |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for cell in cells:
            rr = [r for r in soh_rows if r["cell"] == cell]
            if not rr:
                continue
            import statistics as st
            lines.append(
                f"| {cell} | {st.mean(x['paper']['soh_rmse'] for x in rr):.3f} | "
                f"{st.mean(x['normalized']['soh_rmse'] for x in rr):.3f} | "
                f"{st.mean(x['soh_only']['soh_rmse'] for x in rr):.3f} | "
                f"{st.mean(x['paper']['rul_rmse'] for x in rr):.2f} | "
                f"{st.mean(x['normalized']['rul_rmse'] for x in rr):.2f} | "
                f"{st.mean(y['ratio_rul_over_soh'] for x in rr for y in x['grad_ratio']):.1f} |\n")
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print("done ->", OUT)


if __name__ == "__main__":
    main()