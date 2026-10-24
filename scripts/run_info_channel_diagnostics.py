#!/usr/bin/env python3
"""Channel diagnostics on chronological test windows (no NN training).

Checks L0 countdown, L1/L2 collinearity, and a linear readout vs L0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_info_levels_v2 import (  # noqa: E402
    NASA_CELLS,
    calce_lifetime_table,
    load_calce_rul_windows,
    load_nasa_windows,
    nasa_lifetime_table,
    restrict_pre_eol,
    run_classical,
    sibling_t_eol,
    t_scale_for,
)
from src.data.calce_loader import list_calce_cells  # noqa: E402
from src.data.nasa_loader import chronological_split  # noqa: E402

OUT = ROOT / "outputs" / "info_levels_v2" / "channel_diagnostics.json"
WINDOW, HORIZON = 50, 5


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ols_rmse(x: np.ndarray, y: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    """Least squares with intercept. x is (n, p)."""
    n = len(y)
    xb = np.concatenate([np.ones((n, 1)), x], axis=1)
    xt = np.concatenate([np.ones((len(y_te), 1)), x_te], axis=1)
    try:
        beta, *_ = np.linalg.lstsq(xb, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    pred = xt @ beta
    return float(np.sqrt(np.mean((pred - y_te) ** 2)))


def last_chan(x: np.ndarray, idx: int) -> np.ndarray:
    return np.asarray(x[:, -1, idx], dtype=np.float64).reshape(-1)


def diagnose_cell(data_l0, data_l1, data_l2, data_l1p=None) -> dict:
    def prep(d):
        d = restrict_pre_eol(d)
        x, y = d["x"], np.asarray(d["y_rul"], dtype=np.float64).reshape(-1)
        xtr, xte, ytr, yte = chronological_split(x, y, 0.8)
        return xtr, xte, ytr, yte

    x0tr, x0te, ytr, yte = prep(data_l0)
    x1tr, x1te, _, _ = prep(data_l1)
    x2tr, x2te, _, _ = prep(data_l2)

    c0_tr, c0_te = last_chan(x0tr, 5), last_chan(x0te, 5)
    c1_tr, c1_te = last_chan(x1tr, 5), last_chan(x1te, 5)
    c2_tr, c2_te = last_chan(x2tr, 5), last_chan(x2te, 5)

    row = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "spearman_L0_t_vs_rul_test": spearman(c0_te, yte),
        "pearson_L0_t_vs_rul_test": pearson(c0_te, yte),
        "spearman_L2_frac_vs_rul_test": spearman(c2_te, yte),
        "pearson_L1_vs_L2_test": pearson(c1_te, c2_te),
        "spearman_L1_vs_L2_test": spearman(c1_te, c2_te),
        "ols_rmse_L0_clock": ols_rmse(c0_tr.reshape(-1, 1), ytr, c0_te.reshape(-1, 1), yte),
        "ols_rmse_L1_clock": ols_rmse(c1_tr.reshape(-1, 1), ytr, c1_te.reshape(-1, 1), yte),
        "ols_rmse_L2_clock": ols_rmse(c2_tr.reshape(-1, 1), ytr, c2_te.reshape(-1, 1), yte),
        "ols_rmse_L0_drop5": ols_rmse(
            x0tr[:, -1, :5].reshape(len(ytr), -1),
            ytr,
            x0te[:, -1, :5].reshape(len(yte), -1),
            yte,
        ),
        "ols_rmse_L0_all6": ols_rmse(
            x0tr[:, -1, :].reshape(len(ytr), -1),
            ytr,
            x0te[:, -1, :].reshape(len(yte), -1),
            yte,
        ),
        "mean_rul_test": float(yte.mean()),
        "std_rul_test": float(yte.std()),
    }
    if data_l1p is not None:
        xp_tr, xp_te, _, _ = prep(data_l1p)
        cp_tr, cp_te = last_chan(xp_tr, 5), last_chan(xp_te, 5)
        row["pearson_L1P_vs_L2_test"] = pearson(cp_te, c2_te)
        row["pearson_L1P_vs_rul_test"] = pearson(cp_te, yte)
        row["spearman_L1P_vs_rul_test"] = spearman(cp_te, yte)
        row["ols_rmse_L1P"] = ols_rmse(cp_tr.reshape(-1, 1), ytr, cp_te.reshape(-1, 1), yte)
    return row


def cell_level_wilcoxon(agg_path: Path) -> list[dict]:
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    from collections import defaultdict

    by = defaultdict(dict)
    for r in agg:
        if r.get("model") != "mlp":
            continue
        by[(r["corpus"], r["battery"])][r["info_level"]] = float(r["rul_rmse"])
    e_l0 = []
    e_l1 = []
    cells = []
    for key, d in sorted(by.items()):
        if "L2" not in d or "L0" not in d:
            continue
        cells.append({"corpus": key[0], "cell": key[1], "L2": d["L2"], "L1": d.get("L1"), "L0": d["L0"]})
        e_l0.append(d["L0"] - d["L2"])
        if "L1" in d:
            e_l1.append(d["L1"] - d["L2"])
    out = {"n_cells": len(cells), "cells": cells, "delta_L0_minus_L2": e_l0, "delta_L1_minus_L2": e_l1}
    try:
        from scipy.stats import wilcoxon

        if len(e_l0) >= 6:
            out["wilcoxon_L0_vs_L2_p"] = float(wilcoxon(e_l0).pvalue)
        if len(e_l1) >= 6:
            out["wilcoxon_L1_vs_L2_p"] = float(wilcoxon(e_l1).pvalue)
    except Exception:
        out["wilcoxon_L0_vs_L2_p"] = None
    return out


def main() -> None:
    data_dir = ROOT / "data" / "nasa"
    npy = ROOT / "data" / "calce" / "CALCE.npy"
    nasa_life = nasa_lifetime_table(data_dir)
    nasa_ids = [p.stem.upper() for p in sorted(data_dir.glob("B*.mat"))] or list(NASA_CELLS)
    calce_life = calce_lifetime_table(npy)
    calce_ids = list(calce_life)

    rows = []
    persist_rows = []
    jobs = [("nasa", c, nasa_life) for c in nasa_ids if c in nasa_life]
    jobs += [("calce", c, calce_life) for c in sorted(calce_ids)]

    for corpus, cell_id, table in jobs:
        scales = sibling_t_eol(table, cell_id)
        loaders = {}
        for level in ("L0", "L1", "L2", "L1P"):
            ts = t_scale_for(level, scales)
            if corpus == "nasa":
                loaders[level] = load_nasa_windows(data_dir, cell_id, WINDOW, HORIZON, level, ts)
            else:
                loaders[level] = load_calce_rul_windows(npy, cell_id, WINDOW, HORIZON, level, ts)
        rec = diagnose_cell(loaders["L0"], loaders["L1"], loaders["L2"], loaders["L1P"])
        rec.update(
            {
                "corpus": corpus,
                "cell": cell_id,
                "t_eol": scales["t_target"],
                "t_med_train": scales["t_med"],
                "t_med_all_illegal": scales["t_med_all_illegal"],
                "used_target_in_prior": scales["used_target_in_prior"],
            }
        )
        d0 = restrict_pre_eol(loaders["L0"])
        persist_rows.extend(run_classical(d0, cell_id, "level-invariant", corpus, WINDOW, HORIZON))
        rows.append(rec)
        print(
            json.dumps(
                {
                    "cell": cell_id,
                    "rho_L1_L2": rec["pearson_L1_vs_L2_test"],
                    "rho_L1P_L2": rec.get("pearson_L1P_vs_L2_test"),
                    "ols_L1P": rec.get("ols_rmse_L1P"),
                    "ols_L0": rec["ols_rmse_L0_clock"],
                },
                ensure_ascii=False,
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    persist_path = OUT.parent / "classical_l0valid.json"
    persist_path.write_text(json.dumps(persist_rows, indent=2), encoding="utf-8")
    agg_path = OUT.parent / "agg.json"
    if agg_path.exists():
        cell_stats = cell_level_wilcoxon(agg_path)
        (OUT.parent / "cell_stats.json").write_text(json.dumps(cell_stats, indent=2), encoding="utf-8")
        print("cell n=", cell_stats.get("n_cells"), "p L0-L2", cell_stats.get("wilcoxon_L0_vs_L2_p"))
    print("wrote", OUT, persist_path)


if __name__ == "__main__":
    main()
