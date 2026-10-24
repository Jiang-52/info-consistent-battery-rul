#!/usr/bin/env python3
"""NASA Protocol A: L2 vs L1 vs L1P vs L0 paired bars (MLP)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "outputs" / "info_levels_v2" / "agg.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = ["L2", "L1", "L1P", "L0"]
COLORS = {"L2": "#4C78A8", "L1": "#72B7B2", "L1P": "#B85C38", "L0": "#8C8C8C"}
CELLS = ["B0005", "B0006", "B0007"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def main():
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    vals = {c: {} for c in CELLS}
    err = {c: {} for c in CELLS}
    for r in agg:
        if r.get("corpus") not in (None, "nasa") and r.get("corpus") != "nasa":
            if str(r.get("corpus", "")).lower() != "nasa":
                continue
        if str(r.get("model", "")).lower() != "mlp":
            continue
        bat = r.get("battery")
        lv = r.get("info_level")
        if bat in vals and lv in LEVELS:
            vals[bat][lv] = float(r["rul_rmse"])
            err[bat][lv] = float(r.get("rul_rmse_std", 0.0))

    x = np.arange(len(CELLS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, lv in enumerate(LEVELS):
        ys = [vals[c].get(lv, np.nan) for c in CELLS]
        es = [err[c].get(lv, 0.0) for c in CELLS]
        ax.bar(
            x + (i - 1.5) * width,
            ys,
            width,
            yerr=es,
            label=lv,
            color=COLORS[lv],
            capsize=2,
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("NASA Protocol A: a non-affine local prior (L1P) vs target-life oracle (L2)")
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_l1p_nasa.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_l1p_nasa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_l1p_nasa", vals)


if __name__ == "__main__":
    main()
