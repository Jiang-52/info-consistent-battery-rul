
#!/usr/bin/env python3
"""Mechanism figure for Finding 2 (L1P vs L2 under NASA Protocol A)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "l1p_mechanism" / "windows.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"L2": "#4C78A8", "L1P": "#B85C38", "true": "#222222"}
CELL_MARK = {"B0005": "o", "B0006": "s", "B0007": "^"}

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
    rec = json.loads(SRC.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in rec:
        by[(r["cell"], r["level"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))

    # (A) B0005 trajectory: true vs L2 vs L1P, mean over seeds
    ax = axes[0]
    cell = "B0005"
    for level, color in (("L2", COLORS["L2"]), ("L1P", COLORS["L1P"])):
        rows = by[(cell, level)]
        ts = sorted({r["t"] for r in rows})
        mu, sd, tr = [], [], []
        for t in ts:
            p = np.array([r["pred"] for r in rows if r["t"] == t])
            y = np.array([r["true"] for r in rows if r["t"] == t])
            mu.append(p.mean()); sd.append(p.std()); tr.append(y.mean())
        ax.fill_between(ts, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd),
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(ts, mu, color=color, lw=1.6, label=f"{level} prediction")
    ax.plot(ts, tr, color=COLORS["true"], lw=1.4, ls="--", label="true RUL")
    ax.set_xlabel("Cycle $t$ (test window)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("(A) B0005: L2 over-predicts late life")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_ylim(bottom=0)

    # (B) L2 signed error vs extrapolation distance beyond training support
    ax = axes[1]
    for cell in ("B0005", "B0006", "B0007"):
        rows = by[(cell, "L2")]
        x = np.array([r["c"] - r["c_train_max"] for r in rows])
        y = np.array([r["pred"] - r["true"] for r in rows])
        ax.scatter(x * 100, y, s=12, alpha=0.55, marker=CELL_MARK[cell],
                   color=COLORS["L2"], edgecolor="none", label=cell)
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("L2 channel beyond train max (percentage points of $c$)")
    ax.set_ylabel("Signed error, pred $-$ true (cycles)")
    ax.set_title("(B) L2: error grows outside support")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # (C) RMSE by true-RUL bin: where each level fails
    ax = axes[2]
    bins = [(0, 10), (10, 30)]
    width = 0.19
    xs = np.arange(len(CELL_MARK))
    for bi, (lo, hi) in enumerate(bins):
        for li, (level, hatch) in enumerate((("L2", ""), ("L1P", "///"))):
            vals = []
            for cell in CELL_MARK:
                rows = [r for r in by[(cell, level)] if lo <= r["true"] < hi]
                err = np.array([r["pred"] - r["true"] for r in rows])
                vals.append(float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan)
            pos = xs + (bi - 0.5) * 2 * width + (li - 0.5) * width
            ax.bar(pos, vals, width, color=COLORS[level] if li == 0 else "white",
                   edgecolor=COLORS[level], hatch=hatch, linewidth=1.0,
                   label=f"{level} $<$10" if bi == 0 else f"{level} 10--30")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(CELL_MARK))
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("(C) L2 fails where RUL is short")
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, fontsize=6.6, ncol=4, loc="upper center",
              handlelength=1.2, columnspacing=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_l1p_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_l1p_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_l1p_mechanism.pdf")


if __name__ == "__main__":
    main()