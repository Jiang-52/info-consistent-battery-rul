#!/usr/bin/env python3
"""Fig: NASA SOH trajectories and EOL heterogeneity (explains L1 vs L2 and B0007 persistence)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.nasa_loader import load_battery_soh
from src.data.rul_dataset import eol_cycle_index

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

COLORS = {"B0005": "#0F4D92", "B0006": "#B85C38", "B0007": "#5B8C5A"}


def main() -> None:
    data_dir = ROOT / "data" / "nasa"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    ax = axes[0]
    eols = {}
    for bat in ("B0005", "B0006", "B0007"):
        _, _, soh = load_battery_soh(data_dir / f"{bat}.mat", bat)
        eol = int(eol_cycle_index(soh, 0.7))
        eols[bat] = eol
        t = np.arange(len(soh))
        ax.plot(t, soh, color=COLORS[bat], lw=1.6, label=f"{bat} (EOL={eol})")
        ax.axvline(eol, color=COLORS[bat], ls=":", lw=0.9, alpha=0.8)
    ax.axhline(0.7, color="#666666", ls="--", lw=1.0, label="EOL threshold (0.7)")
    ax.set_xlabel("Discharge cycle $t$")
    ax.set_ylabel("SOH")
    ax.set_title("A  Lifetime $T$ is EOL, not record length")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_ylim(0.55, 1.02)

    ax = axes[1]
    _, _, soh7 = load_battery_soh(data_dir / "B0007.mat", "B0007")
    _, _, soh5 = load_battery_soh(data_dir / "B0005.mat", "B0005")
    t7, t5 = np.arange(len(soh7)), np.arange(len(soh5))
    ax.plot(t5, soh5, color=COLORS["B0005"], lw=1.4, label="B0005")
    ax.plot(t7, soh7, color=COLORS["B0007"], lw=1.4, label="B0007")
    d7 = np.diff(soh7)
    up = np.where(d7 > 1e-4)[0]
    ax.scatter(up + 1, soh7[up + 1], s=10, c=COLORS["B0007"], alpha=0.5, zorder=3)
    ax.axhline(0.7, color="#666666", ls="--", lw=1.0)
    ax.set_xlim(120, 168)
    ax.set_ylim(0.68, 0.90)
    ax.set_xlabel("Discharge cycle $t$ (late life)")
    ax.set_ylabel("SOH")
    ax.set_title("B  B0007 stays above 0.7; fade-rate persistence fails")
    ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("NASA cells share record length 168 but not EOL", fontsize=10)
    pdf = OUT / "fig_soh_trajectories.pdf"
    png = OUT / "fig_soh_trajectories.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("wrote", pdf, "eols", eols)


if __name__ == "__main__":
    main()
