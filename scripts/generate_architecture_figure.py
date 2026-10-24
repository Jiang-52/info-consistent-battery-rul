"""Generate system architecture block diagram for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "system_architecture.png"


def box(ax, xy, w, h, text, fc="#E8F4FD", ec="#1A5276", fontsize=8):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p0, p1, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.75,
        "TCN-Attention Prognostics Framework (Primary + Auxiliary Branch)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # Primary branch
    ax.text(0.35, 5.15, "Primary SOH/RUL branch", fontsize=9, fontweight="bold", color="#1A5276")
    box(ax, (0.3, 4.35), 1.5, 0.65, "SOH window\n$\\mathbf{x}_{t-W:t}$", fc="#FDEBD0")
    box(ax, (2.2, 4.35), 1.55, 0.65, "TCN encoder\n(dilated causal conv)", fc="#E8F4FD")
    box(ax, (4.15, 4.55), 1.35, 0.45, "Channel\nAttention", fc="#D5F5E3")
    box(ax, (4.15, 3.95), 1.35, 0.45, "TPA", fc="#D5F5E3")
    box(ax, (5.85, 4.35), 1.45, 0.65, "Context\n$\\mathbf{c}$", fc="#E8F4FD")
    box(ax, (7.6, 4.55), 1.55, 0.45, "GRU Seq2Seq\ndecoder", fc="#FADBD8")
    box(ax, (7.6, 3.85), 1.55, 0.45, "RUL head", fc="#FADBD8")
    box(ax, (9.35, 4.55), 0.55, 0.45, "$\\hat{y}_{SOH}$", fc="#FFFFFF", fontsize=7)
    box(ax, (9.35, 3.85), 0.55, 0.45, "$\\hat{y}_{RUL}$", fc="#FFFFFF", fontsize=7)

    arrow(ax, (1.8, 4.675), (2.2, 4.675))
    arrow(ax, (3.75, 4.675), (4.15, 4.775))
    arrow(ax, (3.75, 4.675), (4.15, 4.175))
    arrow(ax, (5.5, 4.775), (5.85, 4.775))
    arrow(ax, (5.5, 4.175), (5.85, 4.575))
    arrow(ax, (5.5, 4.775), (5.85, 4.675))
    arrow(ax, (5.5, 4.175), (5.85, 4.675))
    arrow(ax, (7.3, 4.675), (7.6, 4.775))
    arrow(ax, (7.3, 4.675), (7.6, 4.075))
    arrow(ax, (9.15, 4.775), (9.35, 4.775))
    arrow(ax, (9.15, 4.075), (9.35, 4.075))

    ax.annotate(
        "Iterative window roll\n(Algorithm 1)",
        xy=(8.4, 3.55),
        xytext=(6.0, 2.95),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="#666666"),
    )

    # Auxiliary branch
    ax.text(0.35, 2.55, "Auxiliary anomaly branch (separate loss)", fontsize=9, fontweight="bold", color="#7D3C98")
    box(ax, (0.3, 1.55), 1.7, 0.65, "Cycle-feature\nwindow", fc="#FDEBD0")
    box(ax, (2.35, 1.55), 1.65, 0.65, "TCN\nAutoencoder", fc="#E8DAEF")
    box(ax, (4.35, 1.55), 1.55, 0.65, "Reconstruction\nMSE", fc="#E8DAEF")
    box(ax, (6.25, 1.55), 1.45, 0.65, "$p_{95}$\nthreshold", fc="#F5EEF8")
    box(ax, (8.05, 1.55), 1.35, 0.65, "Anomaly\nflag", fc="#FFFFFF")

    arrow(ax, (2.0, 1.875), (2.35, 1.875))
    arrow(ax, (4.0, 1.875), (4.35, 1.875))
    arrow(ax, (5.9, 1.875), (6.25, 1.875))
    arrow(ax, (7.7, 1.875), (8.05, 1.875))

    ax.text(
        5,
        0.55,
        "Shared preprocessing: NASA/CALCE capacity $\\rightarrow$ SOH; chronological 80/20 split; "
        "auxiliary branch uses synthetic Gaussian spikes for feasibility validation only.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#444444",
        wrap=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
