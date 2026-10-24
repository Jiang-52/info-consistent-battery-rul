#!/usr/bin/env python3
"""Headline: RUL RMSE vs information level after T:=EOL definition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "outputs" / "info_levels_v2" / "agg.json"
CLASSICAL = ROOT / "outputs" / "info_levels_v2" / "classical.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LEVEL_ORDER = ["L2", "L1", "L0"]
LEVEL_LABELS = {
    "L2": "L2\n(oracle $T$)",
    "L1": "L1\n(sibling median $T$)",
    "L0": "L0\n(raw $t$)",
}

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


def pivot(agg, corpus, battery):
    out = {}
    for r in agg:
        if r.get("corpus") != corpus or r.get("battery") != battery:
            continue
        out.setdefault(r["model"], {})[r["info_level"]] = (
            float(r["rul_rmse"]),
            float(r.get("rul_rmse_std", 0.0)),
        )
    return out


def persist_fade(classical, corpus, battery):
    for r in classical:
        if (
            r.get("corpus") == corpus
            and r.get("battery") == battery
            and r.get("model") == "persist_fade"
        ):
            return float(r["rul_rmse"])
    return float("nan")


def panel(ax, agg, classical, corpus, battery, colors, markers):
    pv = pivot(agg, corpus, battery)
    x = np.arange(len(LEVEL_ORDER))
    for model, series in pv.items():
        ys, es = [], []
        for lv in LEVEL_ORDER:
            if lv not in series:
                ys.append(np.nan)
                es.append(0.0)
            else:
                ys.append(series[lv][0])
                es.append(series[lv][1])
        ax.errorbar(
            x,
            ys,
            yerr=es,
            label=model,
            color=colors.get(model, "#333"),
            marker=markers.get(model, "o"),
            lw=1.6,
            capsize=2.5,
            elinewidth=0.9,
        )
    pers = persist_fade(classical, corpus, battery)
    if np.isfinite(pers):
        ax.axhline(pers, color="#666666", ls="--", lw=1.1, label="persist_fade")
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[lv] for lv in LEVEL_ORDER])
    ax.set_title(f"{corpus.upper()} {battery}")
    ax.set_ylabel("RUL RMSE (cycles)")


def main() -> None:
    if not AGG.exists():
        print("missing", AGG)
        return
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    classical = json.loads(CLASSICAL.read_text(encoding="utf-8")) if CLASSICAL.exists() else []
    colors = {"patch-tf": "#B85C38", "mlp": "#0F4D92"}
    markers = {"patch-tf": "o", "mlp": "s"}

    nasa_cells = ["B0005", "B0006", "B0007"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.85), sharey=False, constrained_layout=True)
    for ax, bat in zip(axes, nasa_cells):
        panel(ax, agg, classical, "nasa", bat, colors, markers)
        ax.set_xlabel("Information level")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Ranking is a function of prediction-time information (NASA, $T$=EOL)", fontsize=10)
    fig.savefig(OUT / "fig_info_levels.pdf")
    fig.savefig(OUT / "fig_info_levels.png", dpi=300)
    plt.close(fig)

    calce_cells = sorted({r["battery"] for r in agg if r.get("corpus") == "calce"})
    if calce_cells:
        n = len(calce_cells)
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 1.2, 2.7), sharey=False, constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, bat in zip(axes, calce_cells):
            panel(ax, agg, classical, "calce", bat, colors, markers)
            ax.set_xlabel("Information level")
        axes[0].legend(fontsize=6.5)
        fig.suptitle("CALCE CS2: L1 uses leave-target median EOL (L1 $\\neq$ L2)", fontsize=10)
        fig.savefig(OUT / "fig_info_levels_calce.pdf")
        fig.savefig(OUT / "fig_info_levels_calce.png", dpi=300)
        plt.close(fig)
        print("wrote CALCE figure")
    print("wrote", OUT / "fig_info_levels.pdf")


if __name__ == "__main__":
    main()
