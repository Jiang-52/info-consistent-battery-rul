#!/usr/bin/env python3
"""Headline figure: Information level (L2→L1→L0) → RUL RMSE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "outputs" / "info_levels" / "info_levels_agg.json"
CLASSICAL = ROOT / "outputs" / "info_levels" / "info_levels_classical.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LEVEL_ORDER = ["L2", "L1", "L0"]
LEVEL_LABELS = {
    "L2": "L2\n(oracle)",
    "L1": "L1\n(pop. prior)",
    "L0": "L0\n(pred.-time)",
}


def load_pivot(agg: list[dict], battery: str) -> dict[str, dict[str, tuple[float, float]]]:
    """model -> level -> (mean, std)"""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for r in agg:
        if r.get("battery") != battery:
            continue
        if r.get("experiment") != "info_levels":
            continue
        m = r["model"]
        out.setdefault(m, {})[r["info_level"]] = (
            float(r["rul_rmse"]),
            float(r.get("rul_rmse_std", 0.0)),
        )
    return out


def persistence_rmse(classical: list[dict], battery: str) -> float:
    for r in classical:
        if r.get("model") == "persistence" and r.get("battery") == battery:
            return float(r["rul_rmse"])
    return float("nan")


def make_figure() -> Path | None:
    if not AGG.exists():
        print("missing", AGG)
        return None
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    classical = (
        json.loads(CLASSICAL.read_text(encoding="utf-8")) if CLASSICAL.exists() else []
    )
    colors = {
        "patch-tf": "#B85C38",
        "mlp": "#0F4D92",
        "lstm-direct": "#5B8C5A",
        "tcn-direct": "#6B5B95",
    }
    markers = {"patch-tf": "o", "mlp": "s", "lstm-direct": "^", "tcn-direct": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=False)
    for ax, bat in zip(axes, ("B0005", "B0007")):
        pivot = load_pivot(agg, bat)
        if not pivot:
            ax.set_title(f"{bat} (no data)")
            continue
        x = np.arange(len(LEVEL_ORDER))
        for model, series in pivot.items():
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
                color=colors.get(model, "#333333"),
                marker=markers.get(model, "o"),
                linewidth=1.8,
                capsize=3,
            )
        pers = persistence_rmse(classical, bat)
        if np.isfinite(pers):
            ax.axhline(pers, color="#666666", linestyle="--", linewidth=1.2, label="persistence")
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS[lv] for lv in LEVEL_ORDER])
        ax.set_xlabel("Information level")
        ax.set_ylabel("RUL RMSE (cycles)")
        ax.set_title(bat)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Model ranking vs prediction-time information availability",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    pdf = OUT / "fig_info_levels.pdf"
    png = OUT / "fig_info_levels.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    return pdf


if __name__ == "__main__":
    make_figure()
