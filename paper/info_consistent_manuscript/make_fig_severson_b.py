#!/usr/bin/env python3
"""Severson Protocol B information ladder (primary evidence figure)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs" / "info_levels_severson_loco" / "raw.json"
AGG = ROOT / "outputs" / "info_levels_severson_loco" / "agg.json"
SUMMARY = ROOT / "outputs" / "info_levels_severson_loco" / "summary.json"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = ["L0", "L1", "L1P", "L2"]
LABELS = ["L0\n(raw $t$)", "L1\n(prior $T$)", "L1P\n(SOH match)", "L2\n(oracle $T$)"]

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


def cell_level_rmse(raw: list[dict]) -> dict[str, dict[str, float]]:
    """Prefer agg/summary; fall back to mean over seeds in raw."""
    by = {}
    for r in raw:
        if r.get("model") not in (None, "mlp") and r.get("model") != "mlp":
            # keep mlp only
            if str(r.get("model", "")).lower() != "mlp":
                continue
        bat = r.get("battery") or r.get("cell_id") or r.get("cell")
        lv = r.get("info_level")
        if bat is None or lv is None or "rul_rmse" not in r:
            continue
        by.setdefault(bat, {}).setdefault(lv, []).append(float(r["rul_rmse"]))
    return {c: {lv: float(np.mean(v)) for lv, v in d.items()} for c, d in by.items()}


def main():
    if not AGG.exists():
        raise SystemExit(f"missing {AGG}")
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    series = {}
    for r in agg:
        if str(r.get("model", "mlp")).lower() != "mlp":
            continue
        bat = r.get("battery")
        lv = r.get("info_level")
        if bat and lv and "rul_rmse" in r:
            series.setdefault(bat, {})[lv] = float(r["rul_rmse"])
    if not series:
        raise SystemExit("no MLP cell series in agg.json")

    # Build matrix
    cell_ids = sorted(series.keys())
    data = []
    for lv in LEVELS:
        data.append([series[c].get(lv, np.nan) for c in cell_ids])
    data = np.asarray(data, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    # paired lines L0 -> L2
    x = np.arange(len(LEVELS))
    for j, c in enumerate(cell_ids):
        ys = [series[c].get(lv, np.nan) for lv in LEVELS]
        ax.plot(x, ys, color="#B0B0B0", linewidth=0.7, alpha=0.55, zorder=1)
        ax.scatter(x, ys, color="#4C78A8", s=14, alpha=0.7, zorder=2)

    # boxplot overlay
    bp = ax.boxplot(
        [data[i][~np.isnan(data[i])] for i in range(len(LEVELS))],
        positions=x,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#B85C38", linewidth=1.6),
        boxprops=dict(facecolor="#E8F0F8", edgecolor="#4C78A8", linewidth=1.0),
        whiskerprops=dict(color="#4C78A8"),
        capprops=dict(color="#4C78A8"),
        zorder=3,
    )

    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text(encoding="utf-8"))
        med = s.get("median") or {}
        meds = [med.get(lv) for lv in LEVELS]
        if all(v is not None for v in meds):
            ax.plot(
                x,
                meds,
                "o-",
                color="#B85C38",
                linewidth=2.0,
                markersize=6,
                label="cross-cell median",
                zorder=4,
            )
            ax.legend(loc="upper right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_xlabel("Information level")
    ax.set_title("Severson Protocol B ($n{=}24$ cells): information ladder")
    # annotate all L0 > L2
    n_ok = sum(
        1
        for c in cell_ids
        if series[c].get("L0", 0) > series[c].get("L2", 1e9)
    )
    ax.text(
        0.02,
        0.98,
        f"L0 $>$ L2 on {n_ok}/{len(cell_ids)} cells",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_severson_protocol_b.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_severson_protocol_b.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_severson_protocol_b")


if __name__ == "__main__":
    main()
