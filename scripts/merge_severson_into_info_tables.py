"""Merge Severson Protocol A/B ladder results into paper tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "info_levels_v2"
SEV_A = ROOT / "outputs" / "info_levels_severson"
SEV_B = ROOT / "outputs" / "info_levels_severson_loco"
OUT_TEX = ROOT / "paper" / "info_consistent_manuscript" / "generated_severson.tex"
SEL = SEV_A / "selection.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def summarize(agg: list[dict]) -> dict:
    by_level: dict[str, list[float]] = {}
    cells = sorted(
        {r["battery"] for r in agg if r.get("corpus") == "severson" and r.get("model") == "mlp"}
    )
    for lvl in ("L2", "L1", "L1P", "L0"):
        vals = []
        for c in cells:
            for r in agg:
                if (
                    r.get("corpus") == "severson"
                    and r.get("model") == "mlp"
                    and r.get("battery") == c
                    and r.get("info_level") == lvl
                ):
                    vals.append(float(r["rul_rmse"]))
                    break
        by_level[lvl] = vals
    d = []
    for c in cells:
        m = {}
        for r in agg:
            if r.get("corpus") == "severson" and r.get("model") == "mlp" and r.get("battery") == c:
                m[r["info_level"]] = float(r["rul_rmse"])
        if "L0" in m and "L2" in m:
            d.append(m["L0"] - m["L2"])
    wil = None
    try:
        from scipy.stats import wilcoxon

        if len(d) >= 2 and not np.allclose(d, 0):
            wil = float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
    except Exception:
        wil = None
    return {
        "n_cells": len(cells),
        "cells": cells,
        "median": {k: float(np.median(v)) if v else float("nan") for k, v in by_level.items()},
        "mean": {k: float(np.mean(v)) if v else float("nan") for k, v in by_level.items()},
        "delta_L0_minus_L2": d,
        "wilcoxon_p_L0_vs_L2": wil,
        "frac_L0_worse_than_L2": float(np.mean(np.asarray(d) > 0)) if d else float("nan"),
    }


def _cell_map(agg: list[dict], cell: str) -> dict[str, float]:
    m: dict[str, float] = {}
    for r in agg:
        if r.get("corpus") == "severson" and r.get("model") == "mlp" and r.get("battery") == cell:
            m[r["info_level"]] = float(r["rul_rmse"])
    return m


def _picks_by_T(cells: list[str]) -> list[str]:
    sel = _load(SEL)
    order: list[str] = []
    for key in ("short", "mid", "long"):
        for item in sel.get(key, []):
            cid = item["id"] if isinstance(item, dict) else item
            if cid in cells and cid not in order:
                order.append(cid)
    if not order:
        order = list(cells)
    if len(order) >= 9:
        return order[:3] + order[len(order) // 2 - 1 : len(order) // 2 + 2] + order[-3:]
    return order


def classical_medians(path: Path) -> dict[str, float]:
    rows = _load(path)
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(float(r["rul_rmse"]))
    return {k: float(np.median(v)) for k, v in by.items() if v}


def write_tex(sum_a: dict, agg_a: list[dict], sum_b: dict, agg_b: list[dict]) -> None:
    med_a, med_b = sum_a["median"], sum_b["median"]
    base_b = classical_medians(SEV_B / "classical.json")
    lines: list[str] = []
    lines.append("% auto-generated Severson Protocol A/B summary\n")

    # Protocol A (supporting)
    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append(
        "\\caption{Severson Protocol~A within-cell retrospective evaluation "
        f"(chronological fit; MLP; stratified {sum_a['n_cells']} cells; "
        "leave-target $T^{\\mathrm{prior}}$ only). "
        "Cross-cell median RUL RMSE (cycles). Support comparison.}\n"
    )
    lines.append("\\label{tab:severson_A}\n")
    lines.append("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
    lines.append("Fit & L2 & L1 & L1P & L0 \\\\\n\\midrule\n")
    lines.append(
        f"Protocol~A ($n={sum_a['n_cells']}$) & "
        f"{med_a['L2']:.1f} & {med_a['L1']:.1f} & {med_a['L1P']:.1f} & {med_a['L0']:.1f} \\\\\n"
    )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # Protocol B (main)
    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append(
        "\\caption{Severson Protocol~B strict leave-target-cell-out evidence "
        f"(MLP; same stratified {sum_b['n_cells']} cells and same late-20\\% test windows; "
        "train/val on sibling cells only; L1/L1P from the leave-target 119-cell pool). "
        "Cross-cell median RUL RMSE (cycles). Main evidence.}\n"
    )
    lines.append("\\label{tab:severson}\n")
    lines.append("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
    lines.append("Fit & L2 & L1 & L1P & L0 \\\\\n\\midrule\n")
    lines.append(
        f"Protocol~B ($n={sum_b['n_cells']}$) & "
        f"{med_b['L2']:.1f} & {med_b['L1']:.1f} & {med_b['L1P']:.1f} & {med_b['L0']:.1f} \\\\\n"
    )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # L0-valid classical foils on the same Protocol B test windows
    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append(
        "\\caption{Severson L0-valid classical foils on the same late-20\\% test windows "
        f"as Protocol~B ($n={sum_b['n_cells']}$). Cross-cell median RUL RMSE (cycles). "
        "persist\\_hold $=1$ is an oracle label walk, not an L0 foil.}\n"
    )
    lines.append("\\label{tab:severson_base}\n")
    lines.append("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n")
    lines.append("Baseline & persist\\_fade & linear & persist\\_recursive \\\\\n\\midrule\n")
    lines.append(
        f"Protocol~B test windows & "
        f"{base_b.get('persist_fade', float('nan')):.1f} & "
        f"{base_b.get('linear', float('nan')):.1f} & "
        f"{base_b.get('persist_recursive', float('nan')):.1f} \\\\\n"
    )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    p_b = sum_b["wilcoxon_p_L0_vs_L2"]
    p_a = sum_a["wilcoxon_p_L0_vs_L2"]
    lines.append(
        f"% Protocol A Wilcoxon L0 vs L2: p={p_a}, frac={sum_a['frac_L0_worse_than_L2']:.2f}\n"
        f"% Protocol B Wilcoxon L0 vs L2: p={p_b}, frac={sum_b['frac_L0_worse_than_L2']:.2f}\n"
        f"% Protocol B classical medians: {base_b}\n"
    )

    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append(
        "\\caption{Severson Protocol~B per-cell MLP RUL RMSE excerpt (mean over three seeds). "
        "Short / mid / long $T$ strata.}\n"
    )
    lines.append("\\label{tab:severson_cells}\n")
    lines.append("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
    lines.append("Cell & L2 & L1 & L1P & L0 \\\\\n\\midrule\n")
    for c in _picks_by_T(sum_b["cells"]):
        m = _cell_map(agg_b, c)
        if len(m) < 3:
            continue
        lines.append(
            f"{c} & {m.get('L2', float('nan')):.1f} & {m.get('L1', float('nan')):.1f} & "
            f"{m.get('L1P', float('nan')):.1f} & {m.get('L0', float('nan')):.1f} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    OUT_TEX.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    agg_a = _load(SEV_A / "agg.json")
    agg_b = _load(SEV_B / "agg.json")
    if not agg_a:
        raise SystemExit(f"No Protocol A results in {SEV_A}")
    if not agg_b:
        raise SystemExit(f"No Protocol B results in {SEV_B}")
    sum_a = summarize(agg_a)
    sum_b = summarize(agg_b)
    (SEV_A / "summary.json").write_text(json.dumps(sum_a, indent=2), encoding="utf-8")
    (SEV_B / "summary.json").write_text(json.dumps(sum_b, indent=2), encoding="utf-8")
    write_tex(sum_a, agg_a, sum_b, agg_b)
    merged = [r for r in _load(BASE / "agg.json") if r.get("corpus") != "severson"] + agg_b
    (BASE / "agg_with_severson.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("Protocol A median", sum_a["median"], "p", sum_a["wilcoxon_p_L0_vs_L2"])
    print("Protocol B median", sum_b["median"], "p", sum_b["wilcoxon_p_L0_vs_L2"])
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
