#!/usr/bin/env python3
"""Write LaTeX result tables from outputs/info_levels_v2 JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "info_levels_v2"
OUT = ROOT / "paper" / "info_consistent_manuscript" / "generated_tables.tex"


def tex_cell_id(name: str) -> str:
    return str(name).replace("_", "\\_")


def cell(mean, std=None):
    if mean is None:
        return "---"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f}$\\pm${std:.2f}"


def find_agg(agg, corpus, model, bat, level):
    for r in agg:
        if (
            r.get("corpus") == corpus
            and r.get("model") == model
            and r.get("battery") == bat
            and r.get("info_level") == level
        ):
            return r
    return {}


def main() -> None:
    agg = json.loads((SRC / "agg.json").read_text(encoding="utf-8"))
    classical_path = SRC / "classical_l0valid.json"
    if not classical_path.exists():
        classical_path = SRC / "classical.json"
    classical = json.loads(classical_path.read_text(encoding="utf-8"))
    stats = json.loads((SRC / "stats.json").read_text(encoding="utf-8"))
    lines = []
    lines.append("% auto-generated; do not edit by hand\n")

    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append("\\caption{Information ladder RUL RMSE (cycles; mean$\\pm$s.d.\\ over three seeds). ")
    lines.append("$T$ is the EOL cycle (SOH$\\le 0.7$). L0 uses raw $t$. ")
    lines.append("L1 uses $t/T^{\\mathrm{prior}}$ from leave-target training cells. L1P is MLP only: SOH-matched sibling RUL.}\n")
    lines.append("\\label{tab:ladder}\n")
    lines.append("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n")
    lines.append("Model & Cell & L2 & L1 & L1P & L0 \\\\\n\\midrule\n")
    for corpus, cells in (
        ("nasa", ["B0005", "B0006", "B0007"]),
        ("calce", ["CS2\\_35", "CS2\\_36", "CS2\\_37", "CS2\\_38"]),
    ):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{corpus.upper()}}}}} \\\\\n")
        for model in ("mlp", "patch-tf"):
            for bat in cells:
                key = bat.replace("\\", "")
                a2 = find_agg(agg, corpus, model, key, "L2")
                a1 = find_agg(agg, corpus, model, key, "L1")
                ap = find_agg(agg, corpus, model, key, "L1P")
                a0 = find_agg(agg, corpus, model, key, "L0")
                lp = cell(ap.get("rul_rmse"), ap.get("rul_rmse_std")) if ap else "---"
                lines.append(
                    f"{model} & {bat} & {cell(a2.get('rul_rmse'), a2.get('rul_rmse_std'))} & "
                    f"{cell(a1.get('rul_rmse'), a1.get('rul_rmse_std'))} & {lp} & "
                    f"{cell(a0.get('rul_rmse'), a0.get('rul_rmse_std'))} \\\\\n"
                )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append("\\caption{SOH RMSE (dimensionless; mean$\\pm$s.d.\\ over three seeds) under the same information ladder.}\n")
    lines.append("\\label{tab:soh}\n")
    lines.append("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n")
    lines.append("Model & Cell & L2 & L1 & L0 \\\\\n\\midrule\n")
    for corpus, cells in (("nasa", ["B0005", "B0006", "B0007"]),):
        for model in ("mlp", "patch-tf"):
            for bat in cells:
                a2 = find_agg(agg, corpus, model, bat, "L2")
                a1 = find_agg(agg, corpus, model, bat, "L1")
                a0 = find_agg(agg, corpus, model, bat, "L0")
                def sc(a):
                    if not a or "soh_rmse" not in a:
                        return "---"
                    return f"{a['soh_rmse']:.4f}$\\pm${a.get('soh_rmse_std', 0):.4f}"
                lines.append(f"{model} & {bat} & {sc(a2)} & {sc(a1)} & {sc(a0)} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append("\\caption{Classical RUL baselines on chronological test windows with RUL$>0$. ")
    lines.append("persist\\_fade: SOH fade in the window (L0-valid). persist\\_recursive: $\\widehat{\\mathrm{RUL}}_t=\\max(\\widehat{\\mathrm{RUL}}_{t-1}-1,0)$ initialized by persist\\_fade (no labels). ")
    lines.append("persist\\_hold uses true $\\mathrm{RUL}_{t-1}$ and is an oracle walk, not an L0 BMS baseline. ")
    lines.append("linear: least-squares SOH fit. Fade RMSE $>500$ is marked unstable.}\n")
    lines.append("\\label{tab:persist}\n")
    lines.append("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n")
    lines.append("Corpus & Cell & fade & recursive & hold (oracle) & linear \\\\\n\\midrule\n")
    by = {}
    for r in classical:
        by.setdefault((r["corpus"], r["battery"]), {})[r["model"]] = float(r["rul_rmse"])
    for key in sorted(by):
        d = by[key]
        fade = d.get("persist_fade", float("nan"))
        fade_s = f"{fade:.2f}" if fade < 500 else "unstable"
        recu = d.get("persist_recursive", float("nan"))
        recu_s = f"{recu:.2f}" if recu == recu else "---"
        hold = d.get("persist_hold", float("nan"))
        lin = d.get("linear", float("nan"))
        lines.append(
            f"{key[0]} & {tex_cell_id(key[1])} & {fade_s} & {recu_s} & "
            f"{hold:.2f} & {lin:.2f} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    lines.append("\\begin{table}[htbp]\n\\centering\n")
    lines.append("\\caption{Seed-level paired contrasts of RUL RMSE (selected). ")
    lines.append("The experimental unit is the random seed on a fixed cell ($n=3$), not the cell. ")
    lines.append("Wilcoxon $p{=}0.25$ is the smallest two-sided value when all three seeds move in the same direction. ")
    lines.append("Full contrasts are in \\texttt{outputs/info\\_levels\\_v2/stats.json}.}\n")
    lines.append("\\label{tab:stats}\n")
    lines.append("\\begin{tabular}{@{}llllccc@{}}\n\\toprule\n")
    lines.append("Contrast & Corpus & Model & Cell & $\\Delta$RMSE & Cohen's $d$ & $p$ \\\\\n\\midrule\n")
    keep = []
    for s in stats:
        if s["contrast"] not in ("L2-vs-L0", "L2-vs-L1"):
            continue
        if s["corpus"] == "nasa":
            keep.append(s)
        elif s["corpus"] == "calce" and s["model"] == "mlp":
            keep.append(s)
    for s in keep:
        p = s.get("wilcoxon_p")
        ps = f"{p:.3f}" if p is not None else "NA"
        d = s.get("cohens_d")
        ds = f"{d:.2f}" if d is not None and d == d else "NA"
        lines.append(
            f"{s['contrast']} & {s['corpus']} & {s['model']} & {tex_cell_id(s['battery'])} & "
            f"{s['delta_b_minus_a']:.2f} & {ds} & {ps} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    uq_lines = []
    uq_lines.append("% auto-generated UQ appendix; do not edit by hand\n")
    uq_lines.append("\\begin{table}[htbp]\n\\centering\n")
    uq_lines.append("\\caption{Appendix. MC-Dropout 95\\% intervals ($n_{\\mathrm{MC}}=20$) for Patch-TF RUL on NASA. ")
    uq_lines.append("PICP is empirical coverage of mean$\\pm 1.96$ std; PINAW is width normalized by the test RUL range. ")
    uq_lines.append("Not used as a calibration claim.}\n")
    uq_lines.append("\\label{tab:uq}\n")
    uq_lines.append("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
    uq_lines.append("Cell & L2 PICP & L2 PINAW & L0 PICP & L0 PINAW \\\\\n\\midrule\n")
    for bat in ("B0005", "B0006", "B0007"):
        a2 = find_agg(agg, "nasa", "patch-tf", bat, "L2")
        a0 = find_agg(agg, "nasa", "patch-tf", bat, "L0")
        uq_lines.append(
            f"{bat} & {a2.get('picp_rul', float('nan')):.3f}$\\pm${a2.get('picp_rul_std', 0):.3f} & "
            f"{a2.get('pinaw_rul', float('nan')):.3f}$\\pm${a2.get('pinaw_rul_std', 0):.3f} & "
            f"{a0.get('picp_rul', float('nan')):.3f}$\\pm${a0.get('picp_rul_std', 0):.3f} & "
            f"{a0.get('pinaw_rul', float('nan')):.3f}$\\pm${a0.get('pinaw_rul_std', 0):.3f} \\\\\n"
        )
    uq_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    uq_out = ROOT / "paper" / "info_consistent_manuscript" / "generated_uq.tex"
    uq_out.write_text("".join(uq_lines), encoding="utf-8")

    diag_path = SRC / "channel_diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        lines.append("\\begin{table}[htbp]\n\\centering\n")
        lines.append("\\caption{Linear readout of the cycle channel on chronological test windows (RUL$>0$). ")
        lines.append("$\\rho$ is Pearson correlation. OLS uses an intercept plus the last-timestep info channel. ")
        lines.append("Under first-EOL labels $\\mathrm{RUL}=\\max(0,T-t)$, $\\rho(t,\\mathrm{RUL})=-1$ and OLS on raw~$t$ ")
        lines.append("is numerically zero on NASA and CALCE. ")
        lines.append("L1 and L2 clocks are affine in the same $t$, so $\\rho(\\mathrm{L1},\\mathrm{L2})=1$. ")
        lines.append("L1P is the SOH-matched sibling RUL prior and is not affine in~$t$.}\n")
        lines.append("\\label{tab:clock}\n")
        lines.append("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n")
        lines.append("Corpus & Cell & $\\rho(\\mathrm{L1},\\mathrm{L2})$ & $\\rho(\\mathrm{L1P},\\mathrm{L2})$ & OLS $t$ & OLS L1P \\\\\n\\midrule\n")
        for r in diag:
            def f4(x):
                x = float(x)
                if abs(x) < 1e-8:
                    return "0"
                if abs(x) > 10:
                    return f"{x:.2f}"
                return f"{x:.3f}"
            rp = r.get("pearson_L1P_vs_L2_test")
            rps = f"{float(rp):.3f}" if rp is not None else "---"
            olp = r.get("ols_rmse_L1P")
            lines.append(
                f"{r['corpus']} & {tex_cell_id(r['cell'])} & {float(r['pearson_L1_vs_L2_test']):.3f} & "
                f"{rps} & {f4(r['ols_rmse_L0_clock'])} & "
                f"{f4(olp) if olp is not None else '---'} \\\\\n"
            )
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    cell_path = SRC / "cell_stats.json"
    if cell_path.exists():
        cs = json.loads(cell_path.read_text(encoding="utf-8"))
        lines.append("\\begin{table}[htbp]\n\\centering\n")
        lines.append("\\caption{Cell-level MLP RUL RMSE (mean over three seeds). The experimental unit is the cell ($n{=}" + str(cs.get("n_cells", 7)) + "$). ")
        p0 = cs.get("wilcoxon_L0_vs_L2_p")
        p1 = cs.get("wilcoxon_L1_vs_L2_p")
        lines.append(f"Paired Wilcoxon on $\\mathrm{{RMSE}}_{{L0}}-\\mathrm{{RMSE}}_{{L2}}$: $p={p0:.3f}$; ")
        lines.append(f"on $\\mathrm{{RMSE}}_{{L1}}-\\mathrm{{RMSE}}_{{L2}}$: $p={p1:.3f}$. ")
        lines.append("Signs mix across NASA (L0 easier) and CALCE (L0 harder); seed tests are not a substitute.}\n")
        lines.append("\\label{tab:cellstat}\n")
        lines.append("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n")
        lines.append("Corpus & Cell & L2 & L1 & L0 \\\\\n\\midrule\n")
        for r in cs.get("cells", []):
            lines.append(
                f"{r['corpus']} & {tex_cell_id(r['cell'])} & {r['L2']:.2f} & {r['L1']:.2f} & {r['L0']:.2f} \\\\\n"
            )
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
