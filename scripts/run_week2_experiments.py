"""Week-2 experiment runner: RUL comparison + anomaly detection + report tables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.plotting import save_comparison_bar


def run_cmd(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_rul_table(rul_dir: Path) -> pd.DataFrame:
    path = rul_dir / "rul_metrics.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def build_anomaly_table(anomaly_dir: Path) -> pd.DataFrame:
    path = anomaly_dir / "anomaly_metrics.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def ablation_from_rul(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for bid in df["battery"].unique():
        sub = df[df["battery"] == bid]
        base = sub[sub["model"] == "tcn"]
        attn = sub[sub["model"] == "tcn-attn"]
        if len(base) and len(attn):
            rows.append(
                {
                    "battery": bid,
                    "tcn_rul_rmse": base["rul_rmse"].values[0],
                    "tcn_attn_rul_rmse": attn["rul_rmse"].values[0],
                    "improvement_pct": (
                        (base["rul_rmse"].values[0] - attn["rul_rmse"].values[0])
                        / base["rul_rmse"].values[0]
                        * 100
                    ),
                    "tcn_soh_rmse": base["soh_rmse"].values[0],
                    "tcn_attn_soh_rmse": attn["soh_rmse"].values[0],
                }
            )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week-2 experiments")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs-rul", type=int, default=250)
    parser.add_argument("--epochs-ae", type=int, default=150)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "week2")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_train:
        run_cmd(
            [
                py,
                "scripts/train_rul.py",
                "--all",
                "--models",
                "tcn",
                "tcn-attn",
                "tcn-bigru",
                "lstm",
                "--epochs",
                str(args.epochs_rul),
                "--out-dir",
                str(ROOT / "outputs" / "rul"),
            ]
        )
        run_cmd(
            [
                py,
                "scripts/train_anomaly.py",
                "--all",
                "--epochs",
                str(args.epochs_ae),
                "--out-dir",
                str(ROOT / "outputs" / "anomaly"),
            ]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rul_df = build_rul_table(ROOT / "outputs" / "rul")
    anomaly_df = build_anomaly_table(ROOT / "outputs" / "anomaly")
    ablation_df = ablation_from_rul(rul_df)

    if not rul_df.empty:
        rul_df.to_csv(args.out_dir / "rul_comparison.csv", index=False)
        save_comparison_bar(
            rul_df,
            "rul_rmse",
            args.out_dir / "rul_rmse_comparison.png",
            "RUL RMSE Comparison (lower is better)",
        )
        save_comparison_bar(
            rul_df,
            "soh_rmse",
            args.out_dir / "soh_rmse_comparison.png",
            "Multi-step SOH RMSE Comparison",
        )

    if not ablation_df.empty:
        ablation_df.to_csv(args.out_dir / "ablation_tcn_vs_attn.csv", index=False)

    if not anomaly_df.empty:
        anomaly_df.to_csv(args.out_dir / "anomaly_results.csv", index=False)

    md_lines = [
        "# Week 2 Experiment Report",
        "",
        "## RUL Prediction (TCN vs TCN-Attention vs LSTM)",
        "",
    ]
    if not rul_df.empty:
        md_lines.append(df_to_markdown(rul_df.round(4)))
    else:
        md_lines.append("_No RUL results found._")

    md_lines.extend(["", "## Ablation: TCN vs TCN-Attention", ""])
    if not ablation_df.empty:
        md_lines.append(df_to_markdown(ablation_df.round(4)))
    else:
        md_lines.append("_No ablation results._")

    md_lines.extend(["", "## Anomaly Detection (TCN-Autoencoder)", ""])
    if not anomaly_df.empty:
        md_lines.append(df_to_markdown(anomaly_df.round(4)))
    else:
        md_lines.append("_No anomaly results._")

    report = "\n".join(md_lines) + "\n"
    (args.out_dir / "week2_report.md").write_text(report, encoding="utf-8")
    print(f"\nReport saved to {args.out_dir / 'week2_report.md'}")


if __name__ == "__main__":
    main()
