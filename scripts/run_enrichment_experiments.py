"""Enrichment experiments for paper revision: ablation, sensitivity, model size."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import set_seed, train_one_model  # noqa: E402
from src.models.rul_models import build_rul_model, count_parameters  # noqa: E402


ABLATION_MODELS = ["tcn", "tcn-attn-ch", "tcn-attn-tpa", "tcn-attn"]
BATTERIES = ["B0005", "B0006", "B0007"]


def _key(r: dict) -> tuple:
    return (
        r["experiment"],
        r["model"],
        r["battery"],
        r["window_size"],
        r["horizon"],
        round(float(r.get("lambda_rul", 0.5)), 4),
    )


def run_job(args, device, experiment: str, model: str, battery: str, window: int, horizon: int, lam: float) -> dict:
    print(
        f"\n=== [{experiment}] {model} {battery} W={window} H={horizon} lambda={lam} ==="
    )
    res = train_one_model(
        model_name=model,
        battery_id=battery,
        data_dir=args.data_dir,
        out_dir=args.out_dir / "checkpoints",
        window_size=window,
        horizon=horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tf_ratio=args.tf_ratio,
        device=device,
        lambda_rul=lam,
        save_artifacts=args.save_artifacts,
    )
    res["experiment"] = experiment
    return res


def write_param_table(out_dir: Path, horizon: int = 5) -> list[dict]:
    rows = []
    for name in ["tcn", "tcn-attn-ch", "tcn-attn-tpa", "tcn-attn", "tcn-bigru", "lstm"]:
        model = build_rul_model(name, horizon=horizon)
        rows.append({"model": name, "horizon": horizon, "n_params": count_parameters(model)})
    path = out_dir / "model_params.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "horizon", "n_params"])
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_ablation(rows: list[dict], out_dir: Path) -> None:
    abl = [r for r in rows if r["experiment"] == "ablation"]
    if not abl:
        return
    batteries = sorted({r["battery"] for r in abl})
    models = ABLATION_MODELS
    x = np.arange(len(batteries))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(models):
        vals = []
        for b in batteries:
            hit = [r["rul_rmse"] for r in abl if r["battery"] == b and r["model"] == m]
            vals.append(hit[0] if hit else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(batteries)
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("Attention-component ablation")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_components.png", dpi=160)
    plt.close(fig)


def plot_sensitivity(rows: list[dict], out_dir: Path) -> None:
    win_rows = [r for r in rows if r["experiment"] == "sensitivity_window"]
    if win_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in BATTERIES:
            xs, ys = [], []
            for r in sorted(win_rows, key=lambda z: z["window_size"]):
                if r["battery"] == b:
                    xs.append(r["window_size"])
                    ys.append(r["rul_rmse"])
            if xs:
                ax.plot(xs, ys, "o-", label=b)
        ax.set_xlabel("Window size W")
        ax.set_ylabel("RUL RMSE")
        ax.set_title("Window-size sensitivity (TCN-Attention, H=5)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity_window.png", dpi=160)
        plt.close(fig)

    hor_rows = [r for r in rows if r["experiment"] == "sensitivity_horizon"]
    if hor_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in BATTERIES:
            xs, ys = [], []
            for r in sorted(hor_rows, key=lambda z: z["horizon"]):
                if r["battery"] == b:
                    xs.append(r["horizon"])
                    ys.append(r["rul_rmse"])
            if xs:
                ax.plot(xs, ys, "o-", label=b)
        ax.set_xlabel("Horizon H")
        ax.set_ylabel("RUL RMSE")
        ax.set_title("Horizon sensitivity (TCN-Attention, W=50)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity_horizon.png", dpi=160)
        plt.close(fig)


def write_report(rows: list[dict], params: list[dict], out_dir: Path) -> None:
    lines = ["# Enrichment Experiment Report", ""]
    lines.append("## Model size")
    lines.append("")
    lines.append("| model | n_params |")
    lines.append("| --- | ---: |")
    for p in params:
        lines.append(f"| {p['model']} | {p['n_params']} |")
    lines.append("")

    for exp in ["ablation", "sensitivity_window", "sensitivity_horizon", "sensitivity_lambda"]:
        subset = [r for r in rows if r["experiment"] == exp]
        if not subset:
            continue
        lines.append(f"## {exp}")
        lines.append("")
        lines.append(
            "| model | battery | W | H | lambda | rul_rmse | soh_rmse | n_params |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in subset:
            lines.append(
                f"| {r['model']} | {r['battery']} | {r['window_size']} | {r['horizon']} | "
                f"{r.get('lambda_rul', 0.5)} | {r['rul_rmse']:.4f} | {r['soh_rmse']:.4f} | "
                f"{r.get('n_params', '')} |"
            )
        lines.append("")

    (out_dir / "enrichment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper enrichment experiments")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "enrichment")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument(
        "--skip-lambda",
        action="store_true",
        help="Skip lambda sensitivity (still runs ablation + W/H)",
    )
    args = parser.parse_args()

    import torch

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    params = write_param_table(args.out_dir)
    print("Param counts:", params)

    jobs: list[tuple] = []
    # Component ablation (W=50, H=5, lambda=0.5)
    for bid in BATTERIES:
        for model in ABLATION_MODELS:
            jobs.append(("ablation", model, bid, 50, 5, 0.5))

    # Window sensitivity
    for bid in BATTERIES:
        for w in (20, 30, 50):
            jobs.append(("sensitivity_window", "tcn-attn", bid, w, 5, 0.5))

    # Horizon sensitivity
    for bid in BATTERIES:
        for h in (3, 5, 10):
            jobs.append(("sensitivity_horizon", "tcn-attn", bid, 50, h, 0.5))

    # Lambda sensitivity on all cells (default setting)
    if not args.skip_lambda:
        for bid in BATTERIES:
            for lam in (0.1, 0.5, 1.0):
                jobs.append(("sensitivity_lambda", "tcn-attn", bid, 50, 5, lam))

    # Deduplicate identical train settings; keep first experiment label for reporting
    # but still record under each experiment that needs the metric.
    # Strategy: train unique (model,battery,W,H,lambda) once, then fan-out copies.
    unique_train: dict[tuple, tuple] = {}
    for exp, model, bid, w, h, lam in jobs:
        tk = (model, bid, w, h, lam)
        unique_train.setdefault(tk, (exp, model, bid, w, h, lam))

    print(f"Unique training runs: {len(unique_train)} (from {len(jobs)} requested)")

    trained: dict[tuple, dict] = {}
    for tk, job in unique_train.items():
        exp, model, bid, w, h, lam = job
        trained[tk] = run_job(args, device, exp, model, bid, w, h, lam)

    rows: list[dict] = []
    for exp, model, bid, w, h, lam in jobs:
        base = trained[(model, bid, w, h, lam)].copy()
        base["experiment"] = exp
        rows.append(base)

    metrics_path = args.out_dir / "enrichment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Flat CSV
    csv_path = args.out_dir / "enrichment_metrics.csv"
    fields = [
        "experiment",
        "model",
        "battery",
        "window_size",
        "horizon",
        "lambda_rul",
        "epochs",
        "n_params",
        "rul_rmse",
        "rul_mae",
        "soh_rmse",
        "soh_mae",
        "iter_rmse",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    plot_ablation(rows, args.out_dir)
    plot_sensitivity(rows, args.out_dir)
    write_report(rows, params, args.out_dir)
    print(f"\nSaved enrichment outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
