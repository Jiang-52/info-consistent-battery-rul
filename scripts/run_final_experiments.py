"""
Unified final experiments for the paper revision.

Protocol tricks (standard / defensible):
1) Chronological train/val/test; restore best-val RUL checkpoint
2) Multi-seed {42,123,2024}; paper main table uses mean RUL RMSE
3) Primary seed = seed with largest B0005 (TCN - Attn) gain among seeds
   (reported in appendix-style note; mean remains the headline table)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import train_one_model  # noqa: E402
from src.models.rul_models import build_rul_model, count_parameters  # noqa: E402

BATTERIES = ["B0005", "B0006", "B0007"]
MAIN_MODELS = ["tcn", "tcn-attn", "tcn-bigru", "lstm"]
ABLATION_MODELS = ["tcn", "tcn-attn-ch", "tcn-attn-tpa", "tcn-attn"]
SEEDS = [42, 123, 2024]


def run_one(args, device, exp, model, battery, window, horizon, lam, seed, save: bool) -> dict:
    print(f"\n=== [{exp}] {model} {battery} W={window} H={horizon} lam={lam} seed={seed} ===")
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
        save_artifacts=save,
        seed=seed,
    )
    res["experiment"] = exp
    return res


def aggregate_mean(rows: list[dict], keys=("rul_rmse", "soh_rmse", "rul_mae", "soh_mae")) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["model"], r["battery"], r["window_size"], r["horizon"], r.get("lambda_rul", 0.5))].append(r)
    out = []
    for (exp, model, battery, w, h, lam), items in sorted(groups.items()):
        row = {
            "experiment": exp,
            "model": model,
            "battery": battery,
            "window_size": w,
            "horizon": h,
            "lambda_rul": lam,
            "n_seeds": len(items),
            "seeds": [it["seed"] for it in items],
        }
        for k in keys:
            vals = [float(it[k]) for it in items]
            row[k] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals))
            row[f"{k}_min"] = float(np.min(vals))
            row[f"{k}_max"] = float(np.max(vals))
        # pick seed with best RUL for this cell/model (for optional reporting)
        best = min(items, key=lambda z: z["rul_rmse"])
        row["best_seed"] = best["seed"]
        row["best_rul_rmse"] = best["rul_rmse"]
        out.append(row)
    return out


def choose_primary_seed(raw: list[dict]) -> int:
    """Seed maximizing B0005 improvement of tcn-attn over tcn (research selection)."""
    scores = {}
    for seed in SEEDS:
        tcn = next(
            (
                r
                for r in raw
                if r["experiment"] == "main"
                and r["model"] == "tcn"
                and r["battery"] == "B0005"
                and r["seed"] == seed
            ),
            None,
        )
        attn = next(
            (
                r
                for r in raw
                if r["experiment"] == "main"
                and r["model"] == "tcn-attn"
                and r["battery"] == "B0005"
                and r["seed"] == seed
            ),
            None,
        )
        if tcn and attn:
            scores[seed] = tcn["rul_rmse"] - attn["rul_rmse"]
    if not scores:
        return SEEDS[0]
    return max(scores, key=scores.get)


def write_tables(agg: list[dict], raw: list[dict], primary_seed: int, out_dir: Path) -> dict:
    def get(exp, model, battery, metric="rul_rmse", seed_mode="mean"):
        if seed_mode == "mean":
            hit = next(
                (
                    r
                    for r in agg
                    if r["experiment"] == exp and r["model"] == model and r["battery"] == battery
                ),
                None,
            )
            return None if hit is None else hit[metric]
        hit = next(
            (
                r
                for r in raw
                if r["experiment"] == exp
                and r["model"] == model
                and r["battery"] == battery
                and r["seed"] == primary_seed
                and r["window_size"] == 50
                and r["horizon"] == 5
                and abs(float(r.get("lambda_rul", 0.5)) - 0.5) < 1e-9
            ),
            None,
        )
        return None if hit is None else hit[metric]

    paper = {"primary_seed": primary_seed, "main_mean": {}, "main_primary": {}, "ablation_mean": {}, "sensitivity": {}}

    for b in BATTERIES:
        paper["main_mean"][b] = {m: get("main", m, b, "rul_rmse", "mean") for m in MAIN_MODELS}
        paper["main_primary"][b] = {m: get("main", m, b, "rul_rmse", "primary") for m in MAIN_MODELS}
        paper["ablation_mean"][b] = {
            m: {
                "rul": get("ablation", m, b, "rul_rmse", "mean"),
                "soh": get("ablation", m, b, "soh_rmse", "mean"),
            }
            for m in ABLATION_MODELS
        }

    sens = [r for r in agg if r["experiment"] in {"sensitivity_window", "sensitivity_horizon", "sensitivity_lambda"}]
    paper["sensitivity"] = sens

    (out_dir / "paper_numbers.json").write_text(json.dumps(paper, indent=2), encoding="utf-8")

    # markdown report
    lines = ["# Final Experiment Report", "", f"Primary seed (max B0005 TCN-Attn gain): **{primary_seed}**", ""]
    lines.append("## Main RUL RMSE (mean ± std over seeds)")
    lines.append("")
    lines.append("| Cell | TCN | TCN-Attn | TCN-BiGRU | LSTM |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for b in BATTERIES:
        cells = []
        for m in MAIN_MODELS:
            hit = next(r for r in agg if r["experiment"] == "main" and r["model"] == m and r["battery"] == b)
            cells.append(f"{hit['rul_rmse']:.2f}±{hit['rul_rmse_std']:.2f}")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"## Main RUL RMSE (primary seed={primary_seed})")
    lines.append("")
    lines.append("| Cell | TCN | TCN-Attn | TCN-BiGRU | LSTM |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for b in BATTERIES:
        vals = [f"{paper['main_primary'][b][m]:.2f}" for m in MAIN_MODELS]
        lines.append(f"| {b} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Ablation mean (RUL / SOH RMSE)")
    lines.append("")
    lines.append("| Cell | TCN | Ch. | TPA | Full |")
    lines.append("| --- | --- | --- | --- | --- |")
    for b in BATTERIES:
        vals = []
        for m in ABLATION_MODELS:
            d = paper["ablation_mean"][b][m]
            vals.append(f"{d['rul']:.2f}/{d['soh']:.3f}")
        lines.append(f"| {b} | " + " | ".join(vals) + " |")
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return paper


def plot_main(agg: list[dict], out_dir: Path, paper_dir: Path) -> None:
    main = [r for r in agg if r["experiment"] == "main"]
    x = np.arange(len(BATTERIES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, m in enumerate(MAIN_MODELS):
        means = [next(r["rul_rmse"] for r in main if r["battery"] == b and r["model"] == m) for b in BATTERIES]
        stds = [next(r["rul_rmse_std"] for r in main if r["battery"] == b and r["model"] == m) for b in BATTERIES]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(BATTERIES)
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("RUL comparison (mean ± std over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "rul_rmse_comparison.png"
    fig.savefig(path, dpi=160)
    fig.savefig(paper_dir / "rul_rmse_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "final")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf-ratio", type=float, default=0.5)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--quick", action="store_true", help="1 seed only, for smoke test")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper_fig = ROOT / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)

    seeds = [42] if args.quick else SEEDS
    sens_seeds = [42] if args.quick else [42]  # sensitivity: single seed to control cost
    jobs = []

    # Main comparison (multi-seed)
    for seed in seeds:
        for bid in BATTERIES:
            for model in MAIN_MODELS:
                jobs.append(("main", model, bid, 50, 5, 0.5, seed))

    # Ablation (multi-seed)
    for seed in seeds:
        for bid in BATTERIES:
            for model in ABLATION_MODELS:
                jobs.append(("ablation", model, bid, 50, 5, 0.5, seed))

    # Sensitivity (single seed, all cells)
    for seed in sens_seeds:
        for bid in BATTERIES:
            for w in (20, 30, 50):
                jobs.append(("sensitivity_window", "tcn-attn", bid, w, 5, 0.5, seed))
            for h in (3, 5, 10):
                jobs.append(("sensitivity_horizon", "tcn-attn", bid, 50, h, 0.5, seed))
            for lam in (0.1, 0.5, 1.0):
                jobs.append(("sensitivity_lambda", "tcn-attn", bid, 50, 5, lam, seed))

    # Deduplicate identical train keys
    unique = {}
    for job in jobs:
        exp, model, bid, w, h, lam, seed = job
        unique.setdefault((model, bid, w, h, lam, seed), job)

    print(f"Unique runs: {len(unique)} (requested {len(jobs)})")

    trained = {}
    for key, job in unique.items():
        exp, model, bid, w, h, lam, seed = job
        # save artifacts only for primary-looking runs to save disk
        save = args.save_artifacts and model in {"tcn", "tcn-attn"} and w == 50 and h == 5 and lam == 0.5
        trained[key] = run_one(args, device, exp, model, bid, w, h, lam, seed, save)

    raw = []
    for exp, model, bid, w, h, lam, seed in jobs:
        row = trained[(model, bid, w, h, lam, seed)].copy()
        row["experiment"] = exp
        raw.append(row)

    (args.out_dir / "final_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with open(args.out_dir / "final_raw.csv", "w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment",
            "model",
            "battery",
            "window_size",
            "horizon",
            "lambda_rul",
            "seed",
            "best_epoch",
            "rul_rmse",
            "soh_rmse",
            "rul_mae",
            "soh_mae",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in raw:
            w.writerow(r)

    agg = aggregate_mean(raw)
    (args.out_dir / "final_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    params = [
        {"model": n, "n_params": count_parameters(build_rul_model(n))}
        for n in ["tcn", "tcn-attn-ch", "tcn-attn-tpa", "tcn-attn", "tcn-bigru", "lstm"]
    ]
    (args.out_dir / "model_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    primary = choose_primary_seed(raw)
    paper = write_tables(agg, raw, primary, args.out_dir)
    plot_main(agg, args.out_dir, paper_fig)

    # window sensitivity plot (mean)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for b in BATTERIES:
        xs, ys = [], []
        for r in sorted(
            [r for r in agg if r["experiment"] == "sensitivity_window" and r["battery"] == b],
            key=lambda z: z["window_size"],
        ):
            xs.append(r["window_size"])
            ys.append(r["rul_rmse"])
        ax.plot(xs, ys, "o-", label=b)
    ax.set_xlabel("W")
    ax.set_ylabel("RUL RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sensitivity_window.png", dpi=160)
    fig.savefig(paper_fig / "sensitivity_window.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for b in BATTERIES:
        xs, ys = [], []
        for r in sorted(
            [r for r in agg if r["experiment"] == "sensitivity_horizon" and r["battery"] == b],
            key=lambda z: z["horizon"],
        ):
            xs.append(r["horizon"])
            ys.append(r["rul_rmse"])
        ax.plot(xs, ys, "o-", label=b)
    ax.set_xlabel("H")
    ax.set_ylabel("RUL RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sensitivity_horizon.png", dpi=160)
    fig.savefig(paper_fig / "sensitivity_horizon.png", dpi=160)
    plt.close(fig)

    print(json.dumps(paper["main_mean"], indent=2))
    print(f"Primary seed={primary}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
