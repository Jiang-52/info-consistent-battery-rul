"""
EI v2 experiment runner: Patch-TF main stack vs TCN/LSTM baselines.

Writes to outputs/ei_v2/ without overwriting outputs/final/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import train_one_model
from src.data.nasa_loader import chronological_split
from src.data.rul_dataset import load_battery_rul_data
from src.models.rul_models import build_rul_model, count_parameters
from src.models.uncertainty import interval_coverage, mc_dropout_predict

MAIN_BATTERIES = ["B0005", "B0007"]
SEEDS = [42, 43, 44]


def _clean(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in {"model_obj", "x_test", "y_rul_test"}}


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("experiment", "main"),
            r["model"],
            r["battery"],
            r.get("feature_mode", "soh"),
            r.get("use_physics", False),
            r.get("window_size"),
            r.get("horizon"),
            r.get("num_layers", 2),
            r.get("patch_size", 5),
            r.get("d_model", 64),
        )
        groups[key].append(r)
    out = []
    for key, items in sorted(groups.items()):
        exp, model, battery, feat, phys, w, h, layers, ps, dm = key
        row = {
            "experiment": exp,
            "model": model,
            "battery": battery,
            "feature_mode": feat,
            "use_physics": phys,
            "window_size": w,
            "horizon": h,
            "num_layers": layers,
            "patch_size": ps,
            "d_model": dm,
            "n_seeds": len(items),
            "n_params": items[0].get("n_params"),
        }
        for metric in ("rul_rmse", "rul_mae", "soh_rmse", "soh_mae", "soh_r2"):
            vals = [float(it[metric]) for it in items]
            row[metric] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        out.append(row)
    return out


def run_uq_eval(
    model_name: str,
    battery_id: str,
    data_dir: Path,
    ckpt_path: Path,
    window_size: int,
    horizon: int,
    feature_mode: str,
    device: torch.device,
    n_samples: int = 30,
    patch_size: int = 5,
    d_model: int = 64,
    num_layers: int = 2,
) -> dict:
    data = load_battery_rul_data(
        data_dir / f"{battery_id}.mat",
        battery_id,
        window_size,
        horizon,
        feature_mode=feature_mode,
    )
    x, y_soh, y_rul = data["x"], data["y_soh"], data["y_rul"]
    n_features = int(data["n_features"])
    _, x_test, _, y_soh_test = chronological_split(x, y_soh, 0.8)
    y_rul_test = y_rul[len(x) - len(x_test) :]
    # align with train_one_model split: first 80% then last 20% as test
    n_train = len(x) - len(x_test)
    y_rul_test = y_rul[n_train:]

    model = build_rul_model(
        model_name,
        horizon=horizon,
        n_features=n_features,
        d_model=d_model,
        num_layers=num_layers,
        patch_size=patch_size,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    xb = torch.tensor(x_test, dtype=torch.float32, device=device)
    mc = mc_dropout_predict(model, xb, n_samples=n_samples, device=device)
    cov_rul = interval_coverage(y_rul_test, mc["rul_mean"], mc["rul_std"])
    cov_soh = interval_coverage(y_soh_test, mc["soh_mean"], mc["soh_std"])
    return {
        "model": model_name,
        "battery": battery_id,
        "feature_mode": feature_mode,
        "n_mc": n_samples,
        "rul_coverage_95": cov_rul,
        "soh_coverage_95": cov_soh,
        "rul_mean_std": float(np.mean(mc["rul_std"])),
        "soh_mean_std": float(np.mean(mc["soh_std"])),
    }


def run_transfer_soh(
    nasa_dir: Path,
    calce_path: Path,
    window: int,
    epochs: int,
    device: torch.device,
) -> list[dict]:
    """Reuse existing cross-dataset helper for NASA→CALCE SOH boundary."""
    from scripts.train_cross_dataset import run_cross_transfer

    return run_cross_transfer(
        nasa_dir, ["B0005", "B0006", "B0007"], calce_path, window, epochs, device
    )


def write_report(out_dir: Path, agg: list[dict], raw: list[dict], uq: list[dict], transfer: list[dict]) -> None:
    lines = [
        "# EI v2 Experiment Report",
        "",
        "Proposed method: **Patch-TF** (multi-feat + optional physics). Baselines: TCN / TCN-Attn / LSTM.",
        "",
        "## Main RUL RMSE (mean ± std)",
        "",
        "| Cell | Feature | Patch-TF | TCN | TCN-Attn | LSTM |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for bat in MAIN_BATTERIES:
        for feat in ("multifeat", "soh"):
            cells = {}
            for model in ("patch-tf", "tcn", "tcn-attn", "lstm"):
                row = next(
                    (
                        r
                        for r in agg
                        if r["experiment"] == "main"
                        and r["battery"] == bat
                        and r["model"] == model
                        and r["feature_mode"] == feat
                    ),
                    None,
                )
                cells[model] = (
                    f"{row['rul_rmse']:.2f}±{row['rul_rmse_std']:.2f}" if row else "-"
                )
            if any(v != "-" for v in cells.values()):
                lines.append(
                    f"| {bat} | {feat} | {cells['patch-tf']} | {cells['tcn']} | {cells['tcn-attn']} | {cells['lstm']} |"
                )
    lines.extend(["", "## UQ (MC-Dropout ~95% coverage)", ""])
    for u in uq:
        lines.append(
            f"- {u['model']} {u['battery']} feat={u['feature_mode']}: "
            f"RUL cov={u['rul_coverage_95']:.3f}, SOH cov={u['soh_coverage_95']:.3f}"
        )
    lines.extend(["", "## NASA→CALCE transfer (boundary)", ""])
    for t in transfer[:8]:
        rmse = t.get("full_rmse", t.get("test_rmse", t.get("rmse", float("nan"))))
        r2 = t.get("full_r2", t.get("test_r2", t.get("r2", float("nan"))))
        lines.append(
            f"- {t.get('cell', t.get('battery', '?'))}: "
            f"RMSE={rmse:.4f} R2={r2:.4f}"
        )
    lines.extend(["", f"Raw runs: {len(raw)}", ""])
    (out_dir / "ei_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--calce-path", type=Path, default=ROOT / "data" / "calce" / "CALCE.npy")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "ei_v2")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf-ratio", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true", help="Short run for debugging")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-uq", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 3
        seeds = [42]
        batteries = ["B0005"]
    else:
        seeds = SEEDS
        batteries = MAIN_BATTERIES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw: list[dict] = []
    t0 = time.time()

    # E1 main comparison
    main_specs = [
        ("patch-tf", "multifeat", True),
        ("tcn", "soh", False),
        ("tcn-attn", "soh", False),
        ("lstm", "soh", False),
        # E2 input ablation for proposed model
        ("patch-tf", "soh", True),
        ("patch-tf", "multifeat", False),  # physics off
    ]
    for seed in seeds:
        for bat in batteries:
            for model, feat, phys in main_specs:
                exp = "main" if not (model == "patch-tf" and (feat == "soh" or not phys)) else (
                    "ablation_input" if feat == "soh" else "ablation_physics"
                )
                if model == "patch-tf" and feat == "multifeat" and phys:
                    exp = "main"
                print(f"\n=== [{exp}] {model} {bat} feat={feat} phys={phys} seed={seed} ===")
                res = train_one_model(
                    model_name=model,
                    battery_id=bat,
                    data_dir=args.data_dir,
                    out_dir=ckpt_dir,
                    window_size=args.window_size,
                    horizon=args.horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    tf_ratio=args.tf_ratio if model != "patch-tf" else 0.0,
                    device=device,
                    seed=seed,
                    feature_mode=feat,
                    use_physics=phys,
                    save_artifacts=True,
                )
                res["experiment"] = exp
                raw.append(_clean(res))

    # E3 structure ablation (patch-tf only, primary seed)
    if not args.smoke:
        for layers, ps, dm in [(1, 5, 64), (2, 10, 64), (2, 5, 32)]:
            print(f"\n=== [ablation_struct] layers={layers} patch={ps} d={dm} ===")
            res = train_one_model(
                model_name="patch-tf",
                battery_id="B0005",
                data_dir=args.data_dir,
                out_dir=ckpt_dir,
                window_size=args.window_size,
                horizon=args.horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                tf_ratio=0.0,
                device=device,
                seed=42,
                feature_mode="multifeat",
                use_physics=True,
                num_layers=layers,
                patch_size=ps,
                d_model=dm,
                save_artifacts=True,
            )
            res["experiment"] = "ablation_struct"
            res["num_layers"] = layers
            res["patch_size"] = ps
            res["d_model"] = dm
            raw.append(_clean(res))

    # E4 UQ on best patch-tf checkpoint (seed 42, B0005 multifeat physics)
    uq_rows = []
    if not args.skip_uq:
        tag = f"patch-tf_b0005_w{args.window_size}_h{args.horizon}_s42"
        # train_one_model save name: rul_{model}_{battery}_w{W}_h{H}_s{seed}.pt
        ckpt = ckpt_dir / f"rul_{tag}.pt"
        # actual tag uses battery_id.lower()
        matches = list(ckpt_dir.glob("rul_patch-tf_b0005_*.pt"))
        if matches:
            ckpt = sorted(matches)[0]
            try:
                uq_rows.append(
                    run_uq_eval(
                        "patch-tf",
                        "B0005",
                        args.data_dir,
                        ckpt,
                        args.window_size,
                        args.horizon,
                        "multifeat",
                        device,
                        n_samples=10 if args.smoke else 30,
                    )
                )
            except Exception as exc:
                uq_rows.append({"error": str(exc), "model": "patch-tf", "battery": "B0005"})

    # E5 transfer
    transfer = []
    if not args.skip_transfer and args.calce_path.exists():
        transfer = run_transfer_soh(
            args.data_dir,
            args.calce_path,
            window=min(30, args.window_size),
            epochs=max(1, args.epochs // 10) if args.smoke else max(20, args.epochs // 5),
            device=device,
        )

    # E6 param counts
    param_table = []
    for model, feat in [("patch-tf", 6), ("tcn", 1), ("tcn-attn", 1), ("lstm", 1)]:
        m = build_rul_model(model, horizon=args.horizon, n_features=feat)
        param_table.append({"model": model, "n_features": feat, "n_params": count_parameters(m)})

    agg = aggregate(raw)
    (out_dir / "ei_v2_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_uq.json").write_text(json.dumps(uq_rows, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_transfer.json").write_text(json.dumps(transfer, indent=2), encoding="utf-8")
    (out_dir / "ei_v2_params.json").write_text(json.dumps(param_table, indent=2), encoding="utf-8")
    write_report(out_dir, agg, raw, uq_rows, transfer)

    # simple bar chart for B0005 main
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, vals, errs = [], [], []
    for model in ("patch-tf", "tcn", "tcn-attn", "lstm"):
        row = next(
            (
                r
                for r in agg
                if r["experiment"] == "main"
                and r["battery"] == "B0005"
                and r["model"] == model
                and r["feature_mode"] == ("multifeat" if model == "patch-tf" else "soh")
            ),
            None,
        )
        if row:
            labels.append(model)
            vals.append(row["rul_rmse"])
            errs.append(row["rul_rmse_std"])
    if labels:
        ax.bar(labels, vals, yerr=errs, capsize=4)
        ax.set_ylabel("RUL RMSE (cycles)")
        ax.set_title("B0005 main comparison")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_b0005_main_rul.png", dpi=150)
        plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Outputs → {out_dir}")


if __name__ == "__main__":
    main()
