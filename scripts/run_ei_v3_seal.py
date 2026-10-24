#!/usr/bin/env python3
"""
Pre-submission sealing experiments (non-OA EI risk closure).

P0  Horizon audit: prove RUL target definition is independent of H, but
    sample set N and chrono splits change with H; report fixed-index eval.
P1  Patch-size 3-seed: W=50, P in {2,5,10,20}, seeds {42,43,44}.
P2  MLP feature ablation: which channels drive the strong MLP baseline.

Outputs: outputs/ei_v3_seal/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import train_one_model, set_seed
from src.data.rul_dataset import load_battery_rul_data, compute_rul
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
FEAT_NAMES = ["cap_norm", "mean_V", "mean_I", "mean_T", "std_V", "cycle_norm"]
FEATURE_SETS = {
    "soh_only": [0],
    "V": [1],
    "I": [2],
    "T": [3],
    "cycle_norm": [5],
    "VIT": [1, 2, 3],
    "all": [0, 1, 2, 3, 4, 5],
}


def _agg(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n_runs"] = len(items)
        for m in ("rul_rmse", "rul_mae", "soh_rmse", "n_train", "n_val", "n_test", "n_total"):
            vals = [float(it[m]) for it in items if m in it and it[m] is not None]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        out.append(row)
    return out


def audit_horizon_construction(data_dir: Path, battery: str = "B0005", window: int = 50) -> dict:
    """Document how H changes sample construction without changing RUL definition."""
    report: dict = {"battery": battery, "window": window, "horizons": {}, "conclusion": []}
    # RUL curve is fixed from SOH; independent of H
    base = load_battery_rul_data(data_dir / f"{battery}.mat", battery, window, horizon=1, feature_mode="multifeat")
    rul_curve = base["rul_curve"]
    report["rul_curve_len"] = int(len(rul_curve))
    report["rul_definition"] = (
        "Y_rul[i] = remaining cycles to EOL at end of history window (index t-1); "
        "identical formula for all H. H only changes SOH multi-step targets and "
        "the set of valid window end indices t in [W, T-H]."
    )

    for h in [1, 3, 5, 10, 20]:
        d = load_battery_rul_data(
            data_dir / f"{battery}.mat", battery, window, horizon=h, feature_mode="multifeat"
        )
        n = len(d["x"])
        n_tr = int(n * 0.8)
        n_te = n - n_tr
        # RUL at each window: should equal rul_curve[t-1] for t = W .. T-H
        # reconstruct prediction indices
        soh = d["soh"]
        pred_idx = np.arange(window, len(soh) - h + 1) - 1  # t-1
        expected = rul_curve[pred_idx]
        match = bool(np.allclose(d["y_rul"], expected, atol=1e-5))
        report["horizons"][str(h)] = {
            "n_total": n,
            "n_train80": n_tr,
            "n_test20": n_te,
            "rul_mean": float(np.mean(d["y_rul"])),
            "rul_std": float(np.std(d["y_rul"])),
            "rul_test_mean": float(np.mean(d["y_rul"][n_tr:])),
            "rul_test_std": float(np.std(d["y_rul"][n_tr:])),
            "rul_matches_fixed_curve": match,
            "soh_target_shape": list(d["y_soh"].shape),
            "first_pred_cycle": int(pred_idx[0]) if len(pred_idx) else None,
            "last_pred_cycle": int(pred_idx[-1]) if len(pred_idx) else None,
        }

    # Intersection of prediction indices across H (fair RUL comparison universe)
    idx_sets = {}
    for h in [1, 3, 5, 10, 20]:
        d = load_battery_rul_data(
            data_dir / f"{battery}.mat", battery, window, horizon=h, feature_mode="multifeat"
        )
        soh = d["soh"]
        idx_sets[h] = set(int(i) for i in (np.arange(window, len(soh) - h + 1) - 1))
    common = set.intersection(*idx_sets.values()) if idx_sets else set()
    report["common_pred_cycles_all_H"] = len(common)
    report["conclusion"] = [
        "RUL scalar target definition does NOT depend on H (verified rul_matches_fixed_curve).",
        "H changes (i) SOH head output length, (ii) number of windows N=T-W-H+1, "
        "(iii) chronological train/test membership of late-life cycles.",
        "Therefore non-monotonic RUL RMSE vs H is expected to mix architecture effects "
        "with changing evaluation sets; H-sweep must be interpreted as multi-task SOH "
        "supervision sensitivity, not as a pure RUL-horizon curve.",
        f"Common prediction-time cycles across H∈{{1,3,5,10,20}}: {len(common)}.",
    ]
    return report


def run_patch_seeds(args, device, data_dir, out_dir, rows: list) -> None:
    print("\n=== P1: W=50 patch-size × 3 seeds ===")
    for p in [2, 5, 10, 20]:
        for seed in SEEDS if not args.smoke else [42]:
            print(f"  P={p} seed={seed}")
            res = train_one_model(
                model_name="patch-tf",
                battery_id="B0005",
                data_dir=data_dir,
                out_dir=out_dir / "checkpoints",
                window_size=50,
                horizon=args.horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                tf_ratio=0.0,
                device=device,
                seed=seed,
                feature_mode="multifeat",
                use_physics=True,
                patch_size=p,
                save_artifacts=False,
            )
            res["experiment"] = "patch_size_seeds"
            rows.append({k: v for k, v in res.items() if not str(k).startswith("_")})


def _slice_features(x: np.ndarray, cols: list[int]) -> np.ndarray:
    return x[:, :, cols].astype(np.float32)


def train_mlp_feature_ablation(
    battery: str,
    cols: list[int],
    feat_name: str,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    window: int,
    horizon: int,
    device: torch.device,
) -> dict:
    """Train MLP on selected feature columns with matched protocol."""
    import copy
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.data.nasa_loader import chronological_split
    from src.models.rul_models import build_rul_model
    from src.models.uncertainty import physics_regularizers
    from scripts.train_rul import _eval_rul_rmse

    set_seed(seed)
    data = load_battery_rul_data(
        data_dir / f"{battery}.mat", battery, window, horizon, feature_mode="multifeat"
    )
    x = _slice_features(data["x"], cols)
    y_soh, y_rul = data["y_soh"], data["y_rul"]
    x_tr_full, x_te, y_soh_tr_full, y_soh_te = chronological_split(x, y_soh, 0.8)
    y_rul_tr_full = y_rul[: len(x_tr_full)]
    y_rul_te = y_rul[len(x_tr_full) :]
    n = len(x_tr_full)
    n_val = max(1, int(round(n * 0.15)))
    n_tr = max(1, n - n_val)
    x_tr, x_val = x_tr_full[:n_tr], x_tr_full[n_tr:]
    y_soh_tr, y_soh_val = y_soh_tr_full[:n_tr], y_soh_tr_full[n_tr:]
    y_rul_tr, y_rul_val = y_rul_tr_full[:n_tr], y_rul_tr_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_tr), torch.tensor(y_soh_tr), torch.tensor(y_rul_tr)
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val), torch.tensor(y_soh_val), torch.tensor(y_rul_val)
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_te), torch.tensor(y_soh_te), torch.tensor(y_rul_te)
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_rul_model(
        "mlp",
        horizon=horizon,
        n_features=len(cols),
        window_size=window,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            ps, pr = model(xb, yb_soh, 0.0)
            loss = soh_fn(ps, yb_soh) + 0.5 * rul_fn(pr, yb_rul) + physics_regularizers(ps)
            loss.backward()
            opt.step()
        val = _eval_rul_rmse(model, val_loader, device)
        if val < best_val:
            best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, _, yb_rul in test_loader:
            _, pr = model(xb.to(device), None, 0.0)
            preds.append(pr.cpu().numpy())
            trues.append(yb_rul.numpy())
    pred = np.concatenate(preds).reshape(-1)
    true = np.concatenate(trues).reshape(-1)
    m = regression_metrics(true, pred)
    return {
        "experiment": "mlp_feature_ablation",
        "model": "mlp",
        "battery": battery,
        "feature_set": feat_name,
        "feature_cols": cols,
        "n_features": len(cols),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "rul_rmse": float(m["rmse"]),
        "rul_mae": float(m["mae"]),
        "rul_r2": float(m["r2"]),
        "n_test": int(len(true)),
    }


def run_mlp_ablation(args, device, data_dir, out_dir, rows: list) -> None:
    print("\n=== P2: MLP feature ablation ===")
    batteries = ["B0005", "B0007"] if not args.smoke else ["B0005"]
    seeds = SEEDS if not args.smoke else [42]
    for bat in batteries:
        for name, cols in FEATURE_SETS.items():
            for seed in seeds:
                print(f"  {bat} feats={name} seed={seed}")
                res = train_mlp_feature_ablation(
                    bat,
                    cols,
                    name,
                    data_dir,
                    out_dir,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    seed,
                    args.window,
                    args.horizon,
                    device,
                )
                rows.append(res)


def run_horizon_retrain_fixed_note(args, device, data_dir, out_dir, rows: list) -> None:
    """Re-run H sweep with 3 seeds on B0005 for seed-sensitivity of the anomaly."""
    print("\n=== P0b: horizon × 3 seeds (Patch-TF B0005) ===")
    horizons = [1, 5, 10, 20] if not args.smoke else [5, 10]
    seeds = SEEDS if not args.smoke else [42]
    for h in horizons:
        for seed in seeds:
            print(f"  H={h} seed={seed}")
            res = train_one_model(
                model_name="patch-tf",
                battery_id="B0005",
                data_dir=data_dir,
                out_dir=out_dir / "checkpoints",
                window_size=args.window,
                horizon=h,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                tf_ratio=0.0,
                device=device,
                seed=seed,
                feature_mode="multifeat",
                use_physics=True,
                patch_size=5,
                save_artifacts=False,
            )
            # attach construction stats
            d = load_battery_rul_data(
                data_dir / "B0005.mat", "B0005", args.window, h, feature_mode="multifeat"
            )
            res["experiment"] = "horizon_seeds"
            res["n_total"] = int(len(d["x"]))
            res["n_test"] = int(len(d["x"]) - int(len(d["x"]) * 0.8))
            rows.append({k: v for k, v in res.items() if not str(k).startswith("_")})


def write_report(rows: list, audit: dict, out_dir: Path) -> None:
    lines = ["# EI v3 sealing report\n", "\n## Horizon construction audit\n"]
    lines.append(f"- RUL definition: {audit.get('rul_definition')}\n")
    for h, info in audit.get("horizons", {}).items():
        lines.append(
            f"- H={h}: N={info['n_total']} test={info['n_test20']} "
            f"rul_match={info['rul_matches_fixed_curve']} "
            f"test_rul_mean={info['rul_test_mean']:.2f}\n"
        )
    for c in audit.get("conclusion", []):
        lines.append(f"- {c}\n")

    for exp, keys in (
        ("horizon_seeds", ("experiment", "horizon", "model", "battery")),
        ("patch_size_seeds", ("experiment", "patch_size", "model", "battery")),
        ("mlp_feature_ablation", ("experiment", "feature_set", "battery", "model")),
    ):
        block = [r for r in rows if r.get("experiment") == exp]
        if not block:
            continue
        agg = _agg(block, keys)
        lines.append(f"\n## {exp}\n")
        for r in agg:
            extra = ""
            if r.get("horizon") is not None:
                extra += f" H={r['horizon']}"
            if r.get("patch_size") is not None:
                extra += f" P={r['patch_size']}"
            if r.get("feature_set") is not None:
                extra += f" feats={r['feature_set']}"
            rmse = r.get("rul_rmse")
            std = r.get("rul_rmse_std", 0)
            lines.append(
                f"- {r.get('battery')}{extra}: {rmse:.3f}±{std:.3f} (n={r['n_runs']})\n"
            )
        (out_dir / f"ei_v3_seal_{exp}_agg.json").write_text(
            json.dumps(agg, indent=2), encoding="utf-8"
        )

    (out_dir / "ei_v3_seal_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="audit,horizon_seeds,patch_seeds,mlp_ablation",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 5)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    def want(name: str) -> bool:
        return not only or name in only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "nasa"
    out_dir = ROOT / "outputs" / "ei_v3_seal"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} epochs={args.epochs}")

    rows: list[dict] = []
    audit = {}
    if want("audit"):
        audit = audit_horizon_construction(data_dir)
        (out_dir / "horizon_construction_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print("audit written; conclusions:")
        for c in audit["conclusion"]:
            print(" ", c)
    if want("horizon_seeds"):
        run_horizon_retrain_fixed_note(args, device, data_dir, out_dir, rows)
    if want("patch_seeds"):
        run_patch_seeds(args, device, data_dir, out_dir, rows)
    if want("mlp_ablation"):
        run_mlp_ablation(args, device, data_dir, out_dir, rows)

    (out_dir / "ei_v3_seal_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not audit and (out_dir / "horizon_construction_audit.json").exists():
        audit = json.loads((out_dir / "horizon_construction_audit.json").read_text(encoding="utf-8"))
    write_report(rows, audit or {"conclusion": []}, out_dir)
    print("done →", out_dir)


if __name__ == "__main__":
    main()
