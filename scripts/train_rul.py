"""Train RUL / multi-step SOH models with model comparison."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nasa_loader import chronological_split
from src.data.rul_dataset import load_battery_rul_data
from src.models.rul_models import build_rul_model, iterative_multistep_predict
from src.utils.metrics import regression_metrics
from src.utils.plotting import save_loss_curve


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval_rul_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, trues = [], []
    for xb, _, yb_rul in loader:
        xb = xb.to(device)
        _, pred_rul = model(xb, target=None, teacher_forcing_ratio=0.0)
        preds.append(pred_rul.cpu().numpy())
        trues.append(yb_rul.numpy())
    pred = np.concatenate(preds).reshape(-1)
    true = np.concatenate(trues).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def train_one_model(
    model_name: str,
    battery_id: str,
    data_dir: Path,
    out_dir: Path,
    window_size: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tf_ratio: float,
    device: torch.device,
    lambda_rul: float = 0.5,
    save_artifacts: bool = True,
    seed: int = 42,
    val_ratio: float = 0.15,
    feature_mode: str = "soh",
    use_physics: bool = False,
    lambda_mono: float = 0.01,
    lambda_smooth: float = 0.01,
    patch_size: int = 5,
    d_model: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> dict:
    """Train with chronological val split and restore best-val RUL checkpoint."""
    from src.models.uncertainty import physics_regularizers

    set_seed(seed)
    mat_path = data_dir / f"{battery_id}.mat"
    data = load_battery_rul_data(
        mat_path, battery_id, window_size, horizon, feature_mode=feature_mode
    )
    x, y_soh, y_rul = data["x"], data["y_soh"], data["y_rul"]
    n_features = int(data.get("n_features", x.shape[-1]))
    x_train_full, x_test, y_soh_train_full, y_soh_test = chronological_split(x, y_soh, 0.8)
    y_rul_train_full = y_rul[: len(x_train_full)]
    y_rul_test = y_rul[len(x_train_full) :]

    n = len(x_train_full)
    n_val = max(1, int(round(n * val_ratio)))
    n_tr = max(1, n - n_val)
    x_train, x_val = x_train_full[:n_tr], x_train_full[n_tr:]
    y_soh_train, y_soh_val = y_soh_train_full[:n_tr], y_soh_train_full[n_tr:]
    y_rul_train, y_rul_val = y_rul_train_full[:n_tr], y_rul_train_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_soh_train, dtype=torch.float32),
        torch.tensor(y_rul_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_soh_val, dtype=torch.float32),
        torch.tensor(y_rul_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_soh_test, dtype=torch.float32),
        torch.tensor(y_rul_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_rul_model(
        model_name,
        horizon=horizon,
        n_features=n_features,
        d_model=d_model,
        num_layers=num_layers,
        patch_size=patch_size,
        dropout=dropout,
        window_size=window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_loss_fn = nn.MSELoss()
    rul_loss_fn = nn.MSELoss()
    losses: list[float] = []
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb_soh, yb_rul in train_loader:
            xb = xb.to(device)
            yb_soh = yb_soh.to(device)
            yb_rul = yb_rul.to(device)
            opt.zero_grad()
            pred_soh, pred_rul = model(xb, yb_soh, teacher_forcing_ratio=tf_ratio)
            loss = soh_loss_fn(pred_soh, yb_soh) + lambda_rul * rul_loss_fn(pred_rul, yb_rul)
            if use_physics:
                loss = loss + physics_regularizers(pred_soh, lambda_mono, lambda_smooth)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        losses.append(epoch_loss)

        val_rmse = _eval_rul_rmse(model, val_loader, device)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"  [{model_name}|{feature_mode}|s{seed}] epoch {epoch}/{epochs} "
                f"train={epoch_loss:.6f} val_rul_rmse={val_rmse:.4f} best@{best_epoch}"
            )

    model.load_state_dict(best_state)
    model.eval()
    soh_preds, soh_true, rul_preds, rul_true = [], [], [], []
    with torch.no_grad():
        for xb, yb_soh, yb_rul in test_loader:
            xb = xb.to(device)
            pred_soh, pred_rul = model(xb, target=None, teacher_forcing_ratio=0.0)
            soh_preds.append(pred_soh.cpu().numpy())
            soh_true.append(yb_soh.numpy())
            rul_preds.append(pred_rul.cpu().numpy())
            rul_true.append(yb_rul.numpy())

    soh_pred = np.concatenate(soh_preds).reshape(-1)
    soh_true_flat = np.concatenate(soh_true).reshape(-1)
    rul_pred = np.concatenate(rul_preds).reshape(-1)
    rul_true_flat = np.concatenate(rul_true).reshape(-1)
    soh_metrics = regression_metrics(soh_true_flat, soh_pred)
    rul_metrics = regression_metrics(rul_true_flat, rul_pred)

    sample_hist = x_test[-1]
    future_true = y_soh_test[-1]
    if sample_hist.shape[-1] == 1:
        future_pred = iterative_multistep_predict(
            model, sample_hist, horizon, steps=horizon, device=device
        )
        iter_metrics = regression_metrics(future_true, future_pred)
    else:
        # Multi-feat models use direct horizon head (no SOH-only roll)
        with torch.no_grad():
            xb = torch.tensor(sample_hist[None, ...], dtype=torch.float32, device=device)
            future_pred, _ = model(xb, None, 0.0)
            future_pred = future_pred.cpu().numpy().reshape(-1)
        iter_metrics = regression_metrics(future_true, future_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{model_name}_{battery_id.lower()}_w{window_size}_h{horizon}_s{seed}"
        f"_{feature_mode}{'_phys' if use_physics else ''}"
    )
    if save_artifacts:
        torch.save(model.state_dict(), out_dir / f"rul_{tag}.pt")
        save_loss_curve(losses, out_dir / f"rul_loss_{tag}.png", f"{model_name} {battery_id} Loss")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(rul_true_flat, "k.", markersize=3, label="True RUL")
        axes[0].plot(rul_pred, "r.", markersize=3, label="Pred RUL")
        axes[0].set_title(f"{battery_id} RUL ({model_name})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(future_true, "ko-", label="True SOH")
        axes[1].plot(future_pred, "r--", label="Iterative pred")
        axes[1].set_title("Multi-step correction (last test window)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"rul_curve_{tag}.png", dpi=150)
        plt.close()

    return {
        "task": "rul",
        "model": model_name,
        "battery": battery_id,
        "window_size": window_size,
        "horizon": horizon,
        "epochs": epochs,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "lambda_rul": lambda_rul,
        "feature_mode": feature_mode,
        "use_physics": use_physics,
        "patch_size": patch_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "n_features": n_features,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "soh_rmse": soh_metrics["rmse"],
        "soh_mae": soh_metrics["mae"],
        "soh_r2": soh_metrics["r2"],
        "rul_rmse": rul_metrics["rmse"],
        "rul_mae": rul_metrics["mae"],
        "rul_r2": rul_metrics["r2"],
        "iter_rmse": iter_metrics["rmse"],
        "iter_mae": iter_metrics["mae"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RUL prediction models")
    parser.add_argument("--battery", default="B0005")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tcn", "tcn-attn", "tcn-bigru", "lstm"],
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "rul")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", default="soh", choices=["soh", "multifeat"])
    parser.add_argument("--use-physics", action="store_true")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batteries = ["B0005", "B0006", "B0007"] if args.all else [args.battery.upper()]
    results = []
    for bid in batteries:
        for model_name in args.models:
            print(f"\n=== RUL {model_name} on {bid} seed={args.seed} feat={args.feature_mode} ===")
            res = train_one_model(
                model_name,
                bid,
                args.data_dir,
                args.out_dir,
                args.window_size,
                args.horizon,
                args.epochs,
                args.batch_size,
                args.lr,
                args.tf_ratio,
                device,
                seed=args.seed,
                feature_mode=args.feature_mode,
                use_physics=args.use_physics,
                patch_size=args.patch_size,
                d_model=args.d_model,
                num_layers=args.num_layers,
            )
            results.append(res)
            print(
                f"  RUL RMSE={res['rul_rmse']:.3f} MAE={res['rul_mae']:.3f} | "
                f"SOH RMSE={res['soh_rmse']:.4f} best_epoch={res['best_epoch']}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "rul_metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        new_keys = {
            (r["model"], r["battery"], r.get("seed", args.seed), r.get("feature_mode", "soh"))
            for r in results
        }
        merged = [
            r
            for r in existing
            if (r["model"], r["battery"], r.get("seed", 42), r.get("feature_mode", "soh"))
            not in new_keys
        ]
        results = merged + results
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
