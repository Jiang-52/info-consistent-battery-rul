"""Train TCN-Autoencoder for cycle-level anomaly detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cycle_features import (
    build_sequence_windows,
    extract_discharge_cycle_features,
    inject_synthetic_anomalies,
)
from src.data.nasa_loader import chronological_split
from src.models.tcn_autoencoder import TCNAutoencoder
from src.utils.plotting import save_loss_curve


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_autoencoder(
    battery_id: str,
    data_dir: Path,
    out_dir: Path,
    window_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold_pct: float,
    device: torch.device,
) -> dict:
    mat_path = data_dir / f"{battery_id}.mat"
    feats = extract_discharge_cycle_features(mat_path, battery_id)
    windows = build_sequence_windows(feats, window_size)

    # Train on early 80% as "normal"; validate threshold on next 10%
    n = len(windows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    x_train = windows[:n_train]
    x_val = windows[n_train : n_train + n_val]
    x_test = windows[n_train + n_val :]

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TCNAutoencoder(
        n_features=feats.shape[1],
        seq_len=window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        losses.append(epoch_loss)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{epochs} loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        val_err = model.reconstruction_error(
            torch.tensor(x_val, dtype=torch.float32, device=device)
        ).cpu().numpy()
        threshold = float(np.percentile(val_err, threshold_pct))

    # Evaluate on test set with injected anomalies
    x_test_aug, y_test = inject_synthetic_anomalies(x_test, n_anomalies=max(5, len(x_test) // 5))
    with torch.no_grad():
        test_err = model.reconstruction_error(
            torch.tensor(x_test_aug, dtype=torch.float32, device=device)
        ).cpu().numpy()
    y_pred = (test_err > threshold).astype(np.int32)

    metrics = {
        "task": "anomaly",
        "battery": battery_id,
        "window_size": window_size,
        "epochs": epochs,
        "threshold": threshold,
        "threshold_percentile": threshold_pct,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "val_err_mean": float(val_err.mean()),
        "test_err_mean": float(test_err.mean()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = battery_id.lower()
    torch.save({"model": model.state_dict(), "threshold": threshold}, out_dir / f"ae_{tag}.pt")
    save_loss_curve(losses, out_dir / f"ae_loss_{tag}.png", f"TCN-AE {battery_id}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(val_err, label="Val error")
    axes[0].axhline(threshold, color="r", linestyle="--", label=f"Threshold p{threshold_pct}")
    axes[0].set_title("Validation reconstruction error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = np.where(y_test == 1, "red", "green")
    axes[1].scatter(np.arange(len(test_err)), test_err, c=colors, s=20, alpha=0.7)
    axes[1].axhline(threshold, color="r", linestyle="--")
    axes[1].set_title("Test errors (red=anomaly)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"ae_result_{tag}.png", dpi=150)
    plt.close()

    with open(out_dir / f"ae_metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN-Autoencoder anomaly detector")
    parser.add_argument("--battery", default="B0005")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nasa")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "anomaly")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-pct", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batteries = ["B0005", "B0006", "B0007"] if args.all else [args.battery.upper()]
    results = []
    for bid in batteries:
        print(f"\n=== Anomaly detection {bid} ===")
        res = train_autoencoder(
            bid,
            args.data_dir,
            args.out_dir,
            args.window_size,
            args.epochs,
            args.batch_size,
            args.lr,
            args.threshold_pct,
            device,
        )
        results.append(res)
        print(
            f"  F1={res['f1']:.3f} Acc={res['accuracy']:.3f} "
            f"Prec={res['precision']:.3f} Rec={res['recall']:.3f}"
        )

    with open(args.out_dir / "anomaly_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
