"""Plot helpers for experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_loss_curve(losses: list[float], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(df["model"].unique()) * 2), 4))
    models = list(df["model"].unique())
    x = np.arange(len(models))
    width = 0.25
    batteries = df["battery"].unique()
    for i, bat in enumerate(batteries):
        sub = df[df["battery"] == bat]
        vals = [sub[sub["model"] == m][metric].values[0] for m in models]
        plt.bar(x + i * width, vals, width=width, label=bat)
    plt.xticks(x + width, models)
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
