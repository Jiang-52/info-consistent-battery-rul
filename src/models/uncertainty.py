"""MC-Dropout uncertainty utilities for prognostic models."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Keep dropout layers active during eval for MC sampling."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    Monte-Carlo Dropout predictive mean/std for SOH horizon and RUL.

    Returns dict with soh_mean, soh_std, rul_mean, rul_std.
    """
    if device is None:
        device = x.device
    model.eval()
    enable_dropout(model)
    soh_samples, rul_samples = [], []
    xb = x.to(device)
    for _ in range(n_samples):
        soh, rul = model(xb, target=None, teacher_forcing_ratio=0.0)
        soh_samples.append(soh.detach().cpu().numpy())
        rul_samples.append(rul.detach().cpu().numpy())
    soh_arr = np.stack(soh_samples, axis=0)
    rul_arr = np.stack(rul_samples, axis=0)
    return {
        "soh_mean": soh_arr.mean(axis=0),
        "soh_std": soh_arr.std(axis=0),
        "rul_mean": rul_arr.mean(axis=0),
        "rul_std": rul_arr.std(axis=0),
    }


def interval_coverage(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Empirical coverage of mean ± z*std intervals (PICP)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_mean = np.asarray(y_mean).reshape(-1)
    y_std = np.asarray(y_std).reshape(-1)
    lo = y_mean - z * y_std
    hi = y_mean + z * y_std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def picp(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Prediction Interval Coverage Probability."""
    return interval_coverage(y_true, y_mean, y_std, z=z)


def mpiw(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Mean Prediction Interval Width."""
    y_std = np.asarray(y_std).reshape(-1)
    return float(np.mean(2.0 * z * y_std))


def pinaw(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Prediction Interval Normalized Average Width."""
    y_true = np.asarray(y_true).reshape(-1)
    width = mpiw(y_mean, y_std, z=z)
    span = float(np.max(y_true) - np.min(y_true))
    if span < 1e-8:
        span = 1.0
    return float(width / span)


def calibration_curve(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    nominal_levels: list[float] | None = None,
) -> list[dict]:
    """
    Reliability diagram points: for each nominal coverage, empirical PICP.
    z from standard Normal inverse CDF of (1+p)/2.
    """
    # Avoid scipy dependency: common z values
    z_map = {
        0.5: 0.67448975,
        0.7: 1.03643339,
        0.8: 1.28155157,
        0.9: 1.64485363,
        0.95: 1.95996398,
    }
    if nominal_levels is None:
        nominal_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for p in nominal_levels:
        z = float(z_map.get(round(p, 2), 1.96))
        emp = picp(y_true, y_mean, y_std, z=z)
        rows.append(
            {
                "nominal": float(p),
                "empirical": emp,
                "z": z,
                "pinaw": pinaw(y_true, y_mean, y_std, z=z),
                "mpiw": mpiw(y_mean, y_std, z=z),
            }
        )
    return rows


def physics_regularizers(
    pred_soh: torch.Tensor,
    lambda_mono: float = 0.01,
    lambda_smooth: float = 0.01,
) -> torch.Tensor:
    """
    Soft physics losses on multi-step SOH predictions.
    - monotonicity: discourage SOH increases along horizon
    - smoothness: penalize large adjacent jumps
    """
    if pred_soh.size(-1) < 2:
        return pred_soh.new_zeros(())
    diffs = pred_soh[..., 1:] - pred_soh[..., :-1]
    mono = torch.relu(diffs).pow(2).mean()
    smooth = diffs.pow(2).mean()
    return lambda_mono * mono + lambda_smooth * smooth
