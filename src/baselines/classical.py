"""Non-NN RUL baselines (persistence / linear)."""

from __future__ import annotations

import numpy as np


def _soh_from_window(x: np.ndarray) -> np.ndarray:
    """Extract SOH series from window; prefer feature-0 (cap_norm / soh)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[:, 0]
    raise ValueError(f"Expected window (W,F), got {x.shape}")


def persistence_rul(x: np.ndarray, eol_threshold: float = 0.7) -> float:
    """
    Constant-fade persistence (SOH-window): assume a constant fade rate from
    first→last SOH in the window and extrapolate cycles to the EOL threshold.

    This is *not* RUL_t = RUL_{t-1}. Regeneration or a near-zero slope makes
    the implied remaining life explode (see persistence_rul_hold).
    """
    soh = _soh_from_window(x)
    w = len(soh)
    if w < 2:
        return 0.0
    slope = (soh[-1] - soh[0]) / max(w - 1, 1)
    if slope >= -1e-8:
        # non-degrading / flat: return large remaining life proxy
        return float(max(0.0, (soh[-1] - eol_threshold) / 1e-4))
    cycles = (eol_threshold - soh[-1]) / slope
    return float(max(0.0, cycles))


def linear_extrapolation_rul(x: np.ndarray, eol_threshold: float = 0.7) -> float:
    """Least-squares linear fit of SOH vs local cycle index → cycles to EOL."""
    soh = _soh_from_window(x)
    w = len(soh)
    t = np.arange(w, dtype=np.float64)
    # soh = a*t + b
    a, b = np.polyfit(t, soh.astype(np.float64), 1)
    if a >= -1e-8:
        return float(max(0.0, (soh[-1] - eol_threshold) / 1e-4))
    t_eol = (eol_threshold - b) / a
    return float(max(0.0, t_eol - (w - 1)))


def persistence_rul_recursive(
    x_full: np.ndarray,
    n_train: int,
    eol_threshold: float = 0.7,
) -> np.ndarray:
    """L0 recursive persistence: initialize with persist_fade, then hat_t = max(hat_{t-1}-1, 0).

    Does not use ground-truth RUL labels.
    """
    x = np.asarray(x_full)
    n = len(x)
    if n_train < 1 or n_train >= n:
        raise ValueError("n_train must leave a nonempty test set")
    prev = persistence_rul(x[n_train - 1], eol_threshold)
    pred = np.empty(n - n_train, dtype=np.float32)
    for i in range(n - n_train):
        pred[i] = prev
        prev = float(max(prev - 1.0, 0.0))
    return pred


def persistence_rul_hold(
    y_rul_full: np.ndarray,
    n_train: int,
) -> np.ndarray:
    """Oracle label walk: RUL_hat_t = true RUL_{t-1}. Not an L0 BMS baseline."""
    y = np.asarray(y_rul_full, dtype=np.float32).reshape(-1)
    pred = np.empty(len(y) - n_train, dtype=np.float32)
    prev = y[n_train - 1] if n_train > 0 else y[0]
    for i, t in enumerate(range(n_train, len(y))):
        pred[i] = prev
        prev = y[t]
    return pred


def persistence_rul_mean(
    y_rul_full: np.ndarray,
    n_train: int,
) -> np.ndarray:
    """Expanding mean of historical true RUL (train+past test labels)."""
    y = np.asarray(y_rul_full, dtype=np.float32).reshape(-1)
    pred = np.empty(len(y) - n_train, dtype=np.float32)
    for i, t in enumerate(range(n_train, len(y))):
        pred[i] = float(np.mean(y[:t])) if t > 0 else float(y[0])
    return pred


def evaluate_classical(
    x_test: np.ndarray,
    y_rul_test: np.ndarray,
    method: str = "linear",
    eol_threshold: float = 0.7,
    y_rul_full: np.ndarray | None = None,
    n_train: int | None = None,
    x_full: np.ndarray | None = None,
) -> dict:
    """Return RUL metrics for classical method on test windows."""
    from src.utils.metrics import regression_metrics

    true = np.asarray(y_rul_test, dtype=np.float32).reshape(-1)
    if method == "persist_recursive":
        if x_full is None or n_train is None:
            raise ValueError("persist_recursive requires x_full and n_train")
        pred = persistence_rul_recursive(x_full, n_train, eol_threshold)
    elif method == "persist_hold":
        if y_rul_full is None or n_train is None:
            raise ValueError("persist_hold requires y_rul_full and n_train")
        pred = persistence_rul_hold(y_rul_full, n_train)
    elif method == "persist_mean":
        if y_rul_full is None or n_train is None:
            raise ValueError("persist_mean requires y_rul_full and n_train")
        pred = persistence_rul_mean(y_rul_full, n_train)
    else:
        preds = []
        for i in range(len(x_test)):
            if method in {"persistence", "persist_fade"}:
                preds.append(persistence_rul(x_test[i], eol_threshold))
            else:
                preds.append(linear_extrapolation_rul(x_test[i], eol_threshold))
        pred = np.asarray(preds, dtype=np.float32)
    m = regression_metrics(true, pred)
    return {
        "model": method,
        "rul_rmse": float(m["rmse"]),
        "rul_mae": float(m["mae"]),
        "rul_r2": float(m["r2"]),
        "n_test": int(len(true)),
    }
