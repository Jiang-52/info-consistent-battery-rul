"""L1P must be causal on the target: L1P[i] uses only soh[i], never soh[i+1:]."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cycle_features import soh_matched_rul_prior


def test_l1p_invariant_to_target_future_soh() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(80, dtype=np.float64)
    target = np.clip(1.0 - t / 200.0 + 0.01 * rng.normal(size=len(t)), 0.55, 1.05)
    sib_a = np.clip(1.0 - t / 150.0, 0.5, 1.05)
    sib_b = np.clip(1.0 - t / 280.0, 0.5, 1.05)
    base = soh_matched_rul_prior(target, [sib_a, sib_b], eol_threshold=0.8)
    future = target.copy()
    future[40:] = 0.99  # destroy post-t SOH
    mutated = soh_matched_rul_prior(future, [sib_a, sib_b], eol_threshold=0.8)
    np.testing.assert_allclose(base[:40], mutated[:40], rtol=0, atol=1e-6)
    # prefix-only construction matches the vectorized channel at a held-out t
    t0 = 25
    one = soh_matched_rul_prior(target[: t0 + 1], [sib_a, sib_b], eol_threshold=0.8)
    assert abs(float(one[-1]) - float(base[t0])) < 1e-5


if __name__ == "__main__":
    test_l1p_invariant_to_target_future_soh()
    print("L1P causal audit passed")
