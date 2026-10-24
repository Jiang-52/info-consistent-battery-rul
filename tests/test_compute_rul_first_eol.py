"""RUL labels use first EOL crossing, not next forward crossing after regen."""

from __future__ import annotations

import numpy as np

from src.data.rul_dataset import compute_rul, eol_cycle_index


def test_rul_equals_T_minus_t_with_regeneration():
    # Cross 0.7 at t=5, recover above 0.7, cross again at t=12.
    soh = np.array(
        [1.0, 0.95, 0.9, 0.85, 0.8, 0.65, 0.75, 0.78, 0.76, 0.74, 0.72, 0.71, 0.60, 0.55],
        dtype=np.float32,
    )
    T = eol_cycle_index(soh, 0.7)
    assert T == 5
    rul = compute_rul(soh, 0.7)
    t = np.arange(len(soh))
    np.testing.assert_array_equal(rul, np.maximum(0, T - t).astype(np.float32))
    # No positive RUL after first EOL (regeneration must not reopen countdown).
    assert not np.any((t > T) & (rul > 0))


def test_rul_never_hits_eol_counts_to_record_end():
    soh = np.linspace(1.0, 0.75, 20, dtype=np.float32)
    T = eol_cycle_index(soh, 0.7)
    assert T == len(soh) - 1
    rul = compute_rul(soh, 0.7)
    np.testing.assert_array_equal(
        rul, np.maximum(0, T - np.arange(len(soh))).astype(np.float32)
    )
