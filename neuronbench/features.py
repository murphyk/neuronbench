"""Spike-count observables.

The benchmark's scored observable is the **test-window spike count** (the count of action potentials a
protocol elicits after any conditioning pre-pulse). These helpers count spikes as upward zero-crossings
of the membrane voltage, over the whole trace or a sub-window. Pure numpy; operate on a (P, T) array of
voltage traces (P repeats/particles, T time samples).

How a solver chooses to *summarise a raw voltage trace into features* to build a likelihood is its own
modelling decision and is deliberately not part of this package -- the benchmark hands you the noisy
voltage and the spike count, nothing more.
"""
from __future__ import annotations

import numpy as np


def spike_count(V2d, test_start):
    """Spike counts (upward zero-crossings of V) in the test window [test_start:] for each row of a
    (P, T) voltage array."""
    up = (V2d[:, :-1] < 0) & (V2d[:, 1:] >= 0)
    return up[:, test_start:].sum(axis=1).astype(float)


def spikes_in(V2d, a, b):
    """Upward-zero-crossing counts in the window [a, b) for each row of a (P, T) voltage array (e.g. the
    conditioning pre-pulse window vs the test window)."""
    a = max(a, 0); b = min(b, V2d.shape[1])
    if b - a < 2:
        return np.zeros(V2d.shape[0])
    up = (V2d[:, a:b - 1] < 0) & (V2d[:, a + 1:b] >= 0)
    return up.sum(axis=1).astype(float)
