"""Published summary features s(y) of a voltage trace.

These are the observable summaries the benchmark exposes for the stochastic worlds: spike-rate
signatures plus sub-threshold shape, so both the rate worlds (na_fatigue, textbook_M) and the
sub-threshold / burst worlds (h_sag, ca_rebound) leave a signal. Pure numpy; operates on a (P, T)
array of voltage traces (P repeats/particles, T time samples). Contains no likelihood -- how you
turn features into an inference is the solver's choice.
"""
from __future__ import annotations

import numpy as np

# Per-feature standard-deviation floor (spike-count features on a ~unit scale, voltages in mV), so a
# degenerate zero-variance simulated feature cannot dominate a downstream Gaussian score.
FEAT_SD_FLOOR = np.array([0.7, 0.7, 0.7, 0.7, 0.3, 0.3])


def spike_count(V2d, test_start):
    """Spike counts (upward zero-crossings of V) in the test window [test_start:] for each row of a
    (P, T) voltage array."""
    up = (V2d[:, :-1] < 0) & (V2d[:, 1:] >= 0)
    return up[:, test_start:].sum(axis=1).astype(float)


def spikes_in(V2d, a, b):
    """Upward-zero-crossing counts in [a, b) for each row of a (P, T) voltage array."""
    a = max(a, 0); b = min(b, V2d.shape[1])
    if b - a < 2:
        return np.zeros(V2d.shape[0])
    up = (V2d[:, a:b - 1] < 0) & (V2d[:, a + 1:b] >= 0)
    return up.sum(axis=1).astype(float)


def feature_vector(V2d, test_start):
    """The stochastic-benchmark summary feature vector s(y), computed per trace:

      [ n_test,   spikes in the test window (the f-I / rate feature)
        n_pre,    spikes BEFORE the test window (conditioning-pulse / rebound spikes)
        rundown,  n_pre - n_test  (use-dependent / cross-pulse fatigue -- the na_fatigue signature)
        adapt,    early-half minus late-half spikes within the test window (within-pulse adaptation)
        vmin,     min voltage over the trace (sub-threshold sag / hyperpolarisation depth)
        vend ]    mean voltage in the last ~20 ms (steady state / after-hyperpolarisation)

    Returns a (P, 6) array for a (P, T) input.
    """
    T = V2d.shape[1]
    n_test = spikes_in(V2d, test_start, T)
    n_pre = spikes_in(V2d, 0, test_start)
    mid = test_start + (T - test_start) // 2
    adapt = spikes_in(V2d, test_start, mid) - spikes_in(V2d, mid, T)
    vmin = V2d.min(axis=1)
    vend = V2d[:, -800:].mean(axis=1)
    return np.stack([n_test, n_pre, n_pre - n_test, adapt, vmin, vend], axis=1)
