"""Scoring for NeuronBench. Two axes, both pure metrics -- no inference lives here.

1. Mechanism selection: did the agent identify the true mechanism (0/1 accuracy), and how much
   posterior mass did it place on the truth (a Brier score for the two-hypothesis decision)?

2. Held-out interventional forecasting: on a disjoint set of test protocols the agent never ran, how
   well does its predicted test-window spike count match the true cell's? Reported as a floored MSE,
   so that irreducible count noise does not dominate.
"""
from __future__ import annotations

import zlib

import numpy as np

from .worlds import WORLDS, POOL, world_models, spikes as _det_spikes, build_I as _build_I
from .stochastic import run_particles as _stoch_run, DT
from .features import spike_count as _spike_count

MSE_FLOOR = 0.25       # forecasts within +/-0.5 spikes of truth are treated as exact


def true_mechanism(world):
    """The (name, kwargs) of the world's true mechanism (its novel alternative)."""
    name = WORLDS[world]["name"]
    return name, world_models(world)[name]


def held_out_targets(world, stochastic=False, N=100.0, seed=0, reps=200):
    """Ground-truth held-out test-set targets: {protocol_label: expected test-window spike count} for
    the true cell. Deterministic worlds give the exact count; stochastic worlds average `reps` noisy
    rollouts to a stable expected count."""
    _, truth_kw = true_mechanism(world)
    out = {}
    for lab, seg in WORLDS[world]["test"]:
        if stochastic:
            I, ts = _build_I(seg, dt=DT)
            rng = np.random.default_rng(seed + zlib.crc32(lab.encode()) % 10_000)
            V = _stoch_run(reps, I, truth_kw, N, rng, DT)
            out[lab] = float(_spike_count(V, ts).mean())
        else:
            out[lab] = float(_det_spikes(truth_kw, seg))
    return out


def forecast_mse(predictions, targets, floor=MSE_FLOOR):
    """Floored MSE between predicted and true test-window spike counts over their shared protocol
    labels. `predictions` and `targets` are {label: count} dicts."""
    labs = [k for k in targets if k in predictions]
    if not labs:
        raise ValueError("no overlapping protocol labels between predictions and targets")
    err = np.array([predictions[k] - targets[k] for k in labs], dtype=float)
    return float(max(np.mean(err ** 2) - floor, 0.0))


def has_latent(world):
    """Whether the true cell has a latent extra mechanism beyond the plain Na+K+leak spiker. True for
    all six worlds as shipped; a plain-null world (truth == plain) would return False, making the
    presence/absence of a latent mechanism a genuine open-ended decision."""
    return True


def battery_targets(world, stochastic=False, N=100.0, seed=0, reps=200):
    """Ground-truth expected test-window spike counts for the true cell over the FULL design pool (all
    9 protocols) -- a comprehensive behavioural fingerprint used to score how well a submitted model
    *recovers* the mechanism (behavioural equivalence), as opposed to the held-out forecast."""
    _, truth_kw = true_mechanism(world)
    out = {}
    for lab, seg in POOL:
        if stochastic:
            I, ts = _build_I(seg, dt=DT)
            rng = np.random.default_rng(seed + zlib.crc32(lab.encode()) % 10_000)
            V = _stoch_run(reps, I, truth_kw, N, rng, DT)
            out[lab] = float(_spike_count(V, ts).mean())
        else:
            out[lab] = float(_det_spikes(truth_kw, seg))
    return out


def recovery_mse(predictions, world, stochastic=False, N=100.0, seed=0, floor=MSE_FLOOR):
    """Behavioural model-recovery: floored MSE of the submitted model's predicted spike counts against
    the true cell over the FULL design pool. Near-zero means the agent's argmax model is behaviourally
    equivalent to the truth (it recovered the mechanism), even if expressed in the agent's own terms --
    the metric that lets an OPEN-ended `argmax_m p(m|D)` be compared to the truth without a shared label.
    `predictions` is a {protocol_label: count} dict spanning the pool."""
    targets = battery_targets(world, stochastic=stochastic, N=N, seed=seed)
    return forecast_mse(predictions, targets, floor=floor)


# -- deprecated two-hypothesis conveniences (kept for backward compatibility; prefer the open-ended
#    forecast_mse + recovery_mse + has_latent above) --
def selection_correct(chosen_name, world):
    """DEPRECATED (two-hypothesis framing). 1 if `chosen_name` matches the world's true mechanism name."""
    return int(chosen_name == WORLDS[world]["name"])


def selection_brier(prob_true):
    """DEPRECATED (two-hypothesis framing). Brier score (1 - prob_true)^2 for the mass on the truth."""
    p = float(np.clip(prob_true, 0.0, 1.0))
    return (1.0 - p) ** 2
