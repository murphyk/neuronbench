"""Scoring for NeuronBench. Pure metrics -- no inference lives here.

The task is open-ended counterfactual interventional forecasting: on a disjoint set of test protocols the
agent never ran, how well does it predict the true cell's response? Two metrics, at two levels of detail:

1. ``spike_forecast_mse`` (headline): floored MSE of the predicted test-window spike count vs. the true
   cell. Any agent -- LLM or model-based -- can produce it, so it is the comparable number in the plots.
   The floor keeps irreducible count noise from dominating.

2. ``feature_forecast_mse`` (secondary): standardised MSE between a feature vector of the agent's predicted
   voltage trace and the true cell's -- spike counts plus sub-threshold shape (V-min, steady state,
   run-down, adaptation). It rewards trajectory shape that a scalar spike count discards.

(Earlier mechanism-selection conveniences -- a 0/1 identification accuracy and a two-hypothesis Brier
score -- were removed: the benchmark is open-ended and scores what the agent *forecasts*, not whether it
names the truth.)
"""
from __future__ import annotations

import zlib

import numpy as np

from .worlds import WORLDS, world_models, spikes as _det_spikes, build_I as _build_I, simulate as _det_sim
from .stochastic import run_particles as _stoch_run, DT, make_protocol as _stoch_proto
from .features import spike_count as _spike_count, spikes_in as _spikes_in

MSE_FLOOR = 0.25       # forecasts within +/-0.5 spikes of truth are treated as exact

# Per-feature standardisation for the (secondary) feature-forecast metric: spike-count features are on a
# ~unit scale, sub-threshold voltage features are in mV (tens), so we divide each squared error by these
# scales before averaging, giving every feature comparable weight.
_EVAL_FEAT_SCALES = np.array([3.0, 3.0, 3.0, 3.0, 12.0, 8.0])


def eval_features(V, test_start):
    """The benchmark's canonical evaluation feature vector of a single voltage trace `V` (1-D), with the
    scored test window starting at index `test_start`:
      [ n_test, n_pre, run-down (=n_pre-n_test), within-test adaptation, V_min, V_end ].
    This is the benchmark's *internal* scoring feature map --- agents forecast raw traces, not features."""
    V = np.asarray(V, float)[None, :]
    T = V.shape[1]
    n_test = _spikes_in(V, test_start, T)[0]
    n_pre = _spikes_in(V, 0, test_start)[0]
    mid = test_start + (T - test_start) // 2
    adapt = _spikes_in(V, test_start, mid)[0] - _spikes_in(V, mid, T)[0]
    vmin = float(V.min())
    vend = float(V[0, -max(T // 20, 1):].mean())
    return np.array([n_test, n_pre, n_pre - n_test, adapt, vmin, vend], float)


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
    labels. `predictions` and `targets` are {label: count} dicts. This is the HEADLINE spike-forecast
    metric (both LLM and model-based agents can produce it)."""
    labs = [k for k in targets if k in predictions]
    if not labs:
        raise ValueError("no overlapping protocol labels between predictions and targets")
    err = np.array([predictions[k] - targets[k] for k in labs], dtype=float)
    return float(max(np.mean(err ** 2) - floor, 0.0))


def held_out_feature_targets(world, stochastic=False, N=100.0, seed=0, reps=200):
    """Ground-truth held-out feature targets: {label: (feature_vector, trace_len, test_start)} for the
    true cell, where feature_vector = eval_features of the true trace. Used by feature_forecast_mse."""
    _, truth_kw = true_mechanism(world)
    out = {}
    for lab, seg in WORLDS[world]["test"]:
        if stochastic:
            I, obs_idx, ts = _stoch_proto(seg)
            rng = np.random.default_rng(seed + zlib.crc32(lab.encode()) % 10_000)
            V = _stoch_run(reps, I, truth_kw, N, rng, DT)          # (reps, T)
            feats = np.mean([eval_features(V[r], ts) for r in range(reps)], axis=0)
            out[lab] = (feats, V.shape[1], ts)
        else:
            I, ts = _build_I(seg)
            _, V = _det_sim(I, ts, **truth_kw, trace=True)
            out[lab] = (eval_features(V, ts), len(V), ts)
    return out


def _resample(trace, n):
    """Linearly resample a 1-D predicted trace to length `n` (so an agent may submit any sampling)."""
    trace = np.asarray(trace, float).ravel()
    if trace.size == 0:
        return np.zeros(n)
    if trace.size == n:
        return trace
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, trace.size), trace)


def feature_forecast_mse(predicted_traces, world, stochastic=False, N=100.0, seed=0):
    """SECONDARY metric. Standardised MSE between the eval-feature vector of the agent's predicted voltage
    trace and the true cell's, averaged over held-out protocols and features. `predicted_traces` is a
    {label: 1-D voltage array} dict; a missing or empty trace (e.g. an LLM that cannot forecast a
    trajectory) scores against a flat 0-mV trace --- i.e. it is penalised, not excused. The benchmark
    extracts the features itself (agents forecast traces, never features)."""
    targets = held_out_feature_targets(world, stochastic=stochastic, N=N, seed=seed)
    errs = []
    for lab, (true_feat, tlen, ts) in targets.items():
        tr = predicted_traces.get(lab)
        pred_feat = eval_features(_resample(tr, tlen), ts) if tr is not None and len(tr) \
            else eval_features(np.zeros(tlen), ts)
        errs.append(((pred_feat - true_feat) / _EVAL_FEAT_SCALES) ** 2)
    return float(np.mean(errs))


# The held-out interventional forecast MSE (above) is the benchmark's single evaluation metric. Earlier
# two-hypothesis conveniences (selection accuracy/Brier) and the full-pool behavioural-recovery MSE were
# removed: the task is open-ended, and held-out forecasting on the discriminating test protocols already
# captures whether the agent recovered the mechanism -- a separate recovery metric was weaker (diluted by
# the textbook protocols where every mechanism agrees) and largely redundant.
