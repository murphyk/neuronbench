"""Stochastic forward model for the six worlds: the same deterministic mechanisms of ``worlds.py``,
with Fox--Lu channel noise substituted for every gate.

This is the *generative oracle* of the stochastic benchmark. Given a mechanism (a world candidate
``{extra: [Chan], slow_na: bool}``) and a protocol current, it propagates P independent stochastic HH
particles and returns their voltage traces. The observation model on top is a noisy, sub-sampled
voltage (``SIG_OBS`` mV, every ~4 ms) -- partial and noisy, so p(y | mechanism) is intractable and
must be estimated by simulation *by the solver*. No likelihood/inference lives here.
"""
from __future__ import annotations

import numpy as np

from . import _foxlu as fx
from ._foxlu import ENa, EK, EL, GNA, GK, GL
from .blockers import resolve as _resolve_blockers
from .worlds import build_I as _build_I

SIG_OBS = 2.0            # voltage observation noise (mV)
DT = 0.025               # integration step (ms)


def _chan_to_dict(c):
    """Convert a worlds.Chan dataclass into the plain dict the Fox--Lu extra-channel steppers use."""
    return dict(name=c.name, g=c.g, E=c.E, mvh=c.mvh, mk=c.mk, mtau=c.mtau, mpow=c.mpow,
                hvh=c.hvh, hk=c.hk, htau=c.htau, hpow=c.hpow)


def _slow_na_init(v):
    """Init the slow Na inactivation gate s (na_fatigue world) at its steady state."""
    return 1.0 / (1.0 + np.exp((v + 45.0) / 4.0))


def _slow_na_step(s, v, dt, N, rng):
    """One Fox--Lu step of the slow Na inactivation gate: xinf = s_inf(V), tau = tau_s(V)
    (fast inactivation, very slow recovery)."""
    s_inf = 1.0 / (1.0 + np.exp((v + 45.0) / 4.0))
    tau_s = np.where(v > -55.0, 380.0, 1500.0)
    return fx.tau_gate_step(s, s_inf, tau_s, dt, N, rng)


def run_particles(P, I, mechanism, N, rng, dt=DT, V0=-65.0, block=()):
    """Propagate P stochastic HH particles under current I(t) for a world candidate
    ``mechanism = {extra: [Chan], slow_na: bool}``; return the (P, T) voltage array. Fox--Lu channel
    noise is applied to every gate (base m/h/n, the slow-Na gate, and each extra channel's
    activation/inactivation), scaled by 1/N (N -> inf recovers the deterministic trace). `block` is an
    iterable of channel-blocker drug names (see ``blockers``)."""
    extra_chans = [_chan_to_dict(c) for c in mechanism.get("extra", ())]
    gna_base, gk_base, extra = _resolve_blockers(block, GNA, GK, extra_chans)
    slow_na = mechanism.get("slow_na", False)
    T = len(I)
    v = np.full(P, V0)
    am, bm, ah, bh, an, bn = fx.ab(v)
    m = am / (am + bm); h = ah / (ah + bh); nn = an / (an + bn)
    s = np.full(P, _slow_na_init(v[0])) if slow_na else None
    ex = fx.extra_init(v, extra)
    Vs = np.empty((P, T))
    for i in range(T):
        am, bm, ah, bh, an, bn = fx.ab(v)
        m = fx.gate_step(m, am, bm, dt, N, rng)
        h = fx.gate_step(h, ah, bh, dt, N, rng)
        nn = fx.gate_step(nn, an, bn, dt, N, rng)
        gna = gna_base * m**3 * h
        if slow_na:
            s = _slow_na_step(s, v, dt, N, rng)
            gna = gna * s
        cur = gna * (v - ENa) + gk_base * nn**4 * (v - EK) + GL * (v - EL)
        cur = cur + fx.extra_step(ex, v, dt, N, rng)
        v = np.clip(v + dt * (I[i] - cur), -95.0, 60.0)
        Vs[:, i] = v
    return Vs


def make_protocol(segments, dt=DT):
    """Build (I, obs_idx, test_start) for a protocol's segment list: the current I(t), the ~4 ms
    sub-sampled observation index set, and the spike-count test-window start."""
    I, test_start = _build_I(segments, dt=dt)
    obs_idx = np.arange(0, len(I), int(round(4.0 / dt)))
    return I, obs_idx, test_start


def observe_voltage(I, obs_idx, mechanism, N, rng, dt=DT, block=()):
    """Run the (hidden, true) stochastic cell once and return the noisy, sub-sampled voltage the agent
    sees: V(obs_idx) + Gaussian(0, SIG_OBS). This is the benchmark's observation."""
    V = run_particles(1, I, mechanism, N, rng, dt, block=block)[0]
    return V[obs_idx] + rng.normal(0.0, SIG_OBS, len(obs_idx))
