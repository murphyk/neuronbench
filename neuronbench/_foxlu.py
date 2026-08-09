"""Fox--Lu diffusion approximation of ion-channel gating noise.

A finite population of N stochastic channels, in the diffusion limit, makes each Hodgkin--Huxley
gating variable obey a stochastic differential equation

    dx = (a(V)(1-x) - b(V)x) dt  +  sqrt( (a(V)(1-x) + b(V)x) / N ) dW,

i.e. deterministic HH drift plus channel-count-scaled Brownian noise (N -> inf recovers the
deterministic gate). This module holds the noiseless rate functions and the Euler--Maruyama gate
steppers. It is the *generative* core of the stochastic worlds; it contains no inference/likelihood
code (that is the solver's job). numpy only.
"""
from __future__ import annotations

import numpy as np

# HH constants (textbook squid axon), ms / mV / (uA/cm^2) scale, matching worlds.py.
ENa, EK, EL = 50.0, -77.0, -54.4
GNA, GK, GL = 120.0, 36.0, 0.3


def ab(V):
    """Vectorised HH rate functions am, bm, ah, bh, an, bn (arrays in, arrays out)."""
    am = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-9)
    bm = 4.0 * np.exp(-(V + 65) / 18)
    ah = 0.07 * np.exp(-(V + 65) / 20)
    bh = 1.0 / (1 + np.exp(-(V + 35) / 10))
    an = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-9)
    bn = 0.125 * np.exp(-(V + 65) / 80)
    return am, bm, ah, bh, an, bn


def minf(V, vh, k):
    """Activation steady state; k<0 makes the channel hyperpolarisation-activated."""
    return 1.0 / (1.0 + np.exp(-(V - vh) / k))


def hinf(V, vh, k):
    """Inactivation steady state."""
    return 1.0 / (1.0 + np.exp((V - vh) / k))


def gate_step(x, a, b, dt, N, rng):
    """One Euler--Maruyama step of a gating variable with Fox--Lu channel noise:
    dx = (a(1-x) - bx) dt + sqrt((a(1-x) + bx)/N) dW. N = effective channel count."""
    drift = a * (1 - x) - b * x
    diff = np.sqrt(np.maximum((a * (1 - x) + b * x) / N, 0.0) * dt)
    x = x + dt * drift + diff * rng.standard_normal(x.shape)
    return np.clip(x, 0.0, 1.0)


def tau_gate_step(x, xinf, tau, dt, N, rng):
    """Fox--Lu step for a gate parameterised by (xinf, tau): rates a = xinf/tau, b = (1-xinf)/tau."""
    drift = (xinf - x) / tau
    diff = np.sqrt(np.maximum((xinf * (1 - x) + (1 - xinf) * x) / tau / N, 0.0) * dt)
    return np.clip(x + dt * drift + diff * rng.standard_normal(x.shape), 0.0, 1.0)


def extra_init(v, extra):
    """Init activation x (and optional inactivation h) for each extra channel (dict form)."""
    return [{"c": c, "x": minf(v, c["mvh"], c["mk"]).copy(),
             "h": (hinf(v, c["hvh"], c.get("hk", 4.0)).copy() if c.get("hvh") is not None else None)}
            for c in extra]


def extra_step(ex, v, dt, N, rng):
    """Advance the extra channels (activation + optional inactivation, both with channel noise); return
    their total current contribution."""
    cur = 0.0
    for e in ex:
        c = e["c"]
        e["x"] = tau_gate_step(e["x"], minf(v, c["mvh"], c["mk"]), c["mtau"], dt, N, rng)
        gh = 1.0
        if e["h"] is not None:
            e["h"] = tau_gate_step(e["h"], hinf(v, c["hvh"], c.get("hk", 4.0)), c["htau"], dt, N, rng)
            gh = e["h"] ** c.get("hpow", 1)
        cur = cur + c["g"] * e["x"] ** c["mpow"] * gh * (v - c["E"])
    return cur
