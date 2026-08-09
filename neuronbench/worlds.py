"""A registry of NOVEL single-neuron worlds -- the temporal-biology analogue of the six DiscoverPhysics
force-law worlds. Each world is a plain Na+K+leak spiker plus ONE novel membrane mechanism that is *silent
under every textbook probe* (so plain vs. novel fire identically to standard steps/blockers) and is revealed
only by a single NON-textbook protocol. This is the honest stress test: an LLM cannot recall the novel
channel's signature, so LLM-proposed experiment selection fails where numerical VoI succeeds; and a model-based
forecaster that has *expanded* its candidate set to include the novel mechanism forecasts its interventions,
where an LLM-in-context forecaster cannot.

Generic integrator: base Na (m^3 h, optional slow inactivation gate s), K (n^4), leak, plus a list of extra
channels each with an activation gate and optional inactivation gate (voltage half, slope, tau, power). A
negative activation slope makes a channel HYPERPOLARISATION-activated (I_h). Spike counts are taken in the
protocol's TEST window (after any conditioning pre-pulse).

    python -m neuronbench.worlds     # self-test: silence + revelation per world
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ENa, EK, EL = 50.0, -77.0, -54.4


def _ab(V):
    am = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-9); bm = 4 * np.exp(-(V + 65) / 18)
    ah = 0.07 * np.exp(-(V + 65) / 20); bh = 1 / (1 + np.exp(-(V + 35) / 10))
    an = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-9); bn = 0.125 * np.exp(-(V + 65) / 80)
    return am, bm, ah, bh, an, bn


def _clip(x, a, b):
    return a if x < a else b if x > b else x


@dataclass(frozen=True)
class Chan:
    """A novel channel: g*m^mpow*(h^hpow)*(v-E). Activation m_inf = sigmoid((V-mvh)/mk) with mk<0 =>
    hyperpolarisation-activated. Inactivation optional (hvh set): h_inf = 1/(1+exp((V-hvh)/hk))."""
    name: str; g: float; E: float
    mvh: float; mk: float; mtau: float; mpow: int = 1
    hvh: float = None; hk: float = 4.0; htau: float = 1e9; hpow: int = 1


def _minf(V, vh, k):
    return 1.0 / (1.0 + np.exp(-(V - vh) / k))


def _hinf(V, vh, k):
    return 1.0 / (1.0 + np.exp((V - vh) / k))


def build_I(segments, dt=0.01, base=20.0, tail=80.0):
    """Piecewise current I(t) from [(dur_ms, amp), ...], with a leading `base` ms and trailing `tail` ms at 0.
    Returns (I, test_start_idx) where test_start_idx is the sample index after the LAST segment whose amp<=0
    that precedes a positive test (i.e. spikes are counted from the final depolarising test onward)."""
    segs = [(base, 0.0)] + list(segments) + [(tail, 0.0)]
    chunks = [np.full(max(int(round(d / dt)), 1), a) for d, a in segs]
    I = np.concatenate(chunks)
    # count window = the LAST contiguous run of positive (depolarising) current, so spikes are counted in the
    # final test pulse (after any conditioning pre-pulse or paired-pulse gap).
    lens = [len(c) for c in chunks]; amps = [a for _, a in segs]
    start, cur, run_start = 0, 0, None
    for a, L in zip(amps, lens):
        run_start = cur if (a > 0 and run_start is None) else (None if a <= 0 else run_start)
        cur += L
        if a > 0:
            start = run_start
    return I, (start or 0)


def simulate(I, test_start, extra=(), gNa=120.0, gK=36.0, gL=0.3, slow_na=False, dt=0.01, V0=-65.0,
             trace=False):
    """Integrate base Na/K/leak (+ optional slow Na inactivation) plus `extra` channels; return the spike
    count in the test window [test_start:]. If ``trace=True``, also return the full voltage trace as
    ``(cnt, V)`` with ``V`` a length-``len(I)`` array (for qualitative visualisation)."""
    n = len(I); v = V0
    am, bm, ah, bh, an, bn = _ab(v)
    m = am / (am + bm); h = ah / (ah + bh); nn = an / (an + bn)
    s = 1.0                                            # slow Na inactivation (only if slow_na)
    xs = {c.name: _minf(v, c.mvh, c.mk) for c in extra}
    hs = {c.name: (_hinf(v, c.hvh, c.hk) if c.hvh is not None else 1.0) for c in extra}
    Vs = np.empty(n) if trace else None
    cnt = 0; pv = v
    for i in range(n):
        am, bm, ah, bh, an, bn = _ab(v)
        m += dt * (am * (1 - m) - bm * m); h += dt * (ah * (1 - h) - bh * h); nn += dt * (an * (1 - nn) - bn * nn)
        m = _clip(m, 0, 1); h = _clip(h, 0, 1); nn = _clip(nn, 0, 1)
        if slow_na:
            # slow, voltage-dependent Na inactivation: inactivates moderately during a depol pulse and
            # RECOVERS very slowly, so a single long step barely differs from plain but a paired pulse (short
            # gap => little recovery) leaves the test pulse strongly fatigued.
            s_inf = 1.0 / (1.0 + np.exp((v + 45) / 4.0))
            tau_s = 380.0 if v > -55.0 else 1500.0           # fast-ish inactivation, very slow recovery
            s += dt * (s_inf - s) / tau_s; s = _clip(s, 0, 1)
        cur = gNa * m**3 * h * (s if slow_na else 1.0) * (v - ENa) + gK * nn**4 * (v - EK) + gL * (v - EL)
        for c in extra:
            xs[c.name] += dt * (_minf(v, c.mvh, c.mk) - xs[c.name]) / c.mtau
            if c.hvh is not None:
                hs[c.name] += dt * (_hinf(v, c.hvh, c.hk) - hs[c.name]) / c.htau
            xs[c.name] = _clip(xs[c.name], 0, 1); hs[c.name] = _clip(hs[c.name], 0, 1)
            cur += c.g * xs[c.name] ** c.mpow * hs[c.name] ** c.hpow * (v - c.E)
        v += dt * (I[i] - cur); v = _clip(v, -95, 80)
        if trace:
            Vs[i] = v
        if i >= test_start and pv < 0 <= v:
            cnt += 1
        pv = v
    return (cnt, Vs) if trace else cnt


# ---------------------------------------------------------------------------------------------------------
# World registry. Each world: the novel channel/flag, the design POOL (textbook probes + the ONE non-textbook
# discriminator), and a disjoint held-out interventional TEST set. Protocols are (label, segments) with
# segments in (dur_ms, amp); a leading hyperpolarising segment is a conditioning pre-pulse.
# ---------------------------------------------------------------------------------------------------------
Z = Chan("Z", g=4.0, E=120.0, mvh=-57.0, mk=5.0, mtau=4.0, mpow=2, hvh=-88.0, hk=4.0, htau=130.0)
# Ih: hyperpol-activated (mk<0), ~silent at rest (mvh far below -65 so m_inf(-65)~0), strong when the cell is
# driven to ~-90 by a hyperpolarising step -> inward -> sag + post-inhibitory rebound.
IH = Chan("Ih", g=5.0, E=-30.0, mvh=-95.0, mk=-5.0, mtau=140.0, mpow=1)
# T-type Ca: hyperpol-de-inactivated inward (like Z) but FAST/transient -> a low-threshold rebound BURST on
# release from hyperpolarisation (extra spikes), vs Z's slow depolarisation block.
MT = Chan("T", g=3.2, E=120.0, mvh=-54.0, mk=6.0, mtau=2.0, mpow=2, hvh=-87.0, hk=4.0, htau=22.0)
# D-type K (I_D): OUTWARD, de-inactivated by hyperpolarisation; after a hyperpolarising pre-pulse it opens on
# the depolarising test and DELAYS/suppresses firing (fewer spikes) -- the outward analogue of Z.
ID = Chan("D", g=9.0, E=-77.0, mvh=-30.0, mk=10.0, mtau=3.0, mpow=1, hvh=-80.0, hk=5.0, htau=200.0)
# M-current (I_M): TEXTBOOK slow non-inactivating K -> spike-frequency adaptation on a long step. The LLM can
# recall this one, so it is the "names do all the work" control world.
MC = Chan("M", g=2.5, E=-77.0, mvh=-35.0, mk=10.0, mtau=60.0, mpow=1)

TEXTBOOK = [
    ("brief step (12 uA, 40 ms)",   [(40, 12.0)]),
    ("long step (10 uA, 300 ms)",   [(300, 10.0)]),
    ("strong step (18 uA, 120 ms)", [(120, 18.0)]),
    ("weak step (5 uA, 120 ms)",    [(120, 5.0)]),
]

# A SHARED menu of non-standard ("advanced") protocols added to every world's design pool. Exactly one reveals
# each world's novel channel; the rest are decoys FOR THAT WORLD (they probe the wrong voltage/timing regime).
# Because every world gets the same menu and the novel candidate is named opaquely, an LLM cannot know WHICH
# advanced protocol reveals the unidentified channel -- only numerical VoI, which simulates the candidates, can.
ADVANCED = [
    ("hyperpol conditioning + depol test (-30 uA/250 ms, then +12 uA)",  [(250, -30.0), (150, 12.0)]),
    ("hyperpol pre-pulse + weak test (-30 uA/250 ms, gap, +6 uA)",       [(250, -30.0), (120, 0.0), (60, 6.0)]),
    ("paired long pulses (12 uA/300 ms, 60 ms gap, 12 uA/300 ms)",       [(300, 12.0), (60, 0.0), (300, 12.0)]),
    ("depolarising conditioning + test (+15 uA/250 ms, then +12 uA)",    [(250, 15.0), (150, 12.0)]),
    ("brief hyperpol conditioning + test (-30 uA/40 ms, then +12 uA)",   [(40, -30.0), (150, 12.0)]),
]
POOL = TEXTBOOK + ADVANCED       # the shared design pool used for every world

# Each world: `alt` is the non-plain candidate (extra channels + slow_na flag), `name` its label, `hint` the
# description the LLM is given about it, `textbook` marks the recallable control. `discriminator` is the one
# revealing protocol; `test` is held out. For the NOVEL worlds the candidate is named and described OPAQUELY
# ("an unidentified current") so the LLM cannot recall the channel type from the name and pick its probe --
# only numerical VoI, which simulates the candidates, can find the discriminator. The textbook world names its
# channel so the LLM can recall it.
_NOVEL_NAME = "Na+K + unidentified current"
_NOVEL_HINT = ("an additional membrane current of unknown identity, voltage-dependence, and kinetics -- not a "
               "named textbook channel, so how to probe it is not known a priori")
WORLDS = {
    "z_rebound": dict(
        desc="low-threshold inward Z, de-inactivated by hyperpolarisation -> depolarisation block",
        alt=dict(extra=[Z], slow_na=False), name=_NOVEL_NAME, hint=_NOVEL_HINT, textbook=False,
        discriminator=("conditioning pulse (-25 mV/200 ms, then +8 uA test)", [(200, -25.0), (120, 8.0)]),
        test=[("cond -20/+10", [(200, -20.0), (120, 10.0)]),
              ("cond -30/+8", [(200, -30.0), (120, 8.0)]),
              ("cond -35/+12", [(200, -35.0), (120, 12.0)]),
              ("cond -22/+9", [(180, -22.0), (140, 9.0)]),
              ("depol 14 uA/200 ms", [(200, 14.0)]),
              ("strong step 16 uA/150 ms", [(150, 16.0)])]),
    "h_sag": dict(
        desc="hyperpolarisation-activated inward Ih -> voltage sag and post-inhibitory rebound",
        alt=dict(extra=[IH], slow_na=False), name=_NOVEL_NAME, hint=_NOVEL_HINT, textbook=False,
        discriminator=("hyperpol step then release (-30 uA/250 ms -> rebound)", [(250, -30.0), (120, 0.0), (60, 6.0)]),
        test=[("hyperpol -25 then rebound", [(250, -25.0), (120, 0.0), (60, 6.0)]),
              ("hyperpol -35 then rebound", [(300, -35.0), (120, 0.0), (60, 5.0)]),
              ("hyperpol -30 then rebound", [(280, -30.0), (120, 0.0), (60, 6.0)]),
              ("hyperpol -40 then rebound", [(350, -40.0), (100, 0.0), (60, 7.0)]),
              ("long step (10 uA, 300 ms)", [(300, 10.0)]),
              ("brief step (12 uA, 40 ms)", [(40, 12.0)])]),
    "na_fatigue": dict(
        desc="slow Na inactivation -> use-dependent spike-count run-down, revealed by paired long pulses",
        alt=dict(extra=[], slow_na=True), name=_NOVEL_NAME, hint=_NOVEL_HINT, textbook=False,
        discriminator=("paired pulses (12 uA/300 ms, 60 ms gap, 12 uA/300 ms test)", [(300, 12.0), (60, 0.0), (300, 12.0)]),
        test=[("paired 10 uA/300+300", [(300, 10.0), (60, 0.0), (300, 10.0)]),
              ("paired 14 uA/250+250", [(250, 14.0), (50, 0.0), (250, 14.0)]),
              ("paired 12 uA/350+350", [(350, 12.0), (40, 0.0), (350, 12.0)]),
              ("paired 8 uA/300+300", [(300, 8.0), (60, 0.0), (300, 8.0)]),
              ("single long 12 uA/300 ms", [(300, 12.0)]),
              ("single long 10 uA/300 ms", [(300, 10.0)])]),
    "ca_rebound": dict(
        desc="fast low-threshold Ca (T-type), de-inactivated by hyperpolarisation -> rebound BURST on release",
        alt=dict(extra=[MT], slow_na=False), name=_NOVEL_NAME, hint=_NOVEL_HINT, textbook=False,
        discriminator=("hyperpol then release (-30 uA/200 ms -> rebound burst)", [(200, -30.0), (150, 0.0)]),
        test=[("hyperpol -25 then release", [(200, -25.0), (150, 0.0)]),
              ("hyperpol -35 then release", [(250, -35.0), (150, 0.0)]),
              ("hyperpol -30 then release", [(220, -30.0), (150, 0.0)]),
              ("hyperpol -40 then release", [(250, -40.0), (150, 0.0)]),
              ("brief step (12 uA, 40 ms)", [(40, 12.0)]),
              ("long step (10 uA, 300 ms)", [(300, 10.0)])]),
    "d_type": dict(
        desc="D-type K, de-inactivated by hyperpolarisation -> delayed/suppressed firing on the test",
        alt=dict(extra=[ID], slow_na=False), name=_NOVEL_NAME, hint=_NOVEL_HINT, textbook=False,
        discriminator=("conditioning pulse (-30 mV/250 ms, then +12 uA test)", [(250, -30.0), (150, 12.0)]),
        test=[("cond -25/+12", [(250, -25.0), (150, 12.0)]),
              ("cond -35/+10", [(300, -35.0), (150, 10.0)]),
              ("cond -30/+14", [(250, -30.0), (150, 14.0)]),
              ("cond -28/+11", [(280, -28.0), (150, 11.0)]),
              ("long step (10 uA, 300 ms)", [(300, 10.0)]),
              ("brief step (12 uA, 40 ms)", [(40, 12.0)])]),
    "textbook_M": dict(
        desc="TEXTBOOK M-current adaptation (Na+K+M): recallable -- names do all the work",
        alt=dict(extra=[MC], slow_na=False), name="Na+K+M (M-current adaptation)",
        hint="an M-type slow non-inactivating potassium current (I_M) causing spike-frequency ADAPTATION",
        textbook=True,
        discriminator=("sustained step (9 uA, 400 ms)", [(400, 9.0)]),
        test=[("long 8 uA/300 ms", [(300, 8.0)]),
              ("long 12 uA/250 ms", [(250, 12.0)]),
              ("long 10 uA/400 ms", [(400, 10.0)]),
              ("long 6 uA/300 ms", [(300, 6.0)]),
              ("brief 12 uA/40 ms", [(40, 12.0)]),
              ("strong 16 uA/200 ms", [(200, 16.0)])]),
}


def world_models(w):
    """Candidate mechanisms for world `w`: plain Na+K+leak vs. the world's alternative. Returns
    {name: kwargs-for-simulate}."""
    spec = WORLDS[w]
    return {"Na+K (plain)": dict(extra=[], slow_na=False), spec["name"]: dict(**spec["alt"])}


def spikes(kwargs, segments, dt=0.01):
    I, ts = build_I(segments, dt=dt)
    return simulate(I, ts, **kwargs, dt=dt)


def _selftest():
    labels = [lab for lab, _ in POOL]
    for w, spec in WORLDS.items():
        models = world_models(w); alt = spec["name"]
        tag = "TEXTBOOK (recallable)" if spec["textbook"] else "NOVEL"
        print(f"\n=== {w} [{tag}]: {spec['desc']} ===")
        print(f"{'protocol':56s} plain  alt")
        disc = []
        for lab, seg in POOL:
            p = spikes(models["Na+K (plain)"], seg); nv = spikes(models[alt], seg)
            d = abs(p - nv) >= 2; disc.append(lab) if d else None
            std = lab in {t[0] for t in TEXTBOOK}
            flag = "  <-- discriminates" if d else ""
            if d and std and not spec["textbook"]:
                flag = "  !! LEAK (textbook probe reveals the novel channel)"
            print(f"  {lab:54s} {p:4d}  {nv:4d}{flag}")
        print(f"  --> discriminating protocols: {disc}")


if __name__ == "__main__":
    _selftest()
