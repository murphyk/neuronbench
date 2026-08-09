"""NeuronBench: a partially observed single-neuron electrophysiology benchmark.

Public API
----------
    import neuronbench as nb
    world = nb.load_world("ca_rebound", stochastic=True, n_channels=100, seed=0)
    obs   = world.run(nb.protocols.discriminator("ca_rebound"))   # noisy, partial observation
    ...                                                           # an agent proposes a mechanism
    mse   = world.forecast_mse(predicted_test_counts)             # held-out interventional score

The benchmark exposes the *generative* forward model (``world.simulate``) so a solver can score its
own hypotheses, and a hidden *oracle* (``world.run``) that reports only noisy, partial observations of
the true cell under a budget. It ships no inference machinery. The only bundled agent is the pure-LLM
baseline (``neuronbench.agent``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import agent, blockers, evaluator, features, protocols, stochastic, worlds
from .worlds import WORLDS, Chan, world_models
from .stochastic import DT, SIG_OBS

__all__ = ["load_world", "World", "Observation", "list_worlds", "spikes",
           "worlds", "stochastic", "features", "protocols", "blockers", "evaluator", "agent", "Chan"]

__version__ = "0.1.0"

_SUB = 10          # deterministic voltage is returned sub-sampled by this factor (dt 0.01 -> ~0.1 ms)


def spikes(obs):
    """The provided reduction of a raw trace to the scored scalar: the test-window spike count of an
    ``Observation``. A solver may call this, or ignore it and model the raw ``obs.voltage`` directly."""
    return obs.spike_count


def list_worlds():
    """The six world names, in canonical order."""
    return ["z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M"]


@dataclass
class Observation:
    """The result of running one (or `reps`) experiment(s) --- the raw observable plus the provided
    reduction. `voltage` is the (sub-sampled) membrane-potential trace the recording actually yields:
    noisy in the stochastic benchmark, noiseless in the deterministic one. `spike_count` is the
    provided reduction ``spikes(voltage)`` in the scored test window --- the scalar the benchmark
    scores --- but a solver is free to use the raw `voltage` instead (e.g. a particle-filter or feature
    likelihood). `obs_idx` are the sample times of `voltage`; `test_start` indexes where the scored test
    window begins in `voltage`. `cost` is the budget consumed (= reps)."""
    protocol_label: str
    spike_count: float
    reps: int
    cost: int
    voltage: np.ndarray | None = None
    obs_idx: np.ndarray | None = None
    test_start: int | None = None


@dataclass
class World:
    """A single benchmark world. Carries its two candidate mechanisms (plain vs the novel
    alternative), the design pool, the discriminator and held-out test protocols, and an opaque text
    prior. The true mechanism is the novel alternative; a hidden RNG makes repeated observations
    genuinely stochastic (no free noise-free forks)."""
    name: str
    stochastic: bool = True
    n_channels: float = 100.0
    seed: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        if self.name not in WORLDS:
            raise KeyError(f"unknown world {self.name!r}; choose from {list_worlds()}")
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    # -- metadata the agent is allowed to see --
    @property
    def text_prior(self):
        """The description the agent is given of the unidentified mechanism (opaque for novel worlds,
        named for the recallable control)."""
        return WORLDS[self.name]["hint"]

    @property
    def is_control(self):
        """True for the recallable textbook_M control world."""
        return WORLDS[self.name]["textbook"]

    @property
    def reference_model(self):
        """The familiar reference mechanism the agent starts from: plain Na+K+leak
        (``{extra: [], slow_na: False}``). The task is open-ended -- the true cell may be this, or it may
        involve a novel current the agent must discover and propose itself; the truth is never exposed."""
        return {"extra": [], "slow_na": False}

    def problem(self):
        """The leak-free specification handed to the agent (no ground truth): an opaque text prior, the
        design pool + held-out test-protocol labels, the reference (plain) model, and the budget rule.
        The agent runs experiments (``run``), proposes its own hypotheses, and returns a posterior over
        them plus forecasts -- it is never told the true mechanism or that there are exactly two."""
        return {
            "text_prior": self.text_prior,
            "reference_model": self.reference_model,
            "protocols": [(lab, seg) for lab, seg in protocols.POOL],
            "test_protocol_labels": [lab for lab, _ in self.test_protocols],
            "budget_rule": ("each protocol once" if not self.stochastic
                            else "(protocol, repeats): running r times costs r and averages the noise"),
        }

    @property
    def mechanisms(self):
        """DEPRECATED (two-hypothesis framing / leaks the truth). Use ``reference_model`` + ``problem()``.
        Returns only the reference plain model now; the novel mechanism is hidden."""
        return {"Na+K (plain)": self.reference_model}

    @property
    def protocol_pool(self):
        """The shared design pool [(label, segments), ...] the agent may choose experiments from."""
        return list(protocols.POOL)

    @property
    def test_protocols(self):
        """The held-out interventional test protocols used for forecasting scoring."""
        return protocols.test_protocols(self.name)

    def discriminator(self):
        """The single protocol that reveals this world's novel mechanism."""
        return protocols.discriminator(self.name)

    # -- the hidden truth (for scoring; a solver should not read this) --
    @property
    def _truth_name(self):
        return WORLDS[self.name]["name"]

    @property
    def _truth_kwargs(self):
        return world_models(self.name)[self._truth_name]

    def simulate(self, mechanism, protocol, reps=1, rng=None, block=()):
        """Forward model for an arbitrary hypothesis `mechanism` ({extra:[Chan], slow_na:bool}) under
        a protocol (label, segments), optionally with channel blockers. Returns the (reps, T) voltage
        array (stochastic) or the spike count (deterministic). This is the generative model a solver
        uses to score its own hypotheses -- it does not consume budget and does not touch the true cell."""
        _, seg = protocol
        if self.stochastic:
            I, obs_idx, ts = stochastic.make_protocol(seg)
            rng = rng if rng is not None else np.random.default_rng()
            return stochastic.run_particles(reps, I, mechanism, self.n_channels, rng, block=block)
        return worlds.spikes(mechanism, seg, block=block)

    def run(self, protocol, reps=1, block=()):
        """Run the hidden TRUE cell under a protocol (label, segments) `reps` times, optionally applying
        channel blockers, and return a noisy, partial Observation. Consumes `reps` units of budget. Uses
        the world's hidden RNG, so repeated calls are genuinely independent noisy experiments."""
        lab, seg = protocol
        if self.stochastic:
            I, obs_idx, ts = stochastic.make_protocol(seg)
            V = stochastic.run_particles(reps, I, self._truth_kwargs, self.n_channels, self._rng, block=block)
            noisy = V[0][obs_idx] + self._rng.normal(0.0, SIG_OBS, len(obs_idx))
            cnt = float(features.spike_count(V, ts).mean())
            ts_obs = int(np.searchsorted(obs_idx, ts))          # test-window start in the sub-sampled trace
            return Observation(lab, cnt, reps, reps, voltage=noisy, obs_idx=obs_idx, test_start=ts_obs)
        # deterministic: the raw observable is the noiseless voltage trace; spike_count is its reduction
        I, ts = worlds.build_I(seg)
        cnt, V = worlds.simulate(I, ts, **self._truth_kwargs, block=block, trace=True)
        idx = np.arange(0, len(V), _SUB)                        # ~0.1 ms sampling of the trace
        return Observation(lab, float(cnt), reps, reps, voltage=V[idx].astype(float),
                           obs_idx=idx, test_start=int(np.searchsorted(idx, ts)))

    # -- evaluation (a single open-ended metric) --
    def forecast_mse(self, predicted_test_counts, floor=evaluator.MSE_FLOOR):
        """The benchmark's sole metric: floored MSE of the agent's predicted held-out test-window spike
        counts (a {label: count} dict over ``test_protocol_labels``) against the true cell. Fully
        open-ended -- the agent may reach it with any model form -- and it doubles as the model-recovery
        score, since the held-out test protocols are the discriminating regime: only a model that
        recovered the mechanism forecasts them well."""
        targets = evaluator.held_out_targets(self.name, stochastic=self.stochastic,
                                             N=self.n_channels, seed=self.seed)
        return evaluator.forecast_mse(predicted_test_counts, targets, floor=floor)


def load_world(name, stochastic=True, n_channels=100.0, seed=0):
    """Load a benchmark world by name (see ``list_worlds()``). ``stochastic=True`` uses the Fox--Lu
    channel-noise generative model at ``n_channels`` effective channels; ``False`` uses the
    deterministic spike-count model. ``seed`` fixes the hidden observation RNG."""
    return World(name=name, stochastic=stochastic, n_channels=float(n_channels), seed=seed)
