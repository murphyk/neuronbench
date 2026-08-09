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

from . import agent, evaluator, features, protocols, stochastic, worlds
from .worlds import WORLDS, Chan, world_models
from .stochastic import DT, SIG_OBS

__all__ = ["load_world", "World", "Observation", "list_worlds",
           "worlds", "stochastic", "features", "protocols", "evaluator", "agent", "Chan"]

__version__ = "0.1.0"


def list_worlds():
    """The six world names, in canonical order."""
    return ["z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M"]


@dataclass
class Observation:
    """The result of running one (or `reps`) experiment(s). For stochastic worlds `voltage` is the
    noisy sub-sampled trace and `features` is the mean feature vector s(y); for deterministic worlds
    those are None and only `spike_count` is populated. `cost` is the budget consumed (= reps)."""
    protocol_label: str
    spike_count: float
    reps: int
    cost: int
    voltage: np.ndarray | None = None
    obs_idx: np.ndarray | None = None
    features: np.ndarray | None = None


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
    def mechanisms(self):
        """The candidate mechanisms {name: kwargs}: plain Na+K+leak vs the world's alternative."""
        return world_models(self.name)

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

    def simulate(self, mechanism, protocol, reps=1, rng=None):
        """Forward model for an arbitrary hypothesis `mechanism` ({extra:[Chan], slow_na:bool}) under
        a protocol (label, segments). Returns the (reps, T) voltage array (stochastic) or the spike
        count (deterministic). This is the generative model a solver uses to score its own hypotheses
        -- it does not consume budget and does not touch the true cell."""
        _, seg = protocol
        if self.stochastic:
            I, obs_idx, ts = stochastic.make_protocol(seg)
            rng = rng if rng is not None else np.random.default_rng()
            return stochastic.run_particles(reps, I, mechanism, self.n_channels, rng)
        return worlds.spikes(mechanism, seg)

    def run(self, protocol, reps=1):
        """Run the hidden TRUE cell under a protocol (label, segments) `reps` times and return a noisy,
        partial Observation. Consumes `reps` units of budget. Uses the world's hidden RNG, so repeated
        calls are genuinely independent noisy experiments."""
        lab, seg = protocol
        if self.stochastic:
            I, obs_idx, ts = stochastic.make_protocol(seg)
            V = stochastic.run_particles(reps, I, self._truth_kwargs, self.n_channels, self._rng)
            noisy = V[0][obs_idx] + self._rng.normal(0.0, SIG_OBS, len(obs_idx))
            feat = features.feature_vector(V, ts).mean(axis=0)
            cnt = float(features.spike_count(V, ts).mean())
            return Observation(lab, cnt, reps, reps, voltage=noisy, obs_idx=obs_idx, features=feat)
        cnt = float(worlds.spikes(self._truth_kwargs, seg))
        return Observation(lab, cnt, reps, reps)

    # -- scoring --
    def forecast_mse(self, predicted_test_counts, floor=evaluator.MSE_FLOOR):
        """Floored MSE of the agent's predicted held-out test-window spike counts (a {label: count}
        dict) against the true cell."""
        targets = evaluator.held_out_targets(self.name, stochastic=self.stochastic,
                                             N=self.n_channels, seed=self.seed)
        return evaluator.forecast_mse(predicted_test_counts, targets, floor=floor)

    def selection_correct(self, chosen_name):
        """1 if `chosen_name` is the true mechanism, else 0."""
        return evaluator.selection_correct(chosen_name, self.name)

    def selection_brier(self, prob_true):
        """Two-hypothesis Brier score for the posterior mass placed on the true mechanism."""
        return evaluator.selection_brier(prob_true)


def load_world(name, stochastic=True, n_channels=100.0, seed=0):
    """Load a benchmark world by name (see ``list_worlds()``). ``stochastic=True`` uses the Fox--Lu
    channel-noise generative model at ``n_channels`` effective channels; ``False`` uses the
    deterministic spike-count model. ``seed`` fixes the hidden observation RNG."""
    return World(name=name, stochastic=stochastic, n_channels=float(n_channels), seed=seed)
