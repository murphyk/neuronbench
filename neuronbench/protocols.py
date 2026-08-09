"""The intervention API: the experiments an agent may run, and the budget rule.

A *protocol* is ``(label, segments)`` where ``segments`` is a list of ``(duration_ms, amplitude_uA)``
current-clamp steps; a leading negative-amplitude segment is a hyperpolarising conditioning pre-pulse.
Every world shares the same design pool (``POOL`` = ``TEXTBOOK`` standard steps + ``ADVANCED``
non-standard conditioning/paired-pulse protocols); exactly one ADVANCED protocol reveals each world's
novel channel, the rest are decoys for that world.

Budget rule. The deterministic benchmark allows "each protocol once": a run of `budget` experiments
must choose `budget` *distinct* protocol labels. The stochastic benchmark relaxes this to a
(protocol, repeats) design where running a protocol r times costs r units of budget -- averaging r
noisy trials is the experimentalist's response to channel noise. Both rules are advisory here: the
harness reports cost; enforcing the budget is the runner's job.
"""
from __future__ import annotations

from .worlds import TEXTBOOK, ADVANCED, POOL, WORLDS

__all__ = ["TEXTBOOK", "ADVANCED", "POOL", "labels", "by_label", "discriminator", "test_protocols"]


def labels(pool=POOL):
    """The protocol labels in a pool (default: the shared design POOL)."""
    return [lab for lab, _ in pool]


def by_label(label, world=None):
    """Look up a protocol's segments by label, searching the shared POOL and, if `world` is given,
    that world's discriminator + held-out test protocols. Returns (label, segments) or raises."""
    tables = list(POOL)
    if world is not None:
        spec = WORLDS[world]
        tables = tables + [spec["discriminator"]] + list(spec["test"])
    for lab, seg in tables:
        if lab == label:
            return lab, seg
    raise KeyError(f"unknown protocol label {label!r}" + (f" for world {world!r}" if world else ""))


def discriminator(world):
    """The single revealing protocol (label, segments) for a world."""
    return WORLDS[world]["discriminator"]


def test_protocols(world):
    """The held-out interventional test protocols [(label, segments), ...] for a world."""
    return list(WORLDS[world]["test"])
