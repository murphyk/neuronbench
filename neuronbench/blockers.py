"""Pharmacological channel blockers: the second intervention type, alongside current-clamp protocols.

A blocker sets a channel's conductance to zero (``do(g=0)``) -- the classic causal intervention in
electrophysiology. Four are supported:

    TTX    blocks the base spike Na+ current           (g_Na -> 0)
    TEA    blocks the base delayed-rectifier K+        (g_K  -> 0)
    Cd     blocks Ca2+-type currents                   (any extra channel with E >= +100 mV, e.g. T-type)
    XE991  blocks the M-current (Kv7)                  (the extra channel named "M")

Blockers act on *both* candidate mechanisms (plain and novel), so on the six worlds they are not by
themselves discriminating -- TTX silences every spike regardless of the hidden channel. They matter as
identity probes (e.g. TTX-insensitivity flags a Ca2+ spike) and are provided for completeness; the
mechanism-discriminating interventions are the current-clamp protocols (see ``protocols``).
"""
from __future__ import annotations

BLOCKERS = {
    "TTX":   "block Na+ (base spike current)",
    "TEA":   "block K+ (base delayed rectifier)",
    "Cd":    "block Ca2+-type currents",
    "XE991": "block the M-current (Kv7)",
}

CA_E_THRESHOLD = 100.0   # an extra channel with reversal potential >= this (mV) is treated as Ca2+-type


def blocks_base_na(drug):
    return drug == "TTX"


def blocks_base_k(drug):
    return drug == "TEA"


def blocks_channel(drug, name, E):
    """Whether `drug` blocks an extra channel with the given name and reversal potential E (mV)."""
    if drug == "Cd":
        return E >= CA_E_THRESHOLD
    if drug == "XE991":
        return name == "M"
    return False


def resolve(block, base_gna, base_gk, extra):
    """Apply a set of blocker drug names to (base Na conductance, base K conductance, extra-channel list).
    Returns (gna, gk, kept_extra) with blocked conductances zeroed / channels dropped. `extra` items may be
    worlds.Chan dataclasses (``.name``/``.E``) or plain dicts (``["name"]``/``["E"]``)."""
    block = set(block or ())
    gna = 0.0 if any(blocks_base_na(d) for d in block) else base_gna
    gk = 0.0 if any(blocks_base_k(d) for d in block) else base_gk

    def _name_E(c):
        return (c["name"], c["E"]) if isinstance(c, dict) else (c.name, c.E)

    kept = [c for c in extra if not any(blocks_channel(d, *_name_E(c)) for d in block)]
    return gna, gk, kept
