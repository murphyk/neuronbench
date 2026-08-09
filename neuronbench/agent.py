"""The reference baseline: a pure-LLM agent.

This is the benchmark's *baseline*, not a solver -- it does no Bayesian inference, no value-of-information,
no simulation of hypotheses. It simply asks a language model, in context, (a) which experiment to run next
and (b) what the held-out interventions will do, given the experiments run so far. It is what a strong
frontier model achieves on its own, and the number a discovery method must beat.

The LLM is injected as a client with a single method::

    client.ask(system: str, prompt: str, temperature: float = 0.2,
               max_tokens: int = 800, response_format: dict | None = None) -> str

so you can plug in any model. ``openrouter_client()`` builds a reference OpenAI-compatible client
against OpenRouter (needs the optional ``[llm]`` extra and ``OPENROUTER_API_KEY``); it carries no
dependency on any discovery framework.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

import numpy as np


class LLMClient(Protocol):
    """The minimal LLM interface the baseline agent needs."""
    def ask(self, system: str, prompt: str, temperature: float = 0.2,
            max_tokens: int = 800, response_format: dict | None = None) -> str: ...


def _extract_json(txt):
    return json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))


def choose_experiment(world, observed, remaining, client):
    """Ask the LLM which ONE of the `remaining` protocols to run next, given the `observed`
    [(label, count), ...] history. Returns a (label, segments) protocol; falls back to the first
    remaining protocol if the reply cannot be parsed.

    `world` is a neuronbench.World; `remaining` is a list of (label, segments)."""
    labels = [lab for lab, _ in remaining]
    obs_txt = ("\n".join(f"  - {l} -> {c} spikes" for l, c in observed) if observed else "  (none yet)")
    prompt = ("A single-compartment neuron under current clamp; we count action potentials per protocol. "
              "It is a plain Na+K spiker, possibly plus one additional membrane current you must identify: "
              f"{world.text_prior}.\n"
              "Experiments run so far and their observed spike counts:\n" + obs_txt +
              "\n\nAvailable experiments (choose one; each may be run once):\n"
              + "\n".join(f"  - {lab}" for lab in labels)
              + "\n\nWhich ONE experiment best distinguishes the candidate mechanisms next? "
                'Return exactly {"experiment": "<one label copied verbatim>"}.')
    try:
        txt = client.ask("You are an electrophysiologist choosing the next experiment. Reply with ONLY a JSON object.",
                         prompt, temperature=0.2, max_tokens=700, response_format={"type": "json_object"})
        choice = _extract_json(txt)["experiment"].strip()
        pick = next((l for l in labels if l == choice), None) or next((l for l in labels if choice[:14] in l), None)
        pick = pick or labels[0]
    except Exception:
        pick = labels[0]
    return next(p for p in remaining if p[0] == pick)


def forecast_counts(world, observed, client):
    """Ask the LLM to predict the held-out test-protocol spike counts from the `observed`
    [(label, count), ...] history. Returns a {test_label: predicted_count} dict suitable for
    ``world.forecast_mse``. On a parse failure, predicts each test count as the mean observed count
    (or 0 if nothing observed)."""
    test = world.test_protocols
    obs_txt = ("\n".join(f"  - {l} -> {c} spikes" for l, c in observed) if observed else "  (none yet)")
    prompt = ("A single-compartment neuron under current clamp; we count action potentials per protocol. "
              "It is a plain Na+K spiker, possibly plus one additional membrane current of unknown identity: "
              f"{world.text_prior}. "
              "Experiments run and observed spike counts:\n" + obs_txt +
              "\n\nPredict the spike count for each held-out protocol:\n"
              + "\n".join(f"  {i+1}. {lab}" for i, (lab, _) in enumerate(test))
              + '\n\nReturn exactly {"counts": [n1, ...]} with one number per protocol, in order.')
    fallback = float(np.mean([c for _, c in observed])) if observed else 0.0
    try:
        txt = client.ask("You are an electrophysiologist forecasting a neuron's response. Reply with ONLY a JSON object.",
                         prompt, temperature=0.2, max_tokens=800, response_format={"type": "json_object"})
        arr = [float(x) for x in _extract_json(txt)["counts"]]
        return {lab: (arr[i] if i < len(arr) else fallback) for i, (lab, _) in enumerate(test)}
    except Exception:
        return {lab: fallback for lab, _ in test}


def run_baseline(world, client, budget=8, seed=0):
    """Run one full baseline episode on `world`: repeatedly let the LLM choose an experiment (from the
    shared pool, each protocol once) and run it on the true cell, up to `budget` experiments; then let
    the LLM forecast the held-out interventions. Returns a dict with the observed trajectory, the
    predicted test counts, and the held-out forecast MSE."""
    remaining = list(world.protocol_pool)
    observed = []
    for _ in range(min(budget, len(remaining))):
        proto = choose_experiment(world, observed, remaining, client)
        obs = world.run(proto)
        observed.append((proto[0], round(obs.spike_count)))
        remaining = [p for p in remaining if p[0] != proto[0]]
    predictions = forecast_counts(world, observed, client)
    return dict(world=world.name, observed=observed, predictions=predictions,
                forecast_mse=world.forecast_mse(predictions))


def openrouter_client(model="anthropic/claude-opus-4.7", api_key=None, base_url="https://openrouter.ai/api/v1"):
    """A reference OpenAI-compatible LLM client against OpenRouter. Requires the optional ``[llm]``
    extra (``pip install neuronbench[llm]``) and an API key (arg or ``OPENROUTER_API_KEY`` env var).
    No dependency on any discovery framework."""
    import os
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY or pass api_key=")
    _cli = OpenAI(api_key=key, base_url=base_url)

    class _Client:
        def ask(self, system, prompt, temperature=0.2, max_tokens=800, response_format=None):
            r = _cli.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                response_format=response_format,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            return r.choices[0].message.content

    return _Client()
