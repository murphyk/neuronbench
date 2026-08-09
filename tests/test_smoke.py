"""Smoke tests: the package imports, the worlds behave, scoring works, and the baseline agent runs
against a mock LLM client (no network)."""
import numpy as np

import neuronbench as nb


def test_six_worlds():
    assert nb.list_worlds() == ["z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M"]


def test_deterministic_revelation_and_silence():
    """Every novel world's discriminator separates plain from alt; a textbook step does not."""
    for name in nb.list_worlds():
        w = nb.load_world(name, stochastic=False)
        plain = w.mechanisms["Na+K (plain)"]
        alt = w.mechanisms[w._truth_name]
        disc = w.discriminator()
        assert abs(w.simulate(plain, disc) - w.simulate(alt, disc)) >= 2, name
        if not w.is_control:  # a novel channel is silent under the plain brief step
            brief = nb.protocols.by_label("brief step (12 uA, 40 ms)")
            assert w.simulate(plain, brief) == w.simulate(alt, brief), name


def test_perfect_forecast_is_zero():
    w = nb.load_world("ca_rebound", stochastic=False, seed=0)
    targets = nb.evaluator.held_out_targets("ca_rebound", stochastic=False, seed=0)
    assert w.forecast_mse(targets) == 0.0
    assert w.selection_correct(w._truth_name) == 1
    assert w.selection_correct("Na+K (plain)") == 0


def test_stochastic_observation_shape():
    w = nb.load_world("na_fatigue", stochastic=True, n_channels=100, seed=1)
    o = w.run(w.discriminator(), reps=3)
    assert o.cost == 3
    assert o.features.shape == (6,)
    assert o.voltage.ndim == 1


class _MockClient:
    """Returns a fixed experiment pick and a constant forecast, so the agent path runs offline."""
    def ask(self, system, prompt, temperature=0.2, max_tokens=800, response_format=None):
        if "experiment" in prompt:
            lab = prompt.split("Available experiments")[1].split("- ")[1].splitlines()[0].strip()
            return '{"experiment": "%s"}' % lab
        n = prompt.count("\n  ")  # rough count of test protocols listed
        return '{"counts": [%s]}' % ", ".join(["2"] * max(n, 6))


def test_baseline_agent_runs_offline():
    w = nb.load_world("d_type", stochastic=False, seed=0)
    result = nb.agent.run_baseline(w, _MockClient(), budget=4)
    assert len(result["observed"]) == 4
    assert set(result["predictions"]) == {lab for lab, _ in w.test_protocols}
    assert result["forecast_mse"] >= 0.0
