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
        plain = w.reference_model
        alt = w._truth_kwargs            # tests may peek at the hidden truth; agents may not
        disc = w.discriminator()
        assert abs(w.simulate(plain, disc) - w.simulate(alt, disc)) >= 2, name
        if not w.is_control:  # a novel channel is silent under the plain brief step
            brief = nb.protocols.by_label("brief step (12 uA, 40 ms)")
            assert w.simulate(plain, brief) == w.simulate(alt, brief), name


def test_perfect_forecast_is_zero():
    w = nb.load_world("ca_rebound", stochastic=False, seed=0)
    # a perfect forecaster (submitting the true held-out counts) scores 0 on the single metric
    assert w.forecast_mse(nb.evaluator.held_out_targets("ca_rebound", stochastic=False, seed=0)) == 0.0


def test_problem_is_leak_free():
    w = nb.load_world("h_sag", stochastic=False)
    prob = w.problem()
    assert set(prob) >= {"text_prior", "reference_model", "protocols", "test_protocol_labels", "budget_rule"}
    assert prob["reference_model"] == {"extra": [], "slow_na": False}   # only the plain reference is given
    assert w.mechanisms == {"Na+K (plain)": w.reference_model}          # the novel mechanism is not exposed


def test_stochastic_observation_shape():
    w = nb.load_world("na_fatigue", stochastic=True, n_channels=100, seed=1)
    o = w.run(w.discriminator(), reps=3)
    assert o.cost == 3
    assert o.spike_count >= 0
    assert o.voltage.ndim == 1


def test_run_returns_trace_and_reduction():
    """run() always yields the raw voltage trace + its spike-count reduction, deterministic or stochastic;
    spikes(obs) recovers the scored scalar and the count matches counting crossings in the trace's window."""
    for stoch in (False, True):
        w = nb.load_world("ca_rebound", stochastic=stoch, seed=0)
        o = w.run(w.discriminator())
        assert o.voltage is not None and o.voltage.ndim == 1 and o.test_start is not None
        assert nb.spikes(o) == o.spike_count
        if not stoch:   # deterministic trace: reduction is exact
            V = o.voltage[None, :]
            assert nb.features.spike_count(V, o.test_start)[0] == o.spike_count


def test_ttx_silences_spikes():
    """TTX (block base Na) abolishes spiking for every world under a strong step."""
    strong = nb.protocols.by_label("strong step (18 uA, 120 ms)")
    for name in nb.list_worlds():
        w = nb.load_world(name, stochastic=False)
        truth = w._truth_kwargs
        assert w.simulate(truth, strong) > 0, name
        assert w.simulate(truth, strong, block=["TTX"]) == 0, name


def test_xe991_blocks_only_m_current():
    """XE991 blocks the M-current, so it changes textbook_M but leaves the Na-only fast worlds' base
    spiking unaffected on a plain step (na_fatigue's alt has no extra channel)."""
    strong = nb.protocols.by_label("strong step (18 uA, 120 ms)")
    wm = nb.load_world("textbook_M", stochastic=False)
    m = wm._truth_kwargs
    assert wm.simulate(m, strong, block=["XE991"]) >= wm.simulate(m, strong)  # M-block disinhibits firing


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
