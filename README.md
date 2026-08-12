# NeuronBench

A partially observed **single-neuron electrophysiology benchmark** for data-efficient
mechanism discovery. A neuron is patched on the bench; it implements one of a small set of
biophysical mechanisms, but which one is hidden. From a **budget** of interventional
experiments — current-clamp protocols and channel blockers, each returning a noisy, partial
recording — an agent must uncover enough of the mechanism to **predict the outcome of
interventions it never ran**. That — counterfactual interventional forecasting — is what the
benchmark scores; the agent is never asked to *name* the mechanism, and identifying it is only a
means to a better forecast.

NeuronBench ships in **deterministic** and **stochastic** (Fox–Lu channel-noise) forms, a zoo
of Hodgkin–Huxley worlds, an intervention API under a hard budget, an evaluator, and a
**reference baseline LLM agent**.

**Paper:** NeuronBench is introduced in *Model Discovery Agent: LLM-assisted Bayesian experiment
design for data-efficient discovery of mechanistic world models* —
[arxiv.org/abs/2608.09696](https://arxiv.org/abs/2608.09696).

### Interactive demo

**[Patch](https://claude.ai/code/artifact/2848d02d-cdc1-4c1c-99fe-c0034e9714fb)** — a playable
front end for the deterministic benchmark: pick a mystery cell, spend a budget of experiments
(the 9-protocol menu, custom steps, and the four channel blockers), read off the test-window
spike count, work out the hidden mechanism, then forecast the held-out interventions and score
yourself. The whole thing runs in the browser from a faithful port of `worlds.py` — see
[`app/patch.html`](app/patch.html) (self-contained, validated against the Python model to the
spike).

---

## How it works

### The worlds

Each world is a plain Na⁺/K⁺/leak spiker **plus one hidden extra membrane current**. The agent is
**not** told the candidate set — or even that there are exactly two hypotheses; it proposes its own
mechanisms and is scored on what it *forecasts* (see
[Evaluation](#evaluation-criteria--fully-open-ended)), not on picking from a menu. Five of the six
worlds are deliberately **hard**: the extra channel is **silent under every textbook probe** (it fires
identically to a plain cell under standard steps and blockers) and is revealed **only by a single
non-textbook protocol**. The sixth is a recallable control.

| world | hidden mechanism | revealed by |
|---|---|---|
| `z_rebound` | low-threshold inward current, de-inactivated by hyperpolarisation → depolarisation block | a hyperpolarising conditioning pulse, then a depolarising test |
| `h_sag` | hyperpolarisation-activated inward current (Iₕ) → sag + post-inhibitory rebound | a hyperpolarising step, then release |
| `na_fatigue` | slow use-dependent Na⁺ inactivation → spike-count run-down | paired long pulses (the second is fatigued) |
| `ca_rebound` | fast low-threshold Ca²⁺ (T-type) → rebound **burst** on release | a hyperpolarising step, then release |
| `d_type` | D-type K⁺, de-inactivated by hyperpolarisation → delayed/suppressed firing | a hyperpolarising conditioning pulse, then a depolarising test |
| `textbook_M` | M-current (Kv7) → spike-frequency **adaptation** (control: recallable by name) | any sustained step |

For the five novel worlds the mechanism is described to the agent only as "an unidentified
membrane current" — so a language model cannot recall its signature and guess the right probe;
only an experiment-design method that *simulates candidate mechanisms* can find the
discriminating protocol. This is the point of the benchmark: **naming the mechanism is not
enough; you must design the experiment that exposes it.**

### The observation model

Running a protocol returns a **voltage trace**. In the **deterministic** benchmark the scored
observable is the number of action potentials in the protocol's test window. In the
**stochastic** benchmark the latent gating is a finite population of `N` channels (Fox–Lu
diffusion approximation), so the neuron is genuinely noisy and only **partially observed**: you
receive a noisy, sub-sampled voltage trace (σ = 2 mV, every ~4 ms). The likelihood
`p(trace | mechanism)` is then intractable — it must be estimated by simulation — which is what
makes the stochastic form a realistic test. As `N → ∞` the stochastic model recovers the
deterministic one.

The benchmark hands you the (noisy) voltage and the spike count. **How you summarise a trace
into features to build a likelihood is your method's choice** and is not part of this package.

### The interventions

Two intervention types, spent against a fixed **experiment budget**:

- **Current-clamp protocols** — a shared 9-protocol design pool (`neuronbench.protocols.POOL`):
  4 textbook steps + 5 non-standard conditioning / paired-pulse protocols. Exactly one of the
  non-standard protocols reveals each world's mechanism; the rest are decoys *for that world*.
  You may also design a custom `(amplitude, duration)` step.
- **Channel blockers** (`neuronbench.blockers`) — pharmacological `do(g=0)` interventions:
  **TTX** (block Na⁺), **TEA** (block K⁺), **Cd** (block Ca²⁺-type currents), **XE991** (block
  the M-current). Blockers act on *both* hypotheses, so on the novel worlds they are identity
  probes rather than by-themselves discriminating — the discriminating interventions are the
  current-clamp protocols.

**Budget rule.** The deterministic benchmark allows *each protocol once* (a budget of `b`
buys `b` distinct protocols). The stochastic benchmark relaxes this to a `(protocol, repeats)`
design: running a protocol `r` times costs `r` units and averages the channel noise down
(~1/√r) — repeating the informative protocol is the experimentalist's response to noise.

---

## Evaluation criteria — fully open-ended

The task is **counterfactual trajectory forecasting**, open-ended: the agent is given only an opaque
prior (a plain Na+K spiker that *may* also involve one novel current of unknown identity), the reference
plain model, the design pool, and a budget. It runs experiments, then **proposes its own hypotheses**
and **forecasts the cell's response to held-out interventions** — for each, a predicted spike count and
a predicted voltage trace. It is **never told the true mechanism, nor that there are exactly two
hypotheses** (`world.problem()` is the leak-free spec; the truth is private). Two metrics score the two
levels of the forecast:

1. **`spike_forecast_mse`** *(headline)* — floored MSE of the predicted held-out test-window spike
   counts vs. the true cell. Both LLM and model-based agents can produce it, so it is the comparable
   number in the plots. Any model form is allowed; a method that recovered the mechanism forecasts
   interventions it never observed, which a curve-fit to the observed data cannot.
2. **`feature_forecast_mse`** *(secondary; model-based deep-dives)* — standardised MSE between the
   benchmark's internal feature vector (spike counts **plus** sub-threshold shape: V-min, steady state,
   run-down, adaptation) of the agent's **predicted voltage trace** and the true cell's. It rewards
   trajectory *shape* that a scalar spike count discards. Features are the benchmark's *internal*
   scoring device — **never given to agents**; agents forecast raw traces. An agent that cannot predict
   a trajectory (an LLM) submits an empty trace and is penalised here, not excused.

```python
import neuronbench as nb

world = nb.load_world("ca_rebound", stochastic=True, n_channels=100, seed=0)
spec  = world.problem()   # opaque prior + reference model + protocols + test labels + budget rule (no truth)

# ... the agent runs experiments (each consumes budget), proposes hypotheses, forms p(m|D) ...
obs = world.run(world.discriminator(), reps=3)        # noisy, partial; costs 3

# forecast each held-out intervention: a spike count (all agents) and a voltage trace (model-based)
spike_mse = world.spike_forecast_mse({lab: predicted_count for lab, _ in world.test_protocols})  # headline
feat_mse  = world.feature_forecast_mse({lab: predicted_trace for lab, _ in world.test_protocols}) # secondary
# an LLM agent submits {lab: []} traces -> a (penalised) feature_forecast_mse, and competes on spike_mse
```

`world.simulate(mechanism, protocol, block=...)` is the **forward model** a solver uses to score
its *own* candidate hypotheses; it consumes no budget and never touches the true cell.
`world.run(...)` is the **oracle**: it runs the hidden true cell (with a hidden RNG, so repeats
are genuinely independent) and returns a partial observation.

### What an observation contains — raw trace + a provided reduction

`world.run(design)` returns an `Observation` carrying **both** the raw membrane-voltage trace and its
provided reduction, and the solver decides which to use:

```python
obs = world.run(world.discriminator())
obs.voltage       # the (sub-sampled) membrane-potential trace V(t): noisy if stochastic, noiseless if not
obs.obs_idx       # its sample indices;  obs.test_start = where the scored test window begins
obs.spike_count   # the PROVIDED reduction  spikes(obs)  — action-potential count in the test window
```

A solver may reduce the trace to `obs.spike_count` (what the LLM baselines use — they are shown only
`"- <protocol> -> <count> spikes"` lines), or model the raw `obs.voltage` directly (e.g. a particle-filter
or feature likelihood, useful for the sub-threshold *shape* worlds). **The LLM is only ever shown the
spike-count reduction, never the trace** — reducing vs. modelling the trace is the solver's choice, and the
benchmark scores the spike count either way.

So the loop an agent runs is, conceptually:

```python
state = init(world.problem())                       # opaque prior + protocols + budget (no truth)
for _ in range(budget):
    design = state.choose_design()                  # the agent may use an LLM internally here
    obs    = world.run(design)                      # -> Observation(voltage, spike_count = spikes(voltage))
    state  = state.update(design, obs)              # use obs.spike_count and/or obs.voltage
p_of_m, forecasts = state.finish()                  # posterior over the agent's OWN hypotheses + predictions
```

---

## The reference baseline

`neuronbench.agent` is a **pure-LLM baseline** — it chooses experiments and forecasts the
held-out interventions in-context, with no Bayesian inference or experiment design. It is the
number a discovery method must beat. The LLM is injected as a client with a single `.ask(...)`
method, so any model plugs in:

```bash
export OPENROUTER_API_KEY=...
python examples/run_baseline.py --world ca_rebound --stochastic --budget 8
```

---

## Design boundary

NeuronBench is a **benchmark, not a solver.** It defines *what is observable* (noisy, partial
voltage), *what you may do* (a budgeted intervention API), and *how you are scored* — and
nothing about *how* to infer the mechanism. It ships **no** likelihoods, particle filters,
synthetic likelihoods, SMC, or value-of-information; those are the solver's job and bundling
them would leak "the right way to score." The only bundled agent is the pure-LLM baseline.

```
neuronbench/
  worlds.py       # 6-world registry, Chan model, deterministic simulator, protocol pool
  _foxlu.py       # Fox–Lu channel-noise gate steppers
  stochastic.py   # stochastic (finite-N) forward model + noisy observation
  features.py     # spike-count observables (the scored quantity)
  protocols.py    # current-clamp intervention API + budget rule
  blockers.py     # channel-blocker interventions (TTX / TEA / Cd / XE991)
  evaluator.py    # held-out forecast MSE — spike counts (headline) + trace features (secondary)
  agent.py        # reference pure-LLM baseline (injectable client)
```

## Install

```bash
pip install neuronbench              # once released
pip install -e .                     # from a checkout
pip install neuronbench[llm]         # + the reference OpenRouter client
```

Pin it by commit in a downstream project for reproducibility:

```toml
[tool.uv.sources]
neuronbench = { git = "https://github.com/murphyk/neuronbench", rev = "<commit-sha>" }
```

## License

MIT.
