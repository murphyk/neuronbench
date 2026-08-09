# NeuronBench

A partially observed **single-neuron electrophysiology benchmark** for data-efficient
mechanism discovery. A neuron is patched on the bench; it implements one of a small set of
biophysical mechanisms, but which one is hidden. From a **budget** of interventional
experiments — current-clamp protocols and channel blockers, each returning a noisy, partial
recording — an agent must (1) identify the mechanism and (2) **predict the outcome of
interventions it never ran**. That second part, counterfactual interventional forecasting, is
what the benchmark ultimately scores.

NeuronBench ships in **deterministic** and **stochastic** (Fox–Lu channel-noise) forms, a zoo
of Hodgkin–Huxley worlds, an intervention API under a hard budget, an evaluator, and a
**reference baseline LLM agent**.

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

Each world is a plain Na⁺/K⁺/leak spiker **plus one extra mechanism**. The task is the
two-hypothesis decision: *plain* vs. *plain + the world's mechanism*. Five of the six worlds
are deliberately **hard**: the extra channel is **silent under every textbook probe** (plain
and novel fire identically to standard steps and blockers) and is revealed **only by a single
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

## Evaluation criteria

An episode is scored on two axes:

1. **Mechanism selection.** Did the agent identify the true mechanism?
   - `selection_correct` — 0/1 accuracy of the committed mechanism.
   - `selection_brier` — `(1 − p_true)²`, the two-hypothesis Brier score for the posterior mass
     placed on the true mechanism (lower is better; rewards *calibrated* confidence).

2. **Held-out interventional forecasting** *(the headline metric)*. On a disjoint set of
   **test protocols the agent never ran**, how well does its predicted test-window spike count
   match the true cell's?
   - `forecast_mse` — mean squared error between predicted and true spike counts over the
     held-out protocols, **floored** at 0.25 so irreducible ±0.5-spike noise is treated as
     exact. Lower is better; a method that has correctly identified the mechanism can forecast
     interventions it never observed, which a curve-fit to the observed data cannot.

The forecast metric is the one that matters: identifying the mechanism is only useful insofar
as it lets you predict interventions. A data-efficient discovery method should reach a low
`forecast_mse` from **few** experiments.

```python
import neuronbench as nb

world = nb.load_world("ca_rebound", stochastic=True, n_channels=100, seed=0)

# ... an agent runs experiments (each consumes budget) and forms a belief ...
obs = world.run(world.discriminator(), reps=3)        # noisy, partial; costs 3
obs = world.run(nb.protocols.by_label("brief step (12 uA, 40 ms)"), block=["TTX"])

# score it
acc  = world.selection_correct("Na+K + unidentified current")   # 0/1
mse  = world.forecast_mse({lab: predicted_count for lab, _ in world.test_protocols})
```

`world.simulate(mechanism, protocol, block=...)` is the **forward model** a solver uses to score
its *own* candidate hypotheses; it consumes no budget and never touches the true cell.
`world.run(...)` is the **oracle**: it runs the hidden true cell (with a hidden RNG, so repeats
are genuinely independent) and returns only a noisy, partial observation.

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
  evaluator.py    # mechanism selection + held-out forecast MSE
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
