#!/usr/bin/env python
"""Run the reference pure-LLM baseline on a NeuronBench world.

    export OPENROUTER_API_KEY=...
    python examples/run_baseline.py --world ca_rebound --stochastic --budget 8

The baseline lets a language model choose experiments and forecast the held-out interventions, with no
Bayesian inference or experiment design. The reported forecast MSE is the number a discovery method
must beat.
"""
import argparse

import neuronbench as nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="ca_rebound", choices=nb.list_worlds())
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--n-channels", type=float, default=100.0)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="anthropic/claude-opus-4.7")
    args = ap.parse_args()

    world = nb.load_world(args.world, stochastic=args.stochastic, n_channels=args.n_channels, seed=args.seed)
    client = nb.agent.openrouter_client(model=args.model)
    result = nb.agent.run_baseline(world, client, budget=args.budget, seed=args.seed)

    print(f"world: {result['world']}  (truth: {world._truth_name})")
    print("experiments chosen:")
    for lab, cnt in result["observed"]:
        print(f"  - {lab} -> {cnt} spikes")
    print("held-out forecast:")
    for lab, cnt in result["predictions"].items():
        print(f"  - {lab}: {cnt:.1f}")
    print(f"\nheld-out forecast MSE: {result['forecast_mse']:.3f}")


if __name__ == "__main__":
    main()
