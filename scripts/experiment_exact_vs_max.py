"""
Experiment: exact vs max budget mode comparison.

Runs AlphaZero training in both modes with identical hyperparameters,
collecting metrics for comparison. See specs/budget_max_mode_refined.md §9.

Usage:
    python scripts/experiment_exact_vs_max.py [--quick]

    --quick: Use reduced hyperparameters for fast sanity check (~2 min)
    default: Full experiment per spec §9 (~30+ min per seed)
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.bitstring.dsl.derivation_config import DerivationConfig
from alphazeropp.synthesis.derivation_game import compute_max_productions
from alphazeropp.synthesis.budget_grammar import count_programs
from alphazeropp.training.gated_trainer import GatedTrainer


def make_config(mode, n_sites, budget, seed, quick=False):
    """Create a DerivationConfig for the given mode and seed."""
    cfg = DerivationConfig()

    # Problem params
    cfg.game.kwargs["n_sites"] = n_sites
    cfg.game.kwargs["budget"] = budget
    cfg.game.kwargs["n_ones"] = 2
    cfg.game.kwargs["n_frozen_states"] = 1
    cfg.game.kwargs["program_budget_mode"] = mode
    cfg.game.kwargs["potential_name"] = "onemax"
    cfg.game.kwargs["metric"] = "avg_reward"

    # Sync net params
    cfg.net.kwargs["budget"] = budget
    cfg.net.kwargs["n_sites"] = n_sites

    # Seeds
    cfg.agent.random_seeds = {
        "mcts": seed,
        "train": seed + 1,
        "eval": seed + 2,
        "external_policy": seed + 3,
    }

    if quick:
        # Reduced params for fast sanity check
        cfg.agent.mcts_params["n_simulations"] = 25
        cfg.trainer.n_games_per_train = 10
        cfg.trainer.n_past_iterations_to_train = 5
        cfg.evaluator.n_games = 6
        cfg.run.n_iterations = 5
        cfg.net.kwargs["d_model"] = 32
        cfg.net.kwargs["n_heads"] = 2
        cfg.net.kwargs["n_layers"] = 1
        cfg.net.kwargs["training_params"]["epochs"] = 2
        cfg.net.kwargs["training_params"]["batch_size"] = 16
    # else: use defaults from DerivationConfig

    cfg.trainer.n_procs = -1  # sequential (avoid multiprocessing complexity)
    cfg.evaluator.n_procs = -1

    return cfg


def run_one(mode, seed, exp_dir, quick=False, n_sites=6, budget=14):
    """Run a single training experiment and return results."""
    cfg = make_config(mode, n_sites, budget, seed, quick=quick)

    run_name = f"{mode}_seed{seed}"
    run_dir = exp_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.run.plot_path = str(run_dir / "training_metrics.png")
    cfg.save(str(run_dir / "config.json"))

    # Build
    game, net, agent, trainer, evaluator = cfg.build()
    leaf_eval = game.leaf_evaluator

    max_prods = compute_max_productions(budget, n_sites, mode=mode)
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Mode: {mode}, Seed: {seed}")
    print(f"  Action space: {max_prods}")
    print(f"{'='*60}\n")

    # Training loop
    gated_trainer = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold)
    program_log = []
    best_value = float("-inf")
    best_metrics = None
    best_program_str = None
    node_counts = []  # track program sizes for early-termination monitoring

    for i in range(cfg.run.n_iterations):
        t0 = time.time()
        score, accepted = gated_trainer.train_iteration()
        elapsed = time.time() - t0

        # Extract best program
        if leaf_eval._cache:
            best_key = max(leaf_eval._cache, key=leaf_eval._cache.get)
            val = leaf_eval._cache[best_key]
            if val > best_value:
                best_value = val
                best_program_str = best_key
                best_metrics = leaf_eval._full_cache[best_key]

        # Track node counts of programs seen this iteration
        for key, prog in leaf_eval._program_cache.items():
            nc = prog.node_count()
            if nc not in [entry.get("node_count") for entry in program_log
                          if entry.get("iteration") == i + 1]:
                node_counts.append(nc)

        sr = best_metrics.get("solve_rate", 0) if best_metrics else 0
        ar = best_metrics.get("avg_reward", 0) if best_metrics else 0

        entry = {
            "iteration": i + 1,
            "score": score,
            "accepted": bool(accepted),
            "best_solve_rate": sr,
            "best_avg_reward": ar,
            "best_program": best_program_str,
            "unique_programs": leaf_eval.stats()["unique_programs"],
            "elapsed_sec": round(elapsed, 1),
        }
        program_log.append(entry)

        status = "ACC" if accepted else "REJ"
        print(f"  [{run_name}] Iter {i+1}/{cfg.run.n_iterations}: "
              f"score={score*100:.0f}% {status} | "
              f"best_sr={sr:.1%} avg_r={ar:+.3f} | "
              f"progs={leaf_eval.stats()['unique_programs']} | "
              f"{elapsed:.1f}s")

        # Early stop
        if sr >= 1.0:
            print(f"  [{run_name}] Perfect program found at iteration {i+1}!")
            break

    # Save results
    with open(run_dir / "program_log.jsonl", "w") as f:
        for entry in program_log:
            f.write(json.dumps(entry) + "\n")

    trainer.statistics_manager.save_jsonl(str(run_dir / "train_stats.jsonl"))
    evaluator.statistics_manager.save_jsonl(str(run_dir / "eval_stats.jsonl"))

    # Compute node count distribution
    nc_dist = {}
    for key, prog in leaf_eval._program_cache.items():
        nc = prog.node_count()
        nc_dist[nc] = nc_dist.get(nc, 0) + 1

    summary = {
        "mode": mode,
        "seed": seed,
        "n_iterations_completed": len(program_log),
        "best_solve_rate": sr,
        "best_avg_reward": ar,
        "best_program": best_program_str,
        "unique_programs": leaf_eval.stats()["unique_programs"],
        "node_count_distribution": dict(sorted(nc_dist.items())),
        "avg_node_count": (sum(p.node_count() for p in leaf_eval._program_cache.values())
                           / max(1, len(leaf_eval._program_cache))),
        "action_space": max_prods,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exact vs Max mode experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced params for fast sanity check")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to use")
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--budget", type=int, default=14)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "quick" if args.quick else "full"
    exp_dir = Path("experiments") / "exact_vs_max" / f"{timestamp}_{tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: exact vs max budget mode")
    print(f"  N={args.n_sites}, L={args.budget}, seeds={args.seeds}")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Output: {exp_dir}/\n")

    all_summaries = []

    for mode in ["exact", "max"]:
        for seed in args.seeds:
            summary = run_one(
                mode, seed, exp_dir, quick=args.quick,
                n_sites=args.n_sites, budget=args.budget,
            )
            all_summaries.append(summary)

    # Print comparison table
    print(f"\n{'='*80}")
    print("  EXPERIMENT RESULTS")
    print(f"{'='*80}\n")

    print(f"  {'Mode':<8} {'Seed':<6} {'BestSR':<10} {'BestAvgR':<12} "
          f"{'Progs':<8} {'AvgNC':<10} {'Actions':<8}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")

    for s in all_summaries:
        print(f"  {s['mode']:<8} {s['seed']:<6} "
              f"{s['best_solve_rate']:<10.1%} "
              f"{s['best_avg_reward']:<+12.4f} "
              f"{s['unique_programs']:<8} "
              f"{s['avg_node_count']:<10.1f} "
              f"{s['action_space']:<8}")

    # Aggregate by mode
    print(f"\n  Aggregated:")
    for mode in ["exact", "max"]:
        mode_runs = [s for s in all_summaries if s["mode"] == mode]
        avg_sr = np.mean([s["best_solve_rate"] for s in mode_runs])
        avg_ar = np.mean([s["best_avg_reward"] for s in mode_runs])
        avg_progs = np.mean([s["unique_programs"] for s in mode_runs])
        avg_nc = np.mean([s["avg_node_count"] for s in mode_runs])
        print(f"  {mode:<8} avg_sr={avg_sr:.1%}  avg_reward={avg_ar:+.4f}  "
              f"avg_progs={avg_progs:.0f}  avg_node_count={avg_nc:.1f}")

    # Node count distributions
    print(f"\n  Node count distributions:")
    for s in all_summaries:
        dist_str = " ".join(f"{k}:{v}" for k, v in
                            sorted(s["node_count_distribution"].items()))
        print(f"  {s['mode']:<8} seed={s['seed']}: {dist_str}")

    # Save combined results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n  Full results: {exp_dir}/results.json")
    print()


if __name__ == "__main__":
    main()
