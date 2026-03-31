#!/usr/bin/env python3
"""
Pure MCTS baseline for Doors program synthesis (no neural network training).

Runs N independent MCTS derivation rounds using UniformPolicyValueNet
(uniform prior, value=0). This isolates MCTS search quality from the
neural network learning loop.

Purpose: determine whether MCTS alone can find solving programs, and
how many rounds/simulations are needed. Comparison baseline for the
full AlphaZero training loop (run_doors_derivation.py).

Usage:
    python scripts/run_doors_mcts_baseline.py --num_rooms 3 --n_rounds 400 --n_simulations 100
    python scripts/run_doors_mcts_baseline.py --num_rooms 3 --n_rounds 80 --n_simulations 200
    python scripts/run_doors_mcts_baseline.py --no-interactive
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig,
    compute_doors_derived_params,
    doors_initial_state,
)
from alphazeropp.synthesis.derivation_game import (
    DerivationGame,
    UniformPolicyValueNet,
    compute_max_productions,
)
from alphazeropp.synthesis.factored_derivation_game import (
    FactoredDerivationGame,
    compute_max_factored_actions,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator, VALID_METRICS
from alphazeropp.synthesis.budget_grammar import count_programs
from alphazeropp.core.mcts import MCTS


# ---------------------------------------------------------------------------
# Game factory
# ---------------------------------------------------------------------------

def make_game_and_net(mode, doors_cfg, derived, leaf_eval, budget_mode, allow_and):
    """Create a DerivationGame (or FactoredDerivationGame) + UniformPolicyValueNet."""
    budget = derived["budget"]
    n_sites = derived["n_sites"]
    n_actions = doors_cfg.M + doors_cfg.K + 1
    one_hot_groups = [list(range(doors_cfg.M))]

    common = dict(
        budget=budget,
        n_sites=n_sites,
        leaf_evaluator=leaf_eval,
        program_budget_mode=budget_mode,
        allow_and=allow_and,
        allow_not=True,
        n_actions=n_actions,
        one_hot_groups=one_hot_groups,
    )

    if mode == "doors_factored":
        game = FactoredDerivationGame(**common)
    elif mode == "doors_d10_macro":
        from alphazeropp.instances.doors.dsl.doors_macros import make_macro_fn
        game = FactoredDerivationGame(
            **common,
            macro_productions_fn=make_macro_fn(doors_cfg),
            max_condition_budget=12,
        )
    else:  # "doors" (flat)
        game = DerivationGame(**common)

    net = UniformPolicyValueNet(game.action_space.n)
    return game, net


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------

def run_mcts_baseline(
    mode, doors_cfg, derived, leaf_eval, budget_mode, allow_and,
    n_rounds, n_simulations, c_exploration, temperature, seed,
    dirichlet_alpha=0.25, dirichlet_epsilon=0.4,
):
    """Run N independent MCTS rounds, return logs and best program."""
    best_program = None
    best_value = float("-inf")
    best_round = -1
    round_logs = []
    t0 = time.time()

    for r in range(n_rounds):
        np.random.seed(seed + r)

        game, net = make_game_and_net(
            mode, doors_cfg, derived, leaf_eval, budget_mode, allow_and,
        )
        game.reset_wrapper()
        mcts = MCTS(
            game, net,
            n_simulations=n_simulations,
            temperature=temperature,
            c_exploration=c_exploration,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        step_count = 0
        while not game.terminated and not game.truncated:
            probs = mcts.perform_simulations(None, add_noise=True)
            # Sample from distribution (like training self-play), not argmax
            action = int(np.random.choice(len(probs), p=probs))
            game.step_wrapper(action)
            step_count += 1

        program = game.get_program()
        if program is not None:
            value = leaf_eval(program)
            metrics = leaf_eval.get_all_metrics(program)
        else:
            value = 0.0
            metrics = {"solve_rate": 0.0, "avg_reward": 0.0}

        if value > best_value:
            best_value = value
            best_program = program
            best_round = r

        stats = leaf_eval.stats()
        is_best = " <-- NEW BEST" if value >= best_value else ""
        sr = metrics["solve_rate"]
        ar = metrics.get("avg_reward", 0)

        if (r + 1) % max(1, n_rounds // 20) == 0 or sr > 0 or r == 0:
            prog_str = program.pretty().replace("\n", " | ") if program else "DEAD_END"
            print(f"Round {r+1}/{n_rounds}: sr={sr:.2f} reward={ar:+.3f} "
                  f"unique={stats['unique_programs']} steps={step_count}{is_best}")
            if sr > 0:
                print(f"  Program: {prog_str}")

        round_logs.append({
            "round": r + 1,
            "derivation_steps": step_count,
            "solve_rate": round(sr, 6),
            "avg_reward": round(ar, 6),
            "metric_value": round(value, 6),
            "best_metric_value": round(best_value, 6),
            "unique_programs": stats["unique_programs"],
            "cache_hits": stats["cache_hits"],
            "total_env_steps": stats["total_env_steps"],
            "program": program.pretty().replace("\n", " | ") if program else "DEAD_END",
            "seed": seed + r,
        })

    elapsed = time.time() - t0
    return best_program, best_value, best_round, round_logs, elapsed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(exp_dir, round_logs, best_program, best_value, best_round,
                 leaf_eval, args, elapsed):
    """Save JSONL log, summary JSON, and convergence plot."""
    # JSONL
    jsonl_path = exp_dir / "round_log.jsonl"
    with open(jsonl_path, "w") as f:
        for log in round_logs:
            f.write(json.dumps(log) + "\n")

    # Summary
    stats = leaf_eval.stats()
    best_metrics = leaf_eval.get_all_metrics(best_program) if best_program else {}
    summary = {
        "mode": args.mode,
        "num_rooms": args.num_rooms,
        "n_rounds": args.n_rounds,
        "n_simulations": args.n_simulations,
        "seed": args.seed,
        "best_solve_rate": best_metrics.get("solve_rate", 0),
        "best_avg_reward": best_metrics.get("avg_reward", 0),
        "best_round": best_round + 1 if best_round >= 0 else None,
        "best_program": best_program.pretty() if best_program else None,
        "unique_programs": stats["unique_programs"],
        "total_env_steps": stats["total_env_steps"],
        "elapsed_s": round(elapsed, 1),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = [l["round"] for l in round_logs]
        best_so_far = []
        running = float("-inf")
        for l in round_logs:
            running = max(running, l["solve_rate"])
            best_so_far.append(running)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Convergence
        ax1.plot(rounds, best_so_far, "-", color="#4C72B0", linewidth=2,
                 label="Best solve rate so far")
        ax1.plot(rounds, [l["solve_rate"] for l in round_logs], ".",
                 color="#999999", markersize=2, alpha=0.4, label="Per-round")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Solve Rate")
        ax1.set_title(f"Pure MCTS Convergence ({args.mode}, D={args.num_rooms})\n"
                      f"sims={args.n_simulations} rounds={args.n_rounds} seed={args.seed}")
        ax1.set_ylim(-0.05, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Unique programs
        ax2.plot(rounds, [l["unique_programs"] for l in round_logs],
                 "-", color="#2ECC71", linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Unique Programs")
        ax2.set_title(f"Program Exploration | {args.n_rounds} rounds")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = exp_dir / f"convergence_{args.mode}_D{args.num_rooms}_sims{args.n_simulations}.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"  Plot: {plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return jsonl_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pure MCTS baseline for Doors program synthesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/run_doors_mcts_baseline.py --num_rooms 3 --n_rounds 400 --n_simulations 100
              python scripts/run_doors_mcts_baseline.py --num_rooms 3 --n_rounds 80 --n_simulations 200
              python scripts/run_doors_mcts_baseline.py --mode doors_factored --num_rooms 3
        """),
    )
    parser.add_argument("--num_rooms", type=int, default=3)
    parser.add_argument("--locs_per_room", type=int, default=2)
    parser.add_argument("--n_rounds", type=int, default=400)
    parser.add_argument("--n_simulations", type=int, default=100)
    parser.add_argument("--mode", type=str, default="doors",
                        choices=["doors", "doors_factored", "doors_d10_macro"])
    parser.add_argument("--metric", type=str, default="weighted",
                        choices=list(VALID_METRICS))
    parser.add_argument("--blend_alpha", type=float, default=0.7)
    parser.add_argument("--penalty_lambda", type=float, default=0.1)
    parser.add_argument("--budget_mode", type=str, default="max",
                        choices=["max", "exact"])
    parser.add_argument("--c_exploration", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.25)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()

    allow_and = True  # required for Doors

    # Derived parameters
    derived = compute_doors_derived_params(args.num_rooms, args.locs_per_room)
    budget = derived["budget"]
    n_sites = derived["n_sites"]

    # Doors config
    doors_cfg = DoorsGameConfig(
        num_rooms=args.num_rooms,
        locs_per_room=args.locs_per_room,
        horizon=derived["horizon"],
        step_penalty=0.01,
        unlock_bonus=0.1,
    )

    # Frozen states
    frozen_states = [doors_initial_state(doors_cfg)]

    # Leaf evaluator
    leaf_eval = LeafEvaluator(
        n_sites=n_sites,
        frozen_states=frozen_states,
        game_config=doors_cfg,
        metric=args.metric,
        penalty_lambda=args.penalty_lambda,
        blend_alpha=args.blend_alpha,
        is_solved=doors_cfg.is_solved,
    )

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_programs = count_programs(n_sites, budget)
    dirname = (f"{timestamp}_{args.mode}_D{args.num_rooms}"
               f"_sims{args.n_simulations}_rounds{args.n_rounds}")
    exp_dir = Path("experiments") / "doors_mcts_baseline" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    n_actions_env = doors_cfg.M + doors_cfg.K + 1
    max_prods = compute_max_productions(budget, n_sites, mode=args.budget_mode,
                                         allow_and=allow_and, n_actions=n_actions_env)
    print()
    print("=" * 70)
    print("  PURE MCTS BASELINE — Doors Program Synthesis")
    print("=" * 70)
    print()
    print(f"  Mode:          {args.mode}")
    print(f"  Rooms (D):     {args.num_rooms}")
    print(f"  Obs size:      {n_sites}")
    print(f"  Budget:        {budget}")
    print(f"  Max actions:   {max_prods} (flat)")
    print(f"  Total programs: {total_programs}")
    print(f"  Metric:        {args.metric}")
    print()
    print(f"  MCTS sims:     {args.n_simulations}")
    print(f"  Rounds:        {args.n_rounds}")
    print(f"  c_exploration: {args.c_exploration}")
    print(f"  temperature:   {args.temperature}")
    print(f"  Seed:          {args.seed}")
    print()
    print(f"  Network:       UniformPolicyValueNet (no training)")
    print(f"  Experiment:    {exp_dir}/")
    print()
    print("=" * 70)
    print()

    # Run
    best_program, best_value, best_round, round_logs, elapsed = run_mcts_baseline(
        args.mode, doors_cfg, derived, leaf_eval, args.budget_mode, allow_and,
        args.n_rounds, args.n_simulations, args.c_exploration, args.temperature,
        args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )

    # Results
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    stats = leaf_eval.stats()
    if best_program is not None:
        bm = leaf_eval.get_all_metrics(best_program)
        print(f"  Best program (round {best_round + 1}):")
        print(f"    {best_program.pretty()}")
        print(f"  Solve rate: {bm['solve_rate']:.2f}")
        print(f"  Avg reward: {bm['avg_reward']:+.4f}")
    else:
        print("  No valid program found.")
    print(f"  Unique programs: {stats['unique_programs']} / {total_programs} "
          f"({stats['unique_programs']*100/max(total_programs,1):.4f}% coverage)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()

    # Save
    jsonl_path = save_results(exp_dir, round_logs, best_program, best_value,
                              best_round, leaf_eval, args, elapsed)
    print(f"  Log:     {jsonl_path}")
    print(f"  Summary: {exp_dir / 'summary.json'}")
    print()


if __name__ == "__main__":
    main()
