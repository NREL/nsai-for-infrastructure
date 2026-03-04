"""
Estimate the expressivity gap between allow_and=True and allow_and=False
on the DoorsPDDLLiteEnv using pure MCTS search (no neural network training).

Uses UniformPolicyValueNet (uniform prior + zero value) so that MCTS
relies entirely on backed-up leaf evaluations. Any difference in search
outcomes is due to the grammar's structural expressivity.

Usage:
    python scripts/estimate_expressivity_gap.py [--rounds N] [--sims S]
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import time

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.core.config import AgentConfig
from alphazeropp.synthesis.derivation_game import (
    DerivationGame, UniformPolicyValueNet,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


def build_search_agent(
    allow_and: bool,
    budget: int = 18,
    n_simulations: int = 200,
    num_rooms: int = 2,
):
    """Build a DerivationGame + UniformPolicyValueNet agent for pure MCTS search."""
    doors_cfg = DoorsGameConfig(num_rooms=num_rooms)
    n_sites = doors_cfg.obs_size()
    frozen_states = [doors_initial_state(doors_cfg)]

    leaf_eval = LeafEvaluator(
        n_sites,
        frozen_states,
        doors_cfg,
        metric="weighted",
        blend_alpha=0.7,
        is_solved=doors_cfg.is_solved,
    )

    game = DerivationGame(
        budget, n_sites, leaf_eval,
        program_budget_mode="max",
        allow_and=allow_and,
        allow_not=True,
    )

    net = UniformPolicyValueNet(action_size=game.action_space.n)

    agent = Agent(
        game=game,
        net=net,
        mcts_params={
            "n_simulations": n_simulations,
            "temperature": 1.0,
            "c_exploration": 1.5,
            "dirichlet_alpha": 0.25,
            "dirichlet_epsilon": 0.40,
        },
        reward_discount=1.0,
        random_seeds={"mcts": 42, "train": 0, "eval": 0, "external_policy": 0},
    )
    return agent, game, leaf_eval


def run_search_rounds(allow_and: bool, n_rounds: int, budget: int,
                      n_simulations: int, num_rooms: int):
    """Run n_rounds of MCTS self-play games, collecting best program stats."""
    agent, game, leaf_eval = build_search_agent(
        allow_and=allow_and, budget=budget,
        n_simulations=n_simulations, num_rooms=num_rooms,
    )

    tag = "And=True " if allow_and else "And=False"

    for i in range(n_rounds):
        agent.play_game(game)
        prog_str, prog_ast, prog_metrics, prog_value = _extract_best(leaf_eval)
        if prog_metrics:
            sr = prog_metrics.get("solve_rate", 0)
            ar = prog_metrics.get("avg_reward", 0)
        else:
            sr, ar = 0.0, 0.0
        print(f"  [{tag}] Round {i+1}/{n_rounds}: "
              f"best solve_rate={sr:.1%}, best avg_reward={ar:+.4f}, "
              f"unique_programs={leaf_eval.stats()['unique_programs']}")

    return leaf_eval


def _extract_best(leaf_eval):
    """Extract the best program from LeafEvaluator's cache."""
    if not leaf_eval._cache:
        return None, None, None, float("-inf")
    best_key = max(leaf_eval._cache, key=leaf_eval._cache.get)
    return (
        best_key,
        leaf_eval._program_cache[best_key],
        leaf_eval._full_cache[best_key],
        leaf_eval._cache[best_key],
    )


def main():
    parser = argparse.ArgumentParser(description="Estimate expressivity gap")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of MCTS search rounds per condition")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations per step")
    parser.add_argument("--budget", type=int, default=18,
                        help="AST node budget")
    parser.add_argument("--num-rooms", type=int, default=2,
                        help="Number of rooms (D)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  Expressivity Gap Estimation: And vs No-And")
    print("=" * 70)
    print()
    print(f"  Rounds per condition: {args.rounds}")
    print(f"  MCTS simulations:    {args.sims}")
    print(f"  Budget:              {args.budget}")
    print(f"  Rooms (D):           {args.num_rooms}")
    print()

    # --- allow_and=True ---
    print("-" * 70)
    print("  Running with allow_and=True ...")
    print("-" * 70)
    t0 = time.time()
    leaf_eval_and = run_search_rounds(
        allow_and=True, n_rounds=args.rounds, budget=args.budget,
        n_simulations=args.sims, num_rooms=args.num_rooms,
    )
    t_and = time.time() - t0

    # --- allow_and=False ---
    print()
    print("-" * 70)
    print("  Running with allow_and=False ...")
    print("-" * 70)
    t0 = time.time()
    leaf_eval_no = run_search_rounds(
        allow_and=False, n_rounds=args.rounds, budget=args.budget,
        n_simulations=args.sims, num_rooms=args.num_rooms,
    )
    t_no = time.time() - t0

    # --- Results ---
    _, prog_and, metrics_and, val_and = _extract_best(leaf_eval_and)
    _, prog_no, metrics_no, val_no = _extract_best(leaf_eval_no)

    sr_and = metrics_and.get("solve_rate", 0) if metrics_and else 0
    ar_and = metrics_and.get("avg_reward", 0) if metrics_and else 0
    sr_no = metrics_no.get("solve_rate", 0) if metrics_no else 0
    ar_no = metrics_no.get("avg_reward", 0) if metrics_no else 0

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Metric':<25} {'And=True':>12} {'And=False':>12} {'Gap':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Best solve_rate':<25} {sr_and:>12.1%} {sr_no:>12.1%} "
          f"{sr_and - sr_no:>+12.1%}")
    print(f"  {'Best avg_reward':<25} {ar_and:>+12.4f} {ar_no:>+12.4f} "
          f"{ar_and - ar_no:>+12.4f}")
    print(f"  {'Unique programs':<25} "
          f"{leaf_eval_and.stats()['unique_programs']:>12d} "
          f"{leaf_eval_no.stats()['unique_programs']:>12d}")
    print(f"  {'Time (s)':<25} {t_and:>12.1f} {t_no:>12.1f}")
    print()

    if prog_and is not None:
        print("  Best program (And=True):")
        for line in prog_and.pretty().split("\n"):
            print(f"    {line}")
        print()

    if prog_no is not None:
        print("  Best program (And=False):")
        for line in prog_no.pretty().split("\n"):
            print(f"    {line}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
