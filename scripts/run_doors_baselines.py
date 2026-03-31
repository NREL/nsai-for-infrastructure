"""
Doors D=2 baselines: random agent, hardcoded optimal, and optimal DSL program.

Usage:
    python scripts/run_doors_baselines.py
    python scripts/run_doors_baselines.py --n_episodes 5000 --seed 0 --verbose
"""

import argparse

import numpy as np

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.interpreter import (
    run_policy_episode, format_trace,
)


def build_optimal_program():
    """Construct the known-optimal 16-node DSL program for Doors D=2."""
    program = Ite(
        And(Not(IsZero(1)), Not(IsZero(6))),
        Flip(4),
        Ite(
            Not(IsZero(6)),
            Flip(1),
            Ite(IsZero(3), Flip(3), Default(Flip(5))),
        ),
    )
    assert program.node_count() == 16, f"Expected 16 nodes, got {program.node_count()}"
    return program


def run_random_baseline(n_episodes: int = 1000, seed: int = 42) -> dict:
    """Run uniform-random agent on Doors D=2."""
    env = DoorsPDDLLiteEnv.make_d2()
    rng = np.random.default_rng(seed)
    solves = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done:
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
        if env.is_solved(obs):
            solves += 1
        total_reward += ep_reward
        total_steps += ep_steps

    return {
        "solve_rate": solves / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "n_episodes": n_episodes,
    }


def run_optimal_env_baseline() -> dict:
    """Run hardcoded optimal action sequence [1, 4, 3] on Doors D=2."""
    env = DoorsPDDLLiteEnv.make_d2()
    obs, _ = env.reset()
    rewards = []
    for action in [1, 4, 3]:
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
    return {
        "reward": sum(rewards),
        "steps": len(rewards),
        "solved": env.is_solved(obs),
        "per_step_rewards": rewards,
    }


def run_optimal_dsl_baseline(verbose: bool = False):
    """Run optimal 16-node DSL program via run_policy_episode."""
    env = DoorsPDDLLiteEnv.make_d2()
    program = build_optimal_program()
    result = run_policy_episode(
        env, program, is_solved=env.is_solved, verbose=verbose,
    )
    return result, program


def main():
    parser = argparse.ArgumentParser(description="Doors D=2 baselines")
    parser.add_argument("--n_episodes", type=int, default=1000,
                        help="Episodes for random baseline (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random baseline (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full DSL execution trace")
    args = parser.parse_args()

    sep = "=" * 60
    line = "-" * 60

    # --- Random baseline ---
    print(sep)
    print("  Doors D=2 Baselines")
    print(sep)
    print()

    rand = run_random_baseline(args.n_episodes, args.seed)
    print(f"  Random baseline ({rand['n_episodes']} episodes, seed={args.seed}):")
    print(f"    solve_rate:  {rand['solve_rate']:.4f}")
    print(f"    avg_reward: {rand['avg_reward']:+.4f}")
    print(f"    avg_steps:  {rand['avg_steps']:.1f}")
    print()

    # --- Optimal env baseline ---
    print(line)
    opt = run_optimal_env_baseline()
    print(f"  Optimal env baseline (hardcoded [1, 4, 3]):")
    print(f"    reward:      {opt['reward']:.4f}")
    print(f"    steps:       {opt['steps']}")
    print(f"    solved:      {opt['solved']}")
    per = opt["per_step_rewards"]
    print(f"    per_step:    [{per[0]:+.4f}, {per[1]:+.4f}, {per[2]:+.4f}]")
    print()

    # --- Optimal DSL baseline ---
    print(line)
    dsl_result, program = run_optimal_dsl_baseline(verbose=args.verbose)
    print(f"  Optimal DSL baseline (16-node program via run_policy_episode):")
    print(f"    reward:      {dsl_result.cumulative_reward:.4f}")
    print(f"    steps:       {dsl_result.total_env_steps}")
    print(f"    solved:      {dsl_result.solved}")
    print(f"    interp_ops:  {dsl_result.total_interp_ops}")
    print()

    if args.verbose:
        print(line)
        print("  DSL Execution Trace:")
        print(format_trace(dsl_result, program))
        print()

    # --- Summary ---
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  {'Baseline':<20s} {'solved':>8s} {'reward':>8s} {'steps':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6}")
    sr = f"{rand['solve_rate']:.1%}"
    print(f"  {'Random':<20s} {sr:>8s} {rand['avg_reward']:>+8.4f} {rand['avg_steps']:>6.1f}")
    print(f"  {'Optimal (env)':<20s} {'True':>8s} {opt['reward']:>+8.4f} {opt['steps']:>6d}")
    print(f"  {'Optimal (DSL)':<20s} {'True':>8s} {dsl_result.cumulative_reward:>+8.4f} {dsl_result.total_env_steps:>6d}")
    print(sep)


if __name__ == "__main__":
    main()
