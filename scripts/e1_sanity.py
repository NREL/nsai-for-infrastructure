"""
Sanity check script: verifies potential-based shaped rewards are correct.

Usage:
    python scripts/e1_sanity.py --task onemax --N 10 --H 200 --seed 0

Expected output:
    Telescoping check: max_abs_error=0.0
    Greedy oracle solve_rate=1.0
"""

import argparse
import numpy as np

from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY
from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import (
    ShapedBitStringGym,
    make_initial_states,
)


def run_telescoping_check(
    potential_fn,
    n_sites: int,
    n_episodes: int,
    seed: int,
) -> float:
    """Run episodes with random actions and return max abs telescoping error."""
    base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
    env = ShapedBitStringGym(
        env=base_env,
        potential_fn=potential_fn,
        reward_mode="dense_potential",
    )

    rng = np.random.default_rng(seed)
    max_abs_error = 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        f_initial = potential_fn(obs)

        total_int_diff = 0
        done = False
        while not done:
            action = int(rng.integers(0, n_sites))
            obs, reward, terminated, truncated, info = env.step(action)
            total_int_diff += info["potential_curr"] - info["potential_prev"]
            done = terminated or truncated

        f_final = potential_fn(obs)
        int_error = abs(total_int_diff - (f_final - f_initial))
        max_abs_error = max(max_abs_error, float(int_error))

    return max_abs_error


def run_greedy_oracle(
    potential_fn,
    n_sites: int,
    n_states: int,
    seed: int,
) -> float:
    """Run greedy oracle on frozen initial states and return solve rate."""
    frozen = make_initial_states(seed=seed, n_sites=n_sites, n_states=n_states)

    base_env = BitStringGym(n_sites=n_sites, bit_flip=True, sparse_reward=False)
    env = ShapedBitStringGym(
        env=base_env,
        potential_fn=potential_fn,
        reward_mode="dense_potential",
        frozen_states=frozen,
    )

    solves = 0
    for _ in range(n_states):
        obs, _ = env.reset()
        done = False
        while not done:
            zero_idx = np.where(obs == 0.0)[0]
            action = int(zero_idx[0]) if len(zero_idx) > 0 else 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if np.all(obs == 1.0):
            solves += 1

    return solves / n_states


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check for potential-based shaped rewards on BitString"
    )
    parser.add_argument("--task", type=str, default="onemax",
                        choices=list(POTENTIAL_REGISTRY.keys()))
    parser.add_argument("--N", type=int, default=10,
                        help="Bitstring length")
    parser.add_argument("--H", type=int, default=200,
                        help="Number of episodes for telescoping check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    potential_fn = POTENTIAL_REGISTRY[args.task]

    max_abs_error = run_telescoping_check(
        potential_fn=potential_fn,
        n_sites=args.N,
        n_episodes=args.H,
        seed=args.seed,
    )
    print(f"Telescoping check: max_abs_error={max_abs_error}")

    if args.task == "onemax":
        solve_rate = run_greedy_oracle(
            potential_fn=potential_fn,
            n_sites=args.N,
            n_states=100,
            seed=args.seed,
        )
        print(f"Greedy oracle solve_rate={solve_rate}")
    else:
        print("Greedy oracle: skipped (only applicable to onemax)")


if __name__ == "__main__":
    main()
