"""Generic episodic evaluation loop for the benchmark harness.

Runs N episodes with an arbitrary policy function and collects
per-episode statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


@dataclass
class EpisodeResult:
    """Statistics for a single evaluation episode."""
    episode_return: float
    episode_steps: int
    solved: bool
    invalid_action_count: int
    max_room_reached: int
    keys_picked: int


def _count_unlocked_rooms(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
    """Count number of unlocked rooms from observation."""
    count = 0
    for r in range(env.D):
        if obs[env._unlocked_offset + r] == 1.0:
            count += 1
    return count


def _count_keys_picked(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
    """Count keys that have been picked (no longer available)."""
    count = 0
    for k in range(env.K):
        if obs[env._key_offset + k] == 0.0:
            count += 1
    return count


def evaluate_policy(
    policy_fn: Callable[[np.ndarray, DoorsPDDLLiteEnv], int],
    env_factory: Callable[[], DoorsPDDLLiteEnv],
    n_episodes: int = 100,
    seed: int = 0,
) -> list[EpisodeResult]:
    """Run *n_episodes* with the given policy and collect statistics.

    Parameters
    ----------
    policy_fn : callable(obs, env) -> int
        Maps an observation and env to an action index.
    env_factory : callable() -> DoorsPDDLLiteEnv
        Creates a fresh env instance.
    n_episodes : int
        Number of evaluation episodes.
    seed : int
        Base seed; episode *i* uses ``seed + i``.

    Returns
    -------
    list[EpisodeResult]
        Per-episode results.
    """
    results = []
    env = env_factory()

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        invalid_count = 0

        for _ in range(env.horizon):
            action = policy_fn(obs, env)

            # Check if action is invalid (precondition-wise)
            precondition_mask = env.action_masks("precondition")
            if not precondition_mask[action]:
                invalid_count += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        results.append(EpisodeResult(
            episode_return=total_reward,
            episode_steps=env.step_count,
            solved=env.is_solved(obs),
            invalid_action_count=invalid_count,
            max_room_reached=_count_unlocked_rooms(obs, env) - 1,
            keys_picked=_count_keys_picked(obs, env),
        ))

    return results


def aggregate_episodes(results: list[EpisodeResult]) -> dict:
    """Aggregate episode results into summary statistics.

    Returns a dict with keys matching CheckpointResult eval fields.
    """
    if not results:
        return {
            "eval_return_mean": 0.0,
            "eval_return_std": 0.0,
            "eval_success_rate": 0.0,
            "eval_episode_length_mean": 0.0,
            "max_room_reached_mean": 0.0,
            "keys_picked_mean": 0.0,
            "invalid_action_rate": 0.0,
        }

    returns = [r.episode_return for r in results]
    steps = [r.episode_steps for r in results]
    total_actions = sum(steps)

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_success_rate": sum(1 for r in results if r.solved) / len(results),
        "eval_episode_length_mean": float(np.mean(steps)),
        "max_room_reached_mean": float(np.mean([r.max_room_reached for r in results])),
        "keys_picked_mean": float(np.mean([r.keys_picked for r in results])),
        "invalid_action_rate": (
            sum(r.invalid_action_count for r in results) / total_actions
            if total_actions > 0 else 0.0
        ),
    }
