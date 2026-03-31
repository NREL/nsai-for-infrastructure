"""Random policy adapter — lower-bound baseline."""

from __future__ import annotations

import time
from typing import Callable, Iterator

import numpy as np

from alphazeropp.benchmark.adapters.base import BenchmarkAlgorithm
from alphazeropp.benchmark.eval_loop import evaluate_policy, aggregate_episodes
from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


def _random_policy(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
    """Select a random action, respecting mask_mode if set."""
    mask_mode = getattr(env, "benchmark_mask_mode", "none")
    if mask_mode == "precondition":
        mask = env.action_masks("precondition")
        valid = np.where(mask)[0]
        return int(np.random.choice(valid))
    return int(env.action_space.sample())


class RandomAdapter(BenchmarkAlgorithm):
    """Random policy baseline. No training — yields one checkpoint."""

    def __init__(self, D: int, locs_per_room: int = 2, mask_mode: str = "none"):
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode

    def name(self) -> str:
        return "random"

    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        t0 = time.time()
        np.random.seed(seed)

        episodes = evaluate_policy(
            policy_fn=_random_policy,
            env_factory=eval_env_factory,
            n_episodes=eval_episodes,
            seed=seed,
        )
        agg = aggregate_episodes(episodes)

        yield CheckpointResult(
            algorithm=self.name(),
            seed=seed,
            D=self._D,
            locs_per_room=self._locs_per_room,
            mask_mode=self._mask_mode,
            env_steps=0,
            train_episodes=0,
            learner_updates=0,
            eval_checkpoint_idx=0,
            wall_clock_sec=time.time() - t0,
            **agg,
        )
