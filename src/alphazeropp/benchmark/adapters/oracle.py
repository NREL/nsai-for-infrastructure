"""Oracle adapter — runs the optimal greedy policy as a benchmark upper bound."""

from __future__ import annotations

import time
from typing import Callable, Iterator

from alphazeropp.benchmark.adapters.base import BenchmarkAlgorithm
from alphazeropp.benchmark.eval_loop import evaluate_policy, aggregate_episodes
from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.instances.doors.oracle import (
    optimal_return,
    oracle_action,
)


class OracleAdapter(BenchmarkAlgorithm):
    """Wraps the oracle policy from oracle.py into the benchmark interface.

    No training occurs — yields a single checkpoint with optimal evaluation.
    """

    def __init__(self, D: int, locs_per_room: int = 2, mask_mode: str = "none"):
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode

    def name(self) -> str:
        return "oracle"

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

        episodes = evaluate_policy(
            policy_fn=oracle_action,
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
            solved_flag=True,
            solve_env_steps=0,
            solve_wall_clock_sec=0.0,
            **agg,
        )
