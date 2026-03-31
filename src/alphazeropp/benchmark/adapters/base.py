"""Abstract base class for benchmark algorithm adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator

from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


class BenchmarkAlgorithm(ABC):
    """Minimal interface for plugging an algorithm into the benchmark harness.

    Implementations must define ``name()`` and ``train_and_yield_checkpoints()``.
    The generator pattern lets the harness check solve criteria incrementally
    without the algorithm needing to know about them.
    """

    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'alphazero', 'dqn', 'oracle'."""

    @abstractmethod
    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        """Train and periodically yield evaluation results.

        Parameters
        ----------
        env_factory : callable
            Creates a fresh training env.
        eval_env_factory : callable
            Creates a fresh evaluation env.
        total_steps : int
            Total training budget (interpretation is algorithm-specific:
            iterations for AlphaZero, timesteps for SB3).
        eval_interval : int
            Evaluate every *eval_interval* training units.
        eval_episodes : int
            Number of evaluation episodes per checkpoint.
        seed : int
            Random seed for reproducibility.

        Yields
        ------
        CheckpointResult
            Evaluation snapshot after each checkpoint.
        """
