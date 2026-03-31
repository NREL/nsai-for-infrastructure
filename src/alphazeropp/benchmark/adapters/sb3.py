"""SB3 algorithm adapters — stubs with guarded imports.

These adapters require stable-baselines3 (and sb3-contrib for MaskablePPO).
If not installed, attempting to instantiate will raise ImportError with
install instructions.
"""

from __future__ import annotations

from typing import Callable, Iterator

from alphazeropp.benchmark.adapters.base import BenchmarkAlgorithm
from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv

try:
    from stable_baselines3 import DQN  # noqa: F401
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO  # noqa: F401
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False


class DQNAdapter(BenchmarkAlgorithm):
    """DQN adapter using stable-baselines3.

    TODO: Implement when SB3 integration is needed.
    """

    def __init__(self, D: int, locs_per_room: int = 2, mask_mode: str = "none"):
        if not SB3_AVAILABLE:
            raise ImportError(
                "DQN requires stable-baselines3. Install with:\n"
                "  pip install 'stable-baselines3[extra]'"
            )
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode

    def name(self) -> str:
        return "dqn"

    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        # TODO: Implement DQN training loop with SB3
        raise NotImplementedError("DQN adapter not yet implemented")
        yield  # make this a generator


class MaskablePPOAdapter(BenchmarkAlgorithm):
    """MaskablePPO adapter using sb3-contrib.

    TODO: Implement when sb3-contrib integration is needed.
    """

    def __init__(self, D: int, locs_per_room: int = 2, mask_mode: str = "precondition"):
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "MaskablePPO requires sb3-contrib. Install with:\n"
                "  pip install sb3-contrib"
            )
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode

    def name(self) -> str:
        return "maskable_ppo"

    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        # TODO: Implement MaskablePPO training loop with sb3-contrib
        raise NotImplementedError("MaskablePPO adapter not yet implemented")
        yield  # make this a generator


def get_sb3_adapter(algo: str, **kwargs) -> BenchmarkAlgorithm:
    """Factory function for SB3-based adapters."""
    if algo == "dqn":
        return DQNAdapter(**kwargs)
    elif algo == "maskable_ppo":
        return MaskablePPOAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown SB3 algorithm: {algo}")
