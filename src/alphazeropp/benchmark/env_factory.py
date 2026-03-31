"""Environment factory for benchmark harness.

Creates DoorsPDDLLiteEnv instances with standardized configuration
for training and evaluation.
"""

from __future__ import annotations

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig,
    compute_doors_derived_params,
)
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


def make_env(
    D: int = 3,
    locs_per_room: int = 2,
    mask_mode: str = "none",
    seed: int | None = None,
) -> DoorsPDDLLiteEnv:
    """Create a DoorsPDDLLiteEnv with benchmark-standard configuration.

    Parameters
    ----------
    D : int
        Number of rooms.
    locs_per_room : int
        Locations per room.
    mask_mode : str
        Action masking mode: "none" or "precondition".
    seed : int or None
        Random seed for env reset.

    Returns
    -------
    DoorsPDDLLiteEnv
        A fresh environment instance with ``mask_mode`` stored as an attribute.
    """
    params = compute_doors_derived_params(D, locs_per_room)
    cfg = DoorsGameConfig(
        num_rooms=D,
        locs_per_room=locs_per_room,
        horizon=params["horizon"],
    )
    env = cfg.make_env(cfg.obs_size())
    # Store mask_mode for the eval loop to reference
    env.benchmark_mask_mode = mask_mode
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_factory(
    D: int = 3,
    locs_per_room: int = 2,
    mask_mode: str = "none",
    seed: int | None = None,
):
    """Return a zero-argument callable that creates a fresh env.

    Each call returns a new, independent env instance.
    """
    def factory():
        return make_env(D=D, locs_per_room=locs_per_room,
                        mask_mode=mask_mode, seed=seed)
    return factory
