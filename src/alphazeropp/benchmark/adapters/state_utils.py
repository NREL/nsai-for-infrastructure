"""Compact state ID utilities for tabular methods."""

from __future__ import annotations

import numpy as np

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


def obs_to_state_id(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> tuple[int, int]:
    """Convert observation to compact (agent_loc, unlock_prefix) tuple.

    Exploits monotone unlocking: rooms 0..p-1 are open, rooms p..D-1 locked.
    Total distinct states = locs_per_room * D * (D+1) / 2.
    """
    agent_loc = int(np.argmax(obs[:env.M]))
    unlock_prefix = sum(int(obs[env._unlocked_offset + r]) for r in range(env.D))
    return (agent_loc, unlock_prefix)
