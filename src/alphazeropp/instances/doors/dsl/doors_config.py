"""
Game configuration for the DoorsPDDLLiteEnv.

Implements the same duck-typed interface as ``GameConfig``
(``make_env(n_sites, frozen_states)``, ``max_steps(n_sites)``)
so that ``LeafEvaluator`` can use it without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def compute_doors_derived_params(num_rooms: int, locs_per_room: int = 2) -> dict:
    """Compute n_sites, budget, and horizon from room configuration.

    Returns dict with keys: n_sites, budget, horizon, M, K,
    optimal_nodes, optimal_steps.
    """
    D = num_rooms
    M = D * locs_per_room
    n_sites = M + 2 * D - 1

    # Optimal program: (D-1) PICK rules @ 7 nodes + (D-1) MOVE rules @ 3 nodes + Default @ 2
    optimal_nodes = 10 * (D - 1) + 2
    budget = int(optimal_nodes * 1.5)  # 50% headroom
    budget = budget + (budget % 2)     # round up to even

    optimal_steps = 2 * (D - 1) + 1
    horizon = max(15, optimal_steps * 5)

    return {
        "n_sites": n_sites,
        "budget": budget,
        "horizon": horizon,
        "M": M,
        "K": D - 1,
        "optimal_nodes": optimal_nodes,
        "optimal_steps": optimal_steps,
    }


@dataclass
class DoorsGameConfig:
    """Configuration for the PDDL-faithful doors environment."""

    num_rooms: int = 2
    locs_per_room: int = 2
    loc_room: list[int] | None = None
    key_loc: list[int] | None = None
    key_unlocks: list[int] | None = None
    start_loc: int = 0
    goal_loc: int | None = None
    horizon: int = 15
    step_penalty: float = 0.01
    unlock_bonus: float = 0.1

    def __post_init__(self):
        if self.loc_room is None:
            self.loc_room = [r for r in range(self.num_rooms)
                             for _ in range(self.locs_per_room)]
        if self.key_loc is None:
            # Key k at 2nd location of room k (matches DoorsDirectGame)
            self.key_loc = [k * self.locs_per_room + 1
                            for k in range(self.num_rooms - 1)]
        if self.key_unlocks is None:
            self.key_unlocks = list(range(1, self.num_rooms))
        if self.goal_loc is None:
            self.goal_loc = len(self.loc_room) - 1

    @property
    def M(self) -> int:
        return len(self.loc_room)

    @property
    def D(self) -> int:
        return self.num_rooms

    @property
    def K(self) -> int:
        return self.D - 1

    def obs_size(self) -> int:
        """Observation vector size: M + 2D - 1."""
        return self.M + 2 * self.D - 1

    def max_steps(self, n_sites: int) -> int:
        """Maximum episode steps (used by LeafEvaluator for normalization)."""
        return self.horizon

    def make_env(self, n_sites: int, frozen_states=None):
        """Create a DoorsPDDLLiteEnv with optional frozen states."""
        from alphazeropp.instances.doors.doors_pddl_lite import (
            DoorsPDDLLiteEnv,
        )
        return DoorsPDDLLiteEnv(
            num_rooms=self.num_rooms,
            loc_room=self.loc_room,
            key_loc=self.key_loc,
            key_unlocks=self.key_unlocks,
            start_loc=self.start_loc,
            goal_loc=self.goal_loc,
            horizon=self.horizon,
            step_penalty=self.step_penalty,
            unlock_bonus=self.unlock_bonus,
            frozen_states=frozen_states,
        )

    def is_solved(self, obs: np.ndarray) -> bool:
        """Check if observation represents goal state."""
        return bool(obs[self.goal_loc] == 1.0)


def doors_initial_state(config: DoorsGameConfig) -> np.ndarray:
    """Generate the canonical initial state for a Doors layout.

    State: agent at start_loc, room 0 unlocked, all keys available.
    """
    obs_size = config.obs_size()
    M = config.M
    D = config.D
    K = config.K

    state = np.zeros(obs_size, dtype=np.float32)
    state[config.start_loc] = 1.0        # at start location
    state[M] = 1.0                        # room 0 unlocked
    for k in range(K):
        state[M + D + k] = 1.0            # all keys available
    return state
