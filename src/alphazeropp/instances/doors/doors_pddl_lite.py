"""
DoorsPDDLLiteEnv: A PDDL-faithful doors environment for program synthesis.

Based on the PDDLGym "doors" domain where:
  - Rooms r=0..D-1.  Room 0 starts unlocked; others locked.
  - Locations l=0..M-1.  Each location belongs to a room (loc_room[l]).
  - Keys k=0..D-2.  Key k is at key_loc[k] and unlocks key_unlocks[k].
  - MOVE_TO(l): succeeds iff unlocked[loc_room[l]].
  - PICK(k): succeeds iff at_loc[key_loc[k]] AND key_available[k].

State vector (float32, size M + 2D - 1):
    at_loc[0..M-1]           one-hot location
    unlocked[0..D-1]         room lock status
    key_available[0..D-2]    key availability

Actions (Discrete, size obs_size for n_sites alignment):
    [0..M-1]                 MOVE_TO(l)
    [M..M+D-2]               PICK(k)
    [M+D-1]                  NOOP
    [M+D..obs_size-1]        invalid -> NOOP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np


# ---------------------------------------------------------------------------
# Preset layouts
# ---------------------------------------------------------------------------

_LAYOUT_D2 = dict(
    num_rooms=2,
    loc_room=[0, 0, 1, 1],
    key_loc=[1],
    key_unlocks=[1],
    start_loc=0,
    goal_loc=3,
    horizon=15,
)

_LAYOUT_D3 = dict(
    num_rooms=3,
    loc_room=[0, 0, 1, 1, 2, 2],
    key_loc=[1, 2],
    key_unlocks=[1, 2],
    start_loc=0,
    goal_loc=5,
    horizon=25,
)


# ---------------------------------------------------------------------------
# Feature / Action spec helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """Maps an observation index to a semantic predicate."""
    index: int
    name: str
    type: str   # "location", "room", "key"
    param: int


@dataclass(frozen=True)
class ActionSpec:
    """Maps an action index to a semantic action."""
    index: int
    name: str
    type: str   # "move", "pick", "noop", "invalid"
    param: int  # location/key index, or -1


# ---------------------------------------------------------------------------
# DoorsPDDLLiteEnv
# ---------------------------------------------------------------------------

class DoorsPDDLLiteEnv(gym.Env):
    """PDDL-faithful doors environment with fixed-size numpy observations.

    Compatible with the AlphaZero derivation pipeline: Gymnasium API,
    ``.state`` property for ``run_policy_episode``, frozen-state cycling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_rooms: int = 2,
        loc_room: list[int] | None = None,
        key_loc: list[int] | None = None,
        key_unlocks: list[int] | None = None,
        start_loc: int = 0,
        goal_loc: int | None = None,
        horizon: int = 20,
        step_penalty: float = 0.01,
        unlock_bonus: float = 0.1,
        frozen_states: Sequence[np.ndarray] | None = None,
    ):
        super().__init__()
        self.D = num_rooms
        self.horizon = horizon
        self.step_penalty = step_penalty
        self.unlock_bonus = unlock_bonus

        # Layout
        if loc_room is None:
            loc_room = [r for r in range(num_rooms) for _ in range(2)]
        self.loc_room = list(loc_room)
        self.M = len(self.loc_room)
        self.K = self.D - 1  # number of keys

        if key_loc is None:
            key_loc = list(range(1, self.K + 1))
        self.key_loc = list(key_loc)

        if key_unlocks is None:
            key_unlocks = list(range(1, self.K + 1))
        self.key_unlocks = list(key_unlocks)

        self.start_loc = start_loc
        self.goal_loc = goal_loc if goal_loc is not None else self.M - 1

        # Dimensions
        self._obs_size = self.M + 2 * self.D - 1
        self._action_count = self.M + self.K + 1  # M moves + K picks + 1 noop

        # Gym spaces (action space padded to obs_size for n_sites alignment)
        self.observation_space = spaces.MultiBinary(self._obs_size)
        self.action_space = spaces.Discrete(self._obs_size)

        # Frozen states
        self._frozen_states = list(frozen_states) if frozen_states else None
        self._frozen_idx = 0

        # Internal state
        self._state = np.zeros(self._obs_size, dtype=np.float32)
        self.step_count = 0
        self._terminated = False
        self._truncated = False

        # Precompute index offsets
        self._unlocked_offset = self.M
        self._key_offset = self.M + self.D

    # -- Properties --

    @property
    def n_sites(self) -> int:
        return self._obs_size

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy().astype(np.float32)

    # -- Gym API --

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._terminated = False
        self._truncated = False

        if self._frozen_states is not None:
            self._state = self._frozen_states[self._frozen_idx].copy().astype(
                np.float32
            )
            self._frozen_idx = (self._frozen_idx + 1) % len(self._frozen_states)
        else:
            self._state = np.zeros(self._obs_size, dtype=np.float32)
            self._state[self.start_loc] = 1.0
            self._state[self._unlocked_offset] = 1.0  # room 0 unlocked
            for k in range(self.K):
                self._state[self._key_offset + k] = 1.0

        return self._state.copy(), {}

    def step(self, action: int):
        assert not self._terminated and not self._truncated, "Episode is done"
        self.step_count += 1
        reward = -self.step_penalty
        info: dict = {}

        if 0 <= action < self.M:
            # MOVE_TO(l)
            l = action
            room = self.loc_room[l]
            if self._state[self._unlocked_offset + room] == 1.0:
                # Clear old location, set new
                self._state[:self.M] = 0.0
                self._state[l] = 1.0
        elif self.M <= action < self.M + self.K:
            # PICK(k)
            k = action - self.M
            key_l = self.key_loc[k]
            if (self._state[key_l] == 1.0
                    and self._state[self._key_offset + k] == 1.0):
                self._state[self._key_offset + k] = 0.0
                target_room = self.key_unlocks[k]
                if self._state[self._unlocked_offset + target_room] == 0.0:
                    self._state[self._unlocked_offset + target_room] = 1.0
                    reward += self.unlock_bonus
                    info["unlocked_room"] = target_room
        # else: NOOP or invalid -> do nothing

        # Check terminal
        if self._state[self.goal_loc] == 1.0:
            self._terminated = True
            reward += 1.0
        elif self.step_count >= self.horizon:
            self._truncated = True

        obs = self._state.copy()
        return obs, reward, self._terminated, self._truncated, info

    def reset_frozen_index(self):
        """Reset the frozen-state cycling index."""
        self._frozen_idx = 0

    # -- Solved check --

    def is_solved(self, obs: np.ndarray) -> bool:
        """Check whether observation represents the goal state."""
        return bool(obs[self.goal_loc] == 1.0)

    # -- Feature / Action specs --

    @property
    def feature_spec(self) -> list[FeatureSpec]:
        specs = []
        for l in range(self.M):
            specs.append(FeatureSpec(l, f"AtLoc({l})", "location", l))
        for r in range(self.D):
            idx = self._unlocked_offset + r
            specs.append(FeatureSpec(idx, f"Unlocked({r})", "room", r))
        for k in range(self.K):
            idx = self._key_offset + k
            specs.append(FeatureSpec(idx, f"KeyAvail({k})", "key", k))
        return specs

    @property
    def action_spec(self) -> list[ActionSpec]:
        specs = []
        for l in range(self.M):
            specs.append(ActionSpec(l, f"MOVE_TO({l})", "move", l))
        for k in range(self.K):
            idx = self.M + k
            specs.append(ActionSpec(idx, f"PICK({k})", "pick", k))
        specs.append(ActionSpec(self.M + self.K, "NOOP", "noop", -1))
        for i in range(self.M + self.K + 1, self._obs_size):
            specs.append(ActionSpec(i, f"INVALID({i})", "invalid", -1))
        return specs

    # -- Preset layouts --

    @classmethod
    def make_d2(cls, **kwargs) -> DoorsPDDLLiteEnv:
        """D=2, M=4 preset: 2 rooms, 1 key, goal in room 1."""
        kw = dict(_LAYOUT_D2)
        kw.update(kwargs)
        return cls(**kw)

    @classmethod
    def make_d3(cls, **kwargs) -> DoorsPDDLLiteEnv:
        """D=3, M=6 preset: 3 rooms, 2 keys, sequential unlock to goal."""
        kw = dict(_LAYOUT_D3)
        kw.update(kwargs)
        return cls(**kw)
