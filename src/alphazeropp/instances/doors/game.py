"""DoorsDirectGame: EnvGame wrapper for direct-play AlphaZero on DoorsPDDLLiteEnv."""

import numpy as np

from alphazeropp.core.game import EnvGame
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


class DoorsDirectGame(EnvGame):
    """Wraps DoorsPDDLLiteEnv for direct-play AlphaZero.

    Provides action masking, fast stash/unstash (avoids deepcopy),
    and lightweight clone.
    """

    def __init__(self, env=None, num_rooms=2, locs_per_room=2,
                 horizon=None, use_precondition_mask=False,
                 step_penalty=0.01, unlock_bonus=0.1):
        if env is None:
            if num_rooms < 2:
                raise ValueError(f"num_rooms must be >= 2, got {num_rooms}")
            if horizon is None:
                optimal_steps = 2 * (num_rooms - 1) + 1
                horizon = max(15, optimal_steps * 5)
            loc_room = [r for r in range(num_rooms) for _ in range(locs_per_room)]
            # Place key k at 2nd location of room k for a proper sequential chain
            key_loc = [k * locs_per_room + 1 for k in range(num_rooms - 1)]
            env = DoorsPDDLLiteEnv(
                num_rooms=num_rooms,
                loc_room=loc_room,
                key_loc=key_loc,
                horizon=horizon,
                step_penalty=step_penalty,
                unlock_bonus=unlock_bonus,
            )
        super().__init__(env)
        self._num_rooms = num_rooms
        self._locs_per_room = locs_per_room
        self._use_precondition_mask = use_precondition_mask
        # Precompute structural mask: real actions valid, padding invalid
        n_real = env.M + env.K + 1  # MOVE_TO + PICK + NOOP
        n_total = env.action_space.n
        self._structural_mask = np.array(
            [True] * n_real + [False] * (n_total - n_real), dtype=bool
        )

    @property
    def hashable_obs(self):
        """Include step_count in hash so MCTS distinguishes the same obs at
        different time steps (needed because horizon truncation makes
        terminal status step-dependent)."""
        return (self.obs.tobytes(), self.step_count)

    def get_action_mask(self):
        if not self._use_precondition_mask:
            return self._structural_mask

        env = self.env
        state = env._state
        mask = np.zeros(env.action_space.n, dtype=bool)

        # MOVE_TO(l): valid iff room containing l is unlocked
        for l in range(env.M):
            room = env.loc_room[l]
            mask[l] = state[env._unlocked_offset + room] == 1.0

        # PICK(k): valid iff at key location AND key available
        for k in range(env.K):
            key_l = env.key_loc[k]
            mask[env.M + k] = (
                state[key_l] == 1.0
                and state[env._key_offset + k] == 1.0
            )

        # NOOP: always valid
        mask[env.M + env.K] = True

        return mask

    # -- Fast stash/unstash (avoids deepcopy) --

    def stash_state(self):
        """Save mutable state as a lightweight tuple."""
        return (
            self.obs.copy() if self.obs is not None else None,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
            self.env._state.copy(),
            self.env.step_count,
            self.env._terminated,
            self.env._truncated,
            self.env._frozen_idx,
        )

    def unstash_state(self, state):
        """Restore mutable state from a stashed tuple."""
        (
            self.obs, self.reward, self.terminated, self.truncated,
            self.info, self.step_count,
            env_state, env_sc, env_term, env_trunc, env_frozen_idx,
        ) = state
        self.env._state = env_state
        self.env.step_count = env_sc
        self.env._terminated = env_term
        self.env._truncated = env_trunc
        self.env._frozen_idx = env_frozen_idx
        return self

    def clone(self):
        """Return a lightweight independent copy."""
        new_env = DoorsPDDLLiteEnv(
            num_rooms=self._num_rooms,
            loc_room=self.env.loc_room,
            key_loc=self.env.key_loc,
            key_unlocks=self.env.key_unlocks,
            start_loc=self.env.start_loc,
            goal_loc=self.env.goal_loc,
            horizon=self.env.horizon,
            step_penalty=self.env.step_penalty,
            unlock_bonus=self.env.unlock_bonus,
        )
        new_game = DoorsDirectGame(
            new_env, num_rooms=self._num_rooms,
            locs_per_room=self._locs_per_room,
            use_precondition_mask=self._use_precondition_mask,
        )
        new_game.unstash_state(self.stash_state())
        return new_game
