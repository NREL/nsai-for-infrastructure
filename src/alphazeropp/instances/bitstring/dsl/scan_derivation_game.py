"""
ScanDerivationGame: A single-player game for MCTS-guided scan-order synthesis.

Casts permutation construction as a Game (compatible with core MCTS) where:
  - States are partial permutations (which bit indices chosen so far)
  - Actions are bit indices to add to the scan order
  - Terminal reward = leaf evaluation of the completed scan program
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.instances.bitstring.dsl.leaf_evaluator import LeafEvaluator
from alphazeropp.instances.bitstring.dsl.scan_grammar import ScanState


class ScanDerivationGame(Game):
    """Single-player game where actions choose the next bit in a scan order.

    At each step the agent picks a bit index from the remaining set.
    When all indices are assigned (or only one remains, which is forced),
    the permutation is converted to a decision-list program and evaluated
    by the leaf evaluator.

    Observation: flat float32 array of length 2 * n_sites.
      - First n_sites slots: prefix values (chosen indices), padded with -1.
      - Last n_sites slots: remaining mask (1.0 if available, 0.0 if taken).
    """

    def __init__(
        self,
        n_sites: int,
        leaf_evaluator: LeafEvaluator,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.leaf_evaluator = leaf_evaluator

        self.action_space = spaces.Discrete(n_sites)
        self.observation_space = spaces.Box(
            low=-1.0, high=float(n_sites),
            shape=(2 * n_sites,), dtype=np.float32,
        )

        self._scan_state: ScanState | None = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._scan_state = ScanState.initial(self.n_sites)
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._scan_state = self._scan_state.apply(action)

        info: dict[str, Any] = {"action": action}

        if self._scan_state.is_terminal():
            program = self._scan_state.to_program()
            reward = self.leaf_evaluator(program)
            info["program"] = program
            return self._encode_obs(), reward, True, False, info

        return self._encode_obs(), 0.0, False, False, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_sites, dtype=bool)
        for idx in self._scan_state.remaining:
            mask[idx] = True
        return mask

    @property
    def hashable_obs(self) -> str:
        return self._scan_state.pretty()

    def get_program(self):
        """Return the completed program, or None if not terminal."""
        if self._scan_state is not None and self._scan_state.is_terminal():
            return self._scan_state.to_program()
        return None

    # -- Stash / Unstash (lightweight, ScanState is frozen) --

    def stash_state(self) -> tuple:
        return (
            self._scan_state,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        )

    def unstash_state(self, state: tuple):
        (
            self._scan_state,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> "ScanDerivationGame":
        new = ScanDerivationGame(self.n_sites, self.leaf_evaluator)
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new

    # -- Observation encoding --

    def _encode_obs(self) -> np.ndarray:
        obs = np.full(2 * self.n_sites, -1.0, dtype=np.float32)
        # First N slots: prefix
        for i, idx in enumerate(self._scan_state.prefix):
            obs[i] = float(idx)
        # Last N slots: remaining mask
        for idx in range(self.n_sites):
            obs[self.n_sites + idx] = (
                1.0 if idx in self._scan_state.remaining else 0.0
            )
        return obs
