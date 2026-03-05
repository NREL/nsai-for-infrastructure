"""
Game configuration and initial state generation for BitString DSL evaluation.

Extracted from scripts/enumerate_dsl.py for reuse by the leaf evaluator
and MCTS-based program synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import ShapedBitStringGym
from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY


@dataclass
class GameConfig:
    """Bundle of game settings threaded through evaluation functions."""
    bit_flip: bool = True
    sparse_reward: bool = False
    n_ones: int = 2
    potential_fn: Callable[[np.ndarray], int] = None
    potential_name: str = "onemax"

    def __post_init__(self):
        if self.potential_fn is None:
            self.potential_fn = POTENTIAL_REGISTRY[self.potential_name]

    def max_steps(self, n_sites: int) -> int:
        if self.sparse_reward:
            return n_sites - self.n_ones
        return 2 * n_sites

    def make_env(self, n_sites: int, frozen_states=None) -> ShapedBitStringGym:
        """Create a configured ShapedBitStringGym environment."""
        base = BitStringGym(
            n_sites=n_sites,
            bit_flip=self.bit_flip,
            sparse_reward=self.sparse_reward,
            n_ones=self.n_ones,
        )
        return ShapedBitStringGym(
            base, self.potential_fn, "dense_potential",
            frozen_states=frozen_states,
        )


def all_initial_states(n_sites: int, n_ones: int) -> list[np.ndarray]:
    """Return all C(n_sites, n_ones) possible initial bitstring states."""
    states = []
    for ones_indices in combinations(range(n_sites), n_ones):
        s = np.zeros(n_sites, dtype=np.float32)
        s[list(ones_indices)] = 1.0
        states.append(s)
    return states
