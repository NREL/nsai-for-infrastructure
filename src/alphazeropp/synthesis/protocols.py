"""
Protocols (structural typing) for the synthesis module.

Defines the interface that domain-specific game configs must satisfy
to work with LeafEvaluator and the derivation game framework.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class DSLGameConfig(Protocol):
    """Interface that domain game configs must satisfy for LeafEvaluator.

    Implementations include:
      - ``instances.bitstring.dsl.game_config.GameConfig``
      - ``instances.doors.dsl.doors_config.DoorsGameConfig``
    """

    def make_env(self, n_sites: int,
                 frozen_states: Sequence[np.ndarray] | None = None):
        """Create a Gymnasium-compatible environment for DSL evaluation."""
        ...

    def max_steps(self, n_sites: int) -> int:
        """Maximum episode steps (used for normalization)."""
        ...
