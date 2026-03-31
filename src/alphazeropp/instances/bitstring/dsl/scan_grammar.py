"""
Priority-scan grammar for the BitString DSL.

Replaces the size-budget CFG grammar with a permutation-construction grammar
where each derivation step picks the next bit index in a priority scan order.

The resulting permutation σ translates to the decision-list program:
    if IsZero(σ(0)): Flip(σ(0))
    elif IsZero(σ(1)): Flip(σ(1))
    ...
    elif IsZero(σ(N-2)): Flip(σ(N-2))
    else: Flip(σ(N-1))

Provides:
  permutation_to_program()  — convert a permutation to a Program AST
  ScanState                 — partial permutation with remaining indices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Ite, Default, Program,
)


def permutation_to_program(perm: Sequence[int]) -> Program:
    """Convert a permutation of bit indices to a decision-list Program AST.

    The permutation [σ(0), σ(1), ..., σ(N-1)] becomes:
        Ite(IsZero(σ(0)), Flip(σ(0)),
          Ite(IsZero(σ(1)), Flip(σ(1)),
            ...
              Ite(IsZero(σ(N-2)), Flip(σ(N-2)),
                Default(Flip(σ(N-1))))))

    Args:
        perm: A permutation of [0, N). Must contain each index exactly once.

    Returns:
        A complete Program AST node.
    """
    if len(perm) == 0:
        raise ValueError("Permutation must be non-empty")

    # Build from the tail: Default(Flip(last))
    prog: Program = Default(Flip(perm[-1]))

    # Wrap in Ite nodes from second-to-last back to first
    for i in range(len(perm) - 2, -1, -1):
        idx = perm[i]
        prog = Ite(IsZero(idx), Flip(idx), prog)

    return prog


@dataclass(frozen=True)
class ScanState:
    """Partial permutation state for the scan grammar.

    Immutable (frozen dataclass) for safe use with MCTS stash/unstash.

    Attributes:
        prefix: Bit indices chosen so far, in order.
        n_sites: Total number of bit positions (N).
        remaining: Indices not yet chosen.
    """
    prefix: tuple[int, ...]
    n_sites: int
    remaining: frozenset[int]

    @classmethod
    def initial(cls, n_sites: int) -> ScanState:
        """Create the initial state with all indices available."""
        return cls(
            prefix=(),
            n_sites=n_sites,
            remaining=frozenset(range(n_sites)),
        )

    def is_terminal(self) -> bool:
        """True when the full permutation is determined.

        Terminal when all indices are chosen, OR when exactly one remains
        (it is forced and auto-appended by to_program).
        """
        return len(self.remaining) <= 1

    def legal_actions(self) -> list[int]:
        """Return sorted list of choosable bit indices."""
        return sorted(self.remaining)

    def apply(self, action: int) -> ScanState:
        """Choose bit index `action` as the next priority position.

        Args:
            action: A bit index in self.remaining.

        Returns:
            New ScanState with action appended to prefix and removed from remaining.

        Raises:
            ValueError: If action is not in remaining.
        """
        if action not in self.remaining:
            raise ValueError(
                f"Action {action} not in remaining {self.remaining}"
            )
        return ScanState(
            prefix=self.prefix + (action,),
            n_sites=self.n_sites,
            remaining=self.remaining - {action},
        )

    def to_program(self) -> Program:
        """Convert the completed scan state to a Program AST.

        If exactly one index remains, it is appended as the forced last element.

        Raises:
            AssertionError: If more than one index remains (not terminal).
        """
        assert self.is_terminal(), (
            f"Cannot convert non-terminal state (remaining={self.remaining})"
        )
        if len(self.remaining) == 1:
            full_perm = list(self.prefix) + list(self.remaining)
        else:
            full_perm = list(self.prefix)
        return permutation_to_program(full_perm)

    def pretty(self) -> str:
        """Human-readable representation of the partial permutation."""
        slots = []
        for i in range(self.n_sites):
            if i < len(self.prefix):
                slots.append(str(self.prefix[i]))
            else:
                slots.append("_")
        return f"Scan({','.join(slots)})"
