"""
AST nodes for the BitString decision-list DSL.

Node types:
  Actions:    Flip(index)
  Conditions: IsZero(index), Not(child), And(left, right)
  Programs:   Ite(cond, action, else_prog), Default(action)

Every node supports:
  - node_count(): exact AST node count (each constructor counts as 1)
  - pretty():     stable human-readable string representation
  - validate(n):  assert all indices are in [0, n)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Type aliases (forward references resolved after class definitions)
# ---------------------------------------------------------------------------

Condition = Union["IsZero", "Not", "And"]
Program = Union["Ite", "Default"]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Flip:
    """Action: flip bit at position *index*. Returns action index *index*."""
    index: int

    def node_count(self) -> int:
        return 1

    def pretty(self) -> str:
        return f"Flip({self.index})"

    def validate(self, n_sites: int) -> None:
        if not (0 <= self.index < n_sites):
            raise ValueError(
                f"Flip index {self.index} out of range [0, {n_sites})"
            )


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IsZero:
    """Condition: true iff state[index] == 0."""
    index: int

    def node_count(self) -> int:
        return 1

    def pretty(self) -> str:
        return f"IsZero({self.index})"

    def validate(self, n_sites: int) -> None:
        if not (0 <= self.index < n_sites):
            raise ValueError(
                f"IsZero index {self.index} out of range [0, {n_sites})"
            )


@dataclass(frozen=True)
class Not:
    """Condition: logical negation of *child*."""
    child: Condition  # type: ignore[valid-type]

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def pretty(self) -> str:
        return f"Not({self.child.pretty()})"

    def validate(self, n_sites: int) -> None:
        self.child.validate(n_sites)


@dataclass(frozen=True)
class And:
    """Condition: conjunction of *left* and *right*."""
    left: Condition   # type: ignore[valid-type]
    right: Condition  # type: ignore[valid-type]

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def pretty(self) -> str:
        return f"And({self.left.pretty()}, {self.right.pretty()})"

    def validate(self, n_sites: int) -> None:
        self.left.validate(n_sites)
        self.right.validate(n_sites)


# ---------------------------------------------------------------------------
# Programs (decision list)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ite:
    """If *cond* then *action*, else continue to *else_prog*.

    First-match semantics: the first Ite whose condition is true fires.
    """
    cond: Condition     # type: ignore[valid-type]
    action: Flip
    else_prog: Program  # type: ignore[valid-type]

    def node_count(self) -> int:
        return (1
                + self.cond.node_count()
                + self.action.node_count()
                + self.else_prog.node_count())

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [
            f"{pad}if {self.cond.pretty()}:",
            f"{pad}  {self.action.pretty()}",
        ]
        if isinstance(self.else_prog, Ite):
            # Chain as elif
            lines.append(f"{pad}el{self.else_prog.pretty(indent)}")
        else:
            lines.append(f"{pad}else:")
            lines.append(f"{pad}  {self.else_prog.pretty()}")
        return "\n".join(lines)

    def validate(self, n_sites: int) -> None:
        self.cond.validate(n_sites)
        self.action.validate(n_sites)
        self.else_prog.validate(n_sites)


@dataclass(frozen=True)
class Default:
    """Always returns *action*. Terminal case of a decision list."""
    action: Flip

    def node_count(self) -> int:
        return 1 + self.action.node_count()

    def pretty(self, indent: int = 0) -> str:
        return f"{'  ' * indent}{self.action.pretty()}"

    def validate(self, n_sites: int) -> None:
        self.action.validate(n_sites)
