"""
Derivation engine for the BitString DSL size-budget grammar.

Provides:
  DerivationState            — partial AST with holes, canonical expansion
  Production                 — a grammar production (hole → partial subtree)
  format_derivation_trace()  — step-by-step derivation trace
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

from alphazeropp.instances.bitstring.dsl.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Condition, Program,
)
from alphazeropp.instances.bitstring.dsl.budget_grammar import (
    ProgramHole, ConditionHole,
    enumerate_conditions, count_conditions,
    enumerate_programs, count_programs,
    _ccnn,
)


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Production:
    """A grammar production that expands a hole into a (partial) subtree.

    Attributes:
        hole_kind: "P" (ProgramHole) or "C" (ConditionHole).
        hole_budget: budget of the hole being expanded.
        result: the replacement subtree (may contain new holes).
        label: human-readable description (e.g., "P(5) -> Ite(C(1), Flip(0), P(2))").
    """
    hole_kind: str
    hole_budget: int
    result: Any
    label: str


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def _find_leftmost_hole(node: Any) -> Optional[Union[ProgramHole, ConditionHole]]:
    """Find the leftmost hole in a partial AST via preorder traversal."""
    if isinstance(node, (ProgramHole, ConditionHole)):
        return node
    if isinstance(node, Ite):
        for child in (node.cond, node.action, node.else_prog):
            found = _find_leftmost_hole(child)
            if found is not None:
                return found
    elif isinstance(node, Default):
        return _find_leftmost_hole(node.action)
    elif isinstance(node, Not):
        return _find_leftmost_hole(node.child)
    elif isinstance(node, And):
        found = _find_leftmost_hole(node.left)
        if found is not None:
            return found
        return _find_leftmost_hole(node.right)
    # Flip, IsZero: leaf nodes, no holes
    return None


def _count_holes(node: Any) -> int:
    """Count all holes in a partial AST."""
    if isinstance(node, (ProgramHole, ConditionHole)):
        return 1
    if isinstance(node, Ite):
        return (_count_holes(node.cond)
                + _count_holes(node.action)
                + _count_holes(node.else_prog))
    if isinstance(node, Default):
        return _count_holes(node.action)
    if isinstance(node, Not):
        return _count_holes(node.child)
    if isinstance(node, And):
        return _count_holes(node.left) + _count_holes(node.right)
    return 0  # Flip, IsZero


def _replace_leftmost_hole(node: Any, replacement: Any) -> Any:
    """Replace the leftmost hole with `replacement`, returning a new tree.

    Since AST nodes are frozen dataclasses, we reconstruct the path
    from the root to the replaced hole.
    """
    if isinstance(node, (ProgramHole, ConditionHole)):
        return replacement
    if isinstance(node, Ite):
        # Try cond first
        if _count_holes(node.cond) > 0:
            new_cond = _replace_leftmost_hole(node.cond, replacement)
            return Ite(new_cond, node.action, node.else_prog)
        # Then action (shouldn't have holes, but be safe)
        if _count_holes(node.action) > 0:
            new_action = _replace_leftmost_hole(node.action, replacement)
            return Ite(node.cond, new_action, node.else_prog)
        # Then else_prog
        if _count_holes(node.else_prog) > 0:
            new_else = _replace_leftmost_hole(node.else_prog, replacement)
            return Ite(node.cond, node.action, new_else)
    if isinstance(node, Default):
        if _count_holes(node.action) > 0:
            return Default(_replace_leftmost_hole(node.action, replacement))
    if isinstance(node, Not):
        if _count_holes(node.child) > 0:
            return Not(_replace_leftmost_hole(node.child, replacement))
    if isinstance(node, And):
        if _count_holes(node.left) > 0:
            return And(_replace_leftmost_hole(node.left, replacement),
                       node.right)
        if _count_holes(node.right) > 0:
            return And(node.left,
                       _replace_leftmost_hole(node.right, replacement))
    return node  # no holes found


def _partial_pretty(node: Any) -> str:
    """Pretty-print a partial AST, showing holes as [P:k] or [C:k]."""
    if isinstance(node, ProgramHole):
        return f"[P:{node.budget}]"
    if isinstance(node, ConditionHole):
        return f"[C:{node.budget}]"
    if isinstance(node, Flip):
        return node.pretty()
    if isinstance(node, IsZero):
        return node.pretty()
    if isinstance(node, Not):
        return f"Not({_partial_pretty(node.child)})"
    if isinstance(node, And):
        return f"And({_partial_pretty(node.left)}, {_partial_pretty(node.right)})"
    if isinstance(node, Default):
        return f"Default({_partial_pretty(node.action)})"
    if isinstance(node, Ite):
        return (f"Ite({_partial_pretty(node.cond)}, "
                f"{_partial_pretty(node.action)}, "
                f"{_partial_pretty(node.else_prog)})")
    return repr(node)


# ---------------------------------------------------------------------------
# Production generation
# ---------------------------------------------------------------------------

def _program_productions(
    budget: int, n_sites: int, mode: str = "exact",
) -> list[Production]:
    """Generate all productions for a ProgramHole with given budget.

    In exact mode, P(k) -> Default(Flip(j)) only at k == 2, and Ite
    expansions skip else_budgets with zero exact completions.

    In max mode, P(k) -> Default(Flip(j)) at any k >= 2 (early
    termination), and all Ite expansions are valid (else_budget >= 2
    is guaranteed by loop bounds, and any P(m >= 2) can terminate).
    """
    prods: list[Production] = []

    # Terminate: P(k) → Default(Flip(j))
    # Exact: only at k == 2.  Max: at any k >= 2.
    if (mode == "exact" and budget == 2) or (mode == "max" and budget >= 2):
        for j in range(n_sites):
            result = Default(Flip(j))
            prods.append(Production(
                hole_kind="P", hole_budget=budget,
                result=result,
                label=f"P({budget}) -> Default(Flip({j}))",
            ))

    # Expand: P(k) → Ite(C(i), Flip(j), P(k-2-i))  for k >= 5
    if budget >= 5:
        for i in range(1, budget - 3):  # i in [1, k-4]
            else_budget = budget - 2 - i
            if mode == "exact" and count_programs(n_sites, else_budget) == 0:
                continue  # Skip dead-end budgets (e.g., 3 and 4)
            # Max mode: else_budget >= 2 always (loop bounds guarantee it).
            for j in range(n_sites):
                result = Ite(ConditionHole(i), Flip(j), ProgramHole(else_budget))
                prods.append(Production(
                    hole_kind="P", hole_budget=budget,
                    result=result,
                    label=f"P({budget}) -> Ite(C({i}), Flip({j}), P({else_budget}))",
                ))

    return prods


def _condition_productions(
    budget: int, n_sites: int, parent_is_not: bool = False,
    mode: str = "exact",
) -> list[Production]:
    """Generate canonical productions for a ConditionHole with given budget.

    Canonicalization rules:
      1. **Double-negation ban**: When *parent_is_not* is True (this hole is
         the child of a ``Not``), suppress the ``Not(C(k-1))`` production.
      2. **And commutativity**: For ``And(C(i), C(j))``, restrict to
         ``i <= j`` (left budget <= right budget).

    In max mode, C(k) -> IsZero(j) at any k >= 1 (early termination),
    and the _ccnn dead-end guard for Not is bypassed (C(k-1,
    parent_is_not=True) can always early-terminate to IsZero(j)).
    """
    prods: list[Production] = []

    # Terminate: C(k) → IsZero(j)
    # Exact: only at k == 1.  Max: at any k >= 1.
    if (mode == "exact" and budget == 1) or (mode == "max" and budget >= 1):
        for j in range(n_sites):
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=IsZero(j),
                label=f"C({budget}) -> IsZero({j})",
            ))

    # C(k) → Not(C(k-1))  — only if parent is NOT a Not.
    # Exact: also guard with _ccnn to prevent dead-end C(k-1, parent_is_not).
    # Max: _ccnn guard unnecessary (child can early-terminate to IsZero).
    if budget >= 2 and not parent_is_not:
        if mode == "max" or _ccnn(n_sites, budget - 1) > 0:
            child_budget = budget - 1
            result = Not(ConditionHole(child_budget, parent_is_not=True))
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=result,
                label=f"C({budget}) -> Not(C({child_budget}))",
            ))

    # C(k) → And(C(i), C(k-1-i))  — canonical: i <= k-1-i
    if budget >= 3:
        for i in range(1, (budget - 1) // 2 + 1):
            right_budget = budget - 1 - i
            result = And(ConditionHole(i), ConditionHole(right_budget))
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=result,
                label=f"C({budget}) -> And(C({i}), C({right_budget}))",
            ))

    return prods


# ---------------------------------------------------------------------------
# DerivationState
# ---------------------------------------------------------------------------

@dataclass
class DerivationState:
    """Partial AST with holes, supporting canonical leftmost-hole expansion.

    Usage::

        state = DerivationState.initial(budget=5)
        while not state.is_terminal():
            prods = state.legal_productions(n_sites=3)
            state = state.apply(prods[0])  # pick first production
        program = state.to_program()
    """
    root: Any

    @classmethod
    def initial(cls, budget: int) -> DerivationState:
        """Create initial state with a single ProgramHole."""
        return cls(root=ProgramHole(budget))

    def is_terminal(self) -> bool:
        """True if no holes remain in the partial AST."""
        return _count_holes(self.root) == 0

    def leftmost_hole(self) -> Optional[Union[ProgramHole, ConditionHole]]:
        """Find the leftmost hole via preorder traversal."""
        return _find_leftmost_hole(self.root)

    def legal_productions(
        self, n_sites: int, mode: str = "exact",
    ) -> list[Production]:
        """Get all legal productions for the leftmost hole."""
        hole = self.leftmost_hole()
        if hole is None:
            return []
        if isinstance(hole, ProgramHole):
            return _program_productions(hole.budget, n_sites, mode)
        elif isinstance(hole, ConditionHole):
            return _condition_productions(
                hole.budget, n_sites, hole.parent_is_not, mode,
            )
        return []

    def apply(self, production: Production) -> DerivationState:
        """Apply a production to the leftmost hole, returning a new state."""
        new_root = _replace_leftmost_hole(self.root, production.result)
        return DerivationState(root=new_root)

    def to_program(self) -> Program:
        """Convert a terminal derivation to a Program AST node.

        Raises AssertionError if holes remain.
        """
        assert self.is_terminal(), "Cannot convert non-terminal derivation"
        return self.root

    def pretty(self) -> str:
        """Pretty-print the partial AST, showing holes."""
        return _partial_pretty(self.root)

    def hole_count(self) -> int:
        """Count remaining holes."""
        return _count_holes(self.root)


# ---------------------------------------------------------------------------
# Derivation enumeration (via DerivationState)
# ---------------------------------------------------------------------------

def enumerate_via_derivation(
    n_sites: int, budget: int,
) -> list[Program]:
    """Enumerate all terminal programs by expanding DerivationStates.

    Uses DFS with canonical leftmost-hole expansion. This produces the
    same set of programs as enumerate_programs(), but via the derivation
    machinery (useful for testing derivation correctness).
    """
    results: list[Program] = []

    def _dfs(state: DerivationState) -> None:
        if state.is_terminal():
            results.append(state.to_program())
            return
        for prod in state.legal_productions(n_sites):
            _dfs(state.apply(prod))

    _dfs(DerivationState.initial(budget))
    return results


# ---------------------------------------------------------------------------
# Derivation trace formatting
# ---------------------------------------------------------------------------

def format_derivation_trace(
    states: list[DerivationState],
    productions: list[Production],
) -> str:
    """Format a derivation as a step-by-step trace.

    Args:
        states: list of DerivationStates (length = len(productions) + 1).
        productions: list of productions applied at each step.
    """
    lines: list[str] = []
    budget = None
    if states and isinstance(states[0].root, ProgramHole):
        budget = states[0].root.budget
    lines.append(
        f"=== Derivation Trace"
        f"{f' (budget={budget})' if budget is not None else ''} ==="
    )
    lines.append("")

    for step, state in enumerate(states):
        tag = "  [TERMINAL]" if state.is_terminal() else ""
        lines.append(f"Step {step}: {state.pretty()}{tag}")

        if state.is_terminal():
            prog = state.to_program()
            lines.append(f"  node_count = {prog.node_count()}")
            lines.append(f"  Pretty:")
            for pline in prog.pretty().split("\n"):
                lines.append(f"    {pline}")
        elif step < len(productions):
            hole = state.leftmost_hole()
            lines.append(f"  Leftmost hole: {hole.pretty()}")
            lines.append(f"  Apply: {productions[step].label}")

        lines.append("")

    return "\n".join(lines)


def trace_first_derivation(
    n_sites: int, budget: int,
) -> str:
    """Generate and format the trace for the first derivation at a given budget.

    Useful for quick demonstration of the derivation machinery.
    """
    states: list[DerivationState] = []
    productions: list[Production] = []

    state = DerivationState.initial(budget)
    states.append(state)

    while not state.is_terminal():
        prods = state.legal_productions(n_sites)
        if not prods:
            break
        prod = prods[0]  # take the first (canonical) production
        productions.append(prod)
        state = state.apply(prod)
        states.append(state)

    return format_derivation_trace(states, productions)
