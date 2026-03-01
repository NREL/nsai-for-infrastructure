"""
Size-budget grammar for enumerating BitString DSL programs.

Provides:
  ProgramHole, ConditionHole     — hole placeholders for partial derivations
  enumerate_programs(N, budget)  — enumerate ALL programs with node_count == budget
  enumerate_conditions(N, budget)— enumerate ALL conditions with node_count == budget
  count_programs(N, budget)      — count without generating
  count_conditions(N, budget)    — count without generating
  format_grammar_summary(N, max) — human-readable grammar summary
  format_grammar_debug(N, max)   — debug/consistency report
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Union

from alphazeropp.instances.bitstring.dsl.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Condition, Program,
)


# ---------------------------------------------------------------------------
# Hole types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgramHole:
    """Placeholder for a Program subtree with exactly `budget` AST nodes."""
    budget: int

    def pretty(self) -> str:
        return f"[P:{self.budget}]"


@dataclass(frozen=True)
class ConditionHole:
    """Placeholder for a Condition subtree with exactly `budget` AST nodes.

    Attributes:
        budget: Exact number of AST nodes the condition must have.
        parent_is_not: True when this hole is the direct child of a ``Not``
            node.  Used by ``_condition_productions`` to suppress the
            ``Not(C(k-1))`` production and prevent double-negation.
    """
    budget: int
    parent_is_not: bool = False

    def pretty(self) -> str:
        return f"[C:{self.budget}]"


# ---------------------------------------------------------------------------
# Counting (efficient, no allocation)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def count_conditions(n_sites: int, budget: int) -> int:
    """Count conditions with exactly `budget` AST nodes."""
    if budget < 1:
        return 0
    total = 0
    # C(1) → IsZero(j)
    if budget == 1:
        total += n_sites
    # C(k) → Not(C(k-1))  for k >= 2
    if budget >= 2:
        total += count_conditions(n_sites, budget - 1)
    # C(k) → And(C(i), C(k-1-i))  for k >= 3, i in [1, k-2]
    if budget >= 3:
        for i in range(1, budget - 1):
            total += (count_conditions(n_sites, i)
                      * count_conditions(n_sites, budget - 1 - i))
    return total


@functools.lru_cache(maxsize=None)
def count_programs(n_sites: int, budget: int) -> int:
    """Count programs with exactly `budget` AST nodes."""
    if budget < 2:
        return 0
    total = 0
    # P(2) → Default(Flip(j))
    if budget == 2:
        total += n_sites
    # P(k) → Ite(C(i), Flip(j), P(k-2-i))  for k >= 5
    if budget >= 5:
        for i in range(1, budget - 3):  # i in [1, k-4]
            else_budget = budget - 2 - i
            total += (count_conditions(n_sites, i)
                      * n_sites
                      * count_programs(n_sites, else_budget))
    return total


# ---------------------------------------------------------------------------
# Canonical counting (no double-negation, canonical And ordering)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _ccnn(n_sites: int, budget: int) -> int:
    """Count canonical conditions at *budget* that are NOT Not-prefixed.

    Needed by the double-negation ban: inside a Not, only non-Not children
    are allowed.  The Not-prefixed count at budget k equals ``_ccnn(n, k-1)``
    (each such condition is ``Not(child)`` where *child* is a non-Not
    canonical condition at k-1).  Therefore::

        _ccnn(n, k) = cc(n, k) - _ccnn(n, k-1)
    """
    if budget < 1:
        return 0
    if budget == 1:
        return n_sites  # all IsZero, none Not-prefixed
    return count_canonical_conditions(n_sites, budget) - _ccnn(n_sites, budget - 1)


@functools.lru_cache(maxsize=None)
def count_canonical_conditions(n_sites: int, budget: int) -> int:
    """Count conditions with no double-negation and canonical And ordering.

    Rules applied:
      1. Double-negation ban: ``Not(Not(...))`` is forbidden.  The Not
         production at budget k contributes ``_ccnn(n, k-1)`` (only non-Not
         children) instead of ``cc(n, k-1)``.
      2. And commutativity: ``And(C(i), C(j))`` requires ``i <= j``.
         When ``i < j``: ``cc(i) * cc(j)`` ordered pairs.
         When ``i == j``: ``cc(i) * (cc(i)+1) / 2`` unordered pairs with
         repetition (multiset coefficient).
    """
    if budget < 1:
        return 0
    if budget == 1:
        return n_sites  # IsZero(0..n-1)

    total = 0

    # Not(non-Not child at k-1)
    if budget >= 2:
        total += _ccnn(n_sites, budget - 1)

    # And(C(i), C(j)) with i <= j, i + j = k - 1
    if budget >= 3:
        for i in range(1, (budget - 1) // 2 + 1):
            j = budget - 1 - i
            ci = count_canonical_conditions(n_sites, i)
            cj = count_canonical_conditions(n_sites, j)
            if i < j:
                total += ci * cj
            else:  # i == j
                total += ci * (ci + 1) // 2

    return total


@functools.lru_cache(maxsize=None)
def count_canonical_programs(n_sites: int, budget: int) -> int:
    """Count programs whose conditions are all canonical.

    Same structure as ``count_programs`` but uses
    ``count_canonical_conditions`` for condition subtrees.
    """
    if budget < 2:
        return 0
    total = 0
    if budget == 2:
        total += n_sites
    if budget >= 5:
        for i in range(1, budget - 3):  # i in [1, k-4]
            else_budget = budget - 2 - i
            total += (count_canonical_conditions(n_sites, i)
                      * n_sites
                      * count_canonical_programs(n_sites, else_budget))
    return total


# ---------------------------------------------------------------------------
# Enumeration (generates actual AST nodes)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def enumerate_conditions(n_sites: int, budget: int) -> tuple[Condition, ...]:
    """Enumerate ALL conditions with node_count == budget.

    Returns a tuple (for hashability/caching). Production order:
    IsZero (ascending j) → Not → And (ascending i, then children).
    """
    if budget < 1:
        return ()
    results: list[Condition] = []

    # C(1) → IsZero(j)
    if budget == 1:
        for j in range(n_sites):
            results.append(IsZero(j))

    # C(k) → Not(C(k-1))
    if budget >= 2:
        for child in enumerate_conditions(n_sites, budget - 1):
            results.append(Not(child))

    # C(k) → And(C(i), C(k-1-i))
    if budget >= 3:
        for i in range(1, budget - 1):
            for left in enumerate_conditions(n_sites, i):
                for right in enumerate_conditions(n_sites, budget - 1 - i):
                    results.append(And(left, right))

    return tuple(results)


@functools.lru_cache(maxsize=None)
def enumerate_programs(n_sites: int, budget: int) -> tuple[Program, ...]:
    """Enumerate ALL programs with node_count == budget.

    Returns a tuple (for hashability/caching). Production order:
    Default (ascending j) → Ite (ascending i, then j, then else_prog).
    """
    if budget < 2:
        return ()
    results: list[Program] = []

    # P(2) → Default(Flip(j))
    if budget == 2:
        for j in range(n_sites):
            results.append(Default(Flip(j)))

    # P(k) → Ite(C(i), Flip(j), P(k-2-i))
    if budget >= 5:
        for i in range(1, budget - 3):  # i in [1, k-4]
            else_budget = budget - 2 - i
            for cond in enumerate_conditions(n_sites, i):
                for j in range(n_sites):
                    for else_prog in enumerate_programs(n_sites, else_budget):
                        results.append(Ite(cond, Flip(j), else_prog))

    return tuple(results)


# ---------------------------------------------------------------------------
# Canonical enumeration (no double-negation, canonical And ordering)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def enumerate_canonical_conditions(
    n_sites: int, budget: int,
) -> tuple[Condition, ...]:
    """Enumerate canonical conditions with node_count == budget.

    Rules:
      1. No double-negation: ``Not(child)`` only when *child* is not
         ``Not``-prefixed.
      2. And commutativity: ``And(left, right)`` requires
         ``left.node_count() <= right.node_count()``.  When budgets are
         equal, ``left.pretty() <= right.pretty()`` breaks ties.
    """
    if budget < 1:
        return ()
    results: list[Condition] = []

    # C(1) → IsZero(j)
    if budget == 1:
        for j in range(n_sites):
            results.append(IsZero(j))

    # C(k) → Not(child) where child is NOT Not-prefixed
    if budget >= 2:
        for child in enumerate_canonical_conditions(n_sites, budget - 1):
            if not isinstance(child, Not):
                results.append(Not(child))

    # C(k) → And(C(i), C(j)) with i <= j
    if budget >= 3:
        for i in range(1, (budget - 1) // 2 + 1):
            j = budget - 1 - i
            lefts = enumerate_canonical_conditions(n_sites, i)
            rights = enumerate_canonical_conditions(n_sites, j)
            for left in lefts:
                for right in rights:
                    if i == j and left.pretty() > right.pretty():
                        continue  # tiebreaker for equal budgets
                    results.append(And(left, right))

    return tuple(results)


@functools.lru_cache(maxsize=None)
def enumerate_canonical_programs(
    n_sites: int, budget: int,
) -> tuple[Program, ...]:
    """Enumerate programs whose conditions are all canonical.

    Same structure as ``enumerate_programs`` but uses
    ``enumerate_canonical_conditions`` for condition subtrees.
    """
    if budget < 2:
        return ()
    results: list[Program] = []

    # P(2) → Default(Flip(j))
    if budget == 2:
        for j in range(n_sites):
            results.append(Default(Flip(j)))

    # P(k) → Ite(C(i), Flip(j), P(k-2-i))
    if budget >= 5:
        for i in range(1, budget - 3):  # i in [1, k-4]
            else_budget = budget - 2 - i
            for cond in enumerate_canonical_conditions(n_sites, i):
                for j in range(n_sites):
                    for else_prog in enumerate_canonical_programs(
                        n_sites, else_budget
                    ):
                        results.append(Ite(cond, Flip(j), else_prog))

    return tuple(results)


# ---------------------------------------------------------------------------
# Formatting: grammar summary
# ---------------------------------------------------------------------------

def format_grammar_summary(n_sites: int, max_budget: int) -> str:
    """Human-readable summary of the grammar and program/condition counts."""
    lines: list[str] = []
    lines.append(f"=== Budget Grammar (N={n_sites}, max_budget={max_budget}) ===")
    lines.append("")

    # Program productions
    lines.append("Program productions:")
    for k in range(1, max_budget + 1):
        cnt = count_programs(n_sites, k)
        if cnt == 0:
            continue
        if k == 2:
            lines.append(f"  P({k}) -> Default(Flip(j))"
                         f"  [{cnt} programs]")
        else:
            # Show each (i, else_budget) combination
            parts: list[str] = []
            for i in range(1, k - 3):
                else_budget = k - 2 - i
                sub = (count_conditions(n_sites, i)
                       * n_sites
                       * count_programs(n_sites, else_budget))
                if sub > 0:
                    parts.append(
                        f"Ite(C({i}), Flip(j), P({else_budget})): {sub}"
                    )
            first = True
            for part in parts:
                if first:
                    lines.append(f"  P({k}) -> {part}")
                    first = False
                else:
                    lines.append(f"  {' ' * len(f'P({k}) -> ')}{part}")
            lines.append(f"  {'':>{len(f'P({k}) -> ')}}[{cnt} programs total]")

    lines.append("")

    # Condition productions
    lines.append("Condition productions:")
    for k in range(1, max_budget + 1):
        cnt = count_conditions(n_sites, k)
        if cnt == 0:
            continue
        parts: list[str] = []
        if k == 1:
            parts.append(f"IsZero(j): {n_sites}")
        if k >= 2:
            not_cnt = count_conditions(n_sites, k - 1)
            if not_cnt > 0:
                parts.append(f"Not(C({k - 1})): {not_cnt}")
        if k >= 3:
            and_cnt = 0
            for i in range(1, k - 1):
                and_cnt += (count_conditions(n_sites, i)
                            * count_conditions(n_sites, k - 1 - i))
            if and_cnt > 0:
                parts.append(f"And(...): {and_cnt}")
        lines.append(f"  C({k}): {', '.join(parts)}  [{cnt} total]")

    lines.append("")

    # Impossible budgets
    impossible = [k for k in range(1, max_budget + 1)
                  if count_programs(n_sites, k) == 0]
    if impossible:
        lines.append(
            f"Budget levels with zero programs: {', '.join(map(str, impossible))}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting: grammar debug
# ---------------------------------------------------------------------------

def format_grammar_debug(n_sites: int, max_budget: int) -> str:
    """Debug/consistency report for the grammar."""
    lines: list[str] = []
    lines.append(f"=== Grammar Debug (N={n_sites}) ===")
    lines.append("")
    lines.append("Common bugs to check:")
    lines.append(
        "  P(2) is the base case (not P(1)) -- Default(Flip) = 2 nodes"
    )
    lines.append(
        "  P(3), P(4) produce 0 programs (Ite minimum = 5 nodes)"
    )
    lines.append(
        "  Ite budget: 1 + |cond| + 1 + |else| = k"
    )
    lines.append(
        "  And budget: 1 + |left| + |right| = k"
    )
    lines.append(
        "  Production order is stable (sorted by type, then index)"
    )
    lines.append("")

    # Budget achievability
    lines.append("Budget achievability:")
    all_ok = True
    for k in range(1, max_budget + 1):
        cnt = count_programs(n_sites, k)
        if cnt == 0:
            reason = _impossible_reason(k)
            lines.append(f"  L={k}: 0 programs ({reason})")
        else:
            # Verify node_count correctness
            progs = enumerate_programs(n_sites, k)
            bad = [p for p in progs if p.node_count() != k]
            if bad:
                lines.append(
                    f"  L={k}: {cnt} programs -- "
                    f"FAIL: {len(bad)} have wrong node_count!"
                )
                all_ok = False
            else:
                lines.append(f"  L={k}: {cnt} programs -- all node_count == {k}")

    lines.append("")
    if all_ok:
        lines.append("Consistency check: PASSED")
    else:
        lines.append("Consistency check: FAILED")

    return "\n".join(lines)


def _impossible_reason(k: int) -> str:
    """Explain why a budget level produces zero programs."""
    if k < 2:
        return "minimum program is Default(Flip) = 2 nodes"
    elif k in (3, 4):
        return "gap between Default(2) and smallest Ite(5)"
    else:
        return "no valid production combination"


def format_canonical_reduction_report(n_sites: int, max_budget: int) -> str:
    """Print before/after counts showing the effect of canonical pruning."""
    lines: list[str] = []
    lines.append(
        f"=== Canonical Reduction Report (N={n_sites}, "
        f"max_budget={max_budget}) ==="
    )
    lines.append("")
    lines.append(
        f"{'Budget':>6} | {'Conditions':>24} | {'Programs':>30}"
    )
    lines.append("-" * 68)

    for k in range(1, max_budget + 1):
        co = count_conditions(n_sites, k)
        cc = count_canonical_conditions(n_sites, k)
        po = count_programs(n_sites, k)
        pc = count_canonical_programs(n_sites, k)

        c_str = f"{co:>8} -> {cc:>8}" if co > 0 else f"{'—':>19}"
        if po > 0:
            pct = 100 * (po - pc) / po
            p_str = f"{po:>10} -> {pc:>10} ({pct:4.0f}%)"
        elif po == 0 and pc == 0:
            p_str = f"{'—':>30}"
        else:
            p_str = f"{po:>10} -> {pc:>10}"

        lines.append(f"{k:>6} | {c_str} | {p_str}")

    lines.append("")
    total_old = sum(count_programs(n_sites, k) for k in range(1, max_budget + 1))
    total_new = sum(
        count_canonical_programs(n_sites, k) for k in range(1, max_budget + 1)
    )
    if total_old > 0:
        pct = 100 * (total_old - total_new) / total_old
        lines.append(
            f"Total programs: {total_old:,} -> {total_new:,} "
            f"({pct:.1f}% reduction)"
        )

    return "\n".join(lines)
