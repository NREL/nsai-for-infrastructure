"""
Tests for the BitString DSL size-budget grammar and derivation engine.

Categories:
1. Condition counts: golden count verification
2. Program counts: golden count verification (including impossible budgets)
3. Node count correctness: every enumerated program has exact node_count
4. Uniqueness: no duplicate programs in enumeration
5. Determinism: stable enumeration order across calls
6. Derivation: leftmost hole, apply, terminal, to_program
7. Derivation canonicality: derivation produces same set as enumeration
8. Grammar formatting: output structure checks
"""

import numpy as np
import pytest

from alphazeropp.instances.bitstring.dsl.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.instances.bitstring.dsl.budget_grammar import (
    ProgramHole, ConditionHole,
    count_conditions, count_programs,
    enumerate_conditions, enumerate_programs,
    count_canonical_conditions, count_canonical_programs,
    enumerate_canonical_conditions, enumerate_canonical_programs,
    format_grammar_summary, format_grammar_debug,
)
from alphazeropp.instances.bitstring.dsl.derivation import (
    Production, DerivationState,
    enumerate_via_derivation,
    format_derivation_trace, trace_first_derivation,
)


# ---------------------------------------------------------------------------
# Condition counts (golden values for N=3)
# ---------------------------------------------------------------------------

class TestConditionCounts:
    def test_c1(self):
        """C(1) = IsZero(j) for j in {0,1,2} → 3."""
        assert count_conditions(3, 1) == 3

    def test_c2(self):
        """C(2) = Not(IsZero(j)) → 3."""
        assert count_conditions(3, 2) == 3

    def test_c3(self):
        """C(3) = Not(C(2)):3 + And(C(1),C(1)):9 → 12."""
        assert count_conditions(3, 3) == 12

    def test_c4(self):
        """C(4) = Not(C(3)):12 + And(C(1),C(2)):9 + And(C(2),C(1)):9 → 30."""
        assert count_conditions(3, 4) == 30

    def test_c0_and_negative(self):
        """No conditions for budget 0 or negative."""
        assert count_conditions(3, 0) == 0
        assert count_conditions(3, -1) == 0


# ---------------------------------------------------------------------------
# Program counts (golden values for N=3)
# ---------------------------------------------------------------------------

class TestProgramCounts:
    def test_p2(self):
        """P(2) = Default(Flip(j)) → 3."""
        assert count_programs(3, 2) == 3

    def test_p3_impossible(self):
        """P(3) is impossible (gap between Default and Ite)."""
        assert count_programs(3, 3) == 0

    def test_p4_impossible(self):
        """P(4) is impossible."""
        assert count_programs(3, 4) == 0

    def test_p5(self):
        """P(5) = Ite(C(1), Flip(j), P(2)) → 3×3×3 = 27."""
        assert count_programs(3, 5) == 27

    def test_p6(self):
        """P(6) = Ite(C(2), Flip(j), P(2)) → 3×3×3 = 27."""
        assert count_programs(3, 6) == 27

    def test_p7(self):
        """P(7) = Ite(C(3), Flip(j), P(2)) → 12×3×3 = 108."""
        assert count_programs(3, 7) == 108

    def test_p8(self):
        """P(8) = Ite(C(1),Flip,P(5)):243 + Ite(C(4),Flip,P(2)):270 → 513."""
        assert count_programs(3, 8) == 513

    def test_p1_impossible(self):
        """P(1) is impossible (minimum is Default(Flip) = 2 nodes)."""
        assert count_programs(3, 1) == 0

    def test_count_matches_enumerate_length(self):
        """count_programs matches len(enumerate_programs) for small budgets."""
        for budget in range(1, 9):
            cnt = count_programs(3, budget)
            progs = enumerate_programs(3, budget)
            assert len(progs) == cnt, (
                f"budget={budget}: count={cnt} but enumerate gave {len(progs)}"
            )


# ---------------------------------------------------------------------------
# Node count correctness
# ---------------------------------------------------------------------------

class TestNodeCountCorrectness:
    @pytest.mark.parametrize("budget", [2, 5, 6, 7, 8])
    def test_all_programs_have_correct_node_count(self, budget):
        """Every enumerated program at budget L has node_count == L."""
        programs = enumerate_programs(n_sites=3, budget=budget)
        for prog in programs:
            assert prog.node_count() == budget, (
                f"Program {prog.pretty()} has node_count={prog.node_count()}, "
                f"expected {budget}"
            )

    @pytest.mark.parametrize("budget", [1, 2, 3, 4, 5])
    def test_all_conditions_have_correct_node_count(self, budget):
        """Every enumerated condition at budget k has node_count == k."""
        conds = enumerate_conditions(n_sites=3, budget=budget)
        for c in conds:
            assert c.node_count() == budget, (
                f"Condition {c.pretty()} has node_count={c.node_count()}, "
                f"expected {budget}"
            )


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

class TestUniqueness:
    @pytest.mark.parametrize("budget", [2, 5, 7, 8])
    def test_no_duplicate_programs(self, budget):
        """No duplicate programs in enumeration (checked via pretty())."""
        progs = enumerate_programs(n_sites=3, budget=budget)
        pretty_strs = [p.pretty() for p in progs]
        assert len(pretty_strs) == len(set(pretty_strs)), (
            f"Duplicates found at budget={budget}"
        )

    @pytest.mark.parametrize("budget", [1, 2, 3, 4])
    def test_no_duplicate_conditions(self, budget):
        """No duplicate conditions."""
        conds = enumerate_conditions(n_sites=3, budget=budget)
        pretty_strs = [c.pretty() for c in conds]
        assert len(pretty_strs) == len(set(pretty_strs)), (
            f"Duplicates found at condition budget={budget}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_stable_program_order(self):
        """Two calls produce identical program order."""
        # Clear cache to force re-enumeration
        enumerate_programs.cache_clear()
        a = enumerate_programs(n_sites=3, budget=5)
        enumerate_programs.cache_clear()
        b = enumerate_programs(n_sites=3, budget=5)
        assert [p.pretty() for p in a] == [p.pretty() for p in b]

    def test_stable_condition_order(self):
        """Two calls produce identical condition order."""
        enumerate_conditions.cache_clear()
        a = enumerate_conditions(n_sites=3, budget=3)
        enumerate_conditions.cache_clear()
        b = enumerate_conditions(n_sites=3, budget=3)
        assert [c.pretty() for c in a] == [c.pretty() for c in b]


# ---------------------------------------------------------------------------
# Derivation tests
# ---------------------------------------------------------------------------

class TestDerivation:
    def test_initial_state(self):
        """Initial state is a single ProgramHole."""
        state = DerivationState.initial(5)
        assert not state.is_terminal()
        hole = state.leftmost_hole()
        assert isinstance(hole, ProgramHole)
        assert hole.budget == 5

    def test_leftmost_hole_in_ite(self):
        """After expanding P(5), leftmost hole is the ConditionHole in Ite."""
        state = DerivationState.initial(5)
        prods = state.legal_productions(n_sites=3)
        # First production: P(5) -> Ite(C(1), Flip(0), P(2))
        state2 = state.apply(prods[0])
        hole = state2.leftmost_hole()
        assert isinstance(hole, ConditionHole)
        assert hole.budget == 1

    def test_apply_reduces_holes(self):
        """Applying a terminal production reduces hole count."""
        state = DerivationState.initial(5)
        assert state.hole_count() == 1
        # P(5) -> Ite(C(1), Flip(0), P(2)) adds 2 holes (C(1), P(2))
        prods = state.legal_productions(3)
        state2 = state.apply(prods[0])
        assert state2.hole_count() == 2
        # Expand C(1) -> IsZero(0), removes one hole
        prods2 = state2.legal_productions(3)
        state3 = state2.apply(prods2[0])
        assert state3.hole_count() == 1

    def test_terminal_conversion(self):
        """A fully expanded derivation converts to a valid Program."""
        state = DerivationState.initial(5)
        while not state.is_terminal():
            prods = state.legal_productions(3)
            state = state.apply(prods[0])
        prog = state.to_program()
        assert prog.node_count() == 5
        assert isinstance(prog, (Ite, Default))

    def test_pretty_shows_holes(self):
        """Partial AST pretty-print shows holes."""
        state = DerivationState.initial(5)
        prods = state.legal_productions(3)
        state2 = state.apply(prods[0])
        text = state2.pretty()
        assert "[C:1]" in text or "[P:2]" in text


# ---------------------------------------------------------------------------
# Derivation canonicality
# ---------------------------------------------------------------------------

class TestDerivationCanonical:
    @pytest.mark.parametrize("budget", [2, 5, 7])
    def test_derivation_subset_of_full(self, budget):
        """Derivation programs are a subset of full enumeration."""
        enum_progs = set(p.pretty() for p in enumerate_programs(3, budget))
        deriv_progs = set(
            p.pretty() for p in enumerate_via_derivation(3, budget)
        )
        assert deriv_progs.issubset(enum_progs), (
            f"Derivation produced programs not in full enum at budget={budget}"
        )

    @pytest.mark.parametrize("budget", [2, 5, 7])
    def test_derivation_superset_of_canonical(self, budget):
        """Derivation programs are a superset of canonical enumeration."""
        canon_progs = set(
            p.pretty() for p in enumerate_canonical_programs(3, budget)
        )
        deriv_progs = set(
            p.pretty() for p in enumerate_via_derivation(3, budget)
        )
        assert canon_progs.issubset(deriv_progs), (
            f"Canonical programs missing from derivation at budget={budget}"
        )

    def test_derivation_count_in_range(self):
        """Derivation count is between canonical and full counts."""
        for budget in [2, 5, 8]:
            deriv = enumerate_via_derivation(3, budget)
            cp = count_canonical_programs(3, budget)
            fp = count_programs(3, budget)
            assert cp <= len(deriv) <= fp, (
                f"budget={budget}: canonical={cp}, deriv={len(deriv)}, full={fp}"
            )

    def test_only_leftmost_hole_expandable(self):
        """At any state, productions match the leftmost hole's kind and budget."""
        state = DerivationState.initial(8)
        prods = state.legal_productions(3)
        hole = state.leftmost_hole()
        for p in prods:
            assert p.hole_kind == "P"
            assert p.hole_budget == hole.budget

        # After one expansion
        state2 = state.apply(prods[0])
        hole2 = state2.leftmost_hole()
        prods2 = state2.legal_productions(3)
        expected_kind = "C" if isinstance(hole2, ConditionHole) else "P"
        for p in prods2:
            assert p.hole_kind == expected_kind
            assert p.hole_budget == hole2.budget


# ---------------------------------------------------------------------------
# Grammar formatting
# ---------------------------------------------------------------------------

class TestGrammarFormatting:
    def test_grammar_summary_structure(self):
        """format_grammar_summary produces expected sections."""
        text = format_grammar_summary(n_sites=3, max_budget=8)
        assert "Budget Grammar" in text
        assert "Program productions:" in text
        assert "Condition productions:" in text
        assert "P(2)" in text
        assert "P(5)" in text

    def test_grammar_debug_structure(self):
        """format_grammar_debug produces expected sections."""
        text = format_grammar_debug(n_sites=3, max_budget=8)
        assert "Grammar Debug" in text
        assert "Common bugs" in text
        assert "Budget achievability" in text
        assert "Consistency check: PASSED" in text

    def test_derivation_trace_structure(self):
        """trace_first_derivation produces expected output."""
        text = trace_first_derivation(n_sites=3, budget=5)
        assert "Derivation Trace" in text
        assert "Step 0" in text
        assert "TERMINAL" in text
        assert "node_count" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_n_sites_1(self):
        """Grammar works for the degenerate N=1 case."""
        assert count_programs(1, 2) == 1  # Default(Flip(0))
        assert count_programs(1, 5) == 1  # Ite(IsZero(0), Flip(0), Default(Flip(0)))
        progs = enumerate_programs(1, 5)
        assert len(progs) == 1
        assert progs[0].node_count() == 5

    def test_larger_n_sites(self):
        """Grammar scales to larger N."""
        # N=5, P(2) = 5 programs
        assert count_programs(5, 2) == 5
        # N=5, P(5) = 5 × 5 × 5 = 125
        assert count_programs(5, 5) == 125

    def test_program_validates(self):
        """Enumerated programs pass validate()."""
        for prog in enumerate_programs(3, 5):
            prog.validate(n_sites=3)  # should not raise


# ---------------------------------------------------------------------------
# Dead-end prevention
# ---------------------------------------------------------------------------

def _assert_no_dead_holes(node, n_sites):
    """Recursively verify no hole in the subtree has zero completions."""
    if isinstance(node, ProgramHole):
        assert count_programs(n_sites, node.budget) > 0, (
            f"Dead-end ProgramHole({node.budget}): "
            f"count_programs={count_programs(n_sites, node.budget)}"
        )
    if isinstance(node, ConditionHole):
        assert count_conditions(n_sites, node.budget) > 0, (
            f"Dead-end ConditionHole({node.budget}): "
            f"count_conditions={count_conditions(n_sites, node.budget)}"
        )
    if isinstance(node, Ite):
        _assert_no_dead_holes(node.cond, n_sites)
        _assert_no_dead_holes(node.action, n_sites)
        _assert_no_dead_holes(node.else_prog, n_sites)
    elif isinstance(node, Default):
        _assert_no_dead_holes(node.action, n_sites)
    elif isinstance(node, Not):
        _assert_no_dead_holes(node.child, n_sites)
    elif isinstance(node, And):
        _assert_no_dead_holes(node.left, n_sites)
        _assert_no_dead_holes(node.right, n_sites)


class TestDeadEndPrevention:
    @pytest.mark.parametrize("budget", range(2, 15))
    def test_no_dead_end_productions(self, budget):
        """Every hole created by a production must be completable."""
        from alphazeropp.instances.bitstring.dsl.derivation import _program_productions
        n_sites = 6
        for prod in _program_productions(budget, n_sites):
            _assert_no_dead_holes(prod.result, n_sites)

    def test_program_counts_unchanged(self):
        """Filtering dead-end productions must not change program counts."""
        assert count_programs(3, 2) == 3
        assert count_programs(3, 3) == 0
        assert count_programs(3, 4) == 0
        assert count_programs(3, 5) == 27
        assert count_programs(3, 8) == 513
        assert count_programs(6, 14) == 151_173_432

    def test_random_derivations_never_truncate(self):
        """1000 random derivation episodes should all reach terminal state."""
        n_sites = 6
        rng = np.random.default_rng(42)
        truncations = 0
        for _ in range(1000):
            state = DerivationState.initial(14)
            while not state.is_terminal():
                prods = state.legal_productions(n_sites)
                if not prods:
                    truncations += 1
                    break
                state = state.apply(prods[rng.integers(len(prods))])
        assert truncations == 0, f"Got {truncations}/1000 truncations (expected 0)"

    @pytest.mark.parametrize("budget", [2, 5, 7])
    def test_derivation_subset_of_full(self, budget):
        """Derivation programs are a subset of full enumeration."""
        enum_progs = set(p.pretty() for p in enumerate_programs(3, budget))
        deriv_progs = set(
            p.pretty() for p in enumerate_via_derivation(3, budget)
        )
        assert deriv_progs.issubset(enum_progs)


# ---------------------------------------------------------------------------
# Canonical condition/program counts (Phase 1)
# ---------------------------------------------------------------------------

class TestCanonicalConditionCounts:
    """Golden values for count_canonical_conditions (N=3)."""

    @pytest.mark.parametrize("budget,expected", [
        (1, 3), (2, 3), (3, 6), (4, 15), (5, 33), (6, 87), (7, 228),
    ])
    def test_golden_values(self, budget, expected):
        assert count_canonical_conditions(3, budget) == expected

    def test_zero_and_negative(self):
        assert count_canonical_conditions(3, 0) == 0
        assert count_canonical_conditions(3, -1) == 0

    @pytest.mark.parametrize("budget", range(1, 8))
    def test_matches_enumeration_length(self, budget):
        """count matches len(enumerate) for canonical conditions."""
        cc = count_canonical_conditions(3, budget)
        ec = len(enumerate_canonical_conditions(3, budget))
        assert cc == ec, f"C({budget}): count={cc}, enum={ec}"

    @pytest.mark.parametrize("budget", range(1, 8))
    def test_at_most_full_count(self, budget):
        """Canonical count <= full count (it's a subset)."""
        assert (count_canonical_conditions(3, budget)
                <= count_conditions(3, budget))

    @pytest.mark.parametrize("budget", range(1, 13))
    def test_no_dead_budget(self, budget):
        """Every condition budget >= 1 has at least one canonical condition."""
        assert count_canonical_conditions(6, budget) > 0


class TestCanonicalProgramCounts:
    """Golden values for count_canonical_programs (N=3)."""

    @pytest.mark.parametrize("budget,expected", [
        (2, 3), (3, 0), (4, 0), (5, 27), (6, 27), (7, 54), (8, 378),
    ])
    def test_golden_values(self, budget, expected):
        assert count_canonical_programs(3, budget) == expected

    def test_n6_budget14(self):
        """Key config: N=6, budget=14."""
        assert count_canonical_programs(6, 14) == 37_463_688

    @pytest.mark.parametrize("budget", [2, 5, 7, 8])
    def test_matches_enumeration_length(self, budget):
        """count matches len(enumerate) for canonical programs."""
        cp = count_canonical_programs(3, budget)
        ep = len(enumerate_canonical_programs(3, budget))
        assert cp == ep, f"P({budget}): count={cp}, enum={ep}"

    @pytest.mark.parametrize("budget", range(1, 9))
    def test_at_most_full_count(self, budget):
        """Canonical count <= full count."""
        assert (count_canonical_programs(3, budget)
                <= count_programs(3, budget))


class TestCanonicalEnumerationProperties:
    """Verify structural properties of canonical conditions."""

    @pytest.mark.parametrize("budget", range(1, 7))
    def test_no_double_negation(self, budget):
        """No canonical condition contains Not(Not(...))."""
        for cond in enumerate_canonical_conditions(3, budget):
            _assert_no_double_neg(cond)

    @pytest.mark.parametrize("budget", range(3, 7))
    def test_and_canonical_order(self, budget):
        """All And nodes have left_budget <= right_budget."""
        for cond in enumerate_canonical_conditions(3, budget):
            _assert_and_canonical(cond)

    @pytest.mark.parametrize("budget", [2, 5, 7, 8])
    def test_no_duplicate_canonical_programs(self, budget):
        """No duplicate programs in canonical enumeration."""
        progs = enumerate_canonical_programs(3, budget)
        pretty_strs = [p.pretty() for p in progs]
        assert len(pretty_strs) == len(set(pretty_strs))

    @pytest.mark.parametrize("budget", range(1, 7))
    def test_no_duplicate_canonical_conditions(self, budget):
        """No duplicate conditions in canonical enumeration."""
        conds = enumerate_canonical_conditions(3, budget)
        pretty_strs = [c.pretty() for c in conds]
        assert len(pretty_strs) == len(set(pretty_strs))

    @pytest.mark.parametrize("budget", [2, 5, 7, 8])
    def test_canonical_is_subset_of_full(self, budget):
        """Every canonical program also appears in full enumeration."""
        full = set(p.pretty() for p in enumerate_programs(3, budget))
        canon = set(p.pretty() for p in enumerate_canonical_programs(3, budget))
        assert canon.issubset(full)


# ---------------------------------------------------------------------------
# Derivation canonicality checks (Phase 2 + Phase 3)
# ---------------------------------------------------------------------------

class TestDerivationCanonicalProperties:
    """Verify that the canonical derivation engine produces well-formed programs."""

    def test_no_double_neg_in_derivation(self):
        """1000 random derivations at budget=14 produce no Not(Not(...))."""
        rng = np.random.default_rng(123)
        for _ in range(1000):
            state = DerivationState.initial(14)
            while not state.is_terminal():
                prods = state.legal_productions(6)
                assert prods, "Dead-end encountered"
                state = state.apply(prods[rng.integers(len(prods))])
            prog = state.to_program()
            _assert_no_double_neg_program(prog)

    def test_and_budget_ordering_in_derivation(self):
        """1000 random derivations verify And left_budget <= right_budget."""
        rng = np.random.default_rng(456)
        for _ in range(1000):
            state = DerivationState.initial(14)
            while not state.is_terminal():
                prods = state.legal_productions(6)
                assert prods, "Dead-end encountered"
                state = state.apply(prods[rng.integers(len(prods))])
            prog = state.to_program()
            _assert_and_budget_ordering_program(prog)

    def test_condition_productions_parent_is_not(self):
        """_condition_productions with parent_is_not=True never returns Not."""
        from alphazeropp.instances.bitstring.dsl.derivation import (
            _condition_productions,
        )
        for k in range(1, 13):
            prods = _condition_productions(k, 6, parent_is_not=True)
            for p in prods:
                assert not isinstance(p.result, Not), (
                    f"Not production at C({k}) with parent_is_not=True: {p.label}"
                )

    def test_action_space_unchanged(self):
        """compute_max_productions returns the expected value."""
        from alphazeropp.instances.bitstring.dsl.derivation_game import (
            compute_max_productions,
        )
        assert compute_max_productions(14, 6) == 48


class TestCrossValidation:
    """Cross-validate canonical enumeration against canonical derivation."""

    @pytest.mark.parametrize("budget", [2, 5, 6])
    def test_exact_match_no_equal_budget_and(self, budget):
        """At budgets where no And(C(i),C(i)) exists, derivation == canonical."""
        canon = set(
            p.pretty() for p in enumerate_canonical_programs(3, budget)
        )
        deriv = set(
            p.pretty() for p in enumerate_via_derivation(3, budget)
        )
        assert canon == deriv, (
            f"P({budget}): canon={len(canon)}, deriv={len(deriv)}"
        )


def _assert_no_double_neg_program(prog):
    """Check no double-negation in any condition of a program."""
    if isinstance(prog, Ite):
        _assert_no_double_neg(prog.cond)
        _assert_no_double_neg_program(prog.else_prog)
    # Default has no conditions


def _assert_and_budget_ordering_program(prog):
    """Check And budget ordering in all conditions of a program."""
    if isinstance(prog, Ite):
        _assert_and_budget_ordering_cond(prog.cond)
        _assert_and_budget_ordering_program(prog.else_prog)


def _assert_and_budget_ordering_cond(node):
    """Check And left_budget <= right_budget (budget-level only)."""
    if isinstance(node, And):
        assert node.left.node_count() <= node.right.node_count(), (
            f"And budget ordering violated: {node.pretty()}"
        )
        _assert_and_budget_ordering_cond(node.left)
        _assert_and_budget_ordering_cond(node.right)
    elif isinstance(node, Not):
        _assert_and_budget_ordering_cond(node.child)


def _assert_no_double_neg(node):
    """Recursively verify no Not(Not(...)) pattern."""
    if isinstance(node, Not):
        assert not isinstance(node.child, Not), (
            f"Double negation found: {node.pretty()}"
        )
        _assert_no_double_neg(node.child)
    elif isinstance(node, And):
        _assert_no_double_neg(node.left)
        _assert_no_double_neg(node.right)


def _assert_and_canonical(node):
    """Recursively verify And budget ordering: left <= right."""
    if isinstance(node, And):
        assert node.left.node_count() <= node.right.node_count(), (
            f"And not canonical: left={node.left.node_count()}, "
            f"right={node.right.node_count()} in {node.pretty()}"
        )
        if node.left.node_count() == node.right.node_count():
            assert node.left.pretty() <= node.right.pretty(), (
                f"And tiebreaker violated: {node.pretty()}"
            )
        _assert_and_canonical(node.left)
        _assert_and_canonical(node.right)
    elif isinstance(node, Not):
        _assert_and_canonical(node.child)
