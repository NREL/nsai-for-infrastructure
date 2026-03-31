"""
Tests for grammar pruning: Flip restriction and one-hot satisfiability.

Tests:
1. Flip restriction: n_actions < n_sites restricts program productions
2. One-hot pruning: contradictory conditions pruned from legal_productions
3. Backward compatibility: no pruning when params are None
4. Helper function correctness: _extract_literals, _one_hot_contradiction,
   _find_leftmost_hole_with_context
"""

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.budget_grammar import (
    ProgramHole, ConditionHole,
)
from alphazeropp.synthesis.derivation import (
    Production, DerivationState,
    _program_productions, _condition_productions,
    _find_leftmost_hole_with_context,
    _extract_literals,
    _one_hot_contradiction,
)
from alphazeropp.synthesis.derivation_game import (
    DerivationGame, compute_max_productions,
)


# ---------------------------------------------------------------------------
# Flip restriction
# ---------------------------------------------------------------------------

class TestFlipRestriction:
    """Verify n_actions restricts Flip indices in program productions."""

    def test_program_productions_default(self):
        """Without n_actions, Flip ranges over 0..n_sites-1."""
        prods = _program_productions(2, n_sites=7, mode="exact")
        flip_indices = {p.result.action.index for p in prods}
        assert flip_indices == set(range(7))

    def test_program_productions_restricted(self):
        """With n_actions=6, Flip ranges over 0..5 only."""
        prods = _program_productions(2, n_sites=7, mode="exact", n_actions=6)
        flip_indices = {p.result.action.index for p in prods}
        assert flip_indices == set(range(6))
        assert 6 not in flip_indices

    def test_ite_productions_restricted(self):
        """Ite productions also respect n_actions for the Flip action."""
        prods = _program_productions(5, n_sites=7, mode="max", n_actions=6)
        for p in prods:
            if isinstance(p.result, Ite):
                assert p.result.action.index < 6, (
                    f"Ite Flip index {p.result.action.index} >= n_actions=6"
                )

    def test_fewer_productions_with_n_actions(self):
        """n_actions < n_sites should produce fewer program productions."""
        full = _program_productions(5, n_sites=7, mode="max")
        restricted = _program_productions(5, n_sites=7, mode="max", n_actions=6)
        assert len(restricted) < len(full)

    def test_compute_max_productions_reduced(self):
        """compute_max_productions should be smaller with n_actions < n_sites."""
        full = compute_max_productions(18, 7, mode="max", allow_and=True)
        restricted = compute_max_productions(
            18, 7, mode="max", allow_and=True, n_actions=6)
        assert restricted < full

    def test_n_actions_none_is_default(self):
        """n_actions=None should behave identically to no restriction."""
        prods_default = _program_productions(5, n_sites=7, mode="max")
        prods_none = _program_productions(5, n_sites=7, mode="max", n_actions=None)
        assert len(prods_default) == len(prods_none)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestExtractLiterals:
    """Test _extract_literals on various condition shapes."""

    def test_iszero(self):
        assert _extract_literals(IsZero(3)) == {(3, False)}

    def test_not_iszero(self):
        assert _extract_literals(Not(IsZero(0))) == {(0, True)}

    def test_and_chain(self):
        cond = And(Not(IsZero(0)), IsZero(5))
        assert _extract_literals(cond) == {(0, True), (5, False)}

    def test_complex_not_skipped(self):
        """Not(And(...)) is too complex — returns empty set."""
        cond = Not(And(IsZero(0), IsZero(1)))
        assert _extract_literals(cond) == set()

    def test_nested_and(self):
        cond = And(And(Not(IsZero(0)), IsZero(4)), Not(IsZero(2)))
        assert _extract_literals(cond) == {(0, True), (4, False), (2, True)}


class TestOneHotContradiction:
    """Test _one_hot_contradiction detection."""

    def test_direct_contradiction(self):
        """Same index, opposite polarity → contradiction."""
        context = {(0, True)}
        idx_to_group = {0: 0, 1: 0, 2: 0, 3: 0}
        assert _one_hot_contradiction(context, 0, False, idx_to_group)

    def test_one_hot_two_positive(self):
        """Two positive literals in same group → contradiction."""
        context = {(0, True)}
        idx_to_group = {0: 0, 1: 0, 2: 0, 3: 0}
        assert _one_hot_contradiction(context, 1, True, idx_to_group)

    def test_no_contradiction_different_groups(self):
        """Positive literals in different groups → no contradiction."""
        context = {(0, True)}
        idx_to_group = {0: 0, 1: 0, 4: 1, 5: 1}
        assert not _one_hot_contradiction(context, 4, True, idx_to_group)

    def test_no_contradiction_negative_same_group(self):
        """Positive + negative in same group → no contradiction."""
        context = {(0, True)}
        idx_to_group = {0: 0, 1: 0, 2: 0}
        assert not _one_hot_contradiction(context, 1, False, idx_to_group)

    def test_no_group(self):
        """Index not in any group → no one-hot contradiction."""
        context = {(0, True)}
        idx_to_group = {0: 0, 1: 0}
        assert not _one_hot_contradiction(context, 5, True, idx_to_group)


class TestFindLeftmostHoleWithContext:
    """Test context collection during leftmost hole search."""

    def test_simple_program_hole(self):
        """ProgramHole at root has no context."""
        result = _find_leftmost_hole_with_context(ProgramHole(5))
        hole, siblings, negated = result
        assert isinstance(hole, ProgramHole)
        assert siblings == []
        assert negated is False

    def test_condition_hole_in_and_right(self):
        """And(complete_cond, ConditionHole) → sibling is the left cond."""
        tree = And(Not(IsZero(0)), ConditionHole(1))
        result = _find_leftmost_hole_with_context(tree)
        hole, siblings, negated = result
        assert isinstance(hole, ConditionHole)
        assert len(siblings) == 1
        assert isinstance(siblings[0], Not)
        assert negated is False

    def test_condition_hole_under_not(self):
        """And(A, Not(ConditionHole)) → negated=True, sibling is A."""
        tree = And(Not(IsZero(0)), Not(ConditionHole(1)))
        result = _find_leftmost_hole_with_context(tree)
        hole, siblings, negated = result
        assert isinstance(hole, ConditionHole)
        assert len(siblings) == 1
        assert negated is True

    def test_negated_and_clears_context(self):
        """Not(And(A, hole)) → effective Or, context cleared."""
        tree = Not(And(IsZero(0), ConditionHole(1)))
        result = _find_leftmost_hole_with_context(tree)
        hole, siblings, negated = result
        assert isinstance(hole, ConditionHole)
        assert siblings == []  # cleared by negated And
        assert negated is True  # under one Not

    def test_ite_fresh_context(self):
        """Ite starts a fresh context for its condition."""
        tree = Ite(ConditionHole(1), Flip(0), ProgramHole(2))
        result = _find_leftmost_hole_with_context(tree)
        hole, siblings, negated = result
        assert isinstance(hole, ConditionHole)
        assert siblings == []
        assert negated is False

    def test_nested_and_accumulates(self):
        """And(A, And(B, hole)) → siblings = [A, B]."""
        tree = And(
            Not(IsZero(0)),
            And(IsZero(4), ConditionHole(1)),
        )
        result = _find_leftmost_hole_with_context(tree)
        hole, siblings, negated = result
        assert isinstance(hole, ConditionHole)
        assert len(siblings) == 2
        assert negated is False


# ---------------------------------------------------------------------------
# One-hot pruning integration
# ---------------------------------------------------------------------------

class TestOneHotPruningIntegration:
    """End-to-end test: legal_productions filters contradictory conditions."""

    def test_and_not_iszero_prunes_same_group(self):
        """And(Not(IsZero(0)), C(1)): expanding C(1) with one_hot_groups
        should prune IsZero(1..3) because Not(IsZero(j)) for j in
        same group as 0 would be a one-hot contradiction.
        """
        # Build state: And(Not(IsZero(0)), ConditionHole(1))
        # This represents a condition where we already assert "at loc 0"
        # and are about to expand the right condition hole.
        state = DerivationState.initial(5)  # ProgramHole(5)
        # Apply: P(5) -> Ite(C(2), Flip(0), P(2))
        # Then: C(2) -> And(C(1), C(1))  — but we need the right structure.

        # More direct: construct the state manually.
        # We want the partial AST to be:
        # Ite(And(Not(IsZero(0)), ConditionHole(1)), Flip(0), ProgramHole(2))
        root = Ite(
            And(Not(IsZero(0)), ConditionHole(1)),
            Flip(0),
            ProgramHole(2),
        )
        state = DerivationState(root=root)

        # Without pruning: C(1) can expand to IsZero(j) for j=0..6
        n_sites = 7
        prods_no_prune = state.legal_productions(n_sites, mode="max")
        iszero_indices_no_prune = {
            p.result.index for p in prods_no_prune
            if isinstance(p.result, IsZero)
        }
        assert iszero_indices_no_prune == set(range(7))

        # With pruning: one_hot_groups=[[0,1,2,3]]
        # The hole is NOT under a Not → negated=False
        # IsZero(j) is a negative literal. Not(IsZero(0)) is positive.
        # IsZero(0) contradicts Not(IsZero(0)) → direct contradiction.
        # IsZero(1..3) with negated=False → negative literal, NOT positive
        # So only IsZero(0) should be pruned (direct contradiction).
        prods_pruned = state.legal_productions(
            n_sites, mode="max",
            one_hot_groups=[[0, 1, 2, 3]],
        )
        iszero_indices_pruned = {
            p.result.index for p in prods_pruned
            if isinstance(p.result, IsZero)
        }
        # IsZero(0) pruned (direct contradiction with Not(IsZero(0)))
        assert 0 not in iszero_indices_pruned
        # IsZero(1..6) kept (negative literals don't cause one-hot contradiction)
        assert iszero_indices_pruned == {1, 2, 3, 4, 5, 6}

    def test_negated_hole_prunes_one_hot(self):
        """And(Not(IsZero(0)), Not(ConditionHole(1))): the hole is under Not.
        Expanding to IsZero(j) creates Not(IsZero(j)) — a positive literal.
        One-hot contradiction with Not(IsZero(0)) for j in {1,2,3}.
        """
        root = Ite(
            And(Not(IsZero(0)), Not(ConditionHole(1))),
            Flip(0),
            ProgramHole(2),
        )
        state = DerivationState(root=root)
        n_sites = 7
        one_hot_groups = [[0, 1, 2, 3]]

        prods = state.legal_productions(
            n_sites, mode="max",
            one_hot_groups=one_hot_groups,
        )
        iszero_indices = {
            p.result.index for p in prods
            if isinstance(p.result, IsZero)
        }
        # IsZero(0): Not(IsZero(0)) — same positive literal, not a contradiction
        #   (redundant but satisfiable)
        assert 0 in iszero_indices
        # IsZero(1..3): Not(IsZero(j)) — positive, same group as 0 → contradiction
        assert 1 not in iszero_indices
        assert 2 not in iszero_indices
        assert 3 not in iszero_indices
        # IsZero(4..6): Not(IsZero(j)) — positive, NOT in one-hot group → kept
        assert {4, 5, 6}.issubset(iszero_indices)

    def test_no_pruning_without_groups(self):
        """Without one_hot_groups, all IsZero productions are kept."""
        root = Ite(
            And(Not(IsZero(0)), Not(ConditionHole(1))),
            Flip(0),
            ProgramHole(2),
        )
        state = DerivationState(root=root)
        prods = state.legal_productions(7, mode="max")
        iszero_indices = {
            p.result.index for p in prods
            if isinstance(p.result, IsZero)
        }
        assert iszero_indices == set(range(7))


# ---------------------------------------------------------------------------
# DerivationGame integration
# ---------------------------------------------------------------------------

class TestDerivationGamePruning:
    """Test that DerivationGame correctly threads pruning params."""

    @pytest.fixture
    def leaf_eval(self):
        """Dummy leaf evaluator that returns 0.0."""
        class DummyLeafEval:
            def __call__(self, program):
                return 0.0
            def stats(self):
                return {}
        return DummyLeafEval()

    def test_action_space_smaller_with_n_actions(self, leaf_eval):
        """Action space should be smaller when n_actions < n_sites."""
        game_full = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game_restricted = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)
        assert game_restricted.action_space.n < game_full.action_space.n

    def test_clone_preserves_pruning_params(self, leaf_eval):
        """clone() should preserve n_actions and one_hot_groups."""
        game = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max",
            n_actions=6, one_hot_groups=[[0, 1, 2, 3]])
        game.reset()
        cloned = game.clone()
        assert cloned._n_actions == 6
        assert cloned._one_hot_groups == [[0, 1, 2, 3]]
        assert cloned.action_space.n == game.action_space.n
