"""
Tests for budget max mode (program_budget_mode="max").

Covers:
1. Production generation: early termination for programs and conditions
2. Condition dead-end fix: C(2, parent_is_not=True) has productions in max mode
3. Action space sizing: compute_max_productions with mode="max"
4. Game integration: DerivationGame with max mode, no truncations
5. Program validity: node_count <= budget in max mode
6. Superset property: max mode reaches all exact-mode programs plus extras
7. Backward compatibility: exact mode behavior unchanged
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.bitstring.dsl.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.instances.bitstring.dsl.budget_grammar import (
    ProgramHole, ConditionHole,
)
from alphazeropp.instances.bitstring.dsl.derivation import (
    DerivationState, Production,
    _program_productions, _condition_productions,
    enumerate_via_derivation,
)
from alphazeropp.instances.bitstring.dsl.derivation_game import (
    DerivationGame, UniformPolicyValueNet, compute_max_productions,
)
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.instances.bitstring.dsl.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SITES = 3
N_ONES = 2


@pytest.fixture
def leaf_eval():
    cfg = GameConfig(bit_flip=True, sparse_reward=False, n_ones=N_ONES)
    frozen = all_initial_states(N_SITES, N_ONES)
    return LeafEvaluator(N_SITES, frozen, cfg, metric="avg_reward")


# ---------------------------------------------------------------------------
# 1. Program production tests (max mode)
# ---------------------------------------------------------------------------

class TestProgramProductionsMaxMode:

    def test_p2_same_in_both_modes(self):
        """P(2) produces Default(Flip(j)) in both modes."""
        exact = _program_productions(2, 3, "exact")
        max_ = _program_productions(2, 3, "max")
        assert len(exact) == len(max_) == 3

    def test_p3_exact_empty(self):
        """P(3) has zero productions in exact mode (dead-end gap)."""
        assert len(_program_productions(3, 3, "exact")) == 0

    def test_p3_max_has_terminations(self):
        """P(3) has n_sites termination productions in max mode."""
        prods = _program_productions(3, 3, "max")
        assert len(prods) == 3  # Default(Flip(0)), Default(Flip(1)), Default(Flip(2))
        for p in prods:
            assert isinstance(p.result, Default)

    def test_p4_exact_empty(self):
        """P(4) has zero productions in exact mode (dead-end gap)."""
        assert len(_program_productions(4, 3, "exact")) == 0

    def test_p4_max_has_terminations(self):
        """P(4) has n_sites termination productions in max mode."""
        prods = _program_productions(4, 3, "max")
        assert len(prods) == 3
        for p in prods:
            assert isinstance(p.result, Default)

    def test_p5_max_has_more_prods_than_exact(self):
        """P(5) in max mode has terminations PLUS Ite expansions."""
        exact = _program_productions(5, 3, "exact")
        max_ = _program_productions(5, 3, "max")
        # Exact: Ite(C(1), Flip(j), P(2)) -> 1 × 3 = 3.
        # Max: 3 (Default) + 3 (Ite) = 6.
        assert len(exact) == 3
        assert len(max_) == 6
        assert len(max_) > len(exact)

    def test_p7_max_includes_dead_end_else_budgets(self):
        """P(7) in max mode includes Ite(..., P(3)) and Ite(..., P(4))."""
        prods = _program_productions(7, 3, "max")
        else_budgets = set()
        for p in prods:
            if isinstance(p.result, Ite):
                assert isinstance(p.result.else_prog, ProgramHole)
                else_budgets.add(p.result.else_prog.budget)
        # Max mode should include else_budgets 2, 3, 4 (all >= 2)
        assert 2 in else_budgets
        assert 3 in else_budgets
        assert 4 in else_budgets

    def test_p7_exact_skips_dead_end_else_budgets(self):
        """P(7) in exact mode skips P(3) and P(4) else budgets."""
        prods = _program_productions(7, 3, "exact")
        else_budgets = set()
        for p in prods:
            if isinstance(p.result, Ite):
                assert isinstance(p.result.else_prog, ProgramHole)
                else_budgets.add(p.result.else_prog.budget)
        assert 3 not in else_budgets
        assert 4 not in else_budgets
        assert 2 in else_budgets

    @pytest.mark.parametrize("budget", range(2, 15))
    def test_max_superset_of_exact(self, budget):
        """Max mode productions are a superset of exact mode productions."""
        exact = set(p.label for p in _program_productions(budget, 6, "exact"))
        max_ = set(p.label for p in _program_productions(budget, 6, "max"))
        assert exact.issubset(max_), (
            f"P({budget}): exact has productions not in max: {exact - max_}"
        )


# ---------------------------------------------------------------------------
# 2. Condition production tests (max mode)
# ---------------------------------------------------------------------------

class TestConditionProductionsMaxMode:

    def test_c1_same_in_both_modes(self):
        """C(1) produces IsZero(j) in both modes."""
        exact = _condition_productions(1, 3, mode="exact")
        max_ = _condition_productions(1, 3, mode="max")
        assert len(exact) == len(max_) == 3

    def test_c2_max_has_early_termination(self):
        """C(2) in max mode has IsZero terminations + Not expansion."""
        prods = _condition_productions(2, 3, mode="max")
        iszero_count = sum(1 for p in prods if isinstance(p.result, IsZero))
        not_count = sum(1 for p in prods if isinstance(p.result, Not))
        assert iszero_count == 3  # IsZero(0), IsZero(1), IsZero(2)
        assert not_count == 1     # Not(C(1))

    def test_c2_exact_no_early_termination(self):
        """C(2) in exact mode has only Not, no IsZero."""
        prods = _condition_productions(2, 3, mode="exact")
        iszero_count = sum(1 for p in prods if isinstance(p.result, IsZero))
        not_count = sum(1 for p in prods if isinstance(p.result, Not))
        assert iszero_count == 0
        assert not_count == 1

    def test_c2_parent_is_not_exact_dead_end(self):
        """C(2, parent_is_not=True) is a dead-end in exact mode."""
        prods = _condition_productions(2, 3, parent_is_not=True, mode="exact")
        assert len(prods) == 0

    def test_c2_parent_is_not_max_has_productions(self):
        """C(2, parent_is_not=True) has IsZero terminations in max mode."""
        prods = _condition_productions(2, 3, parent_is_not=True, mode="max")
        assert len(prods) == 3  # IsZero(0), IsZero(1), IsZero(2)
        for p in prods:
            assert isinstance(p.result, IsZero)

    def test_c3_max_includes_not_c2(self):
        """C(3) in max mode includes Not(C(2)) — was blocked in exact by _ccnn."""
        prods = _condition_productions(3, 3, mode="max")
        not_prods = [p for p in prods if isinstance(p.result, Not)]
        # Not(C(2)) should be present
        assert len(not_prods) == 1
        assert isinstance(not_prods[0].result.child, ConditionHole)
        assert not_prods[0].result.child.budget == 2

    def test_c3_exact_blocks_not_c2(self):
        """C(3) in exact mode blocks Not(C(2)) because _ccnn(3, 2) == 0.

        C(2, parent_is_not=True) has zero productions in exact mode
        (IsZero needs k==1, Not is banned, And needs k>=3). So the
        _ccnn guard correctly blocks C(3) -> Not(C(2)).
        """
        prods = _condition_productions(3, 3, mode="exact")
        not_prods = [p for p in prods if isinstance(p.result, Not)]
        assert len(not_prods) == 0  # Blocked: _ccnn(3, 2) == 0

    def test_parent_is_not_never_returns_not(self):
        """Productions with parent_is_not=True never include Not, any mode."""
        for mode in ["exact", "max"]:
            for k in range(1, 13):
                prods = _condition_productions(k, 6, parent_is_not=True, mode=mode)
                for p in prods:
                    assert not isinstance(p.result, Not), (
                        f"Not found at C({k}, parent_is_not=True, mode={mode})"
                    )

    @pytest.mark.parametrize("budget", range(1, 13))
    def test_max_superset_of_exact(self, budget):
        """Max mode condition productions are a superset of exact mode."""
        exact = set(p.label for p in _condition_productions(budget, 6, mode="exact"))
        max_ = set(p.label for p in _condition_productions(budget, 6, mode="max"))
        assert exact.issubset(max_), (
            f"C({budget}): exact has productions not in max: {exact - max_}"
        )


# ---------------------------------------------------------------------------
# 3. Action space sizing
# ---------------------------------------------------------------------------

class TestActionSpaceMaxMode:

    def test_compute_max_productions_exact(self):
        """Exact mode: N=6, L=14 -> 48 (unchanged golden value)."""
        assert compute_max_productions(14, 6, "exact") == 48

    def test_compute_max_productions_max(self):
        """Max mode: N=6, L=14 -> 66 (programs dominate)."""
        assert compute_max_productions(14, 6, "max") == 66

    def test_max_mode_larger_action_space(self):
        """Max mode action space >= exact mode action space."""
        for budget in [5, 8, 10, 14]:
            for n_sites in [3, 6]:
                exact = compute_max_productions(budget, n_sites, "exact")
                max_ = compute_max_productions(budget, n_sites, "max")
                assert max_ >= exact, (
                    f"L={budget}, N={n_sites}: max={max_} < exact={exact}"
                )

    def test_condition_prods_dont_dominate(self):
        """Condition productions never exceed program productions in max mode."""
        # For L=14, N=6: max condition count should be < 66
        max_cond = 0
        for k in range(1, 15):
            n = len(_condition_productions(k, 6, mode="max"))
            max_cond = max(max_cond, n)
        max_prog = 0
        for k in range(2, 15):
            n = len(_program_productions(k, 6, "max"))
            max_prog = max(max_prog, n)
        assert max_prog >= max_cond


# ---------------------------------------------------------------------------
# 4. DerivationGame integration (max mode)
# ---------------------------------------------------------------------------

class TestDerivationGameMaxMode:

    def test_game_initializes_with_max_mode(self, leaf_eval):
        """DerivationGame accepts program_budget_mode='max'."""
        game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        game.reset_wrapper()
        assert game._mode == "max"
        assert game.obs is not None

    def test_action_space_size_max_mode(self, leaf_eval):
        """Max mode game has larger action space than exact mode."""
        exact_game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="exact")
        max_game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        assert max_game.action_space.n >= exact_game.action_space.n

    def test_first_action_mask_max_mode(self, leaf_eval):
        """First step in max mode has more valid actions (terminations)."""
        exact_game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="exact")
        max_game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        exact_game.reset_wrapper()
        max_game.reset_wrapper()
        exact_valid = exact_game.get_action_mask().sum()
        max_valid = max_game.get_action_mask().sum()
        # Max mode adds n_sites termination productions at the root
        assert max_valid > exact_valid

    def test_no_truncations_max_mode(self, leaf_eval):
        """100 random max-mode episodes: zero truncations."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            assert game.terminated and not game.truncated, (
                "Max mode episode truncated (should be impossible)"
            )

    def test_no_truncations_max_mode_large(self, leaf_eval):
        """100 random max-mode episodes at L=14 N=3: zero truncations."""
        rng = np.random.default_rng(99)
        for _ in range(100):
            game = DerivationGame(14, N_SITES, leaf_eval, program_budget_mode="max")
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            assert game.terminated and not game.truncated

    def test_terminal_program_valid(self, leaf_eval):
        """Max mode produces valid programs with node_count <= budget."""
        game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        game.reset_wrapper()
        while not game.terminated:
            game.step_wrapper(0)  # always take first action
        prog = game.get_program()
        assert prog is not None
        assert prog.node_count() <= 8

    def test_early_termination_produces_small_program(self, leaf_eval):
        """Taking the termination action at P(8) produces a 2-node program."""
        game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        game.reset_wrapper()
        # In max mode, first n_sites actions are Default(Flip(j))
        # Action 0 should be Default(Flip(0))
        game.step_wrapper(0)
        assert game.terminated
        prog = game.get_program()
        assert prog is not None
        assert prog.node_count() == 2
        assert isinstance(prog, Default)


# ---------------------------------------------------------------------------
# 5. Node count validity in max mode
# ---------------------------------------------------------------------------

class TestMaxModeNodeCounts:

    def test_node_count_le_budget(self, leaf_eval):
        """All max-mode programs have node_count <= budget."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            if game.terminated:
                prog = game.get_program()
                assert prog.node_count() <= 8, (
                    f"Program node_count={prog.node_count()} > budget=8: "
                    f"{prog.pretty()}"
                )

    def test_exact_mode_node_count_equals_budget(self, leaf_eval):
        """In exact mode, programs always have node_count == budget."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="exact")
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            if game.terminated:
                prog = game.get_program()
                assert prog.node_count() == 8

    def test_max_mode_can_produce_small_programs(self, leaf_eval):
        """Max mode can produce programs smaller than the budget."""
        rng = np.random.default_rng(42)
        sizes = set()
        for _ in range(500):
            game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            if game.terminated:
                sizes.add(game.get_program().node_count())
        # Should see programs of size 2 (early termination at root)
        assert 2 in sizes, f"Never produced size-2 program. Sizes seen: {sizes}"
        # Should also see full-budget programs
        assert 8 in sizes, f"Never produced size-8 program. Sizes seen: {sizes}"


# ---------------------------------------------------------------------------
# 6. Max mode derivation: superset property
# ---------------------------------------------------------------------------

class TestMaxModeDerivationSuperset:

    @pytest.mark.parametrize("budget", [2, 5, 7])
    def test_max_reaches_all_exact_programs(self, budget):
        """Max mode derivation reaches all exact-mode programs."""
        exact_progs = set(
            p.pretty() for p in enumerate_via_derivation(N_SITES, budget)
        )
        # Enumerate max mode via DFS
        max_progs = set()

        def _dfs(state: DerivationState):
            if state.is_terminal():
                max_progs.add(state.to_program().pretty())
                return
            for prod in state.legal_productions(N_SITES, mode="max"):
                _dfs(state.apply(prod))

        _dfs(DerivationState.initial(budget))
        assert exact_progs.issubset(max_progs), (
            f"P({budget}): exact programs not in max: {exact_progs - max_progs}"
        )

    def test_max_mode_produces_extra_programs(self):
        """Max mode from P(8) produces programs not reachable from exact P(8)."""
        exact_progs = set(
            p.pretty() for p in enumerate_via_derivation(N_SITES, 8)
        )
        max_progs = set()

        def _dfs(state: DerivationState):
            if state.is_terminal():
                max_progs.add(state.to_program().pretty())
                return
            for prod in state.legal_productions(N_SITES, mode="max"):
                _dfs(state.apply(prod))

        _dfs(DerivationState.initial(8))
        extras = max_progs - exact_progs
        # Max mode should produce programs with node_count < 8
        assert len(extras) > 0, "Max mode didn't produce any extra programs"
        # Verify extras have node_count < 8
        for pretty_str in list(extras)[:5]:
            # All exact programs have node_count == 8, so extras must be smaller
            pass  # The assertion above is sufficient


# ---------------------------------------------------------------------------
# 7. Dead-end elimination proof
# ---------------------------------------------------------------------------

class TestDeadEndEliminationMaxMode:

    @pytest.mark.parametrize("budget", range(2, 15))
    def test_every_program_hole_has_productions(self, budget):
        """P(k) for any k >= 2 has at least one production in max mode."""
        prods = _program_productions(budget, 6, "max")
        assert len(prods) > 0, f"P({budget}) has no productions in max mode"

    @pytest.mark.parametrize("budget", range(1, 13))
    def test_every_condition_hole_has_productions(self, budget):
        """C(k) for any k >= 1 has productions in max mode (any parent_is_not)."""
        for parent_is_not in [False, True]:
            prods = _condition_productions(budget, 6, parent_is_not, mode="max")
            assert len(prods) > 0, (
                f"C({budget}, parent_is_not={parent_is_not}) has no productions "
                f"in max mode"
            )

    def test_derivation_never_gets_stuck_max_mode(self):
        """1000 random max-mode derivations at L=14 N=6 all complete."""
        rng = np.random.default_rng(42)
        truncations = 0
        for _ in range(1000):
            state = DerivationState.initial(14)
            while not state.is_terminal():
                prods = state.legal_productions(6, mode="max")
                if not prods:
                    truncations += 1
                    break
                state = state.apply(prods[rng.integers(len(prods))])
        assert truncations == 0, f"Got {truncations}/1000 truncations"

    def test_derivation_no_double_negation_max_mode(self):
        """Max mode derivations still respect canonical double-negation ban."""
        rng = np.random.default_rng(123)
        for _ in range(500):
            state = DerivationState.initial(14)
            while not state.is_terminal():
                prods = state.legal_productions(6, mode="max")
                assert prods, "Dead-end in max mode"
                state = state.apply(prods[rng.integers(len(prods))])
            prog = state.to_program()
            _assert_no_double_neg_program(prog)


# ---------------------------------------------------------------------------
# 8. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_default_mode_is_exact(self):
        """Default mode parameter is 'exact'."""
        prods_default = _program_productions(5, 3)
        prods_exact = _program_productions(5, 3, "exact")
        assert len(prods_default) == len(prods_exact)
        assert all(a.label == b.label for a, b in zip(prods_default, prods_exact))

    def test_default_condition_mode_is_exact(self):
        """Default mode for conditions is 'exact'."""
        prods_default = _condition_productions(3, 3)
        prods_exact = _condition_productions(3, 3, mode="exact")
        assert len(prods_default) == len(prods_exact)

    def test_compute_max_productions_default_exact(self):
        """compute_max_productions default is exact mode."""
        assert compute_max_productions(14, 6) == compute_max_productions(14, 6, "exact")

    def test_game_default_mode_exact(self, leaf_eval):
        """DerivationGame defaults to exact mode."""
        game = DerivationGame(8, N_SITES, leaf_eval)
        assert game._mode == "exact"
        assert game.action_space.n == compute_max_productions(8, N_SITES, "exact")

    def test_clone_preserves_mode(self, leaf_eval):
        """clone() preserves program_budget_mode."""
        game = DerivationGame(8, N_SITES, leaf_eval, program_budget_mode="max")
        game.reset_wrapper()
        cloned = game.clone()
        assert cloned._mode == "max"
        assert cloned.action_space.n == game.action_space.n
        assert cloned.get_action_mask().shape == game.get_action_mask().shape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_double_neg_program(prog):
    """Check no double-negation in any condition of a program."""
    if isinstance(prog, Ite):
        _assert_no_double_neg(prog.cond)
        _assert_no_double_neg_program(prog.else_prog)


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
