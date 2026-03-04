"""
Tests for domain-aware macro productions (PickRule, MoveRule) and
condition budget cap for the Doors environment.

Tests:
1. Macro production correctness (AST structure, node counts)
2. Template grouping (K variants -> 1 template each)
3. Condition budget cap limits Ite templates
4. max_factored with macros + cap
5. Known-good D=10 program -> solve_rate=1.0
6. Budget guards (macro unavailable when budget too small)
7. FactoredDerivationGame integration with macros
8. Clone preserves macro configuration
"""

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.budget_grammar import ProgramHole, ConditionHole
from alphazeropp.synthesis.derivation import (
    Production, DerivationState, _program_productions,
)
from alphazeropp.synthesis.factored_derivation_game import (
    FactoredDerivationGame, compute_max_factored_actions,
    _structure_key, _group_by_structure,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.doors_macros import (
    doors_macro_productions, make_macro_fn,
    PICK_RULE_COST, MOVE_RULE_COST,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyLeafEval:
    def __call__(self, program):
        return 0.0
    def stats(self):
        return {}


@pytest.fixture
def doors_cfg_d2():
    return DoorsGameConfig(num_rooms=2, locs_per_room=2)


@pytest.fixture
def doors_cfg_d10():
    params = compute_doors_derived_params(10, 2)
    return DoorsGameConfig(num_rooms=10, locs_per_room=2, horizon=params["horizon"])


@pytest.fixture
def leaf_eval():
    return DummyLeafEval()


# ---------------------------------------------------------------------------
# Macro production structure
# ---------------------------------------------------------------------------

class TestMacroProductionStructure:
    """Verify macro productions create correct AST subtrees."""

    def test_pick_rule_structure(self, doors_cfg_d2):
        """PickRule produces Ite(And(Not(IsZero), Not(IsZero)), Flip, ProgramHole)."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        pick_prods = [p for p in prods if p.macro_key == ("PickRule",)]
        assert len(pick_prods) == doors_cfg_d2.K  # 1 key for D=2

        p = pick_prods[0]
        assert isinstance(p.result, Ite)
        assert isinstance(p.result.cond, And)
        assert isinstance(p.result.cond.left, Not)
        assert isinstance(p.result.cond.left.child, IsZero)
        assert isinstance(p.result.cond.right, Not)
        assert isinstance(p.result.cond.right.child, IsZero)
        assert isinstance(p.result.action, Flip)
        assert isinstance(p.result.else_prog, ProgramHole)

    def test_pick_rule_indices_d2(self, doors_cfg_d2):
        """PickRule(0) for D=2: loc=1, avail=6, action=4."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        pick = [p for p in prods if p.macro_key == ("PickRule",)][0]
        # D=2: M=4, D=2, K=1, key_loc=[1], key_unlocks=[1]
        assert pick.result.cond.left.child.index == 1   # key_loc[0]
        assert pick.result.cond.right.child.index == 6   # M+D+0 = 4+2+0
        assert pick.result.action.index == 4              # M+0 = PICK(0)
        assert pick.result.else_prog.budget == 18 - PICK_RULE_COST

    def test_move_rule_structure(self, doors_cfg_d2):
        """MoveRule produces Ite(IsZero, Flip, ProgramHole)."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        move_prods = [p for p in prods if p.macro_key == ("MoveRule",)]
        assert len(move_prods) == doors_cfg_d2.K

        m = move_prods[0]
        assert isinstance(m.result, Ite)
        assert isinstance(m.result.cond, IsZero)
        assert isinstance(m.result.action, Flip)
        assert isinstance(m.result.else_prog, ProgramHole)

    def test_move_rule_indices_d2(self, doors_cfg_d2):
        """MoveRule(0) for D=2: lock=5, action=1."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        move = [p for p in prods if p.macro_key == ("MoveRule",)][0]
        # key_unlocks[0] = 1, M=4, so lock_idx = M+1 = 5
        assert move.result.cond.index == 5               # M + key_unlocks[0]
        assert move.result.action.index == 1              # key_loc[0]
        assert move.result.else_prog.budget == 18 - MOVE_RULE_COST

    def test_pick_rule_indices_d10(self, doors_cfg_d10):
        """PickRule indices correct for all 9 keys in D=10."""
        prods = doors_macro_productions(138, doors_cfg_d10)
        picks = [p for p in prods if p.macro_key == ("PickRule",)]
        assert len(picks) == 9

        for k, p in enumerate(picks):
            expected_loc = doors_cfg_d10.key_loc[k]       # 2k+1
            expected_avail = 20 + 10 + k                   # M+D+k
            expected_action = 20 + k                       # M+k = PICK(k)
            assert p.result.cond.left.child.index == expected_loc
            assert p.result.cond.right.child.index == expected_avail
            assert p.result.action.index == expected_action

    def test_move_rule_indices_d10(self, doors_cfg_d10):
        """MoveRule indices correct for all 9 keys in D=10."""
        prods = doors_macro_productions(138, doors_cfg_d10)
        moves = [p for p in prods if p.macro_key == ("MoveRule",)]
        assert len(moves) == 9

        for k, m in enumerate(moves):
            expected_lock = 20 + doors_cfg_d10.key_unlocks[k]  # M + room
            expected_move = doors_cfg_d10.key_loc[k]           # key location
            assert m.result.cond.index == expected_lock
            assert m.result.action.index == expected_move

    def test_node_count(self, doors_cfg_d2):
        """PickRule = 7 nodes, MoveRule = 3 nodes (excluding ProgramHole)."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        pick = [p for p in prods if p.macro_key == ("PickRule",)][0]
        move = [p for p in prods if p.macro_key == ("MoveRule",)][0]

        def count_real_nodes(node):
            """Count non-hole AST nodes."""
            if isinstance(node, ProgramHole):
                return 0
            if isinstance(node, Ite):
                return 1 + count_real_nodes(node.cond) + count_real_nodes(node.action) + count_real_nodes(node.else_prog)
            if isinstance(node, And):
                return 1 + count_real_nodes(node.left) + count_real_nodes(node.right)
            if isinstance(node, Not):
                return 1 + count_real_nodes(node.child)
            if isinstance(node, (IsZero, Flip, Default)):
                return 1
            return 0

        assert count_real_nodes(pick.result) == PICK_RULE_COST
        assert count_real_nodes(move.result) == MOVE_RULE_COST


# ---------------------------------------------------------------------------
# Template grouping
# ---------------------------------------------------------------------------

class TestMacroTemplateGrouping:
    """Verify _structure_key and _group_by_structure handle macros."""

    def test_macro_key_used_by_structure_key(self, doors_cfg_d2):
        """_structure_key returns macro_key when set."""
        prods = doors_macro_productions(18, doors_cfg_d2)
        for p in prods:
            assert p.macro_key is not None
            assert _structure_key(p) == p.macro_key

    def test_pick_variants_group_together(self, doors_cfg_d10):
        """All 9 PickRule variants share one StructureTemplate."""
        prods = doors_macro_productions(138, doors_cfg_d10)
        templates = _group_by_structure(prods)
        pick_templates = [t for t in templates if t.key == ("PickRule",)]
        assert len(pick_templates) == 1
        assert pick_templates[0].needs_parameter is True
        assert len(pick_templates[0].productions) == 9

    def test_move_variants_group_together(self, doors_cfg_d10):
        """All 9 MoveRule variants share one StructureTemplate."""
        prods = doors_macro_productions(138, doors_cfg_d10)
        templates = _group_by_structure(prods)
        move_templates = [t for t in templates if t.key == ("MoveRule",)]
        assert len(move_templates) == 1
        assert move_templates[0].needs_parameter is True
        assert len(move_templates[0].productions) == 9

    def test_macros_dont_collide_with_base(self, doors_cfg_d2):
        """Macro templates have distinct keys from base Ite templates."""
        base = _program_productions(18, 7, "max", n_actions=6)
        macros = doors_macro_productions(18, doors_cfg_d2)
        combined = base + macros
        templates = _group_by_structure(combined)
        keys = [t.key for t in templates]
        assert ("PickRule",) in keys
        assert ("MoveRule",) in keys
        assert ("Default",) in keys
        # Base Ite keys start with "Ite"
        ite_keys = [k for k in keys if k[0] == "Ite"]
        assert len(ite_keys) > 0


# ---------------------------------------------------------------------------
# Condition budget cap
# ---------------------------------------------------------------------------

class TestConditionBudgetCap:
    """Test max_condition_budget parameter on _program_productions."""

    def test_cap_limits_ite_templates(self):
        """With cap=5, only condition budgets 1..5 are generated."""
        prods = _program_productions(138, 39, "max", n_actions=30,
                                     max_condition_budget=5)
        ite_prods = [p for p in prods if isinstance(p.result, Ite)]
        cond_budgets = set()
        for p in ite_prods:
            cond_budgets.add(p.result.cond.budget)
        assert max(cond_budgets) == 5
        assert min(cond_budgets) == 1

    def test_cap_reduces_template_count(self):
        """Cap=12 gives far fewer templates than uncapped."""
        uncapped = _program_productions(138, 39, "max", n_actions=30)
        capped = _program_productions(138, 39, "max", n_actions=30,
                                      max_condition_budget=12)
        uncapped_templates = _group_by_structure(uncapped)
        capped_templates = _group_by_structure(capped)
        # Uncapped: 1 Default + 134 Ite = 135 templates
        # Capped:   1 Default + 12 Ite = 13 templates
        assert len(uncapped_templates) > 100
        assert len(capped_templates) == 13

    def test_cap_none_preserves_behavior(self):
        """cap=None gives same results as no cap."""
        prods_none = _program_productions(18, 7, "max", n_actions=6,
                                          max_condition_budget=None)
        prods_orig = _program_productions(18, 7, "max", n_actions=6)
        assert len(prods_none) == len(prods_orig)

    def test_cap_at_small_budget(self):
        """At small budgets, cap doesn't change anything (natural range is smaller)."""
        prods_capped = _program_productions(10, 7, "max", n_actions=6,
                                            max_condition_budget=12)
        prods_orig = _program_productions(10, 7, "max", n_actions=6)
        assert len(prods_capped) == len(prods_orig)


# ---------------------------------------------------------------------------
# Budget guards
# ---------------------------------------------------------------------------

class TestMacroBudgetGuards:
    """Macros should not be offered when budget is too small."""

    def test_pick_unavailable_below_9(self, doors_cfg_d2):
        """PickRule needs budget >= 9 (7 cost + 2 min remainder)."""
        for budget in range(1, 9):
            prods = doors_macro_productions(budget, doors_cfg_d2)
            picks = [p for p in prods if p.macro_key == ("PickRule",)]
            assert len(picks) == 0, f"PickRule should not be available at budget={budget}"

    def test_pick_available_at_9(self, doors_cfg_d2):
        prods = doors_macro_productions(9, doors_cfg_d2)
        picks = [p for p in prods if p.macro_key == ("PickRule",)]
        assert len(picks) == doors_cfg_d2.K

    def test_move_unavailable_below_5(self, doors_cfg_d2):
        """MoveRule needs budget >= 5 (3 cost + 2 min remainder)."""
        for budget in range(1, 5):
            prods = doors_macro_productions(budget, doors_cfg_d2)
            moves = [p for p in prods if p.macro_key == ("MoveRule",)]
            assert len(moves) == 0, f"MoveRule should not be available at budget={budget}"

    def test_move_available_at_5(self, doors_cfg_d2):
        prods = doors_macro_productions(5, doors_cfg_d2)
        moves = [p for p in prods if p.macro_key == ("MoveRule",)]
        assert len(moves) == doors_cfg_d2.K


# ---------------------------------------------------------------------------
# compute_max_factored_actions with macros
# ---------------------------------------------------------------------------

class TestMaxFactoredWithMacros:
    """Verify max_factored computation with macros + cap."""

    def test_d10_with_macros_and_cap(self, doors_cfg_d10):
        """D=10 with macros + cap=12 should give max_factored ~ 39."""
        macro_fn = make_macro_fn(doors_cfg_d10)
        result = compute_max_factored_actions(
            138, 39, mode="max", allow_and=True,
            n_actions=30,
            macro_productions_fn=macro_fn,
            max_condition_budget=12,
        )
        # Structures: 12 Ite + 1 Default + 2 macros = 15
        # Params: max(30 Flip, 9 keys, 39 IsZero) = 39
        # max_factored = max(15, 39) = 39
        assert result == 39

    def test_d10_no_macros_no_cap(self):
        """D=10 without macros or cap should give ~135."""
        result = compute_max_factored_actions(
            138, 39, mode="max", allow_and=True, n_actions=30,
        )
        assert result > 100  # 135 expected

    def test_d2_backward_compatible(self):
        """D=2 without macros (default) gives same result as before."""
        result = compute_max_factored_actions(
            18, 7, mode="max", allow_and=True, n_actions=6,
        )
        assert result == 15  # known D=2 factored action size


# ---------------------------------------------------------------------------
# Known-good D=10 program
# ---------------------------------------------------------------------------

class TestOptimalD10Program:
    """Construct and verify the optimal D=10 program."""

    def _build_optimal_d10(self, doors_cfg):
        """Build the 92-node optimal D=10 program manually."""
        M = doors_cfg.M     # 20
        D = doors_cfg.D     # 10
        K = doors_cfg.K     # 9

        def pick_rule(k, else_prog):
            return Ite(
                And(Not(IsZero(doors_cfg.key_loc[k])),
                    Not(IsZero(M + D + k))),
                Flip(M + k),
                else_prog,
            )

        def move_rule(k, else_prog):
            return Ite(
                IsZero(M + doors_cfg.key_unlocks[k]),
                Flip(doors_cfg.key_loc[k]),
                else_prog,
            )

        # Build from inside out: Default, then wrap with MOVE and PICK
        goal_loc = doors_cfg.goal_loc  # 19
        prog = Default(Flip(goal_loc))
        for k in range(K - 1, -1, -1):
            prog = move_rule(k, prog)
            prog = pick_rule(k, prog)
        return prog

    def test_optimal_program_node_count(self, doors_cfg_d10):
        """Optimal D=10 program should be 92 nodes."""
        prog = self._build_optimal_d10(doors_cfg_d10)
        assert prog.node_count() == 92

    def test_optimal_program_solves_d10(self, doors_cfg_d10):
        """Optimal D=10 program achieves solve_rate=1.0."""
        from alphazeropp.synthesis.interpreter import run_policy_episode

        prog = self._build_optimal_d10(doors_cfg_d10)
        env = doors_cfg_d10.make_env(n_sites=39)
        x0 = doors_initial_state(doors_cfg_d10)

        result = run_policy_episode(
            env, prog, x0,
            is_solved=lambda obs: doors_cfg_d10.is_solved(obs),
        )
        assert result.solved, (
            f"Optimal D=10 program should solve, got steps={result.total_env_steps}, "
            f"reward={result.cumulative_reward:.3f}"
        )
        assert result.total_env_steps == 19, (
            f"Expected 19 optimal steps, got {result.total_env_steps}"
        )

    def test_optimal_matches_macro_output(self, doors_cfg_d10):
        """Macros produce the same AST subtrees as the manual program."""
        prods = doors_macro_productions(138, doors_cfg_d10)
        picks = [p for p in prods if p.macro_key == ("PickRule",)]
        moves = [p for p in prods if p.macro_key == ("MoveRule",)]

        prog = self._build_optimal_d10(doors_cfg_d10)
        # First branch is PickRule(0) - check condition matches
        assert isinstance(prog, Ite)
        assert prog.cond.pretty() == picks[0].result.cond.pretty()
        assert prog.action.index == picks[0].result.action.index


# ---------------------------------------------------------------------------
# FactoredDerivationGame integration
# ---------------------------------------------------------------------------

class TestFactoredGameWithMacros:
    """Test FactoredDerivationGame with macro_productions_fn."""

    def test_macro_templates_appear(self, doors_cfg_d2, leaf_eval):
        """After reset, macro templates are available at root."""
        macro_fn = make_macro_fn(doors_cfg_d2)
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, macro_productions_fn=macro_fn,
        )
        game.reset()
        keys = [t.key for t in game._structure_templates]
        assert ("PickRule",) in keys
        assert ("MoveRule",) in keys

    def test_macro_step_applies_correctly(self, doors_cfg_d10, leaf_eval):
        """Choosing a macro structure + parameter applies the full subtree.

        Uses D=10 (K=9) so PickRule has 9 variants and needs_parameter=True.
        D=2 has K=1 so PickRule would apply immediately (no parameter phase).
        """
        macro_fn = make_macro_fn(doors_cfg_d10)
        game = FactoredDerivationGame(
            138, 39, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=30, macro_productions_fn=macro_fn,
            max_condition_budget=12,
        )
        game.reset()

        # Find PickRule template index
        pick_idx = None
        for i, t in enumerate(game._structure_templates):
            if t.key == ("PickRule",):
                pick_idx = i
                break
        assert pick_idx is not None

        # Choose PickRule structure — enters parameter phase (9 key variants)
        obs, reward, term, trunc, info = game.step(pick_idx)
        assert game._phase == "parameter"
        assert game._pending_template.key == ("PickRule",)

        # Choose key_id=0
        obs, reward, term, trunc, info = game.step(0)
        assert game._phase == "structure"
        assert "production" in info
        # After PickRule applied, AST has an Ite with complete condition
        assert "PickRule" in info["production"].label

    def test_complete_derivation_with_macros(self, doors_cfg_d2, leaf_eval):
        """Can complete a full derivation using macros."""
        macro_fn = make_macro_fn(doors_cfg_d2)
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, macro_productions_fn=macro_fn,
        )
        game.reset()

        for _ in range(200):
            mask = game.get_action_mask()
            if not mask.any():
                break
            valid = np.where(mask)[0]
            action = valid[0]
            obs, reward, term, trunc, info = game.step(action)
            if term or trunc:
                break

        assert term or trunc, "Should reach terminal or dead-end"

    def test_clone_preserves_macros(self, doors_cfg_d2, leaf_eval):
        """clone() forwards macro_productions_fn and max_condition_budget."""
        macro_fn = make_macro_fn(doors_cfg_d2)
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, macro_productions_fn=macro_fn, max_condition_budget=12,
        )
        game.reset()

        clone = game.clone()
        assert clone._macro_productions_fn is macro_fn
        assert clone._max_condition_budget == 12
        # Verify macros appear in cloned game
        keys = [t.key for t in clone._structure_templates]
        assert ("PickRule",) in keys

    def test_action_mask_with_macros(self, doors_cfg_d2, leaf_eval):
        """Action mask covers macro templates and their parameters."""
        macro_fn = make_macro_fn(doors_cfg_d2)
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, macro_productions_fn=macro_fn,
        )
        game.reset()
        mask = game.get_action_mask()
        n_templates = len(game._structure_templates)
        assert mask[:n_templates].all()
        assert not mask[n_templates:].any()

    def test_d10_game_initializes(self, doors_cfg_d10, leaf_eval):
        """D=10 game with macros + cap initializes without error."""
        macro_fn = make_macro_fn(doors_cfg_d10)
        game = FactoredDerivationGame(
            138, 39, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=30, macro_productions_fn=macro_fn,
            max_condition_budget=12,
        )
        game.reset()
        assert game.action_space.n == 39
        assert len(game._structure_templates) <= 39

    def test_macros_only_at_program_holes(self, doors_cfg_d2, leaf_eval):
        """Macros are not offered when the leftmost hole is a ConditionHole."""
        macro_fn = make_macro_fn(doors_cfg_d2)
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, macro_productions_fn=macro_fn,
        )
        game.reset()

        # Play a base Ite (not a macro) to get to a ConditionHole
        for i, t in enumerate(game._structure_templates):
            if t.key[0] == "Ite":
                game.step(i)
                if game._phase == "parameter":
                    game.step(0)  # pick any Flip parameter
                break

        # Now at a ConditionHole — macros should NOT appear
        if game._structure_templates:
            keys = [t.key for t in game._structure_templates]
            assert ("PickRule",) not in keys
            assert ("MoveRule",) not in keys
