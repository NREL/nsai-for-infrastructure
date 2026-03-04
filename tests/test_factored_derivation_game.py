"""
Tests for FactoredDerivationGame.

Tests:
1. Structure template grouping correctness
2. Reachability equivalence with DerivationGame
3. Branching sanity (factored < flat)
4. Action space invariant (constant across steps)
5. Phase transitions and immediate-apply for Not/And
6. Stash/unstash and clone preserve phase state
7. Grammar pruning integration (n_actions, one_hot_groups)
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
)
from alphazeropp.synthesis.derivation_game import (
    DerivationGame, compute_max_productions,
)
from alphazeropp.synthesis.factored_derivation_game import (
    FactoredDerivationGame,
    StructureTemplate,
    _structure_key,
    _group_by_structure,
    compute_max_factored_actions,
)


# ---------------------------------------------------------------------------
# Dummy leaf evaluator
# ---------------------------------------------------------------------------

class DummyLeafEval:
    """Leaf evaluator that returns 0.0 for any program."""
    def __call__(self, program):
        return 0.0
    def stats(self):
        return {}


@pytest.fixture
def leaf_eval():
    return DummyLeafEval()


# ---------------------------------------------------------------------------
# Structure template helpers
# ---------------------------------------------------------------------------

class TestStructureKey:
    """Test _structure_key extracts correct template identity."""

    def test_default(self):
        assert _structure_key(Production("P", 2, Default(Flip(0)), "")) == ("Default",)
        assert _structure_key(Production("P", 2, Default(Flip(5)), "")) == ("Default",)

    def test_ite(self):
        result = Ite(ConditionHole(1), Flip(0), ProgramHole(5))
        assert _structure_key(Production("P", 8, result, "")) == ("Ite", 1, 5)

    def test_iszero(self):
        assert _structure_key(Production("C", 1, IsZero(3), "")) == ("IsZero",)

    def test_not(self):
        result = Not(ConditionHole(2, parent_is_not=True))
        assert _structure_key(Production("C", 3, result, "")) == ("Not", 2)

    def test_and(self):
        result = And(ConditionHole(1), ConditionHole(2))
        assert _structure_key(Production("C", 4, result, "")) == ("And", 1, 2)


class TestGroupByStructure:
    """Test _group_by_structure groups correctly."""

    def test_program_productions_grouped(self):
        """Program productions for P(5) should group by structure."""
        prods = _program_productions(5, n_sites=4, mode="max")
        templates = _group_by_structure(prods)

        # P(5) in max mode: Default(Flip(?)) + Ite(C(1), Flip(?), P(2))
        assert len(templates) == 2

        default_t = templates[0]
        assert default_t.key == ("Default",)
        assert default_t.needs_parameter is True
        assert len(default_t.productions) == 4  # 4 Flip indices

        ite_t = templates[1]
        assert ite_t.key == ("Ite", 1, 2)
        assert ite_t.needs_parameter is True
        assert len(ite_t.productions) == 4

    def test_condition_productions_mixed(self):
        """Condition productions have both parameterized and non-parameterized."""
        prods = _condition_productions(3, n_sites=4, mode="max",
                                        allow_and=True, allow_not=True)
        templates = _group_by_structure(prods)

        # C(3) max mode: IsZero(?), Not(C(2)), And(C(1), C(1))
        keys = [t.key for t in templates]
        assert ("IsZero",) in keys
        assert ("Not", 2) in keys
        assert ("And", 1, 1) in keys

        # IsZero needs parameter, Not and And do not
        iszero_t = [t for t in templates if t.key == ("IsZero",)][0]
        assert iszero_t.needs_parameter is True
        assert len(iszero_t.productions) == 4

        not_t = [t for t in templates if t.key[0] == "Not"][0]
        assert not_t.needs_parameter is False
        assert len(not_t.productions) == 1

        and_t = [t for t in templates if t.key[0] == "And"][0]
        assert and_t.needs_parameter is False
        assert len(and_t.productions) == 1


# ---------------------------------------------------------------------------
# Action space computation
# ---------------------------------------------------------------------------

class TestComputeMaxFactoredActions:
    """Test compute_max_factored_actions."""

    def test_smaller_than_flat(self):
        """Factored action space should be smaller than flat."""
        flat = compute_max_productions(18, 7, mode="max", allow_and=True)
        factored = compute_max_factored_actions(18, 7, mode="max", allow_and=True)
        assert factored < flat

    def test_with_n_actions(self):
        """n_actions should reduce parameter count."""
        full = compute_max_factored_actions(18, 7, mode="max", allow_and=True)
        restricted = compute_max_factored_actions(
            18, 7, mode="max", allow_and=True, n_actions=6)
        assert restricted <= full

    def test_doors_d2(self):
        """D=2 config: factored action space ~ 15."""
        factored = compute_max_factored_actions(
            18, 7, mode="max", allow_and=True, n_actions=6)
        assert factored <= 20  # budget-3 = 15, but also check param

    def test_doors_d10(self):
        """D=10 config: factored action space <= 120."""
        factored = compute_max_factored_actions(
            102, 39, mode="max", allow_and=True, n_actions=30)
        assert factored <= 120


# ---------------------------------------------------------------------------
# Game flow
# ---------------------------------------------------------------------------

class TestGameFlow:
    """Test FactoredDerivationGame basic flow."""

    def test_reset(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        obs, info = game.reset()
        assert obs.shape == (2 * 18 + 2,)
        assert game._phase == "structure"

    def test_action_space_constant(self, leaf_eval):
        """Action space size is constant across all steps."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        expected_n = game.action_space.n
        mask = game.get_action_mask()
        assert mask.shape == (expected_n,)

        # Take a few steps
        for _ in range(5):
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            obs, reward, term, trunc, info = game.step(valid[0])
            if term or trunc:
                break
            mask = game.get_action_mask()
            assert mask.shape == (expected_n,)

    def test_structure_to_parameter_transition(self, leaf_eval):
        """Choosing a parameterized structure enters parameter phase."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()

        # Root is ProgramHole(18). All structures need parameters.
        assert game._phase == "structure"
        templates = game._structure_templates
        assert all(t.needs_parameter for t in templates)

        # Pick first structure → should enter parameter phase
        obs, reward, term, trunc, info = game.step(0)
        assert game._phase == "parameter"
        assert reward == 0.0
        assert not term
        assert not trunc

    def test_parameter_applies_production(self, leaf_eval):
        """Choosing a parameter applies the full production."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()

        # Enter parameter phase
        game.step(0)
        assert game._phase == "parameter"

        # Pick a parameter → should apply production and return to structure
        obs, reward, term, trunc, info = game.step(0)
        assert game._phase == "structure"
        assert "production" in info

    def test_no_parameter_immediate_apply(self, leaf_eval):
        """Not/And structures apply immediately without parameter phase."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()

        # Play until we hit a condition hole with Not/And available.
        # Strategy: pick the LAST structure (largest Ite expansion) to
        # create condition holes with high budget, then look for Not/And.
        found_non_param = False
        for _ in range(30):
            if game._phase == "structure":
                templates = game._structure_templates
                # Find a non-parameterized template
                non_param_idx = None
                for i, t in enumerate(templates):
                    if not t.needs_parameter:
                        non_param_idx = i
                        break

                if non_param_idx is not None:
                    obs, _, term, trunc, info = game.step(non_param_idx)
                    # Should stay in structure phase (immediate apply)
                    assert game._phase == "structure"
                    assert "production" in info
                    found_non_param = True
                    break
                else:
                    # Pick last structure (Ite with large condition budget)
                    # to reach condition holes with Not/And
                    game.step(len(templates) - 1)
                    if game._phase == "parameter":
                        game.step(0)
            elif game._phase == "parameter":
                game.step(0)

            mask = game.get_action_mask()
            if not mask.any():
                break

        assert found_non_param, "No non-parameterized structure found in 30 steps"

    def test_complete_derivation(self, leaf_eval):
        """Can complete a full derivation to terminal state."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()

        for _ in range(200):
            mask = game.get_action_mask()
            if not mask.any():
                break
            valid = np.where(mask)[0]
            obs, reward, term, trunc, info = game.step(valid[0])
            if term or trunc:
                break

        assert term or trunc, "Derivation did not terminate in 200 steps"
        if term:
            assert game.get_program() is not None


# ---------------------------------------------------------------------------
# Reachability equivalence
# ---------------------------------------------------------------------------

class TestReachabilityEquivalence:
    """Verify factored game produces identical programs to flat game."""

    def _play_flat_episode(self, flat_game, rng):
        """Play a random episode in the flat game, return (program, productions)."""
        flat_game.reset()
        productions = []
        for _ in range(200):
            mask = flat_game.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = rng.choice(valid)
            obs, reward, term, trunc, info = flat_game.step(action)
            productions.append(info["production"])
            if term or trunc:
                break
        program = flat_game.get_program()
        return program, productions

    def _replay_in_factored(self, factored_game, productions):
        """Replay a sequence of productions in the factored game."""
        factored_game.reset()
        for prod in productions:
            # Find the structure template matching this production
            assert factored_game._phase == "structure"
            prod_key = _structure_key(prod)

            # Find template index
            template_idx = None
            for i, t in enumerate(factored_game._structure_templates):
                if t.key == prod_key:
                    template_idx = i
                    break
            assert template_idx is not None, (
                f"Template {prod_key} not found in "
                f"{[t.key for t in factored_game._structure_templates]}"
            )

            factored_game.step(template_idx)

            if factored_game._phase == "parameter":
                # Find parameter index
                param_idx = None
                for i, p in enumerate(factored_game._parameter_productions):
                    if p.result == prod.result:
                        param_idx = i
                        break
                assert param_idx is not None, (
                    f"Parameter {prod.result} not found in productions"
                )
                factored_game.step(param_idx)

        return factored_game.get_program()

    def test_reachability_equivalence(self, leaf_eval):
        """Random flat episodes produce identical programs in factored game."""
        flat = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        factored = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)

        rng = np.random.default_rng(42)
        n_episodes = 20
        n_matched = 0

        for _ in range(n_episodes):
            program_flat, productions = self._play_flat_episode(flat, rng)
            if program_flat is None:
                continue  # skip dead-end episodes

            program_factored = self._replay_in_factored(factored, productions)
            assert program_factored is not None
            assert program_flat.pretty() == program_factored.pretty()
            n_matched += 1

        assert n_matched >= 5, f"Only {n_matched} episodes completed"

    def test_reachability_with_pruning(self, leaf_eval):
        """Reachability holds with n_actions and one_hot_groups."""
        flat = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, one_hot_groups=[[0, 1, 2, 3]])
        factored = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6, one_hot_groups=[[0, 1, 2, 3]])

        rng = np.random.default_rng(123)
        n_matched = 0

        for _ in range(20):
            program_flat, productions = self._play_flat_episode(flat, rng)
            if program_flat is None:
                continue

            program_factored = self._replay_in_factored(factored, productions)
            assert program_factored is not None
            assert program_flat.pretty() == program_factored.pretty()
            n_matched += 1

        assert n_matched >= 5


# ---------------------------------------------------------------------------
# Branching sanity
# ---------------------------------------------------------------------------

class TestBranchingSanity:
    """Verify factored game has lower branching than flat game."""

    def test_root_branching_d2(self, leaf_eval):
        """D=2 root branching should be much less than flat."""
        flat = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)
        factored = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)

        flat.reset()
        factored.reset()

        flat_branching = flat.get_action_mask().sum()
        factored_branching = factored.get_action_mask().sum()

        assert factored_branching < flat_branching
        assert factored_branching <= 20  # P(18) has ~15 structure templates

    def test_action_space_smaller(self, leaf_eval):
        """Factored action space should be smaller than flat."""
        flat = DerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)
        factored = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)

        assert factored.action_space.n < flat.action_space.n


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestStateManagement:
    """Test stash/unstash and clone preserve phase state."""

    def test_stash_unstash_structure_phase(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        assert game._phase == "structure"

        stashed = game.stash_state()
        game.step(0)  # enter parameter phase
        assert game._phase == "parameter"

        game.unstash_state(stashed)
        assert game._phase == "structure"

    def test_stash_unstash_parameter_phase(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        game.step(0)  # enter parameter phase
        assert game._phase == "parameter"
        pending_key = game._pending_template.key

        stashed = game.stash_state()
        game.step(0)  # apply production
        assert game._phase == "structure"

        game.unstash_state(stashed)
        assert game._phase == "parameter"
        assert game._pending_template.key == pending_key

    def test_clone_structure_phase(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()

        cloned = game.clone()
        assert cloned._phase == "structure"
        assert cloned.action_space.n == game.action_space.n
        assert len(cloned._structure_templates) == len(game._structure_templates)

    def test_clone_parameter_phase(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        game.step(0)  # enter parameter phase

        cloned = game.clone()
        assert cloned._phase == "parameter"
        assert cloned._pending_template.key == game._pending_template.key
        assert len(cloned._parameter_productions) == len(game._parameter_productions)

    def test_clone_independence(self, leaf_eval):
        """Stepping the clone should not affect the original."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        game.step(0)  # enter parameter phase
        original_phase = game._phase
        original_key = game._pending_template.key

        cloned = game.clone()
        cloned.step(0)  # apply production in clone

        assert game._phase == original_phase
        assert game._pending_template.key == original_key

    def test_hashable_obs_includes_phase(self, leaf_eval):
        """hashable_obs should differ between structure and parameter phases."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        hash_structure = game.hashable_obs

        game.step(0)
        hash_parameter = game.hashable_obs

        assert hash_structure != hash_parameter
        assert "|structure|" in hash_structure
        assert "|parameter|" in hash_parameter


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

class TestObservationEncoding:
    """Test observation shape and phase encoding."""

    def test_obs_shape(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        obs, _ = game.reset()
        assert obs.shape == (2 * 18 + 2,)

    def test_phase_encoding_structure(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        obs, _ = game.reset()
        assert obs[2 * 18] == 0.0  # phase_id = structure
        assert obs[2 * 18 + 1] == 0.0  # no pending structure

    def test_phase_encoding_parameter(self, leaf_eval):
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True)
        game.reset()
        obs, _, _, _, _ = game.step(0)
        assert obs[2 * 18] == 1.0  # phase_id = parameter
        assert obs[2 * 18 + 1] > 0  # pending structure id


# ---------------------------------------------------------------------------
# Grammar pruning integration
# ---------------------------------------------------------------------------

class TestPruningIntegration:
    """Test that n_actions and one_hot_groups work with factored game."""

    def test_n_actions_restricts_parameters(self, leaf_eval):
        """With n_actions=6, Flip parameter choices should be ≤ 6."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max", allow_and=True,
            n_actions=6)
        game.reset()

        # Enter parameter phase for a program structure
        game.step(0)
        assert game._phase == "parameter"

        # All parameter productions should have Flip index < 6
        for prod in game._parameter_productions:
            if isinstance(prod.result, Default):
                assert prod.result.action.index < 6
            elif isinstance(prod.result, Ite):
                assert prod.result.action.index < 6

    def test_clone_preserves_pruning(self, leaf_eval):
        """Clone should preserve n_actions and one_hot_groups."""
        game = FactoredDerivationGame(
            18, 7, leaf_eval, program_budget_mode="max",
            n_actions=6, one_hot_groups=[[0, 1, 2, 3]])
        game.reset()

        cloned = game.clone()
        assert cloned._n_actions == 6
        assert cloned._one_hot_groups == [[0, 1, 2, 3]]
        assert cloned.action_space.n == game.action_space.n
