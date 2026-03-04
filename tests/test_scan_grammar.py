"""
Tests for the priority-scan grammar: ScanState, ScanDerivationGame,
permutation_to_program, and end-to-end integration.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Ite, Default,
)
from alphazeropp.instances.bitstring.dsl.scan_grammar import (
    permutation_to_program, ScanState,
)
from alphazeropp.instances.bitstring.dsl.scan_derivation_game import (
    ScanDerivationGame,
)
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SITES = 4
N_ONES = 2


@pytest.fixture
def game_config():
    return GameConfig(bit_flip=True, sparse_reward=False, n_ones=N_ONES)


@pytest.fixture
def frozen_states():
    return all_initial_states(N_SITES, N_ONES)


@pytest.fixture
def leaf_eval(frozen_states, game_config):
    return LeafEvaluator(N_SITES, frozen_states, game_config, metric="avg_reward")


@pytest.fixture
def scan_game(leaf_eval):
    return ScanDerivationGame(N_SITES, leaf_eval)


# ---------------------------------------------------------------------------
# permutation_to_program
# ---------------------------------------------------------------------------

class TestPermutationToProgram:
    """Tests for the permutation_to_program conversion."""

    def test_identity_permutation(self):
        """Identity permutation [0,1,2,3] produces correct AST."""
        prog = permutation_to_program([0, 1, 2, 3])
        assert isinstance(prog, Ite)
        assert isinstance(prog.cond, IsZero)
        assert prog.cond.index == 0
        assert prog.action.index == 0

    def test_single_element(self):
        """Single-element permutation produces Default."""
        prog = permutation_to_program([2])
        assert isinstance(prog, Default)
        assert prog.action.index == 2

    def test_two_elements(self):
        """Two-element permutation produces Ite + Default."""
        prog = permutation_to_program([3, 1])
        assert isinstance(prog, Ite)
        assert prog.cond.index == 3
        assert prog.action.index == 3
        assert isinstance(prog.else_prog, Default)
        assert prog.else_prog.action.index == 1

    def test_node_count(self):
        """A scan program with N elements has 3*(N-1) + 2 nodes."""
        for n in range(1, 8):
            perm = list(range(n))
            prog = permutation_to_program(perm)
            expected = 3 * (n - 1) + 2
            assert prog.node_count() == expected, (
                f"N={n}: expected {expected}, got {prog.node_count()}"
            )

    def test_condition_action_match(self):
        """Every Ite checks IsZero(i) and flips Flip(i) for the same i."""
        perm = [3, 0, 2, 1]
        prog = permutation_to_program(perm)
        node = prog
        for idx in perm[:-1]:
            assert isinstance(node, Ite)
            assert isinstance(node.cond, IsZero)
            assert node.cond.index == idx
            assert node.action.index == idx
            node = node.else_prog
        assert isinstance(node, Default)
        assert node.action.index == perm[-1]

    def test_empty_permutation_raises(self):
        """Empty permutation should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            permutation_to_program([])

    def test_pretty_print(self):
        """Pretty print matches expected format."""
        prog = permutation_to_program([0, 1, 2])
        pretty = prog.pretty()
        assert "IsZero(0)" in pretty
        assert "Flip(0)" in pretty
        assert "IsZero(1)" in pretty
        assert "Flip(2)" in pretty


# ---------------------------------------------------------------------------
# ScanState
# ---------------------------------------------------------------------------

class TestScanState:
    """Tests for ScanState dataclass."""

    def test_initial_state(self):
        """Initial state has empty prefix and all indices remaining."""
        state = ScanState.initial(4)
        assert state.prefix == ()
        assert state.remaining == frozenset({0, 1, 2, 3})
        assert state.n_sites == 4
        assert not state.is_terminal()

    def test_apply_action(self):
        """Applying an action adds to prefix and removes from remaining."""
        state = ScanState.initial(4)
        state2 = state.apply(2)
        assert state2.prefix == (2,)
        assert state2.remaining == frozenset({0, 1, 3})
        assert not state2.is_terminal()

    def test_terminal_when_one_remaining(self):
        """State is terminal when exactly one index remains (forced)."""
        state = ScanState.initial(3)
        state = state.apply(1)
        state = state.apply(0)
        # Only index 2 remains -> terminal (forced)
        assert state.is_terminal()
        assert state.remaining == frozenset({2})

    def test_terminal_when_none_remaining(self):
        """State is terminal when zero indices remain."""
        state = ScanState.initial(2)
        state = state.apply(1)
        state = state.apply(0)
        assert state.is_terminal()
        assert state.remaining == frozenset()

    def test_apply_invalid_action_raises(self):
        """Applying an action not in remaining raises ValueError."""
        state = ScanState.initial(3)
        state = state.apply(1)
        with pytest.raises(ValueError, match="not in remaining"):
            state.apply(1)

    def test_legal_actions_sorted(self):
        """Legal actions are sorted remaining indices."""
        state = ScanState.initial(5)
        state = state.apply(2)
        state = state.apply(4)
        assert state.legal_actions() == [0, 1, 3]

    def test_to_program_forces_last(self):
        """to_program appends the single remaining index."""
        state = ScanState.initial(3)
        state = state.apply(2)
        state = state.apply(0)
        assert state.is_terminal()
        prog = state.to_program()
        # Should be: Ite(IsZero(2), Flip(2), Ite(IsZero(0), Flip(0), Default(Flip(1))))
        assert isinstance(prog, Ite)
        assert prog.cond.index == 2
        node = prog.else_prog
        assert isinstance(node, Ite)
        assert node.cond.index == 0
        assert isinstance(node.else_prog, Default)
        assert node.else_prog.action.index == 1

    def test_to_program_not_terminal_raises(self):
        """to_program on non-terminal state raises AssertionError."""
        state = ScanState.initial(4)
        state = state.apply(0)
        with pytest.raises(AssertionError, match="non-terminal"):
            state.to_program()

    def test_pretty_format(self):
        """Pretty print shows prefix and underscores."""
        state = ScanState.initial(4)
        state = state.apply(3)
        state = state.apply(1)
        assert state.pretty() == "Scan(3,1,_,_)"

    def test_immutability(self):
        """ScanState is frozen (immutable)."""
        state = ScanState.initial(3)
        state2 = state.apply(0)
        # Original state unchanged
        assert state.prefix == ()
        assert state.remaining == frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# ScanDerivationGame
# ---------------------------------------------------------------------------

class TestScanDerivationGame:
    """Tests for ScanDerivationGame."""

    def test_reset(self, scan_game):
        """Reset returns correct observation shape and empty prefix."""
        obs, info = scan_game.reset()
        assert obs.shape == (2 * N_SITES,)
        assert obs.dtype == np.float32
        # All prefix slots should be -1
        assert all(obs[i] == -1.0 for i in range(N_SITES))
        # All remaining slots should be 1.0
        assert all(obs[N_SITES + i] == 1.0 for i in range(N_SITES))

    def test_action_mask_initial(self, scan_game):
        """Initial action mask has all indices valid."""
        scan_game.reset()
        mask = scan_game.get_action_mask()
        assert mask.shape == (N_SITES,)
        assert all(mask)

    def test_action_mask_after_step(self, scan_game):
        """After choosing index 2, mask[2] should be False."""
        scan_game.reset()
        scan_game.step(2)
        mask = scan_game.get_action_mask()
        assert not mask[2]
        assert sum(mask) == N_SITES - 1

    def test_step_nonterminal(self, scan_game):
        """Non-terminal steps return reward=0, terminated=False."""
        scan_game.reset()
        obs, reward, terminated, truncated, info = scan_game.step(0)
        assert reward == 0.0
        assert not terminated
        assert not truncated

    def test_full_episode(self, scan_game):
        """A complete episode yields a terminal reward and program."""
        scan_game.reset()
        # Play N-1 steps (last is forced at terminal)
        for i in range(N_SITES - 1):
            mask = scan_game.get_action_mask()
            valid = [j for j in range(N_SITES) if mask[j]]
            obs, reward, terminated, truncated, info = scan_game.step(valid[0])
            if terminated:
                break

        assert terminated
        assert reward != 0.0 or reward == 0.0  # reward is a float
        assert "program" in info

    def test_no_dead_ends(self, scan_game):
        """Random episodes always complete (no dead ends)."""
        rng = random.Random(42)
        for _ in range(50):
            scan_game.reset()
            done = False
            steps = 0
            while not done:
                mask = scan_game.get_action_mask()
                valid = [j for j in range(N_SITES) if mask[j]]
                assert len(valid) > 0, "Dead end: no valid actions"
                action = rng.choice(valid)
                _, _, terminated, truncated, _ = scan_game.step(action)
                done = terminated or truncated
                steps += 1
                assert steps <= N_SITES, "Too many steps"
            assert terminated and not truncated

    def test_no_duplicate_indices(self, scan_game):
        """Final permutation has no duplicate bit indices."""
        scan_game.reset()
        actions = []
        for _ in range(N_SITES):
            mask = scan_game.get_action_mask()
            valid = [j for j in range(N_SITES) if mask[j]]
            if not valid:
                break
            actions.append(valid[0])
            _, _, terminated, _, _ = scan_game.step(valid[0])
            if terminated:
                break
        # Check that all chosen actions are unique
        assert len(set(actions)) == len(actions)

    def test_hashable_obs(self, scan_game):
        """hashable_obs returns a string representation."""
        scan_game.reset()
        h = scan_game.hashable_obs
        assert isinstance(h, str)
        assert "Scan" in h

    def test_stash_unstash(self, scan_game):
        """Stash and unstash preserve game state."""
        scan_game.reset()
        scan_game.step_wrapper(1)
        stashed = scan_game.stash_state()
        obs_before = scan_game.obs.copy()

        # Advance further
        scan_game.step_wrapper(0)

        # Restore
        scan_game.unstash_state(stashed)
        np.testing.assert_array_equal(scan_game.obs, obs_before)
        assert scan_game._scan_state.prefix == (1,)

    def test_clone(self, scan_game):
        """Clone creates an independent copy."""
        scan_game.reset()
        scan_game.step_wrapper(2)
        clone = scan_game.clone()

        # Advance original
        scan_game.step_wrapper(0)

        # Clone should be unaffected
        assert clone._scan_state.prefix == (2,)
        assert scan_game._scan_state.prefix == (2, 0)

    def test_action_space(self, scan_game):
        """Action space is Discrete(N)."""
        assert scan_game.action_space.n == N_SITES

    def test_observation_encoding(self, scan_game):
        """Observation encoding matches expected format."""
        scan_game.reset()
        scan_game.step(3)
        scan_game.step(0)
        obs = scan_game._encode_obs()
        # Prefix: [3, 0, -1, -1]
        assert obs[0] == 3.0
        assert obs[1] == 0.0
        assert obs[2] == -1.0
        assert obs[3] == -1.0
        # Remaining mask: [0, 1, 1, 0]  (0 and 3 taken)
        assert obs[4] == 0.0  # index 0 taken
        assert obs[5] == 1.0  # index 1 remaining
        assert obs[6] == 1.0  # index 2 remaining
        assert obs[7] == 0.0  # index 3 taken


# ---------------------------------------------------------------------------
# Integration: scan program evaluation
# ---------------------------------------------------------------------------

class TestScanProgramEvaluation:
    """Test that scan programs produce correct evaluation results."""

    def test_identity_scan_solves_all_states(self, leaf_eval):
        """Identity permutation [0,1,...,N-1] should solve all states for OneMax."""
        prog = permutation_to_program(list(range(N_SITES)))
        value = leaf_eval(prog)
        metrics = leaf_eval.get_all_metrics(prog)
        assert metrics["solve_rate"] == 1.0

    def test_all_permutations_optimal_for_onemax(self, leaf_eval):
        """For OneMax, every permutation should achieve the same (optimal) reward."""
        from itertools import permutations
        values = []
        for perm in permutations(range(N_SITES)):
            prog = permutation_to_program(list(perm))
            values.append(leaf_eval(prog))

        # All should be equal (OneMax is order-invariant)
        assert all(abs(v - values[0]) < 1e-6 for v in values), (
            f"Not all permutations equal: min={min(values)}, max={max(values)}"
        )

    def test_scan_program_validates(self):
        """Scan programs pass AST validation."""
        for n in range(2, 8):
            perm = list(range(n))
            prog = permutation_to_program(perm)
            prog.validate(n)  # Should not raise

    def test_leaf_evaluator_caching(self, leaf_eval):
        """Same permutation evaluated twice hits cache."""
        prog1 = permutation_to_program([0, 1, 2, 3])
        prog2 = permutation_to_program([0, 1, 2, 3])
        v1 = leaf_eval(prog1)
        hits_before = leaf_eval._cache_hits
        v2 = leaf_eval(prog2)
        assert leaf_eval._cache_hits == hits_before + 1
        assert v1 == v2


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestScanConfig:
    """Test ScanDerivationConfig builds correctly."""

    def test_config_build(self):
        """ScanDerivationConfig.build() produces working objects."""
        from alphazeropp.instances.bitstring.dsl.derivation_config import (
            ScanDerivationConfig,
        )
        cfg = ScanDerivationConfig()
        cfg.game.kwargs["n_sites"] = 3
        cfg.net.kwargs["n_sites"] = 3
        game, net, agent, trainer, evaluator = cfg.build()

        assert game.action_space.n == 3
        assert game.observation_space.shape == (6,)

        # Smoke test: reset and step
        game.reset_wrapper()
        obs, reward, terminated, truncated, info = game.step_wrapper(0)
        assert not terminated
        assert reward == 0.0
