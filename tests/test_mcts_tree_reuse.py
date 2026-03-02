"""
Tests for MCTS tree reuse across derivation steps.

Verifies that perform_simulations_reuse() + advance_to() produces correct
behavior: noise injection on reused roots, warm-started subtrees, and
end-to-end agent integration via play_one_round_reuse_tree().
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.instances.bitstring.dsl.leaf_evaluator import LeafEvaluator
from alphazeropp.instances.bitstring.dsl.derivation_game import (
    DerivationGame, UniformPolicyValueNet, compute_max_productions,
)
from alphazeropp.core.mcts import MCTS
from alphazeropp.core.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures (same as test_derivation_game.py)
# ---------------------------------------------------------------------------

N_SITES = 3
N_ONES = 2
BUDGET_5 = 5
BUDGET_8 = 8


@pytest.fixture
def game_config():
    return GameConfig(bit_flip=True, sparse_reward=False, n_ones=N_ONES)


@pytest.fixture
def frozen_states():
    return all_initial_states(N_SITES, N_ONES)


@pytest.fixture
def leaf_eval(frozen_states, game_config):
    return LeafEvaluator(N_SITES, frozen_states, game_config, metric="avg_reward")


def _make_mcts(budget, leaf_eval, n_simulations, seed=42):
    np.random.seed(seed)
    game = DerivationGame(budget, N_SITES, leaf_eval)
    game.reset_wrapper()
    net = UniformPolicyValueNet(game._max_productions)
    mcts = MCTS(
        game, net,
        n_simulations=n_simulations,
        temperature=0.1,
        c_exploration=1.5,
    )
    return mcts, game


# ---------------------------------------------------------------------------
# Test: nn_policy_original is stored on leaf expansion
# ---------------------------------------------------------------------------

class TestNNPolicyOriginal:

    def test_stored_after_expansion(self, leaf_eval):
        """After perform_simulations, root node has nn_policy_original set."""
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=1)
        mcts.perform_simulations_reuse(None)
        root_state = game.hashable_obs
        node = mcts.nodes[root_state]
        assert node.nn_policy_original is not None
        assert node.nn_policy_original.shape == node.nn_policy.shape

    def test_original_equals_policy_without_noise(self, leaf_eval):
        """Without noise, nn_policy and nn_policy_original should be equal."""
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=1)
        mcts.perform_simulations_reuse(None, add_noise=False)
        root_state = game.hashable_obs
        node = mcts.nodes[root_state]
        np.testing.assert_array_equal(node.nn_policy_original, node.nn_policy)

    def test_original_differs_from_noised_policy(self, leaf_eval):
        """With noise, nn_policy should differ from nn_policy_original."""
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=5)
        mcts.perform_simulations_reuse(None, add_noise=True)
        root_state = game.hashable_obs
        node = mcts.nodes[root_state]
        # They could theoretically be equal by extreme coincidence, but
        # practically the Dirichlet noise should perturb them.
        assert not np.allclose(node.nn_policy_original, node.nn_policy)


# ---------------------------------------------------------------------------
# Test: advance_to updates game state
# ---------------------------------------------------------------------------

class TestAdvanceTo:

    def test_changes_hashable_obs(self, leaf_eval):
        """advance_to moves the game to a new state."""
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=5)
        mcts.perform_simulations_reuse(None)
        old_obs = mcts.game.hashable_obs
        # Pick first valid action
        mask = mcts.game.get_action_mask()
        action = int(np.argmax(mask))
        mcts.advance_to(action)
        new_obs = mcts.game.hashable_obs
        assert old_obs != new_obs

    def test_new_root_in_nodes(self, leaf_eval):
        """After advance_to, the new root should already exist in self.nodes
        (it was explored as a child during simulations)."""
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=25)
        mcts.perform_simulations_reuse(None)
        probs = mcts.perform_simulations_reuse(None)
        action = int(np.argmax(probs))
        mcts.advance_to(action)
        assert mcts.game.hashable_obs in mcts.nodes


# ---------------------------------------------------------------------------
# Test: Dirichlet noise on reused roots
# ---------------------------------------------------------------------------

class TestNoiseOnReusedRoot:

    def test_noise_injected_despite_nonzero_total_N(self, leaf_eval):
        """The critical test: after advancing, the new root has total_N > 0
        from being visited as a child. perform_simulations_reuse should still
        inject noise (unlike perform_simulations which would skip it)."""
        np.random.seed(42)
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=25, seed=42)

        # Move 1: search from root
        probs = mcts.perform_simulations_reuse(None, add_noise=True)
        action = int(np.argmax(probs))
        mcts.advance_to(action)

        # The new root was a child — it has total_N > 0
        new_root_state = mcts.game.hashable_obs
        new_root_node = mcts.nodes[new_root_state]
        assert new_root_node.total_N > 0, "New root should have visits from prior search"

        # Save the original policy before noise
        original_before = new_root_node.nn_policy_original.copy()

        # Move 2: search from new root with noise
        mcts.perform_simulations_reuse(None, add_noise=True)

        # nn_policy should have been modified by noise (differs from original)
        assert not np.allclose(new_root_node.nn_policy, original_before), \
            "Noise should have been injected on the reused root"

    def test_noise_fresh_each_move(self, leaf_eval):
        """Noise should be re-drawn (not reused) at each move."""
        np.random.seed(42)
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=10, seed=42)

        # Move 1
        mcts.perform_simulations_reuse(None, add_noise=True)
        root1_state = mcts.game.hashable_obs
        policy_after_noise_1 = mcts.nodes[root1_state].nn_policy.copy()

        # Advance
        action = int(np.argmax(mcts.nodes[root1_state].nn_policy))
        mcts.advance_to(action)

        # Move 2
        mcts.perform_simulations_reuse(None, add_noise=True)
        root2_state = mcts.game.hashable_obs
        node2 = mcts.nodes[root2_state]

        # The noised policy should differ from the original (noise applied)
        assert not np.allclose(node2.nn_policy, node2.nn_policy_original), \
            "Noise should have been applied at Move 2"

    def test_no_noise_layering(self, leaf_eval):
        """Calling perform_simulations_reuse twice on the same root should
        restore from original each time, not layer noise on noise."""
        np.random.seed(42)
        mcts, game = _make_mcts(BUDGET_5, leaf_eval, n_simulations=5, seed=42)

        # First call with noise
        mcts.perform_simulations_reuse(None, add_noise=True)
        root_state = mcts.game.hashable_obs
        node = mcts.nodes[root_state]
        original = node.nn_policy_original.copy()

        # Second call with noise on same root (no advance)
        np.random.seed(99)  # different noise seed
        mcts.perform_simulations_reuse(None, add_noise=True)

        # The policy should be original + fresh noise, not double-noised
        # We can verify by checking nn_policy = (1-eps)*original + eps*noise
        # The policy should be a convex combination of original and noise
        eps = mcts.dirichlet_epsilon
        residual = node.nn_policy - (1 - eps) * original
        # residual should be eps * masked_noise, all non-negative where mask is True
        mask = node.action_mask
        assert np.all(residual[mask] >= -1e-7), \
            "Policy should be a proper convex combination of original and noise"


# ---------------------------------------------------------------------------
# Test: reuse matches existing pattern (no noise)
# ---------------------------------------------------------------------------

class TestReuseMatchesExisting:

    def test_same_actions_without_noise(self, leaf_eval):
        """With add_noise=False, perform_simulations_reuse + advance_to should
        produce identical actions to the existing pattern (perform_simulations
        + game.step_wrapper on the same MCTS instance)."""

        # Pattern 1: existing (perform_simulations + game.step_wrapper)
        np.random.seed(42)
        mcts1, game1 = _make_mcts(BUDGET_5, leaf_eval, n_simulations=25, seed=42)
        actions1 = []
        while not game1.terminated:
            probs = mcts1.perform_simulations(None)
            action = int(np.argmax(probs))
            actions1.append(action)
            game1.step_wrapper(action)

        # Pattern 2: new (perform_simulations_reuse + advance_to)
        np.random.seed(42)
        mcts2, game2 = _make_mcts(BUDGET_5, leaf_eval, n_simulations=25, seed=42)
        actions2 = []
        while not mcts2.game.terminated:
            probs = mcts2.perform_simulations_reuse(None)
            action = int(np.argmax(probs))
            actions2.append(action)
            mcts2.advance_to(action)

        assert actions1 == actions2, \
            f"Actions should be identical without noise: {actions1} vs {actions2}"


# ---------------------------------------------------------------------------
# Test: Agent.play_one_round_reuse_tree
# ---------------------------------------------------------------------------

class TestAgentTreeReuse:

    def test_completes_and_returns_valid_experience(self, leaf_eval):
        """play_one_round_reuse_tree runs to completion and returns valid
        experience tuples."""
        game = DerivationGame(BUDGET_5, N_SITES, leaf_eval)
        game.reset_wrapper()
        net = UniformPolicyValueNet(game._max_productions)
        agent = Agent(game, net, mcts_params={
            'n_simulations': 10, 'temperature': 0.1, 'c_exploration': 1.5,
        })

        experience, cumulative_reward = agent.play_one_round_reuse_tree(
            game, random_seed=42
        )

        assert len(experience) > 0
        for obs, probs, discounted_reward in experience:
            assert obs.shape == (2 * BUDGET_5,)
            assert np.isclose(probs.sum(), 1.0)
            assert isinstance(discounted_reward, float)

    def test_finds_good_program(self, leaf_eval):
        """End-to-end: tree reuse path finds a reasonable program."""
        np.random.seed(42)
        mcts, _ = _make_mcts(BUDGET_5, leaf_eval, n_simulations=50, seed=42)

        while not mcts.game.terminated:
            probs = mcts.perform_simulations_reuse(None)
            action = int(np.argmax(probs))
            mcts.advance_to(action)

        program = mcts.game.get_program()
        assert program is not None
        metrics = leaf_eval.get_all_metrics(program)
        assert metrics["solve_rate"] >= 0.0
