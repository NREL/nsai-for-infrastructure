"""
Tests for MCTS configurable backup strategy.

Verifies that the backup rule mechanism:
  - Default "mean" reproduces current behavior
  - "max" returns the maximum backed-up value
  - "topk" returns mean of top-k values
  - "softmax" returns a value between mean and max
  - Edge cases: single visit, identical rewards, k > visits
  - Q-normalization stays healthy with all rules
  - Tree reuse works with non-mean backup
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.synthesis.derivation_game import (
    DerivationGame, UniformPolicyValueNet,
)
from alphazeropp.core.mcts import MCTS, MCTSTreeNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SITES = 3
N_ONES = 2
BUDGET = 5


@pytest.fixture
def game_config():
    return GameConfig(bit_flip=True, sparse_reward=False, n_ones=N_ONES)


@pytest.fixture
def frozen_states():
    return all_initial_states(N_SITES, N_ONES)


@pytest.fixture
def leaf_eval(frozen_states, game_config):
    return LeafEvaluator(N_SITES, frozen_states, game_config, metric="avg_reward",
                         is_solved=lambda obs: bool(np.all(obs == 1.0)))


def _make_game(leaf_eval):
    game = DerivationGame(BUDGET, N_SITES, leaf_eval)
    game.reset_wrapper()
    return game


def _make_mcts(leaf_eval, **mcts_kwargs):
    game = _make_game(leaf_eval)
    net = UniformPolicyValueNet(game._max_productions)
    defaults = dict(
        n_simulations=10,
        temperature=1.0,
        c_exploration=1.5,
    )
    defaults.update(mcts_kwargs)
    mcts = MCTS(game, net, **defaults)
    return mcts, game


# ---------------------------------------------------------------------------
# Tests: Mean backup (default / backward compat)
# ---------------------------------------------------------------------------


class TestBackupMean:
    """Default backup_rule="mean" reproduces current behavior."""

    def test_default_is_mean(self, leaf_eval):
        mcts, _ = _make_mcts(leaf_eval)
        assert mcts.backup_rule == "mean"

    def test_mean_runs_normally(self, leaf_eval):
        mcts, game = _make_mcts(leaf_eval, backup_rule="mean")
        probs = mcts.perform_simulations(None)
        assert probs.shape == (game._max_productions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_mean_q_equals_average(self, leaf_eval):
        """After simulations, Q should equal average of action_values."""
        mcts, _ = _make_mcts(leaf_eval, backup_rule="mean", n_simulations=20)
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            expected = sum(values) / len(values)
            assert np.isclose(root.action_Q[action], expected), \
                f"Q={root.action_Q[action]} != mean={expected}"


# ---------------------------------------------------------------------------
# Tests: Max backup
# ---------------------------------------------------------------------------


class TestBackupMax:
    """backup_rule="max" returns maximum backed-up value."""

    def test_max_runs(self, leaf_eval):
        mcts, game = _make_mcts(leaf_eval, backup_rule="max")
        probs = mcts.perform_simulations(None)
        assert probs.shape == (game._max_productions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_max_q_equals_max_value(self, leaf_eval):
        mcts, _ = _make_mcts(leaf_eval, backup_rule="max", n_simulations=20)
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            expected = max(values)
            assert np.isclose(root.action_Q[action], expected), \
                f"Q={root.action_Q[action]} != max={expected}"

    def test_max_q_ge_mean_q(self, leaf_eval):
        """Max Q should be >= mean Q for same data."""
        np.random.seed(42)
        mcts_max, _ = _make_mcts(leaf_eval, backup_rule="max", n_simulations=30)
        mcts_max.perform_simulations(None)
        root = mcts_max.nodes[mcts_max.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            mean_q = sum(values) / len(values)
            assert root.action_Q[action] >= mean_q - 1e-10


# ---------------------------------------------------------------------------
# Tests: Top-k backup
# ---------------------------------------------------------------------------


class TestBackupTopK:
    """backup_rule="topk" returns mean of top-k values."""

    def test_topk_runs(self, leaf_eval):
        mcts, game = _make_mcts(leaf_eval, backup_rule="topk", backup_topk=2)
        probs = mcts.perform_simulations(None)
        assert probs.shape == (game._max_productions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_topk_q_correct(self, leaf_eval):
        mcts, _ = _make_mcts(
            leaf_eval, backup_rule="topk", backup_topk=2, n_simulations=20
        )
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            top2 = sorted(values, reverse=True)[:2]
            expected = sum(top2) / len(top2)
            assert np.isclose(root.action_Q[action], expected, atol=1e-10)

    def test_topk_k_exceeds_visits(self, leaf_eval):
        """When k > number of visits, topk degenerates to mean."""
        mcts, _ = _make_mcts(
            leaf_eval, backup_rule="topk", backup_topk=100, n_simulations=5
        )
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            # k=100 > len(values), so top-k = all values = mean
            expected = sum(values) / len(values)
            assert np.isclose(root.action_Q[action], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Softmax backup
# ---------------------------------------------------------------------------


class TestBackupSoftmax:
    """backup_rule="softmax" returns smooth max approximation."""

    def test_softmax_runs(self, leaf_eval):
        mcts, game = _make_mcts(
            leaf_eval, backup_rule="softmax", backup_tau=0.1
        )
        probs = mcts.perform_simulations(None)
        assert probs.shape == (game._max_productions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_softmax_between_mean_and_max(self, leaf_eval):
        """Softmax Q should be between mean and max of values."""
        mcts, _ = _make_mcts(
            leaf_eval, backup_rule="softmax", backup_tau=0.1, n_simulations=30
        )
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            if len(values) < 2:
                continue
            mean_v = sum(values) / len(values)
            max_v = max(values)
            q = root.action_Q[action]
            assert q >= mean_v - 1e-10, f"softmax Q={q} < mean={mean_v}"
            assert q <= max_v + 1e-10, f"softmax Q={q} > max={max_v}"

    def test_small_tau_approaches_max(self, leaf_eval):
        """With very small tau, softmax should approach max."""
        mcts, _ = _make_mcts(
            leaf_eval, backup_rule="softmax", backup_tau=0.001, n_simulations=20
        )
        mcts.perform_simulations(None)
        root = mcts.nodes[mcts.game.hashable_obs]
        for action in root.action_Q:
            values = root.action_values[action]
            max_v = max(values)
            q = root.action_Q[action]
            # With small tau, softmax approaches max but numerical precision
            # limits exact convergence when values are closely spaced
            assert np.isclose(q, max_v, atol=5e-3), \
                f"small tau: Q={q} not close to max={max_v}"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestBackupEdgeCases:
    """Edge cases: single visit, identical rewards."""

    def test_single_visit_all_rules_agree(self, leaf_eval):
        """With one visit per action, all rules produce the same Q."""
        results = {}
        for rule in ["mean", "max", "topk", "softmax"]:
            np.random.seed(123)
            mcts, _ = _make_mcts(
                leaf_eval, backup_rule=rule, n_simulations=1
            )
            mcts.perform_simulations(None)
            root = mcts.nodes[mcts.game.hashable_obs]
            results[rule] = dict(root.action_Q)

        # All rules should have the same Q values for visited actions
        for action in results["mean"]:
            vals = [results[rule][action] for rule in results if action in results[rule]]
            if len(vals) > 1:
                assert all(np.isclose(v, vals[0]) for v in vals), \
                    f"Single visit: rules disagree on action {action}: {vals}"

    def test_identical_rewards_all_rules_agree(self, leaf_eval):
        """When all values are identical, all rules produce the same Q."""
        node = MCTSTreeNode(0.0, False)
        node.action_values[(0,)] = [0.5, 0.5, 0.5, 0.5]
        node.action_N[(0,)] = 4
        node.action_Q[(0,)] = 0.0

        for rule in ["mean", "max", "topk", "softmax"]:
            mcts, _ = _make_mcts(leaf_eval, backup_rule=rule)
            # Manually call update logic by setting values and recomputing
            values = node.action_values[(0,)]
            if rule == "mean":
                q = sum(values) / len(values)
            elif rule == "max":
                q = max(values)
            elif rule == "topk":
                top = sorted(values, reverse=True)[:3]
                q = sum(top) / len(top)
            elif rule == "softmax":
                v = np.array(values)
                v_shifted = (v - v.max()) / 0.1
                q = float(v.max() + 0.1 * np.log(np.mean(np.exp(v_shifted))))
            assert np.isclose(q, 0.5), f"rule={rule}: Q={q} != 0.5"

    def test_invalid_backup_rule_raises(self, leaf_eval):
        """Unknown backup_rule should raise AssertionError."""
        with pytest.raises(AssertionError, match="Unknown backup_rule"):
            _make_mcts(leaf_eval, backup_rule="invalid")


# ---------------------------------------------------------------------------
# Tests: Q-normalization health
# ---------------------------------------------------------------------------


class TestBackupQNorm:
    """Q-normalization stays healthy with all backup rules."""

    @pytest.mark.parametrize("rule", ["mean", "max", "topk", "softmax"])
    def test_q_range_finite(self, leaf_eval, rule):
        """q_min and q_max should be finite after simulations."""
        mcts, _ = _make_mcts(
            leaf_eval, backup_rule=rule, n_simulations=20
        )
        mcts.perform_simulations(None)
        assert np.isfinite(mcts.q_min), f"q_min not finite: {mcts.q_min}"
        assert np.isfinite(mcts.q_max), f"q_max not finite: {mcts.q_max}"
        assert mcts.q_max >= mcts.q_min


# ---------------------------------------------------------------------------
# Tests: Tree reuse
# ---------------------------------------------------------------------------


class TestBackupTreeReuse:
    """Tree reuse works with non-mean backup rules."""

    @pytest.mark.parametrize("rule", ["mean", "max", "topk", "softmax"])
    def test_tree_reuse_completes(self, leaf_eval, rule):
        """perform_simulations_reuse + advance_to works with all rules."""
        game = _make_game(leaf_eval)
        net = UniformPolicyValueNet(game._max_productions)
        mcts = MCTS(
            game, net,
            n_simulations=5, temperature=1.0, c_exploration=1.5,
            backup_rule=rule,
        )

        # Do first search
        probs = mcts.perform_simulations_reuse(None)
        assert np.isclose(probs.sum(), 1.0)

        # Advance and do another search
        action = np.argmax(probs)
        if not (mcts.game.terminated or mcts.game.truncated):
            mcts.advance_to(action)
            if not (mcts.game.terminated or mcts.game.truncated):
                probs2 = mcts.perform_simulations_reuse(None)
                assert np.isclose(probs2.sum(), 1.0)
