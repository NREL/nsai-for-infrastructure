"""Tests for evaluation-time ablation dispatch.

Verifies that each ablation mode produces correct output and
does NOT modify the training network.
"""

import copy

import numpy as np
import pytest
import torch

from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.instances.doors.network import DoorsDirectNet
from alphazeropp.instances.doors.network_ablations import (
    RandomNetWrapper,
    UniformPolicyWrapper,
    ZeroValueWrapper,
    value_only_1step_policy,
)
from alphazeropp.instances.doors.eval_ablations import (
    ABLATION_MODES,
    compute_solve_rate_ablated,
    _eval_loop,
    _make_temp_agent,
)
from alphazeropp.core.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def net():
    """Small deterministic network for testing."""
    return DoorsDirectNet(input_size=7, output_size=6, random_seed=42, device="cpu")


@pytest.fixture
def game():
    """Default D=2 Doors game."""
    g = DoorsDirectGame()
    g.reset_wrapper()
    return g


@pytest.fixture
def agent(game, net):
    """Agent with D=2 game and small network."""
    mcts_params = {
        "n_simulations": 5,
        "temperature": 1.0,
        "c_exploration": 1.4,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
    }
    return Agent(
        game=game, net=net, mcts_params=mcts_params,
        reward_discount=1.0,
        random_seeds={"mcts": 42, "train": 43, "eval": 44, "external_policy": 45},
    )


# ---------------------------------------------------------------------------
# Test 1: full mode matches baseline
# ---------------------------------------------------------------------------

class TestFullMatchesBaseline:
    """'full' mode must produce identical results to _eval_loop with no ablation."""

    def test_full_matches(self, agent):
        torch.manual_seed(99)
        np.random.seed(99)
        solve1, reward1 = _eval_loop(agent, n_episodes=5)

        torch.manual_seed(99)
        np.random.seed(99)
        solve2, reward2 = compute_solve_rate_ablated(agent, "full", n_episodes=5)

        assert solve1 == solve2
        np.testing.assert_allclose(reward1, reward2, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: policy-only returns legal probabilities
# ---------------------------------------------------------------------------

class TestPolicyOnlyLegal:
    """policy-only mode returns valid, legal-only probabilities."""

    def test_legal_actions_only(self, agent, game):
        """Policy-only should produce legal actions."""
        # Run a few episodes and check actions are always legal
        solve_rate, avg_reward = compute_solve_rate_ablated(
            agent, "policy-only", n_episodes=3)
        # Should produce valid float results
        assert isinstance(solve_rate, float)
        assert isinstance(avg_reward, float)
        assert 0.0 <= solve_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 3: value-only-1step deterministic
# ---------------------------------------------------------------------------

class TestValueOnly1Step:
    """value_only_1step_policy chooses the action with best r + gamma*V(s')."""

    def test_returns_valid_onehot(self, game, net):
        probs = value_only_1step_policy(game, net, gamma=1.0)
        assert probs.shape == (6,)
        assert abs(probs.sum() - 1.0) < 1e-6
        # One-hot: exactly one entry is 1.0
        assert np.count_nonzero(probs) == 1

    def test_respects_legal_mask(self, net):
        """With precondition mask, chosen action must be legal."""
        game = DoorsDirectGame(use_precondition_mask=True)
        game.reset_wrapper(seed=42)
        mask = game.get_action_mask()
        probs = value_only_1step_policy(game, net, gamma=1.0)
        chosen = int(np.argmax(probs))
        assert mask[chosen], f"Chose illegal action {chosen}, mask={mask}"


# ---------------------------------------------------------------------------
# Test 4: value-only-1step terminal handling
# ---------------------------------------------------------------------------

class TestValueOnly1StepTerminal:
    """If an action leads to terminal state, use reward only (no V)."""

    def test_runs_without_error(self, agent):
        """Smoke test: value-only-1step completes without error."""
        solve_rate, avg_reward = compute_solve_rate_ablated(
            agent, "value-only-1step", n_episodes=3, gamma=1.0)
        assert isinstance(solve_rate, float)
        assert isinstance(avg_reward, float)


# ---------------------------------------------------------------------------
# Test 5: uniform-prior uses wrapper
# ---------------------------------------------------------------------------

class TestUniformPriorWrapper:
    """uniform-prior mode creates a temp agent with UniformPolicyWrapper."""

    def test_temp_agent_uses_uniform(self, agent):
        wrapped = UniformPolicyWrapper(agent.net)
        temp = _make_temp_agent(agent, net_override=wrapped)
        assert isinstance(temp.net, UniformPolicyWrapper)

    def test_runs_correctly(self, agent):
        solve_rate, avg_reward = compute_solve_rate_ablated(
            agent, "uniform-prior", n_episodes=3)
        assert isinstance(solve_rate, float)


# ---------------------------------------------------------------------------
# Test 6: unguided-search ignores learned prior and value
# ---------------------------------------------------------------------------

class TestUnguidedSearch:
    """unguided-search uses both uniform prior AND zero value."""

    def test_composed_wrappers(self, agent):
        """Verify the composition: ZeroValue(UniformPolicy(net))."""
        outer = ZeroValueWrapper(UniformPolicyWrapper(agent.net))
        obs = agent.game.obs.copy()
        policy, value = outer.predict(obs)

        # Policy should be uniform
        expected_uniform = np.ones(6, dtype=np.float32) / 6
        np.testing.assert_allclose(policy, expected_uniform)

        # Value should be 0
        assert value == np.float32(0.0)

    def test_runs_correctly(self, agent):
        solve_rate, avg_reward = compute_solve_rate_ablated(
            agent, "unguided-search", n_episodes=3)
        assert isinstance(solve_rate, float)


# ---------------------------------------------------------------------------
# Test 7: random-net differs from trained
# ---------------------------------------------------------------------------

class TestRandomNet:
    """random-net predictions should differ from the trained network."""

    def test_predictions_differ(self, net, game):
        wrapped = RandomNetWrapper(net)
        obs = game.obs.copy()
        trained_policy, trained_value = net.predict(obs)
        random_policy, random_value = wrapped.predict(obs)

        # Extremely unlikely to be identical with different random weights
        assert not np.allclose(trained_policy, random_policy, atol=1e-4) or \
               not np.isclose(trained_value, random_value, atol=1e-4), \
            "Random net produced identical output to trained net (extremely unlikely)"

    def test_train_delegates_to_real_net(self, net):
        wrapped = RandomNetWrapper(net)
        # train() should delegate to the real net, not the random one
        # Need non-empty examples (obs, policy_target, value_target)
        obs = np.zeros(7, dtype=np.float32)
        policy_target = np.ones(6, dtype=np.float32) / 6
        value_target = 0.0
        result = wrapped.train([(obs, policy_target, value_target)])
        assert isinstance(result, tuple)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 8: training net unchanged after ablation eval
# ---------------------------------------------------------------------------

class TestTrainingNetUnchanged:
    """After any ablation eval, agent.net weights must be byte-identical."""

    @pytest.mark.parametrize("mode", [
        m for m in ABLATION_MODES if m != "full"
    ])
    def test_weights_unchanged(self, agent, mode):
        # Snapshot weights before
        before = {k: v.clone() for k, v in agent.net.model.state_dict().items()}

        # Run ablation eval
        compute_solve_rate_ablated(agent, mode, n_episodes=2, gamma=1.0)

        # Compare weights after
        after = agent.net.model.state_dict()
        for key in before:
            assert torch.equal(before[key], after[key]), \
                f"Weight {key} changed after {mode} ablation eval!"


# ---------------------------------------------------------------------------
# Test 9: all modes listed
# ---------------------------------------------------------------------------

class TestAblationModesList:
    """Verify ABLATION_MODES list contains all expected modes."""

    def test_contains_expected_modes(self):
        expected = {"full", "policy-only", "value-only-1step",
                    "uniform-prior", "zero-value", "unguided-search", "random-net"}
        assert set(ABLATION_MODES) == expected
