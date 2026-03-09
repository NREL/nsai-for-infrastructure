"""Tests verifying semantic assumptions for ablation experiments.

These tests validate the codebase behavior documented in docs/ablation_semantics.md
BEFORE any ablation wrappers are injected into training scripts.
"""

import copy
import pickle

import numpy as np
import pytest
import torch

from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.instances.doors.network import DoorsDirectNet
from alphazeropp.instances.doors.network_ablations import (
    FrozenWrapper,
    UniformPolicyWrapper,
    ZeroValueWrapper,
)
from alphazeropp.core.mcts import MCTS
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
def obs(game):
    """Observation from a reset game."""
    return game.obs.copy()


# ---------------------------------------------------------------------------
# Test 1: __getattr__ delegation
# ---------------------------------------------------------------------------

class TestWrapperDelegation:
    """Verify __getattr__ delegates attribute access to wrapped net."""

    def test_frozen_delegates_model(self, net):
        wrapped = FrozenWrapper(net)
        assert wrapped.model is net.model

    def test_frozen_delegates_device(self, net):
        wrapped = FrozenWrapper(net)
        assert wrapped.DEVICE == net.DEVICE

    def test_frozen_delegates_optimizer(self, net):
        wrapped = FrozenWrapper(net)
        assert wrapped.optimizer is net.optimizer

    def test_uniform_delegates_model(self, net):
        wrapped = UniformPolicyWrapper(net)
        assert wrapped.model is net.model

    def test_zero_value_delegates_model(self, net):
        wrapped = ZeroValueWrapper(net)
        assert wrapped.model is net.model


# ---------------------------------------------------------------------------
# Test 2: FrozenWrapper train returns compatible 5-tuple
# ---------------------------------------------------------------------------

class TestFrozenTrainSignature:
    """FrozenWrapper.train() must return 5-tuple matching trainer expectations."""

    def test_returns_5tuple(self, net):
        wrapped = FrozenWrapper(net)
        result = wrapped.train([])
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_first_element_is_model(self, net):
        wrapped = FrozenWrapper(net)
        model, *_ = wrapped.train([])
        # Must have state_dict() — trainer.py doesn't use it but gated_trainer does
        assert hasattr(model, "state_dict")
        assert model is net.model

    def test_remaining_elements_are_lists(self, net):
        wrapped = FrozenWrapper(net)
        _, batch_losses, train_losses, policy_losses, value_losses = wrapped.train([])
        assert isinstance(batch_losses, list)
        assert isinstance(train_losses, list)
        assert isinstance(policy_losses, list)
        assert isinstance(value_losses, list)


# ---------------------------------------------------------------------------
# Test 3: FrozenWrapper predict unchanged
# ---------------------------------------------------------------------------

class TestFrozenPredictUnchanged:
    """FrozenWrapper.predict() must return identical output to unwrapped net."""

    def test_predict_identical(self, net, obs):
        wrapped = FrozenWrapper(net)
        orig_policy, orig_value = net.predict(obs)
        wrap_policy, wrap_value = wrapped.predict(obs)
        np.testing.assert_array_equal(wrap_policy, orig_policy)
        np.testing.assert_equal(wrap_value, orig_value)


# ---------------------------------------------------------------------------
# Test 4: UniformPolicyWrapper output
# ---------------------------------------------------------------------------

class TestUniformPolicyWrapper:
    """UniformPolicyWrapper replaces policy with uniform, preserves value."""

    def test_policy_is_uniform(self, net, obs):
        wrapped = UniformPolicyWrapper(net)
        policy, _ = wrapped.predict(obs)
        expected = np.ones(6, dtype=np.float32) / 6
        np.testing.assert_allclose(policy, expected)

    def test_value_preserved(self, net, obs):
        wrapped = UniformPolicyWrapper(net)
        _, orig_value = net.predict(obs)
        _, wrap_value = wrapped.predict(obs)
        np.testing.assert_equal(wrap_value, orig_value)


# ---------------------------------------------------------------------------
# Test 5: ZeroValueWrapper output
# ---------------------------------------------------------------------------

class TestZeroValueWrapper:
    """ZeroValueWrapper replaces value with 0.0, preserves policy."""

    def test_value_is_zero(self, net, obs):
        wrapped = ZeroValueWrapper(net)
        _, value = wrapped.predict(obs)
        assert value == np.float32(0.0)

    def test_policy_preserved(self, net, obs):
        wrapped = ZeroValueWrapper(net)
        orig_policy, _ = net.predict(obs)
        wrap_policy, _ = wrapped.predict(obs)
        np.testing.assert_array_equal(wrap_policy, orig_policy)


# ---------------------------------------------------------------------------
# Test 6: Masking remains correct under wrappers
# ---------------------------------------------------------------------------

class TestMaskingUnderWrapper:
    """Action masking must work correctly when net is wrapped."""

    def test_mcts_respects_mask_with_frozen(self, game, net):
        wrapped = FrozenWrapper(net)
        mcts = MCTS(game, wrapped, n_simulations=1)
        probs = mcts.perform_simulations("", add_noise=False)
        mask = game.get_action_mask()
        # No probability mass on invalid actions
        invalid = ~mask
        assert np.all(probs[invalid] == 0.0), \
            f"Invalid actions have nonzero prob: {probs[invalid]}"

    def test_mcts_respects_mask_with_uniform(self, game, net):
        wrapped = UniformPolicyWrapper(net)
        mcts = MCTS(game, wrapped, n_simulations=1)
        probs = mcts.perform_simulations("", add_noise=False)
        mask = game.get_action_mask()
        invalid = ~mask
        assert np.all(probs[invalid] == 0.0)


# ---------------------------------------------------------------------------
# Test 7: n_simulations=-1 returns legal probs only
# ---------------------------------------------------------------------------

class TestNoMCTSReturnsLegalProbs:
    """n_simulations=-1 path must return probs only for legal actions."""

    def test_no_mcts_legal_only(self, game, net):
        mcts = MCTS(game, net, n_simulations=-1)
        probs = mcts.perform_simulations("", add_noise=False)
        mask = game.get_action_mask()
        invalid = ~mask
        assert np.all(probs[invalid] == 0.0), \
            f"Invalid actions have nonzero prob under n_sims=-1: {probs[invalid]}"
        # Should have some nonzero probs for legal actions
        assert probs[mask].sum() > 0

    def test_no_mcts_sums_to_one(self, game, net):
        mcts = MCTS(game, net, n_simulations=-1)
        probs = mcts.perform_simulations("", add_noise=False)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: n_simulations=0 returns all zeros
# ---------------------------------------------------------------------------

class TestZeroSimsReturnsAllZeros:
    """n_simulations=0 returns all-zero probs (documenting broken behavior)."""

    def test_zero_sims_all_zeros(self, game, net):
        mcts = MCTS(game, net, n_simulations=0)
        probs = mcts.perform_simulations("", add_noise=False)
        assert np.all(probs == 0.0), \
            f"Expected all zeros for n_sims=0, got: {probs}"


# ---------------------------------------------------------------------------
# Test 9: Evaluation with add_noise=False disables noise
# ---------------------------------------------------------------------------

class TestEvalNoiseDisabled:
    """Calling with add_noise=False must produce deterministic results."""

    def test_deterministic_without_noise(self, game, net):
        torch.manual_seed(99)
        np.random.seed(99)

        # Two calls with same state, no noise
        game1 = game.clone()
        mcts1 = MCTS(game1, net, n_simulations=5)
        probs1 = mcts1.perform_simulations("", add_noise=False)

        game2 = game.clone()
        mcts2 = MCTS(game2, net, n_simulations=5)
        probs2 = mcts2.perform_simulations("", add_noise=False)

        np.testing.assert_array_equal(probs1, probs2)


# ---------------------------------------------------------------------------
# Test 10: Wrapper pickle round-trip
# ---------------------------------------------------------------------------

class TestWrapperPickle:
    """Wrappers must survive pickle (needed for spawn multiprocessing)."""

    def test_frozen_pickle_roundtrip(self, net, obs):
        wrapped = FrozenWrapper(net)
        # push to CPU for pickling (mirrors multiprocessing flow)
        wrapped.push_multiprocessing()
        data = pickle.dumps(wrapped)
        restored = pickle.loads(data)
        # predict must still work
        policy, value = restored.predict(obs)
        assert policy.shape == (6,)
        assert value.shape == ()

    def test_uniform_pickle_roundtrip(self, net, obs):
        wrapped = UniformPolicyWrapper(net)
        wrapped.push_multiprocessing()
        data = pickle.dumps(wrapped)
        restored = pickle.loads(data)
        policy, value = restored.predict(obs)
        expected = np.ones(6, dtype=np.float32) / 6
        np.testing.assert_allclose(policy, expected)

    def test_zero_value_pickle_roundtrip(self, net, obs):
        wrapped = ZeroValueWrapper(net)
        wrapped.push_multiprocessing()
        data = pickle.dumps(wrapped)
        restored = pickle.loads(data)
        _, value = restored.predict(obs)
        assert value == np.float32(0.0)


# ---------------------------------------------------------------------------
# Test 11: Wrapper push/pop multiprocessing
# ---------------------------------------------------------------------------

class TestWrapperPushPopMultiprocessing:
    """push/pop_multiprocessing must work through __getattr__ delegation."""

    def test_frozen_push_pop(self, net, obs):
        wrapped = FrozenWrapper(net)
        stash = wrapped.push_multiprocessing()
        # Predict should still work after push (model on CPU)
        policy1, value1 = wrapped.predict(obs)
        assert policy1.shape == (6,)

        wrapped.pop_multiprocessing(stash)
        # Predict should still work after pop (model restored)
        policy2, value2 = wrapped.predict(obs)
        assert policy2.shape == (6,)

    def test_uniform_push_pop(self, net, obs):
        wrapped = UniformPolicyWrapper(net)
        stash = wrapped.push_multiprocessing()
        wrapped.pop_multiprocessing(stash)
        policy, _ = wrapped.predict(obs)
        expected = np.ones(6, dtype=np.float32) / 6
        np.testing.assert_allclose(policy, expected)
