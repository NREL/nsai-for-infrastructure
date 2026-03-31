"""Tests for DoorsDirectGame, DoorsDirectNet, and DoorsDirectConfig."""

import numpy as np
import pytest
import torch

from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.instances.doors.network import DoorsDirectNet
from alphazeropp.instances.doors.config import DoorsDirectConfig
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


# ---------------------------------------------------------------------------
# DoorsDirectGame
# ---------------------------------------------------------------------------

class TestDoorsDirectGame:
    """Tests for DoorsDirectGame wrapper."""

    def test_reset_and_step(self):
        """reset_wrapper and step_wrapper produce correct shapes."""
        game = DoorsDirectGame()
        obs, _ = game.reset_wrapper()
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
        assert game.step_count == 0

        obs, reward, terminated, truncated, _ = game.step_wrapper(1)
        assert obs.shape == (7,)
        assert game.step_count == 1

    def test_action_mask_structural(self):
        """Default mask: all 6 actions valid (no padding)."""
        game = DoorsDirectGame()
        game.reset_wrapper()
        mask = game.get_action_mask()
        assert mask.shape == (6,)
        # All actions are real (MOVE_TO x4, PICK x1, NOOP x1)
        assert mask.all()

    def test_action_mask_precondition_initial(self):
        """Precondition mask at initial state."""
        game = DoorsDirectGame(use_precondition_mask=True)
        game.reset_wrapper()
        mask = game.get_action_mask()
        # Initial: at loc 0, room 0 unlocked, room 1 locked, key available
        # MOVE_TO(0): room 0 unlocked -> True
        assert mask[0] is np.True_
        # MOVE_TO(1): room 0 unlocked -> True
        assert mask[1] is np.True_
        # MOVE_TO(2): room 1 locked -> False
        assert mask[2] is np.False_
        # MOVE_TO(3): room 1 locked -> False
        assert mask[3] is np.False_
        # PICK(0): at loc 0, key at loc 1 -> not at key loc -> False
        assert mask[4] is np.False_
        # NOOP: always True
        assert mask[5] is np.True_

    def test_action_mask_precondition_at_key(self):
        """Precondition mask when at key location."""
        game = DoorsDirectGame(use_precondition_mask=True)
        game.reset_wrapper()
        game.step_wrapper(1)  # MOVE_TO(1) -> at key loc
        mask = game.get_action_mask()
        # PICK(0): at loc 1 AND key available -> True
        assert mask[4] is np.True_

    def test_stash_unstash_roundtrip(self):
        """Stash, step several times, unstash restores original state."""
        game = DoorsDirectGame()
        game.reset_wrapper()
        original_obs = game.obs.copy()
        original_step = game.step_count

        stashed = game.stash_state()

        # Step forward
        game.step_wrapper(1)
        game.step_wrapper(4)
        assert not np.array_equal(game.obs, original_obs)

        # Restore
        game = game.unstash_state(stashed)
        np.testing.assert_array_equal(game.obs, original_obs)
        assert game.step_count == original_step

    def test_clone_independence(self):
        """Clone produces independent copy."""
        game = DoorsDirectGame()
        game.reset_wrapper()
        original_obs = game.obs.copy()

        cloned = game.clone()
        np.testing.assert_array_equal(cloned.obs, original_obs)

        # Step the original, clone should be unchanged
        game.step_wrapper(1)
        np.testing.assert_array_equal(cloned.obs, original_obs)
        assert not np.array_equal(game.obs, original_obs)

    def test_play_optimal_sequence(self):
        """Optimal action sequence [1,4,3] yields reward ~1.07."""
        game = DoorsDirectGame()
        game.reset_wrapper()
        total = 0.0
        for action in [1, 4, 3]:
            _, reward, terminated, _, _ = game.step_wrapper(action)
            total += reward
        assert terminated
        assert total == pytest.approx(1.07, abs=0.001)


# ---------------------------------------------------------------------------
# DoorsDirectNet
# ---------------------------------------------------------------------------

class TestDoorsDirectNet:
    """Tests for DoorsDirectNet."""

    def test_predict_shape(self):
        """predict returns (policy (6,), value ()) for D=2 (6 real actions)."""
        net = DoorsDirectNet(input_size=7, output_size=6, device="cpu")
        obs = np.zeros(7, dtype=np.float32)
        policy, value = net.predict(obs)
        assert policy.shape == (6,)
        assert value.shape == ()
        # Policy should sum to ~1.0 (softmax)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_train_loss_decreases(self):
        """Training on a tiny dataset decreases loss."""
        net = DoorsDirectNet(input_size=7, output_size=6, random_seed=42, device="cpu")
        # Create a trivial dataset: all zeros -> uniform policy, value=0.5
        n = 50
        states = [np.zeros(7, dtype=np.float32) for _ in range(n)]
        policies = [np.ones(6, dtype=np.float32) / 6 for _ in range(n)]
        values = [0.5 for _ in range(n)]
        examples = list(zip(states, policies, values))

        _, _, train_losses, _, _ = net.train(examples)
        assert len(train_losses) > 1
        assert train_losses[-1] < train_losses[0]


# ---------------------------------------------------------------------------
# DoorsDirectConfig
# ---------------------------------------------------------------------------

class TestDoorsDirectConfig:
    """Tests for DoorsDirectConfig."""

    def test_build_returns_5tuple(self):
        """build() returns (game, net, agent, trainer, evaluator)."""
        config = DoorsDirectConfig()
        result = config.build()
        assert len(result) == 5
        game, net, agent, trainer, evaluator = result
        assert isinstance(game, DoorsDirectGame)
        assert isinstance(net, DoorsDirectNet)

    def test_default_hyperparameters(self):
        """Default hyperparameters are set correctly."""
        config = DoorsDirectConfig()
        assert config.agent.mcts_params["n_simulations"] == 120
        assert config.agent.mcts_params["c_exploration"] == 1.5
        assert config.agent.mcts_params["dirichlet_alpha"] == 0.25
        assert config.agent.mcts_params["dirichlet_epsilon"] == 0.40
        assert config.trainer.n_games_per_train == 50
        assert config.run.n_iterations == 50
        assert config.run.accept_threshold == 0.0
        assert config.evaluator.n_games == 20
        assert config.game.kwargs["num_rooms"] == 10
        assert config.game.kwargs["locs_per_room"] == 3

    def test_one_iteration(self):
        """Single training iteration completes without error."""
        config = DoorsDirectConfig()
        # Reduce to minimum for speed
        config.trainer.n_games_per_train = 2
        config.agent.mcts_params["n_simulations"] = 5

        game, net, agent, trainer, evaluator = config.build()
        trainer.train_iteration()

        # Verify some training happened
        records = trainer.statistics_manager.to_list()
        assert len(records) >= 1
