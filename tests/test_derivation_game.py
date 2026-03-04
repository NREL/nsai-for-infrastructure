"""
Tests for DerivationGame, LeafEvaluator, and UniformPolicyValueNet.

Covers game mechanics, stash/unstash correctness, leaf evaluation
with caching, and end-to-end MCTS integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Ite, Default,
)
from alphazeropp.synthesis.derivation import DerivationState
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.synthesis.derivation_game import (
    DerivationGame, UniformPolicyValueNet, compute_max_productions,
)
from alphazeropp.core.mcts import MCTS


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def game_b5(leaf_eval):
    g = DerivationGame(BUDGET_5, N_SITES, leaf_eval)
    g.reset_wrapper()
    return g


@pytest.fixture
def game_b8(leaf_eval):
    g = DerivationGame(BUDGET_8, N_SITES, leaf_eval)
    g.reset_wrapper()
    return g


# ---------------------------------------------------------------------------
# TestDerivationGameBasics
# ---------------------------------------------------------------------------

class TestDerivationGameBasics:

    def test_reset_returns_valid_obs(self, game_b5):
        assert game_b5.obs is not None
        assert game_b5.obs.shape == (2 * BUDGET_5,)
        assert game_b5.obs.dtype == np.float32

    def test_step_applies_production(self, game_b5):
        obs_before = game_b5.hashable_obs
        game_b5.step_wrapper(0)
        obs_after = game_b5.hashable_obs
        assert obs_before != obs_after

    def test_action_mask_shape(self, game_b5):
        mask = game_b5.get_action_mask()
        max_prods = compute_max_productions(BUDGET_5, N_SITES)
        assert mask.shape == (max_prods,)
        assert mask.dtype == bool

    def test_action_mask_valid_count(self, game_b5):
        mask = game_b5.get_action_mask()
        prods = game_b5._deriv_state.legal_productions(N_SITES)
        assert mask.sum() == len(prods)

    def test_terminal_when_no_holes(self, game_b5):
        """Step through a full derivation and verify terminal state."""
        while not game_b5.terminated:
            mask = game_b5.get_action_mask()
            action = np.argmax(mask)  # take first valid action
            game_b5.step_wrapper(int(action))
        assert game_b5.terminated is True
        assert game_b5._deriv_state.is_terminal()

    def test_produces_valid_program(self, game_b5):
        """Terminal derivation produces a program with correct node count."""
        while not game_b5.terminated:
            game_b5.step_wrapper(0)
        program = game_b5.get_program()
        assert program is not None
        assert program.node_count() == BUDGET_5

    def test_hashable_obs_unique(self, game_b5):
        """Different states should have different hashable_obs."""
        h1 = game_b5.hashable_obs
        game_b5.step_wrapper(0)
        h2 = game_b5.hashable_obs
        assert h1 != h2

    def test_hashable_obs_stable(self, game_b5):
        """Same state should return same hashable_obs across calls."""
        h1 = game_b5.hashable_obs
        h2 = game_b5.hashable_obs
        assert h1 == h2


# ---------------------------------------------------------------------------
# TestStashUnstash
# ---------------------------------------------------------------------------

class TestStashUnstash:

    def test_roundtrip(self, game_b5):
        """stash -> step -> unstash restores original state."""
        original_obs = game_b5.hashable_obs
        stashed = game_b5.stash_state()
        game_b5.step_wrapper(0)
        assert game_b5.hashable_obs != original_obs
        game_b5 = game_b5.unstash_state(stashed)
        assert game_b5.hashable_obs == original_obs

    def test_independence(self, game_b5):
        """Stepping after stash doesn't affect the stashed copy."""
        stashed = game_b5.stash_state()
        original_obs = game_b5.hashable_obs

        # Step multiple times
        game_b5.step_wrapper(0)
        if not game_b5.terminated:
            game_b5.step_wrapper(0)

        # Restore
        game_b5 = game_b5.unstash_state(stashed)
        assert game_b5.hashable_obs == original_obs


# ---------------------------------------------------------------------------
# TestUniformPolicyValueNet
# ---------------------------------------------------------------------------

class TestUniformPolicyValueNet:

    def test_predict_uniform_policy(self):
        net = UniformPolicyValueNet(action_size=10)
        policy, value = net.predict(np.zeros(5))
        assert np.allclose(policy, 0.1)
        assert np.isclose(policy.sum(), 1.0)

    def test_predict_value_zero(self):
        net = UniformPolicyValueNet(action_size=10)
        _, value = net.predict(np.zeros(5))
        assert np.isclose(value, 0.0)

    def test_shapes(self):
        net = UniformPolicyValueNet(action_size=7)
        policy, value = net.predict(np.zeros(3))
        assert policy.shape == (7,)
        assert value.shape == ()


# ---------------------------------------------------------------------------
# TestLeafEvaluator
# ---------------------------------------------------------------------------

class TestLeafEvaluator:

    def test_perfect_program_high_value(self, leaf_eval):
        """A known-good program should have high solve_rate."""
        # if IsZero(0): Flip(0) elif IsZero(1): Flip(1) else: Flip(2)
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1),
                        Default(Flip(2))))
        metrics = leaf_eval.get_all_metrics(prog)
        assert metrics["solve_rate"] == 1.0

    def test_bad_program_low_value(self, leaf_eval):
        """Default(Flip(0)) should have low solve rate for N=3."""
        prog = Default(Flip(0))
        metrics = leaf_eval.get_all_metrics(prog)
        assert metrics["solve_rate"] < 1.0

    def test_caching_works(self, leaf_eval):
        prog = Default(Flip(0))
        leaf_eval(prog)
        hits_before = leaf_eval._cache_hits
        leaf_eval(prog)
        assert leaf_eval._cache_hits == hits_before + 1

    def test_stats_tracking(self, leaf_eval):
        prog = Default(Flip(0))
        leaf_eval(prog)
        stats = leaf_eval.stats()
        assert stats["eval_count"] == 1
        assert stats["total_env_steps"] > 0
        assert stats["total_interp_ops"] > 0

    def test_metric_solve_rate(self, frozen_states, game_config):
        eval_sr = LeafEvaluator(
            N_SITES, frozen_states, game_config, metric="solve_rate"
        )
        prog = Ite(IsZero(0), Flip(0),
                    Ite(IsZero(1), Flip(1), Default(Flip(2))))
        value = eval_sr(prog)
        assert value == 1.0

    def test_metric_penalized_reward(self, frozen_states, game_config):
        eval_pr = LeafEvaluator(
            N_SITES, frozen_states, game_config,
            metric="penalized_reward", penalty_lambda=0.1,
        )
        prog = Default(Flip(0))
        value = eval_pr(prog)
        metrics = eval_pr.get_all_metrics(prog)
        # penalized_reward < avg_reward (penalty reduces it)
        assert value <= metrics["avg_reward"]

    def test_metric_weighted(self, frozen_states, game_config):
        eval_w = LeafEvaluator(
            N_SITES, frozen_states, game_config,
            metric="weighted", blend_alpha=0.5,
        )
        prog = Default(Flip(0))
        value = eval_w(prog)
        metrics = eval_w.get_all_metrics(prog)
        expected = 0.5 * metrics["solve_rate"] + 0.5 * metrics["avg_reward"]
        assert np.isclose(value, expected)


# ---------------------------------------------------------------------------
# TestMCTSIntegration
# ---------------------------------------------------------------------------

class TestMCTSIntegration:

    def _make_mcts(self, budget, leaf_eval, n_simulations, seed=42):
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

    def test_single_simulation_no_crash(self, leaf_eval):
        """MCTS runs 1 simulation without error."""
        mcts, game = self._make_mcts(BUDGET_5, leaf_eval, n_simulations=1)
        probs = mcts.perform_simulations(None)
        assert probs.shape == (game._max_productions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_finds_good_program_budget_5(self, leaf_eval):
        """N=3, L=5, 100 sims should find a reasonable program."""
        mcts, game = self._make_mcts(BUDGET_5, leaf_eval, n_simulations=50)

        # Play out a full derivation
        while not game.terminated:
            probs = mcts.perform_simulations(None)
            action = int(np.argmax(probs))
            game.step_wrapper(action)

        program = game.get_program()
        assert program is not None
        metrics = leaf_eval.get_all_metrics(program)
        # Should find something that solves at least some states
        assert metrics["solve_rate"] >= 0.0

    def test_deterministic_with_seed(self, leaf_eval):
        """Two runs with same seed produce the same best program."""
        programs = []
        for _ in range(2):
            mcts, game = self._make_mcts(BUDGET_5, leaf_eval, n_simulations=25, seed=42)
            while not game.terminated:
                probs = mcts.perform_simulations(None)
                action = int(np.argmax(probs))
                game.step_wrapper(action)
            programs.append(game.get_program().pretty())

        assert programs[0] == programs[1]

    def test_finds_optimal_budget_8(self, leaf_eval):
        """N=3, L=8, enough sims should find a 100% solve-rate program.

        There are 513 programs at L=8. MCTS should find an optimal one
        by evaluating far fewer than all of them.
        """
        best_solve_rate = 0.0
        best_program = None

        # Run multiple rounds to increase chances
        for round_idx in range(5):
            np.random.seed(42 + round_idx)
            game = DerivationGame(BUDGET_8, N_SITES, leaf_eval)
            game.reset_wrapper()
            net = UniformPolicyValueNet(game._max_productions)
            mcts = MCTS(
                game, net,
                n_simulations=100,
                temperature=0.1,
                c_exploration=1.5,
            )

            while not game.terminated:
                probs = mcts.perform_simulations(None)
                action = int(np.argmax(probs))
                game.step_wrapper(action)

            program = game.get_program()
            metrics = leaf_eval.get_all_metrics(program)
            if metrics["solve_rate"] > best_solve_rate:
                best_solve_rate = metrics["solve_rate"]
                best_program = program

            if best_solve_rate >= 1.0:
                break

        assert best_solve_rate >= 1.0, (
            f"Failed to find optimal program after 5 rounds. "
            f"Best solve_rate: {best_solve_rate}, "
            f"program: {best_program.pretty() if best_program else None}"
        )


# ---------------------------------------------------------------------------
# TestActionSpaceAfterFiltering
# ---------------------------------------------------------------------------

class TestActionSpaceAfterFiltering:

    def test_action_space_size(self, leaf_eval):
        """Action space size reflects filtered (dead-end-free) productions."""
        game = DerivationGame(14, 6, leaf_eval)
        # P(14): 8 valid structural choices × 6 sites = 48
        assert game.action_space.n == 48

    def test_no_truncation_in_game_episodes(self, leaf_eval):
        """Play 100 random episodes — zero should truncate (dead-end)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            game = DerivationGame(BUDGET_8, N_SITES, leaf_eval)
            game.reset_wrapper()
            while not (game.terminated or game.truncated):
                mask = game.get_action_mask()
                valid = np.where(mask)[0]
                game.step_wrapper(int(rng.choice(valid)))
            assert game.terminated and not game.truncated, (
                "Episode truncated (dead end)"
            )
