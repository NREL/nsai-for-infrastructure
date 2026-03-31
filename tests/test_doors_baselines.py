"""Tests for Doors D=2 baselines: random, optimal env, and optimal DSL."""

import numpy as np
import pytest

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.interpreter import run_policy_episode


def _build_optimal_program():
    """Construct the known-optimal 16-node DSL program for Doors D=2."""
    return Ite(
        And(Not(IsZero(1)), Not(IsZero(6))),
        Flip(4),
        Ite(
            Not(IsZero(6)),
            Flip(1),
            Ite(IsZero(3), Flip(3), Default(Flip(5))),
        ),
    )


# ---------------------------------------------------------------------------
# Optimal program structure
# ---------------------------------------------------------------------------

class TestOptimalProgram:
    """Structural checks on the 16-node optimal program."""

    def test_node_count(self):
        """Program has exactly 16 AST nodes."""
        program = _build_optimal_program()
        assert program.node_count() == 16

    def test_validate(self):
        """All Flip/IsZero indices are in [0, 7)."""
        program = _build_optimal_program()
        program.validate(7)  # should not raise


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    """Random agent baseline on Doors D=2."""

    @pytest.fixture()
    def random_stats(self):
        """Run 1000 random episodes and return stats."""
        env = DoorsPDDLLiteEnv.make_d2()
        rng = np.random.default_rng(42)
        n_episodes = 1000
        solves = 0
        total_reward = 0.0
        total_steps = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            while not done:
                action = int(rng.integers(0, env.action_space.n))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated
            if env.is_solved(obs):
                solves += 1
            total_reward += ep_reward
            total_steps += ep_steps

        return {
            "solve_rate": solves / n_episodes,
            "avg_reward": total_reward / n_episodes,
            "avg_steps": total_steps / n_episodes,
        }

    def test_random_solve_rate_bounded(self, random_stats):
        """Random agent solves well below optimal (< 50%) but above zero."""
        assert 0.0 < random_stats["solve_rate"] < 0.50

    def test_random_avg_reward_suboptimal(self, random_stats):
        """Random agent reward is well below optimal 1.07."""
        assert random_stats["avg_reward"] < 0.50


# ---------------------------------------------------------------------------
# Optimal env baseline (hardcoded actions)
# ---------------------------------------------------------------------------

class TestOptimalEnvBaseline:
    """Hardcoded optimal action sequence [1, 4, 3] on Doors D=2."""

    @pytest.fixture()
    def optimal_run(self):
        """Execute the optimal 3-step sequence."""
        env = DoorsPDDLLiteEnv.make_d2()
        obs, _ = env.reset()
        rewards = []
        for action in [1, 4, 3]:
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
        return {
            "obs": obs,
            "rewards": rewards,
            "terminated": terminated,
            "solved": env.is_solved(obs),
            "cumulative_reward": sum(rewards),
        }

    def test_optimal_env_reward(self, optimal_run):
        """Cumulative reward is 1.07."""
        assert optimal_run["cumulative_reward"] == pytest.approx(1.07, abs=0.001)

    def test_optimal_env_solved(self, optimal_run):
        """Episode terminates with env solved."""
        assert optimal_run["terminated"]
        assert optimal_run["solved"]

    def test_optimal_env_steps(self, optimal_run):
        """Exactly 3 actions required."""
        assert len(optimal_run["rewards"]) == 3

    def test_optimal_env_per_step_rewards(self, optimal_run):
        """Per-step rewards: -0.01, +0.09, +0.99."""
        r = optimal_run["rewards"]
        assert r[0] == pytest.approx(-0.01, abs=1e-6)
        assert r[1] == pytest.approx(0.09, abs=1e-6)
        assert r[2] == pytest.approx(0.99, abs=1e-6)


# ---------------------------------------------------------------------------
# Optimal DSL baseline
# ---------------------------------------------------------------------------

class TestOptimalDSLBaseline:
    """Optimal 16-node DSL program via run_policy_episode."""

    @pytest.fixture()
    def dsl_result(self):
        """Run the optimal DSL program on Doors D=2."""
        env = DoorsPDDLLiteEnv.make_d2()
        program = _build_optimal_program()
        return run_policy_episode(env, program, is_solved=env.is_solved)

    def test_dsl_solved(self, dsl_result):
        """DSL program solves the environment."""
        assert dsl_result.solved is True

    def test_dsl_reward(self, dsl_result):
        """DSL program achieves cumulative reward 1.07."""
        assert dsl_result.cumulative_reward == pytest.approx(1.07, abs=0.001)

    def test_dsl_steps(self, dsl_result):
        """DSL program solves in 3 env steps."""
        assert dsl_result.total_env_steps == 3

    def test_dsl_action_sequence(self, dsl_result):
        """DSL program produces action sequence [1, 4, 3]."""
        actions = [step.action for step in dsl_result.steps]
        assert actions == [1, 4, 3]

    def test_dsl_without_is_solved_callback_raises(self):
        """Omitting is_solved raises TypeError (keyword-only, required)."""
        env = DoorsPDDLLiteEnv.make_d2()
        program = _build_optimal_program()
        with pytest.raises(TypeError):
            run_policy_episode(env, program)  # no is_solved


# ---------------------------------------------------------------------------
# Cross-validation: DSL matches env baseline
# ---------------------------------------------------------------------------

class TestDSLEnvAgreement:
    """Verify DSL program produces identical results to hardcoded actions."""

    def test_dsl_matches_env_baseline(self):
        """DSL and env baselines agree on reward, steps, and final state."""
        # Env baseline
        env1 = DoorsPDDLLiteEnv.make_d2()
        obs1, _ = env1.reset()
        env_reward = 0.0
        for action in [1, 4, 3]:
            obs1, reward, _, _, _ = env1.step(action)
            env_reward += reward

        # DSL baseline
        env2 = DoorsPDDLLiteEnv.make_d2()
        program = _build_optimal_program()
        dsl_result = run_policy_episode(env2, program, is_solved=env2.is_solved)

        assert dsl_result.cumulative_reward == pytest.approx(env_reward, abs=1e-6)
        assert dsl_result.total_env_steps == 3
        np.testing.assert_array_equal(dsl_result.final_state, obs1)
