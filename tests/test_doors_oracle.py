"""Tests for Doors oracle policy, reachable state enumeration, and env audits."""

import numpy as np
import pytest

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.oracle import (
    enumerate_reachable_states,
    optimal_return,
    optimal_steps,
    reachable_state_count,
    run_oracle_episode,
)


def _make_env(D, locs_per_room=2):
    """Create a DoorsPDDLLiteEnv for D rooms via DoorsGameConfig."""
    opt_steps = 2 * (D - 1) + 1
    horizon = max(15, opt_steps * 5)
    cfg = DoorsGameConfig(
        num_rooms=D, locs_per_room=locs_per_room, horizon=horizon,
    )
    return cfg.make_env(cfg.obs_size())


# ---------------------------------------------------------------------------
# Oracle optimality
# ---------------------------------------------------------------------------

class TestOracleIsOptimal:
    """Verify oracle solves optimally for small D values."""

    @pytest.mark.parametrize("D", [2, 3, 5])
    def test_oracle_solves_optimally(self, D):
        env = _make_env(D)
        result = run_oracle_episode(env)
        assert result["solved"], f"Oracle failed to solve D={D}"
        assert result["steps"] == optimal_steps(D), (
            f"D={D}: expected {optimal_steps(D)} steps, got {result['steps']}"
        )
        assert result["reward"] == pytest.approx(optimal_return(D), abs=1e-6), (
            f"D={D}: expected reward {optimal_return(D)}, got {result['reward']}"
        )

    def test_oracle_matches_d2_hardcoded_baseline(self):
        """Oracle produces the same action sequence as the known D=2 optimal [1, 4, 3]."""
        env = DoorsPDDLLiteEnv.make_d2()
        result = run_oracle_episode(env)
        assert result["actions"] == [1, 4, 3]
        assert result["reward"] == pytest.approx(1.07, abs=1e-6)

    @pytest.mark.parametrize("D", [10, 20])
    def test_oracle_solves_larger(self, D):
        """Oracle solves larger instances in optimal steps."""
        env = _make_env(D)
        result = run_oracle_episode(env)
        assert result["solved"]
        assert result["steps"] == optimal_steps(D)


# ---------------------------------------------------------------------------
# Reachable state count
# ---------------------------------------------------------------------------

class TestReachableStateCount:
    """Verify the closed-form reachable state formula."""

    @pytest.mark.parametrize("D,locs,expected", [
        (2, 2, 6),
        (3, 2, 12),
        (5, 2, 30),
        (10, 2, 110),
        (2, 3, 9),
        (3, 3, 18),
    ])
    def test_formula(self, D, locs, expected):
        assert reachable_state_count(D, locs) == expected

    @pytest.mark.parametrize("D", [2, 3, 5])
    def test_enumeration_matches_formula(self, D):
        """BFS enumeration of reachable states matches the closed-form count."""
        env = _make_env(D)
        visited = enumerate_reachable_states(env)
        expected = reachable_state_count(D, locs_per_room=2)
        assert len(visited) == expected, (
            f"D={D}: BFS found {len(visited)} states, formula gives {expected}"
        )


# ---------------------------------------------------------------------------
# Reward accounting
# ---------------------------------------------------------------------------

class TestRewardAccounting:
    """Verify reward formulas match actual env execution."""

    @pytest.mark.parametrize("D", [2, 3, 5])
    def test_optimal_reward_matches_formula(self, D):
        env = _make_env(D)
        result = run_oracle_episode(env)
        expected = optimal_return(D)
        assert result["reward"] == pytest.approx(expected, abs=1e-6)

    def test_optimal_steps_formula(self):
        """optimal_steps(D) = 2*(D-1) + 1."""
        for D in range(2, 20):
            assert optimal_steps(D) == 2 * (D - 1) + 1

    def test_optimal_return_components(self):
        """Decompose optimal return into goal + unlocks - penalties."""
        D = 5
        K = D - 1
        steps = optimal_steps(D)
        expected = 1.0 + K * 0.1 - steps * 0.01
        assert optimal_return(D) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Horizon and truncation
# ---------------------------------------------------------------------------

class TestHorizonTruncation:
    """Verify truncation semantics."""

    @pytest.mark.parametrize("D", [2, 3])
    def test_noop_hits_horizon(self, D):
        """NOOPs until horizon causes truncation, not termination."""
        env = _make_env(D)
        obs, _ = env.reset()
        noop = env.M + env.K
        for _ in range(env.horizon - 1):
            obs, _, terminated, truncated, _ = env.step(noop)
            assert not truncated, "Truncated before horizon"
            assert not terminated
        obs, _, terminated, truncated, _ = env.step(noop)
        assert truncated
        assert not terminated

    def test_no_truncation_before_horizon(self):
        """At horizon-1 steps, episode is still active."""
        env = DoorsPDDLLiteEnv.make_d2()
        env.reset()
        noop = env.M + env.K
        for _ in range(env.horizon - 1):
            _, _, terminated, truncated, _ = env.step(noop)
        assert not truncated
        assert not terminated


# ---------------------------------------------------------------------------
# sb3 check_env
# ---------------------------------------------------------------------------

class TestSB3CheckEnv:
    """Verify env passes stable_baselines3 env checker."""

    @pytest.mark.parametrize("D", [2, 3])
    def test_check_env(self, D):
        sb3_checker = pytest.importorskip(
            "stable_baselines3.common.env_checker"
        )
        env = _make_env(D)
        # check_env raises or warns on issues; should not raise
        sb3_checker.check_env(env, warn=True, skip_render_check=True)
