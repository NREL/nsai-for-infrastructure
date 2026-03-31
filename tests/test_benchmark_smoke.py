"""Smoke tests for the benchmark harness."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alphazeropp.benchmark.env_factory import make_env, make_env_factory
from alphazeropp.benchmark.eval_loop import (
    EpisodeResult,
    evaluate_policy,
    aggregate_episodes,
)
from alphazeropp.benchmark.result_schema import CheckpointResult, ResultWriter
from alphazeropp.benchmark.solve_criterion import check_sustained_solve
from alphazeropp.instances.doors.oracle import optimal_return, oracle_action


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

class TestEnvFactory:
    @pytest.mark.parametrize("D", [2, 3, 10])
    def test_creates_valid_env(self, D):
        env = make_env(D=D)
        obs, _ = env.reset()
        assert obs.shape == (env._obs_size,)
        assert env.action_space.n == env.M + env.K + 1

    def test_mask_mode_stored(self):
        env = make_env(D=2, mask_mode="precondition")
        assert env.benchmark_mask_mode == "precondition"

    def test_factory_creates_independent_envs(self):
        factory = make_env_factory(D=2)
        env1 = factory()
        env2 = factory()
        assert env1 is not env2

    def test_gymnasium_check_env(self):
        """Env passes gymnasium's built-in env_checker."""
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            pytest.skip("gymnasium env_checker not available")
        env = make_env(D=2)
        check_env(env, skip_render_check=True)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

class TestEvalLoop:
    def test_oracle_achieves_optimal(self):
        factory = make_env_factory(D=2)
        episodes = evaluate_policy(
            policy_fn=oracle_action,
            env_factory=factory,
            n_episodes=5,
            seed=0,
        )
        assert len(episodes) == 5
        for ep in episodes:
            assert ep.solved
            assert ep.episode_return == pytest.approx(optimal_return(2), abs=1e-6)
            assert ep.invalid_action_count == 0

    def test_random_policy_lower_bound(self):
        def random_policy(obs, env):
            return int(env.action_space.sample())

        factory = make_env_factory(D=3)
        episodes = evaluate_policy(
            policy_fn=random_policy,
            env_factory=factory,
            n_episodes=20,
            seed=42,
        )
        agg = aggregate_episodes(episodes)
        # Random policy should not consistently solve D=3
        assert agg["eval_success_rate"] < 1.0
        assert agg["eval_return_mean"] < optimal_return(3)

    def test_aggregate_empty(self):
        agg = aggregate_episodes([])
        assert agg["eval_return_mean"] == 0.0
        assert agg["eval_success_rate"] == 0.0

    def test_invalid_action_counting(self):
        """Random policy in mask_mode=none should produce some invalid actions."""
        def random_policy(obs, env):
            return int(env.action_space.sample())

        factory = make_env_factory(D=3, mask_mode="none")
        episodes = evaluate_policy(
            policy_fn=random_policy,
            env_factory=factory,
            n_episodes=10,
            seed=42,
        )
        total_invalid = sum(ep.invalid_action_count for ep in episodes)
        # With random policy on D=3, most actions will be invalid
        assert total_invalid > 0


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class TestResultSchema:
    def _make_result(self, **overrides) -> CheckpointResult:
        defaults = dict(
            algorithm="test",
            seed=42,
            D=3,
            locs_per_room=2,
            mask_mode="none",
            env_steps=1000,
            train_episodes=50,
            learner_updates=10,
            eval_checkpoint_idx=1,
            eval_return_mean=0.85,
            eval_return_std=0.1,
            eval_success_rate=0.8,
            eval_episode_length_mean=12.0,
            max_room_reached_mean=2.5,
            keys_picked_mean=1.8,
            invalid_action_rate=0.05,
            wall_clock_sec=30.0,
        )
        defaults.update(overrides)
        return CheckpointResult(**defaults)

    def test_to_dict(self):
        r = self._make_result(extra={"custom_field": 42})
        d = r.to_dict()
        assert d["algorithm"] == "test"
        assert d["extra_custom_field"] == 42
        assert "extra" not in d

    def test_csv_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultWriter(output_dir=tmpdir, run_name="test_run")

            r1 = self._make_result(eval_checkpoint_idx=1)
            r2 = self._make_result(eval_checkpoint_idx=2, env_steps=2000)
            writer.append(r1)
            writer.append(r2)

            # Check JSONL
            jsonl_path = Path(tmpdir) / "test_run.jsonl"
            assert jsonl_path.exists()
            lines = jsonl_path.read_text().strip().split("\n")
            assert len(lines) == 2
            parsed = json.loads(lines[0])
            assert parsed["algorithm"] == "test"
            assert parsed["eval_checkpoint_idx"] == 1

            # Check CSV
            csv_path = Path(tmpdir) / "test_run.csv"
            assert csv_path.exists()
            csv_lines = csv_path.read_text().strip().split("\n")
            assert len(csv_lines) == 3  # header + 2 rows
            assert "algorithm" in csv_lines[0]


# ---------------------------------------------------------------------------
# Solve criterion
# ---------------------------------------------------------------------------

class TestSolveCriterion:
    def _make_cp(self, success_rate, return_mean, env_steps=0, wall_clock=0.0):
        return CheckpointResult(
            algorithm="test", seed=0, D=3, locs_per_room=2,
            mask_mode="none", env_steps=env_steps, train_episodes=0,
            learner_updates=0, eval_checkpoint_idx=0,
            eval_return_mean=return_mean, eval_return_std=0.0,
            eval_success_rate=success_rate,
            eval_episode_length_mean=5.0,
            max_room_reached_mean=2.0, keys_picked_mean=2.0,
            invalid_action_rate=0.0, wall_clock_sec=wall_clock,
        )

    def test_not_enough_checkpoints(self):
        cps = [self._make_cp(1.0, 2.0)]
        assert check_sustained_solve(cps, D=3) is None

    def test_sustained_solve_detected(self):
        oracle_ret = optimal_return(3)
        cps = [
            self._make_cp(1.0, oracle_ret, env_steps=100, wall_clock=1.0),
            self._make_cp(1.0, oracle_ret, env_steps=200, wall_clock=2.0),
            self._make_cp(1.0, oracle_ret, env_steps=300, wall_clock=3.0),
        ]
        result = check_sustained_solve(cps, D=3)
        assert result is not None
        assert result == (300, 3.0)

    def test_intermittent_fail_resets_counter(self):
        oracle_ret = optimal_return(3)
        cps = [
            self._make_cp(1.0, oracle_ret, env_steps=100, wall_clock=1.0),
            self._make_cp(1.0, oracle_ret, env_steps=200, wall_clock=2.0),
            self._make_cp(0.5, 0.5, env_steps=300, wall_clock=3.0),  # fail
            self._make_cp(1.0, oracle_ret, env_steps=400, wall_clock=4.0),
            self._make_cp(1.0, oracle_ret, env_steps=500, wall_clock=5.0),
        ]
        # Only 2 consecutive after reset, not 3
        assert check_sustained_solve(cps, D=3) is None

    def test_solve_after_intermittent_fail(self):
        oracle_ret = optimal_return(3)
        cps = [
            self._make_cp(1.0, oracle_ret, env_steps=100, wall_clock=1.0),
            self._make_cp(0.5, 0.5, env_steps=200, wall_clock=2.0),  # fail
            self._make_cp(1.0, oracle_ret, env_steps=300, wall_clock=3.0),
            self._make_cp(1.0, oracle_ret, env_steps=400, wall_clock=4.0),
            self._make_cp(1.0, oracle_ret, env_steps=500, wall_clock=5.0),
        ]
        result = check_sustained_solve(cps, D=3)
        assert result is not None
        assert result == (500, 5.0)


# ---------------------------------------------------------------------------
# Seeding determinism
# ---------------------------------------------------------------------------

class TestSeedingDeterminism:
    def test_same_seed_same_results(self):
        factory = make_env_factory(D=2)
        ep1 = evaluate_policy(oracle_action, factory, n_episodes=5, seed=123)
        ep2 = evaluate_policy(oracle_action, factory, n_episodes=5, seed=123)
        for a, b in zip(ep1, ep2):
            assert a.episode_return == b.episode_return
            assert a.episode_steps == b.episode_steps
            assert a.solved == b.solved


# ---------------------------------------------------------------------------
# End-to-end smoke test with oracle adapter
# ---------------------------------------------------------------------------

class TestOracleAdapterSmoke:
    def test_oracle_benchmark_end_to_end(self):
        from alphazeropp.benchmark.adapters.oracle import OracleAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = OracleAdapter(D=2)
            train_factory = make_env_factory(D=2)
            eval_factory = make_env_factory(D=2)
            writer = ResultWriter(output_dir=tmpdir, run_name="oracle_D2_seed0")

            for cp in adapter.train_and_yield_checkpoints(
                env_factory=train_factory,
                eval_env_factory=eval_factory,
                total_steps=1,
                eval_interval=1,
                eval_episodes=10,
                seed=0,
            ):
                writer.append(cp)

            # Verify files
            assert writer.csv_path.exists()
            assert writer.jsonl_path.exists()

            # Verify content
            assert len(writer.results) == 1
            cp = writer.results[0]
            assert cp.algorithm == "oracle"
            assert cp.eval_success_rate == 1.0
            assert cp.eval_return_mean == pytest.approx(optimal_return(2), abs=1e-6)
