"""Tests for tabular Q-learning adapter, state utilities, and plotting."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from alphazeropp.benchmark.adapters.state_utils import obs_to_state_id
from alphazeropp.benchmark.adapters.tabular_q import TabularQAdapter
from alphazeropp.benchmark.env_factory import make_env, make_env_factory
from alphazeropp.benchmark.result_schema import ResultWriter
from alphazeropp.instances.doors.oracle import (
    oracle_action,
    optimal_return,
    reachable_state_count,
)


# ---------------------------------------------------------------------------
# State ID utilities
# ---------------------------------------------------------------------------

class TestStateUtils:
    def test_obs_to_state_id_initial(self):
        env = make_env(D=2)
        obs, _ = env.reset()
        sid = obs_to_state_id(obs, env)
        # Initial: agent at start_loc (0), only room 0 unlocked
        assert sid == (0, 1)

    def test_obs_to_state_id_after_unlock(self):
        env = make_env(D=2)
        obs, _ = env.reset()
        # Execute oracle actions to pick key 0 and unlock room 1
        for _ in range(3):
            action = oracle_action(obs, env)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        sid = obs_to_state_id(obs, env)
        # After solving D=2: agent at goal_loc, both rooms unlocked
        assert sid[1] == 2

    def test_state_count_matches_formula(self):
        """BFS over all reachable states matches closed-form count."""
        for D in [2, 3, 5]:
            env = make_env(D=D)
            expected = reachable_state_count(D, locs_per_room=2)

            # BFS to find all reachable state IDs
            obs, _ = env.reset()
            visited = set()
            queue = [obs.copy()]
            visited.add(obs_to_state_id(obs, env))

            while queue:
                current = queue.pop(0)
                env._state = current.copy()
                env.step_count = 0
                env._terminated = False
                env._truncated = False
                mask = env.action_masks("precondition")

                for a in range(env.action_space.n):
                    if not mask[a]:
                        continue
                    env._state = current.copy()
                    env.step_count = 0
                    env._terminated = False
                    env._truncated = False
                    new_obs, _, term, _, _ = env.step(a)
                    new_sid = obs_to_state_id(new_obs, env)
                    if new_sid not in visited:
                        visited.add(new_sid)
                        if not term:
                            queue.append(new_obs.copy())

            assert len(visited) == expected, (
                f"D={D}: found {len(visited)} states, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Tabular Q-learning convergence
# ---------------------------------------------------------------------------

class TestTabularQConvergence:
    def test_tabular_q_solves_d2_masked(self):
        adapter = TabularQAdapter(D=2, mask_mode="precondition")
        factory = make_env_factory(D=2, mask_mode="precondition")
        eval_factory = make_env_factory(D=2, mask_mode="precondition")

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=factory,
            eval_env_factory=eval_factory,
            total_steps=500,
            eval_interval=100,
            eval_episodes=20,
            seed=42,
        ))
        assert len(checkpoints) == 5
        final = checkpoints[-1]
        assert final.eval_success_rate >= 0.9, (
            f"Expected >=0.9, got {final.eval_success_rate}"
        )

    def test_tabular_q_solves_d3_masked(self):
        adapter = TabularQAdapter(D=3, mask_mode="precondition")
        factory = make_env_factory(D=3, mask_mode="precondition")
        eval_factory = make_env_factory(D=3, mask_mode="precondition")

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=factory,
            eval_env_factory=eval_factory,
            total_steps=2000,
            eval_interval=500,
            eval_episodes=20,
            seed=42,
        ))
        final = checkpoints[-1]
        assert final.eval_success_rate >= 0.8, (
            f"Expected >=0.8, got {final.eval_success_rate}"
        )

    def test_tabular_q_monotonic_d2(self):
        adapter = TabularQAdapter(D=2, mask_mode="precondition")
        factory = make_env_factory(D=2, mask_mode="precondition")
        eval_factory = make_env_factory(D=2, mask_mode="precondition")

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=factory,
            eval_env_factory=eval_factory,
            total_steps=500,
            eval_interval=100,
            eval_episodes=20,
            seed=42,
        ))
        # D=2 is easy enough that even early checkpoints may be optimal,
        # so we check >= (non-decreasing) rather than strict improvement.
        assert checkpoints[-1].eval_return_mean >= checkpoints[0].eval_return_mean

    def test_tabular_q_unmasked_learns(self):
        adapter = TabularQAdapter(D=2, mask_mode="none")
        factory = make_env_factory(D=2, mask_mode="none")
        eval_factory = make_env_factory(D=2, mask_mode="none")

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=factory,
            eval_env_factory=eval_factory,
            total_steps=1000,
            eval_interval=200,
            eval_episodes=20,
            seed=42,
        ))
        final = checkpoints[-1]
        assert final.eval_return_mean > 0, (
            f"Expected positive return, got {final.eval_return_mean}"
        )

    def test_q_table_size_bounded(self):
        adapter = TabularQAdapter(D=2, mask_mode="precondition")
        factory = make_env_factory(D=2, mask_mode="precondition")
        eval_factory = make_env_factory(D=2, mask_mode="precondition")

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=factory,
            eval_env_factory=eval_factory,
            total_steps=500,
            eval_interval=500,
            eval_episodes=5,
            seed=42,
        ))
        q_size = checkpoints[-1].extra["q_table_size"]
        max_states = reachable_state_count(2, locs_per_room=2)
        assert q_size <= max_states, (
            f"Q-table has {q_size} entries, max reachable is {max_states}"
        )


# ---------------------------------------------------------------------------
# Heuristic alias
# ---------------------------------------------------------------------------

class TestHeuristicAlias:
    def test_heuristic_alias_solves_optimally(self):
        from alphazeropp.benchmark.run import _build_adapter

        class Args:
            algo = "heuristic"
            D = 3
            locs_per_room = 2
            mask_mode = "none"

        adapter = _build_adapter(Args())
        eval_factory = make_env_factory(D=3)

        checkpoints = list(adapter.train_and_yield_checkpoints(
            env_factory=eval_factory,
            eval_env_factory=eval_factory,
            total_steps=1,
            eval_interval=1,
            eval_episodes=10,
            seed=0,
        ))
        assert len(checkpoints) == 1
        assert checkpoints[0].eval_success_rate == 1.0
        assert checkpoints[0].eval_return_mean == pytest.approx(
            optimal_return(3), abs=1e-6
        )


# ---------------------------------------------------------------------------
# Plotting and summary
# ---------------------------------------------------------------------------

class TestPlotting:
    def _run_and_write(self, tmpdir, algo, D, mask_mode="precondition",
                       total_steps=1, eval_interval=1):
        """Helper: run an adapter and write results to tmpdir."""
        from alphazeropp.benchmark.adapters.oracle import OracleAdapter
        factory = make_env_factory(D=D, mask_mode=mask_mode)
        eval_factory = make_env_factory(D=D, mask_mode=mask_mode)

        if algo == "oracle":
            adapter = OracleAdapter(D=D, mask_mode=mask_mode)
        elif algo == "tabular_q":
            adapter = TabularQAdapter(D=D, mask_mode=mask_mode)
        else:
            raise ValueError(algo)

        writer = ResultWriter(output_dir=tmpdir,
                              run_name=f"{adapter.name()}_D{D}_seed0")
        for cp in adapter.train_and_yield_checkpoints(
            env_factory=factory, eval_env_factory=eval_factory,
            total_steps=total_steps, eval_interval=eval_interval,
            eval_episodes=10, seed=0,
        ):
            writer.append(cp)
        return writer.csv_path

    def test_summary_table_output(self):
        from alphazeropp.benchmark.plotting import print_summary_table
        with tempfile.TemporaryDirectory() as tmpdir:
            csv1 = self._run_and_write(tmpdir, "oracle", D=2)
            csv2 = self._run_and_write(tmpdir, "tabular_q", D=2,
                                       total_steps=200, eval_interval=200)
            table = print_summary_table([csv1, csv2])
            assert "oracle" in table
            assert "tabular_q" in table
            assert "Algorithm" in table

    def test_plot_comparison_panel(self):
        from alphazeropp.benchmark.plotting import plot_comparison_panel
        with tempfile.TemporaryDirectory() as tmpdir:
            csv1 = self._run_and_write(tmpdir, "oracle", D=2)
            csv2 = self._run_and_write(tmpdir, "tabular_q", D=2,
                                       total_steps=200, eval_interval=100)
            out = Path(tmpdir) / "panel.png"
            plot_comparison_panel([csv1, csv2], out)
            assert out.exists()
            assert out.stat().st_size > 0
