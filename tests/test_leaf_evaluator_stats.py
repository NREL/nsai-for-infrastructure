"""
Tests for LeafEvaluator stats export/merge correctness.

Verifies that the stats inflation bug (201x growth per iteration) is fixed:
- export_caches() returns deltas, not inherited totals
- __getstate__ clears caches and stats for pickle
- merge_caches() accumulates correctly
- Multi-iteration simulation shows linear, not exponential, growth
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alphazeropp.synthesis.ast_nodes import Flip, IsZero, Ite, Default
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SITES = 3
N_ONES = 2


@pytest.fixture
def game_config():
    return GameConfig(bit_flip=True, sparse_reward=False, n_ones=N_ONES)


@pytest.fixture
def frozen_states():
    return all_initial_states(N_SITES, N_ONES)


@pytest.fixture
def leaf_eval(frozen_states, game_config):
    return LeafEvaluator(N_SITES, frozen_states, game_config, metric="avg_reward")


def _make_program(idx: int) -> Ite:
    """Create a unique program by varying both condition and action sites.

    Generates N_SITES * N_SITES = 9 distinct programs for N_SITES=3.
    """
    cond_site = idx % N_SITES
    action_site = (idx // N_SITES) % N_SITES
    return Ite(IsZero(cond_site), Flip(action_site), Default(Flip(0)))


# ---------------------------------------------------------------------------
# Tests: export_caches returns deltas
# ---------------------------------------------------------------------------


class TestExportDeltas:

    def test_export_returns_deltas_from_zero(self, leaf_eval):
        """Fresh evaluator: export should match actual work."""
        for i in range(5):
            leaf_eval(_make_program(i))
        export = leaf_eval.export_caches()
        assert export["_eval_count"] == 5
        assert export["_total_env_steps"] > 0
        assert export["_total_interp_ops"] > 0

    def test_export_returns_deltas_after_snapshot(self, leaf_eval):
        """After snapshot_baseline, export only counts new work."""
        # Do some initial work
        for i in range(3):
            leaf_eval(_make_program(i))
        assert leaf_eval._eval_count == 3

        # Snapshot baseline
        leaf_eval.snapshot_baseline()

        # Do more work
        for i in range(3, 7):
            leaf_eval(_make_program(i))
        assert leaf_eval._eval_count == 7

        # Export should only reflect post-snapshot work
        export = leaf_eval.export_caches()
        assert export["_eval_count"] == 4  # 7 - 3

    def test_cache_hits_in_export(self, leaf_eval):
        """cache_hits should appear in export."""
        prog = _make_program(0)
        leaf_eval(prog)  # miss
        leaf_eval(prog)  # hit
        leaf_eval(prog)  # hit
        export = leaf_eval.export_caches()
        assert export["_cache_hits"] == 2


# ---------------------------------------------------------------------------
# Tests: __getstate__ pickle hook
# ---------------------------------------------------------------------------


class TestPickleHook:

    def test_pickle_clears_stats(self, leaf_eval):
        """Pickled evaluator should have zeroed stats."""
        for i in range(5):
            leaf_eval(_make_program(i))
        assert leaf_eval._eval_count == 5

        restored = pickle.loads(pickle.dumps(leaf_eval))
        assert restored._eval_count == 0
        assert restored._cache_hits == 0
        assert restored._total_env_steps == 0
        assert restored._total_interp_ops == 0

    def test_pickle_clears_cache(self, leaf_eval):
        """Pickled evaluator should have empty caches."""
        for i in range(5):
            leaf_eval(_make_program(i))
        assert len(leaf_eval._cache) == 5

        restored = pickle.loads(pickle.dumps(leaf_eval))
        assert len(restored._cache) == 0
        assert len(restored._full_cache) == 0
        assert len(restored._program_cache) == 0

    def test_pickle_preserves_config(self, leaf_eval):
        """Pickled evaluator should preserve configuration (metric, etc.)."""
        restored = pickle.loads(pickle.dumps(leaf_eval))
        assert restored.metric == leaf_eval.metric
        assert restored.n_sites == leaf_eval.n_sites
        assert len(restored.frozen_states) == len(leaf_eval.frozen_states)

    def test_pickled_worker_export_is_pure_delta(self, leaf_eval):
        """Simulate worker: pickle → evaluate → export. Export = pure delta."""
        # Main accumulates some stats
        for i in range(5):
            leaf_eval(_make_program(i))

        # Simulate pickling to worker
        worker_eval = pickle.loads(pickle.dumps(leaf_eval))

        # Worker does its own work
        for i in range(5, 8):
            worker_eval(_make_program(i))

        # Worker export should be pure delta (3 evals, not 5+3=8)
        export = worker_eval.export_caches()
        assert export["_eval_count"] == 3


# ---------------------------------------------------------------------------
# Tests: merge_caches
# ---------------------------------------------------------------------------


class TestMergeCaches:

    def test_merge_accumulates_deltas(self, leaf_eval):
        """Main with E evals + merge of W delta = E + W total."""
        for i in range(5):
            leaf_eval(_make_program(i))
        assert leaf_eval._eval_count == 5

        # Simulate worker export (pure delta)
        worker_export = {
            "_cache": {},
            "_full_cache": {},
            "_program_cache": {},
            "_eval_count": 10,
            "_cache_hits": 3,
            "_total_env_steps": 100,
            "_total_interp_ops": 500,
        }
        leaf_eval.merge_caches(worker_export)
        assert leaf_eval._eval_count == 15
        assert leaf_eval._cache_hits == 3

    def test_merge_cache_hits_accumulated(self, leaf_eval):
        """cache_hits from multiple workers should sum."""
        leaf_eval.merge_caches({
            "_cache": {}, "_full_cache": {}, "_program_cache": {},
            "_eval_count": 0, "_cache_hits": 5,
            "_total_env_steps": 0, "_total_interp_ops": 0,
        })
        leaf_eval.merge_caches({
            "_cache": {}, "_full_cache": {}, "_program_cache": {},
            "_eval_count": 0, "_cache_hits": 7,
            "_total_env_steps": 0, "_total_interp_ops": 0,
        })
        assert leaf_eval._cache_hits == 12


# ---------------------------------------------------------------------------
# Tests: No multiplicative growth
# ---------------------------------------------------------------------------


class TestNoMultiplicativeGrowth:

    def test_linear_growth_over_iterations(self, frozen_states, game_config):
        """Simulate 5 iterations of pickle→work→export→merge.

        With the fix, eval_count should grow linearly (N * work_per_iter),
        not exponentially (201^N).
        """
        n_tasks = 200  # matches n_games_per_train
        work_per_task = 3  # programs evaluated per task

        main_eval = LeafEvaluator(
            N_SITES, frozen_states, game_config, metric="avg_reward"
        )

        for iteration in range(5):
            # Snapshot baseline (for sequential mode)
            main_eval.snapshot_baseline()

            total_delta = 0
            for task_id in range(n_tasks):
                # Simulate pickle to worker
                worker = pickle.loads(pickle.dumps(main_eval))

                # Worker evaluates unique programs
                for j in range(work_per_task):
                    prog_id = iteration * n_tasks * work_per_task + task_id * work_per_task + j
                    worker(_make_program(prog_id % N_SITES))

                # Worker exports delta
                export = worker.export_caches()
                total_delta += export["_eval_count"]

                # Main merges
                main_eval.merge_caches(export)

            # Each iteration should add exactly n_tasks * work_per_task
            expected = (iteration + 1) * n_tasks * work_per_task
            assert main_eval._eval_count == expected, (
                f"Iteration {iteration + 1}: expected {expected}, "
                f"got {main_eval._eval_count}"
            )

    def test_stats_method_reflects_true_count(self, leaf_eval):
        """stats() should return actual cumulative count, not inflated."""
        for i in range(5):
            leaf_eval(_make_program(i))

        # Simulate one round of pickle→work→export→merge
        leaf_eval.snapshot_baseline()
        worker = pickle.loads(pickle.dumps(leaf_eval))
        for i in range(3):
            worker(_make_program(i))
        leaf_eval.merge_caches(worker.export_caches())

        stats = leaf_eval.stats()
        assert stats["eval_count"] == 8  # 5 original + 3 from worker
