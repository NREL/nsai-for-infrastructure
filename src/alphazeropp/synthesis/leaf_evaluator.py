"""
Leaf evaluator for MCTS-guided program synthesis.

Evaluates complete DSL programs on frozen BitString initial states,
returning a scalar value for MCTS to back-propagate. Results are cached
by program structure (via pretty()) to avoid redundant evaluation.

Supports multiple evaluation metrics:
  - avg_reward:        Mean shaped reward across frozen states (continuous)
  - solve_rate:        Fraction of frozen states solved (discrete)
  - penalized_reward:  avg_reward - lambda * avg_interp_ops / max_ops
  - weighted:          alpha * solve_rate + (1-alpha) * avg_reward
                       (adaptive: uses raw avg_reward when solve_rate=0)
  - keys_progress:     keys_picked/total_keys + 0.1 * avg_reward
                       (domain-specific milestone metric for Doors)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from alphazeropp.synthesis.ast_nodes import Program
from alphazeropp.synthesis.interpreter import run_policy_episode


VALID_METRICS = ("avg_reward", "solve_rate", "penalized_reward", "weighted",
                 "keys_progress")


class LeafEvaluator:
    """Evaluates complete DSL programs on frozen BitString initial states.

    When MCTS reaches a terminal DerivationState (complete program),
    this runs the program on all frozen initial states and returns
    a scalar value determined by the metric parameter.
    """

    def __init__(
        self,
        n_sites: int,
        frozen_states: Sequence[np.ndarray],
        game_config,
        metric: str = "avg_reward",
        penalty_lambda: float = 0.1,
        blend_alpha: float = 0.5,
        is_solved: Optional[Callable[[np.ndarray], bool]] = None,
        normalize_rewards: bool = False,
        progress_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}, must be one of {VALID_METRICS}"
            )
        if metric == "keys_progress" and progress_fn is None:
            raise ValueError("keys_progress metric requires progress_fn")
        self.n_sites = n_sites
        self.is_solved = is_solved
        self.frozen_states = list(frozen_states)
        self.game_config = game_config
        self.metric = metric
        self.penalty_lambda = penalty_lambda
        self.blend_alpha = blend_alpha
        self.normalize_rewards = normalize_rewards
        self.progress_fn = progress_fn

        # Max interpretation ops for normalization (penalized_reward metric).
        # A rough upper bound: budget * max_steps. We use n_sites * max_steps
        # as a reasonable normalizer since programs can't exceed budget ops.
        self._max_ops = n_sites * game_config.max_steps(n_sites)

        # Caching
        self._cache: dict[str, float] = {}
        self._full_cache: dict[str, dict] = {}
        self._program_cache: dict[str, Program] = {}

        # Statistics
        self._eval_count = 0
        self._cache_hits = 0
        self._total_env_steps = 0
        self._total_interp_ops = 0

        # Baseline stats for delta export (see snapshot_baseline)
        self._base_eval_count = 0
        self._base_cache_hits = 0
        self._base_total_env_steps = 0
        self._base_total_interp_ops = 0

        # Running EMA for optional reward normalization
        self._ema_mean = 0.0
        self._ema_var = 1.0
        self._ema_count = 0
        self._ema_decay = 0.99

    def __call__(self, program: Program) -> float:
        """Evaluate program, returning cached result if available.

        The returned scalar depends on self.metric.
        """
        key = program.pretty()
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        metrics = self._evaluate(program)
        value = self._compute_metric(metrics)
        if self.normalize_rewards:
            value = self._normalize(value)
        self._cache[key] = value
        self._full_cache[key] = metrics
        self._program_cache[key] = program
        return value

    def get_all_metrics(self, program: Program) -> dict:
        """Return full metrics dict for a program, using cache.

        Ensures the program is evaluated if not already cached.
        Returns dict with: solve_rate, avg_reward, avg_steps, avg_ops.
        """
        key = program.pretty()
        if key not in self._full_cache:
            self(program)  # trigger evaluation and caching
        return self._full_cache[key]

    def stats(self) -> dict:
        """Return evaluation statistics."""
        return {
            "eval_count": self._eval_count,
            "cache_hits": self._cache_hits,
            "unique_programs": len(self._cache),
            "total_env_steps": self._total_env_steps,
            "total_interp_ops": self._total_interp_ops,
        }

    def export_caches(self) -> dict:
        """Export cache data for cross-process aggregation.

        Returns delta stats (work done since baseline) rather than
        cumulative totals, so that merge_caches() on the main process
        accumulates correctly without double-counting inherited stats.
        """
        return {
            "_cache": dict(self._cache),
            "_full_cache": dict(self._full_cache),
            "_program_cache": dict(self._program_cache),
            "_eval_count": self._eval_count - self._base_eval_count,
            "_cache_hits": self._cache_hits - self._base_cache_hits,
            "_total_env_steps": self._total_env_steps - self._base_total_env_steps,
            "_total_interp_ops": self._total_interp_ops - self._base_total_interp_ops,
        }

    def merge_caches(self, other: dict):
        """Merge exported caches from a worker LeafEvaluator.

        Only adds programs not already in this evaluator's cache.
        Stats are accumulated additively (expects delta values from
        export_caches, not cumulative totals).
        """
        for key, value in other["_cache"].items():
            if key not in self._cache:
                self._cache[key] = value
                self._full_cache[key] = other["_full_cache"][key]
                self._program_cache[key] = other["_program_cache"][key]
        self._eval_count += other.get("_eval_count", 0)
        self._cache_hits += other.get("_cache_hits", 0)
        self._total_env_steps += other.get("_total_env_steps", 0)
        self._total_interp_ops += other.get("_total_interp_ops", 0)

    def snapshot_baseline(self):
        """Snapshot current stats as baseline for delta exports.

        Call before pickling into worker processes so that
        export_caches() returns only work done by the worker.
        Needed for sequential mode (n_procs < 0) where __getstate__
        is not invoked.
        """
        self._base_eval_count = self._eval_count
        self._base_cache_hits = self._cache_hits
        self._base_total_env_steps = self._total_env_steps
        self._base_total_interp_ops = self._total_interp_ops

    def __getstate__(self):
        """Pickle hook: workers start with empty caches and zeroed stats.

        This fixes two problems:
        1. Stats inflation: workers no longer inherit accumulated stats
           that get double-counted when merge_caches() adds them back.
        2. Pickle overhead: avoids serializing O(cache_size) entries
           per task (cache can grow to 1M+ programs).
        """
        state = self.__dict__.copy()
        state['_cache'] = {}
        state['_full_cache'] = {}
        state['_program_cache'] = {}
        state['_eval_count'] = 0
        state['_cache_hits'] = 0
        state['_total_env_steps'] = 0
        state['_total_interp_ops'] = 0
        state['_base_eval_count'] = 0
        state['_base_cache_hits'] = 0
        state['_base_total_env_steps'] = 0
        state['_base_total_interp_ops'] = 0
        return state

    def _evaluate(self, program: Program) -> dict:
        """Run the program on all frozen states and collect raw metrics."""
        self._eval_count += 1
        solved_count = 0
        total_steps = 0
        total_ops = 0
        total_reward = 0.0
        total_progress = 0.0

        for x0 in self.frozen_states:
            env = self.game_config.make_env(
                self.n_sites, frozen_states=[x0]
            )
            env.reset()
            result = run_policy_episode(env, program,
                                               is_solved=self.is_solved)
            if result.solved:
                solved_count += 1
            total_steps += result.total_env_steps
            total_ops += result.total_interp_ops
            total_reward += result.cumulative_reward
            if self.progress_fn is not None:
                total_progress += self.progress_fn(result.final_state)

        n = len(self.frozen_states)
        self._total_env_steps += total_steps
        self._total_interp_ops += total_ops

        metrics = {
            "solve_rate": solved_count / n,
            "avg_reward": total_reward / n,
            "avg_steps": total_steps / n,
            "avg_ops": total_ops / n,
            "n_episodes": n,
        }
        if self.progress_fn is not None:
            metrics["keys_progress"] = total_progress / n
        return metrics

    def _compute_metric(self, metrics: dict) -> float:
        """Compute the scalar value from raw metrics based on self.metric."""
        if self.metric == "avg_reward":
            return metrics["avg_reward"]
        elif self.metric == "solve_rate":
            return metrics["solve_rate"]
        elif self.metric == "penalized_reward":
            penalty = self.penalty_lambda * metrics["avg_ops"] / max(self._max_ops, 1)
            return metrics["avg_reward"] - penalty
        elif self.metric == "weighted":
            sr = metrics["solve_rate"]
            ar = metrics["avg_reward"]
            if sr > 0:
                return self.blend_alpha * sr + (1 - self.blend_alpha) * ar
            else:
                # No compression when solve_rate=0: use full avg_reward
                # signal so the value head can discriminate partial progress
                return ar
        elif self.metric == "keys_progress":
            kp = metrics.get("keys_progress", 0.0)
            return kp + 0.1 * metrics["avg_reward"]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _normalize(self, value: float) -> float:
        """Normalize value using running EMA statistics.

        Maps values to approximately zero mean, unit variance based on
        an exponential moving average of observed values.
        """
        self._ema_count += 1
        if self._ema_count == 1:
            self._ema_mean = value
            self._ema_var = 1.0
        else:
            d = self._ema_decay
            delta = value - self._ema_mean
            self._ema_mean = d * self._ema_mean + (1 - d) * value
            self._ema_var = d * self._ema_var + (1 - d) * delta ** 2
        return (value - self._ema_mean) / (self._ema_var ** 0.5 + 1e-8)
