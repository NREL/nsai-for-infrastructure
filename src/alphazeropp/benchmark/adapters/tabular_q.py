"""Tabular Q-learning adapter for the benchmark harness."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Iterator

import numpy as np

from alphazeropp.benchmark.adapters.base import BenchmarkAlgorithm
from alphazeropp.benchmark.adapters.state_utils import obs_to_state_id
from alphazeropp.benchmark.eval_loop import evaluate_policy, aggregate_episodes
from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv


def _make_greedy_policy(
    Q_snapshot: dict[tuple, np.ndarray],
    mask_mode: str,
    n_actions: int,
):
    """Build a greedy policy from a frozen Q-table snapshot."""

    def policy_fn(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
        sid = obs_to_state_id(obs, env)
        q = Q_snapshot.get(sid, np.zeros(n_actions))
        if mask_mode == "precondition":
            mask = env.action_masks("precondition")
            q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    return policy_fn


class TabularQAdapter(BenchmarkAlgorithm):
    """Tabular Q-learning with optional precondition masking.

    total_steps is interpreted as total training episodes.
    """

    def __init__(
        self,
        D: int,
        locs_per_room: int = 2,
        mask_mode: str = "none",
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
    ):
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end

    def name(self) -> str:
        return "tabular_q"

    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        rng = np.random.default_rng(seed)
        train_env = env_factory()
        n_actions = train_env.action_space.n
        use_mask = self._mask_mode == "precondition"

        Q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

        env_steps_total = 0
        checkpoint_idx = 0
        t0 = time.time()

        for ep in range(total_steps):
            # Linear epsilon decay
            frac = ep / max(1, total_steps - 1)
            epsilon = self._epsilon_start + (self._epsilon_end - self._epsilon_start) * frac

            obs, _ = train_env.reset(seed=seed + ep)
            sid = obs_to_state_id(obs, train_env)

            for _ in range(train_env.horizon):
                # Epsilon-greedy action selection
                if rng.random() < epsilon:
                    if use_mask:
                        valid = np.where(train_env.action_masks("precondition"))[0]
                        action = int(rng.choice(valid))
                    else:
                        action = int(rng.integers(n_actions))
                else:
                    q = Q[sid]
                    if use_mask:
                        mask = train_env.action_masks("precondition")
                        q = np.where(mask, q, -np.inf)
                    action = int(np.argmax(q))

                next_obs, reward, terminated, truncated, _ = train_env.step(action)
                next_sid = obs_to_state_id(next_obs, train_env)
                env_steps_total += 1

                # TD target
                if terminated or truncated:
                    td_target = reward
                else:
                    if use_mask:
                        next_valid = train_env.action_masks("precondition")
                        next_q = Q[next_sid]
                        valid_q = next_q[next_valid]
                        max_next = valid_q.max() if len(valid_q) > 0 else 0.0
                    else:
                        max_next = np.max(Q[next_sid])
                    td_target = reward + self._gamma * max_next

                Q[sid][action] += self._alpha * (td_target - Q[sid][action])

                sid = next_sid
                if terminated or truncated:
                    break

            # Checkpoint evaluation
            if (ep + 1) % eval_interval == 0:
                Q_snapshot = {s: v.copy() for s, v in Q.items()}
                greedy = _make_greedy_policy(Q_snapshot, self._mask_mode, n_actions)

                episodes = evaluate_policy(
                    policy_fn=greedy,
                    env_factory=eval_env_factory,
                    n_episodes=eval_episodes,
                    seed=seed,
                )
                agg = aggregate_episodes(episodes)

                yield CheckpointResult(
                    algorithm=self.name(),
                    seed=seed,
                    D=self._D,
                    locs_per_room=self._locs_per_room,
                    mask_mode=self._mask_mode,
                    env_steps=env_steps_total,
                    train_episodes=ep + 1,
                    learner_updates=env_steps_total,
                    eval_checkpoint_idx=checkpoint_idx,
                    wall_clock_sec=time.time() - t0,
                    extra={
                        "epsilon": round(epsilon, 4),
                        "alpha": self._alpha,
                        "gamma": self._gamma,
                        "q_table_size": len(Q),
                    },
                    **agg,
                )
                checkpoint_idx += 1
