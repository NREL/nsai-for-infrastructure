"""AlphaZero adapter — wraps the existing training pipeline for benchmarking."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator

import numpy as np

from alphazeropp.benchmark.adapters.base import BenchmarkAlgorithm
from alphazeropp.benchmark.eval_loop import evaluate_policy, aggregate_episodes
from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.config import DoorsDirectConfig, compute_dims
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.training.gated_trainer import GatedTrainer

logger = logging.getLogger(__name__)


def _make_nn_greedy_policy(agent):
    """Create a greedy policy function from the agent's current network.

    Uses MCTS with very low temperature for near-greedy action selection.
    """
    def policy_fn(obs: np.ndarray, env: DoorsPDDLLiteEnv) -> int:
        # Use the network directly: forward pass, mask, argmax
        policy_probs, _ = agent.net.predict(obs)
        mask = env.action_masks("precondition")
        # Mask out invalid actions
        masked = np.where(mask, policy_probs, -np.inf)
        return int(np.argmax(masked))
    return policy_fn


class AlphaZeroAdapter(BenchmarkAlgorithm):
    """Wraps DoorsDirectConfig + GatedTrainer for benchmarking.

    Runs AlphaZero training iterations and evaluates with the NN greedy
    policy (no MCTS at eval time) after each iteration.
    """

    def __init__(
        self,
        D: int,
        locs_per_room: int = 2,
        mask_mode: str = "none",
        n_simulations: int = 120,
        n_games_per_train: int = 50,
        n_procs: int = -1,
        accept_threshold: float = 0.0,
    ):
        self._D = D
        self._locs_per_room = locs_per_room
        self._mask_mode = mask_mode
        self._n_simulations = n_simulations
        self._n_games_per_train = n_games_per_train
        self._n_procs = n_procs
        self._accept_threshold = accept_threshold

    def name(self) -> str:
        return "alphazero"

    def train_and_yield_checkpoints(
        self,
        env_factory: Callable[[], DoorsPDDLLiteEnv],
        eval_env_factory: Callable[[], DoorsPDDLLiteEnv],
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
        seed: int,
    ) -> Iterator[CheckpointResult]:
        t0 = time.time()

        # Build AlphaZero stack
        cfg = DoorsDirectConfig(
            num_rooms=self._D,
            locs_per_room=self._locs_per_room,
        )
        cfg.agent.mcts_params["n_simulations"] = self._n_simulations
        cfg.trainer.n_games_per_train = self._n_games_per_train
        cfg.trainer.n_procs = self._n_procs
        cfg.agent.random_seeds = {
            "mcts": seed,
            "train": seed + 1,
            "eval": seed + 2,
            "external_policy": seed + 3,
        }
        use_mask = self._mask_mode == "precondition"
        cfg.game.kwargs["use_precondition_mask"] = use_mask

        game, net, agent, trainer, evaluator = cfg.build()
        gated = GatedTrainer(
            trainer=trainer,
            evaluator=evaluator,
            acceptance_threshold=self._accept_threshold,
        )

        total_iterations = total_steps  # for AlphaZero, total_steps = iterations
        cumulative_env_steps = 0
        cumulative_episodes = 0

        for iteration in range(1, total_iterations + 1):
            # Train one iteration
            score, accepted = gated.train_iteration()

            # Count env steps from this iteration's training data
            # trainer.all_training_examples[-1] is the latest iteration's data
            # Each element is a list of (state, (policy, reward)) tuples
            if trainer.all_training_examples:
                latest = trainer.all_training_examples[-1]
                iter_steps = sum(len(game_examples) for game_examples in latest)
                cumulative_env_steps += iter_steps
                cumulative_episodes += len(latest)

            # Evaluate at specified intervals
            if iteration % eval_interval == 0 or iteration == total_iterations:
                policy_fn = _make_nn_greedy_policy(agent)
                episodes = evaluate_policy(
                    policy_fn=policy_fn,
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
                    env_steps=cumulative_env_steps,
                    train_episodes=cumulative_episodes,
                    learner_updates=iteration,
                    eval_checkpoint_idx=iteration,
                    wall_clock_sec=time.time() - t0,
                    **agg,
                )
