# src/alphazeropp/training/trainer.py

import time
import logging
from pathlib import Path
from typing import Optional
import itertools

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.core.policy_value_net import PolicyValueNet
from alphazeropp.core.game import Game
from alphazeropp.utils.multiprocessing import MultiprocessingManager
from alphazeropp.utils.checkpoint import CheckpointManager
from alphazeropp.utils.statistics import StatisticsManager

from functools import partial
logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the training loop for AlphaZero."""
    
    def __init__(self, 
                 agent: Agent,
                 net: PolicyValueNet,
                 game: Game,
                 n_games_per_train: int = 100,
                 n_past_iterations_to_train: Optional[int] = 20,
                 n_procs: Optional[int] = None,
                 checkpoint_dir: str | Path = "checkpoints",
                 use_tree_reuse: bool = False):
        """
        Initialize the Trainer.

        Args:
            agent: Agent instance for playing games
            net: PolicyValueNet instance
            n_games_per_train: Number of games per training iteration
            n_past_iterations_to_train: How many past iterations to keep for training
            n_procs: Number of processes for multiprocessing
            checkpoint_dir: Directory to save checkpoints
            use_tree_reuse: If True, use MCTS tree reuse across moves within each episode
        """
        self.agent = agent
        self.net = net
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_past_iterations_to_train = n_past_iterations_to_train
        self.n_procs = n_procs
        self.use_tree_reuse = use_tree_reuse

        self.all_training_examples = []
        self._last_diagnostics: list = []
        self.run_start_time = int(time.time())
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.statistics_manager = StatisticsManager(self.run_start_time)
        
        logger.debug(f"Trainer initialized: n_games_per_train={n_games_per_train}, "
                    f"n_past_iterations_to_train={n_past_iterations_to_train}")
        
    def push_multiprocessing(self):
        ### It looks like we don't need to do anything here yet
        pass
    
    def pop_multiprocessing(self, *args):
        ### It looks like we don't need to do anything here yet
        pass
    
    
    def _collect_training_examples(self) -> list:
        """
        Collect training examples by playing games.
        
        Returns:
            List of training example sets
        """
        logger.debug(f"Collecting {self.n_games_per_train} training games...")
        
        # Before we start multiprocessing, we need to push the multiprocessing state to all relevant objects (like the agent and the game) to make sure they are in a consistent state across processes. After multiprocessing is done, we need to pop the multiprocessing state to restore the original state.
        mp_manager = MultiprocessingManager(self.agent.net, self)
        mp_manager.push()
        play_fn = (self.agent.play_for_experience_reuse_tree
                   if self.use_tree_reuse
                   else self.agent.play_for_experience)
        multiprocessing_function = partial(play_fn, self.game)
        
        try:
            arg_tuples = [
                (i, self.agent._randseed("train"), self.agent._randseed("mcts"))
                for i in range(self.n_games_per_train)
            ]
            if self.n_procs is not None and self.n_procs < 0:
                # Sequential mode: run with per-game progress logging
                train_example_sets = []
                for j, args in enumerate(arg_tuples):
                    t0 = time.time()
                    result = multiprocessing_function(*args)
                    elapsed = time.time() - t0
                    n_steps = len(result[0])
                    reward = result[1]
                    logger.info(
                        "[TRAIN] Game %d/%d complete (%d steps, %.1fs, reward=%.4f)",
                        j + 1, self.n_games_per_train, n_steps, elapsed, reward,
                    )
                    train_example_sets.append(result)
            else:
                train_example_sets = MultiprocessingManager.starmap(
                    multiprocessing_function, arg_tuples, self.n_procs
                )
        finally:
            mp_manager.pop()
        return train_example_sets # This should be a list of lists of examples, where each inner list corresponds to one game.
    
    def _process_training_examples(self, new_train_examples: list) -> list:
        """
        Process and accumulate training examples.
        
        Args:
            new_train_examples: New examples from this iteration
        
        Returns:
            Flattened list of all examples to train on
        """
        self.all_training_examples.append(new_train_examples)
        
        # Keep only recent iterations
        if self.n_past_iterations_to_train is not None and \
            len(self.all_training_examples) > self.n_past_iterations_to_train:
            self.all_training_examples.pop(0)

        # Flatten list of lists into a single training set.
        flat_examples_over_iteration = list(itertools.chain.from_iterable(self.all_training_examples))
        flat_examples = list(itertools.chain.from_iterable(flat_examples_over_iteration))
        logger.debug(f"Training examples: {len(flat_examples)} total from {len(self.all_training_examples)} recent iterations")
        logger.debug(f"Examples per iteration: {[len(x) for x in self.all_training_examples]}")
        logger.debug(f"Total value: {sum(x[2] for x in flat_examples):.2f}")
        
        return flat_examples
    
    def _train_network(self, flat_examples: list, avg_reward: float = 0.0):
        """
        Train the network on collected examples.

        Args:
            flat_examples: Flattened list of (state, (policy, reward)) tuples
            avg_reward: Average cumulative reward from self-play games
        """
        start_time = time.time()

        _, train_batch_losses, train_losses, policy_losses, value_losses = self.net.train(flat_examples)

        elapsed = time.time() - start_time
        logger.debug(f"Training completed in {elapsed:.2f} seconds")
        statistics = {
            "train_loss": np.mean(train_losses),
            "train_loss_policy": np.mean(policy_losses),
            "train_loss_value": np.mean(value_losses),
            "num_examples": len(flat_examples),
            "avg_reward": avg_reward,
        }
        self.statistics_manager.record(statistics)
    
    def train_iteration(self) -> None:
        """
        Run a single training iteration: collect examples and train.
        """
        start_time = time.time()

        # Step 1: Collect training examples
        train_example_sets = self._collect_training_examples()

        experience = []
        rewards = []
        all_diagnostics = []
        all_leaf_eval_data = []
        for example_set in train_example_sets:
            experience.append(example_set[0])
            rewards.append(example_set[1])
            # Step infos may be present as 3rd element (from diagnostics)
            if len(example_set) > 2:
                all_diagnostics.append(example_set[2])
            # Leaf evaluator caches may be present as 4th element
            if len(example_set) > 3 and example_set[3] is not None:
                all_leaf_eval_data.append(example_set[3])

        self._last_diagnostics = all_diagnostics

        # Merge leaf evaluator caches from workers back into main process
        leaf_eval = getattr(self.game, 'leaf_evaluator', None)
        if leaf_eval is not None and hasattr(leaf_eval, 'merge_caches'):
            for data in all_leaf_eval_data:
                leaf_eval.merge_caches(data)

        avg_reward = float(np.mean(rewards)) if rewards else 0.0

        # Step 2: Process training examples
        flat_examples = self._process_training_examples(experience)

        # Step 3: Train network
        self._train_network(flat_examples, avg_reward=avg_reward)

        elapsed = time.time() - start_time
        logger.debug(f"Training iteration completed in {elapsed:.2f} seconds")
    
    def train_multiple(self, n_iterations: int, start_at: int = 0, 
                      checkpoint_every: Optional[int] = None):
        """
        Run multiple training iterations with optional checkpointing.
        
        Args:
            n_iterations: Total number of iterations to train
            start_at: Starting iteration number (for resuming)
            checkpoint_every: Save checkpoint every N iterations (None = no checkpointing)
        """
        logger.info(f"Starting training: {n_iterations} iterations, "
                   f"starting from iteration {start_at}")
        
        for i in range(start_at, n_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {i+1}/{n_iterations}")
            logger.info(f"{'='*60}")
            
            self.train_iteration()
            
            # Checkpoint if requested
            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                checkpoint_path = Path("checkpoints") / f"{self.run_start_time}_iter_{i+1}"
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                self.checkpoint_manager.save_checkpoint(checkpoint_path, self.agent, self.net)
        
        logger.info(f"Training completed: {n_iterations} iterations")
    
    def save_checkpoint(self, checkpoint_path: str | Path = None):
        """Save current training state."""
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints") / f"{self.run_start_time}_final"
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        self.checkpoint_manager.save_checkpoint(checkpoint_path, self.agent, self.net)
    
    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load training state from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint_manager.load_checkpoint(checkpoint_path, self.agent, self.net)