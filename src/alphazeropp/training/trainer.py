# src/alphazeropp/training/trainer.py

import time
import logging
from pathlib import Path
from typing import Optional
import itertools
import copy

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.core.policy_value_net import PolicyValueNet
from alphazeropp.core.game import Game
from alphazeropp.utils.multiprocessing import MultiprocessingManager
from alphazeropp.utils.checkpoint import CheckpointManager

from functools import partial


logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the training loop for AlphaZero."""
    
    def __init__(self, 
                 agent: Agent,
                 net: PolicyValueNet,
                 game: Game,
                 n_games_per_train: int = 100,
                 n_games_per_eval: int = 20,
                 n_past_iterations_to_train: Optional[int] = 20,
                 threshold_to_keep: float = 0.55,
                 n_procs: Optional[int] = None,
                 checkpoint_dir: str | Path = "checkpoints"):
        """
        Initialize the Trainer.
        
        Args:
            agent: Agent instance for playing games
            net: PolicyValueNet instance
            n_games_per_train: Number of games per training iteration
            n_games_per_eval: Number of games for evaluation
            n_past_iterations_to_train: How many past iterations to keep for training
            threshold_to_keep: Win rate threshold to keep new network
            n_procs: Number of processes for multiprocessing
            checkpoint_dir: Directory to save checkpoints
        """
        self.agent = agent
        self.net = net
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_games_per_eval = n_games_per_eval
        self.n_past_iterations_to_train = n_past_iterations_to_train
        self.threshold_to_keep = threshold_to_keep
        self.n_procs = n_procs
        
        self.all_training_examples = []
        self.run_start_time = int(time.time())
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        logger.info(f"Trainer initialized: n_games_per_train={n_games_per_train}, "
                   f"n_games_per_eval={n_games_per_eval}, "
                   f"threshold_to_keep={threshold_to_keep}")
    
    def _collect_training_examples(self) -> list:
        """
        Collect training examples by playing games.
        
        Returns:
            List of training example sets
        """
        logger.info(f"Collecting {self.n_games_per_train} training games...")
        
        mp_manager = MultiprocessingManager(self.agent)
        mp_manager.push()
        multiprocessing_function = partial(
            self.agent.play_for_experience,
            self.game
        )
        
        try:
            arg_tuples = [
                (i, self.agent._randseed("train"), self.agent._randseed("mcts"))
                for i in range(self.n_games_per_train)
            ]
            train_example_sets = MultiprocessingManager.starmap(
                multiprocessing_function, arg_tuples, self.n_procs
            )
        finally:
            mp_manager.pop()
        
        breakpoint() # Check what is being returned here.
        return train_example_sets
    
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
        
        flat_examples = list(itertools.chain.from_iterable(self.all_training_examples))
        
        logger.info(f"Training examples: {[len(x) for x in self.all_training_examples]}")
        logger.info(f"Total examples: {len(flat_examples)}, Total value: {sum(x[1][1] for x in flat_examples):.2f}")
        
        return flat_examples
    
    def _train_network(self, flat_examples: list):
        """
        Train the network on collected examples.
        
        Args:
            flat_examples: Flattened list of (state, (policy, reward)) tuples
        """
        logger.info(f"Training on {len(flat_examples)} examples...")
        start_time = time.time()
        
        self.net.train(flat_examples)
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds")
    
    def train_iteration(self) -> bool:
        """
        Run a single training iteration: collect examples, train, and evaluate.
        
        Returns:
            True if new network is kept, False if reverted to old network
        """
        logger.info("Starting training iteration...")
        start_time = time.time()
        
        # Step 1: Collect training examples
        train_example_sets = self._collect_training_examples()
        new_train_examples = []
        for train_examples in train_example_sets:
            new_train_examples.extend(train_examples)
        
        # Step 2: Process training examples
        flat_examples = self._process_training_examples(new_train_examples)
        
        # Step 3: Save old network for comparison
        self.game.reset_wrapper()
        agent_before_training = copy.deepcopy(self.agent)
        
        # Step 4: Train network
        self._train_network(flat_examples)
        
        # Step 5: Evaluate new network
        from alphazeropp.training.evaluator import Evaluator
        evaluator = Evaluator(
            n_games=self.n_games_per_eval,
            n_procs=self.n_procs
        )
        
        score = evaluator.pit(
            self.agent, agent_before_training,
            self.agent._randseed("eval"),
            self.agent._randseed("mcts"),
            self.agent._randseed("external_policy")
        )
        
        # Step 6: Keep or revert network
        if score >= self.threshold_to_keep:
            logger.info(f"Keeping new network (score: {score:.2%})")
            keep_new = True
        else:
            logger.info(f"Reverting to old network (score: {score:.2%})")
            self.agent.net = agent_before_training.net
            keep_new = False
        
        elapsed = time.time() - start_time
        logger.info(f"Training iteration completed in {elapsed:.2f} seconds")
        
        return keep_new
    
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