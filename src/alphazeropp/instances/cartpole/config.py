from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphazeropp.instances.cartpole.game import CartPoleGame
from alphazeropp.instances.cartpole.network import CartPolePolicyValueNet
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator

from alphazeropp.core.config import (
    MetaConfig,
    GameConfig,
    NetConfig,
    AgentConfig,
    TrainerConfig,
    EvaluatorConfig,
    RunConfig,
)


@dataclass
class CartPoleConfig(MetaConfig):
    """Configuration for CartPole AlphaZero training."""
    
    def __init__(self):
        super().__init__()
        # Set CartPole-specific defaults
        self.game = GameConfig(
            game_cls=CartPoleGame,
            kwargs={}
        )
        self.net = NetConfig(
            net_cls=CartPolePolicyValueNet,
            kwargs={}
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 25,
                "temperature": 1.0,
                "c_exploration": 1.0,
            },
            reward_discount=1.0,
            random_seeds={
                "mcts": 0,
                "train": 1,
                "eval": 2,
                "external_policy": 3,
            }
        )
        self.trainer = TrainerConfig(
            n_games_per_train=100,
            n_past_iterations_to_train=20,
            n_procs=-1,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=-1,
        )
        self.run = RunConfig(
            n_iterations=20,
            accept_threshold=0.55,
            plot_every=5,
            plot_path="cartpole_training_metrics.png",
        )

    def build(self):
        """Build CartPole game, network, agent, trainer, and evaluator."""

        game = CartPoleGame(**self.game.kwargs)
        net = CartPolePolicyValueNet(**self.net.kwargs)
        agent = Agent(
            game=game,
            net=net,
            mcts_params=self.agent.mcts_params,
            reward_discount=self.agent.reward_discount,
            external_policy=self.agent.external_policy,
            random_seeds=self.agent.random_seeds,
        )
        trainer = Trainer(
            agent=agent,
            net=net,
            game=game,
            n_games_per_train=self.trainer.n_games_per_train,
            n_past_iterations_to_train=self.trainer.n_past_iterations_to_train,
            n_procs=self.trainer.n_procs,
            checkpoint_dir=self.trainer.checkpoint_dir,
        )
        evaluator = Evaluator(
            n_games=self.evaluator.n_games,
            n_procs=self.evaluator.n_procs,
        )
        return game, net, agent, trainer, evaluator
