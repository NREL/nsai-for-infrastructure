from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphazeropp.instances.bitstring.game import BitStringGame
from alphazeropp.instances.bitstring.network import BitStringPolicyValueNet
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
class BitStringConfig(MetaConfig):
    """Configuration for BitString AlphaZero training."""
    
    def __init__(self):
        super().__init__()
        # Set BitString-specific defaults
        self.game = GameConfig(
            game_cls=BitStringGame,
            kwargs={
                "n_sites": 10,
                "bit_flip": True,
                "sparse_reward": True,
            }
        )
        self.net = NetConfig(
            net_cls= BitStringPolicyValueNet,
            kwargs={"n_sites": 10}
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 120,
                "temperature": 1.0,
                "c_exploration": 1.5,
                "dirichlet_alpha": 0.3,
                "dirichlet_epsilon": 0.25,
            },
            reward_discount=1.0,
            random_seeds={
                "mcts": 46,
                "train": 49,
                "eval": 52,
                "external_policy": 68,
            }
        )
        self.trainer = TrainerConfig(
            n_games_per_train=100,
            n_past_iterations_to_train=5,
            n_procs=8,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=8,
        )
        self.run = RunConfig(
            n_iterations=10,
            accept_threshold=0.55,
            plot_every=5,
            plot_path="bitstring_training_metrics.png",
        )

    def build(self):
        """Build BitString game, network, agent, trainer, and evaluator."""

        game = BitStringGame(**self.game.kwargs)
        net = BitStringPolicyValueNet(**self.net.kwargs)
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
