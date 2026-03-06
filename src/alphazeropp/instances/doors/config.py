"""DoorsDirectConfig: MetaConfig for direct-play AlphaZero on Doors."""

from __future__ import annotations

from dataclasses import dataclass

from alphazeropp.instances.doors.game import DoorsDirectGame
from alphazeropp.instances.doors.network import DoorsDirectNet
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

def compute_dims(num_rooms, locs_per_room=2):
    """Compute obs_size and horizon from difficulty parameters.

    obs_size = M + 2*D - 1  where M = num_rooms * locs_per_room.
    horizon  = max(15, 5 * optimal_steps).
    """
    M = num_rooms * locs_per_room
    obs_size = M + 2 * num_rooms - 1
    optimal_steps = 2 * (num_rooms - 1) + 1
    horizon = max(15, optimal_steps * 5)
    return {"obs_size": obs_size, "horizon": horizon, "M": M}


@dataclass
class DoorsDirectConfig(MetaConfig):
    """Configuration for direct-play AlphaZero on Doors.

    Hyperparameters matched to DoorsDerivationConfig for fair comparison.
    """

    def __init__(self, num_rooms=10, locs_per_room=3):
        super().__init__()
        dims = compute_dims(num_rooms, locs_per_room)
        obs_size = dims["obs_size"]

        self.game = GameConfig(
            game_cls=DoorsDirectGame,
            kwargs={
                "num_rooms": num_rooms,
                "locs_per_room": locs_per_room,
                "step_penalty": 0.01,
                "unlock_bonus": 0.1,
            },
        )
        self.net = NetConfig(
            net_cls=DoorsDirectNet,
            kwargs={
                "input_size": obs_size,
                "output_size": obs_size,
                "n_hidden_layers": 1,
                "hidden_size": max(64, obs_size * 4),
            },
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 120,
                "temperature": 1.0,
                "c_exploration": 1.5,
                "dirichlet_alpha": 0.25,
                "dirichlet_epsilon": 0.40,
            },
            reward_discount=1.0,
            random_seeds={
                "mcts": 43,
                "train": 47,
                "eval": 23,
                "external_policy": 68,
            },
        )
        self.trainer = TrainerConfig(
            n_games_per_train=50,
            n_past_iterations_to_train=20,
            n_procs=8,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=8,
        )
        self.run = RunConfig(
            n_iterations=50,
            accept_threshold=0.0,
            plot_every=5,
            plot_path="doors_direct_training_metrics.png",
        )

    def build(self):
        """Build game, network, agent, trainer, and evaluator."""
        game = DoorsDirectGame(**self.game.kwargs)
        net = DoorsDirectNet(**self.net.kwargs)
        agent = Agent(
            game=game,
            net=net,
            mcts_params=self.agent.mcts_params,
            reward_discount=self.agent.reward_discount,
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
