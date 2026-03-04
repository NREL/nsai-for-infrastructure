from __future__ import annotations

from dataclasses import dataclass

from alphazeropp.core.config import (
    MetaConfig,
    GameConfig as CoreGameConfig,
    NetConfig,
    AgentConfig,
    TrainerConfig,
    EvaluatorConfig,
    RunConfig,
)
from alphazeropp.synthesis.derivation_game import DerivationGame
from alphazeropp.synthesis.derivation_network import (
    DerivationPolicyValueNet,
)
from alphazeropp.instances.bitstring.dsl.scan_derivation_game import (
    ScanDerivationGame,
)
from alphazeropp.instances.bitstring.dsl.scan_network import ScanPolicyValueNet
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig as DSLGameConfig,
    all_initial_states,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator


@dataclass
class DerivationConfig(MetaConfig):
    """Configuration for DerivationGame AlphaZero training."""

    def __init__(self):
        super().__init__()
        self.game = CoreGameConfig(
            game_cls=DerivationGame,
            kwargs={
                "budget": 14,
                "n_sites": 6,
                "program_budget_mode": "exact",
                # LeafEvaluator sub-config (extracted in build())
                "n_ones": 2,
                "n_frozen_states": 1,
                "bit_flip": True,
                "sparse_reward": False,
                "potential_name": "onemax",
                "metric": "avg_reward",
                "penalty_lambda": 0.1,
                "blend_alpha": 0.5,
            },
        )
        self.net = NetConfig(
            net_cls=DerivationPolicyValueNet,
            kwargs={
                "budget": 14,
                "n_sites": 6,
                # action_size computed in build()
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.1,
                "training_params": {
                    "epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "policy_weight": 2.0,
                },
            },
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 200,
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
            n_games_per_train=40,
            n_past_iterations_to_train=20,
            n_procs=-1,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=-1,
        )
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.55,
            plot_every=5,
            plot_path="derivation_training_metrics.png",
        )

    def build(self):
        """Build DerivationGame, network, agent, trainer, and evaluator.

        More complex than BitStringConfig.build() because DerivationGame
        requires a LeafEvaluator, which itself requires a DSL GameConfig
        and frozen initial states.
        """
        gk = dict(self.game.kwargs)
        budget = gk["budget"]
        n_sites = gk["n_sites"]
        n_ones = gk["n_ones"]

        # 1. DSL GameConfig -> frozen states -> LeafEvaluator
        dsl_cfg = DSLGameConfig(
            bit_flip=gk["bit_flip"],
            sparse_reward=gk["sparse_reward"],
            n_ones=n_ones,
            potential_name=gk["potential_name"],
        )
        frozen_states = all_initial_states(n_sites, n_ones)
        n_frozen = gk.get("n_frozen_states", len(frozen_states))
        frozen_states = frozen_states[:n_frozen]
        leaf_eval = LeafEvaluator(
            n_sites,
            frozen_states,
            dsl_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
        )

        # 2. DerivationGame
        mode = gk.get("program_budget_mode", "exact")
        game = DerivationGame(
            budget, n_sites, leaf_eval, program_budget_mode=mode,
        )

        # 3. Network (action_size derived from game)
        action_size = game.action_space.n
        net_kwargs = dict(self.net.kwargs)
        net_kwargs["budget"] = budget
        net_kwargs["n_sites"] = n_sites
        net_kwargs["action_size"] = action_size
        net = DerivationPolicyValueNet(**net_kwargs)

        # 4. Agent, Trainer, Evaluator
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


@dataclass
class ScanDerivationConfig(MetaConfig):
    """Configuration for ScanDerivationGame AlphaZero training.

    Uses the priority-scan grammar: each derivation step picks the next
    bit index in a scan order. The search space is N! permutations
    instead of the CFG grammar's millions of AST programs.
    """

    def __init__(self):
        super().__init__()
        self.game = CoreGameConfig(
            game_cls=ScanDerivationGame,
            kwargs={
                "n_sites": 6,
                # LeafEvaluator sub-config (extracted in build())
                "n_ones": 2,
                "n_frozen_states": 1,
                "bit_flip": True,
                "sparse_reward": False,
                "potential_name": "onemax",
                "metric": "avg_reward",
                "penalty_lambda": 0.1,
                "blend_alpha": 0.5,
            },
        )
        self.net = NetConfig(
            net_cls=ScanPolicyValueNet,
            kwargs={
                "n_sites": 6,
                "d_hidden": 128,
                "n_hidden_layers": 2,
                "training_params": {
                    "epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "policy_weight": 2.0,
                },
            },
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 50,
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
            n_games_per_train=40,
            n_past_iterations_to_train=20,
            n_procs=-1,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=-1,
        )
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.55,
            plot_every=5,
            plot_path="scan_training_metrics.png",
        )

    def build(self):
        """Build ScanDerivationGame, network, agent, trainer, and evaluator.

        Reuses the same LeafEvaluator construction as DerivationConfig,
        then instantiates the scan-specific game and network.
        """
        gk = dict(self.game.kwargs)
        n_sites = gk["n_sites"]
        n_ones = gk["n_ones"]

        # 1. DSL GameConfig -> frozen states -> LeafEvaluator
        dsl_cfg = DSLGameConfig(
            bit_flip=gk["bit_flip"],
            sparse_reward=gk["sparse_reward"],
            n_ones=n_ones,
            potential_name=gk["potential_name"],
        )
        frozen_states = all_initial_states(n_sites, n_ones)
        n_frozen = gk.get("n_frozen_states", len(frozen_states))
        frozen_states = frozen_states[:n_frozen]
        leaf_eval = LeafEvaluator(
            n_sites,
            frozen_states,
            dsl_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
        )

        # 2. ScanDerivationGame
        game = ScanDerivationGame(n_sites, leaf_eval)

        # 3. Network
        net_kwargs = dict(self.net.kwargs)
        net_kwargs["n_sites"] = n_sites
        net = ScanPolicyValueNet(**net_kwargs)

        # 4. Agent, Trainer, Evaluator
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
