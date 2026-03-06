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
from alphazeropp.synthesis.factored_derivation_game import (
    FactoredDerivationGame, compute_max_factored_actions,
)
from alphazeropp.instances.doors.dsl.doors_macros import make_macro_fn
from alphazeropp.synthesis.derivation_network import (
    DerivationPolicyValueNet,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator


class DoorsProgressFn:
    """Picklable callable that counts fraction of keys picked from final state.

    For Doors: keys are in obs[M + num_rooms : M + num_rooms + K].
    A key is "picked" when its availability goes to 0.0.
    Must be a class (not a closure) so multiprocessing can pickle it.
    """

    def __init__(self, doors_cfg: DoorsGameConfig):
        self.K = doors_cfg.K
        self.key_start = doors_cfg.M + doors_cfg.num_rooms

    def __call__(self, final_state):
        keys_picked = sum(
            1 for k in range(self.K)
            if final_state[self.key_start + k] == 0.0
        )
        return keys_picked / self.K if self.K > 0 else 1.0


@dataclass
class DoorsDerivationConfig(MetaConfig):
    """Configuration for DerivationGame on the DoorsPDDLLiteEnv.

    Uses the PDDL-faithful doors environment where PICK actions require
    conjunctive preconditions (And). n_sites, budget, and horizon are
    auto-derived from num_rooms and locs_per_room.
    """

    def __init__(self):
        super().__init__()
        num_rooms = 3
        locs_per_room = 2
        derived = compute_doors_derived_params(num_rooms, locs_per_room)

        self.game = CoreGameConfig(
            game_cls=DerivationGame,
            kwargs={
                "budget": derived["budget"],
                "n_sites": derived["n_sites"],
                "program_budget_mode": "max",
                "allow_and": True,
                "allow_not": True,
                # DoorsGameConfig sub-config (extracted in build())
                "num_rooms": num_rooms,
                "locs_per_room": locs_per_room,
                "horizon": derived["horizon"],
                "step_penalty": 0.01,
                "unlock_bonus": 1.0,
                # LeafEvaluator sub-config
                "metric": "weighted",
                "penalty_lambda": 0.1,
                "blend_alpha": 0.7,
            },
        )
        self.net = NetConfig(
            net_cls=DerivationPolicyValueNet,
            kwargs={
                "budget": derived["budget"],
                "n_sites": derived["n_sites"],
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
                "n_simulations": 80,
                "temperature": 1.0,
                "c_exploration": 1.5,
                "dirichlet_alpha": 0.25,
                "dirichlet_epsilon": 0.40,
                "rollout_n": 4,
                "rollout_mode": "max",
                "rollout_blend": 0.3,
                "rollout_budget": 200,
                "backup_rule": "max",
                "backup_topk": 3,
                "backup_tau": 0.1,
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
            n_games_per_train=30,
            n_past_iterations_to_train=20,
            n_procs=8,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=20,
            n_procs=8,
        )
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="doors_training_metrics.png",
        )

    def build(self):
        """Build DerivationGame for doors, network, agent, trainer, evaluator.

        Creates DoorsGameConfig -> canonical initial state -> LeafEvaluator
        (with is_solved callback) -> DerivationGame (with allow_and/not).
        """
        gk = dict(self.game.kwargs)
        budget = gk["budget"]
        n_sites = gk["n_sites"]

        # 1. DoorsGameConfig -> frozen state -> LeafEvaluator
        doors_cfg = DoorsGameConfig(
            num_rooms=gk["num_rooms"],
            locs_per_room=gk.get("locs_per_room", 2),
            horizon=gk["horizon"],
            step_penalty=gk["step_penalty"],
            unlock_bonus=gk["unlock_bonus"],
        )
        frozen_states = [doors_initial_state(doors_cfg)]
        progress_fn = DoorsProgressFn(doors_cfg)
        leaf_eval = LeafEvaluator(
            n_sites,
            frozen_states,
            doors_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
            is_solved=doors_cfg.is_solved,
            normalize_rewards=gk.get("normalize_rewards", False),
            progress_fn=progress_fn,
        )

        # 2. DerivationGame
        mode = gk.get("program_budget_mode", "max")
        allow_and = gk.get("allow_and", True)
        allow_not = gk.get("allow_not", True)
        n_actions = doors_cfg.M + doors_cfg.K + 1
        one_hot_groups = [list(range(doors_cfg.M))]
        game = DerivationGame(
            budget, n_sites, leaf_eval,
            program_budget_mode=mode,
            allow_and=allow_and,
            allow_not=allow_not,
            n_actions=n_actions,
            one_hot_groups=one_hot_groups,
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
            use_tree_reuse=True,
        )
        evaluator = Evaluator(
            n_games=self.evaluator.n_games,
            n_procs=self.evaluator.n_procs,
        )
        return game, net, agent, trainer, evaluator


@dataclass
class DoorsDerivationConfigNoAnd(DoorsDerivationConfig):
    """Doors config with And disabled — baseline for expressivity gap."""

    def __init__(self):
        super().__init__()
        self.game.kwargs["allow_and"] = False
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="doors_no_and_training_metrics.png",
        )


@dataclass
class DoorsFactoredDerivationConfig(DoorsDerivationConfig):
    """Doors config using FactoredDerivationGame (split structure/parameter).

    Reduces per-step branching from O(structures × params) to
    O(max(structures, params)) by factoring each production into
    a structure choice followed by an optional parameter choice.
    """

    def __init__(self):
        super().__init__()
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="doors_factored_training_metrics.png",
        )

    def build(self):
        """Build FactoredDerivationGame for doors."""
        gk = dict(self.game.kwargs)
        budget = gk["budget"]
        n_sites = gk["n_sites"]

        # 1. DoorsGameConfig -> frozen state -> LeafEvaluator
        doors_cfg = DoorsGameConfig(
            num_rooms=gk["num_rooms"],
            locs_per_room=gk.get("locs_per_room", 2),
            horizon=gk["horizon"],
            step_penalty=gk["step_penalty"],
            unlock_bonus=gk["unlock_bonus"],
        )
        frozen_states = [doors_initial_state(doors_cfg)]
        progress_fn = DoorsProgressFn(doors_cfg)
        leaf_eval = LeafEvaluator(
            n_sites,
            frozen_states,
            doors_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
            is_solved=doors_cfg.is_solved,
            normalize_rewards=gk.get("normalize_rewards", False),
            progress_fn=progress_fn,
        )

        # 2. FactoredDerivationGame
        mode = gk.get("program_budget_mode", "max")
        allow_and = gk.get("allow_and", True)
        allow_not = gk.get("allow_not", True)
        n_actions = doors_cfg.M + doors_cfg.K + 1
        one_hot_groups = [list(range(doors_cfg.M))]
        game = FactoredDerivationGame(
            budget, n_sites, leaf_eval,
            program_budget_mode=mode,
            allow_and=allow_and,
            allow_not=allow_not,
            n_actions=n_actions,
            one_hot_groups=one_hot_groups,
        )

        # 3. Network (action_size from factored game)
        action_size = game.action_space.n
        net_kwargs = dict(self.net.kwargs)
        net_kwargs["budget"] = budget
        net_kwargs["n_sites"] = n_sites
        net_kwargs["action_size"] = action_size
        net_kwargs["extra_features"] = 2  # phase_id + pending_structure_id
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
            use_tree_reuse=True,
        )
        evaluator = Evaluator(
            n_games=self.evaluator.n_games,
            n_procs=self.evaluator.n_procs,
        )
        return game, net, agent, trainer, evaluator


@dataclass
class DoorsFactoredD10MacroConfig(DoorsDerivationConfig):
    """D=10 Doors with FactoredDerivationGame + PickRule/MoveRule macros.

    Macros reduce derivation depth from ~119 to ~38 steps.
    Condition budget cap reduces structure templates from ~135 to ~15.
    Combined: max_factored ~ 39, suitable for MCTS with 400 sims.
    """

    def __init__(self):
        super().__init__()
        num_rooms = 3
        locs_per_room = 2
        derived = compute_doors_derived_params(num_rooms, locs_per_room)

        self.game.kwargs.update({
            "num_rooms": num_rooms,
            "locs_per_room": locs_per_room,
            "n_sites": derived["n_sites"],
            "budget": derived["budget"],
            "horizon": derived["horizon"],
        })
        self.net.kwargs.update({
            "budget": derived["budget"],
            "n_sites": derived["n_sites"],
        })
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 80,
                "temperature": 1.0,
                "c_exploration": 1.5,
                "dirichlet_alpha": 0.25,
                "dirichlet_epsilon": 0.40,
                "rollout_n": 4,
                "rollout_mode": "max",
                "rollout_blend": 0.3,
                "rollout_budget": 200,
                "backup_rule": "max",
                "backup_topk": 3,
                "backup_tau": 0.1,
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
            n_games_per_train=30,
            n_past_iterations_to_train=20,
            n_procs=8,
            checkpoint_dir="checkpoints",
        )
        self.run = RunConfig(
            n_iterations=30,
            accept_threshold=0.40,
            plot_every=5,
            plot_path="doors_d10_macro_training_metrics.png",
        )

    def build(self):
        """Build FactoredDerivationGame with macros for D=10 doors."""
        gk = dict(self.game.kwargs)
        budget = gk["budget"]
        n_sites = gk["n_sites"]

        # 1. DoorsGameConfig -> frozen state -> LeafEvaluator
        doors_cfg = DoorsGameConfig(
            num_rooms=gk["num_rooms"],
            locs_per_room=gk.get("locs_per_room", 2),
            horizon=gk["horizon"],
            step_penalty=gk["step_penalty"],
            unlock_bonus=gk["unlock_bonus"],
        )
        frozen_states = [doors_initial_state(doors_cfg)]
        progress_fn = DoorsProgressFn(doors_cfg)
        leaf_eval = LeafEvaluator(
            n_sites,
            frozen_states,
            doors_cfg,
            metric=gk["metric"],
            penalty_lambda=gk["penalty_lambda"],
            blend_alpha=gk["blend_alpha"],
            is_solved=doors_cfg.is_solved,
            normalize_rewards=gk.get("normalize_rewards", False),
            progress_fn=progress_fn,
        )

        # 2. FactoredDerivationGame with macros + condition budget cap
        mode = gk.get("program_budget_mode", "max")
        allow_and = gk.get("allow_and", True)
        allow_not = gk.get("allow_not", True)
        n_actions = doors_cfg.M + doors_cfg.K + 1
        one_hot_groups = [list(range(doors_cfg.M))]
        macro_fn = make_macro_fn(doors_cfg)
        max_condition_budget = 12

        game = FactoredDerivationGame(
            budget, n_sites, leaf_eval,
            program_budget_mode=mode,
            allow_and=allow_and,
            allow_not=allow_not,
            n_actions=n_actions,
            one_hot_groups=one_hot_groups,
            macro_productions_fn=macro_fn,
            max_condition_budget=max_condition_budget,
        )

        # 3. Network (action_size from factored game)
        action_size = game.action_space.n
        net_kwargs = dict(self.net.kwargs)
        net_kwargs["budget"] = budget
        net_kwargs["n_sites"] = n_sites
        net_kwargs["action_size"] = action_size
        net_kwargs["extra_features"] = 2  # phase_id + pending_structure_id
        net = DerivationPolicyValueNet(**net_kwargs)

        # 4. Agent, Trainer, Evaluator (tree reuse for D=10 efficiency)
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
            use_tree_reuse=True,
        )
        evaluator = Evaluator(
            n_games=self.evaluator.n_games,
            n_procs=self.evaluator.n_procs,
        )
        return game, net, agent, trainer, evaluator
