"""
Doors Derivation -- AlphaZero Program Synthesis Training.

Synthesizes Doors navigation policies via grammar-guided MCTS + AlphaZero.
Modes:
  - doors:          And enabled (required for correct PICK preconditions)
  - doors_no_and:   And disabled (expressivity gap baseline)
  - doors_factored: Factored action space (structure/parameter split)
  - doors_d10_macro: D=10 with PickRule/MoveRule macros + condition budget cap

Usage:
    python scripts/run_doors_derivation.py
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import json
import logging
from datetime import datetime
from pathlib import Path

from alphazeropp.instances.doors.dsl.derivation_config import (
    DoorsDerivationConfig, DoorsDerivationConfigNoAnd,
    DoorsFactoredDerivationConfig, DoorsFactoredD10MacroConfig,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    compute_doors_derived_params,
)
from alphazeropp.synthesis.derivation_game import compute_max_productions
from alphazeropp.synthesis.factored_derivation_game import compute_max_factored_actions
from alphazeropp.synthesis.budget_grammar import count_programs
from alphazeropp.synthesis.leaf_evaluator import VALID_METRICS
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)
from alphazeropp.utils.derivation_utils import run_derivation_training


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_sections(cfg):
    """Return sections for Doors DerivationGame config.

    Changing num_rooms or locs_per_room auto-updates n_sites, budget,
    and horizon via compute_doors_derived_params().
    """
    def _update_derived():
        nr = cfg.game.kwargs["num_rooms"]
        lpr = cfg.game.kwargs.get("locs_per_room", 2)
        derived = compute_doors_derived_params(nr, lpr)
        cfg.game.kwargs["n_sites"] = derived["n_sites"]
        cfg.game.kwargs["budget"] = derived["budget"]
        cfg.game.kwargs["horizon"] = derived["horizon"]
        cfg.net.kwargs["n_sites"] = derived["n_sites"]
        cfg.net.kwargs["budget"] = derived["budget"]

    def _set_num_rooms(val):
        cfg.game.kwargs["num_rooms"] = int(val)
        _update_derived()

    def _set_locs_per_room(val):
        cfg.game.kwargs["locs_per_room"] = int(val)
        _update_derived()

    def _set_budget(val):
        cfg.game.kwargs["budget"] = val
        cfg.net.kwargs["budget"] = val

    params = [
        # Problem -- primary controls
        ("num_rooms", cfg.game.kwargs["num_rooms"], _set_num_rooms,
         "Number of rooms (D) -- auto-updates n_sites/budget/horizon", None),
        ("locs_per_room", cfg.game.kwargs.get("locs_per_room", 2),
         _set_locs_per_room,
         "Locations per room", None),
        # Derived (editable for manual override)
        ("budget", cfg.game.kwargs["budget"], _set_budget,
         "AST node budget (auto: ~1.5x optimal)", None),
        ("horizon", cfg.game.kwargs["horizon"],
         dict_setter(cfg.game.kwargs, "horizon"),
         "Max steps per episode (auto: 5x optimal)", None),
        ("budget_mode", cfg.game.kwargs.get("program_budget_mode", "max"),
         dict_setter(cfg.game.kwargs, "program_budget_mode"),
         "Budget mode: exact (==L) or max (<=L)", ["exact", "max"]),
        ("allow_and", cfg.game.kwargs.get("allow_and", True),
         dict_setter(cfg.game.kwargs, "allow_and"),
         "Allow And() in grammar conditions", None),
        # Leaf evaluation
        ("metric", cfg.game.kwargs["metric"], dict_setter(cfg.game.kwargs, "metric"),
         "Leaf evaluation metric", list(VALID_METRICS)),
    ]
    if cfg.game.kwargs["metric"] == "penalized_reward":
        params.append(
            ("penalty_lambda", cfg.game.kwargs["penalty_lambda"],
             dict_setter(cfg.game.kwargs, "penalty_lambda"),
             "Penalty weight for interp ops", None))
    if cfg.game.kwargs["metric"] == "weighted":
        params.append(
            ("blend_alpha", cfg.game.kwargs["blend_alpha"],
             dict_setter(cfg.game.kwargs, "blend_alpha"),
             "Weight of solve_rate in blend", None))

    # MCTS
    mcts_descs = {
        "n_simulations": "MCTS rollouts per derivation step",
        "temperature": "Exploration temperature for action selection",
        "c_exploration": "UCB exploration constant",
        "dirichlet_alpha": "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
        "rollout_n": "Random completions per MCTS leaf (0=disabled)",
        "rollout_mode": "Aggregation: mean or max",
        "rollout_blend": "Blend: (1-b)*rollout + b*nn_value",
        "rollout_budget": "Max total steps for rollouts per leaf",
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon",
              "rollout_n", "rollout_mode", "rollout_blend", "rollout_budget"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k], dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], None))

    # Network architecture
    params.extend([
        ("d_model", cfg.net.kwargs["d_model"], dict_setter(cfg.net.kwargs, "d_model"),
         "Transformer embedding dimension", None),
        ("n_heads", cfg.net.kwargs["n_heads"], dict_setter(cfg.net.kwargs, "n_heads"),
         "Transformer attention heads", None),
        ("n_layers", cfg.net.kwargs["n_layers"], dict_setter(cfg.net.kwargs, "n_layers"),
         "Transformer encoder layers", None),
        ("learning_rate", cfg.net.kwargs["training_params"]["learning_rate"],
         dict_setter(cfg.net.kwargs["training_params"], "learning_rate"),
         "Adam learning rate", None),
        ("batch_size", cfg.net.kwargs["training_params"]["batch_size"],
         dict_setter(cfg.net.kwargs["training_params"], "batch_size"),
         "Training batch size", None),
    ])

    # Agent / Trainer / Evaluator / Run
    params.extend([
        ("reward_discount", cfg.agent.reward_discount,
         attr_setter(cfg.agent, "reward_discount"),
         "Discount factor for future rewards", None),
        ("n_games_per_train", cfg.trainer.n_games_per_train,
         attr_setter(cfg.trainer, "n_games_per_train"),
         "Self-play games per training iteration", None),
        ("n_past_iters", cfg.trainer.n_past_iterations_to_train,
         attr_setter(cfg.trainer, "n_past_iterations_to_train"),
         "Past iterations kept in training buffer", None),
        ("n_procs", cfg.trainer.n_procs,
         attr_setter(cfg.trainer, "n_procs"),
         "Parallel workers for self-play (-1=sequential)", None),
        ("eval_n_games", cfg.evaluator.n_games,
         attr_setter(cfg.evaluator, "n_games"),
         "Games to pit new vs old agent", None),
        ("eval_n_procs", cfg.evaluator.n_procs,
         attr_setter(cfg.evaluator, "n_procs"),
         "Parallel workers for evaluation (-1=sequential)", None),
        ("n_iterations", cfg.run.n_iterations,
         attr_setter(cfg.run, "n_iterations"),
         "Total training iterations", None),
        ("accept_threshold", cfg.run.accept_threshold,
         attr_setter(cfg.run, "accept_threshold"),
         "Win rate to accept new network", None),
        ("plot_every", cfg.run.plot_every,
         attr_setter(cfg.run, "plot_every"),
         "Plot metrics every N iterations", None),
    ])

    all_params = build_param_list(params)

    problem_labels = {"num_rooms", "locs_per_room", "budget",
                      "horizon", "budget_mode", "allow_and"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon",
                   "rollout_n", "rollout_mode", "rollout_blend",
                   "rollout_budget"}
    net_labels = {"d_model", "n_heads", "n_layers", "learning_rate", "batch_size"}
    agent_labels = {"reward_discount"}
    trainer_labels = {"n_games_per_train", "n_past_iters", "n_procs"}
    evaluator_labels = {"eval_n_games", "eval_n_procs"}
    run_labels = {"n_iterations", "accept_threshold", "plot_every"}

    return [
        ("Problem",    [p for p in all_params if p[1] in problem_labels]),
        ("Leaf Eval",  [p for p in all_params if p[1] in eval_labels]),
        ("MCTS",       [p for p in all_params if p[1] in mcts_labels]),
        ("Network",    [p for p in all_params if p[1] in net_labels]),
        ("Agent",      [p for p in all_params if p[1] in agent_labels]),
        ("Trainer",    [p for p in all_params if p[1] in trainer_labels]),
        ("Evaluator",  [p for p in all_params if p[1] in evaluator_labels]),
        ("Run",        [p for p in all_params if p[1] in run_labels]),
    ]


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def setup_experiment_dir(cfg, mode="doors"):
    """Create and return an experiment directory path. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_sites = cfg.game.kwargs["n_sites"]
    budget = cfg.game.kwargs["budget"]
    bmode = cfg.game.kwargs.get("program_budget_mode", "max")
    metric = cfg.game.kwargs["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations
    num_rooms = cfg.game.kwargs["num_rooms"]
    and_tag = "and" if cfg.game.kwargs.get("allow_and", True) else "noand"

    # Derive game_type tag from mode
    if mode == "doors_d10_macro":
        game_tag = "factored_macro"
    elif mode == "doors_factored":
        game_tag = "factored"
    else:
        game_tag = "flat"

    dirname = (f"{timestamp}_D{num_rooms}_{and_tag}_{game_tag}_N{n_sites}_L{budget}"
               f"_{bmode}_{metric}_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "doors_derivation" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Training output helpers
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir, mode="doors"):
    """Print startup banner with experiment info."""
    gk = cfg.game.kwargs
    n_sites = gk["n_sites"]
    num_rooms = gk["num_rooms"]
    lpr = gk.get("locs_per_room", 2)
    budget = gk["budget"]
    horizon = gk["horizon"]
    allow_and = gk.get("allow_and", True)
    metric = gk["metric"]
    derived = compute_doors_derived_params(num_rooms, lpr)

    print()
    print("=" * 80)
    print("  DOORS PROGRAM SYNTHESIS via AlphaZero")
    print("=" * 80)
    print()
    print("  PROBLEM")
    print(f"    Doors PDDL environment: D={num_rooms} rooms, "
          f"{lpr} locs/room")
    print(f"    M={derived['M']} locations, K={derived['K']} keys, "
          f"obs_size={n_sites}")
    print(f"    Synthesize a policy program that navigates rooms and picks up keys.")
    print(f"    Horizon (max steps): {horizon} "
          f"(optimal: {derived['optimal_steps']} steps)")
    print(f"    Allow And:           {allow_and}")
    print()
    M = derived["M"]
    K = derived["K"]
    key_loc = [k * lpr + 1 for k in range(num_rooms - 1)]
    print("  ROOM LAYOUT & KEYS")
    for r in range(num_rooms):
        locs = list(range(r * lpr, (r + 1) * lpr))
        keys_here = [k for k, kl in enumerate(key_loc) if kl // lpr == r]
        room_desc = f"    Room {r}: locations {locs}"
        if keys_here:
            for k in keys_here:
                room_desc += f"  [Key {k} at loc {key_loc[k]} -> unlocks Room {k+1}]"
        if r == num_rooms - 1:
            room_desc += "  [GOAL]"
        print(room_desc)
    print()
    print(f"  OBSERVATION VECTOR (size {n_sites})")
    print(f"    Indices [0..{M-1}]:        Agent location (one-hot)")
    print(f"    Indices [{M}..{M+num_rooms-1}]:      Room unlock status (1=unlocked)")
    print(f"    Indices [{M+num_rooms}..{M+num_rooms+K-1}]:      Key availability (1=available)")
    print()
    print("  GRAMMAR GUIDE")
    print(f"    Flip(j) = test obs[j]==0    (negated predicate)")
    print(f"    Ite(cond, act, else)        if cond then act else recurse")
    print(f"    IsZero(j):  j<{M} -> 'not at loc j'  |  j>={M} -> 'room/key status'")
    print(f"    Actions:    Flip(0..{M-1}) = MOVE(loc)  |  Flip({M}..{M+K-1}) = PICK(key)  |  Flip({M+K}) = NOOP")
    print()
    bmode = gk.get("program_budget_mode", "max")
    n_actions = M + K + 1
    max_prods = compute_max_productions(budget, n_sites, mode=bmode,
                                         allow_and=allow_and,
                                         n_actions=n_actions)
    total_programs = count_programs(n_sites, budget)
    is_factored = mode in ("doors_factored", "doors_d10_macro")
    game_type = "Factored+Macro" if mode == "doors_d10_macro" else (
        "Factored" if mode == "doors_factored" else "Flat")
    print(f"  GRAMMAR: CFG ({'And enabled' if allow_and else 'And disabled'})")
    print(f"    AST node budget:    L = {budget} "
          f"(optimal program ~{derived['optimal_nodes']} nodes, "
          f"50% headroom)")
    print(f"    Budget mode:        {bmode}")
    print(f"    Possible programs:  {total_programs}")
    print(f"    Max productions:    {max_prods} (flat action space)")
    if is_factored:
        factored_kwargs = dict(
            budget=budget, n_sites=n_sites, mode=bmode,
            allow_and=allow_and, n_actions=n_actions,
        )
        if mode == "doors_d10_macro":
            from alphazeropp.instances.doors.dsl.doors_macros import make_macro_fn
            from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
            _dcfg = DoorsGameConfig(
                num_rooms=gk["num_rooms"],
                locs_per_room=gk.get("locs_per_room", 2),
            )
            factored_kwargs["macro_productions_fn"] = make_macro_fn(_dcfg)
            factored_kwargs["max_condition_budget"] = 12
        max_factored = compute_max_factored_actions(**factored_kwargs)
        print(f"    Factored actions:   {max_factored} "
              f"({max_prods / max_factored:.0f}× reduction)")
    print(f"    Game type:          {game_type}")
    print()
    print(f"  EVALUATION")
    print(f"    Metric:             {metric}")
    print(f"    Optimal reward:     ~1.0 (all keys collected, goal reached)")
    print()
    print(f"  Experiment dir: {exp_dir}/")
    print()
    print("  Output legend:")
    print("    [TRAIN]  Self-play data collection & network training")
    print("    [EVAL]   Pitting new network vs old network")
    print("    [ITER]   Iteration summary with key metrics")
    print("    [PROG]   Best program discovered so far")
    print("=" * 80)
    print()


def print_architecture(cfg, mode="doors"):
    """Print the AlphaZero training loop architecture diagram."""
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    budget = cfg.game.kwargs["budget"]
    d_model = cfg.net.kwargs.get("d_model", 64)
    net_desc = f"Transformer(d={d_model})"
    step_desc = f"expand DoorsPolicyHole({budget}) to a complete policy"

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero for Program Synthesis")
    print("=" * 80)
    print()
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  +----- Iteration i " + "-" * 53 + "+")
    print("  |                                                                      |")
    print(f"  |  STEP 1: Self-Play  ({n_games} derivation games)")
    print("  |")
    print(f"  |    Each game: {step_desc}")
    print(f"  |      At each step: MCTS(state, {net_desc}, n_sims={n_sims})")
    print(f"  |        Network reads partial AST state")
    print("  |        -> (pi: production probs, v: predicted program quality)")
    print("  |        MCTS refines pi using UCB + backed-up leaf evaluations")
    print("  |      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  |      SAVE (partial_ast, pi_MCTS) as training target")
    print("  |    Terminal: complete program -> leaf_eval(prog) -> scalar reward")
    print("  |")
    print(f"  |  STEP 2: Train Transformer  (replay buffer: last {n_past} iterations)")
    print("  |")
    print("  |    L = (z - v_theta(ast))^2  -  pi_MCTS * log p_theta(ast)")
    print("  |        value loss              policy loss")
    print("  |    theta <- theta - alpha * grad(L)")
    print("  |")
    print("  |  STEP 3: Evaluate")
    print("  |    Pit new_net vs old_net on fresh derivation games.")
    print("  |                                                                      |")
    print("  +" + "-" * 72 + "+")
    print()
    print("  Virtuous cycle:")
    print("    Better Transformer -> better MCTS -> better data -> better Transformer")
    print("    The network learns which partial ASTs are promising,")
    print("    directing search toward high-quality programs.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_mode():
    """Prompt user to select doors derivation mode."""
    print()
    print("=== Doors Derivation Mode ===")
    print()
    print("  0) doors          -- And enabled (default)")
    print("     PDDL-faithful rooms/keys; requires And for PICK preconditions")
    print()
    print("  1) doors_no_and   -- And disabled")
    print("     Baseline for expressivity gap measurement")
    print()
    print("  2) doors_factored -- Factored action space")
    print("     Splits productions into (structure, parameter) for lower branching")
    print()
    print("  3) doors_factored_macro -- Factored + macros")
    print("     Factored action space + PickRule/MoveRule macros + condition budget cap")
    print()
    choice = input("  Select mode [0]: ").strip()
    if choice in ("1", "doors_no_and"):
        return "doors_no_and"
    if choice in ("2", "doors_factored"):
        return "doors_factored"
    if choice in ("3", "doors_d10_macro", "doors_factored_macro"):
        return "doors_d10_macro"
    return "doors"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mode = select_mode()

    if mode == "doors_no_and":
        cfg = DoorsDerivationConfigNoAnd()
        interactive_edit("Doors DerivationGame Config (And disabled)",
                         lambda: _build_sections(cfg))
    elif mode == "doors_factored":
        cfg = DoorsFactoredDerivationConfig()
        interactive_edit("Doors FactoredDerivationGame Config",
                         lambda: _build_sections(cfg))
    elif mode == "doors_d10_macro":
        cfg = DoorsFactoredD10MacroConfig()
        interactive_edit("Doors D=10 Factored+Macro Config",
                         lambda: _build_sections(cfg))
    else:
        cfg = DoorsDerivationConfig()
        interactive_edit("Doors DerivationGame Config (And enabled)",
                         lambda: _build_sections(cfg))

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg, mode=mode)
    cfg.save(str(exp_dir / "config.json"))

    print_banner(cfg, exp_dir, mode=mode)
    print_architecture(cfg, mode=mode)

    num_rooms = cfg.game.kwargs["num_rooms"]
    step_penalty = cfg.game.kwargs.get("step_penalty", 0.01)
    unlock_bonus = cfg.game.kwargs.get("unlock_bonus", 0.1)
    optimal_steps = 2 * (num_rooms - 1) + 1
    optimal_reward = 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty

    run_derivation_training(cfg, mode, optimal_reward, exp_dir)


if __name__ == "__main__":
    main()
