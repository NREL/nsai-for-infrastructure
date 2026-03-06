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

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

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

def _build_sections(cfg, seed_state=None):
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
             "Weight of solve_rate in blend (adaptive: uses raw reward when sr=0)", None))
    params.append(
        ("normalize_rewards", cfg.game.kwargs.get("normalize_rewards", False),
         dict_setter(cfg.game.kwargs, "normalize_rewards"),
         "Normalize leaf eval rewards via running EMA", None))

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
        "backup_rule": "Q-value backup: mean=avg all, max=best path, topk=avg top-k, softmax=smooth max",
        "backup_topk": "k for topk backup (ignored unless backup_rule=topk)",
        "backup_tau": "tau for softmax backup; tau->0 = max, tau->inf = mean (ignored unless softmax)",
    }
    mcts_choices = {
        "rollout_mode": ["mean", "max"],
        "backup_rule": ["mean", "max", "topk", "softmax"],
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon",
              "rollout_n", "rollout_mode", "rollout_blend", "rollout_budget",
              "backup_rule", "backup_topk", "backup_tau"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k], dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], mcts_choices.get(k)))

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

    if seed_state is not None:
        params.append(
            ("seeds", seed_state["seeds_str"],
             lambda val: seed_state.__setitem__("seeds_str", val),
             "Base seeds (comma-sep for multi-seed CI)", None))

    all_params = build_param_list(params)

    problem_labels = {"num_rooms", "locs_per_room", "budget",
                      "horizon", "budget_mode", "allow_and"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha", "normalize_rewards"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon",
                   "rollout_n", "rollout_mode", "rollout_blend",
                   "rollout_budget", "backup_rule", "backup_topk",
                   "backup_tau"}
    net_labels = {"d_model", "n_heads", "n_layers", "learning_rate", "batch_size"}
    agent_labels = {"reward_discount"}
    trainer_labels = {"n_games_per_train", "n_past_iters", "n_procs"}
    evaluator_labels = {"eval_n_games", "eval_n_procs"}
    run_labels = {"n_iterations", "accept_threshold", "plot_every", "seeds"}

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

    # MCTS backup strategy explanation
    backup_rule = cfg.agent.mcts_params.get("backup_rule", "mean")
    rollout_n = cfg.agent.mcts_params.get("rollout_n", 0)
    rollout_mode = cfg.agent.mcts_params.get("rollout_mode", "mean")
    rollout_blend = cfg.agent.mcts_params.get("rollout_blend", 0.0)
    print(f"  MCTS SEARCH STRATEGY")
    print(f"    Rollouts:           {rollout_n} random completions per leaf "
          f"(mode={rollout_mode}, blend={rollout_blend})")
    print(f"    Backup rule:        {backup_rule}")
    print()
    print("    How backup rules work (controls how Q-values propagate up the tree):")
    print("      During MCTS, each edge (state, action) accumulates reward values")
    print("      from simulations that passed through it. The backup rule determines")
    print("      how these values become the Q-value used in UCB action selection.")
    print()
    print("      mean    : Q = average of ALL values. Standard for two-player games.")
    print("                Risk: rare good paths drowned out by many bad ones.")
    print("      max     : Q = single best value seen. Optimistic: reflects 'a good")
    print("                path exists'. Best for synthesis (find ONE good program).")
    print("                Risk: a single lucky rollout can dominate early on.")
    print("      topk    : Q = average of top-k values. Compromise between mean and")
    print(f"                max. Currently k={cfg.agent.mcts_params.get('backup_topk', 3)}. "
          "Degenerates to mean when k >= visit count.")
    print("      softmax : Q = tau*log(mean(exp(v/tau))). Smooth approximation of max.")
    print(f"                Currently tau={cfg.agent.mcts_params.get('backup_tau', 0.1)}. "
          "tau->0 = max, tau->inf = mean.")
    print()
    print("    Note: Visit counts N(a) for UCB exploration are unchanged by backup rule.")
    print("    Only the exploitation term (normalized Q) changes.")
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


MODE_CONFIGS = {
    "doors": (DoorsDerivationConfig,
              "Doors DerivationGame Config (And enabled)"),
    "doors_no_and": (DoorsDerivationConfigNoAnd,
                     "Doors DerivationGame Config (And disabled)"),
    "doors_factored": (DoorsFactoredDerivationConfig,
                       "Doors FactoredDerivationGame Config"),
    "doors_d10_macro": (DoorsFactoredD10MacroConfig,
                        "Doors D=10 Factored+Macro Config"),
}


def _make_config(mode, interactive=True, seed_state=None):
    """Create config for mode, optionally with interactive editing."""
    cfg_cls, title = MODE_CONFIGS[mode]
    cfg = cfg_cls()
    if interactive:
        interactive_edit(title, lambda: _build_sections(cfg, seed_state))
    return cfg


def _compute_optimal_reward(cfg):
    """Compute optimal reward from config parameters."""
    num_rooms = cfg.game.kwargs["num_rooms"]
    step_penalty = cfg.game.kwargs.get("step_penalty", 0.01)
    unlock_bonus = cfg.game.kwargs.get("unlock_bonus", 0.1)
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _load_program_logs(seed_dirs):
    """Load program_log.jsonl from each seed directory."""
    logs = {}
    for seed, seed_dir in seed_dirs.items():
        path = seed_dir / "program_log.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            logs[seed] = records
    return logs


def _load_train_stats(seed_dirs):
    """Load train_stats.jsonl from each seed directory."""
    stats = {}
    for seed, seed_dir in seed_dirs.items():
        path = seed_dir / "train_stats.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            stats[seed] = records
    return stats


def _print_seed_summary(logs, seeds):
    """Print aggregated summary table across seeds."""
    if not logs:
        return

    # Final iteration stats per seed
    final_solve = []
    final_reward = []
    final_progs = []
    for seed in seeds:
        if seed not in logs or not logs[seed]:
            continue
        last = logs[seed][-1]
        final_solve.append(last.get("best_solve_rate", 0))
        final_reward.append(last.get("best_avg_reward", 0))
        final_progs.append(last.get("unique_programs", 0))

    n = len(final_solve)
    if n == 0:
        return

    print()
    print("=" * 70)
    print(f"  MULTI-SEED SUMMARY  ({n} seeds: {seeds})")
    print("=" * 70)
    print(f"  Solve rate:    {np.mean(final_solve):.2f} ± {np.std(final_solve):.2f}")
    print(f"  Avg reward:    {np.mean(final_reward):+.3f} ± {np.std(final_reward):.3f}")
    print(f"  Unique progs:  {np.mean(final_progs):.0f} ± {np.std(final_progs):.0f}")
    print()
    print(f"  {'Seed':>6}  {'Solve':>6}  {'Reward':>8}  {'Programs':>8}")
    print(f"  {'---':>6}  {'---':>6}  {'---':>8}  {'---':>8}")
    for i, seed in enumerate(seeds):
        if seed not in logs:
            continue
        print(f"  {seed:>6}  {final_solve[i]:>5.0%}  "
              f"{final_reward[i]:>+8.3f}  {final_progs[i]:>8}")
    print("=" * 70)


def _sausage_panel(ax, seeds, data_dict, extractor, ylabel,
                   color="#1f77b4", label=None, ylim=None, yscale=None,
                   hline=None, overlay=False):
    """Plot mean line + std band + individual seed traces on ax.

    Args:
        ax: matplotlib Axes
        seeds: list of seed values
        data_dict: {seed: [records]}
        extractor: str key or callable(record) -> float
        ylabel: y-axis label
        color: line/fill color
        label: legend label (defaults to ylabel)
        ylim: optional (ymin, ymax)
        yscale: optional "log"
        hline: optional (value, label_str, color_str) for reference line
        overlay: if True, skip xlabel/ylabel (for dual-metric panels)
    """
    if isinstance(extractor, str):
        key = extractor
        extractor = lambda r, _k=key: r.get(_k, 0)

    per_seed = []
    seed_list = []
    for seed in seeds:
        if seed not in data_dict:
            continue
        values = [extractor(e) for e in data_dict[seed]]
        per_seed.append(values)
        seed_list.append(seed)

    if not per_seed:
        return

    min_len = min(len(v) for v in per_seed)
    arr = np.array([v[:min_len] for v in per_seed])
    iters = np.arange(1, min_len + 1)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    lbl = label or ylabel
    ax.plot(iters, means, color=color, linewidth=2, label=lbl)
    if len(per_seed) > 1:
        ax.fill_between(iters, means - stds, means + stds,
                        color=color, alpha=0.15)
    # Individual seed traces (no legend entry to avoid clutter)
    for vals in per_seed:
        ax.plot(np.arange(1, min_len + 1), vals[:min_len],
                color=color, alpha=0.2, linewidth=0.8)

    if not overlay:
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if yscale:
        ax.set_yscale(yscale)
    if hline:
        ax.axhline(y=hline[0], color=hline[2], linestyle=":",
                    alpha=0.5, label=hline[1])
    ax.legend(fontsize=7, loc="best")


def _plot_seed_comparison(logs, seeds, exp_dir, optimal_reward=None,
                          train_stats=None):
    """Plot 2x3 grid of sausage plots across seeds.

    Panels:
      (0,0) Best Solve Rate    (0,1) Best Avg Reward    (0,2) Training Loss
      (1,0) Unique Programs    (1,1) Wall Clock / Iter   (1,2) Self-Play Reward
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    # (0,0) Best Solve Rate
    _sausage_panel(axes[0, 0], seeds, logs, "best_solve_rate",
                   "Best Solve Rate", color="#1f77b4",
                   ylim=(-0.05, 1.15))
    axes[0, 0].set_title("Best Solve Rate")

    # (0,1) Best Avg Reward
    hline_arg = None
    if optimal_reward is not None:
        hline_arg = (optimal_reward, f"Optimal ({optimal_reward:.2f})", "green")
    _sausage_panel(axes[0, 1], seeds, logs, "best_avg_reward",
                   "Best Avg Reward", color="#1f77b4", hline=hline_arg)
    axes[0, 1].set_title("Best Avg Reward")

    # (0,2) Training Loss (dual: policy + value, log scale)
    if train_stats:
        _sausage_panel(axes[0, 2], seeds, train_stats, "train_loss_policy",
                       "Loss", color="#d62728", label="Policy Loss")
        _sausage_panel(axes[0, 2], seeds, train_stats, "train_loss_value",
                       "Loss", color="#2ca02c", label="Value Loss",
                       overlay=True, yscale="log")
        axes[0, 2].set_title("Training Loss")
    else:
        axes[0, 2].text(0.5, 0.5, "No train_stats data",
                        transform=axes[0, 2].transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
        axes[0, 2].set_title("Training Loss")

    # (1,0) Unique Programs
    _sausage_panel(axes[1, 0], seeds, logs, "unique_programs",
                   "Unique Programs", color="#9467bd")
    axes[1, 0].set_title("Unique Programs")

    # (1,1) Wall Clock per Iteration
    _sausage_panel(axes[1, 1], seeds, logs, "iter_wall_clock",
                   "Wall Clock (s)", color="#8c564b")
    axes[1, 1].set_title("Wall Clock / Iteration")

    # (1,2) Self-Play Avg Reward
    if train_stats:
        _sausage_panel(axes[1, 2], seeds, train_stats, "avg_reward",
                       "Self-Play Avg Reward", color="#ff7f0e")
        axes[1, 2].set_title("Self-Play Avg Reward")
    else:
        axes[1, 2].text(0.5, 0.5, "No train_stats data",
                        transform=axes[1, 2].transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
        axes[1, 2].set_title("Self-Play Avg Reward")

    fig.suptitle(f"Seed Comparison ({len(seeds)} seeds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = exp_dir / "seed_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Seed comparison saved to {save_path}")


def _plot_seed_deltas(logs, seeds, exp_dir):
    """Plot per-iteration deltas to reveal stagnation and trends.

    Panels:
      (0) New Programs per Iteration   (1) Reward Improvement   (2) Wall Clock Trend
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _delta_extractor(key):
        """Return extractor that computes per-iteration delta from cumulative series."""
        def extract(records):
            values = [r.get(key, 0) for r in records]
            deltas = [values[0]] + [values[i] - values[i - 1]
                                     for i in range(1, len(values))]
            return deltas
        return extract

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    # Compute deltas per seed and plot
    panels = [
        ("unique_programs", "New Programs / Iter", "#9467bd"),
        ("best_avg_reward", "Reward Improvement / Iter", "#1f77b4"),
        ("iter_wall_clock", "Wall Clock (s)", "#8c564b"),
    ]

    for ax, (key, ylabel, color) in zip(axes, panels):
        per_seed = []
        for seed in seeds:
            if seed not in logs:
                continue
            if key == "iter_wall_clock":
                # Wall clock is already per-iteration, not cumulative
                values = [r.get(key, 0) for r in logs[seed]]
            else:
                values = _delta_extractor(key)(logs[seed])
            per_seed.append(values)

        if not per_seed:
            continue

        min_len = min(len(v) for v in per_seed)
        arr = np.array([v[:min_len] for v in per_seed])
        iters = np.arange(1, min_len + 1)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        ax.plot(iters, means, color=color, linewidth=2, label="Mean")
        if len(per_seed) > 1:
            ax.fill_between(iters, means - stds, means + stds,
                            color=color, alpha=0.15)
        for vals in per_seed:
            ax.plot(np.arange(1, min_len + 1), vals[:min_len],
                    color=color, alpha=0.2, linewidth=0.8)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Per-Iteration Deltas ({len(seeds)} seeds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = exp_dir / "seed_deltas.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Seed deltas saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments. Returns None for interactive defaults."""
    parser = argparse.ArgumentParser(
        description="Doors Derivation -- AlphaZero Program Synthesis Training")
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()),
                        default=None,
                        help="Grammar mode (default: interactive selection)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds for multi-seed runs with CI")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive config editing")
    return parser.parse_args()


def _parse_seeds(seeds_str):
    """Parse comma-separated seed string into list of ints."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Mode selection
    mode = args.mode if args.mode else select_mode()

    # Seed state for interactive editing
    default_seed = cfg_seed = str(args.seeds[0]) if args.seeds else "43"
    seed_state = {"seeds_str": default_seed}

    # Config creation (seed_state is editable in interactive mode)
    cfg = _make_config(mode, interactive=not args.non_interactive,
                       seed_state=seed_state)

    # Determine seeds: CLI --seeds overrides interactive input
    if args.seeds and len(args.seeds) > 1:
        seeds = args.seeds
    else:
        seeds = _parse_seeds(seed_state["seeds_str"])

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg, mode=mode)

    print_banner(cfg, exp_dir, mode=mode)
    print_architecture(cfg, mode=mode)

    optimal_reward = _compute_optimal_reward(cfg)

    if len(seeds) <= 1:
        # Single-seed run
        if seeds:
            s = seeds[0]
            cfg.agent.random_seeds = {
                "mcts": s, "train": s + 1,
                "eval": s + 2, "external_policy": s + 3,
            }
        cfg.save(str(exp_dir / "config.json"))
        run_derivation_training(cfg, mode, optimal_reward, exp_dir)
    else:
        # Multi-seed run
        cfg.save(str(exp_dir / "config.json"))
        seed_dirs = {}
        for seed in seeds:
            print()
            print(f"{'='*70}")
            print(f"  SEED {seed}")
            print(f"{'='*70}")
            cfg.agent.random_seeds = {
                "mcts": seed, "train": seed + 1,
                "eval": seed + 2, "external_policy": seed + 3,
            }
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dirs[seed] = seed_dir
            run_derivation_training(cfg, mode, optimal_reward, seed_dir)

        # Aggregate across seeds
        logs = _load_program_logs(seed_dirs)
        train_stats = _load_train_stats(seed_dirs)
        _print_seed_summary(logs, seeds)
        _plot_seed_comparison(logs, seeds, exp_dir,
                              optimal_reward=optimal_reward,
                              train_stats=train_stats)
        _plot_seed_deltas(logs, seeds, exp_dir)

        print()
        print("[Analysis] Plots saved. Key things to look for:")
        print("  - seed_comparison.png: Do seeds converge? "
              "High variance = unstable training")
        print("  - seed_deltas.png: Flat delta curves = stagnation, "
              "consider stopping early")
        print("  - If wall clock grows: check cache size / "
              "tree reuse overhead")


if __name__ == "__main__":
    main()
