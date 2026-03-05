"""
BitString Derivation -- AlphaZero Program Synthesis Training.

Synthesizes BitString policies via grammar-guided MCTS + AlphaZero.
Two grammar modes:
  - scan: Priority-scan permutation grammar (N! search space)
  - cfg:  Size-budget CFG grammar (AST expansion)

Usage:
    python scripts/run_bitstring_derivation.py
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import json
import logging
from math import comb, factorial
from datetime import datetime
from pathlib import Path

from alphazeropp.instances.bitstring.dsl.derivation_config import (
    DerivationConfig, ScanDerivationConfig,
)
from alphazeropp.synthesis.derivation_game import compute_max_productions
from alphazeropp.synthesis.budget_grammar import count_programs
from alphazeropp.synthesis.leaf_evaluator import VALID_METRICS
from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)
from alphazeropp.utils.derivation_utils import run_derivation_training


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_sections_scan(cfg):
    """Return sections for ScanDerivationGame AlphaZero config."""
    def _set_n_sites(val):
        cfg.game.kwargs["n_sites"] = val
        cfg.net.kwargs["n_sites"] = val

    params = [
        # Problem
        ("n_sites", cfg.game.kwargs["n_sites"], _set_n_sites,
         "Number of bits in the bitstring", None),
        ("n_ones", cfg.game.kwargs["n_ones"], dict_setter(cfg.game.kwargs, "n_ones"),
         "Number of 1s in initial states", None),
        ("n_frozen_states", cfg.game.kwargs.get("n_frozen_states", 1),
         dict_setter(cfg.game.kwargs, "n_frozen_states"),
         "Frozen initial states for evaluation", None),
        ("potential", cfg.game.kwargs["potential_name"],
         dict_setter(cfg.game.kwargs, "potential_name"),
         "Reward shaping function", list(POTENTIAL_REGISTRY.keys())),
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
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k], dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], None))

    # Network architecture
    params.extend([
        ("d_hidden", cfg.net.kwargs["d_hidden"], dict_setter(cfg.net.kwargs, "d_hidden"),
         "MLP hidden layer dimension", None),
        ("n_hidden_layers", cfg.net.kwargs["n_hidden_layers"],
         dict_setter(cfg.net.kwargs, "n_hidden_layers"),
         "MLP hidden layers", None),
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

    problem_labels = {"n_sites", "n_ones", "n_frozen_states", "potential"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon"}
    net_labels = {"d_hidden", "n_hidden_layers", "learning_rate", "batch_size"}
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


def _build_sections_cfg(cfg):
    """Return sections for DerivationGame (CFG) AlphaZero config."""
    def _set_n_sites(val):
        cfg.game.kwargs["n_sites"] = val
        cfg.net.kwargs["n_sites"] = val

    def _set_budget(val):
        cfg.game.kwargs["budget"] = val
        cfg.net.kwargs["budget"] = val

    params = [
        # Problem
        ("n_sites", cfg.game.kwargs["n_sites"], _set_n_sites,
         "Number of bits in the bitstring", None),
        ("budget", cfg.game.kwargs["budget"], _set_budget,
         "AST node budget (program size)", None),
        ("n_ones", cfg.game.kwargs["n_ones"], dict_setter(cfg.game.kwargs, "n_ones"),
         "Number of 1s in initial states", None),
        ("n_frozen_states", cfg.game.kwargs.get("n_frozen_states", 1),
         dict_setter(cfg.game.kwargs, "n_frozen_states"),
         "Frozen initial states for evaluation", None),
        ("potential", cfg.game.kwargs["potential_name"],
         dict_setter(cfg.game.kwargs, "potential_name"),
         "Reward shaping function", list(POTENTIAL_REGISTRY.keys())),
        ("budget_mode", cfg.game.kwargs.get("program_budget_mode", "exact"),
         dict_setter(cfg.game.kwargs, "program_budget_mode"),
         "Budget mode: exact (==L) or max (<=L)", ["exact", "max"]),
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
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon"]:
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

    problem_labels = {"n_sites", "budget", "n_ones", "n_frozen_states", "potential",
                      "budget_mode"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon"}
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

def setup_experiment_dir(cfg, mode="scan"):
    """Create and return an experiment directory path. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_sites = cfg.game.kwargs["n_sites"]
    metric = cfg.game.kwargs["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations

    if mode == "scan":
        dirname = (f"{timestamp}_scan_N{n_sites}_{metric}"
                   f"_mcts{sim}_games{games}_iter{iters}")
    else:
        budget = cfg.game.kwargs["budget"]
        bmode = cfg.game.kwargs.get("program_budget_mode", "exact")
        dirname = (f"{timestamp}_cfg_N{n_sites}_L{budget}_{bmode}_{metric}"
                   f"_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "bitstring_derivation" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Training output helpers
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir, mode="scan"):
    """Print startup banner with experiment info."""
    n_sites = cfg.game.kwargs["n_sites"]
    n_ones = cfg.game.kwargs["n_ones"]
    n_frozen = cfg.game.kwargs.get("n_frozen_states", 1)
    n_total = comb(n_sites, n_ones)
    metric = cfg.game.kwargs["metric"]
    optimal_reward = (n_sites - n_ones) / n_sites
    all_ones = ", ".join(["1"] * n_sites)
    example_init = ["0"] * n_sites
    for j in range(n_ones):
        example_init[j] = "1"

    print()
    print("=" * 80)
    print("  BITSTRING PROGRAM SYNTHESIS via AlphaZero")
    print("=" * 80)
    print()
    print("  PROBLEM")
    print(f"    Given a {n_sites}-bit string with {n_ones} ones and "
          f"{n_sites - n_ones} zeros (e.g. [{', '.join(example_init)}]),")
    print(f"    synthesize a program that flips bits to reach the all-ones "
          f"state [{all_ones}].")
    print()
    print("    Programs are decision lists in a DSL of if/elif/else rules:")
    print("      if IsZero(i):    -- test whether bit i is 0")
    print("        Flip(i)        -- flip bit i")
    print("      elif IsZero(j):  -- next rule")
    print("        Flip(j)")
    print("      else:")
    print("        Flip(k)        -- default action")
    print()
    print(f"  GRAMMAR: {mode.upper()}")
    if mode == "scan":
        total_programs = factorial(n_sites)
        print(f"    Scan grammar: construct a priority permutation over {n_sites} bits")
        print(f"    Derivation steps: {n_sites - 1} (last index forced)")
        print(f"    Max branching:    {n_sites} (action = choose next bit index)")
        print(f"    Search space:     {n_sites}! = {total_programs} permutations")
    else:
        budget = cfg.game.kwargs["budget"]
        bmode = cfg.game.kwargs.get("program_budget_mode", "exact")
        total_programs = count_programs(n_sites, budget)
        max_prods = compute_max_productions(budget, n_sites, mode=bmode)
        print(f"    CFG grammar: construct AST by expanding grammar productions")
        print(f"    AST node budget:    L = {budget}")
        print(f"    Budget mode:        {bmode}"
              + (" (programs use exactly L nodes)" if bmode == "exact"
                 else " (programs use <= L nodes)"))
        print(f"    Possible programs:  {total_programs}")
        print(f"    Max productions:    {max_prods} (action space per step)")
    print()
    print(f"  SEARCH SPACE")
    print(f"    Bitstring length:   N = {n_sites}")
    print()
    print(f"  EVALUATION")
    print(f"    Frozen init states: {n_frozen} of {n_total} "
          f"(C({n_sites},{n_ones}) with {n_ones} ones)")
    print(f"    Metric:             {metric}")
    print(f"    Optimal reward:     {optimal_reward:.4f} = "
          f"({n_sites}-{n_ones})/{n_sites}")
    print(f"    A program \"solves\" a state if it reaches [{all_ones}].")
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


def print_architecture(cfg, mode="scan"):
    """Print the AlphaZero training loop architecture diagram."""
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    n_sites = cfg.game.kwargs["n_sites"]
    if mode == "scan":
        d_model = cfg.net.kwargs.get("d_hidden", 128)
        net_desc = f"MLP(d={d_model})"
        step_desc = f"pick next bit index from remaining {n_sites} positions"
    else:
        budget = cfg.game.kwargs["budget"]
        d_model = cfg.net.kwargs.get("d_model", 64)
        net_desc = f"Transformer(d={d_model})"
        step_desc = f"expand ProgramHole({budget}) to a complete program"

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero for Program Synthesis")
    print("=" * 80)
    print()
    print(f"  Key difference from pure MCTS (run_derivation_mcts.py):")
    print(f"    Pure MCTS:    net = uniform(1/K, 0)    -> no learning")
    print(f"    AlphaZero:    net = {net_desc}  -> learns from self-play")
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
    print(f"  |        Network reads partial state")
    print("  |        -> (pi: production probs, v: predicted program quality)")
    print("  |        MCTS refines pi using UCB + backed-up leaf evaluations")
    print("  |      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  |      SAVE (partial_ast, pi_MCTS) as training target")
    print("  |    Terminal: complete program -> leaf_eval(prog) -> scalar reward")
    print("  |")
    print(f"  |  STEP 2: Train network  (replay buffer: last {n_past} iterations)")
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
    print("    Better network -> better MCTS -> better data -> better network")
    print("    The network learns which partial ASTs are promising,")
    print("    directing search toward high-quality programs.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_mode():
    """Prompt user to select derivation grammar mode."""
    print()
    print("=== BitString Derivation Grammar Mode ===")
    print()
    print("  0) scan  -- Priority-scan grammar (default)")
    print("     Builds a permutation: N! search space, branching factor N")
    print()
    print("  1) cfg   -- Size-budget CFG grammar")
    print("     Builds an AST: millions of programs, branching factor ~48")
    print()
    choice = input("  Select mode [0]: ").strip()
    if choice in ("1", "cfg"):
        return "cfg"
    return "scan"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mode = select_mode()

    if mode == "scan":
        cfg = ScanDerivationConfig()
        interactive_edit("ScanDerivationGame Config",
                         lambda: _build_sections_scan(cfg))
    else:
        cfg = DerivationConfig()
        interactive_edit("DerivationGame Config (CFG)",
                         lambda: _build_sections_cfg(cfg))

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg, mode=mode)
    cfg.save(str(exp_dir / "config.json"))

    print_banner(cfg, exp_dir, mode=mode)
    print_architecture(cfg, mode=mode)

    n_sites = cfg.game.kwargs["n_sites"]
    n_ones = cfg.game.kwargs["n_ones"]
    optimal_reward = (n_sites - n_ones) / n_sites

    run_derivation_training(cfg, mode, optimal_reward, exp_dir)


if __name__ == "__main__":
    main()
