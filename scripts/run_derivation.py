from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import copy
import json
import logging
from math import comb
from alphazeropp.training.gated_trainer import GatedTrainer
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.bitstring.dsl.derivation_config import DerivationConfig
from alphazeropp.instances.bitstring.dsl.derivation_game import compute_max_productions
from alphazeropp.instances.bitstring.dsl.budget_grammar import count_programs
from alphazeropp.instances.bitstring.dsl.leaf_evaluator import VALID_METRICS
from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_sections(cfg):
    """Return sections for DerivationGame AlphaZero config."""
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

    problem_labels = {"n_sites", "budget", "n_ones", "n_frozen_states", "potential"}
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

def setup_experiment_dir(cfg):
    """Create and return an experiment directory path. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    budget = cfg.game.kwargs["budget"]
    n_sites = cfg.game.kwargs["n_sites"]
    metric = cfg.game.kwargs["metric"]
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations

    dirname = (f"{timestamp}_N{n_sites}_L{budget}_{metric}"
               f"_mcts{sim}_games{games}_iter{iters}")

    exp_dir = Path("experiments") / "derivation" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Training output helpers
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir):
    """Print startup banner with experiment info."""
    budget = cfg.game.kwargs["budget"]
    n_sites = cfg.game.kwargs["n_sites"]
    n_ones = cfg.game.kwargs["n_ones"]
    metric = cfg.game.kwargs["metric"]
    total_programs = count_programs(n_sites, budget)
    max_prods = compute_max_productions(budget, n_sites)
    n_frozen = cfg.game.kwargs.get("n_frozen_states", 1)
    n_total = comb(n_sites, n_ones)
    optimal_reward = (n_sites - n_ones) / n_sites
    all_ones = ", ".join(["1"] * n_sites)
    example_init = ["0"] * n_sites
    for j in range(n_ones):
        example_init[j] = "1"

    print()
    print("=" * 80)
    print("  PROGRAM SYNTHESIS via AlphaZero")
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
    print(f"  SEARCH SPACE")
    print(f"    Bitstring length:   N = {n_sites}")
    print(f"    AST node budget:    L = {budget}")
    print(f"    Possible programs:  {total_programs}")
    print(f"    Max productions:    {max_prods} (action space per derivation step)")
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


def print_architecture(cfg):
    """Print the AlphaZero training loop architecture diagram."""
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    budget = cfg.game.kwargs["budget"]
    n_sites = cfg.game.kwargs["n_sites"]
    d_model = cfg.net.kwargs.get("d_model", 64)

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero for Program Synthesis")
    print("=" * 80)
    print()
    print(f"  Key difference from pure MCTS (run_derivation_mcts.py):")
    print(f"    Pure MCTS:    net = uniform(1/K, 0)    -> no learning")
    print(f"    AlphaZero:    net = Transformer(d={d_model})  -> learns from self-play")
    print()
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  ┌── Iteration i ──────────────────────────────────────────────────────┐")
    print("  │                                                                      │")
    print(f"  │  STEP 1: Self-Play  ({n_games} derivation games)")
    print("  │")
    print(f"  │    Each game: expand ProgramHole({budget}) to a complete program")
    print(f"  │      At each step: MCTS(partial_ast, Transformer, n_sims={n_sims})")
    print("  │        Transformer reads partial AST (preorder encoding)")
    print("  │        -> (pi: production probs, v: predicted program quality)")
    print("  │        MCTS refines pi using UCB + backed-up leaf evaluations")
    print("  │      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  │      SAVE (partial_ast, pi_MCTS) as training target")
    print("  │    Terminal: complete program -> leaf_eval(prog) -> scalar reward")
    print("  │")
    print(f"  │  STEP 2: Train Transformer  (replay buffer: last {n_past} iterations)")
    print("  │")
    print("  │    L = (z - v_theta(ast))^2  -  pi_MCTS * log p_theta(ast)")
    print("  │        value loss              policy loss")
    print("  │    theta <- theta - alpha * grad(L)")
    print("  │")
    print("  │  STEP 3: Evaluate")
    print("  │    Pit new_net vs old_net on fresh derivation games.")
    print("  │                                                                      │")
    print("  └──────────────────────────────────────────────────────────────────────┘")
    print()
    print("  Virtuous cycle:")
    print("    Better Transformer -> better MCTS -> better data -> better Transformer")
    print("    The network learns which partial ASTs are promising,")
    print("    directing search toward high-quality programs.")
    print()


def print_iteration_header(i, total):
    """Print iteration separator."""
    header = f"--- Iteration {i}/{total} "
    print(header + "-" * (80 - len(header)))


def print_iteration_summary(i, total, score, trainer_stats, evaluator_stats,
                            best_program_str, best_metrics):
    """Print compact iteration summary with best program info."""
    train_recs = [r for r in trainer_stats.to_list() if "train_loss" in r]
    train_rec = train_recs[-1] if train_recs else {}
    eval_rec = evaluator_stats.to_list()[-1] if evaluator_stats.to_list() else {}

    loss = train_rec.get("train_loss", float("nan"))
    p_loss = train_rec.get("train_loss_policy", float("nan"))
    v_loss = train_rec.get("train_loss_value", float("nan"))
    n_examples = train_rec.get("num_examples", 0)

    new_mean = eval_rec.get("new_rewards_mean", float("nan"))
    old_mean = eval_rec.get("old_rewards_mean", float("nan"))

    print(
        f"[ITER {i}/{total}] Score: {score*100:.1f}% "
        f"| New reward: {new_mean:.3f} vs Old: {old_mean:.3f} "
        f"| Loss: {loss:.4f} (P:{p_loss:.3f} V:{v_loss:.3f}) "
        f"| Examples: {n_examples}"
    )

    if best_program_str is not None and best_metrics is not None:
        sr = best_metrics.get("solve_rate", 0)
        ar = best_metrics.get("avg_reward", 0)
        n_episodes = best_metrics.get("n_episodes", 1)
        if n_episodes == 1:
            solved_str = "Solved" if sr >= 1.0 else "Not solved"
            print(f"[PROG] Best so far: {solved_str}, avg_reward={ar:+.4f}")
        else:
            print(
                f"[PROG] Best so far: solve_rate={sr:.1%}, "
                f"avg_reward={ar:+.4f}"
            )
        prog_display = (best_program_str if len(best_program_str) < 70
                        else best_program_str[:67] + "...")
        print(f"       {prog_display}")
    print()


def extract_best_program(leaf_eval):
    """Extract the best program from LeafEvaluator's cache.

    Returns:
        (prog_str, prog_ast, prog_metrics, prog_value) or
        (None, None, None, -inf) if cache is empty.
    """
    if not leaf_eval._cache:
        return None, None, None, float("-inf")
    best_key = max(leaf_eval._cache, key=leaf_eval._cache.get)
    return (
        best_key,
        leaf_eval._program_cache[best_key],
        leaf_eval._full_cache[best_key],
        leaf_eval._cache[best_key],
    )


def print_best_program_traces(program, metrics, leaf_eval, max_traces=5):
    """Show the best program and step-by-step traces on frozen states."""
    from alphazeropp.instances.bitstring.dsl.interpreter import (
        run_policy_episode, format_trace,
    )

    n_sites = leaf_eval.n_sites
    n_episodes = metrics.get("n_episodes", 1)
    sr = metrics.get("solve_rate", 0)
    ar = metrics.get("avg_reward", 0)
    avg_steps = metrics.get("avg_steps", 0)
    avg_ops = metrics.get("avg_ops", 0)

    print()
    print("=" * 80)
    print("  BEST PROGRAM FOUND")
    print("=" * 80)
    print()

    # Pretty-print the program
    print("  Program:")
    for line in program.pretty().split("\n"):
        print(f"    {line}")
    print()
    print(f"  AST nodes: {program.node_count()}")
    print()

    # Aggregate metrics
    print("  Aggregate metrics across frozen initial states:")
    if n_episodes == 1:
        solved_str = "Yes" if sr >= 1.0 else "No"
        print(f"    Solved:          {solved_str}")
    else:
        n_solved = int(sr * n_episodes)
        print(f"    Solve rate:      {n_solved}/{n_episodes} ({sr:.1%})")
    print(f"    Avg reward:      {ar:+.4f}")
    print(f"    Avg steps:       {avg_steps:.1f}")
    print(f"    Avg interp ops:  {avg_ops:.1f}")
    print()

    # Run on frozen states and show traces
    frozen_states = leaf_eval.frozen_states
    n_to_show = min(len(frozen_states), max_traces)

    print(f"  Step-by-step execution on {n_to_show} frozen initial "
          f"state{'s' if n_to_show != 1 else ''}:")
    print("-" * 80)

    for i, x0 in enumerate(frozen_states[:n_to_show]):
        env = leaf_eval.game_config.make_env(n_sites, frozen_states=[x0])
        env.reset()
        result = run_policy_episode(env, program, x0=x0)

        if i == 0:
            # Full trace for the first state
            print(format_trace(result, program=program))
        else:
            # Compact summary for subsequent states
            state_str = "[" + ", ".join(str(int(b)) for b in x0) + "]"
            status = "SOLVED" if result.solved else "NOT SOLVED"
            print(f"  {state_str} -> {status} "
                  f"in {result.total_env_steps} steps, "
                  f"reward={result.cumulative_reward:+.4f}")

    if len(frozen_states) > max_traces:
        print(f"  ... and {len(frozen_states) - max_traces} more states "
              f"(not shown)")

    print("=" * 80)
    print()


def rename_plot_with_stats(cfg, trainer_stats, evaluator_stats, best_metrics):
    """Rename the plot file to include experiment info and final stats."""
    plot_path = Path(cfg.run.plot_path)
    if not plot_path.exists():
        return str(plot_path)

    eval_recs = evaluator_stats.to_list()
    new_mean = eval_recs[-1].get("new_rewards_mean", 0) if eval_recs else 0

    n_sites = cfg.game.kwargs["n_sites"]
    budget = cfg.game.kwargs["budget"]
    metric = cfg.game.kwargs["metric"]
    sr = best_metrics.get("solve_rate", 0) if best_metrics else 0
    sr_str = f"sr{sr:.0%}".replace("%", "pct")
    reward_str = f"reward{new_mean:.2f}".replace(".", "p")
    new_name = plot_path.with_name(
        f"metrics_N{n_sites}_L{budget}_{metric}_{sr_str}_{reward_str}.png"
    )

    plot_path.rename(new_name)
    return str(new_name)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager,
                          program_log, save_path=None, optimal_reward=None):
    """Plot training metrics with derivation-specific best-program tracking."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plot.")
        return

    trainer_history = [
        entry for entry in (trainer_stats_manager.to_list()
                            if trainer_stats_manager else [])
        if "train_loss" in entry
    ]
    evaluator_history = (evaluator_stats_manager.to_list()
                         if evaluator_stats_manager else [])
    max_len = max(len(trainer_history), len(evaluator_history), len(program_log))
    if max_len == 0:
        print("[Warning] No statistics available for plotting.")
        return

    merged = []
    for i in range(max_len):
        row = {"iteration": i + 1}
        if i < len(trainer_history):
            row.update(trainer_history[i])
        if i < len(evaluator_history):
            row.update(evaluator_history[i])
        if i < len(program_log):
            row["best_solve_rate"] = program_log[i].get("best_solve_rate", 0)
            row["best_avg_reward"] = program_log[i].get("best_avg_reward", 0)
        merged.append(row)

    df = pd.DataFrame(merged)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("DerivationGame AlphaZero Training Metrics",
                 fontsize=14, fontweight="bold")

    # Plot 1: Eval reward mean with std band
    ax1 = axes[0, 0]
    if optimal_reward is not None:
        ax1.axhline(y=optimal_reward, color="green", linestyle="--",
                     linewidth=1.5, label=f"Optimal ({optimal_reward:.3f})")
    if "new_rewards_mean" in df.columns:
        ax1.plot(df["iteration"], df["new_rewards_mean"],
                 "b-", linewidth=2, label="New Reward Mean")
        if "new_rewards_std" in df.columns:
            ax1.fill_between(
                df["iteration"],
                df["new_rewards_mean"] - df["new_rewards_std"],
                df["new_rewards_mean"] + df["new_rewards_std"],
                alpha=0.3, color="blue", label="+-1 Std",
            )
    if "old_rewards_mean" in df.columns:
        ax1.plot(df["iteration"], df["old_rewards_mean"],
                 "c--", linewidth=2, label="Old Reward Mean")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reward")
    ax1.set_title("Evaluation Rewards")
    if len(ax1.lines) > 0:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Policy and Value Loss
    ax2 = axes[0, 1]
    has_loss = False
    if "train_loss_policy" in df.columns:
        ax2.plot(df["iteration"], df["train_loss_policy"],
                 "r-", linewidth=2, label="Policy Loss")
        has_loss = True
    if "train_loss_value" in df.columns:
        ax2.plot(df["iteration"], df["train_loss_value"],
                 "g-", linewidth=2, label="Value Loss")
        has_loss = True
    if not has_loss and "train_loss" in df.columns:
        ax2.plot(df["iteration"], df["train_loss"],
                 "k-", linewidth=2, label="Train Loss")
        has_loss = True
    if has_loss:
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Losses")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")
    else:
        ax2.axis("off")

    # Plot 3: Training Set Size
    ax3 = axes[1, 0]
    if "num_examples" in df.columns:
        ax3.plot(df["iteration"], df["num_examples"],
                 "m-", linewidth=2, marker="o", markersize=4)
        ax3.set_ylabel("Num Examples")
        ax3.set_title("Training Set Size")
    ax3.set_xlabel("Iteration")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Best program quality over iterations
    ax4 = axes[1, 1]
    if "best_solve_rate" in df.columns:
        ax4.plot(df["iteration"], df["best_solve_rate"],
                 "g-", linewidth=2, marker="s", markersize=4,
                 label="Best Solve Rate")
    if "best_avg_reward" in df.columns:
        ax4.plot(df["iteration"], df["best_avg_reward"],
                 "b--", linewidth=2, marker="^", markersize=4,
                 label="Best Avg Reward")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Score")
    ax4.set_title("Best Program Quality")
    if len(ax4.lines) > 0:
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved to {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = DerivationConfig()

    # Interactive config editing
    interactive_edit("DerivationGame Config", lambda: _build_sections(cfg))

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg)
    cfg.save(str(exp_dir / "config.json"))

    # Build objects
    game, net, agent, trainer, evaluator = cfg.build()

    print_banner(cfg, exp_dir)
    print_architecture(cfg)

    n_sites = cfg.game.kwargs["n_sites"]
    n_ones = cfg.game.kwargs["n_ones"]
    optimal_reward = (n_sites - n_ones) / n_sites

    # Access LeafEvaluator for program tracking (works in sequential mode)
    leaf_eval = game.leaf_evaluator

    # Track best program across all iterations
    best_program_str = None
    best_program_ast = None
    best_program_metrics = None
    best_program_value = float("-inf")
    program_log = []

    # Training loop
    gated_trainer = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold)
    for i in range(cfg.run.n_iterations):
        print_iteration_header(i + 1, cfg.run.n_iterations)

        score, accepted = gated_trainer.train_iteration()

        # Extract best program from leaf_eval cache
        prog_str, prog_ast, prog_metrics, prog_value = extract_best_program(leaf_eval)
        if prog_str is not None and prog_value > best_program_value:
            best_program_value = prog_value
            best_program_str = prog_str
            best_program_ast = prog_ast
            best_program_metrics = prog_metrics

        program_log.append({
            "iteration": i + 1,
            "best_program": best_program_str,
            "best_solve_rate": (best_program_metrics.get("solve_rate", 0)
                                if best_program_metrics else 0),
            "best_avg_reward": (best_program_metrics.get("avg_reward", 0)
                                if best_program_metrics else 0),
            "unique_programs": leaf_eval.stats()["unique_programs"],
        })

        print_iteration_summary(
            i + 1, cfg.run.n_iterations, score,
            trainer.statistics_manager, evaluator.statistics_manager,
            best_program_str, best_program_metrics,
        )

        # Save logs after each iteration
        trainer.statistics_manager.save_jsonl(str(exp_dir / "train_stats.jsonl"))
        evaluator.statistics_manager.save_jsonl(str(exp_dir / "eval_stats.jsonl"))
        with open(exp_dir / "program_log.jsonl", "w") as f:
            for entry in program_log:
                f.write(json.dumps(entry) + "\n")

        if i % cfg.run.plot_every == 0:
            plot_training_metrics(
                trainer.statistics_manager,
                evaluator.statistics_manager,
                program_log,
                save_path=cfg.run.plot_path,
                optimal_reward=optimal_reward,
            )

        # Early stopping: perfect program found
        if (best_program_metrics is not None
                and best_program_metrics.get("solve_rate", 0) >= 1.0):
            print(f"[EARLY STOP] Perfect program found at iteration {i+1}!")
            print(f"  {best_program_str}")
            break

    # Final plot
    plot_training_metrics(
        trainer.statistics_manager,
        evaluator.statistics_manager,
        program_log,
        save_path=cfg.run.plot_path,
        optimal_reward=optimal_reward,
    )
    final_plot = rename_plot_with_stats(
        cfg, trainer.statistics_manager, evaluator.statistics_manager,
        best_program_metrics,
    )

    # Best program explanation with step-by-step traces
    if best_program_ast is not None:
        print_best_program_traces(
            best_program_ast, best_program_metrics, leaf_eval,
        )

    print(f"Training complete. Results saved to: {exp_dir}/")
    print(f"  Config:       {exp_dir / 'config.json'}")
    print(f"  Plot:         {final_plot}")
    print(f"  Train log:    {exp_dir / 'train_stats.jsonl'}")
    print(f"  Eval log:     {exp_dir / 'eval_stats.jsonl'}")
    print(f"  Program log:  {exp_dir / 'program_log.jsonl'}")
    print(f"  Programs evaluated: {leaf_eval.stats()['unique_programs']}")


if __name__ == "__main__":
    main()
