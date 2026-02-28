from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from alphazeropp.instances import BitStringConfig

import copy
import torch
import logging
from datetime import datetime
from pathlib import Path


def models_equal(m1, m2):
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()

    if sd1.keys() != sd2.keys():
        return False

    for k in sd1:
        if not torch.equal(sd1[k], sd2[k]):
            return False

    return True


# ---------------------------------------------------------------------------
# Config display & interactive editing
# ---------------------------------------------------------------------------

def _build_param_table(cfg):
    """Build a list of (number, label, value, setter, description) for all editable params."""
    params = []
    n = 1

    def add(label, value, setter, desc=""):
        nonlocal n
        params.append((n, label, value, setter, desc))
        n += 1

    # Game
    def _set_n_sites(val):
        cfg.game.kwargs["n_sites"] = val
        cfg.net.kwargs["n_sites"] = val

    add("n_sites", cfg.game.kwargs["n_sites"], _set_n_sites,
        "Length of the binary vector")
    add("bit_flip", cfg.game.kwargs["bit_flip"],
        lambda val: cfg.game.kwargs.__setitem__("bit_flip", val),
        "Actions flip bits (vs set bits)")
    add("sparse_reward", cfg.game.kwargs["sparse_reward"],
        lambda val: cfg.game.kwargs.__setitem__("sparse_reward", val),
        "Reward only at episode end")

    # MCTS
    descs = {
        "n_simulations":    "MCTS rollouts per move",
        "temperature":      "Exploration temperature for action selection",
        "c_exploration":    "UCB exploration constant",
        "dirichlet_alpha":  "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
    }
    for k in ["n_simulations", "temperature", "c_exploration", "dirichlet_alpha", "dirichlet_epsilon"]:
        if k in cfg.agent.mcts_params:
            v = cfg.agent.mcts_params[k]
            add(k, v, lambda val, _k=k: cfg.agent.mcts_params.__setitem__(_k, val),
                descs[k])

    # Agent
    add("reward_discount", cfg.agent.reward_discount,
        lambda val: setattr(cfg.agent, "reward_discount", val),
        "Discount factor for future rewards")

    # Trainer
    add("n_games_per_train", cfg.trainer.n_games_per_train,
        lambda val: setattr(cfg.trainer, "n_games_per_train", val),
        "Self-play games per training iteration")
    add("n_past_iters", cfg.trainer.n_past_iterations_to_train,
        lambda val: setattr(cfg.trainer, "n_past_iterations_to_train", val),
        "Past iterations kept in training buffer")
    add("n_procs", cfg.trainer.n_procs,
        lambda val: setattr(cfg.trainer, "n_procs", val),
        "Parallel workers for self-play")

    # Evaluator
    add("eval_n_games", cfg.evaluator.n_games,
        lambda val: setattr(cfg.evaluator, "n_games", val),
        "Games to pit new vs old agent")
    add("eval_n_procs", cfg.evaluator.n_procs,
        lambda val: setattr(cfg.evaluator, "n_procs", val),
        "Parallel workers for evaluation")

    # Run
    add("n_iterations", cfg.run.n_iterations,
        lambda val: setattr(cfg.run, "n_iterations", val),
        "Total training iterations")
    add("accept_threshold", cfg.run.accept_threshold,
        lambda val: setattr(cfg.run, "accept_threshold", val),
        "Win rate to accept new network")
    add("plot_every", cfg.run.plot_every,
        lambda val: setattr(cfg.run, "plot_every", val),
        "Plot metrics every N iterations")

    return params


def display_config(cfg):
    """Print all hyperparameters as a numbered table with descriptions."""
    params = _build_param_table(cfg)

    sections = [
        ("Game",      [p for p in params if p[1] in ("n_sites", "bit_flip", "sparse_reward")]),
        ("MCTS",      [p for p in params if p[1] in ("n_simulations", "temperature", "c_exploration", "dirichlet_alpha", "dirichlet_epsilon")]),
        ("Agent",     [p for p in params if p[1] in ("reward_discount",)]),
        ("Trainer",   [p for p in params if p[1] in ("n_games_per_train", "n_past_iters", "n_procs")]),
        ("Evaluator", [p for p in params if p[1] in ("eval_n_games", "eval_n_procs")]),
        ("Run",       [p for p in params if p[1] in ("n_iterations", "accept_threshold", "plot_every")]),
    ]

    print("\n=== BitString Config ===\n")
    for section_name, section_params in sections:
        print(f"  {section_name}:")
        for num, label, value, _, desc in section_params:
            line = f"    {num:>2}) {label:<22} = {str(value):<10}"
            if desc:
                line += f"  # {desc}"
            print(line)
        print()


def _parse_value(raw, current_value):
    """Parse a string input to the same type as current_value."""
    t = type(current_value)
    if t is bool:
        return raw.strip().lower() in ("true", "1", "yes")
    return t(raw)


def interactive_edit(cfg):
    """Display config and let user edit parameters by number."""
    while True:
        display_config(cfg)
        params = _build_param_table(cfg)
        param_map = {p[0]: p for p in params}

        choice = input("  Enter number to edit (or 'run' to start): ").strip()
        if choice.lower() in ("run", "start", ""):
            break

        try:
            num = int(choice)
        except ValueError:
            print(f"  Invalid input: '{choice}'. Enter a number or 'run'.")
            continue

        if num not in param_map:
            print(f"  No parameter with number {num}.")
            continue

        _, label, current, setter, _ = param_map[num]
        raw = input(f"  New value for {label} ({type(current).__name__}) [{current}]: ").strip()
        if raw == "":
            continue

        try:
            new_val = _parse_value(raw, current)
            setter(new_val)
            print(f"  Updated: {label} = {new_val}")
        except (ValueError, TypeError) as e:
            print(f"  Invalid value: {e}")


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def setup_experiment_dir(cfg):
    """
    Create and return an experiment directory path. Updates cfg paths.

    Directory layout justification:
      experiments/          - Already in .gitignore; standard ML convention that
                              separates transient run artifacts from source code.
      bitstring/            - Groups by game type so future games (e.g. CartPole)
                              get their own subdirectory.
      YYYYMMDD_HHMMSS_.../ - Timestamp ensures uniqueness and chronological sorting.
                              Key hyperparams (sim, games, iter) in the name let you
                              identify runs at a glance without opening config files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_sites = cfg.game.kwargs.get("n_sites", 0)
    game_type = "bitflip" if cfg.game.kwargs.get("bit_flip", True) else "bitstring"
    reward_mode = "sparse" if cfg.game.kwargs.get("sparse_reward", True) else "dense"
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations
    dirname = f"{timestamp}_{game_type}{n_sites}_{reward_mode}_mcts{sim}_games{games}_iter{iters}"

    exp_dir = Path("experiments") / "bitstring" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Training output helpers
# ---------------------------------------------------------------------------

def print_banner(cfg, exp_dir):
    """Print startup banner with experiment info and output legend."""
    n_sites = cfg.game.kwargs.get("n_sites", "?")
    print()
    print("=" * 80)
    print("  BitString AlphaZero Training")
    print(f"  Goal: Learn to flip all bits to 1. State is a binary vector of length {n_sites}.")
    print(f"  Experiment dir: {exp_dir}/")
    print()
    print("  Output legend:")
    print("    [TRAIN]  Self-play data collection & network training")
    print("    [EVAL]   Pitting new network vs old network")
    print("    [ITER]   Iteration summary with key metrics")
    print("=" * 80)
    print()


def print_iteration_header(i, total):
    """Print iteration separator."""
    header = f"--- Iteration {i}/{total} "
    print(header + "-" * (80 - len(header)))


def print_iteration_summary(i, total, score, trainer_stats, evaluator_stats):
    """Print compact one-line iteration summary."""
    train_rec = trainer_stats.to_list()[-1] if trainer_stats.to_list() else {}
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
    print()


def rename_plot_with_stats(cfg, trainer_stats, evaluator_stats):
    """Rename the plot file to include game info and final stats in the filename."""
    plot_path = Path(cfg.run.plot_path)
    if not plot_path.exists():
        return str(plot_path)

    train_recs = trainer_stats.to_list()
    eval_recs = evaluator_stats.to_list()

    loss = train_recs[-1].get("train_loss", 0) if train_recs else 0
    new_mean = eval_recs[-1].get("new_rewards_mean", 0) if eval_recs else 0

    n_sites = cfg.game.kwargs.get("n_sites", 0)
    game_type = "bitflip" if cfg.game.kwargs.get("bit_flip", True) else "bitstring"
    reward_mode = "sparse" if cfg.game.kwargs.get("sparse_reward", True) else "dense"
    score_str = f"reward{new_mean:.2f}".replace(".", "p")
    loss_str = f"loss{loss:.2f}".replace(".", "p")
    new_name = plot_path.with_name(f"metrics_{game_type}{n_sites}_{reward_mode}_{score_str}_{loss_str}.png")

    plot_path.rename(new_name)
    return str(new_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = BitStringConfig()
    cfg.agent.random_seeds["mcts"] = 43
    cfg.agent.random_seeds["train"] = 47
    cfg.agent.random_seeds["eval"] = 23

    # Interactive config editing
    interactive_edit(cfg)

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg)
    cfg.save(str(exp_dir / "config.json"))

    # Build objects
    game, net, agent, trainer, evaluator = cfg.build()

    print_banner(cfg, exp_dir)

    breakpoint()
    # Training loop
    for i in range(cfg.run.n_iterations):
        print_iteration_header(i + 1, cfg.run.n_iterations)

        old_agent = copy.deepcopy(trainer.agent)
        trainer.train_iteration()
        new_agent = copy.deepcopy(trainer.agent)
        score = evaluator.pit(new_agent=new_agent, old_agent=old_agent)

        print_iteration_summary(
            i + 1, cfg.run.n_iterations, score,
            trainer.statistics_manager, evaluator.statistics_manager,
        )

        # if score >= cfg.run.accept_threshold:
        #     print("Keeping the new network")
        #     trainer.net = new_agent.net
        #     agent.net = new_agent.net
        # else:
        #     print("Reverting to the old network")
        #     trainer.net = old_agent.net
        #     agent.net = old_agent.net

        # Save training logs after each iteration
        trainer.statistics_manager.save_jsonl(str(exp_dir / "train_stats.jsonl"))
        evaluator.statistics_manager.save_jsonl(str(exp_dir / "eval_stats.jsonl"))

        if i % cfg.run.plot_every == 0:
            plot_training_metrics(
                trainer.statistics_manager,
                evaluator.statistics_manager,
                save_path=cfg.run.plot_path,
            )

    # Final plot with stats in filename
    plot_training_metrics(
        trainer.statistics_manager,
        evaluator.statistics_manager,
        save_path=cfg.run.plot_path,
    )
    final_plot = rename_plot_with_stats(cfg, trainer.statistics_manager, evaluator.statistics_manager)
    print(f"\nTraining complete. Results saved to: {exp_dir}/")
    print(f"  Config:     {exp_dir / 'config.json'}")
    print(f"  Plot:       {final_plot}")
    print(f"  Train log:  {exp_dir / 'train_stats.jsonl'}")
    print(f"  Eval log:   {exp_dir / 'eval_stats.jsonl'}")


def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager, save_path=None):
    """
    Plot training metrics.

    Visualization is essential for diagnosing training issues and comparing runs.
    Plots eval rewards, training losses, and training set size over iterations.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plot generation.")
        return

    trainer_history = trainer_stats_manager.to_list() if trainer_stats_manager else []
    evaluator_history = evaluator_stats_manager.to_list() if evaluator_stats_manager else []
    max_len = max(len(trainer_history), len(evaluator_history))
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
        merged.append(row)

    df = pd.DataFrame(merged)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("BitString AlphaZero Training Metrics", fontsize=14, fontweight='bold')

    # Plot 1: Eval reward mean with std band
    ax1 = axes[0, 0]
    if 'new_rewards_mean' in df.columns:
        ax1.plot(df['iteration'], df['new_rewards_mean'], 'b-', linewidth=2, label='New Reward Mean')
        if 'new_rewards_std' in df.columns:
            ax1.fill_between(
                df['iteration'],
                df['new_rewards_mean'] - df['new_rewards_std'],
                df['new_rewards_mean'] + df['new_rewards_std'],
                alpha=0.3, color='blue', label='+-1 Std'
            )
    if 'old_rewards_mean' in df.columns:
        ax1.plot(df['iteration'], df['old_rewards_mean'], 'c--', linewidth=2, label='Old Reward Mean')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Evaluation Rewards')
    if len(ax1.lines) > 0:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Policy and Value Loss
    ax2 = axes[0, 1]
    has_loss = False
    if 'train_loss_policy' in df.columns:
        ax2.plot(df['iteration'], df['train_loss_policy'], 'r-', linewidth=2, label='Policy Loss')
        has_loss = True
    if 'train_loss_value' in df.columns:
        ax2.plot(df['iteration'], df['train_loss_value'], 'g-', linewidth=2, label='Value Loss')
        has_loss = True
    if not has_loss and 'train_loss' in df.columns:
        ax2.plot(df['iteration'], df['train_loss'], 'k-', linewidth=2, label='Train Loss')
        has_loss = True
    if has_loss:
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.axis('off')

    # Plot 3: Training Set Size or Game Length (if available)
    ax3 = axes[1, 0]
    if 'num_examples' in df.columns:
        ax3.plot(df['iteration'], df['num_examples'], 'm-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Num Examples')
        ax3.set_title('Training Set Size')
    elif 'game_length' in df.columns:
        ax3.plot(df['iteration'], df['game_length'], 'm-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Avg Game Length')
        ax3.set_title('Average Episode Length')
    ax3.set_xlabel('Iteration')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined (normalized)
    ax4 = axes[1, 1]
    series_to_plot = []
    if 'new_rewards_mean' in df.columns and df['new_rewards_mean'].max() > 0:
        reward_norm = df['new_rewards_mean'] / df['new_rewards_mean'].max()
        series_to_plot.append((reward_norm, 'b-', 'Reward (norm)'))
    if 'num_examples' in df.columns and df['num_examples'].max() > 0:
        examples_norm = df['num_examples'] / df['num_examples'].max()
        series_to_plot.append((examples_norm, 'm-', 'Num Examples (norm)'))
    if 'train_loss_policy' in df.columns and df['train_loss_policy'].max() > 0:
        policy_norm = 1 - (df['train_loss_policy'] / df['train_loss_policy'].max())
        series_to_plot.append((policy_norm, 'r--', '1 - Policy Loss (norm)'))

    for series, style, label in series_to_plot:
        ax4.plot(df['iteration'], series, style, linewidth=2, label=label)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Combined Metrics (Normalized)')
    if series_to_plot:
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved to {save_path}")

    plt.close(fig)  # Close to free memory


if __name__ == "__main__":
    main()
