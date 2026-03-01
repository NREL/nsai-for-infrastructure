from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from alphazeropp.instances import BitStringConfig
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)

import copy
import torch
import logging
from alphazeropp.training.gated_trainer import GatedTrainer
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

def _build_sections(cfg):
    """Return sections for BitString AlphaZero config."""
    def _set_n_sites(val):
        cfg.game.kwargs["n_sites"] = val
        cfg.net.kwargs["n_sites"] = val

    def _set_sparse_reward(val):
        cfg.game.kwargs["sparse_reward"] = val
        if val:
            cfg.game.kwargs["fitness_fn"] = None

    params = [
        ("n_sites",      cfg.game.kwargs["n_sites"],      _set_n_sites,
         "Length of the binary vector", None),
        ("bit_flip",     cfg.game.kwargs["bit_flip"],      dict_setter(cfg.game.kwargs, "bit_flip"),
         "Actions flip bits (vs set bits)", None),
        ("sparse_reward", cfg.game.kwargs["sparse_reward"], _set_sparse_reward,
         "Reward only at episode end", None),
    ]

    if not cfg.game.kwargs.get("sparse_reward", False):
        params.append(
            ("fitness_fn", cfg.game.kwargs.get("fitness_fn", None),
             dict_setter(cfg.game.kwargs, "fitness_fn"),
             "0:None 1:leading_ones 2:binval",
             [None, "leading_ones", "binval"]))
        if cfg.game.kwargs.get("fitness_fn") is not None:
            params.append(
                ("reward_mode", cfg.game.kwargs.get("reward_mode", "dense_potential"),
                 dict_setter(cfg.game.kwargs, "reward_mode"),
                 "dense_potential or sparse_pm1", None))

    mcts_descs = {
        "n_simulations":    "MCTS rollouts per move",
        "temperature":      "Exploration temperature for action selection",
        "c_exploration":    "UCB exploration constant",
        "dirichlet_alpha":  "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
    }
    for k in ["n_simulations", "temperature", "c_exploration", "dirichlet_alpha", "dirichlet_epsilon"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k], dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], None))

    params.extend([
        ("reward_discount",   cfg.agent.reward_discount,
         attr_setter(cfg.agent, "reward_discount"),
         "Discount factor for future rewards", None),
        ("n_games_per_train", cfg.trainer.n_games_per_train,
         attr_setter(cfg.trainer, "n_games_per_train"),
         "Self-play games per training iteration", None),
        ("n_past_iters",      cfg.trainer.n_past_iterations_to_train,
         attr_setter(cfg.trainer, "n_past_iterations_to_train"),
         "Past iterations kept in training buffer", None),
        ("n_procs",           cfg.trainer.n_procs,
         attr_setter(cfg.trainer, "n_procs"),
         "Parallel workers for self-play", None),
        ("eval_n_games",      cfg.evaluator.n_games,
         attr_setter(cfg.evaluator, "n_games"),
         "Games to pit new vs old agent", None),
        ("eval_n_procs",      cfg.evaluator.n_procs,
         attr_setter(cfg.evaluator, "n_procs"),
         "Parallel workers for evaluation", None),
        ("n_iterations",      cfg.run.n_iterations,
         attr_setter(cfg.run, "n_iterations"),
         "Total training iterations", None),
        ("accept_threshold",  cfg.run.accept_threshold,
         attr_setter(cfg.run, "accept_threshold"),
         "Win rate to accept new network", None),
        ("plot_every",        cfg.run.plot_every,
         attr_setter(cfg.run, "plot_every"),
         "Plot metrics every N iterations", None),
    ])

    all_params = build_param_list(params)

    game_labels = {"n_sites", "bit_flip", "sparse_reward", "fitness_fn", "reward_mode"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration", "dirichlet_alpha", "dirichlet_epsilon"}
    agent_labels = {"reward_discount"}
    trainer_labels = {"n_games_per_train", "n_past_iters", "n_procs"}
    evaluator_labels = {"eval_n_games", "eval_n_procs"}
    run_labels = {"n_iterations", "accept_threshold", "plot_every"}

    return [
        ("Game",      [p for p in all_params if p[1] in game_labels]),
        ("MCTS",      [p for p in all_params if p[1] in mcts_labels]),
        ("Agent",     [p for p in all_params if p[1] in agent_labels]),
        ("Trainer",   [p for p in all_params if p[1] in trainer_labels]),
        ("Evaluator", [p for p in all_params if p[1] in evaluator_labels]),
        ("Run",       [p for p in all_params if p[1] in run_labels]),
    ]


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
    fitness = cfg.game.kwargs.get("fitness_fn") or "none"
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations
    dirname = f"{timestamp}_{game_type}{n_sites}_{reward_mode}_{fitness}_mcts{sim}_games{games}_iter{iters}"

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
    fitness = cfg.game.kwargs.get("fitness_fn") or "default (±1/N)"
    print()
    print("=" * 80)
    print("  BitString AlphaZero Training")
    print(f"  Goal: Learn to flip all bits to 1. State is a binary vector of length {n_sites}.")
    print(f"  Fitness function: {fitness}")
    print(f"  Experiment dir: {exp_dir}/")
    print()
    print("  Output legend:")
    print("    [TRAIN]  Self-play data collection & network training")
    print("    [EVAL]   Pitting new network vs old network")
    print("    [ITER]   Iteration summary with key metrics")
    print("=" * 80)
    print()


def print_architecture(cfg):
    """Print the AlphaZero training loop architecture diagram."""
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    gamma = cfg.agent.reward_discount
    n_sites = cfg.game.kwargs.get("n_sites", "?")

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero (Self-Play + Learning)")
    print("=" * 80)
    print()
    print(f"  Game: {n_sites}-bit binary vector.  Goal: flip all bits to 1.")
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  ┌── Iteration i ──────────────────────────────────────────────────────┐")
    print("  │                                                                      │")
    print(f"  │  STEP 1: Self-Play  ({n_games} games, each ~{n_sites} moves)")
    print("  │")
    print("  │    Game:  s₀ ──move──> s₁ ──move──> s₂ ──> ... ──> terminal")
    print("  │")
    print("  │    Each move creates a FRESH MCTS tree:")
    print(f"  │      mcts = MCTS(state, neural_net, n_sims={n_sims})")
    print("  │      ┌──────────────────────────────────────────────────────────┐")
    print(f"  │      │  for sim in 1..{n_sims}:")
    print("  │      │    SELECT  → traverse tree via UCB")
    print("  │      │              UCB = Q_norm(s,a) + c · π_net(a) · √N/(1+Nₐ)")
    print("  │      │    EXPAND  → hit new node, query net f_θ(s) → (π, v)")
    print("  │      │    BACKUP  → Q(s,a) = running_mean(R(s,a) + V(s'))")
    print("  │      └──────────────────────────────────────────────────────────┘")
    print("  │      π_MCTS = visit_counts^(1/τ) / Σ      ← distilled search result")
    print("  │      action ~ π_MCTS                       ← sample move")
    print("  │      SAVE (state, π_MCTS)                  ← becomes training target")
    print("  │      DISCARD tree                          ← net carries knowledge")
    print("  │")
    print(f"  │    After game ends (γ={gamma}):")
    print("  │      zₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...    (discounted return)")
    print("  │      Training examples: [(s₁, π₁, z₁), (s₂, π₂, z₂), ...]")
    print("  │")
    print(f"  │  STEP 2: Train Neural Net  (replay buffer: last {n_past} iterations)")
    print("  │")
    print("  │    L = (z - v_θ(s))²  −  π_MCTS · log p_θ(s)  +  c·‖θ‖²")
    print("  │        ─────────────     ─────────────────────     ───────")
    print("  │        value loss:        policy loss:             weight decay")
    print("  │        predict return     imitate MCTS")
    print("  │    θ ← θ − α · ∇L")
    print("  │")
    print("  │  STEP 3: Evaluate")
    print("  │    Pit new_net vs old_net.  Accept if win_rate ≥ threshold.")
    print("  │                                                                      │")
    print("  └──────────────────────────────────────────────────────────────────────┘")
    print()
    print("  Knowledge flow:")
    print("    Trees are DISPOSABLE.  They produce training data (s, π_MCTS, z).")
    print("    Neural net f_θ is the KNOWLEDGE CARRIER across iterations.")
    print("    Better net → better MCTS → better data → better net (virtuous cycle)")
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
    fitness = cfg.game.kwargs.get("fitness_fn") or "none"
    score_str = f"reward{new_mean:.2f}".replace(".", "p")
    loss_str = f"loss{loss:.2f}".replace(".", "p")
    new_name = plot_path.with_name(f"metrics_{game_type}{n_sites}_{reward_mode}_{fitness}_{score_str}_{loss_str}.png")

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
    interactive_edit("BitString Config", lambda: _build_sections(cfg))

    # Setup experiment directory
    exp_dir = setup_experiment_dir(cfg)
    cfg.save(str(exp_dir / "config.json"))

    # Build objects
    game, net, agent, trainer, evaluator = cfg.build()

    print_banner(cfg, exp_dir)
    print_architecture(cfg)

    n_sites = cfg.game.kwargs["n_sites"]
    n_ones = 2  # Hardcoded in BitStringGym.__init__
    optimal_reward = (n_sites - n_ones) / n_sites

    # Training loop
    gated_trainer = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold)
    for i in range(cfg.run.n_iterations):
        print_iteration_header(i + 1, cfg.run.n_iterations)

        score, accepted = gated_trainer.train_iteration()

        print_iteration_summary(
            i + 1, cfg.run.n_iterations, score,
            trainer.statistics_manager, evaluator.statistics_manager,
        )

        # Save training logs after each iteration
        trainer.statistics_manager.save_jsonl(str(exp_dir / "train_stats.jsonl"))
        evaluator.statistics_manager.save_jsonl(str(exp_dir / "eval_stats.jsonl"))

        if i % cfg.run.plot_every == 0:
            plot_training_metrics(
                trainer.statistics_manager,
                evaluator.statistics_manager,
                save_path=cfg.run.plot_path,
                optimal_reward=optimal_reward,
            )

    # Final plot with stats in filename
    plot_training_metrics(
        trainer.statistics_manager,
        evaluator.statistics_manager,
        save_path=cfg.run.plot_path,
        optimal_reward=optimal_reward,
    )
    final_plot = rename_plot_with_stats(cfg, trainer.statistics_manager, evaluator.statistics_manager)
    print(f"\nTraining complete. Results saved to: {exp_dir}/")
    print(f"  Config:     {exp_dir / 'config.json'}")
    print(f"  Plot:       {final_plot}")
    print(f"  Train log:  {exp_dir / 'train_stats.jsonl'}")
    print(f"  Eval log:   {exp_dir / 'eval_stats.jsonl'}")


def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager, save_path=None, optimal_reward=None):
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
    if optimal_reward is not None:
        ax1.axhline(y=optimal_reward, color='green', linestyle='--', linewidth=1.5, label=f'Optimal ({optimal_reward:.3f})')
    if 'new_rewards_mean' in df.columns:
        ax1.plot(df['iteration'], df['new_rewards_mean'], 'b-', linewidth=2, label='New Reward Mean')
        if 'new_rewards_std' in df.columns:
            std_upper = df['new_rewards_mean'] + df['new_rewards_std']
            std_lower = df['new_rewards_mean'] - df['new_rewards_std']
            if optimal_reward is not None:
                std_upper = np.minimum(std_upper, optimal_reward)
            ax1.fill_between(
                df['iteration'],
                std_lower,
                std_upper,
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
