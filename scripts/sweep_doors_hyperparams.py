"""
Hyperparameter sweep for Doors Direct Play.

Systematically identifies which hyperparameters enable learning by running
phased sweeps: MCTS budget, network architecture, learning dynamics, and
MCTS exploration tuning.  Phase configs scale automatically with difficulty.

Usage:
    python scripts/sweep_doors_hyperparams.py                      # D=6 (default)
    python scripts/sweep_doors_hyperparams.py --num-rooms 10       # D=10
    python scripts/sweep_doors_hyperparams.py --phase 1            # MCTS + data only
    python scripts/sweep_doors_hyperparams.py --num-rooms 10 --phase 1
    python scripts/sweep_doors_hyperparams.py --resume <dir>       # resume sweep
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.config import DoorsDirectConfig, compute_dims
from alphazeropp.training.gated_trainer import GatedTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep phase definitions
# ---------------------------------------------------------------------------

# Baseline: D=6, L=3, all defaults
BASELINE = {
    "num_rooms": 6,
    "locs_per_room": 3,
    "n_iterations": 30,
    "n_simulations": 20,
    "n_games_per_train": 20,
    "n_hidden_layers": 1,
    "hidden_size": None,  # computed from obs_size
    "learning_rate": 3e-4,
    "epochs": 5,
    "policy_weight": 2.0,
    "c_exploration": 1.5,
    "dirichlet_alpha": 0.25,
    "dirichlet_epsilon": 0.40,
    "accept_threshold": 0.55,
}


def _phase1_configs(num_rooms=6):
    """MCTS budget x data volume.  Sim budgets scale with difficulty."""
    if num_rooms <= 6:
        sims_list = [50, 100, 200]
    elif num_rooms <= 10:
        sims_list = [100, 200, 400]
    else:
        sims_list = [200, 400, 800]
    configs = []
    for sims in sims_list:
        for games in [50, 100]:
            name = f"phase1_sims{sims}_games{games}"
            configs.append((name, {"n_simulations": sims,
                                   "n_games_per_train": games}))
    return configs


def _phase2_configs(obs_size=29):
    """Network architecture.  Hidden sizes scale with observation size."""
    if obs_size <= 30:
        options = [(1, 128), (2, 128), (2, 256), (3, 128)]
    else:
        options = [(1, 256), (2, 256), (2, 512), (3, 256)]
    configs = []
    for layers, hidden in options:
        name = f"phase2_layers{layers}_hidden{hidden}"
        configs.append((name, {"n_hidden_layers": layers,
                               "hidden_size": hidden}))
    return configs


def _phase3_configs():
    """Learning dynamics."""
    configs = []
    for lr, epochs, pw in [
        (1e-4, 10, 2.0),
        (3e-4, 10, 2.0),
        (1e-3, 10, 2.0),
        (3e-4, 20, 2.0),
        (3e-4, 10, 1.0),
        (3e-4, 10, 4.0),
    ]:
        name = f"phase3_lr{lr}_ep{epochs}_pw{pw}"
        configs.append((name, {"learning_rate": lr, "epochs": epochs,
                               "policy_weight": pw}))
    return configs


def _phase4_configs():
    """MCTS exploration tuning."""
    configs = []
    for c_exp, d_alpha in [(1.0, 0.25), (2.0, 0.25), (1.5, 0.10), (1.5, 0.50)]:
        name = f"phase4_c{c_exp}_alpha{d_alpha}"
        configs.append((name, {"c_exploration": c_exp,
                               "dirichlet_alpha": d_alpha}))
    return configs


def _make_phases(num_rooms=6, obs_size=29):
    """Build phase registry scaled to difficulty."""
    return {
        1: ("MCTS Budget x Data Volume",
            lambda: _phase1_configs(num_rooms)),
        2: ("Network Architecture",
            lambda: _phase2_configs(obs_size)),
        3: ("Learning Dynamics", _phase3_configs),
        4: ("MCTS Exploration Tuning", _phase4_configs),
    }

# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(overrides: dict) -> DoorsDirectConfig:
    """Create a DoorsDirectConfig with programmatic overrides."""
    params = {**BASELINE, **overrides}

    nr = params["num_rooms"]
    lpr = params["locs_per_room"]
    dims = compute_dims(nr, lpr)
    obs_size = dims["obs_size"]

    cfg = DoorsDirectConfig(num_rooms=nr, locs_per_room=lpr)

    # Network
    hidden = params["hidden_size"] if params["hidden_size"] else max(64, obs_size * 4)
    cfg.net.kwargs["n_hidden_layers"] = params["n_hidden_layers"]
    cfg.net.kwargs["hidden_size"] = hidden
    cfg.net.kwargs["training_params"] = {
        "epochs": params["epochs"],
        "batch_size": 32,
        "learning_rate": params["learning_rate"],
        "weight_decay": 1e-4,
        "policy_weight": params["policy_weight"],
    }

    # MCTS
    cfg.agent.mcts_params["n_simulations"] = params["n_simulations"]
    cfg.agent.mcts_params["c_exploration"] = params["c_exploration"]
    cfg.agent.mcts_params["dirichlet_alpha"] = params["dirichlet_alpha"]
    cfg.agent.mcts_params["dirichlet_epsilon"] = params["dirichlet_epsilon"]

    # Trainer
    cfg.trainer.n_games_per_train = params["n_games_per_train"]
    cfg.trainer.n_procs = -1

    # Run
    cfg.run.n_iterations = params["n_iterations"]
    cfg.run.accept_threshold = params["accept_threshold"]

    return cfg


# ---------------------------------------------------------------------------
# Solve-rate evaluation (adapted from run_doors_direct.py)
# ---------------------------------------------------------------------------

def compute_solve_rate(agent, n_episodes=20):
    """Play episodes greedily and compute solve rate + avg reward."""
    solves = 0
    total_reward = 0.0

    for i in range(n_episodes):
        game = agent.game.clone()
        game.reset_wrapper(seed=1000 + i)
        cumulative = 0.0

        for _ in range(game.env.horizon):
            probs = agent.policy(game, "", add_noise=False,
                                 temperature_override=0.01)
            action = int(np.argmax(probs))
            _, reward, terminated, truncated, _ = game.step_wrapper(action)
            cumulative += reward
            if terminated or truncated:
                break

        if game.env.is_solved(game.obs):
            solves += 1
        total_reward += cumulative

    return solves / n_episodes, total_reward / n_episodes


# ---------------------------------------------------------------------------
# Single-config training run
# ---------------------------------------------------------------------------

def run_config(name: str, overrides: dict, run_dir: Path) -> dict:
    """Run one configuration and return summary metrics."""
    config_dir = run_dir / name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    summary_file = config_dir / "summary.json"
    if summary_file.exists():
        logger.info("  [SKIP] %s (already completed)", name)
        with open(summary_file) as f:
            return json.load(f)

    cfg = build_config(overrides)
    cfg.trainer.checkpoint_dir = str(config_dir / "checkpoints")

    # Save config
    all_params = {**BASELINE, **overrides}
    with open(config_dir / "config.json", "w") as f:
        json.dump(all_params, f, indent=2)

    # Build pipeline
    game, net, agent, trainer, evaluator = cfg.build()
    gated = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold)

    # Training loop
    iteration_log = []
    best_solve_rate = 0.0
    n_accepted = 0
    t_total = time.time()

    for i in range(cfg.run.n_iterations):
        t0 = time.time()
        score, accepted = gated.train_iteration()
        elapsed = time.time() - t0

        if accepted:
            n_accepted += 1

        solve_rate, avg_reward = compute_solve_rate(agent, n_episodes=20)
        best_solve_rate = max(best_solve_rate, solve_rate)

        # Extract training losses
        train_records = trainer.statistics_manager.to_list()
        last_train = train_records[-1] if train_records else {}
        p_loss = last_train.get("train_loss_policy", float("nan"))
        v_loss = last_train.get("train_loss_value", float("nan"))

        entry = {
            "iteration": i + 1,
            "gate_score": float(score),
            "accepted": bool(accepted),
            "solve_rate": solve_rate,
            "avg_reward": avg_reward,
            "best_solve_rate": best_solve_rate,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "wall_clock_s": round(elapsed, 1),
        }
        iteration_log.append(entry)

        marker = "ACC" if accepted else "REJ"
        logger.info(
            "  [%s] iter %2d/%d  gate=%.0f%% [%s]  solve=%.2f  "
            "reward=%+.3f  P:%.3f V:%.3f  %.1fs",
            name, i + 1, cfg.run.n_iterations,
            score * 100, marker, solve_rate, avg_reward,
            p_loss, v_loss, elapsed,
        )

        # Early stopping
        if solve_rate >= 1.0:
            logger.info("  [%s] EARLY STOP: solve_rate=1.0 at iter %d", name, i + 1)
            break

    total_time = time.time() - t_total

    # Save iteration log
    with open(config_dir / "iteration_log.jsonl", "w") as f:
        for e in iteration_log:
            f.write(json.dumps(e) + "\n")
    trainer.statistics_manager.save_jsonl(str(config_dir / "train_stats.jsonl"))
    evaluator.statistics_manager.save_jsonl(str(config_dir / "eval_stats.jsonl"))

    # Build summary
    final = iteration_log[-1] if iteration_log else {}
    summary = {
        "name": name,
        **all_params,
        "best_solve_rate": best_solve_rate,
        "final_solve_rate": final.get("solve_rate", 0),
        "final_avg_reward": final.get("avg_reward", 0),
        "final_policy_loss": final.get("policy_loss", float("nan")),
        "final_value_loss": final.get("value_loss", float("nan")),
        "n_accepted": n_accepted,
        "n_iters_run": len(iteration_log),
        "total_wall_clock_s": round(total_time, 1),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "name", "n_simulations", "n_games_per_train",
    "n_hidden_layers", "hidden_size",
    "learning_rate", "epochs", "policy_weight",
    "c_exploration", "dirichlet_alpha",
    "best_solve_rate", "final_avg_reward",
    "final_policy_loss", "final_value_loss",
    "n_accepted", "n_iters_run", "total_wall_clock_s",
]


def write_summary_csv(summaries: list[dict], path: Path):
    """Write summary CSV sorted by best_solve_rate desc, then final_avg_reward desc."""
    ranked = sorted(summaries,
                    key=lambda s: (-s.get("best_solve_rate", 0),
                                   -s.get("final_avg_reward", -999)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)


def print_summary_table(summaries: list[dict]):
    """Print a compact results table to terminal."""
    ranked = sorted(summaries,
                    key=lambda s: (-s.get("best_solve_rate", 0),
                                   -s.get("final_avg_reward", -999)))
    sep = "=" * 100
    print(f"\n{sep}")
    print("  SWEEP RESULTS (ranked by best solve rate)")
    print(sep)
    print(f"  {'Config':<35} {'Solve':>6} {'Reward':>8} "
          f"{'P-Loss':>7} {'V-Loss':>7} {'Acc':>4} {'Time':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*4} {'-'*7}")

    for s in ranked:
        sr = s.get("best_solve_rate", 0)
        rw = s.get("final_avg_reward", 0)
        pl = s.get("final_policy_loss", float("nan"))
        vl = s.get("final_value_loss", float("nan"))
        acc = s.get("n_accepted", 0)
        t = s.get("total_wall_clock_s", 0)
        print(f"  {s['name']:<35} {sr:>5.0%} {rw:>+8.3f} "
              f"{pl:>7.3f} {vl:>7.4f} {acc:>4} {t:>6.0f}s")

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette for distinguishing configs
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _load_iteration_log(config_dir: Path) -> list[dict]:
    """Load iteration_log.jsonl from a config directory."""
    path = config_dir / "iteration_log.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_phase_learning_curves(configs: list[tuple[str, dict]],
                               sweep_dir: Path, phase_num: int,
                               phase_name: str):
    """Plot solve rate, reward, and losses over iterations for one phase.

    Produces a 2x2 figure with:
      - Top-left:  Solve rate over iterations
      - Top-right: Avg reward over iterations
      - Bot-left:  Policy loss over iterations
      - Bot-right: Value loss over iterations
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Phase {phase_num}: {phase_name}", fontsize=14, fontweight="bold")

    for idx, (name, _overrides) in enumerate(configs):
        log = _load_iteration_log(sweep_dir / name)
        if not log:
            continue

        color = _COLORS[idx % len(_COLORS)]
        iters = [e["iteration"] for e in log]
        short_name = name.replace(f"phase{phase_num}_", "")

        # Solve rate
        axes[0, 0].plot(iters, [e["solve_rate"] for e in log],
                        color=color, linewidth=2, marker="o", markersize=3,
                        label=short_name)

        # Avg reward
        axes[0, 1].plot(iters, [e["avg_reward"] for e in log],
                        color=color, linewidth=2, label=short_name)

        # Policy loss
        p_losses = [e.get("policy_loss", float("nan")) for e in log]
        axes[1, 0].plot(iters, p_losses, color=color, linewidth=2,
                        label=short_name)

        # Value loss
        v_losses = [e.get("value_loss", float("nan")) for e in log]
        axes[1, 1].plot(iters, v_losses, color=color, linewidth=2,
                        label=short_name)

    for ax, title, ylabel in [
        (axes[0, 0], "Solve Rate", "Solve Rate"),
        (axes[0, 1], "Average Reward", "Reward"),
        (axes[1, 0], "Policy Loss", "Loss"),
        (axes[1, 1], "Value Loss", "Loss"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylim(-0.05, 1.1)
    axes[0, 0].axhline(y=1.0, color="green", linestyle=":", alpha=0.4)

    plt.tight_layout()
    save_path = sweep_dir / f"phase{phase_num}_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Phase %d curves saved to %s", phase_num, save_path)


def plot_sweep_summary(summaries: list[dict], sweep_dir: Path):
    """Plot a summary bar chart comparing all configs.

    Produces a figure with:
      - Left:  Horizontal bar chart of best solve rate per config
      - Right: Horizontal bar chart of final avg reward per config
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    if not summaries:
        return

    ranked = sorted(summaries,
                    key=lambda s: (-s.get("best_solve_rate", 0),
                                   -s.get("final_avg_reward", -999)))

    names = [s["name"] for s in ranked]
    solve_rates = [s.get("best_solve_rate", 0) for s in ranked]
    rewards = [s.get("final_avg_reward", 0) for s in ranked]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.35)))
    fig.suptitle("Sweep Summary: All Configs Ranked", fontsize=14, fontweight="bold")

    y_pos = np.arange(len(names))

    # Solve rate bars
    colors1 = ["#2ca02c" if sr > 0 else "#d62728" for sr in solve_rates]
    ax1.barh(y_pos, solve_rates, color=colors1, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Best Solve Rate")
    ax1.set_title("Best Solve Rate")
    ax1.set_xlim(0, 1.1)
    ax1.axvline(x=1.0, color="green", linestyle=":", alpha=0.4)
    ax1.invert_yaxis()
    ax1.grid(True, axis="x", alpha=0.3)

    # Reward bars
    colors2 = ["#1f77b4" if rw > 0 else "#ff7f0e" for rw in rewards]
    ax2.barh(y_pos, rewards, color=colors2, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Final Avg Reward")
    ax2.set_title("Final Average Reward")
    ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = sweep_dir / "sweep_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Sweep summary saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Doors hyperparameter sweep")
    parser.add_argument("--num-rooms", type=int, default=None,
                        help="Number of rooms (default: 6)")
    parser.add_argument("--locs-per-room", type=int, default=None,
                        help="Locations per room (default: 3)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Run only this phase (default: all)")
    parser.add_argument("--resume", type=str,
                        help="Resume from existing sweep directory")
    parser.add_argument("--override", type=str, action="append", default=[],
                        help="Override baseline param: key=value (e.g. --override n_simulations=100)")
    args = parser.parse_args()

    # Parse overrides that apply to ALL configs in later phases
    base_overrides = {}
    for ov in args.override:
        k, v = ov.split("=", 1)
        # Auto-cast numeric values
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        base_overrides[k] = v

    # Determine difficulty (CLI flags > --override > BASELINE default)
    num_rooms = (args.num_rooms
                 or base_overrides.pop("num_rooms", None)
                 or BASELINE["num_rooms"])
    lpr = (args.locs_per_room
           or base_overrides.pop("locs_per_room", None)
           or BASELINE["locs_per_room"])
    BASELINE["num_rooms"] = num_rooms
    BASELINE["locs_per_room"] = lpr

    # Scale baseline MCTS budget and iterations with difficulty
    if num_rooms <= 6:
        BASELINE["n_simulations"] = 50
        BASELINE["n_games_per_train"] = 50
        BASELINE["n_iterations"] = 30
    elif num_rooms <= 10:
        BASELINE["n_simulations"] = 100
        BASELINE["n_games_per_train"] = 50
        BASELINE["n_iterations"] = 50
    else:
        BASELINE["n_simulations"] = 200
        BASELINE["n_games_per_train"] = 100
        BASELINE["n_iterations"] = 80

    dims = compute_dims(num_rooms, lpr)
    obs_size = dims["obs_size"]
    logger.info("Difficulty: D=%d L=%d  obs_size=%d  horizon=%d",
                num_rooms, lpr, obs_size, dims["horizon"])

    # Build phases scaled to difficulty
    phases = _make_phases(num_rooms, obs_size)

    # Setup sweep directory
    if args.resume:
        sweep_dir = Path(args.resume)
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = Path("experiments") / "doors_sweep" / f"{timestamp}_D{num_rooms}_L{lpr}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Sweep directory: %s", sweep_dir)

    # Collect configs for selected phases
    phases_to_run = [args.phase] if args.phase else sorted(phases.keys())
    phase_configs = {}  # phase_num -> list of (name, overrides)
    for p in phases_to_run:
        phase_name, config_fn = phases[p]
        configs = config_fn()
        phase_configs[p] = configs
        logger.info("Phase %d: %s (%d configs)", p, phase_name, len(configs))

    # Run sweep phase by phase
    summaries = []
    global_idx = 0
    total_configs = sum(len(c) for c in phase_configs.values())

    for p in phases_to_run:
        phase_name = phases[p][0]
        configs = phase_configs[p]

        for name, overrides in configs:
            global_idx += 1
            merged = {**base_overrides, **overrides}
            logger.info("\n[%d/%d] Running: %s", global_idx, total_configs, name)
            logger.info("  Overrides: %s", merged)

            summary = run_config(name, merged, sweep_dir)
            summaries.append(summary)

            # Update summary CSV after each config
            write_summary_csv(summaries, sweep_dir / "summary.csv")

        # Plot learning curves for this phase
        plot_phase_learning_curves(configs, sweep_dir, p, phase_name)

    # Final summary plot + table
    plot_sweep_summary(summaries, sweep_dir)
    print_summary_table(summaries)
    logger.info("Results saved to: %s/summary.csv", sweep_dir)


if __name__ == "__main__":
    main()
