"""
Shared utilities for derivation-based program synthesis training scripts.

Domain-agnostic functions used by both run_bitstring_derivation.py and
run_doors_derivation.py.
"""

from __future__ import annotations

import json
from pathlib import Path


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
    from alphazeropp.synthesis.interpreter import (
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
    metric = cfg.game.kwargs["metric"]
    sr = best_metrics.get("solve_rate", 0) if best_metrics else 0
    sr_str = f"sr{sr:.0%}".replace("%", "pct")
    reward_str = f"reward{new_mean:.2f}".replace(".", "p")
    budget = cfg.game.kwargs.get("budget")
    budget_str = f"_L{budget}" if budget is not None else ""
    new_name = plot_path.with_name(
        f"metrics_N{n_sites}{budget_str}_{metric}_{sr_str}_{reward_str}.png"
    )

    plot_path.rename(new_name)
    return str(new_name)


def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager,
                          program_log, save_path=None, optimal_reward=None):
    """Plot training metrics with derivation-specific best-program tracking.

    Four panels:
      1. Best Program Quality (solve rate + avg reward)
      2. Training Losses (policy + value)
      3. Search Progress (unique programs explored)
      4. Self-Play Reward & Gate (avg reward from self-play + gate decisions)
    """
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
            row["unique_programs"] = program_log[i].get("unique_programs", 0)
        merged.append(row)

    df = pd.DataFrame(merged)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("DerivationGame AlphaZero Training Metrics",
                 fontsize=14, fontweight="bold")

    # Panel 1 (top-left): Best Program Quality — reward with solved markers
    ax1 = axes[0, 0]
    if "best_avg_reward" in df.columns:
        ax1.plot(df["iteration"], df["best_avg_reward"],
                 "b-", linewidth=2, label="Best Avg Reward")
        # Color-code markers: green when solved, red when not
        if "best_solve_rate" in df.columns:
            iters = df["iteration"].values
            rewards = df["best_avg_reward"].values
            solved = df["best_solve_rate"].values >= 1.0
            if solved.any():
                ax1.scatter(iters[solved], rewards[solved],
                            color="green", s=40, zorder=5, label="Solved")
            if (~solved).any():
                ax1.scatter(iters[~solved], rewards[~solved],
                            color="red", s=40, zorder=5, label="Not Solved")
    if optimal_reward is not None:
        ax1.axhline(y=optimal_reward, color="green", linestyle=":",
                     linewidth=1, alpha=0.5, label=f"Optimal ({optimal_reward})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Avg Reward")
    ax1.set_title("Best Program Quality")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # Panel 2 (top-right): Training Losses
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
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")
    else:
        ax2.axis("off")

    # Panel 3 (bottom-left): Search Progress
    ax3 = axes[1, 0]
    if "unique_programs" in df.columns:
        ax3.plot(df["iteration"], df["unique_programs"],
                 "b-", linewidth=2, marker="o", markersize=4,
                 label="Total Unique Programs")
        ax3.set_ylabel("Total Unique Programs", color="blue")
        # New programs per iteration on secondary axis
        unique = df["unique_programs"].values
        new_per_iter = [unique[0]] + [
            unique[j] - unique[j - 1] for j in range(1, len(unique))
        ]
        ax3_r = ax3.twinx()
        ax3_r.bar(df["iteration"], new_per_iter, alpha=0.3, color="gray",
                  label="New Programs / Iter")
        ax3_r.set_ylabel("New Programs / Iter", color="gray")
        # Combine legends
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines3r, labels3r = ax3_r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labels3 + labels3r, loc="best", fontsize=8)
    elif "num_examples" in df.columns:
        # Fallback if program_log has no unique_programs
        ax3.plot(df["iteration"], df["num_examples"],
                 "m-", linewidth=2, marker="o", markersize=4)
        ax3.set_ylabel("Num Examples")
    ax3.set_xlabel("Iteration")
    ax3.set_title("Search Progress")
    ax3.grid(True, alpha=0.3)

    # Panel 4 (bottom-right): Self-Play Reward & Gate
    ax4 = axes[1, 1]
    has_data = False
    # Background shading for gate accept/reject decisions
    if "gate_accepted" in df.columns:
        from matplotlib.patches import Patch
        accepted = df["gate_accepted"].values
        iters = df["iteration"].values
        has_accept = False
        has_reject = False
        for j in range(len(accepted)):
            if accepted[j]:
                ax4.axvspan(iters[j] - 0.5, iters[j] + 0.5,
                            alpha=0.25, color="#d4edda", zorder=0)
                has_accept = True
            else:
                ax4.axvspan(iters[j] - 0.5, iters[j] + 0.5,
                            alpha=0.25, color="#f8d7da", zorder=0)
                has_reject = True
    if "avg_reward" in df.columns:
        ax4.plot(df["iteration"], df["avg_reward"],
                 "b-", linewidth=2, label="Self-Play Avg Reward")
        has_data = True
    if "gate_score" in df.columns:
        ax4_r = ax4.twinx()
        ax4_r.plot(df["iteration"], df["gate_score"],
                   color="orange", linestyle="--", linewidth=1.5,
                   label="Gate Score")
        ax4_r.set_ylim(-0.05, 1.05)
        ax4_r.set_ylabel("Gate Score", color="orange")
        has_data = True
    if not has_data:
        # Fallback: show eval rewards if no self-play/gate data
        if "new_rewards_mean" in df.columns:
            ax4.plot(df["iteration"], df["new_rewards_mean"],
                     "b-", linewidth=2, label="New Reward Mean")
        if "old_rewards_mean" in df.columns:
            ax4.plot(df["iteration"], df["old_rewards_mean"],
                     "c--", linewidth=2, label="Old Reward Mean")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Reward")
    ax4.set_title("Self-Play Reward & Gate")
    # Build legend with gate band entries
    lines4, labels4 = ax4.get_legend_handles_labels()
    if "gate_score" in df.columns:
        lines4r, labels4r = ax4_r.get_legend_handles_labels()
        lines4 = lines4 + lines4r
        labels4 = labels4 + labels4r
    if "gate_accepted" in df.columns:
        if has_accept:
            lines4.append(Patch(facecolor="#d4edda", alpha=0.5))
            labels4.append("Gate Accepted")
        if has_reject:
            lines4.append(Patch(facecolor="#f8d7da", alpha=0.5))
            labels4.append("Gate Rejected")
    if lines4:
        ax4.legend(lines4, labels4, loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved to {save_path}")

    plt.close(fig)


def run_derivation_training(cfg, mode, optimal_reward, exp_dir,
                            iteration_callback=None):
    """Shared training loop for derivation-based program synthesis.

    Args:
        cfg: MetaConfig (DerivationConfig, ScanDerivationConfig,
             DoorsDerivationConfig, etc.)
        mode: String label for logging (e.g. "scan", "cfg", "doors")
        optimal_reward: Known optimal reward for the target domain.
        exp_dir: Path to experiment output directory.
        iteration_callback: Optional callable(iteration, program_log_entry,
            gate_score, gate_accepted, wall_clock_s) invoked after each
            iteration for external metric capture.
    """
    import json
    import logging
    import time
    from pathlib import Path
    from alphazeropp.training.gated_trainer import GatedTrainer

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build objects
    game, net, agent, trainer, evaluator = cfg.build()

    # Access LeafEvaluator for program tracking
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

        t_iter_start = time.time()
        score, accepted = gated_trainer.train_iteration()
        iter_wall_clock = time.time() - t_iter_start

        # Extract best program from leaf_eval cache
        prog_str, prog_ast, prog_metrics, prog_value = extract_best_program(
            leaf_eval)
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

        # Merge leaf_stats into trainer statistics
        if trainer.statistics_manager._records:
            trainer.statistics_manager._records[-1]["leaf_stats"] = (
                leaf_eval.stats()
            )

        # Write per-step diagnostics from self-play
        diag_records = []
        for ep_idx, ep_infos in enumerate(trainer._last_diagnostics):
            for step_idx, step_info in enumerate(ep_infos):
                diag_records.append({
                    "iteration": i + 1,
                    "episode_id": ep_idx,
                    "deriv_step": step_idx + 1,
                    "branching": step_info.get("branching"),
                    "hole_type": step_info.get("hole_type"),
                    "hole_budget": step_info.get("hole_budget"),
                    "is_complete": step_info.get("is_complete"),
                    "is_dead_end": step_info.get("is_dead_end"),
                    "leaf_value": step_info.get("leaf_value"),
                })
        if diag_records:
            with open(exp_dir / "diagnostics.jsonl", "a") as f:
                for rec in diag_records:
                    f.write(json.dumps(rec) + "\n")

        # Save logs after each iteration
        trainer.statistics_manager.save_jsonl(
            str(exp_dir / "train_stats.jsonl"))
        evaluator.statistics_manager.save_jsonl(
            str(exp_dir / "eval_stats.jsonl"))
        with open(exp_dir / "program_log.jsonl", "w") as f:
            for entry in program_log:
                f.write(json.dumps(entry) + "\n")

        if iteration_callback:
            iteration_callback(
                i + 1, program_log[-1], score, accepted, iter_wall_clock,
            )

        if (i + 1) % cfg.run.plot_every == 0 or i == 0:
            plot_training_metrics(
                trainer.statistics_manager,
                evaluator.statistics_manager,
                program_log,
                save_path=cfg.run.plot_path,
                optimal_reward=optimal_reward,
            )


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
    print(f"  Diagnostics:  {exp_dir / 'diagnostics.jsonl'}")
    print(f"  Programs evaluated: {leaf_eval.stats()['unique_programs']}")
