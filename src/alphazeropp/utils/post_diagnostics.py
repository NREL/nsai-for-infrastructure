"""
Post-experiment diagnostic plots for AlphaZero program synthesis.

Generates diagnostic report from experiment JSONL logs.
Can be used standalone via scripts/run_post_diagnostics.py or
auto-called after training via derivation_utils.run_derivation_training().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: Path) -> dict:
    """Load a regular JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_experiment(exp_dir: Path) -> dict:
    """Load all data from an experiment directory."""
    return {
        "dir": exp_dir,
        "config": _load_json(exp_dir / "config.json"),
        "program_log": _load_jsonl(exp_dir / "program_log.jsonl"),
        "train_stats": _load_jsonl(exp_dir / "train_stats.jsonl"),
        "eval_stats": _load_jsonl(exp_dir / "eval_stats.jsonl"),
        "diagnostics": _load_jsonl(exp_dir / "diagnostics.jsonl"),
    }


def _extract_per_iter_leaf_stats(train_stats: list[dict]) -> list[dict]:
    """Extract per-iteration delta leaf_stats from cumulative train_stats."""
    deltas = []
    prev = {"cache_hits": 0, "eval_count": 0, "unique_programs": 0,
            "total_env_steps": 0, "total_interp_ops": 0}
    for rec in train_stats:
        ls = rec.get("leaf_stats", {})
        delta = {}
        for k in prev:
            curr = ls.get(k, 0)
            delta[k] = curr - prev[k]
            prev[k] = curr
        deltas.append(delta)
    return deltas


def _extract_branching_by_iter(diagnostics: list[dict]) -> dict:
    """Group branching factors by iteration."""
    by_iter = {}
    for rec in diagnostics:
        it = rec.get("iteration", 0)
        br = rec.get("branching")
        if br is not None and br > 0:
            by_iter.setdefault(it, []).append(br)
    return by_iter


def run_post_diagnostics(
    exp_dir: Path | str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """Generate diagnostic report for a single experiment.

    Args:
        exp_dir: Path to experiment directory containing JSONL files.
        save_path: Override save location (default: {exp_dir}/diagnostics_report.png).
        show: If True, display plot interactively.

    Returns:
        Path to saved diagnostic report image.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_dir = Path(exp_dir)
    data = _load_experiment(exp_dir)
    prog_log = data["program_log"]
    train_stats = data["train_stats"]
    diagnostics = data["diagnostics"]

    if not prog_log or not train_stats:
        print(f"[Diagnostics] No data found in {exp_dir}")
        return ""

    n_iters = len(prog_log)
    iters = np.arange(1, n_iters + 1)

    # Extract per-iteration deltas
    leaf_deltas = _extract_per_iter_leaf_stats(train_stats)
    branching_by_iter = _extract_branching_by_iter(diagnostics)

    # --- Build figure ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Derive experiment label from directory name
    dir_name = exp_dir.name
    # Extract D=X from directory name
    d_match = [p for p in dir_name.split("_") if p.startswith("D")]
    exp_label = d_match[0] if d_match else dir_name[:30]

    # ---- Panel 1: Reward distribution (self-play avg_reward) ----
    ax = axes[0, 0]
    rewards = [r.get("avg_reward", 0) for r in train_stats]
    ax.hist(rewards, bins=max(10, n_iters // 2), color="#1f77b4",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Self-play avg_reward")
    ax.set_ylabel("Count (iterations)")
    ax.set_title("Self-Play Reward Distribution")
    ax.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=1,
               label=f"mean={np.mean(rewards):.4f}")
    reward_range = max(rewards) - min(rewards) if len(rewards) > 1 else 0
    ax.text(0.95, 0.85, f"range={reward_range:.4f}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    ax.legend(fontsize=8)

    # ---- Panel 2: Exploration rate ----
    ax = axes[0, 1]
    unique = [r.get("unique_programs", 0) for r in prog_log]
    new_per_iter = [unique[0]] + [unique[i] - unique[i-1]
                                   for i in range(1, len(unique))]
    ax.bar(iters, new_per_iter, color="#aaaaaa", edgecolor="white", label="New/iter")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("New unique programs")
    ax.set_title("Exploration Rate")
    ax2 = ax.twinx()
    ax2.plot(iters, unique, color="#1f77b4", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative unique programs", color="#1f77b4")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ---- Panel 3: Value loss trajectory ----
    ax = axes[0, 2]
    val_loss = [r.get("train_loss_value", 0) for r in train_stats]
    pol_loss = [r.get("train_loss_policy", 0) for r in train_stats]
    if any(v > 0 for v in val_loss):
        ax.semilogy(iters, val_loss, color="green", linewidth=2, label="Value loss")
    if any(v > 0 for v in pol_loss):
        ax.semilogy(iters, pol_loss, color="red", linewidth=2, label="Policy loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Cache hit rate ----
    ax = axes[1, 0]
    hit_rates = []
    for d in leaf_deltas:
        total = d["cache_hits"] + d["eval_count"]
        hit_rates.append(d["cache_hits"] / total if total > 0 else 0)
    ax.plot(iters, hit_rates, color="#ff7f0e", linewidth=2, marker="o",
            markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cache hit rate")
    ax.set_title("Cache Hit Rate (per iteration)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    # ---- Panel 5: Branching factor over time ----
    ax = axes[1, 1]
    br_iters = sorted(branching_by_iter.keys())
    if br_iters:
        br_means = [np.mean(branching_by_iter[i]) for i in br_iters]
        br_medians = [np.median(branching_by_iter[i]) for i in br_iters]
        ax.plot(br_iters, br_means, color="#2ca02c", linewidth=2,
                label="Mean branching")
        ax.plot(br_iters, br_medians, color="#d62728", linewidth=1,
                linestyle="--", label="Median branching")
        ax.fill_between(br_iters,
                         [np.percentile(branching_by_iter[i], 25)
                          for i in br_iters],
                         [np.percentile(branching_by_iter[i], 75)
                          for i in br_iters],
                         color="#2ca02c", alpha=0.15, label="IQR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Branching factor")
    ax.set_title("Branching Factor Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: Wall clock per iteration ----
    ax = axes[1, 2]
    wall_clocks = [r.get("iter_wall_clock", 0) for r in prog_log]
    if any(w > 0 for w in wall_clocks):
        ax.bar(iters, wall_clocks, color="#9467bd", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Seconds")
        ax.set_title("Wall Clock per Iteration")
        cumulative_mins = sum(wall_clocks) / 60
        ax.text(0.95, 0.90, f"Total: {cumulative_mins:.1f} min",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.7))
    else:
        ax.set_title("Wall Clock (no data)")

    # ---- Panel 7: Best program quality ----
    ax = axes[2, 0]
    best_rewards = [r.get("best_avg_reward", 0) for r in prog_log]
    solve_rates = [r.get("best_solve_rate", 0) for r in prog_log]
    ax.plot(iters, best_rewards, color="#1f77b4", linewidth=2,
            label="Best avg reward")
    ax2 = ax.twinx()
    ax2.plot(iters, solve_rates, color="#2ca02c", linewidth=2,
             linestyle="--", label="Best solve rate")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_ylabel("Solve rate", color="#2ca02c")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg reward", color="#1f77b4")
    ax.set_title("Best Program Quality")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ---- Panel 8: Gate score ----
    ax = axes[2, 1]
    gate_scores = [r.get("gate_score", 0.5) for r in train_stats]
    gate_accepted = [r.get("gate_accepted", 1) for r in train_stats]
    colors = ["#2ca02c" if a else "#d62728" for a in gate_accepted]
    ax.bar(iters, gate_scores, color=colors, edgecolor="white", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Neutral (0.5)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gate score")
    ax.set_title("Gate Score (green=accepted, red=rejected)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # ---- Panel 9: Reward signal analysis ----
    ax = axes[2, 2]
    # Show the weighted metric values and their compression
    weighted_vals = []
    for r in train_stats:
        ls = r.get("leaf_stats", {})
        weighted_vals.append(r.get("avg_reward", 0))
    if weighted_vals:
        ax.plot(iters, weighted_vals, color="#1f77b4", linewidth=2,
                label="Self-play avg_reward")
    ax.plot(iters, best_rewards, color="#ff7f0e", linewidth=2,
            linestyle="--", label="Best program reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Signal Analysis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Annotate the range
    if weighted_vals:
        r_min, r_max = min(weighted_vals), max(weighted_vals)
        ax.text(0.95, 0.15,
                f"Self-play range: {r_max - r_min:.4f}\n"
                f"Best prog range: {max(best_rewards) - min(best_rewards):.4f}",
                transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          alpha=0.7))

    fig.suptitle(f"Diagnostic Report: {exp_label}\n{dir_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = save_path or str(exp_dir / "diagnostics_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[Diagnostics] Report saved to {out_path}")
    return out_path


def run_comparison_diagnostics(
    exp_dirs: Sequence[Path | str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """Generate comparison diagnostic across multiple experiments.

    Args:
        exp_dirs: List of experiment directory paths.
        save_path: Override save location.
        show: If True, display plot interactively.

    Returns:
        Path to saved comparison image.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = []
    for d in exp_dirs:
        d = Path(d)
        data = _load_experiment(d)
        if data["program_log"] and data["train_stats"]:
            # Derive label
            name = d.name
            d_match = [p for p in name.split("_") if p.startswith("D")]
            mcts_match = [p for p in name.split("_") if p.startswith("mcts")]
            label = f"{d_match[0] if d_match else '?'}"
            if mcts_match:
                label += f" {mcts_match[0]}"
            data["label"] = label
            experiments.append(data)

    if len(experiments) < 2:
        print("[Diagnostics] Need at least 2 experiments for comparison.")
        return ""

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.tab10.colors

    for idx, exp in enumerate(experiments):
        c = colors[idx % len(colors)]
        label = exp["label"]
        prog_log = exp["program_log"]
        train_stats = exp["train_stats"]
        n = len(prog_log)
        iters = np.arange(1, n + 1)
        leaf_deltas = _extract_per_iter_leaf_stats(train_stats)

        # Panel 1: Best reward
        ax = axes[0, 0]
        best_r = [r.get("best_avg_reward", 0) for r in prog_log]
        ax.plot(iters, best_r, color=c, linewidth=2, label=label)

        # Panel 2: Value loss
        ax = axes[0, 1]
        vl = [r.get("train_loss_value", 0) for r in train_stats]
        if any(v > 0 for v in vl):
            ax.semilogy(iters, vl, color=c, linewidth=2, label=label)

        # Panel 3: Policy loss
        ax = axes[0, 2]
        pl = [r.get("train_loss_policy", 0) for r in train_stats]
        if any(v > 0 for v in pl):
            ax.semilogy(iters, pl, color=c, linewidth=2, label=label)

        # Panel 4: Exploration rate (new programs / iter)
        ax = axes[1, 0]
        unique = [r.get("unique_programs", 0) for r in prog_log]
        new_pi = [unique[0]] + [unique[i] - unique[i-1]
                                 for i in range(1, len(unique))]
        ax.plot(iters, new_pi, color=c, linewidth=2, label=label)

        # Panel 5: Cache hit rate
        ax = axes[1, 1]
        hr = []
        for d in leaf_deltas:
            total = d["cache_hits"] + d["eval_count"]
            hr.append(d["cache_hits"] / total if total > 0 else 0)
        ax.plot(iters, hr, color=c, linewidth=2, label=label)

        # Panel 6: Self-play avg reward
        ax = axes[1, 2]
        sp_r = [r.get("avg_reward", 0) for r in train_stats]
        ax.plot(iters, sp_r, color=c, linewidth=2, label=label)

    # Labels and formatting
    axes[0, 0].set_title("Best Program Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Best avg_reward")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss (log)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title("Policy Loss")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Loss (log)")
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_title("New Programs / Iteration")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("New unique programs")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Cache Hit Rate")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Hit rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_title("Self-Play Avg Reward")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("avg_reward")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    labels_str = " vs ".join(e["label"] for e in experiments)
    fig.suptitle(f"Experiment Comparison: {labels_str}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        # Save next to first experiment
        save_path = str(experiments[0]["dir"].parent / "comparison_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[Diagnostics] Comparison saved to {save_path}")
    return save_path
