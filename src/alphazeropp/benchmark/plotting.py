"""Plotting helpers for cross-algorithm benchmark comparison.

Reads CSV result files and produces comparison plots.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_csvs(csv_paths: list[Path]) -> list[dict]:
    """Load all rows from multiple CSV files into a flat list of dicts."""
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    try:
                        row[key] = float(row[key])
                        if row[key] == int(row[key]):
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
                rows.append(row)
    return rows


def _group_by_algo(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by algorithm name."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["algorithm"]].append(row)
    return dict(groups)


def _group_by_d(rows: list[dict]) -> dict[int, list[dict]]:
    """Group rows by D (number of rooms)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["D"]].append(row)
    return dict(groups)


def _separate_static_algos(
    algo_groups: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split algos into static (single checkpoint) and learning (multiple).

    Static algorithms (oracle, random) have only 1 checkpoint and are best
    shown as horizontal reference lines on learning curves.
    Groups by (seed, D) to handle multi-D sweeps correctly.
    """
    static = {}
    learning = {}
    for name, rows in algo_groups.items():
        # Group by (seed, D) to check if any combination has >1 checkpoint
        groups = defaultdict(list)
        for r in rows:
            groups[(r["seed"], r.get("D", 0))].append(r)
        max_checkpoints = max(len(v) for v in groups.values())
        if max_checkpoints <= 1:
            static[name] = rows
        else:
            learning[name] = rows
    return static, learning


def _plot_algo_curves(ax, algo_name, algo_rows, x_axis, metric):
    """Plot one algorithm's learning curve (mean +/- std across seeds)."""
    seed_groups = defaultdict(list)
    for row in algo_rows:
        seed_groups[row["seed"]].append(row)

    all_xs = set()
    seed_curves = {}
    for seed, seed_rows in seed_groups.items():
        sorted_rows = sorted(seed_rows, key=lambda r: r[x_axis])
        xs = [r[x_axis] for r in sorted_rows]
        ys = [r[metric] for r in sorted_rows]
        seed_curves[seed] = (xs, ys)
        all_xs.update(xs)

    if len(seed_curves) == 1:
        xs, ys = list(seed_curves.values())[0]
        ax.plot(xs, ys, marker="o", markersize=3, label=algo_name)
    else:
        x_grid = sorted(all_xs)
        y_matrix = []
        for seed, (xs, ys) in seed_curves.items():
            y_interp = np.interp(x_grid, xs, ys)
            y_matrix.append(y_interp)
        y_matrix = np.array(y_matrix)
        y_mean = y_matrix.mean(axis=0)
        y_std = y_matrix.std(axis=0)
        ax.plot(x_grid, y_mean, marker="o", markersize=2, label=algo_name)
        ax.fill_between(x_grid, y_mean - y_std, y_mean + y_std, alpha=0.2)


def _plot_static_baselines(ax, static_algos, metric):
    """Draw horizontal dashed reference lines for static baselines."""
    colors = {"oracle": "green", "random": "red"}
    for name, rows in sorted(static_algos.items()):
        values = [r[metric] for r in rows]
        mean_val = np.mean(values)
        color = colors.get(name, "gray")
        ax.axhline(y=mean_val, linestyle="--", color=color, alpha=0.7,
                    label=f"{name} ({mean_val:.3f})")


def plot_learning_curves(
    csv_paths: list[Path],
    output_path: Path,
    metric: str = "eval_return_mean",
    x_axis: str = "env_steps",
    title: str | None = None,
):
    """Plot learning curves across algorithms and seeds.

    Static baselines (single-checkpoint algorithms like oracle/random) are
    drawn as horizontal dashed reference lines.
    Learning algorithms are drawn as curves with mean +/- std across seeds.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return

    rows = _load_csvs(csv_paths)
    if not rows:
        print("[Warning] No data to plot.")
        return

    algo_groups = _group_by_algo(rows)
    static, learning = _separate_static_algos(algo_groups)

    fig, ax = plt.subplots(figsize=(10, 6))

    _plot_static_baselines(ax, static, metric)
    for algo_name in sorted(learning.keys()):
        _plot_algo_curves(ax, algo_name, learning[algo_name], x_axis, metric)

    ax.set_xlabel(x_axis.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric} vs {x_axis}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {output_path}")


def plot_comparison_panel(
    csv_paths: list[Path],
    output_path: Path,
    x_axis: str = "train_episodes",
    title: str | None = None,
):
    """Side-by-side panel: return (left) and success rate (right)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return

    rows = _load_csvs(csv_paths)
    if not rows:
        print("[Warning] No data to plot.")
        return

    algo_groups = _group_by_algo(rows)
    static, learning = _separate_static_algos(algo_groups)

    metrics = [
        ("eval_return_mean", "Eval Return Mean"),
        ("eval_success_rate", "Eval Success Rate"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (metric, ylabel) in zip(axes, metrics):
        _plot_static_baselines(ax, static, metric)
        for algo_name in sorted(learning.keys()):
            _plot_algo_curves(ax, algo_name, learning[algo_name], x_axis, metric)
        ax.set_xlabel(x_axis.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title or "Benchmark Comparison", fontsize=14)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {output_path}")


def plot_wall_clock_comparison(
    csv_paths: list[Path],
    output_path: Path,
    title: str | None = None,
):
    """Bar chart comparing wall-clock time to solve across algorithms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot.")
        return

    rows = _load_csvs(csv_paths)
    if not rows:
        return

    algo_groups = _group_by_algo(rows)

    algo_names = []
    solve_times = []

    for algo_name in sorted(algo_groups.keys()):
        algo_rows = algo_groups[algo_name]
        # Find solve_wall_clock_sec (take first non-None per seed, then average)
        seed_times = {}
        for row in algo_rows:
            seed = row["seed"]
            wc = row.get("solve_wall_clock_sec")
            if wc is not None and wc != "" and seed not in seed_times:
                seed_times[seed] = float(wc)

        if seed_times:
            algo_names.append(algo_name)
            solve_times.append(np.mean(list(seed_times.values())))

    if not algo_names:
        print("[Warning] No solve data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algo_names, solve_times, color="steelblue", edgecolor="navy")
    ax.set_ylabel("Wall-clock to Solve (s)")
    ax.set_title(title or "Wall-clock Time to Solve")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, solve_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {output_path}")


def print_summary_table(
    csv_paths: list[Path],
    output_file: io.TextIOBase | None = None,
) -> str:
    """Print a formatted summary table comparing final metrics across algorithms.

    Groups by (algorithm, D). Takes the last checkpoint per (algorithm, D, seed),
    averages across seeds. Returns the table string.
    """
    rows = _load_csvs(csv_paths)
    if not rows:
        return "[No data]\n"

    # Group by (algorithm, D)
    ad_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        ad_groups[(r["algorithm"], r["D"])].append(r)

    summaries = []
    for (algo_name, D) in sorted(ad_groups.keys()):
        group_rows = ad_groups[(algo_name, D)]
        seed_groups = defaultdict(list)
        for r in group_rows:
            seed_groups[r["seed"]].append(r)

        final_returns = []
        final_success = []
        final_env_steps = []
        final_wall_clock = []
        mask_mode = "none"

        for seed, seed_rows in seed_groups.items():
            last = max(seed_rows, key=lambda r: r.get("eval_checkpoint_idx", 0))
            final_returns.append(last["eval_return_mean"])
            final_success.append(last["eval_success_rate"])
            final_env_steps.append(last["env_steps"])
            final_wall_clock.append(last["wall_clock_sec"])
            mask_mode = last.get("mask_mode", "none")

        summaries.append({
            "algorithm": algo_name,
            "D": D,
            "mask_mode": mask_mode,
            "n_seeds": len(seed_groups),
            "return_mean": np.mean(final_returns),
            "return_std": np.std(final_returns),
            "success_mean": np.mean(final_success),
            "env_steps_mean": np.mean(final_env_steps),
            "wall_clock_mean": np.mean(final_wall_clock),
        })

    # Format table
    header = (
        f"{'Algorithm':<16} {'D':>3} {'Mask':<12} {'Seeds':>5} "
        f"{'Return':>12} {'Success':>9} {'Env Steps':>11} {'Wall Clock':>11}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for s in summaries:
        ret_str = f"{s['return_mean']:.3f}"
        if s["n_seeds"] > 1:
            ret_str += f" +/- {s['return_std']:.3f}"
        lines.append(
            f"{s['algorithm']:<16} {s['D']:>3} {s['mask_mode']:<12} {s['n_seeds']:>5} "
            f"{ret_str:>12} {s['success_mean']:>8.2f} "
            f"{s['env_steps_mean']:>11.0f} {s['wall_clock_mean']:>10.1f}s"
        )

    table = "\n".join(lines) + "\n"

    if output_file is not None:
        output_file.write(table)
    else:
        print(table)

    return table
