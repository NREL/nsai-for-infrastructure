"""
Doors Grammar Comparison Suite -- batch runner for comparing grammar variants.

Runs all grammar modes (doors, doors_no_and, doors_factored, doors_d10_macro)
across multiple D values and seeds, producing comparison CSVs and plots.

Usage:
    python scripts/run_doors_derivation_suite.py --quick          # smoke test
    python scripts/run_doors_derivation_suite.py                  # full suite
    python scripts/run_doors_derivation_suite.py --resume <dir>   # resume
    python scripts/run_doors_derivation_suite.py --num-rooms 2 5  # subset of D
    python scripts/run_doors_derivation_suite.py --modes doors doors_factored
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

from alphazeropp.instances.doors.dsl.derivation_config import (
    DoorsDerivationConfig, DoorsDerivationConfigNoAnd,
    DoorsFactoredDerivationConfig, DoorsFactoredD10MacroConfig,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    compute_doors_derived_params, DoorsGameConfig,
)
from alphazeropp.synthesis.derivation_game import compute_max_productions
from alphazeropp.synthesis.factored_derivation_game import compute_max_factored_actions
from alphazeropp.instances.doors.dsl.doors_macros import make_macro_fn
from alphazeropp.utils.derivation_utils import run_derivation_training

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_CONFIGS = {
    "doors": DoorsDerivationConfig,
    "doors_no_and": DoorsDerivationConfigNoAnd,
    "doors_factored": DoorsFactoredDerivationConfig,
    "doors_d10_macro": DoorsFactoredD10MacroConfig,
}

ALL_MODES = ["doors", "doors_no_and", "doors_factored", "doors_d10_macro"]
DEFAULT_D_VALUES = [3, 5]
DEFAULT_SEEDS = [42, 137, 271]

# MCTS sims scale with D
SIMS_BY_D = {3: 100, 5: 200}
DEFAULT_N_ITERS = 20
DEFAULT_N_GAMES = 20

_COLORS = {
    "doors": "#1f77b4",
    "doors_no_and": "#d62728",
    "doors_factored": "#2ca02c",
    "doors_d10_macro": "#ff7f0e",
}

_LABELS = {
    "doors": "Flat (And)",
    "doors_no_and": "Flat (no And)",
    "doors_factored": "Factored",
    "doors_d10_macro": "Factored+Macro",
}

_DIR_LABELS = {
    "doors": "flat_and",
    "doors_no_and": "flat_noand",
    "doors_factored": "factored",
    "doors_d10_macro": "factored_macro",
}


# ---------------------------------------------------------------------------
# Pre-flight static analysis
# ---------------------------------------------------------------------------

def print_static_analysis(modes, d_values):
    """Print theoretical action space sizes for each (mode, D) combination."""
    print()
    print("=" * 90)
    print("  GRAMMAR COMPARISON: Static Analysis")
    print("=" * 90)
    print()
    print(f"  {'Mode':<20} {'D':>3} {'n_sites':>8} {'budget':>7} "
          f"{'opt_nodes':>10} {'max_prods':>10} {'max_fact':>9} {'reduction':>10}")
    print(f"  {'-'*20} {'-'*3} {'-'*8} {'-'*7} {'-'*10} {'-'*10} {'-'*9} {'-'*10}")

    for D in d_values:
        derived = compute_doors_derived_params(D)
        n_sites = derived["n_sites"]
        budget = derived["budget"]
        optimal_nodes = derived["optimal_nodes"]
        M = derived["M"]
        K = derived["K"]
        n_actions = M + K + 1

        for mode in modes:
            if mode == "doors_no_and" and D > 5:
                continue

            allow_and = mode != "doors_no_and"
            max_prods = compute_max_productions(
                budget, n_sites, mode="max",
                allow_and=allow_and, n_actions=n_actions,
            )

            is_factored = mode in ("doors_factored", "doors_d10_macro")
            if is_factored:
                factored_kwargs = dict(
                    budget=budget, n_sites=n_sites, mode="max",
                    allow_and=allow_and, n_actions=n_actions,
                )
                if mode == "doors_d10_macro":
                    doors_cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
                    factored_kwargs["macro_productions_fn"] = make_macro_fn(doors_cfg)
                    factored_kwargs["max_condition_budget"] = 12
                max_factored = compute_max_factored_actions(**factored_kwargs)
                reduction = f"{max_prods / max_factored:.1f}x"
            else:
                max_factored = None
                reduction = "—"

            fact_str = str(max_factored) if max_factored is not None else "—"
            print(f"  {mode:<20} {D:>3} {n_sites:>8} {budget:>7} "
                  f"{optimal_nodes:>10} {max_prods:>10} {fact_str:>9} {reduction:>10}")

        if D != d_values[-1]:
            print()

    print()
    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config_for_suite(mode, num_rooms, n_sims, n_iters, n_games, seed):
    """Programmatically create a derivation config for the suite."""
    cfg = MODE_CONFIGS[mode]()
    derived = compute_doors_derived_params(num_rooms)

    cfg.game.kwargs.update({
        "num_rooms": num_rooms,
        "n_sites": derived["n_sites"],
        "budget": derived["budget"],
        "horizon": derived["horizon"],
    })
    cfg.net.kwargs.update({
        "n_sites": derived["n_sites"],
        "budget": derived["budget"],
    })
    cfg.agent.mcts_params["n_simulations"] = n_sims
    cfg.agent.random_seeds = {
        "mcts": seed,
        "train": seed + 1,
        "eval": seed + 2,
        "external_policy": seed + 3,
    }
    cfg.run.n_iterations = n_iters
    cfg.trainer.n_games_per_train = n_games
    return cfg


# ---------------------------------------------------------------------------
# Single-config training run
# ---------------------------------------------------------------------------

def run_single_config(mode, D, seed, n_sims, n_iters, n_games, suite_dir):
    """Run one (mode, D, seed) configuration. Returns summary dict."""
    dir_label = _DIR_LABELS.get(mode, mode)
    run_name = f"{dir_label}_D{D}_seed{seed}"
    config_dir = suite_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    summary_file = config_dir / "summary.json"
    if summary_file.exists():
        logger.info("  [SKIP] %s (already completed)", run_name)
        with open(summary_file) as f:
            return json.load(f)

    logger.info("\n" + "=" * 70)
    logger.info("  Running: %s  (sims=%d, iters=%d, games=%d)",
                run_name, n_sims, n_iters, n_games)
    logger.info("=" * 70)

    cfg = build_config_for_suite(mode, D, n_sims, n_iters, n_games, seed)

    # Experiment directory for this run's artifacts
    exp_dir = config_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")
    cfg.save(str(exp_dir / "config.json"))

    # Iteration log capture via callback
    iteration_log = []

    def on_iteration(iteration, prog_entry, gate_score, gate_accepted, wall_s):
        entry = {
            "mode": mode,
            "D": D,
            "seed": seed,
            "iteration": iteration,
            "best_solve_rate": prog_entry.get("best_solve_rate", 0),
            "best_avg_reward": prog_entry.get("best_avg_reward", 0),
            "unique_programs": prog_entry.get("unique_programs", 0),
            "gate_score": float(gate_score),
            "gate_accepted": bool(gate_accepted),
            "wall_clock_s": round(wall_s, 2),
        }
        iteration_log.append(entry)

        # Append to JSONL incrementally
        with open(config_dir / "iteration_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    # Compute correct optimal reward (includes unlock bonuses and step penalties)
    optimal_steps = 2 * (D - 1) + 1
    step_penalty = cfg.game.kwargs.get("step_penalty", 0.01)
    unlock_bonus = cfg.game.kwargs.get("unlock_bonus", 0.1)
    optimal_reward = 1.0 + (D - 1) * unlock_bonus - optimal_steps * step_penalty

    t_total = time.time()
    run_derivation_training(
        cfg, mode, optimal_reward=optimal_reward, exp_dir=exp_dir,
        iteration_callback=on_iteration,
    )
    total_wall = time.time() - t_total

    # Build summary
    final = iteration_log[-1] if iteration_log else {}
    summary = {
        "run_name": run_name,
        "mode": mode,
        "D": D,
        "seed": seed,
        "n_sims": n_sims,
        "n_iters_run": len(iteration_log),
        "final_solve_rate": final.get("best_solve_rate", 0),
        "final_avg_reward": final.get("best_avg_reward", 0),
        "final_unique_programs": final.get("unique_programs", 0),
        "total_wall_clock_s": round(total_wall, 1),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Aggregation: CSV + terminal table
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "run_name", "mode", "D", "seed",
    "final_solve_rate", "final_avg_reward", "final_unique_programs",
    "n_iters_run", "total_wall_clock_s",
]


def write_summary_csv(summaries, path):
    """Write summary CSV sorted by D, then solve rate desc."""
    ranked = sorted(summaries,
                    key=lambda s: (s["D"], -s.get("final_solve_rate", 0),
                                   -s.get("final_avg_reward", -999)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)


def print_summary_table(summaries):
    """Print compact results table to terminal."""
    ranked = sorted(summaries,
                    key=lambda s: (s["D"], s["mode"], s["seed"]))
    sep = "=" * 100
    print(f"\n{sep}")
    print("  GRAMMAR COMPARISON RESULTS")
    print(sep)
    print(f"  {'Run':<30} {'D':>3} {'Solve':>6} {'Reward':>8} "
          f"{'Progs':>6} {'Time':>7}")
    print(f"  {'-'*30} {'-'*3} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")

    for s in ranked:
        sr = s.get("final_solve_rate", 0)
        rw = s.get("final_avg_reward", 0)
        progs = s.get("final_unique_programs", 0)
        t = s.get("total_wall_clock_s", 0)
        print(f"  {s['run_name']:<30} {s['D']:>3} {sr:>5.0%} {rw:>+8.3f} "
              f"{progs:>6} {t:>6.0f}s")

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_iteration_logs(suite_dir, modes, d_values, seeds):
    """Load all iteration logs. Returns dict[(mode, D, seed)] -> list[dict]."""
    logs = {}
    for mode in modes:
        for D in d_values:
            if mode == "doors_no_and" and D > 5:
                continue
            for seed in seeds:
                dir_label = _DIR_LABELS.get(mode, mode)
                run_name = f"{dir_label}_D{D}_seed{seed}"
                path = suite_dir / run_name / "iteration_log.jsonl"
                if not path.exists():
                    continue
                records = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                if records:
                    logs[(mode, D, seed)] = records
    return logs


def _aggregate_across_seeds(logs, modes, D, seeds, metric):
    """For a given D, compute per-mode mean and std of a metric across seeds.

    Returns dict[mode] -> (iterations, means, stds).
    """
    result = {}
    for mode in modes:
        per_seed = []
        for seed in seeds:
            key = (mode, D, seed)
            if key not in logs:
                continue
            values = [e.get(metric, 0) for e in logs[key]]
            per_seed.append(values)

        if not per_seed:
            continue

        # Align lengths (use shortest)
        min_len = min(len(v) for v in per_seed)
        trimmed = [v[:min_len] for v in per_seed]
        arr = np.array(trimmed)  # shape (n_seeds, n_iters)
        iters = list(range(1, min_len + 1))
        result[mode] = (iters, arr.mean(axis=0), arr.std(axis=0))

    return result


def plot_comparison(suite_dir, modes, d_values, seeds):
    """Generate all comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    logs = _load_iteration_logs(suite_dir, modes, d_values, seeds)
    if not logs:
        logger.warning("No iteration logs found for plotting")
        return

    # --- Plot 1: Solve rate vs iteration (per D) ---
    _plot_metric_per_d(logs, modes, d_values, seeds,
                       metric="best_solve_rate", ylabel="Best Solve Rate",
                       title="Solve Rate by Grammar Variant",
                       ylim=(-0.05, 1.1),
                       save_path=suite_dir / "solve_rate_curves.png")

    # --- Plot 2: Avg reward vs iteration (per D) ---
    _plot_metric_per_d(logs, modes, d_values, seeds,
                       metric="best_avg_reward", ylabel="Best Avg Reward",
                       title="Average Reward by Grammar Variant",
                       ylim=None,
                       save_path=suite_dir / "avg_reward_curves.png")

    # --- Plot 3: Unique programs vs iteration (per D) ---
    _plot_metric_per_d(logs, modes, d_values, seeds,
                       metric="unique_programs", ylabel="Unique Programs",
                       title="Search Efficiency by Grammar Variant",
                       ylim=None,
                       save_path=suite_dir / "unique_programs_curves.png")

    # --- Plot 4: Summary bar chart ---
    _plot_summary_bars(suite_dir, modes, d_values, seeds, logs)

    logger.info("[Plot] All comparison plots saved to %s/", suite_dir)


def _plot_metric_per_d(logs, modes, d_values, seeds, metric, ylabel,
                       title, ylim, save_path):
    """Plot a metric vs iteration with one subplot per D value."""
    import matplotlib.pyplot as plt

    n_d = len(d_values)
    ncols = min(n_d, 2)
    nrows = (n_d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, D in enumerate(d_values):
        ax = axes[idx // ncols][idx % ncols]
        agg = _aggregate_across_seeds(logs, modes, D, seeds, metric)

        for mode, (iters, means, stds) in agg.items():
            color = _COLORS.get(mode, "#333333")
            label = _LABELS.get(mode, mode)
            ax.plot(iters, means, color=color, linewidth=2, label=label)
            if len(seeds) > 1:
                ax.fill_between(iters, means - stds, means + stds,
                                color=color, alpha=0.15)

        ax.set_title(f"D = {D}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_d, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Saved %s", save_path)


def _plot_summary_bars(suite_dir, modes, d_values, seeds, logs):
    """Plot grouped bar chart: best solve rate per (mode, D)."""
    import matplotlib.pyplot as plt

    # Compute final solve rate mean/std for each (mode, D)
    groups = []  # (mode, D, mean, std)
    for D in d_values:
        for mode in modes:
            if mode == "doors_no_and" and D > 5:
                continue
            values = []
            for seed in seeds:
                key = (mode, D, seed)
                if key in logs and logs[key]:
                    values.append(logs[key][-1].get("best_solve_rate", 0))
            if values:
                groups.append((mode, D, np.mean(values), np.std(values)))

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(groups) * 0.4)))
    fig.suptitle("Grammar Comparison: Best Solve Rate", fontsize=14,
                 fontweight="bold")

    labels = [f"{_LABELS.get(m, m)} (D={D})" for m, D, _, _ in groups]
    means = [m for _, _, m, _ in groups]
    stds = [s for _, _, _, s in groups]
    colors = [_COLORS.get(m, "#333333") for m, _, _, _ in groups]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=stds if len(seeds) > 1 else None,
            color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Best Solve Rate")
    ax.set_xlim(0, 1.15)
    ax.axvline(x=1.0, color="green", linestyle=":", alpha=0.4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = suite_dir / "summary_bars.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Saved %s", save_path)


# ---------------------------------------------------------------------------
# Branching factor aggregation
# ---------------------------------------------------------------------------

def plot_branching(suite_dir, modes, d_values, seeds):
    """Plot mean branching factor per iteration from diagnostics.jsonl."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_d = len(d_values)
    ncols = min(n_d, 2)
    nrows = (n_d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle("Mean Branching Factor by Grammar Variant",
                 fontsize=14, fontweight="bold")

    for idx, D in enumerate(d_values):
        ax = axes[idx // ncols][idx % ncols]

        for mode in modes:
            if mode == "doors_no_and" and D > 5:
                continue

            per_seed_means = []
            for seed in seeds:
                dir_label = _DIR_LABELS.get(mode, mode)
                run_name = f"{dir_label}_D{D}_seed{seed}"
                diag_path = (suite_dir / run_name / "experiment"
                             / "diagnostics.jsonl")
                if not diag_path.exists():
                    continue

                # Load and aggregate: mean branching per iteration
                iter_branching = {}
                with open(diag_path) as f:
                    for line in f:
                        rec = json.loads(line.strip())
                        it = rec["iteration"]
                        b = rec.get("branching")
                        if b is not None:
                            iter_branching.setdefault(it, []).append(b)

                if iter_branching:
                    iters_sorted = sorted(iter_branching.keys())
                    means = [np.mean(iter_branching[it]) for it in iters_sorted]
                    per_seed_means.append((iters_sorted, means))

            if not per_seed_means:
                continue

            # Align seeds
            min_len = min(len(m[1]) for m in per_seed_means)
            arr = np.array([m[1][:min_len] for m in per_seed_means])
            iters = list(range(1, min_len + 1))

            color = _COLORS.get(mode, "#333333")
            label = _LABELS.get(mode, mode)
            mean_vals = arr.mean(axis=0)
            ax.plot(iters, mean_vals, color=color, linewidth=2, label=label)
            if len(seeds) > 1:
                std_vals = arr.std(axis=0)
                ax.fill_between(iters, mean_vals - std_vals,
                                mean_vals + std_vals, color=color, alpha=0.15)

        ax.set_title(f"D = {D}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Branching Factor")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for idx in range(n_d, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    save_path = suite_dir / "branching_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Saved %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Doors grammar comparison suite")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=ALL_MODES,
                        help="Grammar modes to compare (default: all)")
    parser.add_argument("--num-rooms", nargs="+", type=int, default=None,
                        help="D values to test (default: 3 5)")
    parser.add_argument("--sims", type=int, default=None,
                        help="Override MCTS sims for all configs")
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS,
                        help=f"Training iterations (default: {DEFAULT_N_ITERS})")
    parser.add_argument("--n-games", type=int, default=DEFAULT_N_GAMES,
                        help=f"Self-play games per iter (default: {DEFAULT_N_GAMES})")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: 42 137 271)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: D=2 only, 3 iters, 1 seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing suite directory")
    args = parser.parse_args()

    # Resolve parameters
    modes = args.modes or ALL_MODES
    d_values = args.num_rooms or DEFAULT_D_VALUES
    seeds = args.seeds or DEFAULT_SEEDS

    if args.quick:
        d_values = [3]
        seeds = [42]
        args.n_iters = 3
        logger.info("QUICK MODE: D=3, 3 iters, 1 seed")

    # Setup suite directory
    if args.resume:
        suite_dir = Path(args.resume)
        if not suite_dir.exists():
            raise FileNotFoundError(f"Suite dir not found: {suite_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_tag = "_".join(str(d) for d in d_values)
        suite_dir = (Path("experiments") / "doors_grammar_suite"
                     / f"{timestamp}_rooms{d_tag}")
        suite_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Suite directory: %s", suite_dir)
    logger.info("Modes: %s", modes)
    logger.info("D values: %s", d_values)
    logger.info("Seeds: %s", seeds)
    logger.info("Iterations: %d", args.n_iters)

    # Save suite config
    suite_config = {
        "modes": modes,
        "d_values": d_values,
        "seeds": seeds,
        "n_iters": args.n_iters,
        "n_games": args.n_games,
        "sims_by_d": {str(d): args.sims or SIMS_BY_D.get(d, 100)
                      for d in d_values},
    }
    with open(suite_dir / "suite_config.json", "w") as f:
        json.dump(suite_config, f, indent=2)

    # Pre-flight static analysis
    print_static_analysis(modes, d_values)

    # Count total runs
    total_runs = 0
    run_list = []
    for D in d_values:
        n_sims = args.sims or SIMS_BY_D.get(D, 100)
        for mode in modes:
            if mode == "doors_no_and" and D > 5:
                continue
            for seed in seeds:
                run_list.append((mode, D, seed, n_sims))
                total_runs += 1

    logger.info("Total runs: %d", total_runs)

    # Run all configs
    summaries = []
    for run_idx, (mode, D, seed, n_sims) in enumerate(run_list):
        logger.info("\n[%d/%d] %s D=%d seed=%d",
                    run_idx + 1, total_runs, mode, D, seed)

        summary = run_single_config(
            mode, D, seed, n_sims, args.n_iters, args.n_games, suite_dir,
        )
        summaries.append(summary)

        # Update CSV after each run
        write_summary_csv(summaries, suite_dir / "summary.csv")

    # Final outputs
    print_summary_table(summaries)
    plot_comparison(suite_dir, modes, d_values, seeds)
    plot_branching(suite_dir, modes, d_values, seeds)

    logger.info("\nSuite complete. Results in: %s/", suite_dir)
    logger.info("  summary.csv              -- ranked results")
    logger.info("  solve_rate_curves.png    -- solve rate vs iteration")
    logger.info("  avg_reward_curves.png    -- reward vs iteration")
    logger.info("  unique_programs_curves.png -- search efficiency")
    logger.info("  branching_curves.png     -- mean branching factor")
    logger.info("  summary_bars.png         -- bar chart comparison")


if __name__ == "__main__":
    main()
