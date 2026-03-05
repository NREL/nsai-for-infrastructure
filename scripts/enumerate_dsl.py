#!/usr/bin/env python3
"""
Enumerate and analyze BitString DSL programs via the size-budget grammar.

Interactive script that prompts for game configuration, then:
  1. Prints grammar structure with explanations
  2. Evaluates all programs exhaustively on all initial states
  3. Generates 3 analysis plots
  4. Prints a guide table summarizing the results

CLI args override interactive prompts. Use --no-interactive to skip all prompts.

Usage:
    python scripts/enumerate_dsl.py                           # interactive
    python scripts/enumerate_dsl.py --n_sites 5 --no-bit_flip # override some
    python scripts/enumerate_dsl.py --no-interactive           # all defaults
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
    Program,
)
from alphazeropp.synthesis.interpreter import (
    eval_program, interp_ops, run_policy_episode, format_trace,
)
from alphazeropp.synthesis.budget_grammar import (
    enumerate_programs, count_programs, count_conditions,
    format_grammar_summary, format_grammar_debug,
)
from alphazeropp.synthesis.derivation import (
    DerivationState, trace_first_derivation,
)
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import ShapedBitStringGym
from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY

# Output directory for all plots
PLOT_DIR = Path("results/dsl_plots")

# Skip behavioral eval if more than this many programs at a budget level
MAX_EVAL_PROGRAMS = 200


def _prompt(label: str, description: str, default: str) -> str:
    """Prompt user for a value with description and default."""
    print(f"\n{label} [default: {default}]:")
    for line in description.strip().split("\n"):
        print(f"  {line}")
    response = input("> ").strip()
    return response if response else default


def prompt_game_config(args: argparse.Namespace) -> tuple[GameConfig, int, int, int | None]:
    """Interactive prompts for game configuration.

    CLI args override prompts (those questions are skipped).
    Returns (cfg, n_sites, max_budget, eval_budget).
    """
    interactive = not args.no_interactive

    print("=== Game Configuration ===")

    # n_sites
    if args.n_sites is not None:
        n_sites = args.n_sites
        if interactive:
            print(f"\n  N = {n_sites} (from --n_sites)")
    elif interactive:
        val = _prompt(
            "Number of bits (N)",
            "Controls action space Flip(0..N-1) and condition space IsZero(0..N-1).\n"
            "Program counts grow as O(N^k). C(N, n_ones) initial states are tested.",
            "3",
        )
        n_sites = int(val)
    else:
        n_sites = 3
    print(f"\u2192 N = {n_sites}")

    # bit_flip
    if args.bit_flip is not None:
        bit_flip = args.bit_flip
        if interactive:
            print(f"\n  bit_flip = {bit_flip} (from CLI)")
    elif interactive:
        val = _prompt(
            "Bit-flip mode (y/n)",
            "yes: Flip toggles 0\u21941 (mistakes possible \u2014 flipping a 1 back to 0)\n"
            "no:  Flip only sets 0\u21921 (no mistakes, easier game)",
            "y",
        )
        bit_flip = val.lower() in ("y", "yes", "true", "1")
    else:
        bit_flip = True
    print(f"\u2192 bit_flip = {bit_flip}")

    # n_ones
    if args.n_ones is not None:
        n_ones = args.n_ones
        if interactive:
            print(f"\n  n_ones = {n_ones} (from --n_ones)")
    elif interactive:
        val = _prompt(
            "Number of initial 1-bits (n_ones)",
            "How many bits start as 1. Affects difficulty (must flip N - n_ones\n"
            f"zeros to ones) and number of initial states: C(N, n_ones).",
            "2",
        )
        n_ones = int(val)
    else:
        n_ones = 2
    n_inits = comb(n_sites, n_ones)
    print(f"\u2192 n_ones = {n_ones}, initial states = C({n_sites},{n_ones}) = {n_inits}")

    # sparse_reward
    if args.sparse_reward is not None:
        sparse_reward = args.sparse_reward
        if interactive:
            print(f"\n  sparse_reward = {sparse_reward} (from CLI)")
    elif interactive:
        val = _prompt(
            "Reward mode (dense/sparse)",
            f"dense:  reward every step, max_steps = 2N = {2 * n_sites} (room to recover)\n"
            f"sparse: reward only at end, max_steps = N - n_ones = {n_sites - n_ones} (tight)",
            "dense",
        )
        sparse_reward = val.lower() in ("sparse", "s")
    else:
        sparse_reward = False
    ms = (n_sites - n_ones) if sparse_reward else 2 * n_sites
    print(f"\u2192 sparse_reward = {sparse_reward}, max_steps = {ms}")

    # potential
    pot_names = list(POTENTIAL_REGISTRY.keys())
    if args.potential is not None:
        potential_name = args.potential
        if interactive:
            print(f"\n  potential = {potential_name} (from --potential)")
    elif interactive:
        val = _prompt(
            f"Potential function ({'/'.join(pot_names)})",
            "onemax:       sum of bits (most natural \u2014 just count 1s)\n"
            "leading_ones: length of leading all-ones prefix\n"
            "binval:       binary value with bit 0 as MSB",
            "onemax",
        )
        potential_name = val if val in pot_names else "onemax"
    else:
        potential_name = "onemax"
    print(f"\u2192 potential = {potential_name}")

    # max_budget
    if args.max_budget is not None:
        max_budget = args.max_budget
        if interactive:
            print(f"\n  max_budget = {max_budget} (from --max_budget)")
    elif interactive:
        val = _prompt(
            "Max AST budget (L)",
            "Maximum AST node count to enumerate. L=2 is Default(Flip), L=5 is one\n"
            "if/else rule, L=8 is a two-rule chain. L=1,3,4 are impossible (gap\n"
            "between Default=2 and smallest Ite=5).",
            "8",
        )
        max_budget = int(val)
    else:
        max_budget = 8
    print(f"\u2192 max_budget = {max_budget}")

    # eval_budget (not prompted — auto unless specified)
    eval_budget = args.eval_budget

    cfg = GameConfig(
        bit_flip=bit_flip,
        sparse_reward=sparse_reward,
        n_ones=n_ones,
        potential_fn=POTENTIAL_REGISTRY[potential_name],
        potential_name=potential_name,
    )

    # Summary box
    print()
    box_lines = [
        "Active Configuration",
        f"  N={n_sites}, bit_flip={bit_flip}, n_ones={n_ones}",
        f"  sparse_reward={sparse_reward}, potential={potential_name}",
        f"  max_budget={max_budget}, eval_budget={'auto' if eval_budget is None else eval_budget}",
        f"  Initial states: {n_inits}, Max steps: {ms}",
    ]
    width = max(len(line) for line in box_lines) + 2
    print("\u250c" + "\u2500" * width + "\u2510")
    for line in box_lines:
        print(f"\u2502 {line:<{width - 1}}\u2502")
    print("\u2514" + "\u2500" * width + "\u2518")

    return cfg, n_sites, max_budget, eval_budget


# ---------------------------------------------------------------------------
# Exhaustive evaluation
# ---------------------------------------------------------------------------

def evaluate_program(prog: Program, n_sites: int, cfg: GameConfig) -> dict:
    """Evaluate a program on ALL C(n_sites, n_ones) initial states (exact).

    Returns dict with: solve_rate, avg_steps, avg_interp_ops, avg_reward,
                       n_episodes (number of initial states tested).
    """
    inits = all_initial_states(n_sites, cfg.n_ones)
    solved_count = 0
    total_steps = 0
    total_ops = 0
    total_reward = 0.0

    for x0 in inits:
        env = cfg.make_env(n_sites, frozen_states=[x0])
        env.reset()
        result = run_policy_episode(env, prog)
        if result.solved:
            solved_count += 1
        total_steps += result.total_env_steps
        total_ops += result.total_interp_ops
        total_reward += result.cumulative_reward

    n = len(inits)
    return {
        "solve_rate": solved_count / n,
        "avg_steps": total_steps / n,
        "avg_interp_ops": total_ops / n,
        "avg_reward": total_reward / n,
        "n_episodes": n,
    }


# ---------------------------------------------------------------------------
# Enumeration analysis table
# ---------------------------------------------------------------------------

def format_enumeration_analysis(
    n_sites: int,
    budget: int,
    cfg: GameConfig,
) -> tuple[str, list[tuple[Program, dict]]]:
    """Enumerate programs at a given budget and evaluate each one exactly.

    Returns (formatted string, list of (program, metrics) tuples).
    """
    programs = enumerate_programs(n_sites, budget)
    if not programs:
        return f"No programs exist at budget={budget} for N={n_sites}.", []

    n_inits = comb(n_sites, cfg.n_ones)
    results: list[tuple[Program, dict]] = []
    for prog in programs:
        metrics = evaluate_program(prog, n_sites, cfg)
        results.append((prog, metrics))

    lines: list[str] = []
    lines.append(
        f"=== Enumeration Analysis: N={n_sites}, L={budget} "
        f"({len(programs)} programs, {n_inits} initial states each) ==="
    )
    lines.append("")

    # Header
    header = (
        f"{'#':>3}  {'Program':<50}  {'Solve':>6}  "
        f"{'AvgSteps':>8}  {'AvgOps':>6}  {'AvgRew':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for idx, (prog, m) in enumerate(results, 1):
        pretty = _oneline_pretty(prog)
        if len(pretty) > 50:
            pretty = pretty[:47] + "..."
        solved_frac = f"{int(m['solve_rate'] * m['n_episodes'])}/{m['n_episodes']}"
        lines.append(
            f"{idx:>3}  {pretty:<50}  {solved_frac:>6}  "
            f"{m['avg_steps']:>8.2f}  {m['avg_interp_ops']:>6.2f}  "
            f"{m['avg_reward']:>+7.4f}"
        )

    # Best programs
    lines.append("")
    best_solve = max(results, key=lambda x: x[1]["solve_rate"])
    best_ops = min(results, key=lambda x: x[1]["avg_interp_ops"])
    lines.append(
        f"Best by solve rate: {_oneline_pretty(best_solve[0])} "
        f"-- {best_solve[1]['solve_rate']*100:.1f}%"
    )
    lines.append(
        f"Best by avg ops:    {_oneline_pretty(best_ops[0])} "
        f"-- {best_ops[1]['avg_interp_ops']:.2f} avg ops"
    )

    return "\n".join(lines), results


def _oneline_pretty(prog: Program) -> str:
    """One-line version of pretty() for table display."""
    return prog.pretty().replace("\n", " | ")


# ---------------------------------------------------------------------------
# Plot 1: Best solve rate by budget (cumulative)
# ---------------------------------------------------------------------------

def plot_best_by_budget(
    n_sites: int,
    max_budget: int,
    cfg: GameConfig,
    save_path: str | None = None,
) -> dict[int, float]:
    """Bar chart: best achievable solve rate among all programs with size <= L.

    Returns dict mapping budget -> cumulative best solve rate.
    """
    import matplotlib.pyplot as plt

    budgets = list(range(1, max_budget + 1))
    n_inits = comb(n_sites, cfg.n_ones)

    # Evaluate best solve rate at each exact budget, then accumulate
    best_at_exact: dict[int, float] = {}
    cumul_count: list[int] = []
    running_count = 0

    for b in budgets:
        cnt = count_programs(n_sites, b)
        running_count += cnt
        cumul_count.append(running_count)
        if cnt == 0:
            best_at_exact[b] = -1.0
        else:
            progs = enumerate_programs(n_sites, b)
            best_sr = -1.0
            for p in progs:
                m = evaluate_program(p, n_sites, cfg)
                if m["solve_rate"] > best_sr:
                    best_sr = m["solve_rate"]
            best_at_exact[b] = best_sr

    # Build cumulative best
    cumul_best: list[float] = []
    running_best = 0.0
    for b in budgets:
        if best_at_exact[b] >= 0:
            running_best = max(running_best, best_at_exact[b])
        cumul_best.append(running_best)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for b, cb in zip(budgets, cumul_best):
        cnt = count_programs(n_sites, b)
        if cnt == 0:
            colors.append("#CCCCCC")
        elif cb > 0.99:
            colors.append("#2ECC71")
        else:
            colors.append("#4C72B0")
    bars = ax.bar(budgets, cumul_best, color=colors, edgecolor="white", width=0.7)

    # Annotate bars
    for b, cb, bar, cc in zip(budgets, cumul_best, bars, cumul_count):
        cnt = count_programs(n_sites, b)
        if cnt == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                    "impossible", ha="center", va="bottom", fontsize=7,
                    color="#999999", fontstyle="italic")
        else:
            solved_n = int(round(cb * n_inits))
            label = f"{solved_n}/{n_inits}"
            ax.text(bar.get_x() + bar.get_width() / 2, cb + 0.02,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, -0.06,
                    f"({cc} progs)", ha="center", va="top", fontsize=7,
                    color="#666666")

    ax.set_xlabel("Budget (AST node count \u2264 L)")
    ax.set_ylabel("Best Solve Rate")
    ax.set_title(f"Best Achievable Solve Rate by Program Complexity (N={n_sites})")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(budgets)
    ax.axhline(y=1.0, color="#2ECC71", linestyle="--", alpha=0.3, label="perfect")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return dict(zip(budgets, cumul_best))


# ---------------------------------------------------------------------------
# Plot 2: Solve rate distribution
# ---------------------------------------------------------------------------

def plot_solve_rate_distribution(
    n_sites: int,
    budget: int,
    results: list[tuple[Program, dict]],
    cfg: GameConfig,
    save_path: str | None = None,
):
    """Histogram of solve rates across all programs at a given budget."""
    import matplotlib.pyplot as plt

    if not results:
        print(f"No results to plot for budget={budget}")
        return

    n_inits = comb(n_sites, cfg.n_ones)
    solve_rates = [m["solve_rate"] for _, m in results]

    # Bins aligned to exact fractions: 0/n, 1/n, ..., n/n
    bin_edges = [i / n_inits - 0.5 / n_inits for i in range(n_inits + 1)]
    bin_edges.append((n_inits + 0.5) / n_inits)
    bin_centers = [i / n_inits for i in range(n_inits + 1)]

    sr_counts = Counter(round(sr, 10) for sr in solve_rates)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_heights = [sr_counts.get(round(c, 10), 0) for c in bin_centers]
    bar_colors = ["#E74C3C" if c < 0.5 else "#F39C12" if c < 1.0 else "#2ECC71"
                  for c in bin_centers]
    bars = ax.bar(bin_centers, bar_heights, width=0.8 / n_inits,
                  color=bar_colors, edgecolor="white")

    for center, height, bar in zip(bin_centers, bar_heights, bars):
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                    str(height), ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    # Annotate best program(s)
    best_sr = max(solve_rates)
    best_progs = [_oneline_pretty(p) for p, m in results
                  if abs(m["solve_rate"] - best_sr) < 1e-9]
    best_label = best_progs[0][:45]
    if len(best_progs) > 1:
        best_label += f" (+{len(best_progs)-1} more)"
    ax.annotate(
        f"Best: {best_label}",
        xy=(best_sr, sr_counts[round(best_sr, 10)]),
        xytext=(0.02, 0.95), textcoords="axes fraction",
        fontsize=8, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{i}/{n_inits}" for i in range(n_inits + 1)])
    ax.set_xlabel(f"Solve Rate (out of {n_inits} initial states)")
    ax.set_ylabel("Number of Programs")
    ax.set_title(
        f"Solve Rate Distribution: N={n_sites}, L={budget} "
        f"({len(results)} programs)"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Comparative state evolution
# ---------------------------------------------------------------------------

def _run_episode_with_x0(
    prog: Program,
    n_sites: int,
    x0: np.ndarray,
    cfg: GameConfig,
):
    """Run a single episode with a specific initial state."""
    env = cfg.make_env(n_sites, frozen_states=[x0])
    env.reset()
    return run_policy_episode(env, prog)


def _draw_evolution_subplot(
    ax, prog: Program, result, n_sites: int, title: str,
):
    """Draw a state-evolution grid on a given matplotlib axes."""
    from matplotlib.colors import ListedColormap

    states = [step.state for step in result.steps]
    states.append(result.final_state)
    actions = [step.action for step in result.steps]

    n_steps = len(states)
    grid = np.array(states)

    cmap = ListedColormap(["#E74C3C", "#2ECC71"])
    ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")

    for row in range(n_steps):
        for col in range(n_sites):
            val = int(grid[row, col])
            ax.text(col, row, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val == 0 else "black")
        if row < len(actions):
            flipped = actions[row]
            ax.annotate(
                "", xy=(flipped, row + 0.35), xytext=(flipped, row + 0.65),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2),
            )

    ax.set_xticks(range(n_sites))
    ax.set_xticklabels([f"bit {i}" for i in range(n_sites)])
    y_labels = [f"Step {i}" for i in range(n_steps - 1)] + ["Final"]
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(y_labels)

    status = "SOLVED" if result.solved else "NOT SOLVED"
    ax.set_title(
        f"{title}\n{status} | "
        f"Steps: {result.total_env_steps} | Ops: {result.total_interp_ops}",
        fontsize=9,
    )


def plot_comparative_evolution(
    best_prog: Program,
    bad_prog: Program,
    n_sites: int,
    cfg: GameConfig,
    save_path: str | None = None,
) -> dict:
    """Side-by-side state evolution: best program vs bad program, same initial state.

    Returns dict with best_steps, bad_steps, best_solved, bad_solved.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Pick an initial state where the best program solves and the bad one doesn't
    inits = all_initial_states(n_sites, cfg.n_ones)
    chosen_x0 = inits[0]
    for x0 in inits:
        best_result = _run_episode_with_x0(best_prog, n_sites, x0, cfg)
        bad_result = _run_episode_with_x0(bad_prog, n_sites, x0, cfg)
        if best_result.solved and not bad_result.solved:
            chosen_x0 = x0
            break
    else:
        best_result = _run_episode_with_x0(best_prog, n_sites, inits[0], cfg)
        bad_result = _run_episode_with_x0(bad_prog, n_sites, inits[0], cfg)

    n_rows_best = len(best_result.steps) + 1
    n_rows_bad = len(bad_result.steps) + 1
    max_rows = max(n_rows_best, n_rows_bad)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(max(12, n_sites * 2.5), max(3, max_rows * 0.5 + 1)),
    )

    _draw_evolution_subplot(
        ax1, best_prog, best_result, n_sites,
        f"Best: {_oneline_pretty(best_prog)[:55]}",
    )
    _draw_evolution_subplot(
        ax2, bad_prog, bad_result, n_sites,
        f"Bad: {_oneline_pretty(bad_prog)[:55]}",
    )

    legend_elements = [
        mpatches.Patch(facecolor="#E74C3C", label="0 (unset)"),
        mpatches.Patch(facecolor="#2ECC71", label="1 (set)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8)

    init_str = "[" + ", ".join(str(int(b)) for b in chosen_x0) + "]"
    fig.suptitle(
        f"Policy Comparison on Initial State {init_str} (N={n_sites})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return {
        "best_steps": best_result.total_env_steps,
        "bad_steps": bad_result.total_env_steps,
        "best_solved": best_result.solved,
        "bad_solved": bad_result.solved,
    }


# ---------------------------------------------------------------------------
# Guide table
# ---------------------------------------------------------------------------

def _format_guide_table(
    cumul_best: dict[int, float],
    results: list[tuple[Program, dict]],
    comp_info: dict,
    n_sites: int,
    eval_budget: int,
    n_inits: int,
) -> str:
    """Format a guide table explaining what each plot shows."""

    # Row 1: Best by budget progression (only show when cumul best changes)
    parts = []
    prev_solved = -1
    for b in sorted(cumul_best):
        cnt = count_programs(n_sites, b)
        if cnt == 0:
            continue
        cb = cumul_best[b]
        solved_n = int(round(cb * n_inits))
        if solved_n != prev_solved:
            parts.append(f"L={b}\u2192{solved_n}/{n_inits}")
            prev_solved = solved_n
    row1_insight = ", ".join(parts)
    if any(cb > 0.99 for cb in cumul_best.values()):
        row1_insight += " (green, perfect)"

    # Row 2: Solve rate distribution
    sr_counts = Counter(
        int(round(m["solve_rate"] * n_inits)) for _, m in results
    )
    labels = {0: "fail", n_inits: "perfect"}
    dist_parts = []
    for k in range(n_inits + 1):
        count = sr_counts.get(k, 0)
        cat = labels.get(k, "mediocre" if k < n_inits / 2 else "good")
        dist_parts.append(f"{count} {cat} ({k}/{n_inits})")
    row2_insight = ", ".join(dist_parts) + f" at L={eval_budget}"

    # Row 3: Comparative evolution
    best_s = comp_info["best_steps"]
    bad_s = comp_info["bad_steps"]
    best_word = "solves" if comp_info["best_solved"] else "fails"
    bad_word = "fails" if not comp_info["bad_solved"] else "solves"
    row3_insight = (
        f"Best {best_word} in {best_s} step{'s' if best_s != 1 else ''}; "
        f"bad toggles bit forever ({bad_s} steps, {bad_word})"
    )

    # Format table
    rows = [
        ("Plot", "Question answered", f"Key insight at N={n_sites}"),
        ("Best by budget (cumulative \u2264L)", "Does complexity help?", row1_insight),
        ("Solve rate distribution (histogram)", "How many programs are good?", row2_insight),
        ("Comparative evolution (side-by-side)", "What does a smart policy do?", row3_insight),
    ]

    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    sep = "  "
    lines = []
    for i, row in enumerate(rows):
        line = sep.join(cell.ljust(w) for cell, w in zip(row, widths))
        lines.append(line)
        if i == 0:
            lines.append(sep.join("-" * w for w in widths))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enumerate and analyze BitString DSL programs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              Interactive (prompts for each setting):
                python scripts/enumerate_dsl.py

              Override specific settings (skips those prompts):
                python scripts/enumerate_dsl.py --n_sites 5 --no-bit_flip

              Fully non-interactive (all defaults):
                python scripts/enumerate_dsl.py --no-interactive
        """),
    )
    parser.add_argument(
        "--n_sites", type=int, default=None,
        help="Number of bits (N). Default: 3",
    )
    parser.add_argument(
        "--max_budget", type=int, default=None,
        help="Max AST node count (L). Default: 8",
    )
    parser.add_argument(
        "--eval_budget", type=int, default=None,
        help="Budget level for detailed analysis. Default: auto",
    )
    parser.add_argument(
        "--bit_flip", dest="bit_flip", action="store_true", default=None,
        help="Enable bit-flip mode (Flip toggles 0/1). Default: True",
    )
    parser.add_argument(
        "--no-bit_flip", dest="bit_flip", action="store_false",
        help="Disable bit-flip (Flip only sets 0->1)",
    )
    parser.add_argument(
        "--sparse_reward", dest="sparse_reward", action="store_true", default=None,
        help="Use sparse reward (reward only at end). Default: False (dense)",
    )
    parser.add_argument(
        "--dense_reward", dest="sparse_reward", action="store_false",
        help="Use dense reward (reward every step)",
    )
    parser.add_argument(
        "--n_ones", type=int, default=None,
        help="Number of initial 1-bits. Default: 2",
    )
    parser.add_argument(
        "--potential", type=str, default=None,
        choices=list(POTENTIAL_REGISTRY.keys()),
        help="Potential function for reward shaping. Default: onemax",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip all interactive prompts, use defaults for unspecified args",
    )
    args = parser.parse_args()

    # --- 1. Interactive configuration ---
    cfg, N, max_L, eval_budget = prompt_game_config(args)
    n_inits = comb(N, cfg.n_ones)
    out = PLOT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # --- 2. Grammar structure ---
    print()
    print("=" * 70)
    print("=== Grammar Structure ===")
    print("The DSL has two node types: Programs (decision lists) and Conditions")
    print("(boolean tests on individual bits).")
    print()
    print("Programs are built from:")
    print("  Default(Flip(j))              [base case, 2 AST nodes]")
    print("  Ite(cond, Flip(j), else_prog) [if-then-else, >= 5 nodes]")
    print()
    print("Conditions are built from:")
    print("  IsZero(j)          [leaf, 1 node]")
    print("  Not(cond)          [negation, >= 2 nodes]")
    print("  And(left, right)   [conjunction, >= 3 nodes]")
    print()
    print(format_grammar_summary(N, max_L))
    print()
    print(format_grammar_debug(N, max_L))
    print()

    # --- 3. Derivation trace example ---
    trace_budget = None
    for b in range(5, max_L + 1):
        if count_programs(N, b) > 0:
            trace_budget = b
            break
    if trace_budget:
        print("The derivation engine expands programs left-to-right, filling")
        print("one hole at a time. Here is the first derivation at budget L="
              f"{trace_budget}:")
        print()
        print(trace_first_derivation(N, trace_budget))
        print()

    # --- 4. Behavioral analysis ---
    if eval_budget is None:
        # Prefer smallest Ite budget (L>=5) for interesting variety;
        # fall back to L=2 (Default-only) if no Ite programs exist
        for b in range(5, max_L + 1):
            if count_programs(N, b) > 0:
                eval_budget = b
                break
        if eval_budget is None and count_programs(N, 2) > 0:
            eval_budget = 2

    results: list[tuple[Program, dict]] = []
    if eval_budget is not None and count_programs(N, eval_budget) > 0:
        cnt = count_programs(N, eval_budget)
        print("=" * 70)
        print("=== Behavioral Evaluation ===")
        print(f"Each program is run on all {n_inits} initial states "
              f"(C({N},{cfg.n_ones})).")
        flip_desc = ("toggles 0\u21941" if cfg.bit_flip
                     else "only sets 0\u21921 (no undo)")
        print(f"A program \"solves\" if all bits become 1 within "
              f"max_steps={cfg.max_steps(N)}.")
        print(f"Game uses bit_flip={cfg.bit_flip} (Flip {flip_desc}), "
              f"potential={cfg.potential_name}.")
        print()

        if cnt <= MAX_EVAL_PROGRAMS:
            text, results = format_enumeration_analysis(N, eval_budget, cfg)
            print(text)
            print()
        else:
            print(f"Skipping detailed table for L={eval_budget} "
                  f"({cnt} programs exceeds {MAX_EVAL_PROGRAMS} cap).")
            print()

    # --- 5. Plots ---
    cumul_best_data: dict[int, float] | None = None
    comp_info: dict | None = None
    try:
        import matplotlib
        matplotlib.use("Agg")

        print("=" * 70)
        print("=== Generating Plots ===")
        print(f"Three plots saved to {out}/:")
        print("  1. Best by budget     -- does more complexity help?")
        print("  2. Solve distribution -- how are programs distributed?")
        print("  3. Comparison         -- what does a smart vs dumb policy look like?")
        print()

        # Plot 1: Best solve rate by budget (cumulative)
        cumul_best_data = plot_best_by_budget(
            N, max_L, cfg,
            save_path=str(out / f"best_by_budget_N{N}.png"),
        )

        # Plot 2: Solve rate distribution
        if results:
            plot_solve_rate_distribution(
                N, eval_budget, results, cfg,
                save_path=str(out / f"solve_dist_N{N}_L{eval_budget}.png"),
            )

        # Plot 3: Comparative evolution (good vs bad)
        if results:
            best_prog = max(results, key=lambda x: x[1]["solve_rate"])[0]
            worst_prog = min(results, key=lambda x: x[1]["solve_rate"])[0]
            if _oneline_pretty(best_prog) == _oneline_pretty(worst_prog):
                for b in range(2, max_L + 1):
                    for p in enumerate_programs(N, b):
                        if isinstance(p, Default):
                            worst_prog = p
                            break
                    if isinstance(worst_prog, Default):
                        break
            comp_info = plot_comparative_evolution(
                best_prog, worst_prog, N, cfg,
                save_path=str(out / f"comparison_N{N}.png"),
            )

    except ImportError:
        print("matplotlib not available, skipping plots.")

    # --- 6. Plot guide table ---
    if cumul_best_data and results and comp_info:
        print()
        print(_format_guide_table(
            cumul_best_data, results, comp_info,
            N, eval_budget, n_inits,
        ))
        print()

    # --- 7. Example trace for the best program ---
    if results:
        best_prog, best_m = max(
            results, key=lambda x: x[1]["solve_rate"]
        )
        print("=" * 70)
        print("=== Example Episode Trace (best program) ===")
        print(f"Program: {_oneline_pretty(best_prog)}")
        n_solved = int(best_m['solve_rate'] * best_m['n_episodes'])
        print(f"Solve rate: {n_solved}/{best_m['n_episodes']}")
        print()
        x0 = all_initial_states(N, cfg.n_ones)[0]
        env = cfg.make_env(N, frozen_states=[x0])
        env.reset()
        episode = run_policy_episode(env, best_prog)
        print(format_trace(episode, program=best_prog))


if __name__ == "__main__":
    main()
