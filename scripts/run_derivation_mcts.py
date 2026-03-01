#!/usr/bin/env python3
"""
MCTS-guided program synthesis for the BitString DSL.

Uses MCTS to search the derivation grammar for high-quality BitString
policies, instead of exhaustive enumeration. Each "round" plays out a
full derivation from ProgramHole(budget) to a complete program, guided
by MCTS simulations at each expansion step.

Usage:
    python scripts/run_derivation_mcts.py                            # defaults
    python scripts/run_derivation_mcts.py --budget 8 --n_rounds 10   # custom
    python scripts/run_derivation_mcts.py --metric solve_rate        # different metric
    python scripts/run_derivation_mcts.py --no-interactive           # skip prompts
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from math import comb
from pathlib import Path

import numpy as np

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from alphazeropp.instances.bitstring.dsl.ast_nodes import (
    Flip, IsZero, Ite, Default, Program,
)
from alphazeropp.instances.bitstring.dsl.interpreter import (
    run_policy_episode, format_trace,
)
from alphazeropp.instances.bitstring.dsl.budget_grammar import (
    count_programs, format_grammar_summary, enumerate_programs,
)
from alphazeropp.instances.bitstring.dsl.derivation import (
    DerivationState, format_derivation_trace,
)
from alphazeropp.instances.bitstring.dsl.game_config import (
    GameConfig, all_initial_states,
)
from alphazeropp.instances.bitstring.dsl.leaf_evaluator import (
    LeafEvaluator, VALID_METRICS,
)
from alphazeropp.instances.bitstring.dsl.derivation_game import (
    DerivationGame, UniformPolicyValueNet, compute_max_productions,
)
from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY
from alphazeropp.core.mcts import MCTS
from alphazeropp.utils.statistics import StatisticsManager
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("results/derivation_mcts")


# ---------------------------------------------------------------------------
# One-line pretty helper
# ---------------------------------------------------------------------------
def _oneline(prog: Program) -> str:
    return prog.pretty().replace("\n", " | ")


# ---------------------------------------------------------------------------
# Phase 1: Configuration & Grammar Overview
# ---------------------------------------------------------------------------

def print_configuration(
    n_sites: int, budget: int, cfg: GameConfig,
    n_simulations: int, n_rounds: int, metric: str,
    c_exploration: float, temperature: float,
    penalty_lambda: float, blend_alpha: float,
):
    n_inits = comb(n_sites, cfg.n_ones)
    total_programs = count_programs(n_sites, budget)
    max_prods = compute_max_productions(budget, n_sites)

    print("=" * 70)
    print("=== MCTS Program Synthesis ===")
    print("Goal: Use MCTS to search the derivation grammar for high-quality")
    print("      BitString policies, instead of exhaustive enumeration.")
    print()
    print(f"Game: N={n_sites} bits, bit_flip={cfg.bit_flip}, n_ones={cfg.n_ones}")
    inits = all_initial_states(n_sites, cfg.n_ones)
    state_strs = [
        "[" + ",".join(str(int(b)) for b in s) + "]"
        for s in inits[:5]
    ]
    suffix = f" ..." if n_inits > 5 else ""
    print(f"  -> {n_inits} initial states: {', '.join(state_strs)}{suffix}")
    print(f"  -> max_steps = {cfg.max_steps(n_sites)}"
          f" ({'sparse' if cfg.sparse_reward else 'dense'} reward)")
    print(f"  -> potential = {cfg.potential_name}")
    print()
    print(f"Grammar: budget L={budget} ({total_programs} total programs)")
    print(f"  -> Action space: Discrete({max_prods}) "
          f"(max productions at any step)")
    print()
    print(f"MCTS: {n_simulations} simulations/step, "
          f"c_exploration={c_exploration}, temperature={temperature}")
    print(f"Rounds: {n_rounds} (each round = one full derivation)")
    print()

    # Metric explanation
    print("=" * 70)
    print("=== Leaf Evaluation Metric ===")
    print(f"When MCTS reaches a complete program, it runs the program on all")
    print(f"{n_inits} frozen initial states and computes a scalar value:")
    print()
    print("  avg_reward:        mean shaped reward (continuous, best gradient)")
    print(f"  solve_rate:        fraction of states solved "
          f"(discrete: 0/{n_inits}..{n_inits}/{n_inits})")
    print(f"  penalized_reward:  avg_reward - {penalty_lambda} * avg_ops/max_ops "
          f"(prefers efficient programs)")
    print(f"  weighted:          {blend_alpha} * solve_rate + "
          f"{1-blend_alpha:.1f} * avg_reward (blend)")
    print()
    print(f"Currently using: {metric}")
    print(f"Change via: --metric <name>")
    print()

    # Grammar summary
    print("=" * 70)
    print(format_grammar_summary(n_sites, budget))
    print()


def print_architecture(n_simulations, n_rounds):
    """Print the MCTS program synthesis architecture diagram."""
    print("=" * 70)
    print("=== Algorithm Architecture: Pure MCTS (Planning Only) ===")
    print("=" * 70)
    print()
    print(f"  Outer loop: {n_rounds} independent rounds (different random seeds)")
    print("  Each round: build one complete derivation, guided by MCTS")
    print()
    print("  ┌── Round r ──────────────────────────────────────────────────────┐")
    print("  │                                                                  │")
    print("  │  game = DerivationGame(budget, n_sites, leaf_eval)    ← FRESH    │")
    print("  │  net  = UniformPolicyValueNet()  π=uniform, v=0  (no learning)   │")
    print("  │  mcts = MCTS(game, net, ...)                      ← NEW tree    │")
    print("  │")
    print("  │  Derivation steps (MCTS tree PERSISTS across all steps):")
    print("  │")
    print("  │    Step 1: state = [?program(budget)]    (one big hole)")
    print(f"  │      ┌─ {n_simulations} MCTS simulations ──────────────────────────────┐")
    print("  │      │  SELECT  → UCB = Q_norm + c · (1/K) · √N/(1+Nₐ)          │")
    print("  │      │  EXPAND  → try a production, reach new partial AST        │")
    print("  │      │            at leaf: query net → (uniform 1/K, value 0)    │")
    print("  │      │  BACKUP  → Q(s,a) = running_mean(leaf_eval(program))      │")
    print("  │      │            value ONLY comes from COMPLETE programs         │")
    print("  │      └────────────────────────────────────────────────────────────┘")
    print("  │      best_action = argmax(visit_counts)")
    print("  │      Apply production → e.g. expand into if-then-else")
    print("  │")
    print("  │    Step 2: state = [if ?cond: ?prog(3) else: ?prog(3)]")
    print(f"  │      {n_simulations} simulations  ← REUSES tree nodes from step 1")
    print("  │      Subtrees already explored are NOT re-expanded")
    print("  │      Q-values from step 1 inform step 2 decisions")
    print("  │")
    print("  │    Steps 3..K: fill remaining holes left-to-right")
    print("  │      Tree keeps growing, Q-values accumulate across all steps")
    print("  │")
    print("  │    Terminal: all holes filled → complete Program")
    print("  │      Score = leaf_eval(program)    ← run on all initial states")
    print("  │")
    print("  │  Record best program.  DISCARD entire MCTS tree.")
    print("  │  No training.  No net update.  Next round starts fresh.")
    print("  │                                                                  │")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()
    print("  Key properties:")
    print("    WITHIN a round:  tree PERSISTS → Q-values reused across steps")
    print("    ACROSS rounds:   tree DISCARDED → each round is independent")
    print("    No learning:     net always returns π=uniform, v=0")
    print("    Signal source:   leaf_eval(program) at terminal derivation states")
    print()
    print("  Comparison with AlphaZero (run_bitstring.py):")
    print("    ┌────────────────────┬──────────────────┬──────────────────────┐")
    print("    │                    │ AlphaZero        │ This script          │")
    print("    ├────────────────────┼──────────────────┼──────────────────────┤")
    print("    │ Tree lifetime      │ 1 move           │ 1 round (all steps)  │")
    print("    │ Knowledge carrier  │ Neural net f_θ   │ Nothing              │")
    print("    │ Net provides       │ Learned π(s),v(s)│ Uniform 1/K, v=0    │")
    print("    │ Training           │ Yes (SGD)        │ None                 │")
    print("    │ Cross-episode      │ Net improves     │ Independent restarts │")
    print("    └────────────────────┴──────────────────┴──────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Phase 2: Ground Truth via Exhaustive Enumeration
# ---------------------------------------------------------------------------

def run_ground_truth(
    n_sites: int, budget: int, leaf_eval: LeafEvaluator,
) -> tuple[float, Program | None, list[tuple[Program, dict]]]:
    """Enumerate all programs and return (best_metric_value, best_program, all_results)."""
    total = count_programs(n_sites, budget)
    print("=" * 70)
    print("=== Ground Truth via Exhaustive Enumeration ===")
    print(f"Enumerating all {total} programs at budget={budget}...")

    programs = enumerate_programs(n_sites, budget)
    results: list[tuple[Program, dict]] = []
    best_value = float("-inf")
    best_prog = None

    for prog in programs:
        value = leaf_eval(prog)
        metrics = leaf_eval.get_all_metrics(prog)
        results.append((prog, metrics))
        if value > best_value:
            best_value = value
            best_prog = prog

    best_metrics = leaf_eval.get_all_metrics(best_prog)
    worst_prog = min(results, key=lambda x: leaf_eval(x[0]))[0]

    n_solved = int(best_metrics["solve_rate"] * best_metrics["n_episodes"])
    print(f"Best: {_oneline(best_prog)}")
    print(f"  -> Solves {n_solved}/{best_metrics['n_episodes']} "
          f"({best_metrics['solve_rate']*100:.1f}%)")
    print(f"  -> Avg reward: {best_metrics['avg_reward']:+.4f}")
    print(f"Worst: {_oneline(worst_prog)}")
    print()
    print(f"This is the 'ceiling' -- can MCTS find the optimal program")
    print(f"without enumerating all {total}?")
    print()

    return best_value, best_prog, results


# ---------------------------------------------------------------------------
# Phase 3: MCTS Search
# ---------------------------------------------------------------------------

def run_mcts_search(
    n_sites: int, budget: int, leaf_eval: LeafEvaluator,
    n_simulations: int, n_rounds: int,
    c_exploration: float, temperature: float,
    seed: int,
    stats_mgr: StatisticsManager,
    metric: str,
    gt_best_value: float | None = None,
) -> tuple[Program | None, float, list[dict]]:
    """Run MCTS search for n_rounds, return (best_program, best_value, round_logs)."""
    print("=" * 70)
    print("=== MCTS Search ===")
    print()

    best_program = None
    best_value = float("-inf")
    best_round = -1
    round_logs: list[dict] = []

    for r in range(n_rounds):
        np.random.seed(seed + r)

        game = DerivationGame(budget, n_sites, leaf_eval)
        game.reset_wrapper()
        net = UniformPolicyValueNet(game._max_productions)
        mcts = MCTS(
            game, net,
            n_simulations=n_simulations,
            temperature=temperature,
            c_exploration=c_exploration,
        )

        # Track derivation steps for this round
        derivation_states = [DerivationState.initial(budget)]
        derivation_prods = []

        step_count = 0
        while not game.terminated and not game.truncated:
            probs = mcts.perform_simulations(None)
            action = int(np.argmax(probs))
            game.step_wrapper(action)
            step_count += 1

            # Record derivation
            if game.info and "production" in game.info:
                derivation_prods.append(game.info["production"])
            derivation_states.append(
                DerivationState(root=game._deriv_state.root)
            )

        program = game.get_program()
        eval_stats = leaf_eval.stats()

        if program is not None:
            value = leaf_eval(program)
            metrics = leaf_eval.get_all_metrics(program)
        else:
            value = 0.0
            metrics = {"solve_rate": 0.0, "avg_reward": 0.0,
                        "avg_steps": 0.0, "avg_ops": 0.0, "n_episodes": 0}

        if value > best_value:
            best_value = value
            best_program = program
            best_round = r

        # Print round summary
        n_inits = metrics.get("n_episodes", 0)
        n_solved = int(metrics["solve_rate"] * n_inits) if n_inits else 0
        is_best = " <-- NEW BEST" if value >= best_value else ""
        is_dead = " [DEAD END]" if program is None else ""

        print(f"Round {r+1}/{n_rounds}:")
        if program is not None:
            print(f"  Program: {_oneline(program)}")
            print(f"  Solve rate: {n_solved}/{n_inits} "
                  f"({metrics['solve_rate']*100:.1f}%)  "
                  f"Avg reward: {metrics['avg_reward']:+.4f}{is_best}")
        else:
            print(f"  Dead-end derivation (no valid program){is_dead}")
        print(f"  Unique programs evaluated: {eval_stats['unique_programs']}  "
              f"Cache hits: {eval_stats['cache_hits']}")

        if gt_best_value is not None and best_value >= gt_best_value - 1e-9:
            print(f"  *** OPTIMAL FOUND (matches exhaustive enumeration best)")
        print()

        # Log
        log = {
            "round": r + 1,
            "metric": metric,
            "mcts_simulations_per_step": n_simulations,
            "derivation_steps": step_count,
            "program": _oneline(program) if program else "DEAD_END",
            "metric_value": round(value, 6),
            "solve_rate": round(metrics["solve_rate"], 6),
            "avg_reward": round(metrics["avg_reward"], 6),
            "best_metric_value": round(best_value, 6),
            "unique_programs_seen": eval_stats["unique_programs"],
            "cache_hits": eval_stats["cache_hits"],
            "total_env_steps": eval_stats["total_env_steps"],
            "total_interp_ops": eval_stats["total_interp_ops"],
            "seed": seed + r,
        }
        round_logs.append(log)
        stats_mgr.record(log)

    # Search complete summary
    print("=" * 70)
    print("=== Search Complete ===")
    if best_program is not None:
        print(f"Best program: {_oneline(best_program)}")
        best_metrics = leaf_eval.get_all_metrics(best_program)
        n_inits = best_metrics["n_episodes"]
        n_solved = int(best_metrics["solve_rate"] * n_inits)
        print(f"  Solve rate: {n_solved}/{n_inits} "
              f"({best_metrics['solve_rate']*100:.1f}%)")
        print(f"  Avg reward: {best_metrics['avg_reward']:+.4f}")
        print(f"  Found in round {best_round + 1}")
        if gt_best_value is not None:
            matches = best_value >= gt_best_value - 1e-9
            print(f"  Matches exhaustive-enumeration best: "
                  f"{'YES' if matches else 'NO'}")
    else:
        print("No valid program found (all rounds hit dead ends)")
    print()

    return best_program, best_value, round_logs


# ---------------------------------------------------------------------------
# Phase 4: Best Program Episode Trace
# ---------------------------------------------------------------------------

def print_episode_traces(
    program: Program,
    n_sites: int,
    cfg: GameConfig,
):
    print("=" * 70)
    print("=== Best Program in Action ===")
    print()

    inits = all_initial_states(n_sites, cfg.n_ones)
    for i, x0 in enumerate(inits):
        env = cfg.make_env(n_sites, frozen_states=[x0])
        env.reset()
        result = run_policy_episode(env, program)
        state_str = "[" + ", ".join(str(int(b)) for b in x0) + "]"

        if i == 0:
            # Full trace for first state
            print(format_trace(result, program=program))
        else:
            # Summary for remaining states
            status = "SOLVED" if result.solved else "NOT SOLVED"
            print(f"Initial state: {state_str} -> {status} "
                  f"in {result.total_env_steps} steps, "
                  f"reward={result.cumulative_reward:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Phase 5: Derivation Trace
# ---------------------------------------------------------------------------

def print_derivation_trace(program: Program, n_sites: int, budget: int):
    """Show how MCTS constructed the program via grammar derivation."""
    print("=" * 70)
    print("=== How MCTS Built This Program ===")
    print("The derivation expands holes left-to-right:")
    print()

    # Reconstruct derivation by replaying first-match
    from alphazeropp.instances.bitstring.dsl.derivation import (
        _program_productions, _condition_productions,
    )

    states = []
    prods = []
    state = DerivationState.initial(budget)
    states.append(state)

    # Try to find the derivation that produces this exact program
    # by DFS matching
    target_pretty = program.pretty()

    def _find_derivation(st: DerivationState, depth=0):
        if st.is_terminal():
            if st.to_program().pretty() == target_pretty:
                return True
            return False
        legal = st.legal_productions(n_sites)
        for prod in legal:
            next_st = st.apply(prod)
            states.append(next_st)
            prods.append(prod)
            if _find_derivation(next_st, depth + 1):
                return True
            states.pop()
            prods.pop()
        return False

    if _find_derivation(state):
        print(format_derivation_trace(states, prods))
    else:
        print(f"(Could not reconstruct derivation trace for this program)")
    print()


# ---------------------------------------------------------------------------
# Phase 6: Summary Statistics
# ---------------------------------------------------------------------------

def print_summary(
    n_simulations: int, n_rounds: int,
    leaf_eval: LeafEvaluator,
    budget: int, n_sites: int,
):
    total_programs = count_programs(n_sites, budget)
    stats = leaf_eval.stats()

    print("=" * 70)
    print("=== Compute Budget ===")
    print(f"MCTS simulations:    {n_simulations}/step x {n_rounds} rounds")
    print(f"Programs evaluated:  {stats['unique_programs']} "
          f"(out of {total_programs} possible)  "
          f"<- {stats['unique_programs']*100/max(total_programs,1):.1f}% coverage")
    print(f"Cache hit rate:      {stats['cache_hits']}"
          f"/{stats['cache_hits'] + stats['eval_count']} "
          f"({stats['cache_hits']*100/max(stats['cache_hits']+stats['eval_count'],1):.1f}%)")
    print(f"Total env steps:     {stats['total_env_steps']}")
    print(f"Total interp ops:    {stats['total_interp_ops']}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(
    round_logs: list[dict],
    gt_results: list[tuple[Program, dict]] | None,
    best_program: Program | None,
    n_sites: int,
    budget: int,
    cfg: GameConfig,
    leaf_eval: LeafEvaluator,
    gt_best_value: float | None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("=== Generating Plots ===")
    print()

    # Plot 1: MCTS Convergence
    _plot_convergence(round_logs, gt_best_value, plt)

    # Plot 2: Program Quality Distribution
    if gt_results is not None:
        _plot_quality_distribution(
            leaf_eval, gt_results, n_sites, cfg, plt,
        )

    # Plot 3: Best vs Worst Program Execution
    if best_program is not None and gt_results is not None:
        _plot_comparison(
            best_program, gt_results, n_sites, cfg, plt,
        )

    # Plot 4: Derivation Depth Profile
    _plot_depth_profile(round_logs, plt)


def _plot_convergence(
    round_logs: list[dict],
    gt_best_value: float | None,
    plt,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    rounds = [log["round"] for log in round_logs]
    best_so_far = []
    running_best = float("-inf")
    for log in round_logs:
        running_best = max(running_best, log["solve_rate"])
        best_so_far.append(running_best)

    ax.plot(rounds, best_so_far, "o-", color="#4C72B0", linewidth=2,
            markersize=6, label="MCTS best-so-far (solve rate)")

    # Also plot per-round solve rate
    per_round = [log["solve_rate"] for log in round_logs]
    ax.plot(rounds, per_round, "x", color="#999999", markersize=5,
            alpha=0.6, label="Per-round solve rate")

    if gt_best_value is not None:
        ax.axhline(y=gt_best_value, color="#2ECC71", linestyle="--",
                    linewidth=2, alpha=0.7, label="Ground truth optimum")

    ax.set_xlabel("Round")
    ax.set_ylabel("Solve Rate")
    ax.set_title("MCTS Convergence: Best Solve Rate Over Rounds")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = str(OUT_DIR / "convergence.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_quality_distribution(
    leaf_eval: LeafEvaluator,
    gt_results: list[tuple[Program, dict]],
    n_sites: int,
    cfg: GameConfig,
    plt,
):
    from collections import Counter

    n_inits = comb(n_sites, cfg.n_ones)

    # Ground truth distribution
    gt_rates = [m["solve_rate"] for _, m in gt_results]
    gt_counts = Counter(round(sr, 10) for sr in gt_rates)

    # MCTS-discovered distribution
    mcts_rates = []
    for key, metrics in leaf_eval._full_cache.items():
        mcts_rates.append(metrics["solve_rate"])
    mcts_counts = Counter(round(sr, 10) for sr in mcts_rates)

    bin_centers = [i / n_inits for i in range(n_inits + 1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35 / n_inits

    # Ground truth bars (ghost)
    gt_heights = [gt_counts.get(round(c, 10), 0) for c in bin_centers]
    ax.bar([c - width for c in bin_centers], gt_heights, width=width * 1.8,
           color="#CCCCCC", edgecolor="white", alpha=0.7, label="All programs (enumeration)")

    # MCTS-discovered bars
    mcts_heights = [mcts_counts.get(round(c, 10), 0) for c in bin_centers]
    ax.bar([c + width for c in bin_centers], mcts_heights, width=width * 1.8,
           color="#4C72B0", edgecolor="white", alpha=0.9, label="MCTS-discovered")

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{i}/{n_inits}" for i in range(n_inits + 1)])
    ax.set_xlabel(f"Solve Rate (out of {n_inits} initial states)")
    ax.set_ylabel("Number of Programs")
    ax.set_title("Program Quality Distribution: MCTS vs Full Enumeration")
    ax.legend()

    path = str(OUT_DIR / "quality_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_comparison(
    best_program: Program,
    gt_results: list[tuple[Program, dict]],
    n_sites: int,
    cfg: GameConfig,
    plt,
):
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    # Find worst program for comparison
    worst_prog = min(gt_results, key=lambda x: x[1]["solve_rate"])[0]

    inits = all_initial_states(n_sites, cfg.n_ones)
    chosen_x0 = inits[0]

    # Try to find an initial state where best solves and worst doesn't
    for x0 in inits:
        env_b = cfg.make_env(n_sites, frozen_states=[x0])
        env_b.reset()
        res_b = run_policy_episode(env_b, best_program)
        env_w = cfg.make_env(n_sites, frozen_states=[x0])
        env_w.reset()
        res_w = run_policy_episode(env_w, worst_prog)
        if res_b.solved and not res_w.solved:
            chosen_x0 = x0
            break
    else:
        chosen_x0 = inits[0]

    env_b = cfg.make_env(n_sites, frozen_states=[chosen_x0])
    env_b.reset()
    best_result = run_policy_episode(env_b, best_program)
    env_w = cfg.make_env(n_sites, frozen_states=[chosen_x0])
    env_w.reset()
    worst_result = run_policy_episode(env_w, worst_prog)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, prog, result, title in [
        (ax1, best_program, best_result, f"MCTS Best: {_oneline(best_program)[:45]}"),
        (ax2, worst_prog, worst_result, f"Worst: {_oneline(worst_prog)[:45]}"),
    ]:
        states = [step.state for step in result.steps] + [result.final_state]
        actions = [step.action for step in result.steps]
        grid = np.array(states)
        n_steps = len(states)

        cmap = ListedColormap(["#E74C3C", "#2ECC71"])
        ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")

        for row in range(n_steps):
            for col in range(n_sites):
                val = int(grid[row, col])
                ax.text(col, row, str(val), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val == 0 else "black")
            if row < len(actions):
                ax.annotate(
                    "", xy=(actions[row], row + 0.35),
                    xytext=(actions[row], row + 0.65),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                )

        ax.set_xticks(range(n_sites))
        ax.set_xticklabels([f"bit {i}" for i in range(n_sites)])
        y_labels = [f"Step {i}" for i in range(n_steps - 1)] + ["Final"]
        ax.set_yticks(range(n_steps))
        ax.set_yticklabels(y_labels)
        status = "SOLVED" if result.solved else "NOT SOLVED"
        ax.set_title(f"{title}\n{status} | Steps: {result.total_env_steps}")

    legend_elements = [
        mpatches.Patch(facecolor="#E74C3C", label="0 (unset)"),
        mpatches.Patch(facecolor="#2ECC71", label="1 (set)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8)
    init_str = "[" + ", ".join(str(int(b)) for b in chosen_x0) + "]"
    fig.suptitle(f"Policy Comparison on {init_str} (N={n_sites})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    path = str(OUT_DIR / "comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_depth_profile(round_logs: list[dict], plt):
    from collections import defaultdict

    depth_data = defaultdict(list)
    for log in round_logs:
        depth = log["derivation_steps"]
        depth_data[depth].append(log["solve_rate"])

    if not depth_data:
        return

    depths = sorted(depth_data.keys())
    counts = [len(depth_data[d]) for d in depths]
    avg_quality = [np.mean(depth_data[d]) for d in depths]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(depths, counts, color="#4C72B0", alpha=0.7, label="Count")
    ax2.plot(depths, avg_quality, "ro-", linewidth=2, markersize=8,
             label="Avg solve rate")

    ax1.set_xlabel("Derivation Depth (expansion steps)")
    ax1.set_ylabel("Number of Programs", color="#4C72B0")
    ax2.set_ylabel("Avg Solve Rate", color="red")
    ax1.set_title("Derivation Depth Profile")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    path = str(OUT_DIR / "depth_profile.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Interactive config editing
# ---------------------------------------------------------------------------

def _build_sections(args):
    """Return sections for MCTS program synthesis config."""
    params = [
        # Problem
        ("n_sites",       args.n_sites,       attr_setter(args, "n_sites"),       "Number of bits in the bitstring",     None),
        ("n_ones",        args.n_ones,        attr_setter(args, "n_ones"),        "Number of 1s in initial states",      None),
        ("budget",        args.budget,        attr_setter(args, "budget"),        "AST node budget (program size)",      None),
        ("potential",     args.potential,      attr_setter(args, "potential"),     "Reward shaping function",             list(POTENTIAL_REGISTRY.keys())),
        # MCTS
        ("n_simulations", args.n_simulations, attr_setter(args, "n_simulations"), "MCTS rollouts per derivation step",   None),
        ("n_rounds",      args.n_rounds,      attr_setter(args, "n_rounds"),      "Complete derivations to run",         None),
        ("c_exploration", args.c_exploration,  attr_setter(args, "c_exploration"), "UCB exploration constant",            None),
        ("temperature",   args.temperature,    attr_setter(args, "temperature"),   "Action selection temperature",        None),
        # Evaluation
        ("metric",        args.metric,         attr_setter(args, "metric"),        "Leaf evaluation metric",              list(VALID_METRICS)),
    ]
    if args.metric == "penalized_reward":
        params.append(
            ("penalty_lambda", args.penalty_lambda, attr_setter(args, "penalty_lambda"),
             "Penalty weight for interp ops", None))
    if args.metric == "weighted":
        params.append(
            ("blend_alpha", args.blend_alpha, attr_setter(args, "blend_alpha"),
             "Weight of solve_rate in blend", None))
    # Run
    params.extend([
        ("seed",             args.seed,             attr_setter(args, "seed"),             "Random seed",                  None),
        ("skip_enumeration", args.skip_enumeration, attr_setter(args, "skip_enumeration"), "Skip ground-truth enumeration", None),
    ])

    all_params = build_param_list(params)

    problem_labels = {"n_sites", "n_ones", "budget", "potential"}
    mcts_labels = {"n_simulations", "n_rounds", "c_exploration", "temperature"}
    eval_labels = {"metric", "penalty_lambda", "blend_alpha"}
    run_labels = {"seed", "skip_enumeration"}

    return [
        ("Problem",    [p for p in all_params if p[1] in problem_labels]),
        ("MCTS",       [p for p in all_params if p[1] in mcts_labels]),
        ("Evaluation", [p for p in all_params if p[1] in eval_labels]),
        ("Run",        [p for p in all_params if p[1] in run_labels]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCTS-guided program synthesis for the BitString DSL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/run_derivation_mcts.py
              python scripts/run_derivation_mcts.py --budget 8 --n_rounds 10
              python scripts/run_derivation_mcts.py --metric solve_rate
              python scripts/run_derivation_mcts.py --no-interactive
        """),
    )
    parser.add_argument("--n_sites", type=int, default=3)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--n_simulations", type=int, default=200)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="avg_reward",
        choices=list(VALID_METRICS),
    )
    parser.add_argument("--penalty_lambda", type=float, default=0.1)
    parser.add_argument("--blend_alpha", type=float, default=0.5)
    parser.add_argument("--c_exploration", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--potential", type=str, default="onemax",
        choices=list(POTENTIAL_REGISTRY.keys()),
    )
    parser.add_argument("--n_ones", type=int, default=2)
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--skip-enumeration", action="store_true")
    args = parser.parse_args()

    # Interactive config editing
    if not args.no_interactive:
        interactive_edit("MCTS Program Synthesis Config", lambda: _build_sections(args))

    # Setup
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = GameConfig(
        bit_flip=True,
        sparse_reward=False,
        n_ones=args.n_ones,
        potential_name=args.potential,
    )
    frozen_states = all_initial_states(args.n_sites, args.n_ones)

    leaf_eval = LeafEvaluator(
        args.n_sites, frozen_states, cfg,
        metric=args.metric,
        penalty_lambda=args.penalty_lambda,
        blend_alpha=args.blend_alpha,
    )

    stats_mgr = StatisticsManager()

    # Phase 1: Configuration
    print_configuration(
        args.n_sites, args.budget, cfg,
        args.n_simulations, args.n_rounds, args.metric,
        args.c_exploration, args.temperature,
        args.penalty_lambda, args.blend_alpha,
    )

    # Architecture diagram
    print_architecture(args.n_simulations, args.n_rounds)

    # Phase 2: Ground Truth (optional)
    gt_best_value = None
    gt_best_prog = None
    gt_results = None
    total_programs = count_programs(args.n_sites, args.budget)
    if not args.skip_enumeration and total_programs <= 2000:
        gt_best_value, gt_best_prog, gt_results = run_ground_truth(
            args.n_sites, args.budget, leaf_eval,
        )

    # Phase 3: MCTS Search
    best_program, best_value, round_logs = run_mcts_search(
        args.n_sites, args.budget, leaf_eval,
        args.n_simulations, args.n_rounds,
        args.c_exploration, args.temperature,
        args.seed,
        stats_mgr, args.metric,
        gt_best_value,
    )

    # Phase 4: Best Program Episode Traces
    if best_program is not None:
        print_episode_traces(best_program, args.n_sites, cfg)

    # Phase 5: Derivation Trace
    if best_program is not None:
        print_derivation_trace(best_program, args.n_sites, args.budget)

    # Phase 6: Summary
    print_summary(
        args.n_simulations, args.n_rounds, leaf_eval,
        args.budget, args.n_sites,
    )

    # Plots
    generate_plots(
        round_logs, gt_results,
        best_program, args.n_sites, args.budget, cfg, leaf_eval,
        gt_best_value,
    )

    # Save JSONL
    jsonl_path = OUT_DIR / "run_log.jsonl"
    stats_mgr.save_jsonl(jsonl_path, append=False)
    print(f"\nRun log saved: {jsonl_path}")


if __name__ == "__main__":
    main()
