# Comparison Script Spec: Direct Play vs Grammar Game

## 1. Purpose

`scripts/compare_doors_experiments.py` loads experiment results from Direct Play and Grammar Game training runs on the Doors environment, produces side-by-side comparison plots and a summary table.

This script answers: **Given comparable computational budget, which approach solves Doors faster?**

---

## 2. Context: The Two Approaches

### Direct Play
AlphaZero agent observes the environment state directly (39-dim binary vector for D=10) and selects from 39 env actions (20 MOVE_TO + 9 PICK + 1 NOOP + 9 invalid). Uses an MLP policy-value network. Gets shaped rewards every step (-0.01/step, +0.1/unlock, +1.0/goal).

**Training script**: `scripts/run_doors_direct.py`
**Config**: `src/alphazeropp/instances/doors/config.py` (DoorsDirectConfig)
**Output dir**: `experiments/doors_direct/<timestamp>_D<n>_L<m>_...`

### Grammar Game
AlphaZero agent synthesizes a DSL program by selecting grammar productions on an AST state. The completed program is then executed as a reactive policy on the environment. Uses a Transformer policy-value network. Gets sparse reward (0 until program is complete, then leaf_evaluator score).

**Training script**: `scripts/run_doors_derivation.py`
**Config**: `src/alphazeropp/instances/doors/dsl/derivation_config.py` (DoorsDerivationConfig)
**Output dir**: `experiments/doors_derivation/<timestamp>_D<n>_<and_tag>_N<sites>_L<budget>_...`

---

## 3. CLI Interface

```bash
python scripts/compare_doors_experiments.py \
  --direct experiments/doors_direct/<dirname> \
  --grammar experiments/doors_derivation/<dirname> \
  --output experiments/comparison/<output_dir>/
```

- `--direct`: Path to a Direct Play experiment directory (optional, can provide multiple)
- `--grammar`: Path to a Grammar Game experiment directory (optional, can provide multiple)
- `--output`: Output directory for plots and summary (default: `experiments/comparison/`)

At least one of `--direct` or `--grammar` must be provided.

---

## 4. Input Data Schemas

### 4.1 Direct Play: `iteration_log.jsonl`

Each line is a JSON object (one per training iteration):

```json
{
  "iteration": 1,
  "gate_score": 0.6,
  "accepted": true,
  "solve_rate": 0.15,
  "avg_train_reward": -0.12,
  "avg_eval_reward": -0.08,
  "best_solve_rate": 0.15,
  "wall_clock_s": 45.2
}
```

Source: `scripts/run_doors_direct.py` (lines 958-967)

**Key fields for comparison:**
- `iteration` (int): 1-indexed iteration number
- `solve_rate` (float): Fraction of 20 greedy evaluation episodes that reach the goal
- `avg_eval_reward` (float): Mean reward across evaluation episodes
- `best_solve_rate` (float): Running maximum of solve_rate
- `wall_clock_s` (float): Wall clock seconds for this iteration
- `gate_score` (float): Win rate of new vs old network (0-1)
- `accepted` (bool): Whether new network was accepted

### 4.2 Direct Play: `train_stats.jsonl`

Written by `Trainer.statistics_manager.save_jsonl()`. Each line contains training metrics from one iteration:

```json
{
  "timestamp": 1709500000,
  "train_loss": 0.45,
  "train_loss_policy": 0.30,
  "train_loss_value": 0.15,
  "num_examples": 2000,
  "avg_reward": 0.3,
  "gate_score": 0.6,
  "gate_accepted": true
}
```

The `gate_score` and `gate_accepted` fields are appended by `GatedTrainer` (see `src/alphazeropp/training/gated_trainer.py` lines 71-72).

### 4.3 Direct Play: `eval_stats.jsonl`

Written by `Evaluator.statistics_manager.save_jsonl()`:

```json
{
  "timestamp": 1709500100,
  "new_rewards_mean": 0.5,
  "new_rewards_std": 0.3,
  "old_rewards_mean": 0.2,
  "old_rewards_std": 0.4
}
```

### 4.4 Grammar Game: `program_log.jsonl`

Each line records the best program found at each iteration:

```json
{
  "iteration": 1,
  "best_program": "if Not(IsZero(1)):\n  Flip(4)\nelse:\n  Flip(3)",
  "best_solve_rate": 0.5,
  "best_avg_reward": 0.3,
  "unique_programs": 23
}
```

Source: `src/alphazeropp/utils/derivation_utils.py` (lines 345-353)

### 4.5 Grammar Game: `train_stats.jsonl` and `eval_stats.jsonl`

Same schema as Direct Play (from shared GatedTrainer infrastructure).

### 4.6 Both: `config.json`

```json
{
  "game": {"kwargs": {"num_rooms": 10, ...}},
  "agent": {"mcts_params": {"n_simulations": 120, ...}},
  "trainer": {"n_games_per_train": 50, ...},
  "run": {"n_iterations": 50, ...}
}
```

---

## 5. Unified Comparison Metrics

| Metric | Direct Play Source | Grammar Source | Comparable? |
|--------|-------------------|---------------|-------------|
| solve_rate | `iteration_log.jsonl` → `solve_rate` | `program_log.jsonl` → `best_solve_rate` | Yes (primary) |
| best_solve_rate | `iteration_log.jsonl` → `best_solve_rate` | `program_log.jsonl` → `best_solve_rate` | Yes |
| avg_reward | `iteration_log.jsonl` → `avg_eval_reward` | `program_log.jsonl` → `best_avg_reward` | Caution: different scales |
| gate_score | `train_stats.jsonl` → `gate_score` | `train_stats.jsonl` → `gate_score` | Yes |
| accepted | `train_stats.jsonl` → `gate_accepted` | `train_stats.jsonl` → `gate_accepted` | Yes |
| wall_clock_s | `iteration_log.jsonl` → `wall_clock_s` | Computed from timestamps | Yes |
| policy_loss | `train_stats.jsonl` → `train_loss_policy` | `train_stats.jsonl` → `train_loss_policy` | Yes |
| value_loss | `train_stats.jsonl` → `train_loss_value` | `train_stats.jsonl` → `train_loss_value` | Yes |

**Primary metric**: `solve_rate` -- directly comparable between approaches.

**Reward caveat**: Direct Play rewards are shaped environment rewards (range ~[-1, +1.71] for D=10). Grammar Game "rewards" depend on the metric (e.g., `weighted` = 0.7 × solve_rate + 0.3 × avg_reward, range [0, 1]). These are NOT directly comparable. Use `solve_rate` for cross-approach comparison.

---

## 6. Output Plots

### 6.1 Main Figure: 2x2 Grid

**Plot 1 (top-left): Solve Rate vs Iteration**
- One line per experiment (labeled "Direct Play", "Grammar Game", etc.)
- X-axis: iteration (1..N)
- Y-axis: solve_rate (0..1)
- Horizontal dashed green line at 1.0 (optimal)
- Use `best_solve_rate` (monotonically non-decreasing)

**Plot 2 (top-right): Cumulative Wall Clock**
- X-axis: iteration
- Y-axis: cumulative wall_clock_s (seconds)
- Shows which approach is faster in real time

**Plot 3 (bottom-left): Training Losses**
- Two lines per experiment: policy loss (solid), value loss (dashed)
- X-axis: iteration
- Y-axis: loss (log scale)

**Plot 4 (bottom-right): Gate Acceptance Rate**
- Running acceptance ratio = cumulative_accepted / iteration
- X-axis: iteration
- Y-axis: acceptance rate (0..1)

### 6.2 Optional Grammar-Only Panel

If `--grammar` is provided, add a secondary figure:

**Plot 5: Unique Programs Explored**
- X-axis: iteration
- Y-axis: cumulative unique programs from `program_log.jsonl`

**Plot 6: Best Program Display**
- Text panel showing the final best program's `pretty()` output

---

## 7. Summary Table

Printed to terminal and saved as `comparison_summary.md` in the output directory:

```
| Metric                    | Direct Play     | Grammar Game    |
|---------------------------|-----------------|-----------------|
| Final solve_rate          | X.XX            | X.XX            |
| Iters to solve_rate=1.0   | N or "never"    | N or "never"    |
| Best solve_rate           | X.XX            | X.XX            |
| Final avg_reward          | +X.XXX          | +X.XXX          |
| Total wall clock (s)      | X.X             | X.X             |
| Gate acceptance rate       | X.XX            | X.XX            |
| MCTS sims/move            | N               | N               |
| Games/iteration           | N               | N               |
| Total iterations          | N               | N               |
| Best program              | N/A             | <pretty_print>  |
| Unique programs explored  | N/A             | N               |
```

---

## 8. Validation Checks

The script should warn (not error) on:
1. **Config mismatch**: If `n_games_per_train` or `n_iterations` differ between experiments
2. **Missing files**: If any expected JSONL file is missing, skip that metric
3. **Non-monotone best_solve_rate**: Indicates a logging bug

---

## 9. Reference Baselines

Overlay on solve_rate plot:
- **Random agent**: solve_rate ~ 0.001 (horizontal gray dashed line, labeled "Random")
- **Optimal**: solve_rate = 1.0 (horizontal green dashed line, labeled "Optimal")

For D=10, optimal_reward = +1.71. Overlay on reward plot as horizontal green dashed.

These values can be computed by running `python scripts/run_doors_baselines.py` or hardcoded as constants.

---

## 10. Dependencies

- `matplotlib`, `pandas` (already used by existing plotting code in the project)
- `json` (stdlib)
- `argparse` (stdlib)
- `pathlib` (stdlib)

---

## 11. Existing Code to Reuse

- **Plot style**: follow `scripts/run_doors_direct.py` for matplotlib figure setup, subplot grid, and save logic
- **JSONL loading**: `open(path).readlines()` + `json.loads()` per line (used throughout existing scripts)
- **Sweep overlay pattern**: `scripts/sweep_doors_hyperparams.py` loads multiple experiment dirs and overlays curves -- reuse its loading pattern
- **Derivation plotting**: `src/alphazeropp/utils/derivation_utils.py:plot_training_metrics()` for reference on grammar-side subplot layout
