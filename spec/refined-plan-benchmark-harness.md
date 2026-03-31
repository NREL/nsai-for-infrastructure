# Benchmark Harness for Doors Direct Play

**Date**: 2026-03-09

## Context

This repository contains an AlphaZero-based RL system for the Doors environment. The scientific goal of this stage is NOT about improving learning — it is about building a **common benchmark harness** so different algorithms (heuristic, tabular Q, DQN, MaskablePPO, AlphaZero) can be compared fairly and reproducibly.

The harness must support: common seeding, common evaluation, common logging (CSV + JSONL), common CLI, common plotting, and resumable runs.

---

## Codebase Mapping

| Plan Requirement | Existing Code | Classification |
|---|---|---|
| 1. Env compatibility | `DoorsPDDLLiteEnv` already has **compact** action space `Discrete(M+K+1)`. No padding. | **Exists** — no adapter needed |
| 2. Gymnasium/SB3 check_env | `observation_space = MultiBinary` but returns `float32` — minor mismatch. SB3 not installed. | **Modify** obs space declaration |
| 3. Train/eval separation | No separate constructors. `run_doors_direct.py` has `compute_solve_rate()` for greedy eval but coupled to AlphaZero Agent. | **New** — generic env factory + eval loop |
| 4. Result schema (CSV+JSONL) | `StatisticsManager` does JSONL only. No CSV. No standardized fields. | **New** — schema dataclass + dual writer |
| 5. Solve criterion | `compute_solve_rate()` in `run_doors_direct.py:522` tracks solve rate but no sustained-solve logic | **New** |
| 6. CLI | `scripts/run_doors_direct.py` uses argparse + interactive config. Not a `python -m` entry. | **New** — unified CLI |
| 7. Plotting | `plot_training_metrics()` in `run_doors_direct.py:552` exists but is AlphaZero-specific and per-run only | **New** — cross-algo comparison plots |
| 8. Docs | No benchmark protocol doc | **New** |
| 9. Fallback behavior | N/A | Built into stubs |
| 10. Acceptance tests | `test_doors_oracle.py` covers env audit; no benchmark smoke tests | **New** |

### Key Existing Utilities to Reuse

- `oracle.py` — `optimal_return(D)`, `optimal_steps(D)`, `oracle_action(obs, env)`, `run_oracle_episode(env)` — for oracle adapter and solve thresholds
- `DoorsGameConfig` in `doors_config.py` — parameterized env factory for arbitrary D
- `DoorsPDDLLiteEnv.action_masks(mode)` — mode="none" or "precondition" — maps directly to `--mask-mode`
- `StatisticsManager.save_jsonl()` — pattern for JSONL output
- `compute_solve_rate()` in `run_doors_direct.py:522` — reference for greedy eval loop (but coupled to Agent, needs rewrite)
- `DoorsDirectConfig.build()` — returns `(game, net, agent, trainer, evaluator)` 5-tuple for AlphaZero

### Key Tensions

1. **SB3 not installed** — DQN/MaskablePPO adapters must be stubs with guarded imports
2. **No standard RL algorithms exist** — Only AlphaZero implemented; tabular Q-learning not implemented
3. **AlphaZero training loop is fundamentally different** from SB3 — iteration-based with gated acceptance vs `model.learn(total_timesteps)` with callbacks
4. **Existing eval is pit-based** (new vs old network) — plan wants standard episodic eval (N episodes, mean return/success)
5. **`observation_space = MultiBinary` but obs is `float32`** — Gymnasium check_env will flag this

---

## Architecture

```
src/alphazeropp/benchmark/
    __init__.py
    run.py              # CLI: python -m alphazeropp.benchmark.run
    env_factory.py      # make_train_env(), make_eval_env()
    eval_loop.py        # evaluate_policy(policy_fn, env_factory, n_episodes) -> [EpisodeResult]
    result_schema.py    # CheckpointResult dataclass + ResultWriter (CSV + JSONL)
    solve_criterion.py  # check_sustained_solve(checkpoints) -> solve_step | None
    plotting.py         # Cross-algo comparison plots from CSV files
    adapters/
        __init__.py
        base.py         # BenchmarkAlgorithm ABC with train_and_yield_checkpoints()
        oracle.py       # Wraps oracle.oracle_action
        random_agent.py # Random policy baseline
        alphazero.py    # Wraps DoorsDirectConfig + GatedTrainer
        sb3.py          # Guarded stubs for DQN, MaskablePPO (raises ImportError)
tests/
    test_benchmark_smoke.py   # Env, eval loop, schema, solve criterion, seeding
docs/
    benchmark_protocol.md
```

### Algorithm Plugin Interface

```python
class BenchmarkAlgorithm(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def train_and_yield_checkpoints(
        self, env_factory, eval_env_factory, total_steps, eval_interval,
        eval_episodes, seed,
    ) -> Iterator[CheckpointResult]: ...
```

The **generator pattern** lets:
- AlphaZero yield after each gated training iteration
- SB3 algorithms yield from callbacks every N steps
- Oracle/random yield a single checkpoint immediately (no training)

The harness calls this once and checks solve criteria incrementally.

### Result Schema

```python
@dataclass
class CheckpointResult:
    algorithm: str
    seed: int
    D: int
    locs_per_room: int
    mask_mode: str
    env_steps: int              # cumulative training env steps
    train_episodes: int
    learner_updates: int
    eval_checkpoint_idx: int
    eval_return_mean: float
    eval_return_std: float
    eval_success_rate: float
    eval_episode_length_mean: float
    max_room_reached_mean: float
    keys_picked_mean: float
    invalid_action_rate: float
    wall_clock_sec: float
    solved_flag: bool           # computed by solve criterion
    solve_env_steps: int | None
    solve_wall_clock_sec: float | None
    # Optional
    extra: dict = field(default_factory=dict)
```

### Eval Loop

```python
def evaluate_policy(
    policy_fn: Callable[[np.ndarray, DoorsPDDLLiteEnv], int],
    env_factory: Callable[[], DoorsPDDLLiteEnv],
    n_episodes: int = 100,
    seed: int = 0,
) -> list[EpisodeResult]:
```

Runs N episodes. Tracks per-episode: return, steps, solved, invalid_action_count, max_room_reached, keys_picked. Invalid actions detected by checking `env.action_masks("precondition")` before each step.

### Solve Criterion

Solved when for 3 consecutive checkpoints:
- `eval_success_rate >= 0.95`
- `eval_return_mean >= 0.95 * oracle_return(D)`

Returns earliest `(solve_env_steps, solve_wall_clock_sec)`.

---

## Implementation Plan

### Day 1: Core infrastructure (no algorithm training)

**Step 1**: `src/alphazeropp/benchmark/__init__.py` — empty

**Step 2**: `src/alphazeropp/benchmark/env_factory.py`
- `make_env(D, locs_per_room, mask_mode, seed) -> DoorsPDDLLiteEnv`
- Uses `DoorsGameConfig` internally, computes horizon via `compute_doors_derived_params`
- `mask_mode` stored as env attribute for eval loop to query
- Justification: `DoorsGameConfig.make_env()` already handles arbitrary D; this adds seed + mask_mode wrapping

**Step 3**: `src/alphazeropp/benchmark/eval_loop.py`
- `EpisodeResult` dataclass: return, steps, solved, invalid_actions, max_room_reached, keys_picked
- `evaluate_policy(policy_fn, env_factory, n_episodes, seed)` — generic eval loop
- `aggregate_episodes(results) -> dict` — compute means/stds for CheckpointResult fields
- Justification: `run_oracle_episode()` in oracle.py is the reference pattern but only works for oracle; this generalizes to any policy_fn

**Step 4**: `src/alphazeropp/benchmark/result_schema.py`
- `CheckpointResult` dataclass with all required fields
- `ResultWriter` class: `append_jsonl(path, result)`, `write_csv(path, results)`
- CSV uses `csv.DictWriter` with fixed column order
- Justification: follows `StatisticsManager.save_jsonl()` pattern for JSONL; adds CSV for pandas analysis

**Step 5**: `src/alphazeropp/benchmark/solve_criterion.py`
- `check_sustained_solve(checkpoints, D, success_threshold=0.95, return_ratio_threshold=0.95, sustained_count=3) -> (solve_env_steps, solve_wall_clock) | None`
- Pure function scanning checkpoint history
- Justification: simple sliding window check, no external deps

**Step 6**: `src/alphazeropp/benchmark/adapters/base.py`
- `BenchmarkAlgorithm` ABC

**Step 7**: `src/alphazeropp/benchmark/adapters/oracle.py`
- Wraps `oracle.oracle_action` into `BenchmarkAlgorithm`
- `train_and_yield_checkpoints` immediately evaluates and yields one checkpoint
- Justification: reuses existing `oracle.oracle_action()` directly

**Step 8**: `src/alphazeropp/benchmark/adapters/random_agent.py`
- Random policy: `env.action_space.sample()` (or masked sample if mask_mode=precondition)
- Same yield-once pattern as oracle

### Day 2: AlphaZero adapter + CLI

**Step 9**: `src/alphazeropp/benchmark/adapters/alphazero.py`
- Builds `DoorsDirectConfig` → calls `.build()` → gets `(game, net, agent, trainer, evaluator)`
- Wraps in `GatedTrainer`
- Training loop: calls `gated_trainer.train_iteration()` in a loop
- After each iteration: extracts greedy NN policy (forward pass, mask, argmax) and runs `evaluate_policy()`
- Yields `CheckpointResult` per iteration
- Env step counting: `n_games_per_train * avg_steps_per_game` per iteration (tracked from trainer stats)
- Seed handling: derives 4 agent seeds from single benchmark seed
- Justification: reuses `DoorsDirectConfig.build()` and `GatedTrainer.train_iteration()` unchanged; only the eval path is new (standard episodic vs pit-based)

**Step 10**: `src/alphazeropp/benchmark/run.py`
- argparse CLI: `--algo {oracle,random,alphazero}`, `--D`, `--seed`, `--mask-mode {none,precondition}`, `--locs-per-room`, `--total-iterations`, `--eval-episodes`, `--eval-interval`, `--output-dir`
- Creates output dir, instantiates algorithm adapter, runs `train_and_yield_checkpoints`, writes results
- Invocation: `python -m alphazeropp.benchmark.run --algo oracle --D 3 --seed 42`

**Step 11**: `src/alphazeropp/benchmark/adapters/sb3.py`
- Guarded imports: `try: from stable_baselines3 import DQN; ...`
- Class skeletons for `DQNAdapter`, `MaskablePPOAdapter` that raise `ImportError` with install instructions if SB3 not available
- Full method signatures defined so the contract is clear

### Day 3: Plotting + tests + docs

**Step 12**: `src/alphazeropp/benchmark/plotting.py`
- `plot_learning_curves(csv_paths, output_path, metric)` — return/success_rate vs env_steps, one line per seed with mean+std band
- `plot_wall_clock_comparison(csv_paths, output_path)` — bar chart across algos
- Reads CSV files, groups by algorithm, aggregates over seeds
- Uses matplotlib (already a dependency)

**Step 13**: `tests/test_benchmark_smoke.py`
- `test_env_factory_creates_valid_env` — D=2,3, check spaces, step, reset
- `test_eval_loop_oracle` — oracle achieves optimal_return(D) for D=2
- `test_eval_loop_random` — random policy returns < oracle_return
- `test_result_schema_roundtrip` — JSONL and CSV serialization/deserialization
- `test_solve_criterion_logic` — 3-consecutive detection, edge cases
- `test_seeding_determinism` — same seed produces same eval results
- `test_smoke_benchmark_run` — run oracle adapter end-to-end for D=2, verify CSV+JSONL output
- `test_gymnasium_check_env` — `gymnasium.utils.env_checker.check_env` passes (after obs space fix)

**Step 14**: `docs/benchmark_protocol.md`
- What the benchmark measures
- Train/eval separation rationale
- Metric definitions (all schema fields)
- Solve criterion definition
- How to reproduce: one run, one sweep
- Example output row

### Day 3 (also): Minor env fix

**Step 15**: Fix observation space declaration in `doors_pddl_lite.py:134`
- Change `spaces.MultiBinary(self._obs_size)` to `spaces.Box(0, 1, shape=(self._obs_size,), dtype=np.float32)`
- Justification: obs returned as float32 but MultiBinary expects int8. This causes check_env warnings. Box(0,1,float32) matches actual behavior.
- Risk: may break code that checks `isinstance(obs_space, MultiBinary)`. Search codebase first.

---

## Files Changed

| File | Action | Description |
|---|---|---|
| `src/alphazeropp/benchmark/__init__.py` | **Create** | Package init |
| `src/alphazeropp/benchmark/env_factory.py` | **Create** | Train/eval env constructors |
| `src/alphazeropp/benchmark/eval_loop.py` | **Create** | Generic episodic evaluation |
| `src/alphazeropp/benchmark/result_schema.py` | **Create** | CheckpointResult + CSV/JSONL I/O |
| `src/alphazeropp/benchmark/solve_criterion.py` | **Create** | Sustained solve detection |
| `src/alphazeropp/benchmark/plotting.py` | **Create** | Cross-algo comparison plots |
| `src/alphazeropp/benchmark/run.py` | **Create** | CLI entrypoint |
| `src/alphazeropp/benchmark/adapters/__init__.py` | **Create** | Package init |
| `src/alphazeropp/benchmark/adapters/base.py` | **Create** | BenchmarkAlgorithm ABC |
| `src/alphazeropp/benchmark/adapters/oracle.py` | **Create** | Oracle adapter |
| `src/alphazeropp/benchmark/adapters/random_agent.py` | **Create** | Random baseline adapter |
| `src/alphazeropp/benchmark/adapters/alphazero.py` | **Create** | AlphaZero adapter |
| `src/alphazeropp/benchmark/adapters/sb3.py` | **Create** | SB3 stubs (guarded) |
| `tests/test_benchmark_smoke.py` | **Create** | Smoke tests |
| `docs/benchmark_protocol.md` | **Create** | Protocol documentation |
| `src/alphazeropp/instances/doors/doors_pddl_lite.py:134` | **Modify** | MultiBinary → Box(0,1,float32) |

**Existing files reused (not modified):**
- `src/alphazeropp/instances/doors/oracle.py` — optimal_return, oracle_action
- `src/alphazeropp/instances/doors/dsl/doors_config.py` — DoorsGameConfig, compute_doors_derived_params
- `src/alphazeropp/instances/doors/config.py` — DoorsDirectConfig
- `src/alphazeropp/training/gated_trainer.py` — GatedTrainer
- `src/alphazeropp/utils/statistics.py` — StatisticsManager (pattern reference)

---

## Verification

```bash
# Run smoke tests
pytest tests/test_benchmark_smoke.py -v

# Run existing tests to verify no regression (esp. after obs space fix)
pytest tests/ -v

# Run oracle benchmark (sanity check)
python -m alphazeropp.benchmark.run --algo oracle --D 3 --seed 42 --output-dir /tmp/bench_test

# Run random baseline benchmark
python -m alphazeropp.benchmark.run --algo random --D 3 --seed 42 --output-dir /tmp/bench_test

# Run tiny AlphaZero benchmark (D=2, 3 iterations)
python -m alphazeropp.benchmark.run --algo alphazero --D 2 --seed 42 \
    --total-iterations 3 --eval-episodes 10 --output-dir /tmp/bench_test

# Multi-seed sweep
for seed in 1 2; do
  python -m alphazeropp.benchmark.run --algo oracle --D 3 --seed $seed \
      --output-dir /tmp/bench_sweep
done

# Generate comparison plots
python -c "
from alphazeropp.benchmark.plotting import plot_learning_curves
from pathlib import Path
csvs = list(Path('/tmp/bench_sweep').glob('*.csv'))
plot_learning_curves(csvs, Path('/tmp/bench_sweep/curves.png'))
"

# Verify output files
cat /tmp/bench_test/oracle_D3_seed42.jsonl | head -1  # JSONL row
head -2 /tmp/bench_test/oracle_D3_seed42.csv          # CSV header + row
```

---

## Post-Plan: Save to spec/

Save this plan to `spec/refined-plan-benchmark-harness.md` (create `spec/` directory if needed).

---

## Resolved Design Notes

1. **Observation space fix**: Verified — only `doors_pddl_lite.py:134` uses MultiBinary in the Doors pipeline. `nsai_experiments/new_games_old_engine/helpers.py:39` checks `isinstance(space, MultiBinary)` but is legacy code not used by Doors. Safe to change.

2. **AlphaZero env step counting**: Trainer does NOT track cumulative env steps. `n_steps = len(result[0])` is computed per-game but not accumulated. The AlphaZero adapter will sum steps from `train_example_sets` returned by `_collect_training_data()` (each element is `(examples, total_reward)` where `len(examples)` = steps). The adapter accumulates this across iterations.

3. **mask_mode in eval**: Eval should match training mask_mode by default (measures what the agent actually learned under those conditions). The CLI `--mask-mode` applies to both train and eval envs.
