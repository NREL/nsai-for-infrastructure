# Phase 0: Baseline Stabilization & Instrumentation

_Make the system auditable before swapping the benchmark. Stable runs, consistent logging, diagnostic metrics that survive problem changes._

---

## Task 1 — Run Metadata

**Goal**: Record reproducibility info once per run.

**Where**: New helper in `scripts/run_derivation.py`. Write `{exp_dir}/run_metadata.jsonl` (single line).

**Schema**:
```json
{
  "timestamp": "2026-03-03T14:22:01Z",
  "git_hash": "ef19660",
  "git_dirty": true,
  "device": "mps",
  "seeds": {"mcts": 43, "train": 47, "eval": 23},
  "hyperparams": {
    "n_iterations": 30,
    "n_games_per_train": 40,
    "n_simulations": 200,
    "budget": 14,
    "n_sites": 6,
    "learning_rate": 0.001,
    "policy_weight": 2.0,
    "temperature": 1.0
  }
}
```

**Implementation**: Use `subprocess.run(["git", "rev-parse", "--short", "HEAD"])` for hash, `git diff --quiet` for dirty flag. Wrap in try/except for non-git environments.

**Files**: `scripts/run_derivation.py` (add ~20 lines after `cfg.save()`).

---

## Task 2 — Iteration-Level Diagnostic Metrics

**Goal**: Extend `program_log.jsonl` with metrics that diagnose search health independent of the benchmark.

**New fields** added to the existing `program_log` dict at `run_derivation.py:811`:

| Field | Source | Computation |
|-------|--------|-------------|
| `leaf_eval_mean` | `leaf_eval._cache` | `np.mean(list(cache.values()))` |
| `leaf_eval_std` | `leaf_eval._cache` | `np.std(list(cache.values()))` |
| `action_mask_density` | `game.get_action_mask()` | `mask.sum() / len(mask)` at fresh reset |
| `mcts_visit_entropy` | `trainer.all_training_examples` | Entropy of `move_probs` from latest iteration, averaged across steps |
| `productions_used_by_type` | `leaf_eval._program_cache` | Count AST node types across all cached programs |

**Entropy formula**: `H(p) = -sum(p * log(p))` where `p` is the MCTS visit distribution (already stored as `move_probs` in training examples).

**Production type counting**: Walk each program AST in `leaf_eval._program_cache.values()`, count occurrences of `Default`, `Ite`, `IsZero`, `Not`, `And`. Report as fractions of total nodes.

**Extended record example**:
```json
{
  "iteration": 5,
  "best_program": "if IsZero(0): Flip(0) ...",
  "best_solve_rate": 0.67,
  "best_avg_reward": 0.28,
  "unique_programs": 142,
  "leaf_eval_mean": 0.15,
  "leaf_eval_std": 0.08,
  "action_mask_density": 0.19,
  "mcts_visit_entropy": 1.82,
  "productions_used_by_type": {
    "Default": 0.35, "Ite": 0.30, "IsZero": 0.25, "Not": 0.05, "And": 0.05
  }
}
```

**Files**: `scripts/run_derivation.py` (~30 lines: helper functions + extend the `program_log.append()` block).

---

## Task 3 — Fix predict() Device Thrashing

**Problem**: `self.model.cpu()` is called at the top of every `predict()` call. During a single iteration there are ~72,000 predict calls (200 sims × 9 steps × 40 games). This is redundant because `push_multiprocessing()` already moves the model to CPU before self-play, and `pop_multiprocessing()` restores it to DEVICE after.

**Fix**: Delete the `self.model.cpu()` line in each `predict()` method:

| File | Line |
|------|------|
| `src/alphazeropp/instances/bitstring/dsl/derivation_network.py` | 166 |
| `src/alphazeropp/instances/bitstring/dsl/scan_network.py` | 68 |
| `src/alphazeropp/instances/bitstring/network.py` | 109 |
| `src/alphazeropp/instances/cartpole/network.py` | 107 |

No other changes needed. Input tensors default to CPU (matching model location during predict). The push/pop lifecycle handles device placement:
```
init → DEVICE
push → CPU (before self-play/eval)
predict → CPU (no-op, already there)
pop → DEVICE (before train)
train → DEVICE
```

---

## Task 4 — `--dry_run` Flag

**Goal**: One-command pipeline validation.

**Implementation**: Add `argparse` at top of `main()` in `scripts/run_derivation.py`:

```python
parser = argparse.ArgumentParser(description="AlphaZero derivation training")
parser.add_argument("--dry_run", action="store_true",
                    help="Run 1 iteration with minimal games for pipeline validation")
args = parser.parse_args()
```

After config is built:
```python
if args.dry_run:
    cfg.run.n_iterations = 1
    cfg.trainer.n_games_per_train = min(cfg.trainer.n_games_per_train, 2)
    cfg.evaluator.n_games = min(cfg.evaluator.n_games, 2)
```

Interactive mode selection and config editor still run. All logs are still written (the point is to validate the full pipeline end-to-end).

**Files**: `scripts/run_derivation.py` (~10 lines).

---

## Task 5 — Unit Tests + conftest Fix

### 5a. Fix `tests/conftest.py`

Change import from `nsai_experiments.general_az_1p.setup_utils` to `alphazeropp.utils.common`:

```python
from alphazeropp.utils.common import (
    disable_numpy_multithreading,
    use_deterministic_cuda,
)
```

### 5b. New test file: `tests/test_phase0_instrumentation.py`

**TestActionMaskDensity**:
- Initial density equals `n_legal_productions / max_productions`
- Density decreases after taking a step (scan game)
- Density stays in `(0, 1]` for all non-terminal states

**TestHashableObsDeterminism**:
- Two games with identical `reset_wrapper(seed=42)` produce identical `hashable_obs`
- Two games taking the same action sequence produce identical `hashable_obs` at each step

Uses existing fixtures pattern from `test_derivation_game.py` (N_SITES=3, N_ONES=2, BUDGET=5).

---

## Files Changed Summary

| File | Change type | Lines |
|------|-------------|-------|
| `specs/2026-03-03_phase0_stabilization.md` | New | this file |
| `scripts/run_derivation.py` | Edit | ~60 |
| `src/.../dsl/derivation_network.py` | Edit | -1 |
| `src/.../dsl/scan_network.py` | Edit | -1 |
| `src/.../bitstring/network.py` | Edit | -1 |
| `src/.../cartpole/network.py` | Edit | -1 |
| `tests/conftest.py` | Edit | ~2 |
| `tests/test_phase0_instrumentation.py` | New | ~80 |

---

## Verification

```bash
# 1. Run dry-run to validate pipeline + new logging
python scripts/run_derivation.py --dry_run

# 2. Check outputs exist
ls experiments/derivation/*/run_metadata.jsonl
cat experiments/derivation/*/program_log.jsonl | python -m json.tool

# 3. Run new tests
pytest tests/test_phase0_instrumentation.py -v

# 4. Run all tests (confirm no regressions)
pytest tests/ -v
```

---

## Constraints

- No algorithm behavior changes (only device fix + logging)
- Trainer/evaluator APIs unchanged
- Small diffs, clear comments
