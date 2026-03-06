# LeafEvaluator Stats Inflation Bug: Diagnosis & Fix

**Date:** 2026-03-06
**Experiment:** `20260305_232833_D4_and_factored_macro_N15_L48_max_weighted_mcts200_games200_iter50`
**Status:** Crashed after 4 of 50 iterations

---

## 1. Problem Statement

The `eval_count` stat in `train_stats.jsonl` grows by exactly **201x per iteration**, causing the counters to overflow into the trillions and obscuring real performance data:

| Iteration | eval_count (cumulative) | Per-iter delta | Growth ratio |
|-----------|------------------------|----------------|--------------|
| 1 | 1,739,137 | 1,739,137 | — |
| 2 | 351,504,324 | 349,765,187 | **201.0x** |
| 3 | 70,653,850,921 | 70,302,346,597 | **201.0x** |
| 4 | 14,201,425,543,137 | 14,130,771,692,216 | **201.0x** |

The same 201x multiplication applies to `total_env_steps` and `total_interp_ops`. Meanwhile `cache_hits` is always 0 (a secondary bug: it's not exported).

The 201x factor = **1 + n_games_per_train** (config: `n_games_per_train = 200`).

---

## 2. Process Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN PROCESS                            │
│                                                             │
│  game.leaf_evaluator                                        │
│    ._eval_count = E  (accumulated from prior iterations)    │
│    ._cache = {prog1: v1, prog2: v2, ...}  (N entries)       │
│                                                             │
│  trainer._collect_training_examples()                       │
│    fn = partial(play_for_experience_reuse_tree, self.game)   │
│    pool.starmap(fn, [(0, seed, seed), (1, ...), ...])       │
│                        │                                    │
│         ┌──────────────┼──────────────┐                     │
│         │     PICKLE fn for each of 200 tasks               │
│         │     (spawn context serializes game + leaf_eval)    │
│         ▼              ▼              ▼                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │  Worker 1  │ │  Worker 2  │ │  Worker 3  │  ...          │
│  │  (Pool)    │ │  (Pool)    │ │  (Pool)    │              │
│  │            │ │            │ │            │              │
│  │ Runs ~50   │ │ Runs ~50   │ │ Runs ~50   │              │
│  │ tasks      │ │ tasks      │ │ tasks      │              │
│  │ serially   │ │ serially   │ │ serially   │              │
│  └────────────┘ └────────────┘ └────────────┘              │
│         │              │              │                     │
│         │   Each TASK unpickles a fresh copy:               │
│         │     leaf_eval._eval_count = E  (inherited!)       │
│         │     leaf_eval._cache = {prog1: v1, ...}           │
│         │                                                   │
│         │   Task plays 1 game → does w new evaluations      │
│         │     leaf_eval._eval_count = E + w                 │
│         │                                                   │
│         │   Task exports: {_eval_count: E + w, ...}         │
│         ▼                                                   │
│                                                             │
│  trainer.train_iteration()                                  │
│    for each of 200 exports:                                 │
│      leaf_eval.merge_caches(export)                         │
│        self._eval_count += export["_eval_count"]            │
│                          += (E + w)                         │
│                                                             │
│  Result: E_new = E + 200 * (E + w) = 201*E + 200*w         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Each task's export contains `E + w`, not just `w`. The main process adds `(E + w)` 200 times to its existing `E`, producing `201E + 200w`.

---

## 3. Lifecycle of Stats Through One Iteration

```
BEFORE iteration:
  main.leaf_eval._eval_count = E

STEP 1: pool.starmap pickles fn (which contains game.leaf_evaluator)
         for each of 200 tasks independently
                │
                ▼
STEP 2: Each task unpickles → gets independent copy
         task_i.leaf_eval._eval_count = E        ← INHERITS main's total
                │
                ▼
STEP 3: Task plays 1 game, evaluates w_i unique programs
         task_i.leaf_eval._eval_count = E + w_i
                │
                ▼
STEP 4: Task calls export_caches()
         returns {_eval_count: E + w_i, ...}      ← EXPORTS inherited + new
                │
                ▼
STEP 5: Main calls merge_caches() for each of 200 exports
         main._eval_count += (E + w_0)
         main._eval_count += (E + w_1)
         ...
         main._eval_count += (E + w_199)
                │
                ▼
AFTER iteration:
  main._eval_count = E + Σ(E + w_i)
                   = E + 200*E + Σ(w_i)
                   = 201*E + W_total

  where W_total = Σ(w_i) ≈ actual work done this iteration
```

---

## 4. Worked Numerical Example

Assume each iteration does `W = 8,695` real evaluations per game × 200 games = 1,739,000 total real evals.

```
ITERATION 1:
  Start:  E = 0
  Work:   W = 1,739,000
  Result: 201 * 0 + 1,739,000 = 1,739,000  ✓ (matches data: 1,739,137)

ITERATION 2:
  Start:  E = 1,739,000
  Work:   W ≈ 1,739,000
  Result: 201 * 1,739,000 + 1,739,000 = 349,539,000 + 1,739,000
        = 351,278,000  ✓ (matches data: 351,504,324, ratio = 201.0)

ITERATION 3:
  Start:  E = 351,278,000
  Work:   W ≈ 1,739,000
  Result: 201 * 351,278,000 + 1,739,000
        ≈ 70,606,878,000  ✓ (matches data: 70,653,850,921, ratio = 201.0)

ITERATION 4:
  Start:  E ≈ 70,607,000,000
  Work:   W ≈ 1,739,000
  Result: 201 * 70,607,000,000 + 1,739,000
        ≈ 14,192,007,000,000  ✓ (matches data: 14,201,425,543,137)

PROJECTED ITERATION 5 (if it hadn't crashed):
  ≈ 201 * 14.2T ≈ 2.85 QUADRILLION
```

The actual real work per iteration is ~1.74M evals. By iteration 4, the counter says 14.2T — **inflated by 8,167,000x**.

---

## 5. Why `cache_hits = 0`?

Secondary bug: `export_caches()` does not include `_cache_hits`:

```python
# leaf_evaluator.py lines 118-125 (BEFORE fix)
def export_caches(self) -> dict:
    return {
        "_cache": dict(self._cache),
        "_full_cache": dict(self._full_cache),
        "_program_cache": dict(self._program_cache),
        "_eval_count": self._eval_count,
        "_total_env_steps": self._total_env_steps,
        "_total_interp_ops": self._total_interp_ops,
        # NOTE: _cache_hits is MISSING
    }
```

And `merge_caches()` reads it with a default of 0:
```python
# This line doesn't exist in the current code, so cache_hits
# from workers are silently discarded
```

Workers DO get cache hits (when the same program appears multiple times within one game's MCTS search), but these are never reported back to the main process.

---

## 6. Root Cause Summary

| Bug | Location | Mechanism |
|-----|----------|-----------|
| **Stats inflation** | `export_caches()` | Exports inherited stats + new work, instead of just new work |
| **201x factor** | `pool.starmap` + `merge_caches` | 200 tasks × inherited stats = 200× multiplication per iter |
| **Missing cache_hits** | `export_caches()` | `_cache_hits` field not included in export dict |
| **Pickle overhead** | `pool.starmap` | Entire cache (up to 1M entries) serialized per task |

---

## 7. Fix Specification

### Approach: `__getstate__` pickle hook

Add `__getstate__` to `LeafEvaluator` so that when it's pickled for worker processes, it arrives with:
- **Empty caches** (no O(cache_size) serialization cost)
- **Zeroed stats** (exports are pure deltas by construction)

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state['_cache'] = {}
    state['_full_cache'] = {}
    state['_program_cache'] = {}
    state['_eval_count'] = 0
    state['_cache_hits'] = 0
    state['_total_env_steps'] = 0
    state['_total_interp_ops'] = 0
    state['_base_eval_count'] = 0
    state['_base_cache_hits'] = 0
    state['_base_total_env_steps'] = 0
    state['_base_total_interp_ops'] = 0
    return state
```

Also add `_cache_hits` to `export_caches()` and `merge_caches()`.

### After fix — expected stats flow:

```
ITERATION 1:
  Start:  E = 0
  Each task starts with eval_count = 0 (via __getstate__)
  Each task exports: {_eval_count: w_i}  (pure delta)
  Main merges: 0 + Σ(w_i) = W ≈ 1,739,000

ITERATION 2:
  Start:  E = 1,739,000
  Each task starts with eval_count = 0 (via __getstate__)
  Each task exports: {_eval_count: w_i}  (pure delta)
  Main merges: 1,739,000 + Σ(w_i) ≈ 3,478,000

ITERATION N:
  E ≈ N * 1,739,000  (LINEAR growth, as expected)
```

---

## 8. Did This Bug Cause the Crash?

**No, not directly.** The inflated counters are just integers in memory — they don't cause OOM.

The crash was caused by **wall clock time growth** (separate issue):

| Iter | Wall clock | Growth |
|------|-----------|--------|
| 1 | 13.3 min | — |
| 2 | 20.2 min | 1.5x |
| 3 | 41.8 min | 2.1x |
| 4 | 111.9 min | 2.7x |

This comes from:
1. **Growing cache pickle overhead**: 251K → 972K cache entries serialized 200 times per iteration
2. **Tree reuse accumulating larger MCTS trees**: More nodes per game across moves

The `__getstate__` fix addresses cause #1 (empty caches in workers = fast pickle). Cause #2 is a separate concern for future optimization.

---

## 9. Files Modified

| File | Change |
|------|--------|
| `src/alphazeropp/synthesis/leaf_evaluator.py` | Add `__getstate__`, `snapshot_baseline`, fix `export_caches`/`merge_caches` |
| `src/alphazeropp/training/trainer.py` | Call `snapshot_baseline()` before spawning (for sequential mode) |
| `tests/test_leaf_evaluator_stats.py` | New test file verifying linear growth |
