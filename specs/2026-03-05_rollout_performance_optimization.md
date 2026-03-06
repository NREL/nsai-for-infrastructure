# Rollout Performance Optimization

**Date:** 2026-03-05
**Status:** Proposed
**Goal:** Reduce the wall-clock time of MCTS training after adding Monte Carlo rollouts, without sacrificing rollout signal quality.

---

## Context

After implementing nonterminal rollout evaluation in MCTS (`rollout_n=4, rollout_mode=max, rollout_budget=200`), training is significantly slower. The rollout feature is correct and provides the intended bootstrap signal, but the computational cost needs to be reduced.

---

## What `rollout_n` Does (with examples)

`rollout_n` = how many times MCTS randomly completes the game from each unexpanded leaf to estimate that leaf's value.

### Example: `rollout_n=4`

MCTS reaches a partial AST (a leaf it hasn't explored):
```
Ite(IsZero(3), Flip(2), ProgramHole(28))
```
Instead of trusting the neural network's uninformed value, it **randomly fills in ProgramHole 4 times**:

| Rollout | Random completion | Reward |
|---------|-------------------|--------|
| 1 | `Default(Flip(0))` | -0.105 |
| 2 | `Ite(IsZero(5), Flip(4), Default(Flip(1)))` — picks a key | **-0.075** |
| 3 | `Default(Flip(3))` | -0.105 |
| 4 | `Ite(IsZero(0), Flip(0), Default(Flip(1)))` | -0.105 |

With `rollout_mode="max"`: value = **max(-0.105, -0.075, -0.105, -0.105) = -0.075**

MCTS now knows this partial program has potential. It will explore this subtree more.

**Cost: 4 x (clone game + ~10 random steps + run program in Doors env) ~ 4-20ms per leaf.**

### Example: `rollout_n=2` (proposed)

Same partial AST, but only 2 random completions:

| Rollout | Random completion | Reward |
|---------|-------------------|--------|
| 1 | `Default(Flip(0))` | -0.105 |
| 2 | `Ite(IsZero(5), Flip(4), Default(Flip(1)))` | **-0.075** |

value = **max(-0.105, -0.075) = -0.075** — same signal, **half the cost**.

The tradeoff: fewer samples = higher chance of missing the good completion:
- `rollout_n=4`: P(>=1 good) = 1-(0.85)^4 = **48%**
- `rollout_n=2`: P(>=1 good) = 1-(0.85)^2 = **28%**

28% is still meaningful — MCTS still preferentially explores productive subtrees.

---

## Root Cause Analysis

### The cost chain per MCTS leaf expansion

When MCTS expands a new leaf node (`search()` at `mcts.py:261-270`), `_rollout_value()` is called:

1. **Clones the game `rollout_n` times** (`game.clone()`) — lightweight for DerivationGame (~5us each)
2. **Runs random actions** until terminal or budget exhausted — ~6-17 steps per rollout for D=3
3. **On terminal: `DerivationGame.step()` calls `leaf_evaluator(program)`** — this is the bottleneck

### The real bottleneck: `LeafEvaluator._evaluate()`

`leaf_evaluator.py:112-143` — for each complete program:
- Creates a Doors environment (`make_env()`)
- Runs `run_policy_episode()` — interprets the DSL program step-by-step for up to `horizon` steps
- For Doors D=3: 1 frozen state, horizon=15, cost ~1-5ms per program

### Cost multiplier (D=3, current config)

| Parameter | Value |
|-----------|-------|
| `n_simulations` | 150 |
| New leaves per search | ~20-50 |
| `rollout_n` per leaf | 4 |
| Rollouts reaching terminal | ~60-80% |
| **Extra LeafEvaluator calls per search** | **~50-160** |
| Derivation steps per game | ~6-17 |
| MCTS searches per game | ~6-17 |
| Games per iteration | 30 |
| **Estimated slowdown** | **~5-10x** |

---

## Proposed Optimizations

### 1. Reduce `rollout_n` from 4 to 2 (config change)

**Impact: ~2x speedup.**

Half the rollouts per leaf. Signal drops from 48% to 28% detection rate for productive ASTs, but this is still substantial.

### 2. Reduce `n_simulations` from 150 to 80 (config change)

**Impact: ~1.8x speedup.**

Rollouts provide better per-leaf value estimates than the uninformed network. With better leaf values, fewer simulations are needed to distinguish good from bad subtrees.

### 3. Share `rollout_budget` across the entire MCTS search (code change)

**Impact: ~2-3x speedup on rollout cost.**

Currently `rollout_budget=200` resets independently per leaf. With 20+ new leaves per search, that allows 4000+ rollout steps. Instead, share a single budget pool across all leaf expansions in one `perform_simulations()` call. Early leaves get full rollouts; later leaves get fewer or none (graceful degradation).

### 4. Enable multiprocessing (`n_procs=4`) (config change)

**Impact: ~3-4x speedup (scales with cores).**

#### Multiprocessing safety verification

Investigated the full code path. **Verdict: safe to enable.**

| Component | Independent per worker? | Mechanism |
|-----------|------------------------|-----------|
| Game instance | YES | `game.clone()` in `agent.py:229` |
| MCTS tree | YES | New MCTS per game in `agent.py:164` |
| Rollout state | YES | Local variables in `_rollout_value()` |
| Network model | YES | Pickled CPU copy via `push_multiprocessing()` |
| LeafEvaluator | YES (copies) | Pickled copy per worker (spawn context) |

**How it works:**
- `trainer.py:82-113` uses `torch.multiprocessing.get_context("spawn")`
- Network moved to CPU before spawn (`push_multiprocessing()` in `policy_value_net.py:135`)
- Each worker gets an independent pickled copy of game + network + evaluator
- Workers run `play_for_experience_reuse_tree()` independently

**Caveats (non-blocking):**
- LeafEvaluator cache is **not shared** across workers — each rebuilds independently. This means some redundant program evaluations, but no correctness issue.
- Rollout RNG uses global `np.random.randint()` — non-deterministic across workers but intentionally random (Monte Carlo).
- `n_procs=-1` means sequential (current setting). `n_procs=None` uses all cores. We'll set `n_procs=4` explicitly.

---

## Recommended Configuration

```python
mcts_params={
    "n_simulations": 80,       # was 150 (fewer sims needed with better leaf values)
    "rollout_n": 2,            # was 4 (half cost, 28% detection vs 48%)
    "rollout_mode": "max",     # unchanged
    "rollout_blend": 0.0,      # unchanged
    "rollout_budget": 200,     # unchanged (but now shared across search)
    ...
}
self.trainer = TrainerConfig(
    n_games_per_train=30,      # unchanged
    n_procs=4,                 # was -1 (sequential)
    ...
)
```

**Expected combined speedup: ~8-15x** relative to current config.

---

## Implementation Plan

### Step 1: Shared rollout budget across MCTS search

**File:** `src/alphazeropp/core/mcts.py`

Add `self._search_rollout_budget` initialized in `perform_simulations()` and `perform_simulations_reuse()`:

```python
# In perform_simulations(), before the simulation loop:
self._search_rollout_budget = self.rollout_budget

# In perform_simulations_reuse(), before the simulation loop:
if not hasattr(self, '_search_rollout_budget') or self._search_rollout_budget <= 0:
    self._search_rollout_budget = self.rollout_budget
```

Update `_rollout_value()` to use the shared budget:

```python
def _rollout_value(self, msg) -> float | None:
    rewards = []
    for _ in range(self.rollout_n):
        if self._search_rollout_budget <= 0:
            break
        rollout_game = self.game.clone()
        cumulative_reward = 0.0
        while not (rollout_game.terminated or rollout_game.truncated):
            if self._search_rollout_budget <= 0:
                break
            mask = rollout_game.get_action_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break
            action = valid[np.random.randint(len(valid))]
            if len(rollout_game.action_space.shape) == 0:
                rollout_game.step_wrapper(int(action))
            else:
                rollout_game.step_wrapper(
                    np.unravel_index(action, mask.shape))
            cumulative_reward += rollout_game.reward
            self._search_rollout_budget -= 1
        if rollout_game.terminated or rollout_game.truncated:
            rewards.append(cumulative_reward)
    if not rewards:
        return None
    if self.rollout_mode == "max":
        return float(max(rewards))
    return float(sum(rewards) / len(rewards))
```

### Step 2: Update config defaults

**File:** `src/alphazeropp/instances/doors/dsl/derivation_config.py`

In `DoorsDerivationConfig.__init__()` (base class):
- `"n_simulations": 80` (was 150)
- `"rollout_n": 2` (was 4)
- `n_procs=4` (was -1)

In `DoorsFactoredD10MacroConfig.__init__()`:
- Same changes to its overridden `mcts_params` and `TrainerConfig`

### Step 3: Verification

```bash
# Unit tests (rollout behavior)
python -m pytest tests/test_mcts_rollout.py -v

# Full test suite (backward compatibility)
python -m pytest tests/ -v --ignore=tests/test_zoning_game.py

# Smoke test: run 1-2 iterations and check timing
python scripts/run_doors_derivation.py
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/alphazeropp/core/mcts.py` | Shared rollout budget across search (`_search_rollout_budget`) |
| `src/alphazeropp/instances/doors/dsl/derivation_config.py` | `n_simulations=80`, `rollout_n=2`, `n_procs=4` |
