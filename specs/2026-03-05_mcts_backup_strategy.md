# MCTS Configurable Backup Strategy for Program Synthesis

**Date:** 2026-03-05
**Status:** Proposed
**Goal:** Replace the fixed mean-backup in MCTS with a configurable backup strategy (mean/max/topk/softmax) to better handle single-player program synthesis, where rare good programs are drowned out by averaging.

---

## Context

### Observed Failure

Experiment `D4_and_factored_macro_N15_L48_max_weighted_mcts80_games30_iter30` is stuck:

| Metric | Iteration 1 | Iteration 12 | Change |
|--------|-------------|--------------|--------|
| Best solve rate | 0% | 0% | None |
| Best avg reward | -0.250 | -0.250 | None |
| Best program | (same) | (same) | Identical |
| Unique programs | 7,372 | 144,408 | +137K |
| Training loss | 4.6 | 3.7 | Decreasing |

The system discovers 144K unique programs but the best never improves. Training loss decreases but produces no reward improvement — the network learns to predict the flat reward landscape accurately, which is useless.

For contrast, `doors_direct` (L=3, 100 sims) reaches **100% solve rate by iteration 6**.

### Root Cause Chain

```
Mean backup    →  rare good signals averaged away
               →  Q-values uniformly low (~-0.25 everywhere)
               →  Q-normalization degeneracy (q_max ≈ q_min)
               →  UCB = 0.5 + exploration term (pure policy prior)
               →  MCTS degenerates to policy-guided BFS
               →  no exploitation of discovered good paths
               →  no learning signal for the value network
```

### Why Mean Backup Is Wrong for Synthesis

In **two-player games** (Go, Chess), mean backup approximates the expected outcome under optimal opponent play — both players get to choose, so averaging over trajectories is correct.

In **single-player program synthesis**, the agent wants to find ONE good program. A node with 9 bad paths and 1 good path should be valued highly (there exists a good path), not averaged down. The correct operator is closer to **max** than **mean**.

| Scenario | Mean Q | Max Q | Correct for synthesis? |
|----------|--------|-------|----------------------|
| 1 good (0.5) + 9 bad (-0.25) | -0.175 | 0.5 | Max |
| 3 good (0.3) + 7 bad (-0.25) | -0.085 | 0.3 | Max or topk |
| All similar (-0.105 ± 0.001) | -0.105 | -0.104 | Either (no signal) |

### Contributing Factors (Not Addressed Here)

These are real issues but are separate from this spec:

1. **Budget L=48 for D=4 is excessive** — optimal program needs ~15-20 AST nodes. 48 creates an exponentially larger search space.
2. **80 MCTS sims for ~48-depth trees** — impossibly sparse coverage.
3. **Flat reward landscape** — most random programs score identically (-0.25), providing no gradient for any backup strategy. The rollout system (already implemented per spec `2026-03-05_mcts_nonterminal_rollout_evaluation.md`) partially addresses this by injecting ground-truth reward variance at leaves.

---

## Design

### Approach: Configurable Backup Rule

Add a `backup_rule` parameter to MCTS controlling how `action_Q[a]` is computed from the history of backed-up values through edge `a`.

Four strategies, selectable via config:

#### 1. `mean` (current default)
```
Q(a) = sum(values) / len(values)
```
Backward-compatible. Correct for two-player games.

#### 2. `max`
```
Q(a) = max(values)
```
Optimistic: reflects "the best outcome achievable through this edge." Correct when searching for a single good solution. Risk: can over-exploit a single lucky rollout early on, starving exploration of other branches.

#### 3. `topk`
```
Q(a) = mean(sorted(values, reverse=True)[:k])
```
Compromise: averages the top-k values. Less noisy than pure max, still preserves rare good signals. When `k >= len(values)`, degenerates to mean. When `k=1`, equals max.

#### 4. `softmax`
```
Q(a) = max(v) + tau * log(mean(exp((v - max(v)) / tau)))
```
Smooth approximation of max. As `tau → 0`: approaches max. As `tau → ∞`: approaches mean. Numerically stable via the shifted log-sum-exp trick.

### Key Design Decisions

**1. Store all values, not incremental accumulators.**

Adding `action_values: dict[action, list[float]]` to `MCTSTreeNode`. With 80 sims spread across many edges (~1-8 visits per edge), this is negligible memory. Simpler than maintaining separate heaps/accumulators per strategy.

**2. action_Q remains the single source of truth.**

`calc_masked_ucbs` reads `action_Q` without knowing which backup rule produced it. No changes needed to UCB computation. The backup rule only affects what value goes into `action_Q`.

**3. Visit counts unchanged.**

`action_N` and `total_N` are incremented exactly as before. The exploration term `c * P(a) * sqrt(N) / (1 + N(a))` is unchanged. Only the exploitation term (normalized Q) changes.

**4. Q-normalization handles all rules.**

The existing min-max normalization in `calc_masked_ucbs` (lines 401-407) already handles:
- `q_min == inf` or `q_max == -inf`: returns 0 (no data yet)
- `q_max > q_min`: normalizes to [0, 1]
- `q_max == q_min`: returns 0.5 (degenerate case)

With max backup, Q values have more spread (good), so the degenerate case is less likely.

---

## Implementation

### File: `src/alphazeropp/core/mcts.py`

#### MCTSTreeNode (line 36)
```python
# Add after self.action_N = {}
self.action_values = {}  # All backed-up values per action
```

#### MCTS.__init__ (lines 46-56)
```python
def __init__(self, game, net,
             n_simulations=25, temperature=1.0, c_exploration=1.0,
             dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
             rollout_n=0, rollout_mode="mean", rollout_blend=0.0,
             rollout_budget=500,
             # -- Backup strategy --
             backup_rule="mean",
             backup_topk=3,
             backup_tau=0.1):
    # ... existing init ...
    self.backup_rule = backup_rule
    self.backup_topk = backup_topk
    self.backup_tau = backup_tau
    assert backup_rule in ("mean", "max", "topk", "softmax"), \
        f"Unknown backup_rule: {backup_rule}"
```

#### update_edge (lines 436-450)
```python
def update_edge(self, mynode, action, reward):
    if action not in mynode.action_N:
        assert action not in mynode.action_Q
        mynode.action_N[action] = 0
        mynode.action_Q[action] = 0.0
        mynode.action_values[action] = []

    mynode.action_values[action].append(reward)
    mynode.action_N[action] += 1

    values = mynode.action_values[action]
    if self.backup_rule == "mean":
        mynode.action_Q[action] = sum(values) / len(values)
    elif self.backup_rule == "max":
        mynode.action_Q[action] = max(values)
    elif self.backup_rule == "topk":
        k = self.backup_topk
        top = sorted(values, reverse=True)[:k]
        mynode.action_Q[action] = sum(top) / len(top)
    elif self.backup_rule == "softmax":
        tau = self.backup_tau
        v = np.array(values)
        v_shifted = (v - v.max()) / tau
        mynode.action_Q[action] = float(
            v.max() + tau * np.log(np.mean(np.exp(v_shifted)))
        )

    new_q = mynode.action_Q[action]
    if new_q < self.q_min:
        self.q_min = new_q
    if new_q > self.q_max:
        self.q_max = new_q
```

**No changes to `calc_masked_ucbs`** — it already reads `action_Q` generically.

### File: `src/alphazeropp/instances/doors/dsl/derivation_config.py`

Add to `mcts_params` dict in all 4 config classes:
- `DoorsDerivationConfig` (line ~86)
- `DoorsDerivationConfigNoAnd` (line ~180)
- `DoorsFactoredDerivationConfig` (line ~260)
- `DoorsFactoredD10MacroConfig` (line ~334)

```python
"backup_rule": "max",
"backup_topk": 3,
"backup_tau": 0.1,
```

Default to `"max"` for synthesis configs. The `mcts_params` dict is splatted into `MCTS()` at `agent.py:98`, so no changes needed in agent.py.

### File: `scripts/run_doors_derivation.py`

In `_build_sections`, add to `mcts_descs` dict (line ~111):
```python
"backup_rule": "Backup strategy: mean, max, topk, softmax",
"backup_topk": "Top-k values for topk backup",
"backup_tau": "Temperature for softmax backup",
```

Add `"backup_rule"`, `"backup_topk"`, `"backup_tau"` to the iteration list at lines ~122-124.
Add to `mcts_labels` set at line ~188.
For `backup_rule`, provide choices `["mean", "max", "topk", "softmax"]`.

### File: `tests/test_mcts_backup.py` (new)

Follow `tests/test_mcts_rollout.py` patterns:
- Bitstring `DerivationGame` with `UniformPolicyValueNet`
- `N_SITES=3`, `BUDGET=5` fixtures
- `_make_mcts(leaf_eval, **kwargs)` helper

Test classes:

| Test Class | What It Verifies |
|-----------|-----------------|
| `TestBackupMean` | Default `backup_rule="mean"` reproduces current behavior |
| `TestBackupMax` | `Q = max(values)`, `Q >= mean` for same data |
| `TestBackupTopK` | `Q = mean(top-k)`, handles `k > len(values)` gracefully |
| `TestBackupSoftmax` | `mean <= Q <= max`, `tau→0` approaches max |
| `TestBackupEdgeCases` | Single visit: all rules agree. Identical rewards: all rules agree |
| `TestBackupQNorm` | `q_min` and `q_max` finite after search with each rule |
| `TestBackupTreeReuse` | `perform_simulations_reuse` + `advance_to` works with `backup_rule="max"` |

---

## Edge Cases

| Case | Behavior |
|------|----------|
| First visit (1 value) | All rules return the same Q — single value |
| topk with k > N(a) | `sorted(values)[:k]` returns all values — degenerates to mean |
| Softmax with 1 value | `log(mean(exp(0))) = 0` → `Q = v[0]` — correct |
| All values identical | All rules return the same value |
| q_max == q_min | Existing guard returns 0.5 — unchanged |
| Negative rewards | max, topk, softmax all handle negative values correctly |
| Tree reuse | `action_values` persists across searches (intentional: reused tree retains history) |

---

## Interaction with Existing Features

### Rollouts (`rollout_n > 0`)
Rollouts inject ground-truth reward variance at leaves. Backup strategy affects how these diverse values propagate up the tree. **Max backup + rollouts is the strongest combination**: rollouts provide diverse leaf values, max backup preserves the best ones.

### Tree reuse (`perform_simulations_reuse`)
Tree reuse retains the MCTS tree across derivation steps. `action_values` lists grow across searches for the same node. This is correct behavior — reused nodes should retain their full value history.

### Dirichlet noise
Noise is applied to `nn_policy`, not to Q-values. No interaction with backup strategy.

### Q-normalization reset
`q_min` and `q_max` are reset at the start of each `perform_simulations()` call (line 87-88). This is per-move normalization, independent of backup rule.

---

## Verification

1. **Unit tests**: `pytest tests/test_mcts_backup.py -v`
2. **Regression**: `pytest tests/ -v` (all existing tests pass with default `backup_rule="mean"`)
3. **Integration**: Run D=4 factored+macro experiment with `backup_rule="max"`, compare learning curve against mean baseline
4. **If max is too aggressive**: Try `topk` with `k=3` or `softmax` with `tau=0.1`

---

## Recommended Follow-up Experiments (Out of Scope)

- Reduce budget L from 48 to 24-30 for D=4 (reduce search space)
- Increase n_simulations from 80 to 200+ (better tree coverage)
- Add intermediate reward shaping for partial key collection
- Curriculum: start with D=2, transfer to D=4
