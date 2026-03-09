# Reward Signal Diagnosis: D=3 and D=4 Doors Derivation

**Date:** 2026-03-06
**Experiments:**
- `20260306_101040_D3_and_factored_macro_N11_L34_max_weighted_mcts50_games30_iter30`
- `20260306_084523_D4_and_factored_macro_N15_L48_max_weighted_mcts200_games50_iter30`

## Summary

Analysis of D=3 and D=4 training runs reveals that **learning is not driving program discovery** -- brute-force exploration is. The reward signal is too compressed for the value head to distinguish programs of different quality, causing it to collapse to predicting a constant. D=3 solved by luck at iteration 14; D=4 never solved in 30 iterations despite evaluating 654K unique programs.

**Root cause:** Two compounding compression layers reduce the entire non-solving reward range to just 0.06 units -- below the value head's ability to discriminate.

---

## 1. Experiment Comparison

| | D=3 | D=4 |
|---|---|---|
| Rooms / Keys | 3 / 2 | 4 / 3 |
| Budget (L) | 34 | 48 |
| Horizon | 25 | 35 |
| MCTS sims | 50 | 200 |
| Games/iter | 30 | 50 |
| Solved? | Yes (iter 14) | No (never) |
| Best pre-solve reward | -0.05 (2/2 keys) | -0.15 (2/3 keys) |
| Programs explored | 350K | 654K |
| Value loss at convergence | ~0.0001 | ~0.0001 |
| Gate score (typical) | 0.5 | 0.5 |

Both experiments show identical pathologies:
- Value head collapses by iteration 5
- Gate score ~0.5 = no selection pressure
- Best program found early, never improved by learning
- Self-play avg_reward essentially flat

D=3 solved because random search over 213K programs found a solution. D=4 requires coordinating 3 key pickups (vs 2), making the correct program rarer in the search space.

## 2. The Reward Pipeline

Two layers compound compression:

### Layer 1: Environment rewards (Doors PDDL)
```
per step:      -0.01  (step_penalty)
per key pick:  +0.10  (unlock_bonus)
reach goal:    +1.00
```

### Layer 2: Leaf evaluator metric
```
weighted = 0.7 * solve_rate + 0.3 * avg_reward
```

The derivation game gives reward=0 at non-terminal steps and `leaf_eval(program)` at terminal. With `reward_discount=1.0`, every derivation step gets value target = leaf_eval score.

## 3. Quantifying the Compression

### D=4 (horizon=35, 3 keys)

| Keys picked | Env reward | Weighted metric | Gap to previous |
|---|---|---|---|
| 0 | 0(0.1) - 35(0.01) = **-0.350** | 0.3(-0.35) = **-0.105** | -- |
| 1 | 1(0.1) - 35(0.01) = **-0.250** | 0.3(-0.25) = **-0.075** | 0.030 |
| 2 | 2(0.1) - 35(0.01) = **-0.150** | 0.3(-0.15) = **-0.045** | 0.030 |
| 3+goal | 3(0.1) + 1.0 - 7(0.01) = **+1.230** | 0.7 + 0.3(1.23) = **+1.069** | **1.114** |

- Non-solving range: **0.060** (from -0.105 to -0.045)
- Cliff to solving: **1.114** (18.6x the inter-level gap)
- Value head target variance: ~0.0009, MSE converges to ~0.0001

### Compression source 1: unlock_bonus dominated by step_penalty
- Total step penalty over horizon: 35 * 0.01 = 0.35
- Each key adds only 0.10
- Signal-to-noise ratio: 0.10 / 0.35 = **0.29** (signal drowned by noise)

### Compression source 2: blend_alpha=0.7 multiplier
- When solve_rate=0: weighted = 0.7 * 0 + 0.3 * avg_reward = 0.3 * avg_reward
- The 0.7 * solve_rate term is dead weight (always 0), while 0.3 multiplier compresses the continuous signal by 3.3x

## 4. Best Program Analysis

### D=4 best program (found at iteration 2, never improved)
```
if And(Not(IsZero(1)), Not(IsZero(12))):   # at loc 1 AND key 0 available
  Flip(8)                                   # PICK(key 0) -- CORRECT
elif And(IsZero(13), Not(IsZero(7))):       # key 1 gone AND at loc 7
  Flip(7)                                   # MOVE(goal)
elif And(Not(IsZero(3)), Not(IsZero(13))):  # at loc 3 AND key 1 available
  Flip(9)                                   # PICK(key 1) -- CORRECT
elif IsZero(9):                              # room 1 locked
  Flip(1)                                   # MOVE(loc 1) -- navigate to key 0
elif IsZero(14):                             # key 2 gone
  Flip(7)                                   # MOVE(goal)
elif IsZero(1):                              # not at loc 1
  Flip(3)                                   # MOVE(loc 3)
elif IsZero(10):                             # room 2 locked
  Flip(3)                                   # MOVE(loc 3)
else:
  Flip(3)                                   # MOVE(loc 3)
```

**Picks 2 of 3 keys correctly** but is missing:
- `And(Not(IsZero(5)), Not(IsZero(14))) -> Flip(10)` (pick key 2 at loc 5)
- A rule to navigate to loc 5

Result: picks 2 keys, runs out horizon, reward = 2(0.1) - 35(0.01) = -0.15.

### D=3 solving program (found at iteration 14)
```
if And(Not(IsZero(1)), Not(IsZero(9))):     # at loc 1 AND key 0 available
  Flip(6)                                   # PICK(key 0)
elif Not(IsZero(9)):                         # key 0 available
  Flip(1)                                   # MOVE(loc 1) -- navigate to key 0
elif IsZero(3):                              # not at loc 3
  Flip(3)                                   # MOVE(loc 3) -- navigate to key 1
elif And(Not(IsZero(3)), Not(IsZero(10))):   # at loc 3 AND key 1 available
  Flip(7)                                   # PICK(key 1)
elif And(IsZero(9), ...):                    # key 0 gone (used)
  Flip(5)                                   # MOVE(goal)
else:
  Flip(6)
```

This program solves 100% of states with avg_reward=1.15. It was found by random exploration, not guided by learning -- the value head was already collapsed at iteration 14.

## 5. Evidence That Learning Is Not Helping

1. **Value loss collapse:** train_loss_value drops to ~0.0001 by iteration 5 in both D=3 and D=4. The network predicts a constant (~-0.07) for all partial ASTs.

2. **Gate score = 0.5:** New and old networks are indistinguishable. 27/30 iterations in D=4 have gate_score exactly 0.5. No model selection pressure.

3. **Eval rewards have zero variance:** eval_stats show std ≈ 1.4e-17 for iterations 15-30 (D=4). Both new and old networks produce programs with identical reward.

4. **Best program frozen early:** D=4 best program never changes after iteration 2. D=3 best program changes at iteration 2 (finds 2-key non-solver) then at iteration 14 (finds solver by luck).

5. **Exploration decelerates:** New programs/iter drops from ~48K (iter 1) to ~4K (iter 30) in D=4. The network converges to producing the same programs, yielding cache hits instead of exploration.

## 6. Proposed Reward Changes

### A. Increase unlock_bonus (0.1 -> 1.0)

Each key pickup becomes 10x more valuable. SNR improves from 0.29 to 2.86.

| Keys | Env reward (new) | Weighted (alpha=0.7) |
|---|---|---|
| 0 | -0.350 | -0.105 |
| 1 | +0.650 | +0.195 |
| 2 | +1.650 | +0.495 |
| 3+goal | +4.230 | +1.969 |

Non-solving range: 0.600 (10x wider). Does not change optimal policy.

### B. Adaptive alpha

When solve_rate=0, return raw avg_reward (no 0.3x compression):
```python
if solve_rate > 0:
    return alpha * solve_rate + (1 - alpha) * avg_reward
else:
    return avg_reward
```

Combined with A: non-solving range = **2.000** (33x wider than original 0.060).

Design choice: a non-solver with 2 keys (+1.65) outranks a partial solver at 10% (+0.27). This is correct -- consistent partial progress is genuinely better than unreliable solving.

### C. Reward normalization (optional)

Running EMA normalization to auto-adapt to any reward scale:
```python
normalized = (value - ema_mean) / sqrt(ema_var + eps)
```

Useful for robustness across domains. Less critical after A+B.

### D. Per-key progress metric

Direct milestone counting: keys_progress = keys_picked / total_keys.
Perfectly uniform spacing (0, 0.33, 0.67, 1.0 for D=4). Domain-specific but theoretically grounded in potential-based reward shaping (Ng et al., 1999).

## 7. Systematic Reward Design Principles

1. **Granularity >= quality levels:** If N meaningfully different program quality levels exist, reward must have N distinguishable values.

2. **Gap >> intra-level variance:** Inter-level gap must be much larger than reward variation within the same level (target: gap/variance > 5).

3. **Milestone decomposition:** Reward should decompose into monotone, measurable, progressively harder milestones.

4. **No cliffs:** Largest inter-level gap should be < 5x the smallest. Otherwise the value head treats everything below the cliff as equivalent.

5. **Consistent scale:** Value targets should be in [-1, 1] or [0, 1] for stable gradient flow.

## 8. Implementation

Files to modify:
- `src/alphazeropp/instances/doors/dsl/derivation_config.py` -- unlock_bonus 0.1 -> 1.0
- `src/alphazeropp/synthesis/leaf_evaluator.py` -- adaptive alpha, normalization, keys_progress
- `src/alphazeropp/utils/post_diagnostics.py` -- NEW: diagnostic plotting
- `scripts/run_post_diagnostics.py` -- NEW: standalone diagnostic script
- `src/alphazeropp/utils/derivation_utils.py` -- hook post-diagnostics after training
- `scripts/run_doors_derivation.py` -- expose new params in interactive editor
