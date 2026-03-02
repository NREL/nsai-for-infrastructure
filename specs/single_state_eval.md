# Spec: Single-State Evaluation for Derivation Training

## 1. The Problem

With the current defaults (`budget=14, n_sites=6, n_ones=2`), the `LeafEvaluator`
evaluates every synthesized program on **all** C(6,2) = 15 frozen initial states.
This is mathematically broken: no budget-14 program can solve all 15 states.

### 1.1 Proof: 100% solve_rate is impossible

A budget-14 decision-list program has the structure:

```
if C(i₁): Flip(j₁)      <- 2 + i₁ nodes
elif C(i₂): Flip(j₂)     <- 2 + i₂ nodes
...
else: Flip(j_d)           <- 2 nodes

Budget constraint: sum(2 + i_k) + 2 = 14
```

The number of distinct bit positions the program can act on is at most
the number of Flip actions. With budget L, the maximum number of `Ite` branches
(plus one `Default`) is achieved when all conditions use the minimum budget
(i=1, meaning `IsZero(j)`):

```
d branches of Ite(IsZero(_), Flip(_), ...) = 3 nodes each
1 Default(Flip(_))                          = 2 nodes

Budget: 3d + 2 = L  =>  d = (L-2)/3
Total Flip actions: d + 1 = (L-2)/3 + 1 = (L+1)/3
```

For L=14: `(14+1)/3 = 5` distinct bit positions coverable.

With n_sites=6, there is always **at least 1 uncovered position**. When the
env reaches a state where only uncovered zeros remain, the program's default
branch flips an already-correct bit (1->0 under `bit_flip=True`), creating an
oscillation that runs until `max_steps`. The state is never solved.

**For any single frozen state** with n_ones=2 (meaning 4 zeros): the program
needs to cover only 4 positions. Since 5 > 4, the program has room to cover
all 4 zeros plus 1 spare. **100% solve_rate is achievable for any single state.**

Concrete optimal program for state `[1,1,0,0,0,0]`:

```
if IsZero(2): Flip(2)      <- 3 nodes
elif IsZero(3): Flip(3)     <- 3 nodes
elif IsZero(4): Flip(4)     <- 3 nodes
elif IsZero(5): Flip(5)     <- 3 nodes
else: Flip(0)               <- 2 nodes
Total: 14 nodes
```

### 1.2 Consequence: broken training signal

With 15 states, the best possible solve_rate is 5/15 = 33.3% (a program covering
5 positions solves at most 5 of the 15 states -- those where all zeros happen to
be in covered positions). This means:

- **Early stopping never triggers** (requires `solve_rate >= 1.0`)
- **`avg_reward` is dragged down** by unsolvable states (~0.44 optimal vs 0.667)
- **The reward signal is misleading**: a structurally good program is punished
  for failing to solve states that *no* budget-14 program can solve
- **15x slower evaluation**: each program runs 15 episodes instead of 1

### 1.3 Existing bug in print_banner

`run_derivation.py:198` displays `total_programs` (151,173,432) where it should
show C(n_sites, n_ones). The banner incorrectly reads:
`Initial states: C(6,2) = 151173432 frozen states with 2 ones`

---

## 2. The Fix

Add a `n_frozen_states` parameter (default=1) that controls how many of the
C(n_sites, n_ones) frozen states are used for evaluation.

### 2.1 Why `n_frozen_states=1` is the right default

| Criterion | Justification |
|-----------|---------------|
| **Matches `run_bitstring.py` semantics** | `run_bitstring.py` evaluates its neural-net policy on a single game per evaluation. The derivation system should do the same: evaluate the synthesized program on one problem instance. |
| **100% solve_rate is achievable** | With 1 state (4 zeros), budget=14 covers 5 positions. The optimal program solves it in 4 steps. |
| **Reward signal is clean** | Program quality directly maps to reward. No noise from unsolvable states. |
| **15x faster evaluation** | 1 episode per `LeafEvaluator.__call__()` instead of 15. |
| **Backward-compatible** | Setting `n_frozen_states=15` recovers the old behavior exactly. |

### 2.2 Why NOT change n_sites or budget instead

Reducing `n_sites` to 5 (so budget=14 covers all positions) would change the
problem difficulty and the grammar's action space. Increasing budget would
increase the program space exponentially. The fix should change how we *evaluate*
programs, not the problem itself.

---

## 3. Do Other Algorithm Aspects Need Refinement?

For each concern from `specs/algorithm_audit.md`, we assess whether the
single-state change interacts with it and whether a fix is needed **now**.

### 3.1 Reward metric choice -- NO change needed

With 1 frozen state:
- `solve_rate`: binary (0.0 or 1.0) -- coarse but useful for early stopping
- `avg_reward`: continuous, ranges from ~-0.33 to +0.667 -- this is the
  primary training signal
- `penalized_reward` and `weighted`: work correctly with 1 state

The default metric `avg_reward` is the right choice. It provides a continuous
gradient of quality across programs (unlike `solve_rate` which is all-or-nothing
with 1 state). No change needed.

### 3.2 Optimal reward reference line -- ADD to plots

`run_bitstring.py` draws an `optimal_reward` horizontal line on the eval plot.
`run_derivation.py` does not. With 1 frozen state, the optimal reward is
computable: `optimal_reward = (n_sites - n_ones) / n_sites = 0.667`.

Adding this reference line to the plot helps the user see how close training
is to the theoretical maximum. It applies for any `n_frozen_states`.

### 3.3 Gating mechanism -- NO change needed now

The 2-game evaluation gate (audit section G1) is likely a no-op. This exists
independently of `n_frozen_states` and is out of scope.

### 3.4 MCTS tree reuse -- NO change needed now

MCTS rebuilds the search tree from scratch at every step (audit section B2).
Performance issue, not correctness. Orthogonal to this fix.

### 3.5 Value target uniformity -- NO change needed now

All steps share the same value target (audit section E1). Inherent to
terminal-only rewards with discount=1.0. Not affected by this change.

### 3.6 Grammar dead-end filtering -- ALREADY FIXED

See `specs/dead_end_fix.md`. Action space reduced from 60 to 48.

### 3.7 Semantic redundancy in grammar -- NO change needed now

`Not(Not(c))`, `And(c1,c2)` vs `And(c2,c1)`, etc. (audit section A2).
Orthogonal to evaluation setup.

---

## 4. Changes Made

| File | Changes |
|------|---------|
| `src/.../derivation_config.py` | Add `n_frozen_states=1` to kwargs; slice frozen_states in `build()` |
| `scripts/run_derivation.py` | Fix banner bug; add `n_frozen_states` to interactive config; optimal_reward on plot; single-state output formatting |

### Impact

- 100% solve_rate is now achievable (was capped at 33.3%)
- Optimal avg_reward is 0.667 (was ~0.44)
- Early stopping works (was never triggered)
- Leaf evaluation is 15x faster (1 episode vs 15 per program)
- Banner correctly shows frozen state count (was showing 151M)
- Plot shows optimal_reward reference line (was missing)
- Fully backward-compatible: set `n_frozen_states=15` to recover old behavior
