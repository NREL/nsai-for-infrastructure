# Dead-End Grammar Production Fix

## 1. Summary of Changes

### Problem
The budget-constrained CFG has two program production templates:
- `P(2) → Default(Flip(j))` — requires exactly 2 AST nodes
- `P(k) → Ite(C(i), Flip(j), P(k-2-i))` — requires at least 5 AST nodes

Budgets 3 and 4 fall in a **gap**: no production template applies. Before this fix, `_program_productions()` would generate Ite productions whose else-branch `ProgramHole` had budget 3 or 4. These holes could never be filled, causing the derivation to reach a dead end — the game truncates with reward=0, wasting all computation spent reaching that state.

### Files Modified

| File | Change |
|------|--------|
| `src/.../dsl/derivation.py` | Added 2-line guard in `_program_productions()` to skip dead-end productions |
| `tests/test_cfg_grammar.py` | Added `_assert_no_dead_holes` helper + `TestDeadEndPrevention` class (4 tests) |
| `tests/test_derivation_game.py` | Added `TestActionSpaceAfterFiltering` class (2 tests) |

### Impact
- Action space reduced from 60 to 48 (for budget=14, n_sites=6)
- Zero dead-end truncations in any derivation episode
- No change to the set of reachable programs — only unreachable dead-end paths were removed

### How this improves learning

A common misconception is that the `count_programs` guard adds computational overhead. It doesn't — `count_programs` uses `@lru_cache`, so after the first call for a given `(n_sites, budget)` pair, every subsequent call is an O(1) dictionary lookup. The guard runs once per production when building the action list, which is negligible.

The real benefit is **not about the cost of the guard itself** — it's about what happens **downstream in MCTS** when dead-end productions are available vs. removed:

**Without the fix — wasted MCTS simulations:**

Each MCTS `perform_simulations()` call runs 50 simulations. Each simulation is a single path from the root of the search tree down to a leaf. When a simulation descends through a dead-end production (e.g., one that creates `P(4)`), the entire path proceeds through several more derivation steps (expanding other holes in the partial AST) until it finally reaches the dead `P(4)` hole. At that point, the game truncates with reward=0.

This simulation returns Q=0 to all edges on its path. That Q=0 provides **no useful signal** — it doesn't tell MCTS which programs are good or bad, only that this particular path was a dead end. The simulation was wasted.

With 50 simulations and (for P(7)) 67% of productions being dead ends, a significant fraction of simulations will follow dead-end paths, especially early in training when the network's policy is near-uniform.

**With the fix — every simulation reaches a real program:**

Dead-end productions are removed from the action list before MCTS ever sees them. Every production MCTS can choose leads to a derivation that will eventually complete. Every simulation produces a finished program with a real reward from `LeafEvaluator`. Every Q-value backup carries genuine signal about program quality.

Additionally, the action space shrinks (60 → 48), so MCTS's 50 simulations are spread over fewer actions, giving each viable action more visits on average. More visits per action = more accurate Q-value estimates = better action selection.

**Summary:** The guard's own cost is negligible (cached O(1) lookup). The benefit is that it prevents MCTS from spending simulations on paths that can never produce a program. Every simulation now contributes useful search signal.

---

## 2. The Problem — A Concrete Example (Before the Fix)

Let's use a small example: **budget=7, n_sites=3**.

### 2.1 The grammar productions for P(7)

The Ite template is: `P(k) → Ite(C(i), Flip(j), P(k-2-i))`.

For P(7), the loop iterates `i` from 1 to `k-4 = 3`:

| i | else_budget = 7-2-i | Production template |
|---|---------------------|---------------------|
| 1 | 4 | `P(7) → Ite(C(1), Flip(j), P(4))` |
| 2 | 3 | `P(7) → Ite(C(2), Flip(j), P(3))` |
| 3 | 2 | `P(7) → Ite(C(3), Flip(j), P(2))` |

Each template generates 3 productions (one per `j ∈ {0,1,2}`), so **before the fix**, P(7) had **9 productions** total... but only the `i=3` row is viable.

### 2.2 Why P(4) and P(3) are dead ends

Let's trace `count_programs(3, 4)` through the code in `budget_grammar.py`:

```python
def count_programs(n_sites=3, budget=4):
    # budget < 2? No (4 >= 2), continue
    total = 0

    # budget == 2? No (4 != 2), skip Default branch

    # budget >= 5? No (4 < 5), skip Ite branch

    return 0   # ← ZERO programs exist at budget 4
```

The function returns 0 because:
- The `Default(Flip(j))` template requires exactly 2 nodes — budget 4 is too large
- The `Ite(C(i), Flip(j), P(k-2-i))` template requires at least 5 nodes — budget 4 is too small
- Budget 4 falls in the gap between these two templates

The same logic applies to `count_programs(3, 3)`:

```python
def count_programs(n_sites=3, budget=3):
    total = 0
    # budget == 2? No (3 != 2), skip Default
    # budget >= 5? No (3 < 5), skip Ite
    return 0   # ← ZERO programs at budget 3
```

### 2.3 A derivation that hits a dead end

Suppose MCTS (or a random policy) picks the production `P(7) → Ite(C(1), Flip(0), P(4))`.

**Step 0:** Start with `[P:7]`

**Step 1:** Apply `P(7) → Ite(C(1), Flip(0), P(4))`:
```
Ite([C:1], Flip(0), [P:4])
```
The partial AST now has two holes: `[C:1]` and `[P:4]`.

**Step 2:** The leftmost hole is `[C:1]`. Suppose we apply `C(1) → IsZero(0)`:
```
Ite(IsZero(0), Flip(0), [P:4])
```
Now only one hole remains: `[P:4]`.

**Step 3:** The leftmost hole is `[P:4]`. The game calls `_program_productions(4, 3)` to get legal productions for this hole. The function returns an **empty list** — no production exists for budget 4.

The game detects this:
```python
# derivation_game.py lines 154-168
is_dead_end = not is_complete and len(self._current_productions) == 0
truncated = is_dead_end
if is_dead_end:
    reward = 0.0   # ← All computation wasted
```

**Result:** The episode is truncated with reward=0. The work done expanding C(1) in step 2 was pointless — the derivation was doomed from the moment `P(4)` was created in step 1.

### 2.4 How bad was this?

For P(7) with n_sites=3, **before** the fix:

| i | else_budget | Productions generated | Outcome |
|---|-------------|----------------------|---------|
| 1 | P(4) | 3 (j=0,1,2) | **Dead end** — P(4) can never complete |
| 2 | P(3) | 3 (j=0,1,2) | **Dead end** — P(3) can never complete |
| 3 | P(2) | 3 (j=0,1,2) | Valid — P(2) completes via Default(Flip(j)) |

**6 out of 9 productions (67%) were dead ends.**

A uniform-random policy picking from these 9 productions had a 67% chance of immediately dooming the derivation at the very first step.

---

## 3. How the Guard Works — Same Example (After the Fix)

### 3.1 The code change

Two lines were added inside the Ite production loop in `_program_productions()`:

```python
# derivation.py, lines 168-180
# P(k) → Ite(C(i), Flip(j), P(k-2-i))  for k >= 5
if budget >= 5:
    for i in range(1, budget - 3):  # i in [1, k-4]
        else_budget = budget - 2 - i
        if count_programs(n_sites, else_budget) == 0:   # ← NEW LINE 1
            continue  # Skip dead-end budgets (e.g., 3 and 4)   # ← NEW LINE 2
        for j in range(n_sites):
            result = Ite(ConditionHole(i), Flip(j), ProgramHole(else_budget))
            prods.append(Production(...))
```

### 3.2 Tracing the loop for P(7), n_sites=3

Let's walk through `_program_productions(7, 3)` iteration by iteration.

**budget=7, so `budget >= 5` is true. Enter the Ite loop.**

---

**Iteration: i=1**
```
else_budget = 7 - 2 - 1 = 4
count_programs(3, 4) = 0    ← Is this zero? YES
→ continue                   ← SKIP this entire i value, don't generate any productions
```
The guard asks: "If I create a `ProgramHole(4)`, can it ever be completed into a full program?" The answer is no (count is 0), so we skip.

---

**Iteration: i=2**
```
else_budget = 7 - 2 - 2 = 3
count_programs(3, 3) = 0    ← Is this zero? YES
→ continue                   ← SKIP
```
Same situation — `ProgramHole(3)` can never be completed.

---

**Iteration: i=3**
```
else_budget = 7 - 2 - 3 = 2
count_programs(3, 2) = 3    ← Is this zero? NO (3 programs: Default(Flip(0/1/2)))
→ proceed to inner loop
```
`ProgramHole(2)` can be completed — there are 3 valid programs at budget 2. So we generate productions:

```
j=0: P(7) → Ite(C(3), Flip(0), P(2))   ✓ kept
j=1: P(7) → Ite(C(3), Flip(1), P(2))   ✓ kept
j=2: P(7) → Ite(C(3), Flip(2), P(2))   ✓ kept
```

---

**Result:** `_program_productions(7, 3)` returns **3 productions** (down from 9).

### 3.3 Before vs After comparison

```
BEFORE the fix — _program_productions(7, 3) returns 9 productions:
  i=1: Ite(C(1), Flip(0), P(4))  ← DEAD END
  i=1: Ite(C(1), Flip(1), P(4))  ← DEAD END
  i=1: Ite(C(1), Flip(2), P(4))  ← DEAD END
  i=2: Ite(C(2), Flip(0), P(3))  ← DEAD END
  i=2: Ite(C(2), Flip(1), P(3))  ← DEAD END
  i=2: Ite(C(2), Flip(2), P(3))  ← DEAD END
  i=3: Ite(C(3), Flip(0), P(2))  ✓
  i=3: Ite(C(3), Flip(1), P(2))  ✓
  i=3: Ite(C(3), Flip(2), P(2))  ✓

AFTER the fix — _program_productions(7, 3) returns 3 productions:
  i=3: Ite(C(3), Flip(0), P(2))  ✓
  i=3: Ite(C(3), Flip(1), P(2))  ✓
  i=3: Ite(C(3), Flip(2), P(2))  ✓
```

### 3.4 Larger example: P(14), n_sites=6

For the default training configuration (budget=14, n_sites=6):

| i | else_budget | count_programs(6, else_budget) | Kept? |
|---|-------------|-------------------------------|-------|
| 1 | 11 | 27,060,480 | Yes |
| 2 | 10 | 4,665,600 | Yes |
| 3 | 9 | 419,904 | Yes |
| 4 | 8 | 46,656 | Yes |
| 5 | 7 | 7,776 | Yes |
| 6 | 6 | 1,296 | Yes |
| 7 | 5 | 216 | Yes |
| 8 | 4 | **0** | **No — skipped** |
| 9 | 3 | **0** | **No — skipped** |
| 10 | 2 | 6 | Yes |

**Before:** 10 structural choices × 6 sites = 60 productions
**After:** 8 structural choices × 6 sites = 48 productions

The action space (`game.action_space.n`) shrinks from 60 to 48. This propagates automatically through `compute_max_productions()` → `DerivationGame.__init__()` → neural network policy head.

---

## 4. Why It's Correct

### 4.1 What `count_programs(n_sites, budget)` computes

`count_programs` is a recursive function (with `@lru_cache`) in `budget_grammar.py` that counts how many **complete, terminal programs** — fully expanded ASTs with zero holes — exist at exactly `budget` nodes:

```python
@functools.lru_cache(maxsize=None)
def count_programs(n_sites, budget):
    if budget < 2: return 0
    total = 0
    if budget == 2:
        total += n_sites                         # Default(Flip(j)) for j in [0, N)
    if budget >= 5:
        for i in range(1, budget - 3):
            else_budget = budget - 2 - i
            total += (count_conditions(n_sites, i)   # ways to fill condition
                      * n_sites                       # ways to pick Flip(j)
                      * count_programs(n_sites, else_budget))  # ways to fill else
    return total
```

This function mirrors the grammar rules exactly. It does NOT generate programs — it computes the count via multiplication of sub-counts.

### 4.2 What the guard means

The guard `count_programs(n_sites, else_budget) == 0` asks:

> "If I create a `ProgramHole(else_budget)`, is there any sequence of further productions that can fill it into a complete program?"

- **count > 0**: Yes, the hole can be completed. Keep the production.
- **count == 0**: No. The hole will eventually become the leftmost hole, have no legal productions, and trigger a dead-end truncation with reward=0. Skip the production.

### 4.3 Soundness

1. **`count_programs` is validated against physical enumeration.** The existing test `test_count_matches_enumerate_length` in `test_cfg_grammar.py` verifies that `count_programs(3, b) == len(enumerate_programs(3, b))` for budgets 1-8. The enumeration physically generates every program object and counts them; the count function gives the same number. So `count_programs == 0` truly means "zero complete programs exist at this budget."

2. **No valid programs are removed.** The terms we skip were always multiplied by zero in `count_programs` anyway — they contribute nothing to the total program count. `count_programs(6, 14)` returns 151,173,432 both before and after the fix.

3. **One-level check is sufficient.** After filtering, every remaining `ProgramHole` has budget 2 or ≥5. Budget 2 always has `n_sites` completions (via Default). Budget ≥5 always has at least one completion (Ite with the largest possible `i`, giving `else_budget=2`). No deeper dead-ends can arise.

---

## 5. Tests Added

### In `test_cfg_grammar.py`

**Helper: `_assert_no_dead_holes(node, n_sites)`**
Recursively walks a partial AST subtree. For every `ProgramHole(b)` found, asserts `count_programs(n_sites, b) > 0`. For every `ConditionHole(b)` found, asserts `count_conditions(n_sites, b) > 0`. This verifies that no hole in the subtree is a dead end.

**Test 1: `test_no_dead_end_productions`** (parametrized, budgets 2-14)
For each budget, calls `_program_productions(budget, 6)` and runs `_assert_no_dead_holes` on every production's result subtree. Verifies the core invariant: no production creates an uncompletable hole.

**Test 2: `test_program_counts_unchanged`**
Asserts golden values for `count_programs` at several budgets (including `count_programs(3, 3) == 0` and `count_programs(6, 14) == 151_173_432`). Proves that filtering dead-end productions did NOT change the number of reachable programs.

**Test 3: `test_random_derivations_never_truncate`**
Runs 1000 random derivation episodes at budget=14, n_sites=6. Each episode picks random legal productions until terminal. Asserts zero truncations. This is the end-to-end empirical check — before the fix, many episodes would truncate.

**Test 4: `test_derivation_enumeration_unchanged`** (parametrized, budgets 2, 5, 7)
Compares programs from `enumerate_via_derivation()` (which uses `_program_productions`) against `enumerate_programs()` (direct enumeration). Asserts the sets are identical. Proves the derivation machinery still finds all valid programs.

### In `test_derivation_game.py`

**Test 5: `test_action_space_size`**
Asserts `DerivationGame(14, 6, leaf_eval).action_space.n == 48`. Verifies the action space reduction propagated correctly through `compute_max_productions`.

**Test 6: `test_no_truncation_in_game_episodes`**
Plays 100 random game episodes through `DerivationGame` using `step_wrapper()` and `get_action_mask()`. Asserts every episode ends with `terminated=True` and `truncated=False`. Tests the complete game interface, not just the grammar layer.
