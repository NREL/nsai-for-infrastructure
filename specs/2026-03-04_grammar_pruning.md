# Grammar Pruning: Restrict Flip Indices & One-Hot Satisfiability

**Date:** 2026-03-04
**Status:** Proposed
**Goal:** Reduce grammar junk by (a) restricting Flip parameters to valid env actions and (b) pruning unsatisfiable conditions caused by one-hot observation constraints.

---

## Context

The shared synthesis grammar (`derivation.py`) is domain-agnostic: `Flip(j)` for j in `[0, n_sites)` and `IsZero(j)` for j in `[0, n_sites)`. In the **Doors** domain this creates junk:

1. **Invalid Flip indices**: Doors has `n_actions = M + K + 1` valid actions (MOVE_TO 0..M-1, PICK M..M+K-1, NOOP M+K) but `n_sites = M + 2D - 1` observation slots. Programs with `Flip(j)` for j >= n_actions produce NOOP/invalid — wasted search.

2. **Unsatisfiable conditions**: `at_loc[0..M-1]` is one-hot. An `And` chain with two positive location literals (e.g., `And(Not(IsZero(0)), Not(IsZero(1)))` = "at loc 0 AND at loc 1") is always false. MCTS wastes rollouts exploring these dead branches.

**Example (D=2, M=4, K=1):**
- n_sites=7, n_actions=6. Flip(6) is invalid → 1 junk action per program terminal.
- `And(Not(IsZero(0)), Not(IsZero(1)))`: impossible (one-hot) → entire subtree is wasted.

---

## Issues with Original Proposal

1. **Wrong file paths**: Proposal says `instances/bitstring/dsl/derivation.py` and `instances/bitstring/dsl/doors_config.py`. Actual paths are `src/alphazeropp/synthesis/derivation.py` and `src/alphazeropp/instances/doors/dsl/doors_config.py`.

2. **Parameter threading not addressed**: Proposal says "implement a helper" but doesn't address how `n_actions` and `one_hot_groups` flow from Doors config through `DerivationGame` → `legal_productions()` → `_program_productions()`.

3. **Satisfiability check needs careful design**: The check requires knowing the And-context of the leftmost hole (what conditions are conjoined with it). This requires a new tree traversal function that tracks negation parity and And-sibling accumulation. Also must handle De Morgan correctly: under an odd number of `Not` ancestors, an `And` becomes an effective `Or`, so siblings don't constrain each other.

---

## Design

### Task 1: Restrict Flip to valid action indices

**Approach**: Add `n_actions: int | None` parameter (default `None` → use `n_sites`) to:
- `_program_productions(budget, n_sites, mode, n_actions=None)` — use `n_actions or n_sites` for `range()` in Flip generation (lines 172, 187)
- `compute_max_productions(budget, n_sites, mode, ..., n_actions=None)` — pass through
- `DerivationState.legal_productions(..., n_actions=None)` — pass through
- `DerivationGame.__init__(..., n_actions=None)` — store and pass through

**Doors config**: Set `n_actions = M + K + 1` in `DoorsDerivationConfig.build()`.

**Backward compat**: Bitstring config doesn't pass `n_actions` → defaults to `None` → uses `n_sites`. All existing tests and callers unchanged.

### Task 2: One-hot satisfiability pruning

**Approach**: Add `one_hot_groups: list[list[int]] | None` parameter (default `None` → no pruning) through the same chain as `n_actions`.

**New functions in `derivation.py`:**

#### `_find_leftmost_hole_with_context(node)`

Like `_find_leftmost_hole()` but also returns the **And-context** and **negation parity** for the hole:

```
Returns: (hole, and_siblings: list[Condition], negated: bool) | None
```

- `and_siblings`: complete condition subtrees that are AND-conjoined with the hole
- `negated`: True if the hole is under an odd number of `Not` ancestors (within the condition tree)

**Key rules:**
- When entering `Not(child)`: flip `negated`
- When entering `And(left, right)` with `negated=False` (real conjunction):
  - Recurse left. If hole found there, return (no sibling context from right).
  - If not, recurse right with `and_siblings + [left]` (left is complete, add to context).
- When entering `And(left, right)` with `negated=True` (effective disjunction via De Morgan):
  - Recurse both children with **empty** `and_siblings` (siblings don't constrain under Or).
- When entering `Ite(cond, action, else_prog)`:
  - Fresh context: recurse `cond` with `negated=False, and_siblings=[]`
  - Then `else_prog` with `negated=False, and_siblings=[]`
  - (Cross-Ite implication reasoning is out of scope.)

#### `_extract_literals(cond) → set[(index, is_positive)]`

Extract simple literals from a complete condition:
- `IsZero(j)` → `{(j, False)}` (state[j]==0, "negative")
- `Not(IsZero(j))` → `{(j, True)}` (state[j]!=0, "positive")
- `And(l, r)` → recurse both sides, union
- `Not(And(...))` or other complex → `set()` (conservative: skip)

#### `_one_hot_contradiction(context_literals, new_index, new_positive, index_to_group) → bool`

Returns True if adding literal `(new_index, new_positive)` contradicts any existing context literal:
1. **Direct contradiction**: same index, opposite polarity → always False
2. **One-hot contradiction**: two positive ("is at") literals in same one-hot group → impossible

`index_to_group: dict[int, int]` maps observation index → group ID (precomputed from `one_hot_groups`).

#### Integration in `legal_productions()`

After generating candidate productions for a ConditionHole:
- If `one_hot_groups` is set and the candidate is a **terminal** (IsZero(j)):
  1. Compute `(hole, and_siblings, negated)` via `_find_leftmost_hole_with_context()`
  2. Collect `context_literals` from `and_siblings` via `_extract_literals()`
  3. Determine effective polarity: `new_positive = negated` (if under Not, IsZero becomes positive)
  4. Filter: keep production only if `_one_hot_contradiction()` returns False

**Why only terminal conditions?** At higher budgets, productions create structural nodes (Not, And) with new holes — we can't predict what literals they'll eventually produce. Conservative: only prune when we know the exact literal.

### Files Modified

| # | File | Change |
|---|------|--------|
| 1 | `src/alphazeropp/synthesis/derivation.py` | Add `_find_leftmost_hole_with_context()`, `_extract_literals()`, `_one_hot_contradiction()`. Update `_program_productions()` with `n_actions`. Update `legal_productions()` with `n_actions` + `one_hot_groups` filtering. |
| 2 | `src/alphazeropp/synthesis/derivation_game.py` | Add `n_actions`, `one_hot_groups` params to `__init__()`. Pass to `legal_productions()` in `reset()` and `step()`. Update `compute_max_productions()` with `n_actions`. |
| 3 | `src/alphazeropp/instances/doors/dsl/derivation_config.py` | In `build()`: compute `n_actions = M + K + 1`, `one_hot_groups = [list(range(M))]`. Pass to `DerivationGame()`. |
| 4 | `scripts/run_doors_derivation.py` | Update `print_banner()` call to `compute_max_productions()` with `n_actions`. |
| 5 | `tests/` | New test file for grammar pruning (Flip restriction + one-hot contradiction). |

### Backward Compatibility

- All new parameters default to `None` → no behavior change for bitstring or existing tests
- `DerivationGame` constructor: new kwargs are optional
- `_program_productions()`: `n_actions=None` → `range(n_sites)` as before
- `legal_productions()`: `one_hot_groups=None` → no filtering
- `compute_max_productions()`: `n_actions=None` → passed through unchanged

---

## Verification

1. **Unit test — Flip restriction**: Create `DerivationGame(budget=5, n_sites=7, ..., n_actions=6)`. Check that all program productions use `Flip(j)` with `j < 6` only.

2. **Unit test — One-hot pruning**: Build a partial AST `And(Not(IsZero(0)), ConditionHole(1))` with `one_hot_groups=[[0,1,2,3]]`. Expand `ConditionHole(1)` → confirm `IsZero(1)`, `IsZero(2)`, `IsZero(3)` are pruned (would create `Not(IsZero(j))` contradicting `Not(IsZero(0))` in one-hot group). `IsZero(0)` and `IsZero(4..6)` should remain.

3. **Regression**: Run `pytest tests/ -x -q --ignore=tests/test_zoning_game.py` — all existing tests pass.

4. **Branching comparison**: Run a 1-iteration derivation job and compare max/avg branching factor at root vs before the change.

5. **Program count check**: Verify `compute_max_productions()` returns a smaller value with `n_actions=6` vs `n_actions=None` for the Doors D=2 config.
