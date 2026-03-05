# BitString Size-Budget Grammar & Program Enumeration

## Overview

The size-budget grammar extends the BitString decision-list DSL with the ability
to enumerate ALL valid programs of a given AST size. Instead of hand-writing
policies or learning them via MCTS, the grammar systematically generates every
possible decision-list program with exactly L AST nodes, enabling:

- **Exhaustive search** for optimal programs at small sizes
- **Program space analysis** — understanding how the space grows with size
- **Baseline comparison** — finding the best possible program at each size
- **Foundation for grammar-guided MCTS** — the DerivationState provides the
  game interface for program synthesis via tree search

**Key capabilities:**
- Enumerate all programs with exact AST node count (budget correctness)
- Count programs efficiently without generating them
- Expand programs step-by-step via canonical leftmost-hole derivation
- Generate human-readable grammar summaries, derivation traces, and debug reports
- Evaluate enumerated programs on the BitString game with behavioral analysis
- Produce matplotlib plots for visual analysis

## Architecture

```
Budget Grammar                     Derivation Engine
  ProgramHole(budget)                DerivationState
  ConditionHole(budget)                .initial(budget)
  enumerate_programs(N, L)             .leftmost_hole()
  enumerate_conditions(N, k)           .legal_productions(N)
  count_programs(N, L)                 .apply(production)
  count_conditions(N, k)               .is_terminal()
         │                             .to_program()
         │                                │
         ▼                                ▼
  Existing DSL AST (unchanged)      Grammar → Program
    Flip, IsZero, Not, And            ┌─────────────────┐
    Ite, Default                      │ run_policy_episode│
    node_count(), pretty()            │ interp_ops()      │
    validate()                        │ format_trace()    │
         │                            └─────────────────┘
         ▼                                │
  Analysis Script                         ▼
    enumerate_dsl.py               Behavioral Results
    ├── Grammar summary              solve_rate, avg_steps,
    ├── Derivation trace             avg_interp_ops, avg_reward
    ├── Enumeration table
    └── Matplotlib plots
```

The grammar sits alongside the existing DSL. It does NOT modify the AST nodes
or interpreter — it provides a systematic way to generate programs that the
existing interpreter can evaluate.

## Hole Types

Holes are placeholders in a partial AST during derivation.

| Hole | Semantics | Pretty |
|------|-----------|--------|
| `ProgramHole(k)` | Placeholder for a Program subtree with exactly k nodes | `[P:k]` |
| `ConditionHole(k)` | Placeholder for a Condition subtree with exactly k nodes | `[C:k]` |

Holes are frozen dataclasses defined in `budget_grammar.py`. At runtime, they
can be placed directly into existing AST node fields (Python doesn't enforce
type annotations on frozen dataclasses), avoiding the need for parallel
"partial" node types.

## Grammar Productions

### Budget Accounting

Each production specifies how a hole with budget k expands into a (partial)
subtree whose total node count equals k. The budget math is derived directly
from the existing `node_count()` methods on AST nodes:

| Node | node_count() | Budget consumed |
|------|-------------|----------------|
| `Flip(i)` | 1 | 1 |
| `IsZero(i)` | 1 | 1 |
| `Not(c)` | 1 + c.node_count() | 1 + child budget |
| `And(l, r)` | 1 + l + r | 1 + left + right |
| `Ite(c, a, e)` | 1 + c + a + e | 1 + cond + 1 (Flip) + else |
| `Default(a)` | 1 + a.node_count() | 1 + 1 (Flip) = 2 |

### Program Productions: P(k)

| Budget | Production | Budget breakdown | Constraint |
|--------|-----------|-----------------|------------|
| k = 2 | `Default(Flip(j))` | Default(1) + Flip(1) = 2 | j in [0, N) |
| k >= 5 | `Ite(C(i), Flip(j), P(k-2-i))` | Ite(1) + C(i) + Flip(1) + P(k-2-i) = k | i in [1, k-4], j in [0, N) |

**Why k >= 5 for Ite:** Ite needs 1 (itself) + 1 (smallest condition) + 1
(action) + 2 (smallest else = Default(Flip)) = 5 minimum.

**Why k-2-i >= 2:** The else_prog must be at least `Default(Flip(j))` = 2
nodes. So i <= k-4.

**Impossible budgets:** P(1) is below minimum. P(3) and P(4) fall in the gap
between Default (2 nodes) and the smallest Ite (5 nodes).

### Condition Productions: C(k)

| Budget | Production | Budget breakdown | Constraint |
|--------|-----------|-----------------|------------|
| k = 1 | `IsZero(j)` | IsZero(1) = 1 | j in [0, N) |
| k >= 2 | `Not(C(k-1))` | Not(1) + C(k-1) = k | — |
| k >= 3 | `And(C(i), C(k-1-i))` | And(1) + C(i) + C(k-1-i) = k | i in [1, k-2] |

### Production Ordering

For deterministic, stable enumeration order:
1. **Programs:** Default before Ite; within Ite: ascending i, then ascending j
2. **Conditions:** IsZero (ascending j) before Not before And (ascending i)

## Worked Examples

### Example 1: Budget Accounting for Ite

```
P(5) -> Ite(C(1), Flip(j), P(2))

Budget check:
  Ite node:         1
  C(1) = IsZero(j): 1
  Flip(j):          1
  P(2) = Default:   2
  Total:            5 ✓
```

### Example 2: Impossible Budgets

```
P(3): need Default(2) or Ite(>=5). Neither fits budget 3. → 0 programs.
P(4): need Default(2) or Ite(>=5). Neither fits budget 4. → 0 programs.
```

### Example 3: Multiple Structural Patterns at P(8)

```
P(8) has two structural patterns:

Pattern 1: Ite(C(1), Flip(j), P(5))    — two-rule chain
  Budget: 1 + 1 + 1 + 5 = 8 ✓
  Count:  C(1)=3 × Flip=3 × P(5)=27 = 243

Pattern 2: Ite(C(4), Flip(j), P(2))    — complex condition + default
  Budget: 1 + 4 + 1 + 2 = 8 ✓
  Count:  C(4)=30 × Flip=3 × P(2)=3 = 270

Total P(8) = 243 + 270 = 513
```

### Golden Counts for N=3

| Budget | Programs | Conditions |
|--------|----------|------------|
| 1 | 0 | 3 |
| 2 | 3 | 3 |
| 3 | 0 | 12 |
| 4 | 0 | 30 |
| 5 | 27 | 111 |
| 6 | 27 | 363 |
| 7 | 108 | 1,353 |
| 8 | 513 | 4,917 |

## Derivation State

### Overview

A `DerivationState` represents a partial AST that may contain holes. It
supports canonical leftmost-hole expansion, ensuring each program has exactly
one derivation path.

### Canonical Leftmost-Hole Expansion

The leftmost hole is found via **preorder traversal** (root → left → right):
- `Ite`: visit cond, then action, then else_prog
- `Not`: visit child
- `And`: visit left, then right
- `ProgramHole`/`ConditionHole`: return immediately

At any `DerivationState`, only the leftmost hole can be expanded. This means:
- Each terminal program has exactly one derivation path
- No duplicate programs in the enumeration
- The derivation is deterministic and canonical

### Derivation Trace Example

```
=== Derivation Trace (budget=5) ===

Step 0: [P:5]
  Leftmost hole: [P:5]
  Apply: P(5) -> Ite(C(1), Flip(0), P(2))

Step 1: Ite([C:1], Flip(0), [P:2])
  Leftmost hole: [C:1]
  Apply: C(1) -> IsZero(0)

Step 2: Ite(IsZero(0), Flip(0), [P:2])
  Leftmost hole: [P:2]
  Apply: P(2) -> Default(Flip(0))

Step 3: Ite(IsZero(0), Flip(0), Default(Flip(0)))  [TERMINAL]
  node_count = 5 ✓
  Pretty:
    if IsZero(0):
      Flip(0)
    else:
      Flip(0)
```

### DerivationState API

| Method | Returns | Description |
|--------|---------|-------------|
| `DerivationState.initial(budget)` | `DerivationState` | Create initial state with a single ProgramHole |
| `.is_terminal()` | `bool` | True if no holes remain |
| `.leftmost_hole()` | `ProgramHole \| ConditionHole \| None` | Find the leftmost hole |
| `.legal_productions(n_sites)` | `list[Production]` | Productions for the leftmost hole |
| `.apply(production)` | `DerivationState` | Apply production, return new state |
| `.to_program()` | `Program` | Convert terminal derivation to AST (asserts terminal) |
| `.pretty()` | `str` | Pretty-print with holes shown as `[P:k]` or `[C:k]` |
| `.hole_count()` | `int` | Count remaining holes |

## Enumeration

### Direct Enumeration

`enumerate_programs(n_sites, budget)` generates all programs via recursive
decomposition with memoization (`@functools.lru_cache`). Sub-problems overlap
heavily (e.g., P(8) reuses P(5) and P(2)), so caching is effective.

### Derivation Enumeration

`enumerate_via_derivation(n_sites, budget)` generates the same programs by
DFS through the DerivationState space. This is used in tests to verify that
the derivation machinery produces the same set as direct enumeration.

### Counting

`count_programs(n_sites, budget)` counts programs without generating them,
using the same recursive structure but only tracking integers. Efficient for
estimating the size of larger budget levels.

## Behavioral Analysis

The analysis script (`scripts/enumerate_dsl.py`) evaluates each enumerated
program on the BitString game:

1. For each program, run `run_policy_episode()` across multiple random seeds
2. Track solve rate, average steps, average interpretation ops, average reward
3. Print a ranked table with the best programs highlighted
4. Generate matplotlib plots

### Example Output

```
=== Enumeration Analysis: N=3, L=5 (27 programs, 8 seeds each) ===

  #  Program                                        Solve%  AvgSteps  AvgOps   AvgRew
---------------------------------------------------------------------------------------
  1  if IsZero(0): Flip(0) else: Flip(0)             25.0%     4.75    9.50   +0.0833
  2  if IsZero(0): Flip(0) else: Flip(1)             87.5%     1.62    3.25   +0.2917
  3  if IsZero(0): Flip(0) else: Flip(2)             37.5%     4.12    8.25   +0.1250
  ...

Best by solve rate: if IsZero(0): Flip(0) else: Flip(1) -- 87.5%
Best by avg ops:    if IsZero(0): Flip(0) else: Flip(1) -- 3.25 avg ops
```

### Plots

**Plot 1: Program count by budget level** — Bar chart showing exponential
growth of the program space. Impossible budgets (L=3,4) are annotated with "0".

**Plot 2: Behavioral scatter** — Each dot is one program, plotted at (avg
interp_ops, solve rate). Color indicates root condition type (IsZero=blue,
Not=red, And=purple, Default=green). Pareto-optimal programs are annotated.

**Plot 3: State evolution** — For a selected program, shows a grid of bitstring
states across an episode. Bits are colored red (0) or green (1), with arrows
marking flipped bits. Shows how the policy plays the game step by step.

## Grammar Debug

The `format_grammar_debug()` function checks for common implementation bugs:

```
=== Grammar Debug (N=3) ===

Common bugs to check:
  P(2) is the base case (not P(1)) -- Default(Flip) = 2 nodes
  P(3), P(4) produce 0 programs (Ite minimum = 5 nodes)
  Ite budget: 1 + |cond| + 1 + |else| = k
  And budget: 1 + |left| + |right| = k
  Production order is stable (sorted by type, then index)

Budget achievability:
  L=1: 0 programs (minimum program is Default(Flip) = 2 nodes)
  L=2: 3 programs -- all node_count == 2
  L=3: 0 programs (gap between Default(2) and smallest Ite(5))
  ...

Consistency check: PASSED
```

### Common Pitfalls

1. **Wrong base case:** Plan `<a>` used `P(1) → Default(A)`, but in our AST
   `Default(Flip(j))` has `node_count() = 2`. The base case must be `P(2)`.

2. **Wrong budget subtraction for Ite:** The Ite node itself costs 1, and the
   action (Flip) costs 1. So else_prog budget = k - 2 - i (not k - 1 - i).

3. **Allowing P(k) → Default(A) for k > 2:** Default(Flip) always costs exactly
   2 nodes. There is no Default production for any budget other than 2.

4. **Unstable production ordering:** Productions must be generated in a
   deterministic order. We use: Default before Ite, IsZero before Not before
   And, ascending indices within each type.

5. **Short-circuit in And cost:** The cost model charges both sides of And,
   even if the left side is false. This is structural, not state-dependent.

## Integration with Existing Components

```
Existing DSL (unchanged)           Grammar (new)
    ast_nodes.py                      budget_grammar.py
    interpreter.py                    derivation.py
        │                                 │
        └────────────┬────────────────────┘
                     │
              dsl/__init__.py
              (re-exports all public API)
                     │
              scripts/enumerate_dsl.py
              (analysis + plots)
```

The grammar imports from `ast_nodes.py` but does not modify it. Enumerated
programs are standard AST nodes that work with the existing interpreter:
`eval_program()`, `interp_ops()`, `run_policy_episode()`, `format_trace()`.

## Usage

### Counting programs

```python
from alphazeropp.instances.bitstring.dsl import count_programs

print(count_programs(n_sites=3, budget=5))   # 27
print(count_programs(n_sites=3, budget=8))   # 513
print(count_programs(n_sites=3, budget=3))   # 0 (impossible)
```

### Enumerating programs

```python
from alphazeropp.instances.bitstring.dsl import enumerate_programs

for prog in enumerate_programs(n_sites=3, budget=5):
    print(prog.pretty())
    print(f"  node_count = {prog.node_count()}")
```

### Using the derivation engine

```python
from alphazeropp.instances.bitstring.dsl import DerivationState

state = DerivationState.initial(budget=5)
while not state.is_terminal():
    print(state.pretty())
    prods = state.legal_productions(n_sites=3)
    state = state.apply(prods[0])  # pick first production
prog = state.to_program()
print(prog.pretty())
```

### Running the analysis script

```bash
python scripts/enumerate_dsl.py --n_sites 3 --max_budget 8 --eval_seeds 32
python scripts/enumerate_dsl.py --n_sites 3 --max_budget 8 --eval_budget 5
```

### Tests

```bash
pytest tests/test_cfg_grammar.py -v
# 50 tests: counts, node_count correctness, uniqueness, determinism,
#           derivation, canonicality, formatting, edge cases
```

## File Index

| File | Role |
|------|------|
| `src/alphazeropp/instances/bitstring/dsl/budget_grammar.py` | Hole types, counting, enumeration, grammar formatting |
| `src/alphazeropp/instances/bitstring/dsl/derivation.py` | DerivationState, Production, derivation traces |
| `src/alphazeropp/instances/bitstring/dsl/__init__.py` | Public API re-exports (updated) |
| `scripts/enumerate_dsl.py` | Analysis script: enumerate, evaluate, plot |
| `tests/test_cfg_grammar.py` | 50 tests across 10 test classes |
| `specs/bitstring_cfg_grammar.md` | This document |
