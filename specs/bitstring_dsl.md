# BitString Decision-List DSL

## Overview

The decision-list DSL provides a way to express interpretable, hand-crafted
policies for the BitString game as explicit programs. Instead of learning a
policy via MCTS + neural network, you write a decision list that maps bitstring
states to actions. The interpreter evaluates the program on each state and
tracks a well-defined cost metric (`interp_ops`), enabling comparison of
programmatic policies against learned ones.

**Key capabilities:**
- Build policies as if/elif/else decision lists over bit conditions
- Evaluate them on `BitStringGym` or `ShapedBitStringGym` environments
- Track interpretation cost (interp_ops) per step and per episode
- Generate human-readable execution traces explaining each decision

## Architecture

```
DSL Program (AST)
    Ite(IsZero(0), Flip(0),
        Ite(IsZero(1), Flip(1),
            Default(Flip(3))))
    │
    ▼
Interpreter
    eval_program(prog, state) → action index
    interp_ops(prog, state)   → cost
    │
    ▼
run_policy_episode(env, prog) → EpisodeResult
    │                              ├── steps[]       (per-step records)
    │                              ├── total_env_steps
    │                              ├── total_interp_ops
    │                              ├── cumulative_reward
    │                              └── solved (bool)
    │
    ▼
format_trace(result, prog) → human-readable string
```

The DSL sits alongside the existing AlphaZero pipeline. It does NOT replace
MCTS/Agent — it provides an alternative policy mechanism for analysis.

```
BitStringConfig.build()
│
├─ AlphaZero path (existing)
│   └─ Agent(game, net, mcts_params) → MCTS-based policy
│
└─ DSL path (new)
    └─ run_policy_episode(env, program) → programmatic policy
```

## AST Node Reference

### Actions

| Node | Semantics | Example |
|------|-----------|---------|
| `Flip(i)` | Return action index `i` (flip bit at position `i`) | `Flip(0)` → action 0 |

### Conditions

| Node | Semantics | Example |
|------|-----------|---------|
| `IsZero(i)` | True iff `state[i] == 0` | `IsZero(2)` on `[1,1,0]` → True |
| `Not(c)` | Logical negation | `Not(IsZero(0))` on `[1,0]` → True |
| `And(l, r)` | Logical conjunction | `And(IsZero(0), IsZero(1))` on `[0,0]` → True |

### Programs

| Node | Semantics |
|------|-----------|
| `Ite(cond, action, else_prog)` | If `cond` is true, return `action`; otherwise evaluate `else_prog` |
| `Default(action)` | Always return `action`. Terminal case — guarantees totality |

**First-match semantics:** The decision list is evaluated top-down. The first
`Ite` whose condition is true fires. If no condition matches, `Default` at the
bottom always fires.

**Totality guarantee:** Every `Program` must end with a `Default` node. This is
enforced structurally by the type system: `Ite.else_prog` is a `Program`, and
the only base case is `Default`.

### Methods on All Nodes

| Method | Returns | Description |
|--------|---------|-------------|
| `node_count()` | `int` | Exact AST node count (each constructor = 1) |
| `pretty()` | `str` | Stable human-readable string |
| `validate(n_sites)` | `None` | Raises `ValueError` if any index is out of `[0, n_sites)` |

### pretty() Output Format

Programs print as an if/elif/else cascade:

```python
prog = Ite(IsZero(0), Flip(0),
           Ite(IsZero(1), Flip(1),
               Ite(IsZero(2), Flip(2),
                   Default(Flip(3)))))
print(prog.pretty())
```
```
if IsZero(0):
  Flip(0)
elif IsZero(1):
  Flip(1)
elif IsZero(2):
  Flip(2)
else:
  Flip(3)
```

### node_count() Examples

| Expression | Count | Breakdown |
|-----------|-------|-----------|
| `Flip(0)` | 1 | Flip |
| `Default(Flip(0))` | 2 | Default + Flip |
| `IsZero(0)` | 1 | IsZero |
| `Not(IsZero(0))` | 2 | Not + IsZero |
| `And(IsZero(0), IsZero(1))` | 3 | And + IsZero + IsZero |
| `Ite(IsZero(0), Flip(0), Default(Flip(1)))` | 5 | Ite + IsZero + Flip + Default + Flip |

## Interpreter Semantics

### eval_condition(cond, state) → bool

Evaluates a condition on a `numpy` bitstring state array.

### eval_program(program, state) → int

Returns the action index selected by the decision list for the given state.

### Cost Model: interp_ops(program, state) → int

The cost model tracks how many primitive operations the interpreter performs.

| Node | Cost | Notes |
|------|------|-------|
| `IsZero(i)` | 1 | One bit read |
| `Not(c)` | 1 + cost(c) | One boolean op + child |
| `And(l, r)` | 1 + cost(l) + cost(r) | **No short-circuit** — both sides always counted |
| `Flip(i)` | 1 | One action emit |
| `Ite(cond, act, else)` when cond=true | cost(cond) + 1 | Condition cost + Flip |
| `Ite(cond, act, else)` when cond=false | cost(cond) + interp_ops(else) | Condition cost + rest of list |
| `Default(act)` | 1 | Just the Flip |

**Key design decision:** `And` does NOT short-circuit in the cost model. Even
if the left operand is false, the cost of evaluating the right operand is still
charged. This is the simplest, most deterministic cost model.

**Condition cost is structural.** `_condition_ops(cond)` computes cost without
looking at the state. State-dependence only enters at the `Ite` level (to decide
which branch to charge).

### Worked Examples

**Program:** `Ite(IsZero(1), Flip(1), Default(Flip(3)))`

| State | Trace | Ops |
|-------|-------|-----|
| `[1,0,1,0]` | IsZero(1)=true (1) + Flip(1) (1) | **2** |
| `[1,1,1,0]` | IsZero(1)=false (1) + Default→Flip(3) (1) | **2** |

**Program:** `Ite(And(IsZero(0), IsZero(1)), Flip(0), Default(Flip(1)))`

| State | Trace | Ops |
|-------|-------|-----|
| `[1,0]` | And(1+1+1)=3, false → Default(1) | **4** |
| `[0,0]` | And(1+1+1)=3, true → Flip(1) | **4** |

**Program:** `Ite(IsZero(0), Flip(0), Ite(IsZero(1), Flip(1), Default(Flip(2))))`

| State | Trace | Ops |
|-------|-------|-----|
| `[0,1,1]` | IsZero(0)=true (1) + Flip(1) | **2** |
| `[1,0,1]` | IsZero(0)=false (1) + IsZero(1)=true (1) + Flip(1) | **3** |
| `[1,1,0]` | IsZero(0)=false (1) + IsZero(1)=false (1) + Default(1) | **3** |

## Episode Runner

### run_policy_episode(env, program, x0=None, verbose=False)

Runs the DSL program as a policy on a BitString environment. At each step:
1. Evaluate the program on the current state → get action
2. Compute interp_ops for this step
3. Build a rule trace (which rules matched/failed)
4. Execute `env.step(action)`

**Parameters:**
- `env`: A `BitStringGym` or `ShapedBitStringGym` instance
- `program`: An `Ite` or `Default` AST node
- `x0`: Optional initial state override (numpy array)
- `verbose`: If `True`, log detailed trace via `logging.debug()`

**Returns:** `EpisodeResult` with fields:
- `steps`: list of `StepRecord` (one per env step)
- `total_env_steps`: number of environment steps taken
- `total_interp_ops`: sum of interp_ops across all steps
- `final_state`: numpy array of the final state
- `cumulative_reward`: sum of rewards
- `solved`: `True` if all bits are 1 at the end

### format_trace(result, program=None) → str

Formats the complete episode as a human-readable trace:

```
=== Policy ===
if IsZero(0):
  Flip(0)
elif IsZero(1):
  Flip(1)
elif IsZero(2):
  Flip(2)
else:
  Flip(3)

Step 1: [0, 1, 0, 1]
  Rule 1: IsZero(0) -> TRUE -> Flip(0)
  reward=+0.2500, interp_ops=2

Step 2: [1, 1, 0, 1]
  Rule 1: IsZero(0) -> FALSE (bit 0 = 1)
  Rule 2: IsZero(1) -> FALSE (bit 1 = 1)
  Rule 3: IsZero(2) -> TRUE -> Flip(2)
  reward=+0.2500, interp_ops=4

=== Summary ===
Final state: [1, 1, 1, 1] -- SOLVED
Env steps: 2 | Total interp ops: 6 | Cumulative reward: 0.5000
```

The trace explains:
- Which rules were tried and why they failed (with bit values)
- Which rule finally matched
- The reward and interpretation cost for each step
- Whether the episode solved the puzzle

## Integration with Existing Components

The DSL works with both raw `BitStringGym` and the `ShapedBitStringGym` wrapper:

```
BitStringGym (raw)          ShapedBitStringGym (shaped rewards)
    │                              │
    └──────────┬───────────────────┘
               │
       run_policy_episode(env, program)
               │
           EpisodeResult
```

Both expose `.n_sites`, `.step()`, `.reset()`. The `ShapedBitStringGym` proxies
`.n_sites` via `__getattr__`, so the interpreter works transparently with either.

When using `ShapedBitStringGym`, the rewards in the trace are the shaped rewards
(potential differences), not the raw BitStringGym rewards.

## Usage

### Building and printing a policy

```python
from alphazeropp.instances.bitstring.dsl import (
    Flip, IsZero, Not, And, Ite, Default,
)

# Greedy OneMax: flip the first zero bit
prog = Ite(IsZero(0), Flip(0),
       Ite(IsZero(1), Flip(1),
       Ite(IsZero(2), Flip(2),
       Default(Flip(3)))))

print(prog.pretty())
print(f"Node count: {prog.node_count()}")
prog.validate(n_sites=4)  # checks all indices in [0, 4)
```

### Running an episode

```python
from alphazeropp.instances.bitstring.dsl import run_policy_episode
from alphazeropp.instances.bitstring.dsl.interpreter import format_trace
from alphazeropp.instances.bitstring.game import BitStringGym
from alphazeropp.instances.bitstring.shaped_env import ShapedBitStringGym
from alphazeropp.instances.bitstring.potentials import onemax

env = ShapedBitStringGym(
    BitStringGym(n_sites=4, bit_flip=True, sparse_reward=False),
    onemax, "dense_potential",
)

result = run_policy_episode(env, prog)
print(format_trace(result, program=prog))
print(f"Solved: {result.solved}")
print(f"Steps: {result.total_env_steps}")
print(f"Interp ops: {result.total_interp_ops}")
```

### Running with a specific initial state

```python
import numpy as np

x0 = np.array([1.0, 0.0, 1.0, 0.0])
result = run_policy_episode(env, prog, x0=x0)
```

### Tests

```bash
pytest tests/test_dsl.py -v
# 52 tests: node_count, pretty, eval, totality, interp_ops, behavioral, runner, validation
```

## File Index

| File | Role |
|------|------|
| `src/alphazeropp/instances/bitstring/dsl/__init__.py` | Public API re-exports |
| `src/alphazeropp/instances/bitstring/dsl/ast_nodes.py` | AST node dataclasses (Flip, IsZero, Not, And, Ite, Default) |
| `src/alphazeropp/instances/bitstring/dsl/interpreter.py` | eval_condition, eval_program, interp_ops, run_policy_episode, format_trace |
| `tests/test_dsl.py` | 52 tests across 8 test classes |
| `specs/bitstring_dsl.md` | This document |
