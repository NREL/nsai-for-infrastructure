# Refined Plan: Separate Generic Synthesis Logic from Domain-Specific Code

**Date:** 2026-03-09
**Branch:** `refactor/code_architecture`
**Status:** Ready for execution

## Context

The `synthesis/` module is the core DSL engine shared by both bitstring and doors domains. Its architecture is already structurally correct (both domains share the same AST nodes, grammar, and interpreter), but:

1. **Naming/documentation misleadingly brands the shared DSL as "BitString-specific"**, creating the false impression that doors inherits bitstring code.
2. **The interpreter has a bitstring-specific fallback** (`np.all(obs == 1.0)`) that already caused a documented bug for Doors (`test_doors_baselines.py:168-178`).
3. **Doors config files contain ~120 lines of copy-pasted `build()` methods** across 3 config variants with only 2-3 lines of actual variation.
4. **The duck-typed `GameConfig` interface** (`make_env`, `max_steps`) is implicit, making it hard to discover for new domains.

The scan grammar (`instances/bitstring/dsl/scan_grammar.py`) is a **separate search strategy** (permutation-based), not a duplicate CFG class. It is correctly located in `instances/bitstring/` and requires no changes.

---

## Codebase Comparison

| Plan Item | Classification | Existing Code |
|-----------|---------------|---------------|
| Fix docstrings in synthesis/ | **Modify** | `synthesis/ast_nodes.py`, `interpreter.py`, `leaf_evaluator.py`, `budget_grammar.py` |
| Make `is_solved` required | **Modify** | `synthesis/interpreter.py:259`, `leaf_evaluator.py:221`, ~15 call sites |
| Extract `build()` helpers | **Modify** | `instances/doors/dsl/derivation_config.py` (3 nearly-identical build methods) |
| Add Protocol for GameConfig | **New** | `synthesis/protocols.py` (new file) |
| Rename AST nodes (Flip, IsZero) | **Rejected** | Used in 21+ files; names are accurate DSL concepts |
| Move interpreter.py | **Rejected** | Already generic; both domains use it identically |
| Move scan_grammar.py | **Rejected** | Correctly in `instances/bitstring/dsl/` (bitstring-only) |
| Consolidate /scripts/ | **Rejected** | `benchmark/` already provides modern harness |

---

## Execution Plan

### Day 1: Change 1 — Fix docstrings in `synthesis/` (no code changes)

**Justification:** Zero-risk change that resolves the "doors inherits bitstring" confusion.

| File | Line(s) | Current | New |
|------|---------|---------|-----|
| `synthesis/ast_nodes.py` | 1-2 | "AST nodes for the BitString decision-list DSL" | "AST nodes for the decision-list DSL" |
| `synthesis/ast_nodes.py` | 35 | "flip bit at position *index*" | "select action at *index*" |
| `synthesis/interpreter.py` | 2 | "Interpreter for the BitString decision-list DSL" | "Interpreter for the decision-list DSL" |
| `synthesis/interpreter.py` | 173 | `f"bit {cond.index} = {int(state[cond.index])}"` | `f"obs[{cond.index}] = {int(state[cond.index])}"` |
| `synthesis/interpreter.py` | 128 | `solved: bool  # all bits == 1` | `solved: bool` |
| `synthesis/interpreter.py` | 198 | `env: A BitStringGym or ShapedBitStringGym instance` | `env: A Gymnasium-compatible environment` |
| `synthesis/leaf_evaluator.py` | 4 | "frozen BitString initial states" | "frozen initial states" |

### Day 1: Change 3 — Extract shared `build()` helpers in Doors configs (highest value)

**Justification:** Eliminates ~120 lines of duplication. All changes internal to one file.

**File:** `src/alphazeropp/instances/doors/dsl/derivation_config.py`

Add 5 helper methods to `DoorsDerivationConfig`:

- `_make_doors_cfg()` — creates `DoorsGameConfig` from `self.game.kwargs`
- `_make_leaf_evaluator(doors_cfg, n_sites)` — creates `LeafEvaluator` with progress_fn
- `_make_game(game_cls, leaf_eval, doors_cfg, **extra)` — creates game with shared params
- `_make_net(game, extra_features=0)` — creates `DerivationPolicyValueNet`
- `_make_training_stack(game, net)` — creates Agent, Trainer, Evaluator

Each of the 3 existing `build()` methods collapses to ~6 lines calling these helpers, showing only what varies (game class, macros, extra_features).

### Day 2: Change 4 — Add `Protocol` for duck-typed `GameConfig`

**Justification:** Makes implicit interface explicit; zero breakage (structural typing).

**New file:** `src/alphazeropp/synthesis/protocols.py`
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DSLGameConfig(Protocol):
    def make_env(self, n_sites: int, frozen_states=None): ...
    def max_steps(self, n_sites: int) -> int: ...
```

**Update:** `synthesis/leaf_evaluator.py` — type-annotate `game_config` parameter as `DSLGameConfig`.

### Day 2: Change 2 — Make `is_solved` required in `run_policy_episode`

**Justification:** Prevents the documented Doors pitfall from recurring for future domains.

**File:** `synthesis/interpreter.py`
- Remove `Optional` from `is_solved` parameter type
- Remove the `np.all(obs == 1.0)` fallback (line 259-260)
- Make `is_solved` a required keyword argument

**Call sites to update:**

| File | Action |
|------|--------|
| `synthesis/leaf_evaluator.py:48` | When `is_solved` is None, raise early in `__init__` |
| `instances/bitstring/dsl/derivation_config.py` | Pass `is_solved=lambda obs: bool(np.all(obs == 1.0))` to `LeafEvaluator` |
| `tests/test_dsl.py` (~10 calls) | Add `is_solved=lambda obs: bool(np.all(obs == 1.0))` |
| `tests/test_doors_baselines.py:177` | Update test to expect `ValueError` (or provide explicit callback) |
| `tests/test_doors_pddl_lite.py` | Already passes `is_solved` — no change |
| `tests/test_doors_macros.py:379` | Already passes `is_solved` — verify |
| `scripts/enumerate_dsl.py` (3 calls) | Add bitstring `is_solved` |
| `scripts/run_derivation_mcts.py` (4 calls) | Add bitstring `is_solved` |
| `utils/derivation_utils.py:137` | Accept `is_solved` as parameter, pass through |

---

## What NOT to Change

- **Do NOT rename `Flip`, `IsZero`, etc.** — 21+ import sites; names are accurate DSL concepts
- **Do NOT move `interpreter.py`** — generic for this DSL; both domains use it
- **Do NOT move `scan_grammar.py`** — correctly in `instances/bitstring/dsl/`
- **Do NOT consolidate `/scripts/`** — `benchmark/` already provides modern harness

---

## Verification

1. `python -m pytest tests/ -v` — all 25 test files pass
2. `python -c "from alphazeropp.instances.doors.dsl.derivation_config import DoorsDerivationConfig; DoorsDerivationConfig().build()"` — Doors builds correctly
3. `python -c "from alphazeropp.instances.bitstring.dsl.derivation_config import DerivationConfig; DerivationConfig().build()"` — Bitstring builds correctly
4. Key test files: `test_doors_baselines.py`, `test_dsl.py`, `test_doors_pddl_lite.py`, `test_doors_macros.py`, `test_factored_derivation_game.py`
