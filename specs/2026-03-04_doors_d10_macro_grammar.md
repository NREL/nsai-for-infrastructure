# Domain-Aware Macro Grammar for Doors D=10

**Date**: 2026-03-04
**Status**: Implementation plan
**Depends on**: `2026-03-04_grammar_redesign_for_scale.md` (problem analysis), `FactoredDerivationGame` (already implemented)

---

## Goal

Introduce domain-aware macro productions (PickRule, MoveRule) and a condition budget cap to make the factored derivation game tractable at D=10. Target: max_factored ≤ 40, derivation depth ≤ 40 steps.

## Approach: (B) Macro Productions (not Sketch)

Sketch mode (fixed decision-list template with parameter holes) was rejected:
- Too rigid — hardcodes optimal structure, can't discover novel programs
- Not really synthesis — reduces to parameter search
- Harder to extend to new domains

Macros co-exist with the base grammar. They are production-level shortcuts that expand a ProgramHole into a larger pre-shaped subtree. The resulting AST uses only existing node types (Ite, And, Not, IsZero, Flip). No interpreter or evaluator changes needed.

---

## 1. D=10 Parameters

From `compute_doors_derived_params(10, 2)`:

| Parameter | Value | Formula |
|-----------|-------|---------|
| num_rooms (D) | 10 | given |
| locs_per_room | 2 | given |
| M (locations) | 20 | D × locs_per_room |
| K (keys) | 9 | D - 1 |
| n_sites | 39 | M + 2D - 1 |
| n_actions | 30 | M + K + 1 |
| optimal_nodes | 92 | 10(D-1) + 2 |
| budget | 138 | int(92 × 1.5), rounded to even |
| horizon | 95 | max(15, 5 × optimal_steps) |
| optimal_steps | 19 | 2(D-1) + 1 |

**Note**: The earlier spec used optimal_nodes=68 / budget=102 (PICK-only program without MOVE rules). The code correctly uses 92 / 138 which includes both PICK (7 nodes × 9) and MOVE (3 nodes × 9) rules + Default (2 nodes).

---

## 2. Macro Design

### 2.1 PickRule Macro (7 AST nodes)

```
P(k) → Ite(And(Not(IsZero(key_loc[key_id])), Not(IsZero(M+D+key_id))),
            Flip(M + key_id),
            ProgramHole(k - 7))
```

- **Semantics**: "If agent is at key location AND key is available → PICK that key"
- **Budget cost**: 7 nodes (Ite + And + Not + IsZero + Not + IsZero + Flip)
- **Available when**: ProgramHole budget ≥ 9 (7 for macro + 2 minimum for remaining Default)
- **Parameters**: key_id ∈ [0, K-1] = [0, 8] for D=10
- **macro_key**: `("PickRule",)`

### 2.2 MoveRule Macro (3 AST nodes)

```
P(k) → Ite(IsZero(M + key_unlocks[key_id]),
            Flip(key_loc[key_id]),
            ProgramHole(k - 3))
```

- **Semantics**: "If target room is locked → move to key location"
- **Budget cost**: 3 nodes (Ite + IsZero + Flip)
- **Available when**: ProgramHole budget ≥ 5
- **Parameters**: key_id ∈ [0, K-1]
- **macro_key**: `("MoveRule",)`

### 2.3 Condition Budget Cap

New `max_condition_budget` parameter restricts base Ite templates to condition budgets [1, cap]:

```python
# In _program_productions(), the Ite loop becomes:
max_i = min(budget - 3, max_condition_budget + 1) if max_condition_budget else budget - 3
for i in range(1, max_i):
```

For doors: max useful condition = And(Not(IsZero), Not(IsZero)) = 5 nodes. Set cap = 12 (generous headroom).

### 2.4 Why Both Features

| Config | Structures at root | max_factored | Derivation depth |
|--------|-------------------|-------------|-----------------|
| Baseline (no macros, no cap) | 135 | 135 | ~119 steps |
| Cap only (cap=12) | 13 | 39 | ~119 steps |
| Macros only | 137 | 137 | ~38 steps |
| **Both (cap=12 + macros)** | **15** | **39** | **~38 steps** |

Macros cut depth 3.1×. Cap cuts branching 9×. Together: max_factored=39, depth=38.

---

## 3. Optimal D=10 Program (92 nodes)

The optimal program interleaves PICK and MOVE rules (PICK before MOVE for each key):

```
Ite(And(Not(IsZero(1)),  Not(IsZero(30))), Flip(20),     # PickRule(0): PICK key 0
Ite(IsZero(21), Flip(1),                                   # MoveRule(0): go to key 0 loc
Ite(And(Not(IsZero(3)),  Not(IsZero(31))), Flip(21),     # PickRule(1): PICK key 1
Ite(IsZero(22), Flip(3),                                   # MoveRule(1): go to key 1 loc
Ite(And(Not(IsZero(5)),  Not(IsZero(32))), Flip(22),     # PickRule(2)
Ite(IsZero(23), Flip(5),                                   # MoveRule(2)
Ite(And(Not(IsZero(7)),  Not(IsZero(33))), Flip(23),     # PickRule(3)
Ite(IsZero(24), Flip(7),                                   # MoveRule(3)
Ite(And(Not(IsZero(9)),  Not(IsZero(34))), Flip(24),     # PickRule(4)
Ite(IsZero(25), Flip(9),                                   # MoveRule(4)
Ite(And(Not(IsZero(11)), Not(IsZero(35))), Flip(25),     # PickRule(5)
Ite(IsZero(26), Flip(11),                                  # MoveRule(5)
Ite(And(Not(IsZero(13)), Not(IsZero(36))), Flip(26),     # PickRule(6)
Ite(IsZero(27), Flip(13),                                  # MoveRule(6)
Ite(And(Not(IsZero(15)), Not(IsZero(37))), Flip(27),     # PickRule(7)
Ite(IsZero(28), Flip(15),                                  # MoveRule(7)
Ite(And(Not(IsZero(17)), Not(IsZero(38))), Flip(28),     # PickRule(8)
Ite(IsZero(29), Flip(17),                                  # MoveRule(8)
Default(Flip(19))                                           # MOVE_TO(goal=19)
))))))))))))))))))
```

**Index mappings** (D=10, locs_per_room=2):
- key_loc[k] = 2k + 1 → [1, 3, 5, 7, 9, 11, 13, 15, 17]
- key_avail[k] = M + D + k = 30 + k → [30, 31, ..., 38]
- PICK(k) = M + k = 20 + k → [20, 21, ..., 28]
- room_unlocked[r] = M + r = 20 + r → [20, 21, ..., 29]

**Ordering matters**: PICK_k must precede MOVE_k for each key. Otherwise the agent would loop at a key location (MOVE fires → already there → MOVE fires again). MCTS discovers this ordering through self-play.

**Derivation with macros**: 9 PickRules × 2 steps + 9 MoveRules × 2 steps + 1 Default × 2 steps = **38 game steps**.

---

## 4. Integration with FactoredDerivationGame

### 4.1 Production.macro_key

New frozen field on Production dataclass:

```python
@dataclass(frozen=True)
class Production:
    hole_kind: str
    hole_budget: int
    result: Any
    label: str
    macro_key: tuple | None = None  # NEW
```

When `macro_key` is set, `_structure_key()` returns it directly instead of inspecting `prod.result`. This is **mandatory** — macro Ite results have complete conditions (e.g., `And(Not(IsZero(1)), Not(IsZero(30)))`) that lack a `.budget` attribute, which would crash the existing `_structure_key` logic.

### 4.2 Template Grouping

All 9 PickRule(key_id=0..8) variants share `macro_key=("PickRule",)`:
- Grouped into 1 `StructureTemplate(key=("PickRule",), productions=[...], needs_parameter=True)`
- Structure phase: agent sees "PickRule" as one option
- Parameter phase: agent chooses key_id from 0..8

Same for MoveRule.

### 4.3 `_compute_templates` Flow

```python
def _compute_templates(self):
    prods = self._deriv_state.legal_productions(
        ..., max_condition_budget=self._max_condition_budget)
    if self._macro_productions_fn is not None:
        hole = self._deriv_state.leftmost_hole()
        if isinstance(hole, ProgramHole):
            prods = prods + self._macro_productions_fn(hole.budget)
    return _group_by_structure(prods)
```

Macros are only appended when the current leftmost hole is a ProgramHole (not a ConditionHole).

### 4.4 MCTS Compatibility

- **hashable_obs**: Macro-produced AST nodes are standard (Ite/And/Not/IsZero/Flip), so `pretty()` output is correct regardless of production path. Macro vs base grammar reaching the same AST state merges correctly in the MCTS tree.
- **stash/unstash**: No new state to save (macros are stateless production generators).
- **clone()**: Must forward `macro_productions_fn` and `max_condition_budget` to the new instance.

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `src/alphazeropp/synthesis/derivation.py` | Add `macro_key` to Production; add `max_condition_budget` to `_program_productions` and `legal_productions` |
| `src/alphazeropp/synthesis/factored_derivation_game.py` | `_structure_key` macro check; `__init__`, `_compute_templates`, `compute_max_factored_actions`, `clone()` updates |
| `src/alphazeropp/instances/doors/dsl/derivation_config.py` | Add `DoorsFactoredD10MacroConfig` |
| `scripts/run_doors_derivation.py` | Add `doors_d10_macro` mode |
| **NEW** `src/alphazeropp/instances/doors/dsl/doors_macros.py` | `doors_macro_productions(budget, doors_cfg)` |
| **NEW** `tests/test_doors_macros.py` | Comprehensive tests |

**NOT modified**: `ast_nodes.py`, `interpreter.py`, `leaf_evaluator.py` — macros use existing AST nodes.

---

## 6. Verification Plan

1. **Construct optimal D=10 program** manually, verify `solve_rate=1.0` under LeafEvaluator with DoorsPDDLLiteEnv(D=10)
2. **Action space**: `compute_max_factored_actions(138, 39, "max", macros, cap=12)` = 39
3. **Reachability**: Macro-produced AST identical to base-grammar-produced AST for same program
4. **Training smoke test**: 2-iteration run with D=10 macro config completes, best program > random baseline
5. **Regression**: All existing tests pass (macros default to None, cap defaults to None)

---

## 7. Training Configuration

```python
DoorsFactoredD10MacroConfig:
    num_rooms = 10
    budget = 138
    max_condition_budget = 12
    n_simulations = 400          # up from 200 (more sims for harder problem)
    n_games_per_train = 80       # up from 40 (shorter games → more games needed)
    n_iterations = 50            # up from 30
    accept_threshold = 0.55
```
