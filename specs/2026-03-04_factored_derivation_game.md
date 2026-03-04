# Factored Derivation Game: Split (Structure, Parameter) Actions

**Date:** 2026-03-04
**Status:** Proposed
**Goal:** Reduce per-step branching from O(structures × parameters) to O(max(structures, parameters)) by splitting each grammar production into a structure choice then an optional parameter choice. Lossless: same reachable programs. Designed for the Doors domain.

---

## Context

The current `DerivationGame` presents each grammar production as a single atomic action. For program holes P(k), productions combine a structure template with a Flip index:
- `Default(Flip(j))` — 1 template × n_actions parameters
- `Ite(C(i), Flip(j), P(k-2-i))` — (k-4) templates × n_actions parameters each

This cross-product blows up the action space. For Doors D=10 (budget=102, n_actions=30), the flat action space is **2970** at the root. MCTS must spread simulations across all arms.

**Factored approach:** Each derivation step becomes 1–2 game steps:
1. **Structure phase** — pick a template (e.g., "Default" or "Ite(C(3), ?, P(97))")
2. **Parameter phase** — pick the Flip/IsZero index (only if template is parameterized)

If the chosen structure has no parameter (e.g., `Not(C(k-1))`, `And(C(i), C(j))`), apply immediately in the same `step()` — no parameter phase.

**Branching impact (Doors configs):**

| Config | budget | n_sites | n_actions | Flat max | Factored max | Reduction |
|--------|--------|---------|-----------|----------|--------------|-----------|
| D=2 L=2 | 18 | 7 | 6 | 90 | 15 | 6× |
| D=5 L=2 | 50 | 19 | 15 | ~705 | 47 | ~15× |
| D=10 L=2 | 102 | 39 | 30 | 2970 | 99 | 30× |

---

## Design

### Files

| # | File | Action |
|---|------|--------|
| 1 | `src/alphazeropp/synthesis/factored_derivation_game.py` | **CREATE** — domain-agnostic factored game class |
| 2 | `src/alphazeropp/synthesis/derivation_network.py` | **MODIFY** — add `extra_features` param for extended obs `(2*budget+2,)` |
| 3 | `src/alphazeropp/instances/doors/dsl/derivation_config.py` | **MODIFY** — add `DoorsFactoredDerivationConfig` |
| 4 | `scripts/run_doors_derivation.py` | **MODIFY** — add `doors_factored` mode to menu + banner |
| 5 | `tests/test_factored_derivation_game.py` | **CREATE** — unit + integration tests |

### Structure Templates

**ProgramHole(k):** All templates are parameterized by Flip index j.
- Template 0: `Default(Flip(?))` → needs parameter
- Template 1..s: `Ite(C(i), Flip(?), P(k-2-i))` for each valid i → needs parameter
- Count (max mode): `1 + max(0, k - 4)` for k ≥ 2

**ConditionHole(k):** Mixed — some need parameter, some don't.
- `IsZero(?)` → needs parameter (observation index)
- `Not(C(k-1))` → **no parameter** → apply immediately
- `And(C(i), C(k-1-i))` for each valid i → **no parameter** → apply immediately
- Count (max mode): `1 + (1 if allow_not and not parent_is_not and k≥2) + ((k-1)//2 if allow_and and k≥3)`

### Game Flow

```
reset() → structure phase (for root ProgramHole)

step(action) in structure phase:
  ├─ template.needs_parameter == True
  │    → store pending template, switch to parameter phase
  │    → obs encodes phase=1 + pending_structure_id
  │    → reward = 0, terminated = False
  │
  └─ template.needs_parameter == False  (Not, And)
       → apply the single Production to DerivationState immediately
       → switch to structure phase for next leftmost hole
       → reward/terminated as in DerivationGame

step(action) in parameter phase:
  → build full Production from pending template + parameter
  → apply to DerivationState
  → switch to structure phase for next leftmost hole
  → reward/terminated as in DerivationGame
```

### Action Space

```python
action_space = Discrete(max(max_structures, max_params))
```

Where:
- `max_structures`: max over all possible hole types/budgets of template count
  - Dominated by P(budget) in max mode: `1 + (budget - 4)` = `budget - 3`
- `max_params`: `max(n_actions or n_sites, n_sites)` = `n_sites`

**`compute_max_factored_actions(budget, n_sites, mode, ...)`**: iterates over all possible hole budgets, computes template count for each, returns `max(max_structures, max_params)`.

**Action mask:**
- Structure phase: `mask[:len(current_templates)] = True`
- Parameter phase: `mask[:len(current_params)] = True`

### Observation Encoding

Reuse `_preorder_items()` from `derivation_game.py`, append 2 floats:
- `phase_id`: 0.0 (structure) or 1.0 (parameter)
- `pending_structure_id`: 0.0 (none) or 1-indexed template ordinal

Shape: `(2 * budget + 2,)`.

### Network Changes (`derivation_network.py`)

`DerivationTransformerModel.__init__`:
- Add `extra_features: int = 0` parameter (default 0 → backward compatible)
- If `extra_features > 0`: add `self.extra_proj = nn.Linear(extra_features, d_model)`

`forward()`:
- Split obs: AST part = `x[:, :2*budget]`, extra = `x[:, 2*budget:]`
- If extra features exist: project and add to CLS embedding before transformer
- Existing configs pass `extra_features=0` → no change in behavior

### Key Data Structures (`factored_derivation_game.py`)

```python
@dataclass
class StructureTemplate:
    key: tuple                     # ("Default",) or ("Ite", cond_budget, else_budget)
    productions: list[Production]  # all parameter variants
    needs_parameter: bool          # True iff len(productions) > 1

def _structure_key(prod: Production) -> tuple:
    """Extract template identity from a production's result."""
    r = prod.result
    if isinstance(r, Default):   return ("Default",)
    if isinstance(r, Ite):       return ("Ite", r.cond.budget, r.else_prog.budget)
    if isinstance(r, IsZero):    return ("IsZero",)
    if isinstance(r, Not):       return ("Not", r.child.budget)
    if isinstance(r, And):       return ("And", r.left.budget, r.right.budget)

def _group_by_structure(prods: list[Production]) -> list[StructureTemplate]:
    groups = OrderedDict()
    for p in prods:
        key = _structure_key(p)
        groups.setdefault(key, []).append(p)
    return [StructureTemplate(key=k, productions=ps, needs_parameter=len(ps) > 1)
            for k, ps in groups.items()]
```

### State Management

**Extra fields** (beyond what DerivationGame tracks):
- `_phase: str` — "structure" or "parameter"
- `_pending_template: StructureTemplate | None`
- `_structure_templates: list[StructureTemplate]`
- `_parameter_productions: list[Production]` (when in parameter phase)

**stash_state / unstash_state**: include all 4 extra fields in the tuple.

**hashable_obs**: `f"{deriv_state.pretty()}|{phase}|{pending_key or ''}"`

**clone()**: copy all extra fields (lightweight — templates are shared frozen references).

### Grammar Pruning Integration

Both `n_actions` and `one_hot_groups` work naturally:
1. Call `DerivationState.legal_productions(n_sites, mode, ..., n_actions, one_hot_groups)` → pruned list
2. Group pruned productions by structure template via `_group_by_structure()`
3. Templates fully pruned (empty production list) are automatically removed
4. Parameter phase shows only surviving parameters for the chosen template

### Doors Config (`derivation_config.py`)

Add `DoorsFactoredDerivationConfig`:
- Inherits from `DoorsDerivationConfig`
- Overrides `build()`:
  - Imports `FactoredDerivationGame` instead of `DerivationGame`
  - Passes `extra_features=2` to `DerivationPolicyValueNet`
  - Computes `action_size` via `compute_max_factored_actions()` instead of `compute_max_productions()`
- Sets `plot_path="doors_factored_training_metrics.png"`

### Script Changes (`run_doors_derivation.py`)

- Import `DoorsFactoredDerivationConfig`
- Add mode `"doors_factored"` to `select_mode()` menu
- Import and use `compute_max_factored_actions` in `print_banner()` when mode is factored
- Route to `DoorsFactoredDerivationConfig` in `main()`

---

## Verification

1. **Reachability equivalence**: Run N random episodes in `DerivationGame`, record each `Production`. Replay in `FactoredDerivationGame` by mapping each production to (structure_idx, param_idx). Assert identical `program.pretty()`.

2. **Branching sanity**: For D=2 (budget=18), assert root structure count ≤ 20. For D=10 (budget=102), assert ≤ 120.

3. **Action space invariant**: `game.action_space.n` is constant across all steps and both phases.

4. **Phase round-trip**: Stash in parameter phase, unstash, verify phase + pending template preserved.

5. **Clone preserves phase**: Clone in parameter phase, verify clone has same phase and pending template.

6. **No-parameter immediate apply**: Choose a Not/And structure → verify game stays in structure phase (no parameter phase entered), DerivationState updated.

7. **Pruning integration**: With `n_actions=6, one_hot_groups=[[0,1,2,3]]`, verify parameter choices are restricted and contradictory templates removed.

8. **Regression**: `pytest tests/ -x -q` — all existing tests pass.

9. **End-to-end**: Run `python scripts/run_doors_derivation.py` with `doors_factored` mode for 1 iteration. Verify completion.
