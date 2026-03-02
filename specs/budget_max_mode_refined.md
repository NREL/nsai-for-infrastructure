# Budget-as-Maximum Mode: Implementation Spec

## 1. Problem

The derivation grammar has a **dead-end gap** at program budgets 3 and 4:

- `Default(Flip(j))` requires exactly `budget == 2`.
- `Ite(C(i), Flip(j), P(m))` requires `budget >= 5` (cost: 1 + i + 1 + m, minimum 1+1+1+2 = 5).
- Therefore `P(3)` and `P(4)` have zero legal productions.

Any derivation that creates `ProgramHole(3)` or `ProgramHole(4)` hits a dead-end: `truncated=True`, `reward=0.0`. The current exact-budget guard (`count_programs(n, else_budget) == 0`) prevents creating these holes at the *top level*, but the rigidity forces all programs to use exactly `L` AST nodes, wasting budget on syntactic padding (unnecessary `Not`/`And` in conditions).


## 2. Solution

Add an opt-in `program_budget_mode`:

- `"exact"` (default) -- current behavior, programs use exactly `L` nodes.
- `"max"` -- `ProgramHole(k)` can terminate early with `Default(Flip(j))` for any `k >= 2`, yielding programs with `node_count <= L`.

This eliminates dead-end budgets and lets the network learn program complexity organically through reward signal rather than grammar constraint.


## 3. Current vs Desired Behavior

### 3.1 Current: exact mode dead-end (`N=3, L=7`)

```
1. Start:   ProgramHole(7)
2. Expand:  P(7) -> Ite(C(1), Flip(0), P(4))
3. Fill:    C(1) -> IsZero(0)
4. Stuck:   P(4) has zero productions  -->  dead-end, reward=0.0
```

Why: `P(7) -> Ite(C(1), Flip(0), P(4))` is currently BLOCKED by the guard
`count_programs(3, 4) == 0`. But even if it weren't blocked, P(4) itself
would be a dead-end. The guard prevents reaching P(4), but at the cost of
eliminating all Ite expansions that would create P(3) or P(4) children --
reducing the diversity of reachable programs.

### 3.2 Desired: max mode early termination (`N=3, L=7`)

```
1. Start:   ProgramHole(7)
2. Expand:  P(7) -> Ite(C(1), Flip(0), P(4))       <-- now legal
3. Fill:    C(1) -> IsZero(0)
4. Term:    P(4) -> Default(Flip(1))                 <-- early terminate
5. Result:  Ite(IsZero(0), Flip(0), Default(Flip(1)))
6. Nodes:   5  (satisfies 5 <= 7)  ✓
```

### 3.3 Immediate termination sanity case

At `ProgramHole(14)`, max mode allows `Default(Flip(j))` immediately:

- Final program: `Default(Flip(j))`, 2 nodes, `2 <= 14`. Valid.
- Quality is governed by leaf reward, not grammar constraint.

### 3.4 Reachable programs: ALL programs of every valid size

Max mode reaches **every program** of every structurally valid size, not just
some. Sizes 3 and 4 don't exist in the DSL (Default = 2 nodes, smallest Ite =
5 nodes). All other sizes up to L are fully covered.

**Proof (induction).** Claim: from `ProgramHole(k)` in max mode, every program
of size s in {2} ∪ {5, ..., k} is reachable.

- **Base (s=2):** `P(k) -> Default(Flip(j))` for any j. All n_sites programs covered. ✓
- **Inductive (s >= 5):** every s-node program is `Ite(C(c), Flip(j), Q)` where
  Q has size `s-2-c`, `c >= 1`, `s-2-c >= 2`. From P(k), choose
  `Ite(C(c), Flip(j), P(k-2-c))`. Since `k >= s`, we have `k-2-c >= s-2-c`.
  By induction, `P(k-2-c)` can reach Q (size `s-2-c` is in {2} ∪ {5, ..., k-2-c}).
  Condition `C(c)` can reach any c-node condition (no condition dead-ends in max
  mode; see §4.1). ✓

**Example (L=7, N=3):** reachable final sizes are {2, 5, 6, 7}:

| Derivation path | Cond | Else | Total |
|---|---|---|---|
| `P(7) -> Default(Flip(j))` (terminate) | -- | -- | **2** |
| `P(7) -> Ite(C(1), Flip, P(4))`, P(4) terminates | 1 | 2 | **5** |
| `P(7) -> Ite(C(2), Flip, P(3))`, P(3) terminates | 2 | 2 | **6** |
| `P(7) -> Ite(C(3), Flip, P(2))` | 3 | 2 | **7** |

All 6-node programs have form `Ite(Not(IsZero(a)), Flip(j), Default(Flip(k)))`.
Every combination of a, j, k is reachable via the C(2)/P(3) path above. ✓


## 4. Design

### 4.1 Production rules by mode

**Programs** -- `ProgramHole(k)`:

| Production | Exact mode | Max mode |
|---|---|---|
| `P(k) -> Default(Flip(j))` | `k == 2` only | `k >= 2` |
| `P(k) -> Ite(C(i), Flip(j), P(k-2-i))` | `k >= 5`, skip if `count_programs(n, k-2-i) == 0` | `k >= 5`, no skip needed (see §4.2) |

**Conditions** -- `ConditionHole(k)`:

| Production | Exact mode | Max mode |
|---|---|---|
| `C(k) -> IsZero(j)` | `k == 1` only | `k >= 1` (early terminate) |
| `C(k) -> Not(C(k-1))` | `k >= 2`, skip if `_ccnn(n, k-1) == 0` | `k >= 2`, `_ccnn` guard removed (see §4.2) |
| `C(k) -> And(C(i), C(k-1-i))` | `k >= 3`, `i <= (k-1)//2` | unchanged |

In max mode, every hole type has at least one production:
- `P(k)` for k >= 2: `Default(Flip(j))`. ✓
- `C(k, parent_is_not=False)` for k >= 1: `IsZero(j)`. ✓
- `C(k, parent_is_not=True)` for k >= 1: `IsZero(j)` (non-Not-prefixed, valid under double-negation ban). ✓

**Dead-ends are completely eliminated in max mode.**

### 4.2 Dead-end guard simplification

**Program guard** in `_program_productions`:
```python
if count_programs(n_sites, else_budget) == 0:
    continue  # Skip P(3), P(4)
```

In max mode, this guard is **unnecessary**. Proof: for `k >= 5` and `i in range(1, k-3)`:
- `else_budget = k - 2 - i`
- Minimum: `k - 2 - (k-4) = 2` (when i = k-4)
- Maximum: `k - 2 - 1 = k-3` (when i = 1)

So `else_budget >= 2` always, and any `P(m)` with `m >= 2` can terminate via `Default(Flip(j))`. The guard is provably a no-op.

**Condition guard** in `_condition_productions`:
```python
if budget >= 2 and not parent_is_not and _ccnn(n_sites, budget - 1) > 0:
    # produce Not(C(budget-1))
```

The `_ccnn` guard prevents `Not(C(k-1))` when `C(k-1, parent_is_not=True)` has
zero non-Not-prefixed completions. Currently `_ccnn(n, 2) = 0`, so `C(3) ->
Not(C(2))` is blocked (because `C(2, parent_is_not=True)` is dead in exact mode).

In max mode, `C(k, parent_is_not=True)` can always early-terminate to
`IsZero(j)` (a non-Not-prefixed condition). So the `_ccnn` guard is unnecessary.

Implementation: in max mode, bypass both guards. Add comments explaining the invariants.

### 4.3 Production ordering

For deterministic action indexing, both production functions order terminate-first:

**Programs:**
1. `Default(Flip(0)), ..., Default(Flip(n_sites-1))` -- indices `[0, n_sites)`.
2. `Ite(C(i), Flip(j), P(m))` in nested order: outer `i`, inner `j` -- indices `[n_sites, ...)`.

**Conditions:**
1. `IsZero(0), ..., IsZero(n_sites-1)` -- terminate first.
2. `Not(C(k-1))` -- if applicable.
3. `And(C(i), C(k-1-i))` -- in order of increasing `i`.

This matches the current code structure (base-case block before compound blocks). The only change is widening the base-case guard from `k == exact_value` to `k >= base`.

### 4.4 Action space impact

Adding terminate productions increases `compute_max_productions`:

| Config (N, L) | Exact | Max | Change |
|---|---|---|---|
| (3, 7) | 3 | 12 | +300% |
| (3, 8) | 6 | 15 | +150% |
| (6, 14) | 48 | 66 | +37.5% |
| (3, 14) | 24 | 33 | +37.5% |

**Formula** for P(k) production count in max mode:
- `k < 2`: 0
- `2 <= k < 5`: `n_sites` (terminate only)
- `k >= 5`: `n_sites + (k-4) * n_sites = (k-3) * n_sites`

The max over all reachable holes is always at `P(budget)`: **(budget - 3) * n_sites** for budget >= 5.

**Justification for spelling this out**: The original spec said action space "may differ" without quantifying. The 37.5% increase for the target config (N=6, L=14) has real implications for MCTS branching factor and checkpoint compatibility. Making it concrete enables informed hyperparameter choices.

### 4.5 Observation encoding

**No change required.** The observation shape `(2 * budget,)` encodes a fixed-capacity preorder traversal. Shorter programs produce more zero-padding, which the Transformer already handles via its padding mask (`type_ids == 0`).

### 4.6 MCTS and training

**No algorithmic changes required.**
- Terminal reward remains leaf-evaluator output.
- Variable episode length is already supported.
- Tree reuse keys on `hashable_obs` (partial AST string), unaffected by mode.
- Action masking is recomputed per step.

### 4.7 Checkpoint compatibility

Exact and max modes have different `action_space.n` (e.g. 48 vs 66). Checkpoints from one mode cannot be loaded into the other.

Requirements:
1. `compute_max_productions(budget, n_sites, mode)` must be mode-aware.
2. Experiment metadata must include `program_budget_mode`.
3. Checkpoint loading will naturally fail on action-size mismatch (policy head dimension).
4. Experiment directory naming includes mode suffix.

### 4.8 Counting and enumeration

`count_programs(n_sites, budget)` in `budget_grammar.py` retains exact-count semantics. No change.

The `print_banner` in `run_derivation.py` currently shows `count_programs(n, L)`. In max mode, annotate the banner output with the mode and note that the reachable space is larger. Defer adding a `count_programs_upto` helper -- it's cosmetic and avoids scope creep.

**Justification**: The original spec proposed an optional `count_programs_upto`. This is unnecessary for the core feature; the banner annotation is sufficient.


## 5. Parameter Threading

The mode parameter flows through five functions across two files:

```
DerivationConfig.build()                      [derivation_config.py:102]
  extracts mode from gk["program_budget_mode"]
  │
  ├─> DerivationGame.__init__(..., program_budget_mode=mode)  [derivation_game.py:116]
  │     stores self._mode
  │     │
  │     └─> compute_max_productions(budget, n_sites, mode)    [derivation_game.py:79]
  │           ├─> _program_productions(k, n_sites, mode)      [derivation.py:155]
  │           └─> _condition_productions(k, n_sites, mode=mode) [derivation.py:186]
  │
  └─> DerivationGame.step() / reset()
        └─> DerivationState.legal_productions(n_sites, mode)  [derivation.py:264]
              ├─> _program_productions(budget, n_sites, mode) [derivation.py:155]
              └─> _condition_productions(budget, n_sites, parent_is_not, mode) [derivation.py:186]
```

All five functions get a new `mode: str = "exact"` parameter with backward-compatible default.


## 6. Risks

### 6.1 Checkpoint incompatibility (from original spec)

Mode-specific action size breaks old checkpoints.

Mitigation: mode metadata in experiment config + natural dimension-mismatch failure on load.

### 6.2 MCTS branching factor increase (NEW)

At P(14) with N=6, max mode has 66 legal actions vs 48. Each MCTS simulation explores less deeply with the same `n_simulations` budget.

Mitigation: monitor search quality in experiments. If needed, scale `n_simulations` proportionally (~37.5% more for N=6 L=14).

**Justification for adding this risk**: The original spec used identical hyperparameters across modes. With 37.5% more actions, fixed simulation count means shallower search. This should be monitored explicitly.

### 6.3 Early-termination bias / policy laziness (NEW)

Terminate actions are always available and produce immediate reward. The network may learn to always terminate early, producing trivial 2-node programs and collapsing exploration.

Mitigation:
1. Monitor average `node_count` of generated programs during training.
2. The leaf evaluator naturally penalizes bad programs (low solve_rate).
3. If collapse is observed: consider minimum-depth constraint (terminate only when `k <= threshold`) or reward shaping.

**Justification for adding this risk**: The original spec didn't address the optimization landscape. Terminate is an "easy out" -- the network can trivially learn to always pick it. Leaf reward should counteract this, but it needs explicit monitoring.

### 6.4 Nondeterministic production ordering (from original spec)

Mitigation: deterministic ordering (terminate-first, §4.3) and index-stability tests.


## 7. File-Level Implementation Plan

### 7.1 `derivation.py` (production generation)

File: `src/alphazeropp/instances/bitstring/dsl/derivation.py`

**`_program_productions(budget, n_sites, mode="exact")`** -- add `mode` parameter:

```python
def _program_productions(budget: int, n_sites: int, mode: str = "exact") -> list[Production]:
    prods: list[Production] = []

    # Terminate: P(k) -> Default(Flip(j))
    # Exact: only at k == 2.  Max: at any k >= 2.
    if (mode == "exact" and budget == 2) or (mode == "max" and budget >= 2):
        for j in range(n_sites):
            prods.append(Production(
                hole_kind="P", hole_budget=budget,
                result=Default(Flip(j)),
                label=f"P({budget}) -> Default(Flip({j}))",
            ))

    # Expand: P(k) -> Ite(C(i), Flip(j), P(k-2-i))  for k >= 5
    if budget >= 5:
        for i in range(1, budget - 3):
            else_budget = budget - 2 - i
            if mode == "exact" and count_programs(n_sites, else_budget) == 0:
                continue
            # Max mode: else_budget >= 2 is guaranteed by loop bounds (see spec §4.2).
            for j in range(n_sites):
                prods.append(Production(
                    hole_kind="P", hole_budget=budget,
                    result=Ite(ConditionHole(i), Flip(j), ProgramHole(else_budget)),
                    label=f"P({budget}) -> Ite(C({i}), Flip({j}), P({else_budget}))",
                ))

    return prods
```

**`_condition_productions(budget, n_sites, parent_is_not, mode="exact")`** -- add `mode` parameter:

```python
def _condition_productions(
    budget: int, n_sites: int, parent_is_not: bool = False, mode: str = "exact",
) -> list[Production]:
    prods: list[Production] = []

    # Terminate: C(k) -> IsZero(j)
    # Exact: only at k == 1.  Max: at any k >= 1.
    if (mode == "exact" and budget == 1) or (mode == "max" and budget >= 1):
        for j in range(n_sites):
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=IsZero(j),
                label=f"C({budget}) -> IsZero({j})",
            ))

    # C(k) -> Not(C(k-1))
    # Exact: guard with _ccnn to prevent dead-end C(k-1, parent_is_not=True).
    # Max: C(k-1, parent_is_not=True) can always early-terminate to IsZero(j).
    if budget >= 2 and not parent_is_not:
        if mode == "max" or _ccnn(n_sites, budget - 1) > 0:
            child_budget = budget - 1
            result = Not(ConditionHole(child_budget, parent_is_not=True))
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=result,
                label=f"C({budget}) -> Not(C({child_budget}))",
            ))

    # C(k) -> And(C(i), C(k-1-i)) — unchanged
    if budget >= 3:
        for i in range(1, (budget - 1) // 2 + 1):
            right_budget = budget - 1 - i
            result = And(ConditionHole(i), ConditionHole(right_budget))
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=result,
                label=f"C({budget}) -> And(C({i}), C({right_budget}))",
            ))

    return prods
```

**`DerivationState.legal_productions(n_sites, mode="exact")`** -- pass `mode` to both production functions:

```python
def legal_productions(self, n_sites: int, mode: str = "exact") -> list[Production]:
    hole = self.leftmost_hole()
    if hole is None:
        return []
    if isinstance(hole, ProgramHole):
        return _program_productions(hole.budget, n_sites, mode)
    elif isinstance(hole, ConditionHole):
        return _condition_productions(hole.budget, n_sites, hole.parent_is_not, mode)
    return []
```

### 7.2 `derivation_game.py` (game + action space)

File: `src/alphazeropp/instances/bitstring/dsl/derivation_game.py`

**`compute_max_productions(budget, n_sites, mode="exact")`** -- add `mode` parameter:

```python
def compute_max_productions(budget: int, n_sites: int, mode: str = "exact") -> int:
    max_prods = 0
    for k in range(2, budget + 1):
        n = len(_program_productions(k, n_sites, mode))
        max_prods = max(max_prods, n)
    for k in range(1, budget + 1):
        n = len(_condition_productions(k, n_sites, mode=mode))  # pass mode
        max_prods = max(max_prods, n)
    return max_prods
```

**`DerivationGame.__init__`** -- accept and store mode:

```python
def __init__(self, budget, n_sites, leaf_evaluator, program_budget_mode="exact"):
    super().__init__()
    self.budget = budget
    self.n_sites = n_sites
    self.leaf_evaluator = leaf_evaluator
    self._mode = program_budget_mode

    self._max_productions = compute_max_productions(budget, n_sites, mode=program_budget_mode)
    self.action_space = spaces.Discrete(self._max_productions)
    # ... rest unchanged
```

**`DerivationGame.reset` and `step`** -- pass mode to `legal_productions`:

```python
def reset(self, **kwargs):
    self._deriv_state = DerivationState.initial(self.budget)
    self._current_productions = self._deriv_state.legal_productions(
        self.n_sites, mode=self._mode
    )
    # ...

def step(self, action):
    prod = self._current_productions[action]
    self._deriv_state = self._deriv_state.apply(prod)
    self._current_productions = self._deriv_state.legal_productions(
        self.n_sites, mode=self._mode
    )
    # ... rest unchanged
```

### 7.3 `derivation_config.py` (config wiring)

File: `src/alphazeropp/instances/bitstring/dsl/derivation_config.py`

Add `"program_budget_mode": "exact"` to `self.game.kwargs` in `__init__`.

In `build()`, extract and pass to game:

```python
mode = gk.get("program_budget_mode", "exact")
game = DerivationGame(budget, n_sites, leaf_eval, program_budget_mode=mode)
```

### 7.4 `run_derivation.py` (UI + experiment naming)

File: `scripts/run_derivation.py`

1. Add `program_budget_mode` to `_build_sections()` in the Problem section, with choices `["exact", "max"]`.
2. Include mode in `setup_experiment_dir` dirname: `..._mode{mode}_...`.
3. Annotate `print_banner` with mode. When `mode == "max"`, note that program sizes may be `<= L`.


## 8. Test Plan

### 8.1 Exact mode backward compatibility

All existing tests pass unchanged. Specifically:
- `test_produces_valid_program`: `node_count() == BUDGET` (exact mode).
- `test_action_space_size`: `action_space.n == 48` (exact mode, L=14 N=6).
- `test_no_truncation_in_game_episodes`: zero dead-ends (exact mode).
- All grammar count tests in `test_cfg_grammar.py`.

### 8.2 Max mode: production generation

**Programs:**
1. `P(3)` has `n_sites` terminate productions (was 0).
2. `P(4)` has `n_sites` terminate productions (was 0).
3. `P(5)` has `2 * n_sites` productions: `n_sites` terminate + `n_sites` Ite (was `n_sites` Ite only).
4. `P(14)` with N=6 has 66 productions (was 48).

**Conditions:**
5. `C(2, parent_is_not=True)` has `n_sites` IsZero productions (was 0 -- dead-end fixed).
6. `C(3, parent_is_not=False)` includes `Not(C(2))` (was blocked by `_ccnn` guard).
7. `C(5)` has `n_sites + 1 + 2` productions: `n_sites` IsZero + 1 Not + 2 And.

### 8.3 Max mode: end-to-end episodes

1. Random max-mode episodes: zero truncations (dead-ends eliminated).
2. `node_count() <= BUDGET` for all completed programs.
3. Regression trace: L=7, N=3 example from §3.2 reaches terminal with node count 5.

### 8.4 Action space consistency

1. `action_space.n == compute_max_productions(budget, n_sites, mode="max")`.
2. For N=6 L=14: `action_space.n == 66`.

### 8.5 Test parameterization strategy

Use `@pytest.mark.parametrize("mode", ["exact", "max"])` on `TestDerivationGameBasics` and `TestActionSpaceAfterFiltering`. Add mode-conditional assertions:
- Exact: `node_count() == BUDGET`, `action_space.n == 48`.
- Max: `node_count() <= BUDGET`, `action_space.n == 66`.

**Justification**: The original spec listed test categories but not the parameterization strategy. Pytest parametrize avoids duplicating test classes while making both modes first-class.


## 9. Experiment Plan (Exact vs Max)

Target: `N=6, L=14, onemax`.

### 9.1 Protocol

1. Both modes with identical hyperparameters (same seeds, same n_frozen_states).
2. At least 3 fixed seeds per mode.
3. Monitor whether max mode needs more `n_simulations` due to larger branching factor (§6.2). If early results show search quality degradation, run a follow-up batch with scaled simulations.

### 9.2 Metrics

1. Best `avg_reward` per iteration.
2. `new_rewards_mean` learning curve.
3. Unique programs explored.
4. Best-program `node_count` distribution (expect max mode to have programs below L).
5. Average `node_count` across all generated programs (monitor for early-termination collapse, §6.3).
6. `Not` and `And` usage from `program_log.jsonl`.

### 9.3 Expected outcomes

1. Max mode reduces syntactic padding (fewer unnecessary `Not`/`And`).
2. Best program sizes diversify below `L`.
3. If `Not`/`And` were mainly budget-filler, their usage decreases in max mode.


## 10. Acceptance Criteria

1. `program_budget_mode = "max"` exists and is opt-in (default `"exact"`).
2. All existing exact-mode tests pass unchanged.
3. Max mode: P(3) and P(4) have terminate productions.
4. Max mode: C(k) for k >= 2 has IsZero terminate productions.
5. Max mode: C(2, parent_is_not=True) is no longer dead (was 0 productions, now n_sites).
6. Max mode: completed programs have `node_count <= L`.
7. Max mode: zero dead-end truncations in random episodes (ALL hole types covered).
8. Max mode: `action_space.n == 66` for N=6 L=14 (conditions don't change the max).
9. L=7 regression trace passes as a test.
10. Experiment directory and banner include mode label.


## 11. Implementation TODO

Ordered by dependency. Each step is independently testable.

- [ ] **Step 1: `_program_productions` mode parameter** (`derivation.py:155`)
  - Add `mode: str = "exact"` parameter.
  - Widen Default guard: `(mode == "exact" and budget == 2) or (mode == "max" and budget >= 2)`.
  - Remove dead-end guard in max mode: `if mode == "exact" and count_programs(...) == 0: continue`.
  - Verify: `len(_program_productions(4, 3, "max")) == 3`, `len(_program_productions(4, 3, "exact")) == 0`.

- [ ] **Step 2: `_condition_productions` mode parameter** (`derivation.py:186`)
  - Add `mode: str = "exact"` parameter.
  - Widen IsZero guard: `(mode == "exact" and budget == 1) or (mode == "max" and budget >= 1)`.
  - Relax `_ccnn` guard for Not: `if mode == "max" or _ccnn(n_sites, budget - 1) > 0`.
  - Verify: `len(_condition_productions(2, 3, True, "max")) == 3` (was 0).
  - Verify: `len(_condition_productions(3, 3, False, "max"))` includes Not(C(2)) (was blocked).

- [ ] **Step 3: `legal_productions` mode parameter** (`derivation.py:264`)
  - Add `mode: str = "exact"` parameter, pass to both `_program_productions` and `_condition_productions`.

- [ ] **Step 4: `compute_max_productions` mode parameter** (`derivation_game.py:79`)
  - Add `mode: str = "exact"` parameter, pass to both production functions.
  - Verify: `compute_max_productions(14, 6, "max") == 66`, `compute_max_productions(14, 6, "exact") == 48`.

- [ ] **Step 5: `DerivationGame` accepts and stores mode** (`derivation_game.py:116`)
  - Add `program_budget_mode="exact"` to `__init__`.
  - Store as `self._mode`.
  - Pass to `compute_max_productions` in `__init__`.
  - Pass to `legal_productions` in `reset()` and `step()`.

- [ ] **Step 6: `DerivationConfig` wiring** (`derivation_config.py:34`)
  - Add `"program_budget_mode": "exact"` to `self.game.kwargs`.
  - Extract in `build()` and pass to `DerivationGame` constructor.

- [ ] **Step 7: `run_derivation.py` UI** (`scripts/run_derivation.py`)
  - Add `program_budget_mode` to interactive config (Problem section, choices `["exact", "max"]`).
  - Include mode in experiment directory name.
  - Annotate banner with mode.

- [ ] **Step 8: Tests**
  - Parametrize `TestDerivationGameBasics` and `TestActionSpaceAfterFiltering` over `["exact", "max"]`.
  - Add mode-conditional assertions for `node_count` and `action_space.n`.
  - Add regression test for L=7 N=3 max-mode trace (§3.2).
  - Add unit tests for `_program_productions` AND `_condition_productions` output counts by mode.
  - Add test: `C(2, parent_is_not=True, mode="max")` has `n_sites` productions (dead-end fixed).
  - Add test: 100 random max-mode episodes, zero truncations (covers both hole types).

- [ ] **Step 9: Experiment run**
  - Run exact vs max with N=6 L=14, 3 seeds each.
  - Collect metrics from §9.2.
  - Monitor for early-termination collapse (avg node_count).


## 12. Changes From Previous Spec Version

| Area | Previous spec | This spec | Why |
|---|---|---|---|
| Condition early-term | "Conditions unchanged in both modes" | C(k) -> IsZero(j) for k >= 2; _ccnn guard relaxed | Fixes C(2, parent_is_not=True) dead-end; makes MCTS exploration more forgiving; eliminates ALL dead-ends |
| Reachability proof | Listed reachable sizes without completeness proof | Inductive proof: ALL programs of sizes {2} ∪ {5..L} reachable | Addresses concern that only "some" programs of each size were reachable |
| Action space math | "may differ by mode" (unquantified) | Exact formula + reference table (48 -> 66 for target config) | Enables informed hyperparameter decisions; exposes 37.5% branching factor increase |
| Dead-end guards | Program guard only | Both program (`count_programs`) and condition (`_ccnn`) guards relaxed | Condition guard was blocking Not(C(2)) via _ccnn(n,2)==0; now unnecessary |
| Parameter threading | Listed 4 wiring points | Full call chain with 5 functions across 2 files | _condition_productions also needs mode parameter |
| Risk: MCTS branching | Not mentioned | Added §6.2 with quantified impact | 37.5% more actions means shallower search with fixed n_simulations |
| Risk: policy laziness | Not mentioned | Added §6.3 with monitoring plan | Terminate is an "easy out"; network may collapse to trivial programs |
| Test strategy | Listed categories | Pytest parametrize + condition-specific dead-end tests | C(2, parent_is_not=True) dead-end fix needs explicit test |
| Concrete code | Not provided | Full code for all 5 modified functions | Implementation-ready; no ambiguity about what changes |
| Implementation TODO | Not present | Ordered 9-step checklist with verification criteria | Each step independently testable; clear execution order |
