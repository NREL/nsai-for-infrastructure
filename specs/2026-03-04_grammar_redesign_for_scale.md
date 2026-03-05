# Grammar Redesign for Scalable Program Synthesis on Doors D=10

## 1. Problem: The Doors Environment

### 1.1 Domain Description

The Doors environment is a key-and-lock navigation puzzle. An agent must navigate through D sequentially-locked rooms to reach a goal location. Each room (except room 0) is locked and requires picking up a specific key to unlock it. Keys form a sequential dependency chain: key 0 unlocks room 1, key 1 unlocks room 2, etc.

### 1.2 D=10 Configuration (locs_per_room=2)

| Parameter | Value | Formula |
|-----------|-------|---------|
| D (rooms) | 10 | given |
| M (locations) | 20 | D × locs_per_room |
| K (keys) | 9 | D - 1 |
| obs_size (n_sites) | 39 | M + 2D - 1 |
| action_space | Discrete(39) | obs_size (indices 30..38 are invalid → NOOP) |
| valid_actions | 30 | M + K + 1 (20 MOVE + 9 PICK + 1 NOOP) |
| horizon | 95 | max(15, 5 × optimal_steps) |
| optimal_steps | 19 | 2K + 1 |
| optimal_reward | +1.71 | 1.0 + 9×0.1 - 19×0.01 |
| optimal_nodes | 68 | 7 × (D-1) + 5 |
| budget | 102 | int(optimal_nodes × 1.5), rounded to even |

### 1.3 State Vector Layout

The observation is a 39-dimensional binary vector:

```
Index  0..19:  at_loc[0..19]     -- one-hot agent position (M=20 bits)
Index 20..29:  unlocked[0..9]    -- room lock status (D=10 bits)
Index 30..38:  key_avail[0..8]   -- key availability (K=9 bits)
```

Room 0 starts unlocked (`unlocked[0] = 1`). All keys start available (`key_avail[k] = 1` for k in [0,8]).

### 1.4 Environment Actions

| Action | Index Range | Precondition | Effect |
|--------|-------------|--------------|--------|
| MOVE_TO(l) | 0..19 | Room containing loc l is unlocked | Agent teleports to location l |
| PICK(k) | 20..28 | Agent at key_loc[k] AND key_avail[k] | key_avail[k]=0, unlocked[key_unlocks[k]]=1 |
| NOOP | 29 | Always valid | No effect |
| (invalid) | 30..38 | Never valid | Treated as NOOP |

Failed preconditions result in a silent no-op (no state change, step penalty still applies).

### 1.5 Reward Structure

- **Step penalty**: -0.01 per step (every step, regardless of action outcome)
- **Unlock bonus**: +0.1 when a key successfully unlocks a room
- **Goal bonus**: +1.0 when agent reaches goal location
- **Optimal reward**: +1.71 = 1.0 + 9×0.1 - 19×0.01

### 1.6 Layout for D=10, locs_per_room=2

```
loc_room:     [0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9]
              (loc 0-1 in room 0, loc 2-3 in room 1, ..., loc 18-19 in room 9)

key_loc:      [1, 3, 5, 7, 9, 11, 13, 15, 17]
              (key k at location k * locs_per_room + 1, i.e., 2nd loc of room k)
              Formula: DoorsGameConfig.__post_init__ in instances/doors/dsl/doors_config.py

key_unlocks:  [1, 2, 3, 4, 5, 6, 7, 8, 9]
              (key k unlocks room k+1)

start_loc:    0   (loc 0 in room 0)
goal_loc:     19  (loc 19 in room 9, last location)
```

### 1.7 Optimal Action Sequence

```
MOVE_TO(1) -> PICK(0) -> MOVE_TO(3) -> PICK(1) -> MOVE_TO(5) -> PICK(2) ->
MOVE_TO(7) -> PICK(3) -> MOVE_TO(9) -> PICK(4) -> MOVE_TO(11) -> PICK(5) ->
MOVE_TO(13) -> PICK(6) -> MOVE_TO(15) -> PICK(7) -> MOVE_TO(17) -> PICK(8) ->
MOVE_TO(19)
```

19 steps total: 9 MOVE_TO (to key locations at odd indices) + 9 PICK + 1 final MOVE_TO (to goal).

---

## 2. Current Grammar Architecture

### 2.1 DSL (Decision-List Language)

The DSL produces **reactive policies** -- deterministic functions from observation to action index. Programs are decision lists evaluated top-to-bottom with first-match semantics.

#### AST Node Types

```
Program  ::= Ite(cond: Condition, action: Flip, else_prog: Program)
           | Default(action: Flip)

Condition ::= IsZero(index: int)     -- true iff state[index] == 0
            | Not(child: Condition)   -- logical negation
            | And(left: Condition, right: Condition)  -- conjunction

Flip(index: int)  -- returns action index (MOVE_TO, PICK, or NOOP)
```

Every node is a frozen dataclass. `node_count()` returns the total AST node count (each constructor = 1 node).

Source: `src/alphazeropp/synthesis/ast_nodes.py`

#### Interpreter Semantics

```python
def eval_condition(cond, state):
    if isinstance(cond, IsZero):  return state[cond.index] == 0
    if isinstance(cond, Not):     return not eval_condition(cond.child, state)
    if isinstance(cond, And):     return eval_condition(cond.left, state) and eval_condition(cond.right, state)

def eval_program(program, state):
    if isinstance(program, Default): return program.action.index
    if isinstance(program, Ite):
        if eval_condition(program.cond, state): return program.action.index
        return eval_program(program.else_prog, state)
```

Source: `src/alphazeropp/synthesis/interpreter.py`

#### Why And Is Necessary

For PICK(k) to be correct, the agent must be at `key_loc[k]` AND `key_avail[k]` must be true. Without `And`, a condition like `Not(IsZero(1))` (agent at loc 1) would fire PICK(0) even when key 0 is already consumed, causing an infinite no-op loop.

With `And`: `And(Not(IsZero(1)), Not(IsZero(30)))` fires PICK(0) only when the agent is at loc 1 AND key 0 is available. Once the key is consumed, the condition becomes false and the program falls through to the next rule.

### 2.2 Budget Grammar

Programs are synthesized by expanding **holes** in a partial AST. Each hole carries a **budget** -- the maximum number of AST nodes it can produce.

#### Hole Types

```python
ProgramHole(budget: int)    -- expands into a Program subtree
ConditionHole(budget: int, parent_is_not: bool = False)  -- expands into a Condition subtree
```

#### Production Rules

**Program productions** for `ProgramHole(k)`:

```
P(k) -> Default(Flip(j))                              if k >= 2    [n_sites choices for j]
P(k) -> Ite(ConditionHole(i), Flip(j), ProgramHole(k-2-i))
                                                       if k >= 5    [i in [1, k-4], j in [0, n_sites)]
```

**Condition productions** for `ConditionHole(k)`:

```
C(k) -> IsZero(j)                                     if k >= 1    [n_sites choices for j]
C(k) -> Not(ConditionHole(k-1, parent_is_not=True))   if k >= 2, parent is NOT a Not
C(k) -> And(ConditionHole(i), ConditionHole(k-1-i))   if k >= 3    [i in [1, (k-1)//2], canonical]
```

Source: `src/alphazeropp/synthesis/derivation.py` (lines 155-255)

#### Canonicalization

- **Double-negation ban**: `Not(Not(...))` is suppressed (parent_is_not flag)
- **And commutativity**: `And(C(i), C(j))` requires `i <= j` (left budget <= right budget)

#### Budget Mode

- **"exact"**: Program must use EXACTLY budget nodes. Dead-end budgets (3, 4) are pruned.
- **"max"**: Program can use <= budget nodes. Early termination (Default/IsZero) allowed at any budget >= 2/1.

Doors uses **"max" mode** for flexibility.

### 2.3 DerivationGame (MCTS Interface)

The derivation is cast as a single-player game for AlphaZero MCTS:

- **State**: Partial AST with holes, encoded as a fixed-size float array
- **Action**: Index into the list of legal productions for the current leftmost hole
- **Reward**: 0.0 at non-terminal steps; `leaf_evaluator(completed_program)` at terminal
- **Action space**: `Discrete(max_productions)` where `max_productions` is the maximum production count across all possible hole budgets
- **Action mask**: First `len(current_productions)` positions are True, rest False
- **Observation**: Preorder traversal of partial AST encoded as `(node_type_id, parameter)` pairs, padded to `2 × budget` floats

Source: `src/alphazeropp/synthesis/derivation_game.py`

#### Observation Encoding

```
obs[2i]     = node_type_id   (PAD=0, Flip=1, IsZero=2, Not=3, And=4, Ite=5, Default=6, ProgramHole=7, ConditionHole=8)
obs[2i + 1] = parameter      (Flip/IsZero: index, Holes: budget, others: 0.0)
```

Total observation size: `2 × budget` floats.

### 2.4 LeafEvaluator

When a derivation reaches a terminal state (complete program), the `LeafEvaluator`:

1. Runs `run_policy_episode(env, program)` on all frozen initial states
2. Collects: `solve_rate`, `avg_reward`, `avg_steps`, `avg_ops`
3. Computes scalar value using chosen metric:
   - `"weighted"` (default for Doors): `alpha × solve_rate + (1 - alpha) × avg_reward` (alpha=0.7)
4. Caches results by `program.pretty()` string key

Source: `src/alphazeropp/synthesis/leaf_evaluator.py`

Tracked statistics: `eval_count`, `cache_hits`, `unique_programs`, `total_env_steps`, `total_interp_ops`.

---

## 3. The Scaling Problem

### 3.1 Production Count Explosion

The number of productions at any derivation step is `O(structural_choices × n_sites)` because every structural template is replicated for each possible `Flip(j)` or `IsZero(j)` index:

```python
# From _program_productions() in synthesis/derivation.py:
for i in range(1, budget - 3):       # structural: condition budget allocation
    for j in range(n_sites):          # parameter: which Flip action
        Ite(ConditionHole(i), Flip(j), ProgramHole(k-2-i))
```

For `ProgramHole(k)` in max mode, the production count is:
- Default: `n_sites` (one per Flip index)
- Ite: `(k - 4) × n_sites` (one per condition budget × Flip index)
- Total: `n_sites + (k - 4) × n_sites = (k - 3) × n_sites`

The maximum occurs at the root hole (`k = budget`), giving `(budget - 3) × n_sites`.

#### Production Counts by Configuration

| Config | n_sites | Budget | Max Prods/Step | Formula | MCTS 200 Coverage |
|--------|---------|--------|----------------|---------|-------------------|
| D=2, lpr=2 | 7 | 18 | ~105 | (18-3)×7 | ~190% |
| D=3, lpr=2 | 11 | 28 | ~275 | (28-3)×11 | ~73% |
| D=10, lpr=2 | 39 | 102 | **~3,861** | (102-3)×39 | **~5%** |
| D=10, lpr=3 | 49 | 102 | **~4,851** | (102-3)×49 | **~4%** |

At D=10, MCTS with 200 simulations covers only ~5% of available productions per step. The neural policy prior cannot meaningfully guide search in a 3,800+ action space without orders-of-magnitude more training data.

### 3.2 Root Cause Analysis

The explosion comes from **coupling** structural and parametric choices:

```
Production = (structure_template, parameter_value)
```

Each structural template (e.g., "Ite with condition budget 3") is replicated `n_sites` times for each possible `Flip(j)`. The action space is the Cartesian product:

```
|actions| = sum_{templates} |parameters_per_template|
          ~ |structural_templates| × n_sites
```

For D=10 with n_sites=39: the root ProgramHole(102) has 99 structural templates (1 Default + 98 Ite variants), each replicated 39 times → 3,861 actions.

### 3.3 Secondary Issues

1. **Derivation depth**: Budget=102 means ~50 derivation steps. Combined with branching, the total search tree is enormous.
2. **Sparse reward**: Only complete programs get evaluated. Intermediate derivation steps give reward=0. The value head must learn from very delayed signal.
3. **Observation size**: `2 × budget = 204` floats for the AST encoding. Larger input → harder to learn.

---

## 4. Proposed Solution: Two-Stage Factorized Grammar

### 4.1 Core Idea

Split each derivation step into two sequential sub-actions:

```
Current:  action = (structure + parameter)     -- one step, |A| = O(S × n_sites)
Proposed: action_1 = structure                  -- step A, |A1| = O(S)
          action_2 = parameter                  -- step B, |A2| = O(n_sites)
```

This reduces branching from multiplicative to additive: `O(S × n_sites)` → `O(S) + O(n_sites)`.

### 4.2 Concrete Factorization

#### For ProgramHole(k):

**Step A -- Choose structure:**

| Structure ID | Template | Choices |
|-------------|----------|---------|
| 0 | `Default(Flip(?))` | 1 (terminate) |
| 1..k-4 | `Ite(C(i), Flip(?), P(k-2-i))` for each valid i | up to k-4 |

Total structure choices: ~k-3 (bounded by budget). For budget=102: 99 structures.

**Step B -- Choose parameter (Flip index):**

Given the chosen structure, select `j in [0, n_sites)` for the `Flip(j)` action.

Total parameter choices: n_sites (= 39 for D=10, lpr=2).

#### For ConditionHole(k):

**Step A -- Choose structure:**

| Structure ID | Template | Choices |
|-------------|----------|---------|
| 0 | `IsZero(?)` | 1 (terminate) |
| 1 | `Not(C(k-1))` | 0 or 1 (suppressed if parent is Not) |
| 2..h | `And(C(i), C(k-1-i))` for each valid i | up to (k-1)//2 |

Total structure choices: ~k/2 + 2.

**Step B -- Choose parameter (IsZero index):**

Only applies if structure = `IsZero(?)`. Select `j in [0, n_sites)`.

For `Not` and `And` structures, step B is skipped (no parameter to choose -- they create new holes).

### 4.3 Action Space Analysis

| Config | Current Max Actions | Factorized Max (Step A) | Factorized Max (Step B) | Effective Max |
|--------|-------------------|------------------------|------------------------|---------------|
| D=2, budget=18 | ~105 | ~15 | 7 | max(15, 7) = 15 |
| D=3, budget=28 | ~275 | ~25 | 11 | max(25, 11) = 25 |
| D=10, budget=102 | ~3,861 | ~99 | 39 | max(99, 39) = 99 |

MCTS with 200 sims on a 99-action space gives ~2× coverage per step -- comparable to D=2's current setup.

### 4.4 Implementation Sketch

```python
class FactorizedDerivationGame(Game):
    """Two-phase derivation: choose structure, then choose parameter."""

    def __init__(self, budget, n_sites, leaf_evaluator, ...):
        self._inner_state = DerivationState.initial(budget)
        self._phase = "structure"  # alternates: "structure" / "parameter"
        self._chosen_structure = None

        # Single Discrete action space = max(max_structures, n_sites)
        max_structures = self._compute_max_structures()
        self.action_space = spaces.Discrete(max(max_structures, n_sites))

    def step(self, action):
        if self._phase == "structure":
            structures = self._get_legal_structures()
            self._chosen_structure = structures[action]

            if self._structure_needs_parameter(self._chosen_structure):
                self._phase = "parameter"
                return obs, 0.0, False, False, {}  # no reward yet
            else:
                # Not/And structures don't need a parameter -- apply immediately
                production = self._assemble_production(self._chosen_structure, param=None)
                return self._apply_production(production)
        else:
            # Parameter phase
            production = self._assemble_production(self._chosen_structure, param=action)
            result = self._apply_production(production)
            self._phase = "structure"
            self._chosen_structure = None
            return result

    def get_action_mask(self):
        if self._phase == "structure":
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[:len(self._get_legal_structures())] = True
            return mask
        else:
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[:self.n_sites] = True  # all site indices valid
            return mask

    def _structure_needs_parameter(self, structure):
        """IsZero and Default(Flip) need a parameter. Not/And don't."""
        return structure.kind in ("IsZero", "Default_Flip", "Ite_Flip")
```

### 4.5 Observation Encoding for Factorized Game

The observation must encode:
1. The partial AST (same as current: preorder `(type_id, param)` pairs)
2. The current phase ("structure" or "parameter")
3. If in parameter phase: which structure was chosen

Proposed encoding: Append 2 extra floats to the existing `2 × budget` array:
```
obs[2*budget]     = phase (0.0 = structure, 1.0 = parameter)
obs[2*budget + 1] = chosen_structure_id (0.0 if in structure phase)
```

Total observation size: `2 × budget + 2` floats (= 206 for budget=102).

### 4.6 Derivation Depth Impact

Factorization doubles the number of game steps for parameter-bearing productions:
- Current: ~50 derivation steps for budget=102
- Factorized: ~75-80 game steps (some structures skip parameter phase)

This is acceptable -- the reduced branching factor more than compensates.

### 4.7 Properties

1. **Lossless**: Every program reachable via the current grammar is still reachable. The factorization just decomposes each production into two sub-actions.
2. **Compatible**: Uses the same AST nodes, DerivationState, LeafEvaluator, and interpreter.
3. **Same action space type**: Single `Discrete(N)` -- no Tuple spaces needed. Compatible with existing MCTS, Agent, Trainer infrastructure.

---

## 5. Alternative Approaches (For Reference)

### 5.1 Scan-Style Synthesis

Reuse existing `ScanDerivationGame` pattern: each step picks one site index from remaining pool → builds a fixed-structure flat decision list.

- Action space: `Discrete(n_sites)` = 39 per step
- Depth: n_sites = 39 steps
- **Cannot express `And` conditions** → cannot solve Doors optimally
- Useful only as a negative control

Source: `src/alphazeropp/instances/bitstring/dsl/scan_derivation_game.py`

### 5.2 Fixed Template with Slot Filling

Fix the program structure as a depth-K decision list (one branch per key), only synthesize the condition/action indices per slot.

- Very tractable: 39 + 39 = 78 actions per step, ~18 steps
- **Cannot learn shorter or structurally different programs**
- Could work if conditions are extended to And(IsZero(a), IsZero(b)) per slot

### 5.3 Increased MCTS Budget

Keep current grammar, increase simulations from 200 to 5,000+.

- Computationally expensive: 25× slowdown
- May suffice for ~3,800 action space but leaves no headroom
- Does not address the fundamental structural issue

---

## 6. Key Files in the Codebase

| File | Role |
|------|------|
| `src/alphazeropp/synthesis/ast_nodes.py` | AST node definitions (Flip, IsZero, Not, And, Ite, Default) |
| `src/alphazeropp/synthesis/derivation.py` | DerivationState, Production, production generation |
| `src/alphazeropp/synthesis/derivation_game.py` | DerivationGame (MCTS Game interface), observation encoding |
| `src/alphazeropp/synthesis/budget_grammar.py` | Program/condition counting and enumeration |
| `src/alphazeropp/synthesis/leaf_evaluator.py` | Program evaluation on frozen states, caching, metrics |
| `src/alphazeropp/synthesis/interpreter.py` | eval_condition, eval_program, run_policy_episode |
| `src/alphazeropp/synthesis/derivation_network.py` | Transformer policy-value network for derivation |
| `src/alphazeropp/instances/doors/dsl/doors_config.py` | DoorsGameConfig, compute_doors_derived_params() |
| `src/alphazeropp/instances/doors/dsl/derivation_config.py` | DoorsDerivationConfig (MetaConfig wiring everything) |
| `src/alphazeropp/instances/doors/doors_pddl_lite.py` | DoorsPDDLLiteEnv (Gymnasium environment) |
| `src/alphazeropp/core/game.py` | Game base class (interface for MCTS) |
| `src/alphazeropp/core/mcts.py` | MCTS implementation |
| `scripts/run_doors_derivation.py` | Training script for doors grammar game |
| `scripts/run_doors_direct.py` | Training script for doors direct play |

---

## 7. Optimal Program Structure for D=10

The optimal program for D=10 (locs_per_room=2) is a 9-branch decision list with `And` conditions:

```
if And(Not(IsZero(1)), Not(IsZero(30))):     # at loc 1 AND key 0 available
  Flip(20)                                    # PICK(0)
elif And(Not(IsZero(3)), Not(IsZero(31))):   # at loc 3 AND key 1 available
  Flip(21)                                    # PICK(1)
elif And(Not(IsZero(5)), Not(IsZero(32))):   # at loc 5 AND key 2 available
  Flip(22)                                    # PICK(2)
elif And(Not(IsZero(7)), Not(IsZero(33))):   # at loc 7 AND key 3 available
  Flip(23)                                    # PICK(3)
elif And(Not(IsZero(9)), Not(IsZero(34))):   # at loc 9 AND key 4 available
  Flip(24)                                    # PICK(4)
elif And(Not(IsZero(11)), Not(IsZero(35))):  # at loc 11 AND key 5 available
  Flip(25)                                    # PICK(5)
elif And(Not(IsZero(13)), Not(IsZero(36))):  # at loc 13 AND key 6 available
  Flip(26)                                    # PICK(6)
elif And(Not(IsZero(15)), Not(IsZero(37))):  # at loc 15 AND key 7 available
  Flip(27)                                    # PICK(7)
elif And(Not(IsZero(17)), Not(IsZero(38))):  # at loc 17 AND key 8 available
  Flip(28)                                    # PICK(8)
elif IsZero(19):                              # not at goal
  Flip(19)                                    # MOVE_TO(goal=19)
else:
  Flip(29)                                    # NOOP
```

**Index mapping**:
- `key_loc[k] = k * 2 + 1` → locations [1, 3, 5, 7, 9, 11, 13, 15, 17]
- `key_avail[k] = M + D + k = 30 + k` → indices [30, 31, 32, ..., 38]
- `PICK(k) = M + k = 20 + k` → actions [20, 21, 22, ..., 28]

**Node count**: Each PICK branch = Ite(1) + And(1) + Not(1) + IsZero(1) + Not(1) + IsZero(1) + Flip(1) = 7 nodes.
9 branches × 7 = 63, plus navigation branch (Ite + IsZero + Flip = 3) + default (Default + Flip = 2) = **68 nodes**.

Required budget: >= 68 (current default: 102, with "max" mode for flexibility).

**Navigation gap**: The above program assumes the agent is already at each key location when the PICK rule fires. In practice, the agent also needs MOVE_TO rules to reach each key location. Adding explicit MOVE_TO branches would add ~9 more branches × 3 nodes = 27 nodes, totaling **~95 nodes**. Synthesizing such a large program is genuinely hard -- the agent must discover the interleaved MOVE-PICK pattern.

---

## 8. Design Constraints for Any Grammar Redesign

1. **Must express And conditions** -- PICK preconditions require conjunction
2. **Must be compatible with existing MCTS + AlphaZero infrastructure** -- Game interface with Discrete action space, action masks, stash/unstash
3. **Must use existing LeafEvaluator** -- programs evaluated by running on DoorsPDDLLiteEnv
4. **Must produce valid AST Programs** -- same Ite/Default/Flip/IsZero/Not/And nodes
5. **Action space per step should be <= ~100** -- for MCTS with 200 sims to provide meaningful guidance
6. **Derivation depth should be <= ~100 steps** -- for MCTS tree to not be too deep

---

## 9. Open Questions for Grammar Design

1. **Should navigation logic be hardcoded or synthesized?** For D=10, the MOVE_TO pattern is deterministic (always go to the next key location). Hardcoding it as a macro-action reduces the synthesis task to just ordering PICK conditions.

2. **Can we decompose into sub-programs?** Instead of one monolithic program, synthesize one sub-program per key, then compose them.

3. **Should the grammar be domain-aware?** E.g., restrict Flip indices to valid MOVE/PICK/NOOP actions (not padding), and restrict IsZero indices to semantically meaningful state bits.

4. **Is the budget grammar the right formalism at all?** The decision-list structure (chain of Ite) is fixed. Could a different program representation (e.g., state machines, rule tables, or planning operators) be more natural for sequential key-pickup tasks?

5. **What about curriculum learning?** Start with D=2 (budget=18, n_sites=7), transfer the learned network to D=3, then D=4, etc. Does the grammar generalize across D values?
