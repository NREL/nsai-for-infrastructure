# Grammar Modes Guide: Four Ways to Search for Doors Programs

**Date:** 2026-03-12
**Status:** Reference
**Goal:** Self-contained guide for newcomers explaining the Doors game, the shared program grammar, and the four derivation modes used to synthesize policies.

---

## 1. The Doors Game

The agent must navigate through a sequence of locked rooms to reach a goal location. Each room (except the first) is locked by a door that requires a specific key.

### Layout (D=3 rooms, 2 locations per room)

```
Room 0              Room 1              Room 2
[loc 0] [loc 1]     [loc 2] [loc 3]     [loc 4] [loc 5]
              Key 0 here           Key 1 here         GOAL
         (unlocks Room 1)    (unlocks Room 2)
```

- The agent starts at loc 0. Room 0 is already unlocked.
- Key 0 sits at loc 1. Picking it up unlocks the door to Room 1.
- Key 1 sits at loc 3. Picking it up unlocks the door to Room 2.
- The goal is to reach loc 5 (the last location in the last room).

### Observation Vector

The agent sees a binary vector of size `n_sites = M + 2D - 1` (here: 6 + 5 = 11):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | at loc0 | at loc1 | at loc2 | at loc3 | at loc4 | at loc5 | room0 open | room1 open | room2 open | key0 avail | key1 avail |

Indices 0-5 are one-hot (agent is at exactly one location). Room/key indices are independent binary flags.

**Initial state:** `[1,0,0,0,0,0, 1,0,0, 1,1]` — at loc 0, room 0 open, both keys available.

### Actions

There are `n_actions = M + K + 1 = 9` actions:

| Action index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Meaning | MOVE(0) | MOVE(1) | MOVE(2) | MOVE(3) | MOVE(4) | MOVE(5) | PICK(key0) | PICK(key1) | NOOP |

- `MOVE(loc)` succeeds only if the destination room is unlocked.
- `PICK(key)` succeeds only if the agent is at the key's location AND the key is available.

### Optimal Solution (D=3)

```
1. MOVE to loc 1      (where key 0 is)
2. PICK key 0         (unlocks room 1)
3. MOVE to loc 3      (where key 1 is)
4. PICK key 1         (unlocks room 2)
5. MOVE to loc 5      (goal)
```

5 steps. Generalizes to `2(D-1) + 1` steps for any D.

---

## 2. The Shared Grammar (DSL)

All four modes synthesize programs from the same context-free grammar. A program is an if-then-else decision tree that maps observations to actions.

### AST Node Types

| Node | Meaning | AST cost |
|---|---|---|
| `Flip(j)` | Execute action j. For j < M: MOVE. For j >= M: PICK or NOOP. | 1 node |
| `IsZero(j)` | Test: is `obs[j] == 0`? Returns true/false. | 1 node |
| `Not(cond)` | Negate a condition. | 1 node + child |
| `And(cond1, cond2)` | Conjunction of two conditions. | 1 node + children |
| `Ite(cond, action, else)` | If cond then action else recurse into else-branch. | 1 node + children |
| `Default(action)` | Base case: always execute this action. | 1 node + child |

### Production Rules

```
Program(k) → Default(Flip(j))                          when k = 2  (base case)
           → Ite(Cond(i), Flip(j), Program(k-2-i))     when k >= 5

Cond(k)    → IsZero(j)                                  when k = 1
           → Not(Cond(k-1))                             when k >= 2
           → And(Cond(i), Cond(k-1-i))                  when k >= 3  (modes 0, 2, 3 only)
```

The parameter `k` is the **budget** — the exact number of AST nodes the subtree must use. The total budget `L` is set to ~1.5x the optimal program size, giving headroom for suboptimal solutions.

### Reading Conditions

`IsZero(j)` checks `obs[j] == 0`. Since the observation uses 1 for "true" states:

- `IsZero(1)` = "agent is NOT at loc 1" (because obs[1]=0 means absent)
- `Not(IsZero(1))` = "agent IS at loc 1"
- `Not(IsZero(9))` = "key 0 IS available"
- `IsZero(7)` = "room 1 is LOCKED"

This double-negation pattern (`Not(IsZero(...))` = positive test) is inherent to the grammar.

---

## 3. Mode 0: `doors` — Flat Action Space, And Enabled

**Config:** `DoorsDerivationConfig`
**Game class:** `DerivationGame`

This is the baseline. The grammar includes `And(...)`, and every production is presented as a single atomic action.

### How Derivation Works

The system starts with a root hole `ProgramHole(L)` and repeatedly expands the leftmost hole by choosing a production. Each production is one MCTS action.

**Example derivation (budget L=22, the optimal for D=3):**

```
Step 1: ProgramHole(22) → Ite(CondHole(5), Flip(6), ProgramHole(15))
        "If <5-node condition>, then PICK(key0), else <continue>"

Step 2: CondHole(5) → And(CondHole(2), CondHole(2))
        "Condition is: <2 nodes> AND <2 nodes>"

Step 3: CondHole(2) → Not(CondHole(1))

Step 4: CondHole(1) → IsZero(1)
        Combined: Not(IsZero(1)) = "agent IS at loc 1"

Step 5: CondHole(2) → Not(CondHole(1))

Step 6: CondHole(1) → IsZero(9)
        Combined: Not(IsZero(9)) = "key 0 IS available"

... continue expanding ProgramHole(15) ...
```

### Action Space

At each step, MCTS chooses from ALL productions for the current hole. For `ProgramHole(22)`:
- `Default(Flip(0))`, `Default(Flip(1))`, ..., `Default(Flip(8))` — 9 options
- `Ite(Cond(1), Flip(0), P(19))`, `Ite(Cond(1), Flip(1), P(19))`, ... — many options
- `Ite(Cond(2), Flip(0), P(18))`, ...
- ...up to `Ite(Cond(17), Flip(8), P(3))` — each (template, parameter) is a separate action

**Branching factor: ~279 actions** at the root. This is the cross-product of structure templates and parameter choices.

### Complete Optimal Program

```
Ite(And(Not(IsZero(1)), Not(IsZero(9))),      # at loc 1 AND key 0 available?
    Flip(6),                                    #   → PICK key 0
    Ite(IsZero(7),                              # room 1 locked?
        Flip(1),                                #   → MOVE to loc 1
        Ite(And(Not(IsZero(3)), Not(IsZero(10))), # at loc 3 AND key 1 available?
            Flip(7),                              #   → PICK key 1
            Ite(IsZero(8),                        # room 2 locked?
                Flip(3),                          #   → MOVE to loc 3
                Default(Flip(5))))))              #   → MOVE to goal (loc 5)
```

Node count: 7 + 3 + 7 + 3 + 2 = **22 nodes** (2 PickRules + 2 MoveRules + 1 Default).

---

## 4. Mode 1: `doors_no_and` — Flat Action Space, And Disabled

**Config:** `DoorsDerivationConfigNoAnd`
**Game class:** `DerivationGame`
**Only difference from Mode 0:** `allow_and = False`

### What Changes

The `And(Cond(i), Cond(k-1-i))` production is removed. Conditions can only be:
- `IsZero(j)` — a single test
- `Not(Cond(k-1))` — negation of a condition

### Why This Matters

PICK requires checking two things simultaneously: "I'm at the key's location" AND "the key is available." Without And, the program must use nested Ite chains:

```
# WITH And (7 nodes for a pick rule):
Ite(And(Not(IsZero(1)), Not(IsZero(9))),   # at loc 1 AND key 0 available
    Flip(6),                                # → PICK key 0
    ...)

# WITHOUT And (9+ nodes, deeper nesting):
Ite(Not(IsZero(1)),                         # at loc 1?
    Ite(Not(IsZero(9)),                     #   key 0 available?
        Flip(6),                            #     → PICK key 0
        ...),                               #     else: key not here
    ...)                                    # else: not at loc 1
```

**Consequences:**
- Programs are deeper and use more budget for the same logic
- MCTS must discover the correct nesting pattern (harder to find)
- The optimal program may not fit within the budget at all for larger D

### Purpose

This mode exists as an **ablation baseline** — it measures how much the `And` connective contributes to learning. If mode 0 solves D=3 but mode 1 doesn't, that demonstrates the expressivity gap.

---

## 5. Mode 2: `doors_factored` — Factored Action Space, And Enabled

**Config:** `DoorsFactoredDerivationConfig`
**Game class:** `FactoredDerivationGame`

The grammar and reachable programs are **identical to Mode 0**. The only change is how productions are presented to MCTS.

### The Problem with Flat Actions

In Mode 0, each production combines a structural template with a parameter:

```
Ite(Cond(3), Flip(0), P(17))    ← one action
Ite(Cond(3), Flip(1), P(17))    ← different action
Ite(Cond(3), Flip(2), P(17))    ← different action
...
```

With 18 templates and 9 Flip indices, that's 18 x 9 = 162 actions just for Ite productions. MCTS must spread its simulations across all of them.

### The Factored Solution

Split each derivation step into two phases:

**Phase 1 — Structure:** Choose the template shape.
```
[0] Default(Flip(?))
[1] Ite(Cond(1), Flip(?), P(19))
[2] Ite(Cond(2), Flip(?), P(18))
...
[17] Ite(Cond(17), Flip(?), P(3))
→ 18 choices (not 162)
```

**Phase 2 — Parameter:** Choose the concrete index.
```
[0] Flip(0)    MOVE(loc 0)
[1] Flip(1)    MOVE(loc 1)
...
[8] Flip(8)    NOOP
→ 9 choices
```

**Some productions skip Phase 2** because they have no parameter:
- `Not(Cond(k-1))` — applied immediately (structural only)
- `And(Cond(i), Cond(k-1-i))` — applied immediately

### Example Derivation (Factored)

```
Step 1a (structure):  ProgramHole(22) → choose template "Ite(Cond(5), Flip(?), P(15))"
Step 1b (parameter):  choose Flip(6)  → result: Ite(CondHole(5), Flip(6), ProgramHole(15))

Step 2a (structure):  CondHole(5) → choose template "And(Cond(2), Cond(2))"
Step 2b:              (skipped — And has no parameter, applied immediately)

Step 3a (structure):  CondHole(2) → choose template "Not(Cond(1))"
Step 3b:              (skipped — Not has no parameter)

Step 4a (structure):  CondHole(1) → choose template "IsZero(?)"
Step 4b (parameter):  choose index 1  → result: IsZero(1)
...
```

### Branching Factor

```
Flat (Mode 0):     ~279 actions per step  (templates × parameters)
Factored (Mode 2): ~31 actions per step   (max of templates, parameters)
                    ≈ 9× reduction
```

**This is lossless** — the exact same set of complete programs is reachable. Only the derivation path changes: one flat step becomes 1-2 factored steps.

### Network Changes

The network receives 2 extra observation features:
- `phase_id`: 0.0 (structure) or 1.0 (parameter)
- `pending_template_id`: which template was chosen (0 if in structure phase)

This lets the network output different policy distributions for each phase.

---

## 6. Mode 3: `doors_d10_macro` — Factored + Macros + Condition Cap

**Config:** `DoorsFactoredD10MacroConfig`
**Game class:** `FactoredDerivationGame` (with macro productions)

Builds on Mode 2 with two domain-specific optimizations that dramatically reduce search depth.

### 6a. Macro Productions

Two macro templates encode common Doors sub-patterns as single actions:

#### PickRule(k) — "If at key k's location AND key k is available, pick it up"

Expands to 7 AST nodes in one derivation step:

```
PickRule(k) →
    Ite(And(Not(IsZero(key_loc[k])),       # am I at the key's location?
            Not(IsZero(M + D + k))),        # is the key still available?
        Flip(M + k),                        # → PICK action for key k
        ProgramHole(budget - 7))            # else: continue
```

**PickRule(0) for D=3:**
```
Ite(And(Not(IsZero(1)), Not(IsZero(9))),   # at loc 1 AND key 0 available
    Flip(6),                                # → PICK key 0
    ProgramHole(budget - 7))
```

**PickRule(1) for D=3:**
```
Ite(And(Not(IsZero(3)), Not(IsZero(10))),  # at loc 3 AND key 1 available
    Flip(7),                                # → PICK key 1
    ProgramHole(budget - 7))
```

#### MoveRule(k) — "If the room this key unlocks is locked, move toward the key"

Expands to 3 AST nodes in one derivation step:

```
MoveRule(k) →
    Ite(IsZero(M + key_unlocks[k]),        # is the target room locked?
        Flip(key_loc[k]),                   # → MOVE to the key's location
        ProgramHole(budget - 3))            # else: room already open, continue
```

**MoveRule(0) for D=3:**
```
Ite(IsZero(7),                             # room 1 locked?
    Flip(1),                                # → MOVE to loc 1
    ProgramHole(budget - 3))
```

**MoveRule(1) for D=3:**
```
Ite(IsZero(8),                             # room 2 locked?
    Flip(3),                                # → MOVE to loc 3
    ProgramHole(budget - 3))
```

### How Macros Appear in the Factored Game

In the structure phase, macros show up as additional template choices:

```
Structure templates at ProgramHole(22):
[0]  Default(Flip(?))
[1]  Ite(Cond(1), Flip(?), P(19))
...
[12] PickRule(?)              ← macro template
[13] MoveRule(?)              ← macro template
```

In the parameter phase after choosing PickRule:
```
[0] key 0    → expands full 7-node PickRule(0) subtree
[1] key 1    → expands full 7-node PickRule(1) subtree
```

### 6b. Condition Budget Cap

Limits the condition budget `i` in `Ite(Cond(i), ...)` to a maximum of 12.

```
Without cap: Ite(Cond(1),...), Ite(Cond(2),...), ..., Ite(Cond(48),...)  → 48 templates
With cap=12: Ite(Cond(1),...), Ite(Cond(2),...), ..., Ite(Cond(12),...)  → 12 templates
```

Useful Doors conditions are small (1-7 nodes). Conditions with 30+ nodes are never needed and just waste MCTS simulations.

### Combined Impact on Derivation Depth

The optimal D=3 program built from macros:

```
Step 1: structure → PickRule(?)     parameter → key 0     (7 nodes placed at once)
Step 2: structure → MoveRule(?)     parameter → key 0     (3 nodes placed at once)
Step 3: structure → PickRule(?)     parameter → key 1     (7 nodes)
Step 4: structure → MoveRule(?)     parameter → key 1     (3 nodes)
Step 5: structure → Default(Flip(?)) parameter → Flip(5)  (2 nodes)
```

**5 structure + 5 parameter = 10 total MCTS actions** to produce the complete 22-node program.

Without macros (Mode 2), the same program requires ~15 factored steps. Without factoring (Mode 0), ~15 flat steps but each with 9x more actions to choose from.

For **D=10** (the target scale), the difference is dramatic:
- Mode 0 (flat): ~119 steps × ~2970 branching
- Mode 2 (factored): ~119 steps × ~31 branching
- Mode 3 (factored+macros): ~38 steps × ~39 branching

### Trade-off

Macros are **not lossless** — they bias the search toward domain-specific patterns. A program that doesn't follow the PickRule/MoveRule structure must still be built from primitive productions, which remain available. The macros add options; they don't remove any.

---

## 7. Summary

| | Mode 0: `doors` | Mode 1: `doors_no_and` | Mode 2: `doors_factored` | Mode 3: `doors_d10_macro` |
|---|---|---|---|---|
| **And enabled** | Yes | **No** | Yes | Yes |
| **Game class** | DerivationGame | DerivationGame | FactoredDerivationGame | FactoredDerivationGame |
| **Action space** | Flat (S x P) | Flat (S x P) | Factored (max(S, P)) | Factored + macros |
| **Branching (D=3)** | ~279 | ~200 | ~31 | ~39 |
| **Derivation depth (D=3)** | ~15 | ~15 | ~15 | ~10 |
| **Derivation depth (D=10)** | ~119 | ~119 | ~119 | ~38 |
| **Lossless vs Mode 0** | -- | No (fewer programs) | Yes (same programs) | Yes (superset of actions) |
| **Purpose** | Baseline | Ablation: measure And's value | Reduce branching | Reduce depth + branching |

### Progression

```
Mode 0 (flat + And)
  │
  ├── Mode 1: remove And ──────────► measures expressivity gap
  │
  └── Mode 2: factor action space ─► same programs, 9× less branching
        │
        └── Mode 3: add macros ────► 3× less depth, domain-aware search
```

### When to Use Each

- **Mode 0:** Small problems (D=2-3), debugging, baseline comparison.
- **Mode 1:** Ablation experiments measuring And's contribution.
- **Mode 2:** Medium problems (D=3-5) where branching is the bottleneck.
- **Mode 3:** Large problems (D=5-10+) where both depth and branching matter. This is the recommended mode for production runs.

---

## Key Source Files

| File | Contents |
|---|---|
| `src/alphazeropp/instances/doors/dsl/derivation_config.py` | All four config classes |
| `src/alphazeropp/synthesis/budget_grammar.py` | Grammar production rules, program counting |
| `src/alphazeropp/synthesis/derivation.py` | Production generation, DerivationState |
| `src/alphazeropp/synthesis/derivation_game.py` | DerivationGame (flat action space) |
| `src/alphazeropp/synthesis/factored_derivation_game.py` | FactoredDerivationGame (structure/parameter split) |
| `src/alphazeropp/instances/doors/dsl/doors_macros.py` | PickRule and MoveRule macro definitions |
| `src/alphazeropp/instances/doors/dsl/doors_config.py` | DoorsGameConfig, `compute_doors_derived_params()` |
| `scripts/run_doors_derivation.py` | Entry point with mode selection |
