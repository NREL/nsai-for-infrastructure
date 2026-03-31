# Masking Reference: Comprehensive Catalog

**Date:** 2026-03-12
**Status:** Reference
**Goal:** Document every masking mechanism in the AlphaZero program synthesis codebase — what is masked, why, where, and with concrete examples.

---

## Overview

Masking appears at five layers of the system. Each layer progressively narrows the search space so MCTS only explores valid, non-redundant programs.

| Layer | Masks | Purpose |
|---|---|---|
| **Grammar** | Budget dead-ends, double-negation ban, action range, condition cap | Prune impossible/wasteful productions at generation time |
| **Semantic** | One-hot group constraints | Prune logically contradictory conditions |
| **Game** | Flat action mask, factored action mask, environment preconditions | Expose valid actions to MCTS |
| **MCTS** | Policy masking, Dirichlet masking, UCB masking, rollout masking | Restrict search to valid actions |
| **Network** | Transformer attention padding mask | Ignore unused AST slots |

---

## Running Example

All examples use **D=3 rooms, 2 locs/room**:

```
Room 0            Room 1            Room 2
[loc 0, loc 1]    [loc 2, loc 3]    [loc 4, loc 5 ← GOAL]
          Key 0 at loc 1             Key 1 at loc 3
          (unlocks Room 1)           (unlocks Room 2)
```

Constants: `M=6` locations, `D=3` rooms, `K=2` keys, `n_sites=11`, `n_actions=9`.

Observation vector (size 11):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | loc0 | loc1 | loc2 | loc3 | loc4 | loc5 | room0 unlocked | room1 unlocked | room2 unlocked | key0 avail | key1 avail |

Action indices: `Flip(0..5)` = MOVE, `Flip(6..7)` = PICK, `Flip(8)` = NOOP.

---

## 1. Grammar-Level Masking

These masks are applied when generating the list of legal productions for a hole. They eliminate productions that can never lead to a valid complete program.

### 1a. Budget Dead-End Pruning

**File:** `src/alphazeropp/synthesis/derivation.py:305-322`

In `exact` budget mode, a production is suppressed if the remaining budget has zero valid completions.

```
P(k) → Ite(C(i), Flip(j), P(k-2-i))

For each i, else_budget = k - 2 - i.
If count_programs(n_sites, else_budget) == 0 → skip this production.
```

**Example:** `P(4)` in exact mode — no grammar rule produces exactly 4 AST nodes, so any `Ite(..., ..., P(4))` is a dead end and never generated.

### 1b. Double-Negation Ban

**File:** `src/alphazeropp/synthesis/derivation.py:372-380`

When expanding a `ConditionHole` whose parent is `Not(...)`, the `Not(C(k-1))` production is suppressed.

```
Not(ConditionHole(3))
  → allowed:  IsZero(j), And(...)
  → blocked:  Not(C(2))          ← would create Not(Not(...)) = identity, wastes 2 nodes
```

### 1c. Action Range Restriction

**File:** `src/alphazeropp/synthesis/derivation.py:300-312`

`Flip(j)` indices are capped at `n_actions` instead of `n_sites`.

```
n_sites = 11  (observation size)
n_actions = 9  (6 moves + 2 picks + 1 noop)

Flip(0)..Flip(8)  → generated (valid actions)
Flip(9), Flip(10) → never generated (observation-only indices, not actions)
```

### 1d. Max Condition Budget Cap (mode 3 only)

**File:** `src/alphazeropp/synthesis/derivation.py:314-319`

Caps the condition budget `i` in `Ite(C(i), ...)` to a configurable maximum (default 12 for doors_d10_macro).

```
Without cap (budget=51): Ite(C(1),...), Ite(C(2),...), ..., Ite(C(48),...)  → 48 templates
With cap=12:             Ite(C(1),...), Ite(C(2),...), ..., Ite(C(12),...)  → 12 templates
```

Conditions beyond 12 nodes are far more complex than any useful Doors predicate requires.

---

## 2. Semantic Masking

### 2a. One-Hot Group Constraints

**File:** `src/alphazeropp/synthesis/derivation.py:246-268, 428-482`

Within an `And(...)` node, `IsZero` productions that contradict existing sibling literals are pruned. Contradiction is checked against mutual-exclusion groups.

**Setup:** `one_hot_groups = [[0, 1, 2, 3, 4, 5]]` — agent location indices are one-hot (agent is at exactly one location).

**Two contradiction rules:**

1. **Direct contradiction** — same index, opposite polarity:
   ```
   Existing sibling: Not(IsZero(1))    → "agent IS at loc 1"
   Candidate:        IsZero(1)         → "agent is NOT at loc 1"
   Result: BLOCKED (logically impossible)
   ```

2. **One-hot contradiction** — two positive literals in the same group:
   ```
   Existing sibling: Not(IsZero(1))    → "agent IS at loc 1"  (positive on index 1)
   Candidate:        Not(IsZero(3))    → "agent IS at loc 3"  (positive on index 3)
   Indices 1 and 3 are both in group [0,1,2,3,4,5]
   Result: BLOCKED (can't be at two locations simultaneously)
   ```

**What passes:**
```
   Existing sibling: Not(IsZero(1))    → "agent IS at loc 1"
   Candidate:        Not(IsZero(9))    → "key 0 IS available"
   Index 9 is not in the location group
   Result: ALLOWED
```

---

## 3. Game-Level Masking

These masks are returned by `get_action_mask()` and consumed by MCTS at every search step.

### 3a. DerivationGame — Flat Action Mask

**File:** `src/alphazeropp/synthesis/derivation_game.py:207-210`

Fixed-size boolean vector. The first N entries are `True` where N = number of legal productions at the current hole.

```python
mask = np.zeros(max_productions, dtype=bool)  # e.g. size 279
mask[:len(current_productions)] = True         # e.g. 40 legal → [T]*40 + [F]*239
```

**Example:** At a `ProgramHole(2)` (terminal), only `Default(Flip(j))` for j in 0..8 → 9 productions. Mask: `[T]*9 + [F]*270`.

### 3b. FactoredDerivationGame — Phase-Dependent Action Mask

**File:** `src/alphazeropp/synthesis/factored_derivation_game.py:320-326`

The mask changes meaning based on the current phase:

```python
if phase == "structure":
    mask[:len(structure_templates)] = True
    # e.g. 15 templates: [Default, Ite(C(1),...), ..., PickRule, MoveRule, ...]
else:  # "parameter"
    mask[:len(parameter_productions)] = True
    # e.g. 9 Flip indices after choosing Default(Flip(?))
```

**Example flow:**
```
Step 1 (structure): mask = [T]*15 + [F]*24    → choose PickRule
Step 2 (parameter): mask = [T]*2  + [F]*37    → choose key=0 or key=1
Step 3 (structure): mask = [T]*12 + [F]*27    → choose MoveRule
...
```

### 3c. DoorsDirectGame — Environment Precondition Mask

**File:** `src/alphazeropp/instances/doors/game.py:50-74`

When evaluating a synthesized program on the actual Doors environment, each action has PDDL preconditions:

```
Agent at loc 0, room 1 locked, keys 0 and 1 available:

MOVE(0): room 0 unlocked → T     PICK(0): at loc 1? NO  → F
MOVE(1): room 0 unlocked → T     PICK(1): at loc 3? NO  → F
MOVE(2): room 1 unlocked → F     NOOP:    always         → T
MOVE(3): room 1 unlocked → F
MOVE(4): room 2 unlocked → F
MOVE(5): room 2 unlocked → F

mask = [T, T, F, F, F, F,  F, F,  T]
```

---

## 4. MCTS Masking

These masks ensure the search tree only expands valid actions at every stage.

### 4a. Policy Masking (post-network)

**File:** `src/alphazeropp/core/mcts.py:382-398`

After the network outputs raw policy logits, invalid actions are zeroed and renormalized:

```
Network output:  [0.3, 0.2, 0.1, 0.05, 0.01, ...]   (279 values)
Action mask:     [T,   T,   T,   ...,   F,    ...]   (40 valid)

After masking:   [0.3, 0.2, 0.1, ...,   0.0,  ...]   (zeros on invalid)
After renorm:    [0.38, 0.25, 0.13, ..., 0.0,  ...]   (sums to 1.0 over valid)
```

Fallback: if all valid actions have zero probability, assigns uniform distribution over valid actions.

### 4b. Dirichlet Noise Masking

**File:** `src/alphazeropp/core/mcts.py:136-141`

Exploration noise is generated for all actions, then masked and renormalized before mixing:

```
raw_noise = Dirichlet([α]*279)         → mass spread over 279 actions
masked    = raw_noise * action_mask    → zero out invalid
masked   /= masked.sum()              → renormalize over 40 valid actions

final_policy = (1-ε) * network_policy + ε * masked_noise
```

Without this, noise mass on invalid actions would dilute exploration of valid ones.

### 4c. UCB Score Masking

**File:** `src/alphazeropp/core/mcts.py:400-447`

Invalid actions are assigned UCB = −∞ so `argmax` never selects them:

```
ucb_scores = np.full(279, -np.inf)
ucb_scores[valid_indices] = Q_normalized + exploration_bonus

best_action = argmax(ucb_scores)   → guaranteed valid
```

### 4d. Rollout Dead-End Masking

**File:** `src/alphazeropp/core/mcts.py:259-301`

During random MCTS rollouts, actions are sampled uniformly from valid ones only:

```python
while not terminal:
    mask = game.get_action_mask()
    valid = np.flatnonzero(mask)    # e.g. [0, 1, 2, 5, 8]
    if len(valid) == 0:
        break                        # dead end — stop rollout
    action = valid[randint(len(valid))]
```

---

## 5. Network Masking

### 5a. Transformer Attention Padding Mask

**File:** `src/alphazeropp/synthesis/derivation_network.py:80-114`

The AST is encoded as a fixed-length preorder traversal (size = budget). Unused slots have `type_id = 0` and are masked in transformer attention:

```
Budget = 51 slots.  Partial AST has 12 nodes filled.

type_ids:          [5, 3, 1, 7, ..., 0, 0, 0, ..., 0]
                    ←── 12 nodes ──→  ←── 39 padding ──→

pad_mask_tokens:   [F, F, F, F, ..., T, T, T, ..., T]
cls_mask:          [F]

key_padding_mask:  [F, F, F, F, ..., F, T, T, T, ..., T]
                    CLS  ←─ 12 nodes ─→  ←─ 39 padded ─→
                    (shape: batch × 52)
```

The CLS token (prepended) is never masked. Padding tokens are ignored by all attention heads.

---

## Mask Flow Diagram

```
Grammar rules
  │
  ├─ Budget dead-end pruning ──────────────── (§1a)
  ├─ Double-negation ban ──────────────────── (§1b)
  ├─ Action range restriction ─────────────── (§1c)
  └─ Condition budget cap ─────────────────── (§1d)
  │
  ▼
Legal productions (per hole)
  │
  └─ One-hot group filtering ──────────────── (§2a)
  │
  ▼
get_action_mask()  ─── flat (§3a) or factored (§3b)
  │
  ├──────────────────────► MCTS policy mask ── (§4a)
  ├──────────────────────► Dirichlet mask ──── (§4b)
  ├──────────────────────► UCB mask ────────── (§4c)
  └──────────────────────► Rollout mask ────── (§4d)
  │
  ▼
Network forward pass
  │
  └─ Attention padding mask ───────────────── (§5a)

Environment execution (leaf eval)
  │
  └─ Precondition mask ────────────────────── (§3c)
```

---

## Key Design Decisions

1. **Masking is layered, not redundant.** Grammar-level masks prune at generation time (never stored). Semantic masks prune at query time. Game masks are recomputed per MCTS step. Each layer catches issues the layer above cannot.

2. **Training labels are implicitly masked.** The network is trained on MCTS policy targets that already have zero mass on invalid actions. No explicit loss mask is needed — CrossEntropyLoss naturally ignores zero-target classes.

3. **Factored masking changes shape, not content.** The factored game (§3b) masks the same productions as the flat game (§3a), but splits them into two smaller masks (structure phase, then parameter phase). The reachable program set is identical.

4. **One-hot masking is context-dependent.** Unlike all other masks which depend only on the current hole's budget and type, one-hot masking (§2a) requires inspecting And-sibling context in the partial AST. This makes it the most expensive mask to compute, but it eliminates entire impossible subtrees.
