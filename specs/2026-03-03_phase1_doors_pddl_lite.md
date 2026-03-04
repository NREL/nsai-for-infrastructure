# Phase 1: PDDL-Faithful DoorsPDDLLiteEnv + Grammar Restriction

**Date:** 2026-03-03
**Status:** Implemented
**Branch:** refactor/code_architecture

## Overview

Phase 1 introduces a PDDL-faithful doors environment (`DoorsPDDLLiteEnv`) based on
the PDDLGym "doors" domain. The environment models rooms gated by locked doors, keys
at specific locations, and movement between locations. The key property: `PICK` actions
require conjunctive preconditions (`And`), creating a measurable expressivity gap
between `allow_and=True` and `allow_and=False` grammar configurations.

## PDDL Domain Reference

From `pddlgym/pddl/doors.pddl`:

- **Types:** location, room, key
- **`moveto(sloc, eloc, eroom)`**: requires `at(sloc) AND unlocked(eroom) AND locinroom(eloc, eroom)`.
  Effect: `not at(sloc), at(eloc)`.
- **`pick(loc, key, room)`**: requires `at(loc) AND keyat(key, loc) AND keyforroom(key, room)`.
  Effect: `not keyat(key, loc), unlocked(room)`.

Our lite version uses fixed-size numpy observations, discrete integer actions, and no
PDDL dependency.

## Environment Design: DoorsPDDLLiteEnv

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rooms` | int | 2 | Number of rooms (D) |
| `loc_room` | list[int] | 2 per room | Room assignment for each location |
| `key_loc` | list[int] | [1, 2, ...] | Location of each key |
| `key_unlocks` | list[int] | [1, 2, ...] | Room each key unlocks |
| `start_loc` | int | 0 | Starting location |
| `goal_loc` | int | M-1 | Goal location |
| `horizon` | int | 20 | Max steps per episode |
| `step_penalty` | float | 0.01 | Per-step penalty |
| `unlock_bonus` | float | 0.1 | Bonus for unlocking a room |
| `frozen_states` | list | None | Frozen initial states for cycling |

### State Vector (M + 2D - 1 bits, float32)

```
at_loc[0..M-1]           : one-hot location          (M bits)
unlocked[0..D-1]         : room lock status           (D bits, room 0 always 1)
key_available[0..D-2]    : key availability           (D-1 bits)
```

### Actions (Discrete, size = obs_size for n_sites alignment)

```
[0..M-1]                 : MOVE_TO(l)    -- requires unlocked[loc_room[l]]
[M..M+D-2]               : PICK(k)      -- requires at_loc[key_loc[k]] AND key_available[k]
[M+D-1]                  : NOOP
[M+D..obs_size-1]        : invalid -> NOOP
```

### n_sites Resolution

`n_sites = obs_size = M + 2D - 1` (always >= action_count = M+D). Grammar generates
`IsZero(0..n_sites-1)` and `Flip(0..n_sites-1)`. Flip indices beyond `M+D-1` map to
NOOP in the environment. No grammar changes needed.

### Transition Rules (PDDL-faithful)

- **MOVE_TO(l):** If `unlocked[loc_room[l]]==1`, set at_loc to one-hot at l. Else noop.
- **PICK(k):** If `at_loc[key_loc[k]]==1 AND key_available[k]==1`, consume key and
  unlock target room. Else noop.
- **NOOP/Invalid:** No state change.

### Reward Structure

- `-step_penalty` per step (default -0.01)
- `+unlock_bonus` when unlocking a new room (default +0.1, only on 0->1 flip)
- `+1.0` when goal reached

### Terminal Conditions

- **Solved:** `at_loc[goal_loc] == 1` -> terminated
- **Timeout:** `step_count >= horizon` -> truncated

## Feature/Action Mapping: D=2, M=4 Example

### Layout

```python
loc_room    = [0, 0, 1, 1]         # 2 locs per room
key_loc     = [1]                   # key 0 at loc 1 (room 0)
key_unlocks = [1]                   # key 0 -> room 1
start_loc   = 0, goal_loc = 3
```

### Features (IsZero indices)

| Idx | Name | Meaning |
|-----|------|---------|
| 0 | AtLoc(0) | Room 0, start |
| 1 | AtLoc(1) | Room 0, key_0 here |
| 2 | AtLoc(2) | Room 1 |
| 3 | AtLoc(3) | Room 1, goal |
| 4 | Unlocked(0) | Always 1 |
| 5 | Unlocked(1) | Locked initially |
| 6 | KeyAvail(0) | Key 0 available |

### Actions (Flip indices)

| Idx | Action | Precondition |
|-----|--------|-------------|
| 0 | MOVE_TO(0) | unlocked[0] |
| 1 | MOVE_TO(1) | unlocked[0] |
| 2 | MOVE_TO(2) | unlocked[1] |
| 3 | MOVE_TO(3) | unlocked[1] |
| 4 | PICK(0) | at_loc[1] AND key_avail[0] |
| 5 | NOOP | -- |
| 6 | (invalid) | NOOP |

## Feature/Action Mapping: D=3, M=6 Example

### Layout

```python
loc_room    = [0, 0, 1, 1, 2, 2]   # 2 locs per room
key_loc     = [1, 2]                # key 0 at loc 1, key 1 at loc 2
key_unlocks = [1, 2]                # key 0 -> room 1, key 1 -> room 2
start_loc   = 0, goal_loc = 5
```

Sequential dependency: pick key0 -> unlock room1 -> pick key1 -> unlock room2 -> reach goal.

### Features (IsZero indices -> PDDL predicates)

| Idx | Predicate | Meaning |
|-----|-----------|---------|
| 0 | AtLoc(0) | Room 0, start |
| 1 | AtLoc(1) | Room 0, key_0 here |
| 2 | AtLoc(2) | Room 1, key_1 here |
| 3 | AtLoc(3) | Room 1 |
| 4 | AtLoc(4) | Room 2 |
| 5 | AtLoc(5) | Room 2, goal |
| 6 | Unlocked(0) | Always 1 |
| 7 | Unlocked(1) | Locked initially |
| 8 | Unlocked(2) | Locked initially |
| 9 | KeyAvail(0) | Key 0 |
| 10 | KeyAvail(1) | Key 1 |

### Actions (Flip indices -> PDDL actions)

| Idx | Action | Precondition |
|-----|--------|-------------|
| 0-5 | MOVE_TO(l) | unlocked[loc_room[l]] |
| 6 | PICK(0) | at_loc[1] AND key_avail[0] |
| 7 | PICK(1) | at_loc[2] AND key_avail[1] |
| 8 | NOOP | -- |
| 9-10 | (invalid) | NOOP |

## Expressivity Gap Justification

### Why And is Required

PICK(k) requires conjunctive preconditions: `at_loc[key_loc[k]] AND key_available[k]`.

**Without And** -- decision list gets stuck:

```
if Not(IsZero(1)): Flip(4)      # at loc 1 -> PICK(0)
elif Not(IsZero(6)): Flip(1)    # key avail -> go to key loc
elif IsZero(3): Flip(3)         # not at goal -> go to goal
else: Flip(5)                   # noop
```

After picking key 0, agent is still at loc 1. Rule 1 fires again: `Not(IsZero(1))`
is true -> `Flip(4)` (PICK noop, key already taken). **Stuck forever at loc 1.**

**With And** -- correct program:

```
if And(Not(IsZero(1)), Not(IsZero(6))): Flip(4)   # at key AND key avail -> pick
elif Not(IsZero(6)): Flip(1)                       # key avail -> go to key
elif IsZero(3): Flip(3)                            # not at goal -> go there
else: Flip(5)                                      # noop
```

After picking, `Not(IsZero(6))` is false (key consumed) -> And fails -> falls through
to movement rules correctly.

### Budget Analysis

**D=2, M=4 (n_sites=7)** -- optimal program ~16 AST nodes:

```
Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4),   # pick key
  Ite(Not(IsZero(6)), Flip(1),                       # go to key
    Ite(IsZero(3), Flip(3),                           # go to goal
      Default(Flip(5)))))                             # noop
```

Recommended: budget=18, max mode (16-node optimal + slack).

## Grammar Restriction: allow_and / allow_not

Two boolean flags thread through the grammar to control condition production:

- `allow_and=False`: Suppresses all `And(C(i), C(j))` productions
- `allow_not=False`: Suppresses all `Not(C(k-1))` productions

These flags flow through:
- `_condition_productions()` in `derivation.py`
- `DerivationState.legal_productions()` in `derivation.py`
- `compute_max_productions()` in `derivation_game.py`
- `DerivationGame.__init__/reset/step/clone` in `derivation_game.py`

## is_solved Callback

The `is_solved` parameter was added to:
- `run_policy_episode()` in `interpreter.py` -- optional `Callable[[ndarray], bool]`
- `LeafEvaluator.__init__()` in `leaf_evaluator.py` -- stored and passed to interpreter

This allows non-bitstring environments (like Doors) to define their own solved condition
instead of the default `np.all(obs == 1.0)`.

## Implementation Files

| File | Change |
|------|--------|
| `src/alphazeropp/instances/bitstring/envs/__init__.py` | New (empty package) |
| `src/alphazeropp/instances/bitstring/envs/doors_pddl_lite.py` | New: DoorsPDDLLiteEnv |
| `src/alphazeropp/instances/bitstring/dsl/doors_config.py` | New: DoorsGameConfig |
| `src/alphazeropp/instances/bitstring/dsl/interpreter.py` | Added is_solved param |
| `src/alphazeropp/instances/bitstring/dsl/leaf_evaluator.py` | Added is_solved param |
| `src/alphazeropp/instances/bitstring/dsl/derivation.py` | Added allow_and/allow_not |
| `src/alphazeropp/instances/bitstring/dsl/derivation_game.py` | Threaded allow_and/allow_not |
| `src/alphazeropp/instances/bitstring/dsl/derivation_config.py` | DoorsDerivationConfig, DoorsDerivationConfigNoAnd |
| `scripts/run_derivation.py` | Added doors/doors_no_and modes |
| `scripts/estimate_expressivity_gap.py` | New: gap estimation script |
| `tests/test_doors_pddl_lite.py` | New: 23 tests |

## Configuration Defaults

### DoorsDerivationConfig (And enabled)

| Parameter | Value |
|-----------|-------|
| budget | 18 |
| n_sites | 7 (D=2, M=4) |
| program_budget_mode | max |
| allow_and | True |
| num_rooms | 2 |
| horizon | 15 |
| metric | weighted |
| blend_alpha | 0.7 |
| n_simulations | 200 |

### DoorsDerivationConfigNoAnd (And disabled)

Same as above except `allow_and=False`.

## Verification

```bash
# Unit tests (23 tests)
pytest tests/test_doors_pddl_lite.py -v

# Full regression (438 tests)
pytest tests/ --ignore=tests/test_zoning_game.py -v

# Expressivity gap estimation
python scripts/estimate_expressivity_gap.py --rounds 20 --sims 200

# Training dry run
python scripts/run_derivation.py  # Select mode 2 (doors) or 3 (doors_no_and)
```

## Future Work (Track B)

**PDDLGym Adapter:** A wrapper around `pddlgym.make("PDDLEnvDoors-v0")` that converts
PDDL literal observations to fixed bitvectors and maps integer actions to ground PDDL
actions. This would enable testing against the canonical PDDL implementation. Outlined
but not implemented in Phase 1.
