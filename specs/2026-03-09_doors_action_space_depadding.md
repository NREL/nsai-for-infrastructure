# 2026-03-09 — Doors Action-Space Depadding

## 1. Problem: Action Space Is Padded to Match Observation Size

The Doors environment (`DoorsPDDLLiteEnv`) artificially inflates its action space so that `action_space.n == obs_size`. This has no semantic purpose — it exists only so the network's `output_size` matches `input_size` for symmetry.

### Concrete example: D=2, L=2

```
Observation vector (obs_size = 7):
  [0] AtLoc(0)      ─┐
  [1] AtLoc(1)       │ M = 4 location bits
  [2] AtLoc(2)       │
  [3] AtLoc(3)      ─┘
  [4] Unlocked(0)   ─┐ D = 2 room-lock bits
  [5] Unlocked(1)   ─┘
  [6] KeyAvail(0)   ── K = 1 key-availability bit

Action space (action_space.n = 7, but only 6 are real):
  [0] MOVE_TO(0)    ─┐
  [1] MOVE_TO(1)     │ M = 4 move actions
  [2] MOVE_TO(2)     │
  [3] MOVE_TO(3)    ─┘
  [4] PICK(0)       ── K = 1 pick action
  [5] NOOP          ── 1 noop
  [6] INVALID       ── PADDING (always masked out, treated as NOOP if called)
```

Real actions = M + K + 1 = 4 + 1 + 1 = **6**.
Declared action_space = obs_size = **7**.
Padding = 7 − 6 = **1 ghost action**.

### Concrete example: D=10, L=3

```
M = 10 × 3 = 30 locations
D = 10 rooms
K = D − 1 = 9 keys  (room 0 starts unlocked, so 9 keys for rooms 1–9)

Observation vector (3 segments):
  at_loc[0..29]       →  M  = 30 bits   (one-hot location)
  unlocked[0..9]      →  D  = 10 bits   (room lock status)
  key_available[0..8] →  K  =  9 bits   (key availability)
  obs_size = M + D + K = 30 + 10 + 9 = 49   (equivalently M + 2D − 1, since K = D − 1)

Action space:
  MOVE_TO(0..29)      →  M  = 30 actions
  PICK(0..8)          →  K  =  9 actions
  NOOP                →       1 action
  real_actions = M + K + 1 = 40

Padding = obs_size − real_actions = 49 − 40 = 9 ghost actions (indices 40–48)
```

The network outputs a 49-dim policy vector, 9 entries of which are always masked to zero by MCTS. Those 9 output neurons receive no gradient (masked before softmax in training targets) but still consume parameters and computation.

### Where this lives in code

| File | Line | What it does |
|------|------|--------------|
| `doors_pddl_lite.py` | 132 | `self._action_count = self.M + self.K + 1` (correct real count) |
| `doors_pddl_lite.py` | 136 | `self.action_space = spaces.Discrete(self._obs_size)` **(bug: uses obs_size)** |
| `config.py` | 61 | `"output_size": obs_size` (network sized to match padded action space) |
| `run_doors_direct.py` | 44 | `cfg.net.kwargs["output_size"] = dims["obs_size"]` (same coupling) |
| `game.py` | 40–45 | Structural mask: `[True]*n_real + [False]*(n_total-n_real)` (masks out padding) |
| `doors_pddl_lite.py` | 260–261 | `action_spec` emits `INVALID(i)` entries for padding slots |

### Why this matters

1. **Scientific confound:** When comparing algorithms on this benchmark, the declared `action_space.n` misrepresents the true branching factor. An RL baseline that doesn't use masking (e.g., vanilla PPO) would waste exploration on 9 dead actions.
2. **Wasted capacity:** 9 network output neurons that never receive gradient.
3. **Misleading API:** `env.action_space.n` should tell you how many actions the environment accepts. Currently it lies.

---

## 2. Plan

**Approach:** Design A — fix the base env to expose `Discrete(real_action_count)`. No compatibility wrapper needed because the AlphaZero core (MCTS, Agent, Trainer) is fully generic: MCTS uses `len(nn_policy)`, Agent uses `len(move_probs)`, Trainer passes through. Zero core code changes required.

### Step 1: Add `action_count` to `compute_dims()`

**File:** [config.py](src/alphazeropp/instances/doors/config.py)

```python
# In compute_dims() — add to return dict:
action_count = M + (num_rooms - 1) + 1   # M moves + K picks + 1 noop
return {"obs_size": obs_size, "horizon": horizon, "M": M, "action_count": action_count}
```

Then change line 61:
```python
"output_size": dims["action_count"],   # was: dims["obs_size"]
```

**Why safe:** `output_size` flows only to the network constructor (`PolicyValueNetModel`), which accepts any positive integer. No other code reads this config value.

### Step 2: Fix action_space in base env

**File:** [doors_pddl_lite.py](src/alphazeropp/instances/doors/doors_pddl_lite.py)

- Line 134: Update comment — remove "padded to obs_size for n_sites alignment"
- Line 136: `self.action_space = spaces.Discrete(self._action_count)`
- Lines 260–261: Delete the loop that generates `INVALID` entries in `action_spec`
- Add public property:
  ```python
  @property
  def action_count(self) -> int:
      return self._action_count
  ```
- Add encode/decode helpers:
  ```python
  def encode_action(self, action_type: str, param: int = -1) -> int:
      if action_type == "move":
          assert 0 <= param < self.M
          return param
      elif action_type == "pick":
          assert 0 <= param < self.K
          return self.M + param
      elif action_type == "noop":
          return self.M + self.K
      raise ValueError(f"Unknown action type: {action_type}")

  def decode_action(self, action: int) -> tuple[str, int]:
      if 0 <= action < self.M:
          return ("move", action)
      elif self.M <= action < self.M + self.K:
          return ("pick", action - self.M)
      elif action == self.M + self.K:
          return ("noop", -1)
      raise ValueError(f"Invalid action: {action}")
  ```

**Why safe:** `n_sites` property (line 155) returns `_obs_size`, not `action_space.n`. The derivation pipeline's `game.action_space.n` (derivation_config.py:193) refers to the grammar game, not this env. No code outside `game.py` reads `DoorsPDDLLiteEnv.action_space.n`.

### Step 3: Simplify structural mask in game wrapper

**File:** [game.py](src/alphazeropp/instances/doors/game.py)

Replace lines 40–45:
```python
# Before (padding mask):
n_real = env.M + env.K + 1
n_total = env.action_space.n
self._structural_mask = np.array([True]*n_real + [False]*(n_total - n_real), dtype=bool)

# After (all actions are real):
self._structural_mask = np.ones(env.action_space.n, dtype=bool)
```

The precondition mask path (lines 58–78) already uses `env.action_space.n` and `env.M`, `env.K` — works unchanged since those values haven't changed.

### Step 4: Update run script dimension sync

**File:** [run_doors_direct.py](scripts/run_doors_direct.py)

Line 44:
```python
cfg.net.kwargs["output_size"] = dims["action_count"]   # was: dims["obs_size"]
```

Update any diagnostic prints that reference padding count or action space alignment.

### Step 5: Add `action_masks()` method to env

**File:** [doors_pddl_lite.py](src/alphazeropp/instances/doors/doors_pddl_lite.py)

Add a Gymnasium-standard `action_masks()` method on the base env (currently masking only lives in `game.py`):

```python
def action_masks(self, mode: str = "precondition") -> np.ndarray:
    """Return boolean mask over real semantic actions.

    mode="none": all actions available (invalid semantics → noop in step)
    mode="precondition": only currently legal actions
    """
    if mode == "none":
        return np.ones(self._action_count, dtype=bool)

    mask = np.zeros(self._action_count, dtype=bool)
    for l in range(self.M):
        room = self.loc_room[l]
        mask[l] = self._state[self._unlocked_offset + room] == 1.0
    for k in range(self.K):
        mask[self.M + k] = (
            self._state[self.key_loc[k]] == 1.0
            and self._state[self._key_offset + k] == 1.0
        )
    mask[self.M + self.K] = True  # NOOP always valid
    return mask
```

This duplicates logic from `game.py:get_action_mask()` but lives on the base env so non-AlphaZero consumers (model-free baselines) can access it directly. The `game.py` version can delegate to this.

### Step 6: Update tests

**Files:** [test_doors_direct.py](tests/test_doors_direct.py), [test_doors_pddl_lite.py](tests/test_doors_pddl_lite.py)

| Test | Current assertion | New assertion | Why |
|------|-------------------|---------------|-----|
| `test_action_mask_structural` | `mask.shape == (7,)`, `not mask[6]` | `mask.shape == (6,)`, all True | No padding slot |
| `test_action_mask_precondition_initial` | `mask.shape == (7,)` | `mask.shape == (6,)` | Smaller space |
| `test_predict_shape` | `policy.shape == (7,)` | `policy.shape == (6,)` | Network output matches |
| `test_invalid_action_noop` | Steps with action 6 | Remove or change to action 5 (NOOP) | Action 6 no longer exists |

**New tests to add:**

```python
def test_action_space_equals_real_count():
    env = DoorsPDDLLiteEnv.make_d2()
    assert env.action_space.n == env.M + env.K + 1  # 6, not 7

def test_encode_decode_roundtrip():
    env = DoorsPDDLLiteEnv.make_d2()
    for a in range(env.action_space.n):
        atype, param = env.decode_action(a)
        assert env.encode_action(atype, param) == a

def test_no_invalid_in_action_spec():
    env = DoorsPDDLLiteEnv.make_d2()
    for spec in env.action_spec:
        assert spec.action_type != "invalid"

def test_action_masks_precondition():
    env = DoorsPDDLLiteEnv.make_d2()
    env.reset(seed=0)
    mask = env.action_masks("precondition")
    assert mask.shape == (6,)
    assert mask[env.M + env.K]  # NOOP always valid
    # Room 0 unlocked → moves to room 0 locs valid
    assert mask[0] and mask[1]
    # Room 1 locked → moves to room 1 locs invalid
    assert not mask[2] and not mask[3]
```

### Step 7: Add documentation

**File:** [docs/doors_benchmark_interface.md](docs/doors_benchmark_interface.md) (new)

Short markdown note covering:
- Old interface: `action_space = Discrete(obs_size)` with masking for padding
- New interface: `action_space = Discrete(M + K + 1)`, all actions are semantically real
- Why padding was removed: it conflated network architecture with environment semantics
- `encode_action` / `decode_action` API
- `action_masks(mode)` API

---

## 3. What Does NOT Change

- **Reward function:** Untouched (step penalty, unlock bonus, goal reward)
- **Observation space:** Still `MultiBinary(obs_size)`, obs_size unchanged
- **`n_sites` property:** Still returns `obs_size` (used by derivation pipeline)
- **Core AlphaZero:** `mcts.py`, `agent.py`, `trainer.py`, `network.py` — zero changes
- **Derivation pipeline:** `derivation_config.py` uses grammar game's action space
- **Optimal policy semantics:** Same actions in same order achieve same reward

## 4. Backward Compatibility Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Old checkpoints have `output_size=obs_size` | Low | No production checkpoints need preserving. If needed: slice policy head weights `[:action_count]` |
| Scripts that hardcode `action_space.n == obs_size` | Low | Grep found zero such cases outside the files being modified |
| Ablation experiments (planned) | None | They use the config pipeline; updated config flows through automatically |

## 5. Verification

```bash
# 1. Run all Doors tests
pytest tests/test_doors_direct.py tests/test_doors_pddl_lite.py tests/test_doors_baselines.py -v

# 2. Smoke-test: 1 iteration D=2 training
python scripts/run_doors_direct.py --non-interactive --override n_iterations=1

# 3. Verify dimensions for D=10/L=3
python -c "
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
env = DoorsPDDLLiteEnv(num_rooms=10, locs_per_room=3)
assert env.action_space.n == 40, f'Expected 40, got {env.action_space.n}'
assert env.observation_space.n == 49
print(f'OK: actions={env.action_space.n}, obs={env.observation_space.n}')
"
```
