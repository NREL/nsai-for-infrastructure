# MCTS Tree Reuse: Detailed Before/After With Soundness Analysis

## 1. What Happens Today (BEFORE)

Consider a 3-move episode: `S0 --(a1)--> S1 --(a2)--> S2 --(a3)--> S3(terminal)`

### Move 1 at S0

```python
# agent.py:122 — play_one_round calls policy()
move_probs = self.policy(current_game_state, ...)

# agent.py:93-101 — policy() creates a FRESH MCTS
current_game_state = state.clone()                          # clone #1
mcts = MCTS(current_game_state, self.net, **self.mcts_params)
#   mcts.nodes = {}                                         # empty tree!
move_probs = mcts.perform_simulations("", add_noise=True)
#   Simulation loop builds the tree:
#   nodes = {
#     S0: Node(total_N=25, action_N={(a1,):10, (a2,):8, (a3,):7}, action_Q={...}),
#     S1: Node(total_N=8,  action_N={(a4,):3, (a5,):5},           action_Q={...}),
#     S2: Node(total_N=5,  ...),
#     ... maybe 15 more nodes explored deeper ...
#   }
#   Returns visit-count probs from S0

# After policy() returns, mcts goes out of scope. ALL NODES DISCARDED.
action_idx = rng.choice(len(move_probs), p=move_probs)      # pick a1
current_game_state.step_wrapper(action_idx)                  # game -> S1
```

### Move 2 at S1

```python
move_probs = self.policy(current_game_state, ...)

# policy() creates ANOTHER FRESH MCTS
current_game_state = state.clone()                          # clone #2
mcts = MCTS(current_game_state, self.net, **self.mcts_params)
#   mcts.nodes = {}                                         # empty tree AGAIN!
move_probs = mcts.perform_simulations("", add_noise=True)
#   Must re-discover S1, S2, S3, ... from scratch
#   Even though Move 1 already explored S1 extensively!
#   25 more simulations, rebuilding the same subtree

# mcts goes out of scope. ALL NODES DISCARDED AGAIN.
action_idx = rng.choice(...)                                 # pick a2
current_game_state.step_wrapper(action_idx)                  # game -> S2
```

### Move 3 at S2

```python
# Same thing. THIRD fresh MCTS, THIRD empty tree, 25 more simulations
# rebuilding subtrees we've seen twice before.
```

### Summary of waste

```
Move 1: MCTS() → build 20 nodes → throw away
Move 2: MCTS() → build 18 nodes → throw away    (many overlap with Move 1)
Move 3: MCTS() → build 12 nodes → throw away    (many overlap with Moves 1+2)

Total: 3 clones, 3 MCTS instances, 75 simulations, ~50 nodes built, 0 reused
```

---

## 2. What Happens With Tree Reuse (AFTER)

Same episode: `S0 --(a1)--> S1 --(a2)--> S2 --(a3)--> S3(terminal)`

### Episode start

```python
# play_one_round_reuse_tree creates ONE MCTS for the whole episode:
mcts = MCTS(game.clone(), self.net, **self.mcts_params)
#   mcts.nodes = {}
#   mcts.game = the cloned game (at S0)
```

### Move 1 at S0

```python
probs = mcts.perform_simulations_reuse("", add_noise=True)
#   mystate = S0
#   S0 not in nodes → expand (creates S0 node, queries NN)
#   Inject Dirichlet noise into S0's policy:
#     Save original:  S0.nn_policy_original = [0.3, 0.3, 0.4]
#     Mix noise:      S0.nn_policy = 0.75 * [0.3, 0.3, 0.4] + 0.25 * [noise]
#   Run 25 simulations from S0
#   Tree after Move 1:
#     nodes = {
#       S0: Node(total_N=25, action_N={(a1,):10, (a2,):8, (a3,):7}),
#       S1: Node(total_N=8,  nn_policy_original=[0.5, 0.5], action_N={(a4,):3, (a5,):5}),
#       S2: Node(total_N=5,  nn_policy_original=[0.6, 0.4], action_N={(a6,):3, (a7,):2}),
#       S4: Node(total_N=2,  ...),
#       S5: Node(total_N=1,  ...),
#       ... ~15 more nodes ...
#     }

action = sample(probs)    # pick a1
mcts.advance_to(a1)
#   Internally: self.game.step_wrapper(a1)
#   mcts.game is now at S1
#   mcts.nodes is UNCHANGED — still has all ~20 nodes
#   S1 node has: total_N=8, action_N={(a4,):3, (a5,):5}, action_Q={...}
```

### Move 2 at S1

```python
probs = mcts.perform_simulations_reuse("", add_noise=True)
#   mystate = mcts.game.hashable_obs = S1
#   S1 IS ALREADY IN nodes! (total_N=8 from Move 1's exploration)
#
#   Noise injection:
#     S1.total_N == 8 (NOT zero!)
#     OLD code would SKIP noise here — BUG!
#     NEW code: restore S1.nn_policy from S1.nn_policy_original, mix FRESH noise
#     S1.nn_policy = 0.75 * [0.5, 0.5] + 0.25 * [new_noise]
#
#   Run 25 simulations from S1:
#     Simulation 1: UCB at S1 sees action_N={(a4,):3, (a5,):5}
#       exploration term for a4: c * P(a4) * sqrt(8) / (1+3) = higher
#       exploration term for a5: c * P(a5) * sqrt(8) / (1+5) = lower
#       → tends to explore a4 more (less visited) — WARM START WORKING!
#     ... 24 more sims, deepening the subtree below S1 ...
#
#   Tree after Move 2:
#     nodes = {
#       S0: Node(total_N=25, ...),              ← stale ancestor, still in dict
#       S1: Node(total_N=33, ...),              ← 8 old + 25 new
#       S2: Node(total_N=15, ...),              ← 5 old + 10 new
#       S4: Node(total_N=8,  ...),              ← warmed up from Move 1
#       S5: Node(total_N=6,  ...),              ← warmed up from Move 1
#       S6: Node(total_N=2,  ...),              ← newly discovered
#       ... ~25 nodes total ...
#     }

action = sample(probs)    # pick a2
mcts.advance_to(a2)       # game → S2
```

### Move 3 at S2

```python
probs = mcts.perform_simulations_reuse("", add_noise=True)
#   S2 IS ALREADY IN nodes! (total_N=15 from Moves 1+2)
#   Warm-started Q values and visit counts guide search
#   25 more simulations, going MUCH deeper than a fresh tree would

action = sample(probs)
mcts.advance_to(a3)       # terminal
```

### Summary of gains

```
Move 1: 25 sims → build ~20 nodes
Move 2: 25 sims → reuse 20 nodes + expand ~5 new = 25 total (vs 18 rebuilt from scratch)
Move 3: 25 sims → reuse 25 nodes + expand ~3 new = 28 total (vs 12 rebuilt from scratch)

Total: 1 clone, 1 MCTS instance, 75 simulations, ~28 nodes built, ~45 node-visits reused
Bonus: each move benefits from warm-started Q values from prior moves
```

---

## 3. The Dirichlet Noise Bug — Detailed Trace

This is the critical correctness issue.

### What happens WITHOUT the fix (using original `perform_simulations`):

```python
# mcts.py:95
if add_noise and mynode.total_N == 0:   # <-- the check
    # inject noise into mynode.nn_policy
```

```
Move 1, root=S0:
  S0 just created → total_N == 0 → NOISE INJECTED ✓
  S0.nn_policy = 0.75 * [0.3, 0.3, 0.4] + 0.25 * [0.1, 0.7, 0.2]
               = [0.25, 0.40, 0.35]  ← encourages exploring a2

Move 2, root=S1 (after advance_to):
  S1.total_N == 8 (visited as child of S0)
  total_N == 0 is FALSE → NOISE SKIPPED ✗
  S1.nn_policy stays at NN's raw output [0.5, 0.5]
  → NO exploration noise! Search follows NN policy blindly.

Move 3, root=S2:
  S2.total_N == 5 → NOISE SKIPPED AGAIN ✗
  → NO exploration noise again!
```

**Result**: Only the first move gets exploration noise. All subsequent moves have no Dirichlet noise, making the search overly greedy and less likely to discover good actions.

### What happens WITH the fix (using `perform_simulations_reuse`):

```python
# Always restore from original, then mix fresh noise:
if add_noise and mynode.nn_policy_original is not None:
    mynode.nn_policy = mynode.nn_policy_original.copy()  # restore clean
    noise = np.random.dirichlet(...)                     # fresh draw
    mynode.nn_policy = (1-eps) * mynode.nn_policy + eps * masked_noise
```

```
Move 1, root=S0:
  S0.nn_policy_original = [0.3, 0.3, 0.4]  (saved at expansion)
  Restore from original, mix noise draw #1:
  S0.nn_policy = 0.75 * [0.3, 0.3, 0.4] + 0.25 * [0.1, 0.7, 0.2]
               = [0.25, 0.40, 0.35] ✓

Move 2, root=S1:
  S1.nn_policy_original = [0.5, 0.5]  (saved when S1 was first expanded in Move 1)
  S1.total_N == 8 — DOESN'T MATTER, we don't check total_N
  Restore from original, mix noise draw #2 (different random values):
  S1.nn_policy = 0.75 * [0.5, 0.5] + 0.25 * [0.8, 0.2]
               = [0.575, 0.425] ✓ NOISE APPLIED!

Move 3, root=S2:
  S2.nn_policy_original = [0.6, 0.4]
  Restore from original, mix noise draw #3:
  S2.nn_policy = 0.75 * [0.6, 0.4] + 0.25 * [0.3, 0.7]
               = [0.525, 0.475] ✓ NOISE APPLIED!
```

### Why nn_policy_original is needed (not just removing the total_N check):

If we just removed the `total_N == 0` check and injected noise every time:

```
Move 1: S0.nn_policy = mix(NN_output, noise_1) = [0.25, 0.40, 0.35]
Move 2: S1.nn_policy = mix(NN_output, noise_2) = [0.575, 0.425]   ← OK so far

But what if S1 became root AGAIN (e.g., through game transpositions)?
  S1.nn_policy is ALREADY noisy from Move 2: [0.575, 0.425]
  Injecting noise again: mix([0.575, 0.425], noise_3)
  = noise layered on noise — WRONG! Should be mix(ORIGINAL, noise_3)
```

Storing `nn_policy_original` guarantees we always mix noise with the **clean NN output**.

---

## 4. The Game Identity Issue — Why We Read From `mcts.game`

```python
# In perform_simulations, each simulation does:
for i in range(self.n_simulations):
    old_game_state = self.game.stash_state()    # save
    self.search(...)                             # mutates self.game
    self.game = self.game.unstash_state(old_game_state)  # restore
```

### Base Game class (deepcopy-based):

```python
def stash_state(self):
    return copy.deepcopy(self)       # returns a NEW object (the copy)

def unstash_state(self, state):
    return state                     # returns the copy, NOT self
```

Trace:
```
before:  mcts.game = <Game object A at 0x1000>
stash:   old = deepcopy(A) → <Game object B at 0x2000>
search:  mutates A
unstash: self.game = unstash_state(B) = B  → mcts.game = <Game object B at 0x2000>

mcts.game is now object B, NOT object A!
```

### DerivationGame (tuple-based, returns self):

```python
def stash_state(self):
    return (self._deriv_state, self._current_productions, ...)  # tuple

def unstash_state(self, state):
    (self._deriv_state, self._current_productions, ...) = state
    return self                      # returns SELF
```

Trace:
```
before:  mcts.game = <DerivationGame object A at 0x1000>
stash:   old = (field1, field2, ...)
search:  mutates A's fields
unstash: restores A's fields, self.game = A  → mcts.game = <DerivationGame A at 0x1000>

mcts.game is still object A. Same identity.
```

### Why this matters for play_one_round_reuse_tree:

```python
# WRONG (works for DerivationGame but breaks for base Game):
current_game_state = game.clone()
mcts = MCTS(current_game_state, ...)
mcts.perform_simulations_reuse(...)
# current_game_state might NOT be mcts.game anymore!
obs = current_game_state.obs          # ← stale reference!

# CORRECT (works for ALL game types):
mcts.perform_simulations_reuse(...)
obs = mcts.game.obs                   # ← always the live reference
mcts.advance_to(action)
reward = mcts.game.reward             # ← always correct
terminated = mcts.game.terminated     # ← always correct
```

---

## 5. Warm-Started Q Values — Are They Sound?

When S1 becomes root at Move 2, it has `action_Q` and `action_N` from being explored as S0's child during Move 1.

### Trace of how Q values were built:

```
During Move 1's simulation #3 (for example):
  At S0, UCB picks a1 → step to S1
  At S1, UCB picks a4 → step to S4
  At S4, leaf → NN returns value=0.7
  Backprop: immediate_reward(S1→S4) + future_value(0.7) = 0.0 + 0.7 = 0.7
  update_edge(S1, a4, 0.7):
    S1.action_Q[(a4,)] = 0.7
    S1.action_N[(a4,)] = 1

During Move 1's simulation #7:
  At S0, UCB picks a1 → step to S1
  At S1, UCB picks a4 again → step to S4
  At S4, already expanded, UCB picks a8 → step to S8
  S8 is leaf → NN returns value=0.3
  Backprop: 0.0 + 0.3 = 0.3
  update_edge(S4, a8, 0.3) then backprop to S1:
  total for S1→a4 = reward(S1→S4) + [reward(S4→S8) + value(S8)] = 0.0 + 0.0 + 0.3 = 0.3
  update_edge(S1, a4, 0.3):
    S1.action_Q[(a4,)] = (1*0.7 + 0.3) / 2 = 0.5
    S1.action_N[(a4,)] = 2
```

### At Move 2, S1 becomes root with warm data:

```
S1.total_N = 8
S1.action_Q = {(a4,): 0.5, (a5,): 0.3}
S1.action_N = {(a4,): 3,   (a5,): 5}
```

### Are these Q values valid? YES, because:

1. **Same NN**: Q values were estimated using the same network. No weight updates happened between moves within an episode.
2. **Same game dynamics**: The transition probabilities and rewards haven't changed.
3. **Standard practice**: This is exactly how AlphaGo/AlphaZero reuse trees — the subtree below the chosen action is retained with its statistics.

### One subtlety: Q normalization resets

```python
# perform_simulations_reuse resets q_min/q_max at the start:
self.q_min = float('inf')
self.q_max = float('-inf')
```

This means on Move 2's first simulation at S1:
```
calc_masked_ucbs(S1):
  q_min = inf, q_max = -inf  (just reset, no edges updated yet this round)
  for action a4: q = 0.5, BUT q_normalized = 0.0 (because q_min is inf)
  for action a5: q = 0.3, BUT q_normalized = 0.0

  UCB = 0.0 + exploration_term
  → First sim is purely exploration-driven (policy prior + visit counts)
```

After the first simulation updates at least one edge, `q_min`/`q_max` are set and subsequent simulations normalize properly. This is a minor inefficiency (1 out of 25 sims doesn't use warm Q for ranking), not a correctness issue.

---

## 6. Code Changes — Exactly What Changes Where

### Change 1: `MCTSTreeNode` — add field ([mcts.py:14-36](src/alphazeropp/core/mcts.py#L14-L36))

```python
class MCTSTreeNode():
    # ... existing fields ...
    nn_policy_original: Any  # NEW: NN policy before noise, for re-injection

    def __init__(self, direct_reward, is_terminal_state):
        # ... existing init ...
        self.nn_policy_original = None  # NEW
```

### Change 2: `MCTS.search()` — store original on expansion ([mcts.py:176-180](src/alphazeropp/core/mcts.py#L176-L180))

```python
# BEFORE (lines 176-180):
mypolicy, myvalue, myaction_mask = self.query_net_masked(msg)
mynode.nn_policy = mypolicy
mynode.nn_value = myvalue
mynode.action_mask = myaction_mask
return myvalue

# AFTER:
mypolicy, myvalue, myaction_mask = self.query_net_masked(msg)
mynode.nn_policy = mypolicy
mynode.nn_policy_original = mypolicy.copy()  # NEW LINE
mynode.nn_value = myvalue
mynode.action_mask = myaction_mask
return myvalue
```

### Change 3: New method `MCTS.perform_simulations_reuse()` ([mcts.py](src/alphazeropp/core/mcts.py))

Same as `perform_simulations` except lines 94-117 (noise section) become:
```python
# Instead of: if add_noise and mynode.total_N == 0:
if add_noise and mynode.nn_policy_original is not None:
    mynode.nn_policy = mynode.nn_policy_original.copy()
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))
    mask = mynode.action_mask
    masked_noise = noise * mask
    sum_noise = masked_noise.sum()
    if sum_noise > 0:
        masked_noise /= sum_noise
        mynode.nn_policy = (1 - self.dirichlet_epsilon) * mynode.nn_policy + self.dirichlet_epsilon * masked_noise
```

### Change 4: New method `MCTS.advance_to()` ([mcts.py](src/alphazeropp/core/mcts.py))

```python
def advance_to(self, action):
    """Step the game forward. The new hashable_obs becomes the implicit root."""
    self.game.step_wrapper(action)
```

### Change 5: New method `Agent.play_one_round_reuse_tree()` ([agent.py](src/alphazeropp/core/agent.py))

```python
def play_one_round_reuse_tree(self, game, max_moves=10_000,
                               random_seed=None, msg="",
                               add_noise=True, temperature_override=None):
    mcts = MCTS(game.clone(), self.net, **self.mcts_params)
    if temperature_override is not None:
        mcts.temperature = temperature_override
    rng = np.random.default_rng(random_seed)

    collected_experience = []
    collected_rewards = []
    cumulative_reward = 0.0

    for i in range(max_moves):
        move_probs = mcts.perform_simulations_reuse("", add_noise=add_noise)
        assert len(move_probs.shape) == 1
        action_idx = rng.choice(len(move_probs), p=move_probs)

        collected_experience.append((mcts.game.obs.copy(), move_probs))
        #                            ^^^^^^^^^ NOT a local var — always mcts.game

        mcts.advance_to(action_idx)

        reward = mcts.game.reward
        terminated = mcts.game.terminated
        truncated = mcts.game.truncated

        collected_rewards.append(reward)
        cumulative_reward += reward
        if terminated or truncated:
            break

    # Discounted rewards — identical to play_one_round
    discounted_rewards = []
    cumulative_reward = 0.0
    for reward in reversed(collected_rewards):
        cumulative_reward = reward + self.reward_discount * cumulative_reward
        discounted_rewards.append(cumulative_reward)
    discounted_rewards.reverse()

    collected_experience = [
        (obs, move_probs, dr)
        for (obs, move_probs), dr in zip(collected_experience, discounted_rewards)
    ]
    return collected_experience, sum(collected_rewards)
```

### Change 6: New method `Agent.play_for_experience_reuse_tree()` ([agent.py](src/alphazeropp/core/agent.py))

Same as `play_for_experience` but calls `play_one_round_reuse_tree`.

### Change 7: `Trainer` flag ([trainer.py](src/alphazeropp/training/trainer.py))

Add `use_tree_reuse: bool = False` to `__init__`. In `_collect_training_examples`, select the appropriate `play_for_experience` variant.

---

## 7. What Does NOT Change

- `perform_simulations()` — untouched, original behavior preserved
- `policy()` — untouched, still creates fresh MCTS per call
- `play_one_round()` — untouched, still calls `policy()` per move
- `play_for_experience()` — untouched
- `search()` — only change is one added line (`nn_policy_original = mypolicy.copy()`)
- All existing tests — no behavior changes

---

## 8. Tests

New file: `tests/test_mcts_tree_reuse.py`

| Test | What it verifies |
|------|-----------------|
| `test_nn_policy_original_stored` | `nn_policy_original` is set after leaf expansion |
| `test_advance_to_updates_state` | `advance_to` changes `mcts.game.hashable_obs` |
| `test_noise_on_reused_root` | Noise IS injected when `total_N > 0` (the fix) |
| `test_noise_fresh_each_move` | Noise is re-drawn, not stale/doubled |
| `test_reuse_matches_existing_pattern` | With `add_noise=False`, produces identical actions to existing test pattern |
| `test_play_one_round_reuse_tree` | Agent method runs, produces valid experience tuples |
| `test_finds_good_program` | End-to-end: finds good program via tree reuse path |

## 9. Verification

```bash
pytest tests/test_mcts_tree_reuse.py -v    # new tests
pytest tests/ -v                            # no regressions
```
