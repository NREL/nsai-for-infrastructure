# AlphaZero on Doors: Game, MCTS, and Ablation Study

This document is self-contained. It explains the Doors environment, how AlphaZero solves it, and how the ablation study isolates the contribution of each component.

---

## 1. The Doors Game

### 1.1 Physical setup

The world is a grid of **rooms** connected by **locked doors**. The agent must navigate from a start location to a goal location, but most rooms are locked. To unlock a room, the agent must first collect the right key.

Concrete parameters:
- **D** = number of rooms (e.g. D=3 means rooms 0, 1, 2)
- **L** = locations per room (e.g. L=2 means each room has 2 locations)
- **M** = D * L = total locations (e.g. M=6 for D=3, L=2)
- **K** = D - 1 = number of keys (one key per locked room; room 0 starts unlocked)

### 1.2 Layout example (D=3, L=2)

```
Room 0 (unlocked)     Room 1 (locked)      Room 2 (locked)
+--------+--------+   +--------+--------+   +--------+--------+
| Loc 0  | Loc 1  |   | Loc 2  | Loc 3  |   | Loc 4  | Loc 5  |
| START  | KEY 0  |   | KEY 1  |        |   |        | GOAL   |
+--------+--------+   +--------+--------+   +--------+--------+
                  |   |                  |   |
               Door 0→1              Door 1→2
            (needs Key 0)         (needs Key 1)
```

- Key 0 is at location 1 (room 0). Picking it unlocks room 1.
- Key 1 is at location 2 (room 1). Picking it unlocks room 2.
- The goal is location 5 (room 2).

### 1.3 State representation

The observation is a flat float32 vector of size M + D + K = M + 2D - 1:

```
obs = [at_loc(0), at_loc(1), ..., at_loc(M-1),    # one-hot: which location am I at?
       unlocked(0), unlocked(1), ..., unlocked(D-1), # binary: which rooms are unlocked?
       key_avail(0), key_avail(1), ..., key_avail(K-1)] # binary: which keys remain?
```

For D=3, L=2, the initial state is:

```
obs = [1, 0, 0, 0, 0, 0,   1, 0, 0,   1, 1]
       ^                    ^          ^
       at loc 0             room 0     both keys
                            unlocked   available
```

### 1.4 Actions

There are M + K + 1 discrete actions:

| Action index | Name       | Meaning |
|-------------|------------|---------|
| 0 to M-1    | MOVE_TO(l) | Teleport to location l. Only succeeds if the room containing l is unlocked. |
| M to M+K-1  | PICK(k)    | Pick up key k. Only succeeds if you are at the key's location AND the key is still available. Consumes the key and unlocks the corresponding room. |
| M+K         | NOOP       | Do nothing. Always valid. |

For D=3, L=2: actions are MOVE_TO(0..5), PICK(0), PICK(1), NOOP = 8 actions total.

### 1.5 Rewards

| Event | Reward |
|-------|--------|
| Every step | -0.01 (step penalty, encourages efficiency) |
| Picking a key that unlocks a new room | +0.10 (unlock bonus) |
| Reaching the goal location | +1.00 (terminal reward) |

### 1.6 Optimal solution

The agent must follow a strict sequential chain:

```
MOVE_TO(1) → PICK(0) → MOVE_TO(2) → PICK(1) → MOVE_TO(5) → SOLVED
```

That's 2*(D-1) + 1 = 5 steps for D=3. The horizon is set to max(15, 5 * optimal_steps) to give the agent room to explore.

### 1.7 Why this problem is hard

- **Sequential dependency**: You can't skip ahead. Key 1 is in room 1, which requires key 0, which is in room 0.
- **Sparse reward**: The +1.0 terminal reward only fires when you reach the goal. Until then, the agent only sees -0.01 penalties and occasional +0.10 bonuses.
- **Large action space**: With D=14, L=3, there are M+K+1 = 42+13+1 = 56 actions per step.
- **Long horizon**: D=14 requires 27 optimal steps but has a horizon of 135 steps.

---

## 2. AlphaZero: How It Learns

AlphaZero has three components that work together:

1. **A neural network** with two heads:
   - **Policy head**: Given state s, outputs P(a|s) — a probability distribution over actions. "Which action looks promising?"
   - **Value head**: Given state s, outputs V(s) — a scalar estimate of expected future reward. "How good is this state?"

2. **Monte Carlo Tree Search (MCTS)**: A planning algorithm that uses the network to look ahead many steps before choosing an action.

3. **A training loop**: Self-play generates data, then the network is trained on that data, then the improved network generates better data.

### 2.1 The Training Loop

```
for iteration = 1 to N:
    1. SELF-PLAY: Play 50 games using MCTS + current network.
       Each game produces training examples: (state, MCTS_policy, outcome).
    2. TRAIN: Update the neural network on all collected examples.
       Loss = CrossEntropy(network_policy, MCTS_policy) + MSE(network_value, outcome)
    3. EVALUATE: Play 20 test games with the new network. Measure solve rate.
```

### 2.2 How a single self-play game works

```
game.reset()  →  state s₀

while not done:
    action_probs = MCTS.perform_simulations(s)  ← 120 simulations
    action = sample(action_probs)                ← pick action from MCTS distribution
    s, reward, done = game.step(action)
    store (s, action_probs) as training example

after game ends:
    label each stored example with the final return (discounted cumulative reward)
```

---

## 3. MCTS: The Full Algorithm

This is the core of AlphaZero. MCTS builds a search tree to evaluate actions before committing to one.

### 3.1 Data structures

Each **node** in the tree corresponds to a game state. Each node stores:
- `nn_policy`: P(a|s) from the neural network (set once when the node is first expanded)
- `nn_value`: V(s) from the neural network (set once when the node is first expanded)
- `action_N[a]`: number of times action a was taken from this node
- `action_Q[a]`: average return observed after taking action a from this node
- `total_N`: total visits to this node (sum of all action_N)

### 3.2 The three phases of each simulation

**Each of the 120 simulations starts from the ROOT node** (the current real game state) and walks down the tree:

#### Phase 1: SELECT — Walk down existing tree using UCB

Starting at the root, repeatedly pick the action with the highest UCB score until reaching a node that has an unexplored child (a state we haven't seen before).

```
UCB(s, a) = Q_normalized(s, a) + c * P(a|s) * sqrt(total_N(s)) / (1 + N(s, a))
            \_________________/   \________________________________________/
            exploitation:          exploration:
            "how good was this     "try actions the network likes (high P)
             action in the past?"   and that haven't been tried much (low N)"
```

Where:
- `Q_normalized` = Q(s,a) scaled to [0,1] using the global min/max Q seen during this search
- `c` = 1.5 (exploration constant)
- `P(a|s)` = the neural network's prior probability for action a
- `total_N(s)` = total visits to state s
- `N(s,a)` = visits to action a from state s

**Critical insight**: Early in the search, N(s,a) is small, so the exploration term dominates. The network's policy P(a|s) determines which actions get explored first. As the search progresses, N(s,a) grows, the exploration term shrinks, and the exploitation term Q(s,a) takes over.

#### Phase 2: EXPAND — Reach a new state, query the neural network

When SELECT reaches a state not yet in the tree:

```
policy, value = neural_network.predict(new_state)
create new node with nn_policy=policy, nn_value=value
return value  ← this is the "leaf evaluation"
```

The value returned here is the neural network's estimate of how good this state is. It is propagated back up the tree.

#### Phase 3: BACKUP — Propagate the value back up the path

Walk back up the path from leaf to root, updating each edge:

```
For each (state, action) pair on the path from leaf to root:
    Q(s, a) = R(s, a) + gamma * V(s')     # Bellman equation
    N(s, a) += 1
```

Where R(s,a) is the immediate reward from taking action a in state s, and V(s') is the value returned from the child.

### 3.3 Concrete example: 4 simulations on D=3

Starting state: at location 0, room 0 unlocked, both keys available.

```
ROOT: obs = [1,0,0,0,0,0, 1,0,0, 1,1]
Network says: P = {MOVE(1):0.6, MOVE(0):0.1, NOOP:0.3}, V = -0.2
Legal actions: MOVE(0), MOVE(1), NOOP  (only room 0 locations + noop)
```

**Simulation 1** — SELECT picks MOVE(1) (highest UCB because P=0.6 and N=0):
```
ROOT → take MOVE(1) → NEW STATE: at loc 1, room 0
    This state is not in the tree yet → EXPAND:
    Network says: P = {PICK(0):0.7, MOVE(0):0.2, ...}, V = +0.3
    BACKUP: Q(ROOT, MOVE(1)) = -0.01 + 1.0 * 0.3 = +0.29
            N(ROOT, MOVE(1)) = 1
```

**Simulation 2** — SELECT picks NOOP (P=0.3, N=0, so UCB is high):
```
ROOT → take NOOP → NEW STATE: still at loc 0 (but different step count)
    EXPAND: Network says: V = -0.3
    BACKUP: Q(ROOT, NOOP) = -0.01 + 1.0 * (-0.3) = -0.31
            N(ROOT, NOOP) = 1
```

**Simulation 3** — SELECT picks MOVE(1) again (Q=+0.29 is high, exploration for MOVE(0) not enough to beat it):
```
ROOT → take MOVE(1) → EXISTING NODE at loc 1
    This node IS in the tree → keep SELECTING within it
    SELECT picks PICK(0) (P=0.7, N=0)
    ROOT → MOVE(1) → PICK(0) → NEW STATE: at loc 1, key 0 consumed, room 1 unlocked
        EXPAND: Network says: V = +0.6
        BACKUP (walking back up):
            Q(loc1, PICK(0)) = (-0.01 + 0.10) + 1.0 * 0.6 = +0.69
                                ^^^^^^^^^^^^^^^^
                                step penalty + unlock bonus
            Q(ROOT, MOVE(1)) = average of [+0.29, (-0.01 + 0.69)] = +0.485
            N(ROOT, MOVE(1)) = 2
```

**Simulation 4** — SELECT picks MOVE(0) (only unexplored action left):
```
ROOT → take MOVE(0) → NEW STATE: still at loc 0 (moved to same room)
    EXPAND: V = -0.25
    BACKUP: Q(ROOT, MOVE(0)) = -0.01 + (-0.25) = -0.26
```

**After 4 simulations**, visit counts are:
```
N(MOVE(0)) = 1, N(MOVE(1)) = 2, N(NOOP) = 1
```

After 120 simulations, MOVE(1) would dominate because the search discovered that MOVE(1) → PICK(0) leads to unlocking room 1 (high value). The visit counts become the action probabilities.

### 3.4 Key question: "Where does each simulation start?"

**Every simulation starts from the root.** The root is the agent's CURRENT real game state (the state it needs to make a decision for). Here is what happens mechanically:

1. Before the search begins, the game state is **stashed** (saved).
2. Each simulation walks down the tree by actually stepping the game forward (calling `game.step()`).
3. After each simulation completes, the game state is **restored** to the stash — back to the root state.
4. The next simulation starts from the root again.

```python
# From mcts.py, perform_simulations():
for i in range(120):
    old_game_state = self.game.stash_state()    # save root state
    self.search(...)                             # walk down tree (modifies game state)
    self.game = self.game.unstash_state(old_game_state)  # restore root state
```

The tree persists across simulations — nodes created in simulation 1 are reused in simulation 2. This is how the tree grows: each simulation adds one new leaf node while reusing the existing tree structure above it.

**After all 120 simulations**, the tree might be 5-10 levels deep along the most-explored branches. The agent then picks the action with the most visits from the root, takes that ONE action in the real game, discards the tree, and starts a fresh MCTS search for the next decision.

### 3.5 Why MCTS is powerful

Without MCTS, the neural network makes a single forward pass and picks an action. If the network is uncertain or wrong, there's no recovery.

With MCTS, even a mediocre network can make good decisions:
- The network's policy P(a|s) tells MCTS which actions to try first (but MCTS will eventually try others)
- The network's value V(s) tells MCTS how to evaluate leaf states (but MCTS combines many evaluations)
- MCTS effectively does a **multi-step lookahead**: it discovers that MOVE(1) → PICK(0) → MOVE(2) → PICK(1) → MOVE(5) is valuable even if the network's single-step policy is unsure

---

## 4. Ablation Study: What Component Matters?

The ablation study trains the full AlphaZero system normally, then at evaluation time, disables one component to measure its individual contribution.

**Important**: Training always uses the full system (policy + value + MCTS). Only the evaluation changes. This isolates the question: "Given a trained network, which components are needed to actually solve problems?"

### 4.1 Mode: `full` (baseline)

**What happens**: Standard AlphaZero evaluation. MCTS runs 120 simulations using both the learned policy and learned value.

**Code path**:
```
eval_ablations.py:108 → _eval_loop(agent, n_episodes)
    → agent.policy(game)
        → MCTS.perform_simulations()  [120 sims, full network]
            → for each simulation:
                SELECT using UCB with learned P(a|s)
                EXPAND using learned V(s)
                BACKUP
        → return visit_count_probabilities
    → pick argmax action
```

**Concrete D=3 example**:

Step 0: State = at loc 0, room 0 open.
```
MCTS runs 120 simulations. Network says P(MOVE(1)) = 0.6.
MCTS explores MOVE(1) → PICK(0) → MOVE(2) → ... deeply.
Visit counts: MOVE(1)=95, MOVE(0)=12, NOOP=13
Action chosen: MOVE(1)
```

Step 1: State = at loc 1, key 0 here.
```
MCTS runs 120 simulations. Network says P(PICK(0)) = 0.8.
MCTS confirms picking the key leads to room 1 → room 2 → goal.
Visit counts: PICK(0)=110, MOVE(0)=5, NOOP=5
Action chosen: PICK(0)
```

This continues for 5 steps → SOLVED.

### 4.2 Mode: `policy-only` (no MCTS, raw network output)

**What happens**: Skip MCTS entirely. Just query the neural network once and use its raw policy output.

**Code path**:
```
eval_ablations.py:111-119
    → set n_simulations = -1  (special flag: no search)
    → mcts.py:107-109: if n_simulations < 0, just call net.predict()
    → return raw network policy (no search tree, no lookahead)
    → pick argmax action
```

**What's disabled**: No tree search. No lookahead. No visit count aggregation. The network's single forward pass IS the entire decision.

**Concrete D=3 example** (same trained network as above):

Step 0: State = at loc 0, room 0 open.
```
net.predict(obs) → policy = [0.08, 0.60, 0.02, 0.01, 0.01, 0.01, 0.24, 0.02, 0.01]
                              M(0)  M(1)  M(2)  M(3)  M(4)  M(5)  P(0)  P(1)  NOOP
Mask to legal actions: [0.08, 0.60, 0, 0, 0, 0, 0, 0, 0.01] → normalize
Action chosen: MOVE(1) ← same as full, because the network learned this well
```

Step 1: State = at loc 1, key 0 here.
```
net.predict(obs) → policy says PICK(0) = 0.80
Action chosen: PICK(0) ← correct
```

**When policy-only fails** (D=14, deeper in the episode):

Step 12: State = at loc 22, room 7, keys 0-6 collected.
```
net.predict(obs) → policy = [0.02, 0.02, ..., 0.04, 0.03, 0.04, ...]
                              all actions have similar probability (network is uncertain)
argmax might pick MOVE(18) when the correct action is MOVE(22) → PICK(7)
```

With MCTS, 120 simulations would explore both options several steps deep and discover which one leads to progress. Without MCTS, a near-uniform policy output is essentially a coin flip.

### 4.3 Mode: `value-only-1step` (no policy, no MCTS, greedy 1-step lookahead)

**What happens**: For each legal action, simulate taking it one step, then ask the network "how good is the resulting state?" Pick the action that leads to the best-valued next state.

**Code path**:
```
eval_ablations.py:121-128 → value_only_1step_policy()
    network_ablations.py:113-152:
        for each legal action a:
            clone the game
            take action a → get reward R, new state s'
            if terminal: Q(a) = R
            else: Q(a) = R + gamma * V(s')   ← network's value estimate
        return one-hot on argmax Q
```

**What's disabled**: No policy head (ignored entirely). No tree search. Only 1-step lookahead (not 120-simulation deep search). The value head alone must correctly rank which next-state is better.

**Concrete D=3 example**:

Step 0: State = at loc 0, room 0 open. Legal actions: MOVE(0), MOVE(1), NOOP.

```
Action MOVE(0): clone → step → still at loc 0 (same room)
    R = -0.01, V(new_state) = -0.20
    Q(MOVE(0)) = -0.01 + 1.0 * (-0.20) = -0.21

Action MOVE(1): clone → step → at loc 1 (where key 0 is)
    R = -0.01, V(new_state) = +0.45
    Q(MOVE(1)) = -0.01 + 1.0 * 0.45 = +0.44

Action NOOP: clone → step → still at loc 0
    R = -0.01, V(new_state) = -0.22
    Q(NOOP) = -0.01 + 1.0 * (-0.22) = -0.23

Best action: MOVE(1) (Q = +0.44) ← correct!
```

**When value-only-1step fails** (D=14, mid-episode):

Step 15: State = at loc 25, room 8. Legal: MOVE to many locations in rooms 0-8.
```
Action MOVE(22): Q = -0.01 + V(at loc 22) = -0.01 + 0.38 = +0.37
Action MOVE(25): Q = -0.01 + V(at loc 25) = -0.01 + 0.40 = +0.39  ← network thinks this is better
Action MOVE(24): Q = -0.01 + V(at loc 24) = -0.01 + 0.36 = +0.35

Chosen: MOVE(25), but the correct move was MOVE(22) → PICK(8)
```

The value network thinks "being at loc 25" is slightly better than "being at loc 22", but it can't see that loc 22 has key 8 which unlocks room 9. Only a deeper search (MCTS) would discover this 2-step dependency.

### 4.4 Summary: What each mode tests

```
                    Uses policy?    Uses value?    Uses MCTS search?
full                YES             YES            YES (120 sims)
policy-only         YES             NO             NO
value-only-1step    NO              YES            NO (1-step only)
uniform-prior       NO (uniform)    YES            YES (120 sims)
zero-value          YES             NO (always 0)  YES (120 sims)
unguided-search     NO (uniform)    NO (always 0)  YES (120 sims)
random-net          NO (random)     NO (random)    YES (120 sims)
```

### 4.5 What the results tell you

| Comparison | If full >> ablated | Meaning |
|---|---|---|
| full vs policy-only | MCTS search is critical | The policy alone can't solve the problem; multi-step lookahead is needed |
| full vs value-only-1step | Multi-step search is critical | 1-step greedy lookahead isn't enough; the sequential key dependencies require planning |
| full vs uniform-prior | Learned policy matters | Without a good prior, MCTS wastes simulations exploring bad actions |
| full vs zero-value | Learned value matters | Without value estimates at leaves, MCTS can't evaluate which branches are promising |
| full vs unguided-search | Learning matters for search | Raw MCTS with no learned guidance can't solve the problem |
| full vs random-net | Training matters | An untrained network can't guide MCTS effectively |

The Doors problem specifically tests **sequential planning under dependency constraints**. We expect:
- `full` solves reliably (the system works)
- `policy-only` degrades with D (network uncertainty grows with problem size)
- `value-only-1step` fails for large D (can't see multi-step dependencies)
- `uniform-prior` degrades (MCTS wastes its 120 sim budget exploring 56 actions with no guidance)
- `zero-value` may partially work (policy guides exploration, but backup has no signal)
- `unguided-search` and `random-net` should fail for large D (no learned guidance at all)

---

## 5. Reference: Key Code Locations

| Component | File | Key lines |
|---|---|---|
| Doors environment | `src/alphazeropp/instances/doors/doors_pddl_lite.py` | `step()` at line 235, `reset()` at line 215 |
| Game wrapper | `src/alphazeropp/instances/doors/game.py` | `get_action_mask()` at line 50 |
| MCTS search | `src/alphazeropp/core/mcts.py` | `perform_simulations()` at line 91, `search()` at line 303 |
| UCB calculation | `src/alphazeropp/core/mcts.py` | `calc_masked_ucbs()` at line 400 |
| Network wrappers | `src/alphazeropp/instances/doors/network_ablations.py` | All wrappers |
| Ablation dispatch | `src/alphazeropp/instances/doors/eval_ablations.py` | `compute_solve_rate_ablated()` at line 88 |
| Training loop | `scripts/run_doors_direct.py` | `_run_single_seed()` at line 1130 |
| Sweep runner | `scripts/run_ablation_sweep.py` | `main()` at line 100 |
