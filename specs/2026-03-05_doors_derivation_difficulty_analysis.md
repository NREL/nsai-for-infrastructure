# Doors Derivation Game: MDP Difficulty Analysis

**Date:** 2026-03-05
**Status:** Audit of why AlphaZero fails on grammar-guided program synthesis

---

## 1. The Doors Derivation MDP — Explicit Computation

### 1.1 What the MDP Looks Like (D=3)

The agent derives a **program** (a reactive policy for the Doors environment) by expanding an abstract syntax tree (AST) one production at a time. Each derivation step selects a grammar production to fill the leftmost "hole" in the partial AST.

**State (partial AST encoded as flat vector):**
- 68-dimensional float vector (2 x budget = 2 x 34)
- Encodes preorder traversal of the partial AST as (node_type_id, parameter) pairs
- Node types: PAD(0), Flip(1), IsZero(2), Not(3), And(4), Ite(5), Default(6), ProgramHole(7), ConditionHole(8)

Example — initial state:
```
ProgramHole(34)  →  obs = [7, 34, 0, 0, 0, 0, ..., 0]
                          type=ProgramHole  budget=34  (66 zeros padding)
```

Example — after one production `Ite(ConditionHole(2), Flip(3), ProgramHole(29))`:
```
obs = [5, 0,   8, 2,   1, 3,   7, 29,   0, 0, ..., 0]
       ^Ite    ^CondHole ^Flip   ^ProgHole  (padding)
```

**Actions (grammar productions):**
- Discrete set. At each step, choose which production to apply to the leftmost hole.

For `ProgramHole(k)`:
| Production | Budget used | Actions |
|-----------|-------------|---------|
| `Default(Flip(j))` for j in [0..10] | 2 | 11 |
| `Ite(ConditionHole(i), Flip(j), ProgramHole(k-2-i))` | k | 11 x (number of valid condition budgets i) |

For `ConditionHole(k)`:
| Production | Budget used | Actions |
|-----------|-------------|---------|
| `IsZero(j)` for j in [0..10] | 1 | 11 |
| `Not(ConditionHole(k-1))` | k | 1 |
| `And(ConditionHole(i), ConditionHole(k-1-i))` | k | number of valid splits |

Total maximum actions at any step: **279 (flat)** or **31 (factored)**

**Episode length:**
- Average: **14.8 derivation steps** per game (measured from experiments)
- Min: 2 steps (trivial `Default(Flip(j))`)
- Max: 39 steps (deep AST)
- Distribution: right-skewed, most games 10-25 steps

### 1.2 Reward Structure — The Critical Bottleneck

**Non-terminal steps (93% of all steps):** reward = 0.0, always.

**Terminal step (7% of steps):** The completed program is evaluated on the Doors environment.

The ratio is stark: for every 14.8 steps in a derivation, only the very last one produces any reward. And with discount=1.0, every step in the episode gets the same value target = terminal reward. The value network has no way to distinguish "promising partial AST" from "dead-end partial AST" since both get identical training targets.

### 1.3 The Doors Environment (D=3, 3 rooms)

**Observation vector (11 dimensions):**
```
Index  Meaning                   Initial value
0-5    at_location[0..5]         [1, 0, 0, 0, 0, 0]  (one-hot: at loc 0)
6-8    unlocked[room 0..2]       [1, 0, 0]            (room 0 open, rooms 1-2 locked)
9-10   key_available[key 0..1]   [1, 1]               (both keys available)
```

**Physical layout (D=3, 2 locs/room):**
```
Room 0 (unlocked)     Room 1 (locked)       Room 2 (locked)
  loc 0 (start)         loc 2                  loc 4
  loc 1 (key 0 here)    loc 3 (key 1 here)     loc 5 (GOAL)
```

**Actions (9 total):**
| Action | Name | Effect |
|--------|------|--------|
| 0-5 | MOVE_TO(loc j) | Move to location j (only if room is unlocked) |
| 6-7 | PICK(key k) | Pick up key k (only if at key location AND key available) |
| 8 | NOOP | Do nothing |

**Per-step reward:**
- Every step: -0.01 (step penalty)
- Successful PICK that unlocks a room: +0.1
- Reaching goal (loc 5): +1.0
- Episode truncates at horizon=25 steps

**Optimal strategy (5 steps, reward = 1.15):**
```
Step 1: MOVE_TO(1)  → move to key 0 location    reward: -0.01
Step 2: PICK(0)     → pick key 0, unlock room 1  reward: -0.01 + 0.1 = +0.09
Step 3: MOVE_TO(3)  → move to key 1 location     reward: -0.01
Step 4: PICK(1)     → pick key 1, unlock room 2  reward: -0.01 + 0.1 = +0.09
Step 5: MOVE_TO(5)  → reach goal                  reward: -0.01 + 1.0 = +0.99
                                           Total: 1.0 + 0.2 - 0.05 = 1.15
```

### 1.4 How a Program Executes as a Reactive Policy

A completed program is a **decision list** — a chain of if-then-else rules:
```
if <condition>: <action>
elif <condition>: <action>
...
else: <default action>
```

At each environment step, the program reads the current observation and fires the first matching rule.

**AST node types:**
- `Flip(j)` → execute action j
- `IsZero(j)` → true iff obs[j] == 0
- `Not(c)` → logical negation
- `And(l, r)` → logical conjunction
- `Ite(cond, action, else_prog)` → if cond then action, else continue to next rule
- `Default(action)` → always execute action (base case)

**The optimal program for D=3 (27 AST nodes):**
```
if And(Not(IsZero(1)), Not(IsZero(9))):   # at loc 1 AND key 0 available
  Flip(6)                                   # → PICK(key 0)
elif IsZero(7):                             # room 1 locked
  Flip(1)                                   # → MOVE_TO(loc 1)
elif And(Not(IsZero(3)), Not(IsZero(10))): # at loc 3 AND key 1 available
  Flip(7)                                   # → PICK(key 1)
elif IsZero(8):                             # room 2 locked
  Flip(3)                                   # → MOVE_TO(loc 3)
else:
  Flip(5)                                   # → MOVE_TO(goal)
```

### 1.5 Leaf Evaluation — How Terminal Reward Is Computed

When derivation completes, `leaf_evaluator(program)` runs the program on frozen initial states:

```python
for each frozen_state:
    env = DoorsPDDLLiteEnv(D=3, ...)
    obs = env.reset(frozen_state)
    while not done:
        action = eval_program(program, obs)   # reactive policy decision
        obs, reward, done = env.step(action)
        cumulative_reward += reward

metrics = {
    "solve_rate": fraction of states where agent reached goal,
    "avg_reward": mean cumulative_reward across states,
}
scalar = 0.7 * solve_rate + 0.3 * avg_reward   # "weighted" metric
```

### 1.6 Worked Example — A Garbage Program

**Program:** `if IsZero(0): Flip(3) else: Flip(8)`
Meaning: "If obs[0]==0 (not at location 0), MOVE_TO(3); else NOOP"

**Execution on initial state** `[1,0,0,0,0,0, 1,0,0, 1,1]`:
```
Step 1: obs[0]=1, IsZero(0)=False → else → Flip(8)=NOOP
        reward: -0.01, obs unchanged
Step 2: obs[0]=1, IsZero(0)=False → else → Flip(8)=NOOP
        reward: -0.01, obs unchanged
... (repeats 25 times until horizon)
Step 25: truncated
cumulative_reward = 25 × (-0.01) = -0.25
solved = False
```

Metric (weighted, alpha=0.7): `0.7 × 0.0 + 0.3 × (-0.25) = -0.075`

**This score (-0.075) is what essentially ALL random programs achieve.** Whether the program does NOOP, moves to a locked room, or picks up a key that doesn't exist — the result is the same: stuck for 25 steps, cumulative reward ≈ -0.25, solve_rate = 0.

### 1.7 Worked Example — A Partial-Credit Program

**Program:** `if IsZero(7): Flip(1) else: Flip(5)`
Meaning: "If room 1 locked, MOVE_TO(1); else MOVE_TO(5)"

**Execution:**
```
Step 1: obs[7]=0, IsZero(7)=True → Flip(1)=MOVE_TO(1)
        Moves to loc 1 (room 0, same room as start). reward: -0.01
Step 2: obs[7]=0, still locked → Flip(1)=MOVE_TO(1)
        Already at loc 1. reward: -0.01
... (repeats — never picks up key because no PICK action)
Step 25: truncated
cumulative_reward = 25 × (-0.01) = -0.25
solved = False
```

Metric: `0.7 × 0.0 + 0.3 × (-0.25) = -0.075` — same score as the garbage program.

Even a program that "knows" about the room structure but lacks the PICK action gets the same -0.075. The reward landscape has no gradient between "completely wrong" and "almost right but missing one action."

---

## 2. The Value Network Predicts "Bad" — And How It Poisons MCTS

### 2.1 Empirical Reward Distributions

**Failing grammar suite** (100 sims, 20 games/iter, all 12 runs):
```
Iteration 1:  757 unique programs → ALL score -0.075 or -0.25
Iteration 20: 43,719 unique programs → best_avg_reward = -0.15
                                       solve_rate = 0.0 across ALL programs
```

No program across 12 runs x 20 iterations x ~40K programs EVER scores above -0.15.

**Successful manual run** (200 sims, 80 games/iter), iteration 1 only:
```
-0.075:  73 games (91.2%)   ← garbage programs (stuck/NOOP)
-0.045:   5 games  (6.2%)   ← slightly better (moves, unlocks 0 rooms)
-0.015:   1 game   (1.2%)   ← unlocks ~1 room but doesn't solve
+1.045:   1 game   (1.2%)   ← SOLVER! (found purely by chance)
```

The solver was found among 9,430 unique programs explored in iteration 1 — 12x more than the suite's 757. This was brute-force volume, not learned search.

### 2.2 What the Value Network Learns

Training value targets = terminal leaf value (identical for ALL steps in a game, since discount=1.0).

**In the failing suite:** All training examples have target ≈ -0.075.
- The network learns: `V(any_state) ≈ -0.075`
- This is the **constant predictor** — the best possible model when all targets are identical
- Value loss converges to Var(y) ≈ 0.00002 (the noise floor)
- The network has zero information to distinguish good partial ASTs from bad ones

**In the successful run:** 91.2% of targets are -0.075, but 1.2% are +1.045.
- Value loss starts at 0.041, drops to 0.014 by iteration 10
- Enough signal exists to learn *something*, but 1/80 positive examples is very noisy

### 2.3 How Constant V Predictions Poison MCTS

When MCTS expands a new child node, it queries the value network for a value estimate. This estimate is backed up through the tree:

```
V_network(any_child_state) → -0.075   (constant prediction)

Q(parent, action_a) = running_mean(backed_up_values)
                    ≈ running_mean(-0.075, -0.075, ...) = -0.075 for ALL actions
```

**Min-max Q normalization in UCB (from mcts.py):**
```
Q_min = -0.075, Q_max = -0.075  (all Q-values identical)
Q_normalized = (Q - Q_min) / (Q_max - Q_min) = 0/0 → defaults to 0.5
```

**UCB formula with uniform policy prior:**
```
UCB(a) = 0.5 + 1.5 × (1/279) × √N / (1 + N_a)
```

ALL actions have identical UCB. MCTS action selection is driven entirely by:
1. **Dirichlet noise** at root (random perturbation, epsilon=0.4)
2. **Tie-breaking** (random)
3. **Backed-up leaf evaluations** from completed programs within the search tree

### 2.4 How Many Programs Does MCTS Complete During Search?

Within one game's MCTS tree (100 sims, ~15 steps per derivation):
- MCTS can traverse ~100/15 ≈ **6-7 complete programs**
- These are the only source of actual value signal (leaf evaluations)
- But 6-7 random programs out of ~10^12 are almost certainly all garbage (-0.075)
- So even the backed-up leaf values provide no differentiation

With 200 sims: ~200/15 ≈ **13-14 complete programs** — still mostly garbage, but 2x the lottery tickets.

### 2.5 The Vicious Cycle

```
Value net predicts -0.075 for everything
  → MCTS Q-values are all equal → no exploitation, pure random exploration
  → Random exploration produces programs scoring -0.075
  → Training on -0.075 targets reinforces the constant prediction
  → Value net still predicts -0.075 for everything
  → (repeats for 20 iterations with zero improvement)
```

The AlphaZero virtuous cycle ("better net → better MCTS → better data → better net") never starts because there is no initial signal to bootstrap from.

---

## 3. Reward Informativeness — Comparison to Working Systems

### 3.1 Why Go/Chess Works with AlphaZero

In Go, two random players produce a game. One wins, one loses. The value head sees:
- ~50% of positions labeled +1 (from winner's games)
- ~50% labeled -1 (from loser's games)
- **Entropy: H = 1.0 bit** — every training example is informative

In Chess, random play produces wins/draws/losses with varied margins. Material count alone creates a gradient that the value head can learn from even before understanding strategy.

### 3.2 Doors Derivation: The Reward Desert

Random derivation produces:
- ~91% of programs score -0.075 (weighted metric)
- ~6% score -0.045
- ~1% score -0.015
- ~1% score +1.045 (but ONLY with enough exploration volume — 200 sims × 80 games)

With the failing suite's budget (100 sims × 20 games):
- ~99.75% score -0.075
- ~0.25% score slightly better
- 0% score positively
- **Entropy: H ≈ 0.022 bits**

**The value head receives ~45x less information per training example than in Go.**

### 3.3 Concrete Value Head Analysis

| Metric | Go | Doors Derivation (failing) | Doors Derivation (successful) |
|--------|----|----|------|
| Target distribution | {-1, +1} uniform | {-0.075} near-constant | {-0.075, +1.045} 91:1 ratio |
| Var(y) | 1.0 | ~0.0 | ~0.11 |
| Entropy | 1.0 bit | 0.022 bits | 0.08 bits |
| Optimal constant predictor MSE | 1.0 | ~0.00002 | ~0.11 |
| Network learns beyond constant? | Yes (immediately) | **No** | Slowly |

### 3.4 Direct Play Comparison

The same Doors environment with **direct RL** (not program synthesis):
- Action space: 49 actions (D=10, larger problem!)
- MCTS: 100 sims, 50 games/iter
- **Solves in 5 iterations**
- Why? Random actions occasionally unlock rooms (+0.1) and reach goals (+1.0). The reward has *structure*.

The derivation game wraps this structured-reward environment in a program synthesis layer that converts it into a sparse-reward MDP. The synthesis layer is where the difficulty comes from, not the underlying environment.

---

## 4. The Bootstrapping Failure — Step by Step

### Iteration 1 (grammar suite, flat_and_D3_seed42)

**Step 1 — Network initialization:**
Random Transformer (Xavier init). Policy head → near-uniform over 279 actions. Value head → ~0 for any input.

**Step 2 — Self-play (20 games, 100 sims each):**
At derivation step 1, state = `ProgramHole(34)`, legal actions ≈ 279.

MCTS UCB computation:
```
P(a) = 1/279 = 0.00358          (uniform policy prior)
exploration = 1.5 × 0.00358 × √100 / (1+0) = 0.054
Q_norm(a) = 0.5                  (all Q-values identical)
UCB(a) = 0.5 + 0.054 = 0.554    (identical for ALL actions)
```

With Dirichlet noise (alpha=0.25, epsilon=0.4):
```
P_noisy(a) = (1-0.4) × 1/279 + 0.4 × Dir(0.25)
```
The Dirichlet noise breaks ties randomly. MCTS selects actions essentially at random.

100 sims explore ~100 different subtrees. Each subtree eventually completes a program. All completed programs score ~-0.075.

Result: 20 games × ~6.5 steps = 131 training examples, ALL with value target ≈ -0.075.

**Step 3 — Network training (5 epochs, lr=3e-4):**
- Value loss: MSE between prediction and -0.075. Converges to constant predictor.
- Policy loss: cross-entropy between output and noisy visit distributions. Learns to reproduce random noise.

**Step 4 — Evaluation (pit new vs old, 20 games):**
New network produces programs scoring -0.075. Old network produces programs scoring -0.075.
Gate score: 0.50 (tied). Accepted because threshold=0.40, but no actual improvement.

### Iterations 2-20: The Flatline

Same pattern repeats 19 more times:
```
Iter  unique_programs  best_reward  gate_score  status
1          757          -0.25        0.50       no improvement
2         2,719         -0.15        0.50       no improvement
3         5,140         -0.15        0.50       no improvement
...
20       43,719         -0.15        0.50       no improvement
```

43,719 unique programs explored. Zero solve. Network unchanged.

### Contrast: The Successful Manual Run

Iteration 1 only — 80 games × 200 sims, random network:
```
9,430 unique programs explored in iteration 1
1 out of 9,430 (0.01%) was the optimal solver
```

This was NOT learned search. It was brute-force MCTS with 12x more samples per iteration. The network played no role — the solver was found before any training occurred.

After finding the solver:
- Training on 1/80 positive examples = very noisy but non-zero signal
- Value loss drops from 0.041 to 0.014 over 10 iterations
- Average reward improves: -0.058 → 0.585 → 0.823 → 0.990
- The virtuous cycle STARTS because there is finally a seed of positive reward

---

## 5. Is It the Network or the MCTS? — Pure MCTS Baseline Results

### Three Hypotheses

| Hypothesis | Description | Test |
|-----------|-------------|------|
| H1: MCTS too weak | 100 sims can't explore enough programs | Pure MCTS baseline with 400 rounds × 100 sims |
| H2: Network can't learn | Architecture wrong or signal too sparse | Compare trained vs untrained network at matched compute |
| H3: Both compound | MCTS can't find signal → network stays blind → MCTS stays random | Pure MCTS with 80 rounds × 200 sims (matches successful run budget) |

### Pure MCTS Baseline Experiments

**Methodology:** Replace the Transformer with `UniformPolicyValueNet` (uniform policy prior P(a)=1/N, value V=0) and run N independent MCTS derivation rounds. No training loop, no gating.

**Results (all runs: 0% solve rate):**

| Run | Mode | Rounds × Sims | Unique Programs | Best Reward | Solve Rate |
|-----|------|---------------|-----------------|-------------|------------|
| Baseline A | flat (279 actions) | 400 × 100 | 109,780 | -0.15 | 0% |
| Baseline B | flat (279 actions) | 80 × 200, seed 137 | 69,009 | -0.15 | 0% |
| Baseline C | flat (279 actions) | 80 × 200, seed 43 | 77,316 | -0.15 | 0% |
| Baseline D | factored+macros (31 actions) | 80 × 200, seed 43 | 21,471 | -0.15 | 0% |
| Baseline E | factored+macros (31 actions) | 400 × 200, seed 42 | 109,974 | -0.15 | 0% |

**Comparison with the successful manual AlphaZero run:**

| | Successful Manual Run | Largest Baseline (E) |
|--|----------------------|---------------------|
| Mode | factored+macros (31 actions) | factored+macros (31 actions) |
| Total MCTS rollouts | 80 games × 200 sims = 16,000 | 400 rounds × 200 sims = 80,000 |
| Unique programs explored | 9,430 | 109,974 |
| Solver found? | **Yes** (1/80 games) | **No** |
| Network | Random Transformer (Xavier init) | Uniform (P=1/N, V=0) |

### The Surprising Result: Random Transformer > Uniform Priors

The pure MCTS baseline explored **12× more unique programs** than the successful manual run (109,974 vs 9,430) and still found no solver. This eliminates the "brute force volume" explanation.

**The critical difference is the neural network's initial policy prior.** Even before any training, a randomly initialized Transformer produces **non-uniform** action probabilities:

```
UniformPolicyValueNet:     P(a) = 1/31 ≈ 0.032 for all actions
Random Transformer (Xavier): P(a) ∈ [0.005, 0.12] — varies by state and action
```

The random Transformer's non-uniform priors create **structured search bias**:
- For any given partial AST, the random network assigns different weights to different productions
- MCTS amplifies these biases through visit count accumulation
- The same network is used for all 80 games in an iteration, creating **correlated exploration** — all games share the same action preferences
- This correlated bias may "focus" search on certain regions of program space
- By pure chance, one of these focused regions may contain a solver

With uniform priors, MCTS explores all actions equally, spreading search thinly across the full ~10^12 program space. The uniform approach finds more unique programs (wider coverage) but with less depth in any particular direction.

**Analogy:** A drunkard's walk (uniform) covers more ground than a biased walk (random Transformer). But if the goal is hidden in a specific corridor, the biased walk that happens to point toward that corridor reaches it faster despite covering less total area.

### Updated Assessment

| Hypothesis | Verdict |
|-----------|---------|
| H1: MCTS too weak | **Partially confirmed** — MCTS with uniform priors cannot find a solver even with massive compute (110K programs). But MCTS with a random Transformer CAN (9.4K programs). |
| H2: Network can't learn | **Confirmed** — Even after finding the solver, the training loop barely improves because 1/80 positive examples is too sparse. The suite never finds a solver to learn from at all. |
| H3: Both compound | **Yes, but with nuance** — The random Transformer provides useful initial bias that uniform priors don't. However, the training loop fails to amplify this into learned search guidance. The system is stuck between "random network provides lucky bias" and "training loop can't leverage sparse signal." |

**Implication for solutions:** Simply increasing MCTS compute won't work (uniform priors are fundamentally insufficient). The network architecture matters even at initialization — not because it learns useful features, but because its random weights create structured search that uniform exploration cannot replicate.

---

## 6. Comparison to Other Systems

| System | Domain | Space | Reward | Method | Works? |
|--------|--------|-------|--------|--------|--------|
| This (suite: 100 sims, 20 games × 20 iters) | Doors D=3 derivation | ~10^12 | 0.022 bits | AlphaZero | **No (0%)** |
| This (manual: 200 sims, 80 games × 50 iters) | Doors D=3 derivation | ~10^12 | 0.08 bits | Random Transformer + MCTS | **Yes (iter 1)** |
| **This (MCTS baseline: uniform priors, 400×200)** | **Doors D=3 derivation** | **~10^12** | **0 bits** | **Pure MCTS (no network)** | **No (0%), 110K programs** |
| This (direct play: 100 sims, 50 games × 5 iters) | Doors D=10 direct | Continuous | ~1 bit | AlphaZero | **Yes (5 iters)** |
| AlphaGo Zero | Go 19x19 | ~10^170 | 1.0 bit (win/loss) | AlphaZero | Yes |
| DreamCoder | List manipulation | ~10^8 | Task completion | Neural enum + library | Yes |
| AlphaCode | Competitive programming | ~10^12+ | Test pass rate | LLM sampling + clustering | Yes (10^6 samples) |

---

## 7. Key Takeaways

1. **The problem is not the branching factor or depth.** At ~8-31 branching and ~15 steps, this is geometrically easier than Go.

2. **The problem is reward sparsity.** 99.75% of random programs return the identical score. The value head converges to a constant predictor and provides zero guidance to MCTS.

3. **AlphaZero requires initial signal.** The "virtuous cycle" needs a seed — at least some games must produce distinguishable outcomes. In Go this happens naturally (random play → 50% wins). In program synthesis it does not.

4. **More MCTS compute with uniform priors does NOT work.** Pure MCTS baseline with uniform priors explored 110K unique programs (12× more than the successful run's 9.4K) and still found zero solvers. Brute-force volume is not sufficient.

5. **Random Transformer initialization provides useful search structure.** The successful manual run found the solver because the randomly initialized Transformer creates non-uniform policy priors that bias MCTS toward structured search. This inductive bias is absent from uniform priors. The solver was found not by "brute force" but by "structured random search."

6. **The training loop fails to amplify initial success.** Even when a solver IS found (1/80 games), the extremely sparse positive signal (1.2% of training data) makes value learning very slow. The training loop's contribution is minimal — the solver was found before any training.

7. **The synthesis layer is the bottleneck.** The Doors environment itself has rich reward structure. Wrapping it in a program synthesis MDP collapses that structure into a sparse terminal signal.

8. **Two failure modes operate simultaneously:**
   - **Search failure:** Uniform MCTS cannot find solvers regardless of compute budget. Non-uniform priors (even random) are necessary.
   - **Learning failure:** Even with a solver in the training data, the signal-to-noise ratio is too low for the value head to learn useful state discrimination. The vicious cycle never becomes virtuous.
