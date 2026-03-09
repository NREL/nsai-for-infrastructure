
## System Description

We use **AlphaZero (MCTS + Transformer neural network)** for **grammar-guided program synthesis**. The system derives programs (reactive policies) for a navigation/key-pickup task ("Doors") by expanding an abstract syntax tree (AST) one grammar production at a time. At each derivation step, MCTS guided by a Transformer selects the next production. Complete programs are evaluated on a frozen environment to produce a scalar reward.

The system follows the standard AlphaZero loop:
1. **Self-play**: N games, each using MCTS with the current neural network to derive a complete program
2. **Train**: Update Transformer on (partial_AST, MCTS_visit_distribution, backed_up_reward) tuples
3. **Evaluate**: Pit new vs old network on fresh games; accept if win rate >= threshold
4. **Repeat** for K iterations

### The Doors Domain

The Doors environment is a deterministic navigation/key-pickup task:
- **D rooms**, each with `locs_per_room` locations (default 2)
- **M = D x locs_per_room** total locations, **K = D-1** keys
- Agent starts in room 0 (unlocked). All other rooms are locked.
- Key k is at fixed location `k * locs_per_room + 1` (2nd location of room k) and unlocks room k+1
- Goal: reach the last room's location

**Example (D=3, locs_per_room=2):**
```
Room 0: locations [0, 1]  -- Key 0 at loc 1, unlocks Room 1
Room 1: locations [2, 3]  -- Key 1 at loc 3, unlocks Room 2
Room 2: locations [4, 5]  -- [GOAL]
```
Optimal play: move to loc 1, pick key 0, move to loc 3, pick key 1, move to loc 5 (goal). 5 steps.

**Observation vector** (size M + 2D - 1 = 11 for D=3):
```
Indices [0..M-1]:        Agent location (one-hot)
Indices [M..M+D-1]:      Room unlock status (1=unlocked)
Indices [M+D..M+2D-2]:   Key availability (1=still available, 0=picked up)
```

**Actions**: M MOVE actions + K PICK actions + 1 NOOP = M+K+1 total.

### The Grammar and Derivation Game

Programs are built from a context-free grammar over this AST:
```
Program  ::= Ite(Cond, Action, Program)   -- if-then-else chain
           | Default(Action)               -- terminal action
Cond     ::= IsZero(j)                    -- test obs[j] == 0
           | And(Cond, Cond)              -- conjunction
Action   ::= Flip(j)                      -- execute action j
```

`Flip(j)` maps to: j < M means MOVE_TO(loc j), M <= j < M+K means PICK(key j-M), j = M+K means NOOP.

`IsZero(j)` tests whether observation index j is zero:
- j < M: "agent is NOT at location j"
- j >= M, j < M+D: "room j-M is LOCKED"
- j >= M+D: "key j-M-D has BEEN PICKED UP"

Each derivation step expands one hole in the partial AST by choosing a grammar production. The derivation game is an MDP where states are partial ASTs and actions are productions.

**Example — the optimal D=3 program** (found by successful run, iter 50, 100% solve rate):
```python
if And(Not(IsZero(1)), Not(IsZero(9))):    # at loc 1 AND key 0 available
  Flip(6)                                    # PICK(key 0)
elif IsZero(7):                             # room 1 locked
  Flip(1)                                    # MOVE to loc 1 (go get key 0)
elif And(Not(IsZero(3)), Not(IsZero(10))):  # at loc 3 AND key 1 available
  Flip(7)                                    # PICK(key 1)
elif IsZero(8):                             # room 2 locked
  Flip(3)                                    # MOVE to loc 3 (go get key 1)
else:
  Flip(5)                                    # MOVE to loc 5 (goal)
```

### MCTS Mechanics

At each derivation step, MCTS runs `n_simulations` rollouts:

1. **Selection**: Traverse tree using UCB: `UCB(a) = Q_norm(a) + c * P(a) * sqrt(N_total) / (1 + N(a))`
   - Q-normalization: `Q_norm = (Q - Q_min) / (Q_max - Q_min)` using global min/max across the tree
2. **Expansion**: At a new leaf node, query the neural network for `(policy, value)`
3. **Backpropagation**: Back up the value through the tree, updating Q-values as running averages

**Two types of leaf nodes:**
- **Non-terminal** (partial program with holes): returns the **network's predicted value** — no program execution
- **Terminal** (complete program): runs the program on frozen environment states via `LeafEvaluator`, returns **actual execution reward**

The value network is initialized with **random PyTorch weights** (random policy prior, random value ~0).

### Reward Calculation

Programs are evaluated on frozen initial states. The raw reward combines:
- `+1.0` for reaching the goal
- `+unlock_bonus` (0.1) per room unlocked
- `-step_penalty` (0.01) per step taken
- Horizon: 35 steps max (D=4) or 25 steps max (D=3)

The `weighted` metric (used in experiments) is: `alpha * solve_rate + (1 - alpha) * avg_reward`, with alpha=0.7.

**Concrete reward tiers for D=4 (horizon=35):**

| Program behavior | Raw reward | Weighted (alpha=0.7) |
|---|---|---|
| Wanders randomly, no keys | -35 x 0.01 = -0.35 | 0.3 x (-0.35) = **-0.105** |
| Picks up 1 key | 0.1 - 0.35 = -0.25 | 0.3 x (-0.25) = **-0.075** |
| Picks up 2 keys | 0.2 - 0.35 = -0.15 | 0.3 x (-0.15) = **-0.045** |
| Picks up 3 keys | 0.3 - 0.35 = -0.05 | 0.3 x (-0.05) = **-0.015** |
| Solves (all keys + goal) | 1.0 + 0.3 - 7x0.01 = 1.23 | 0.7 + 0.3 x 1.23 = **+1.069** |

The gap between "0 keys" and "1 key" is only **0.03** in weighted reward. The gap between "0 keys" and "solved" is **1.17**. This creates a reward landscape that is flat desert with rare spikes.

## Experimental Evidence

### Experiment 1: Grammar Suite (D=3, systematic benchmark)

- 13 runs: 4 grammar modes x 3 seeds + 1 D=5 run
- Config: 100 MCTS sims, 15 games/iter, 20 iterations, metric=avg_reward
- **Result: 0% solve rate across ALL 13 runs, all converge to avg_reward = -0.15**

**Value network trajectory (flat_and_D3_seed42):**
```
iter  1: val_loss=0.045620  pol_loss=5.070  avg_rew=-0.0750
iter  5: val_loss=0.005449  pol_loss=3.692  avg_rew=-0.0735
iter 10: val_loss=0.003183  pol_loss=3.512  avg_rew=-0.0750
iter 15: val_loss=0.000547  pol_loss=3.509  avg_rew=-0.0750
iter 20: val_loss=0.000028  pol_loss=3.480  avg_rew=-0.0735  <-- value collapsed
```

Value loss monotonically drops to ~0: the network learns to output a constant for all inputs.

### Experiment 2: D=4 Macro Run (more compute, still fails)

- Config: 150 MCTS sims, 80 games/iter, 17 iterations, metric=weighted, factored+macro grammar
- **Result: 0% solve rate, best program unchanged across all 17 iterations**

**Leaf value distribution across 1,440 complete episodes:**

| Leaf value | Count | % | Meaning |
|---|---|---|---|
| -0.105 | 1,286 | 89.3% | No keys picked |
| -0.075 | 154 | 10.7% | Picked 1 key |

Only **2 distinct reward values** across 175,384 unique programs. No program ever found 2+ keys.

**Best program (unchanged for all 17 iterations):**
```python
if And(Not(IsZero(1)), Not(IsZero(12))):  # at key 0 loc AND key 0 available
  Flip(8)                                   # PICK(key 0)
else:
  Flip(1)                                   # MOVE to key 0 loc
```
This picks up only key 0 — handles 1 of 3 required keys. The system never discovers a 2-key program.

### Experiment 3: D=3 Macro Run (succeeds)

- Config: 200 MCTS sims, 80 games/iter, 50 iterations, metric=weighted, factored+macro grammar
- **Result: 100% solve rate by iteration 20, optimal program found**

**Value network trajectory:**
```
iter  1: val_loss=0.0412  avg_rew=-0.058   <-- 4 distinct leaf values from start
iter 11: val_loss=0.0158  avg_rew=-0.022
iter 16: val_loss=0.0315  avg_rew= 0.114   <-- breakthrough: solving programs found
iter 21: val_loss=0.0689  avg_rew= 0.849
iter 36: val_loss=0.0659  avg_rew= 0.990   <-- near-optimal
iter 50: val_loss=0.0679  avg_rew= 0.976
```

**Leaf value distribution (D=3 success):**

| Leaf value | Count | % |
|---|---|---|
| -0.075 | 1,267 | 31.7% |
| -0.045 | 199 | 5.0% |
| -0.015 | 116 | 2.9% |
| +1.045 | 2,379 | **59.5%** |

The critical difference: MCTS stumbled onto **solving programs** early, creating a +1.045 reward spike that broke the flat landscape. Value std jumps from 0.01 to 0.44, giving the value network meaningful signal.

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Domain | Doors: D rooms, navigate + pickup keys |
| Observation | M + 2D - 1 floats (location one-hot + room locks + key availability) |
| Action space | 540 productions (flat grammar, D=4) or 15 (factored+macro) |
| Derivation length | 2-59 steps (mean ~17 for D=4, ~6 for D=3) |
| Program space | ~10^12+ reachable programs (budget=34 for D=3, budget=48 for D=4) |
| Network | Transformer: d=64, 2 layers, 4 heads, ~100K params |
| MCTS sims | 100-200 per derivation step |

## The Root Cause: Reward Desert

### 1. Near-zero reward variance from random programs

99% of randomly synthesized programs score identically (-0.105 for D=4, -0.075 for D=3). Unlike Go where random play produces a 50/50 split of wins and losses, random program derivation produces a unimodal reward distribution with near-zero variance.

The MSE-optimal value prediction is a constant:
```
V*(any_partial_AST) = E[reward] ≈ -0.105
MSE of constant prediction ≈ 0.00002 (near-zero — already optimal)
```

### 2. Value network collapse is mathematically correct

The value network is not broken — it is **correctly identifying** that the reward landscape is flat. When all training targets cluster at the same value, the optimal MSE predictor IS a constant. Value loss drops to ~0.00003 because there is nothing left to learn.

### 3. Constant value predictions neutralize MCTS

When all backed-up Q-values are identical, Q-normalization (min-max) hits the degenerate case:
```python
# In mcts.py calc_masked_ucbs():
if self.q_max > self.q_min:
    q_norm = (q - q_min) / (q_max - q_min)    # normal: differentiates actions
else:
    q_norm = 0.5                                # ALL actions get Q = 0.5
```

UCB collapses to: `UCB(a) = 0.5 + c * P(a) * sqrt(N) / (1+N(a))`

The Q-component provides **zero differentiation**. Action selection depends only on the policy prior P(a) and visit-count exploration. MCTS degenerates into policy-guided breadth-first search without any value-based exploitation.

### 4. The vicious cycle vs virtuous cycle

**Program synthesis (vicious — never starts):**
```
Random derivations → all programs score ~-0.105 → value learns constant
→ Q-values identical → MCTS has no exploitation signal → random derivations → ...
```

**Go (virtuous — starts immediately):**
```
Random play → 50% wins, 50% losses → value learns position features
→ MCTS exploits value differences → better play → richer signal → ...
```

### 5. Why Go bootstraps but synthesis doesn't

The fundamental structural difference:

| Property | Go | Program Synthesis |
|---|---|---|
| Reward structure | **Competitive** (zero-sum, one player always wins) | **Absolute** (single-agent, no opponent) |
| Random play outcomes | **Bimodal** (+1/-1), variance = 1.0 | **Unimodal** (~-0.105), variance ≈ 0.00002 |
| Variance guarantee | **Structural**: someone always wins, creating differentiation | **None**: good programs are exponentially rare |
| Value network at iter 1 | Can learn "positions like X win more" | Learns "everything scores -0.105" |
| Bootstrapping | Starts immediately from game 1 | Requires lucky discovery of a qualitatively better program |

Go has a **structural guarantee of reward variance**: in any game, one player wins (+1) and one loses (-1). This bimodal distribution forces the value network to differentiate positions from the very first iteration.

Program synthesis has no such guarantee. The value network's training loss (MSE) at iteration 1:
- **Go**: targets = {+1, -1, +1, -1, ...} → MSE of constant ≈ 1.0 → strong gradient to learn
- **Synthesis**: targets = {-0.105, -0.105, -0.105, -0.075, ...} → MSE of constant ≈ 0.00002 → gradient vanishes

### 6. Gate score confirms zero learning

Evaluation pitting new vs old network: gate_score = 0.50 every iteration (coin flip). The new_rewards_std = 1.4e-17 (numerical zero). The network never improves because there is no signal to improve on.

### 7. Pure MCTS baseline reveals the role of random initialization

**Critical experiment:** We ran a pure MCTS baseline replacing the Transformer with `UniformPolicyValueNet` (uniform prior P(a)=1/N, value=0). No training loop.

| Run | Mode | Rounds × Sims | Unique Programs | Solve Rate |
|-----|------|---------------|-----------------|------------|
| Baseline (flat) | 279 actions | 400 × 100 | 109,780 | **0%** |
| Baseline (flat) | 279 actions | 80 × 200 | 77,316 | **0%** |
| Baseline (factored+macros) | 31 actions | 400 × 200 | 109,974 | **0%** |
| Successful AlphaZero iter 1 | factored+macros | 80 × 200 | 9,430 | **100%** (1/80 games) |

**The baseline explored 12× more unique programs and still found zero solvers.** The only difference is the policy prior: uniform (baseline) vs random Transformer (successful run).

This means the randomly initialized Transformer provides useful search structure even before training. Its non-uniform policy priors bias MCTS toward certain programs, and by chance one such bias leads to the optimal solver. Uniform search explores more broadly but less deeply in any particular direction.

**Implication:** The failure is not just about reward sparsity — it's also about search structure. A random Transformer's inductive bias (correlated action preferences across similar partial ASTs) creates structured exploration that uniform random search cannot replicate, regardless of compute budget.

## Code Artifacts Available

For deeper analysis, the following code files are available:
- `leaf_evaluator.py` — reward computation (runs program on environment, caches results)
- `mcts.py` — UCB search with min-max Q normalization and Dirichlet noise
- `derivation_game.py` — the MDP (state encoding, action masking, production application)
- `derivation_network.py` — Transformer architecture (CLS token + AST token embedding)
- `agent.py` — self-play game execution
- `gated_trainer.py` — pit new vs old network, accept/reject
- `doors_pddl_lite.py` — Doors environment (observation, transitions, reward)
- `doors_config.py` — key placement, room layout, derived parameters
- `interpreter.py` — program execution engine (runs synthesized policy on environment)
