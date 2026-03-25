# Doors Program Synthesis: Full System Context for Debugging

**Date:** 2026-03-20
**Status:** Reference (for LLM-assisted diagnosis)
**Intended reader:** An LLM assistant being asked to diagnose why the system fails at D=3 and suggest improvements. This document is self-contained — no external context is needed.
**Goal:** Self-contained document describing the entire Doors program synthesis system — the game, the grammar, the training loop, the reward pipeline, known failure modes, and experimental results — so an LLM can reason about what is broken and propose improvements.

**Contents:**
1. The Problem
2. AlphaZero Primer
3. The Doors Environment
4. The Grammar (DSL)
5. The Four Derivation Modes
6. The Training Pipeline
7. Current Hyperparameters
8. What Has Been Tried and What Failed
9. The Optimal Program (D=3)
10. D=2 as a Debugging Testbed
11. Known Structural Issues
12. Project Structure and Codebase Map
13. Diagnosis Summary and Open Questions

---

## 1. The Problem

We want to **synthesize a program** (a reactive if-then-else policy) that solves a navigation task. The program reads observations and outputs actions. It must generalize: one program must solve all starting states.

The system uses **AlphaZero** (MCTS + neural network; see Section 2 for a primer) to search the space of programs by treating program construction as a game: each "move" expands the program's AST by one grammar production. The terminal reward is the quality of the completed program when run on the actual environment.

**D=2 (2 rooms) is reliably solved.** The factored+macro grammar finds a solver in iteration 1 (8,460 programs). Even flat grammar solves D=2 by iteration 7-8 (~5K programs). **D=3 (3 rooms) is unreliable.** With factored+macros, D=3 solves in some runs (found at iteration 1 via brute-force exploration) but fails in others despite exploring 200K+ programs. Flat grammar never solves D=3. The gap between D=2 and D=3 is where the system breaks.

---

## 2. AlphaZero Primer

### 2.1 What AlphaZero Is

AlphaZero is a reinforcement learning algorithm originally developed for two-player board games (Go, Chess, Shogi). It combines two components:

1. **A neural network** with two output heads:
   - **Policy head:** predicts a probability distribution over legal actions (which move to play)
   - **Value head:** predicts the expected outcome from the current position (a scalar)

2. **Monte Carlo Tree Search (MCTS):** a lookahead search that uses the network's outputs to guide exploration. At each node, MCTS selects actions using UCB (Upper Confidence Bound), which balances exploitation (high Q-value actions) with exploration (under-visited actions weighted by the policy prior). When MCTS reaches an unexpanded leaf, the network evaluates it, providing both a policy prior for the leaf's children and a value estimate for backpropagation.

**Training loop:** Play games using MCTS+network → collect (state, MCTS_policy, outcome) triples → train the network to predict the MCTS policy and game outcome from the state → repeat. MCTS improves the policy beyond what the raw network would suggest; training the network on MCTS outputs gradually improves both.

### 2.2 How Program Synthesis Maps to AlphaZero

In our system, the "game" is **program construction** — building an AST node by node:

| Board Game Concept | Program Synthesis Equivalent |
|---|---|
| Game state | Partial AST (some nodes filled, some holes remaining) |
| Legal move | Grammar production to fill the leftmost hole |
| Game over | AST has no remaining holes — a complete program |
| Outcome/reward | Quality of the completed program when executed on the target environment |
| Policy head learns | "Given this partial AST, which production should I expand next?" |
| Value head learns | "Given this partial AST, how good will the final program be?" |

This is a **single-player** variant — there is no opponent. The "game" is program construction against a fixed environment.

### 2.3 Why This Mapping Is Challenging

AlphaZero was designed for domains with four properties that program synthesis violates:

| Property | Board Games | Program Synthesis |
|---|---|---|
| **Intermediate value** | Board positions have intrinsic value (who is winning) | Partial ASTs have almost no predictable value — quality depends entirely on unfilled holes |
| **Curriculum from opponent** | Weak opponents early, strong opponents later | No opponent — problem difficulty is fixed from the start |
| **Reward variance** | Win/loss/draw provides real signal | 99% of programs score identically (~-0.075) |
| **Branching structure** | Narrow critical moves amid moderate branching | High branching (39-279 actions) at every step with no narrowing |

---

## 3. The Doors Environment

### 3.1 Physical Layout (D=3 example)

```
Room 0 (unlocked)     Room 1 (locked)       Room 2 (locked)
  loc 0 (start)         loc 2                  loc 4
  loc 1 [Key 0]         loc 3 [Key 1]          loc 5 (GOAL)
```

- D rooms, each with `locs_per_room` locations (default 2).
- D-1 keys. Key k sits at a fixed location and unlocks room k+1.
- Agent starts at loc 0. Goal: reach the last location.

### 3.2 Observation Vector

Size: `n_sites = M + 2D - 1` where `M = D * locs_per_room`.

For D=3: M=6, n_sites=11.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | at_loc0 | at_loc1 | at_loc2 | at_loc3 | at_loc4 | at_loc5 | room0_open | room1_open | room2_open | key0_avail | key1_avail |

Initial state: `[1,0,0,0,0,0, 1,0,0, 1,1]` — at loc 0, room 0 open, both keys available.

Indices 0-5 are **one-hot** (agent at exactly one location). Room/key flags are independent.

### 3.3 Actions

`n_actions = M + K + 1` (D=3: 9 actions).

| Index | 0-5 | 6 | 7 | 8 |
|---|---|---|---|---|
| Meaning | MOVE(loc 0-5) | PICK(key 0) | PICK(key 1) | NOOP |

**Preconditions:**
- MOVE(loc): target room must be unlocked.
- PICK(key k): agent must be at key k's location AND key k must be available.
- Failed preconditions: action becomes NOOP (step penalty still applies).

### 3.4 Rewards

```
Every step:            -0.01  (step_penalty)
PICK that unlocks:     +0.10  (unlock_bonus)
Reach goal:            +1.00
Truncation at horizon: episode ends (horizon = max(15, 5 * optimal_steps))
```

For D=3: horizon=25, optimal_steps=5, optimal_reward = 1.0 + 2(0.1) - 5(0.01) = **1.15**.

### 3.5 Reward Landscape by Program Quality (D=3)

| Keys picked | Env reward (over 25 steps) | Weighted metric (alpha=0.7) |
|---|---|---|
| 0 (stuck/NOOP) | 0 - 25(0.01) = **-0.25** | 0.3(-0.25) = **-0.075** |
| 0 (moves but no picks) | ~same **-0.25** | ~**-0.075** |
| 1 key | 0.1 - 25(0.01) = **-0.15** | 0.3(-0.15) = **-0.045** |
| 2 keys | 0.2 - 25(0.01) = **-0.05** | 0.3(-0.05) = **-0.015** |
| Solved (2 keys + goal) | 0.2 + 1.0 - 5(0.01) = **+1.15** | 0.7(1.0) + 0.3(1.15) = **+1.045** |

**Critical observation:** The non-solving range spans only **0.060** units (-0.075 to -0.015). The cliff to solving is **1.060**. This is an 18x jump. Programs picking 0, 1, or 2 keys are nearly indistinguishable by reward.

---

## 4. The Grammar (DSL)

Programs are decision-list ASTs built from a context-free grammar with budget constraints.

### 4.1 AST Nodes

| Node | Semantics | Cost |
|---|---|---|
| `Flip(j)` | Execute action j | 1 |
| `IsZero(j)` | Test obs[j]==0 (returns bool) | 1 |
| `Not(cond)` | Logical negation | 1 + child |
| `And(cond1, cond2)` | Conjunction | 1 + children |
| `Ite(cond, Flip(j), else_prog)` | If cond then action j else continue | 1 + cond + 1 + else |
| `Default(Flip(j))` | Always action j (base case) | 2 |

### 4.2 Production Rules

```
Program(k):
  k == 2:  Default(Flip(j))                         for j in [0, n_actions)
  k >= 5:  Ite(Cond(i), Flip(j), Program(k-2-i))    for i in [1, k-4], j in [0, n_actions)

Condition(k):
  k == 1:  IsZero(j)                                 for j in [0, n_sites)
  k >= 2:  Not(Cond(k-1))                            (banned if parent is Not)
  k >= 3:  And(Cond(i), Cond(k-1-i))                 for i in [1, floor((k-1)/2)]
```

**Note on IsZero semantics:** `IsZero(j)` tests `obs[j]==0`. Since obs uses 1 for "true":
- `IsZero(1)` = "NOT at loc 1" (obs[1]=0 means absent)
- `Not(IsZero(1))` = "at loc 1"
- `Not(IsZero(9))` = "key 0 is available"
- `IsZero(7)` = "room 1 is locked"

### 4.3 Budget and Program Space

The total budget `L` controls AST size. It is set to ~1.5x the optimal program's node count.

For D=3: optimal = 22 nodes (2 PickRules@7 + 2 MoveRules@3 + 1 Default@2), budget L=34.

**Budget gaps:** P(3) and P(4) have zero valid programs (Default needs exactly 2, Ite needs at least 5). In "exact" mode, productions leading to these budgets are pruned.

### 4.4 Constraints That Prune Productions

1. **Action range:** `Flip(j)` only for j in [0, n_actions), not [0, n_sites).
2. **Double-negation ban:** `Not(Not(...))` is suppressed (wastes 2 nodes).
3. **One-hot groups:** Within `And(...)`, two positive literals in the same one-hot group (e.g., "at loc 1 AND at loc 3") are pruned.
4. **Dead-end pruning (exact mode):** Skip productions leading to budgets with zero completions.
5. **Condition budget cap (mode 3):** Caps condition size in Ite to <=12 nodes.

---

## 5. The Four Derivation Modes

The program is built by repeatedly expanding the leftmost hole in the partial AST. Each expansion is one "game step." The four modes differ in how productions are presented to the search algorithm.

### 5.1 Mode 0: `doors` — Flat, And Enabled

Each production is a single atomic action. `Ite(Cond(3), Flip(5), Program(29))` is one choice among ~279 options.

- **Branching:** ~279 actions per step
- **Depth:** ~15 steps per game
- **Grammar:** Full (And + Not + IsZero)

### 5.2 Mode 1: `doors_no_and` — Flat, And Disabled

Identical to Mode 0 but `And(...)` removed from grammar. PICK preconditions (at location AND key available) cannot be expressed in a single condition — must use nested Ite chains.

- **Branching:** ~200 actions
- **Purpose:** Ablation to measure And's contribution

### 5.3 Mode 2: `doors_factored` — Factored, And Enabled

Same grammar as Mode 0, but each production split into two phases:
1. **Structure phase:** Choose template shape (e.g., "Ite with condition budget 3")
2. **Parameter phase:** Choose index (e.g., "Flip(5)")

- **Branching:** max(~15 structures, ~9 parameters) ≈ 31 per step
- **Depth:** ~15 steps (same programs, but 1 flat step = 1-2 factored steps)
- **Lossless:** Identical program space to Mode 0

### 5.4 Mode 3: `doors_d10_macro` — Factored + Macros + Condition Cap

Builds on Mode 2 with domain-specific macro productions:

**PickRule(k)** (7 AST nodes, 1 derivation step):
```
Ite(And(Not(IsZero(key_loc[k])), Not(IsZero(key_avail_idx[k]))),
    Flip(PICK_action[k]),
    ProgramHole(budget - 7))
```

**MoveRule(k)** (3 AST nodes, 1 derivation step):
```
Ite(IsZero(room_unlock_idx[k]),
    Flip(MOVE_to_key_loc[k]),
    ProgramHole(budget - 3))
```

- **Branching:** ~39 per step (macros add to structure templates)
- **Depth:** ~10 steps for D=3 (down from ~15)
- **Condition cap:** Max condition budget = 12 (prunes useless large conditions)
- **Not lossless:** Macros add options but don't remove primitive productions

---

## 6. The Training Pipeline

### 6.1 AlphaZero Loop

```
for iteration in 1..n_iterations:
    1. SELF-PLAY: Run n_games_per_train derivation games using MCTS + neural network
       - Each game step: MCTS(state, network, n_sims) → action probabilities π
       - Collect training examples: (partial_ast_obs, π, terminal_reward)
    2. TRAIN: Update network on recent examples (last n_past_iterations)
       - Loss = MSE(value, reward) + 2.0 * CrossEntropy(policy, π)
    3. EVALUATE: Pit new network vs old network on fresh games
       - If win_rate >= accept_threshold (0.40): keep new weights
       - Else: revert to old weights
```

**Note:** This is a single-player variant of AlphaZero. The original algorithm was designed for two-player games; here the "game" is program construction against a fixed environment, with no opponent. See Section 2 for details on why this mapping is challenging.

### 6.2 Observation Encoding

Partial AST → fixed-size vector of `2 * budget` floats via preorder traversal:
```
obs = [type_id_0, param_0, type_id_1, param_1, ..., 0, 0, ...]
```

Node type IDs: PAD=0, Flip=1, IsZero=2, Not=3, And=4, Ite=5, Default=6, ProgramHole=7, ConditionHole=8.

Example — initial state `ProgramHole(34)`:
```
obs = [7, 34, 0, 0, 0, 0, ..., 0]    (68 floats)
```

For factored modes: 2 extra floats appended (phase_id, pending_template_id).

### 6.3 Network Architecture

Transformer encoder:
- Input: Parse obs into (type_id, param) pairs → embed type + project param → add positional → prepend CLS
- Padding mask: type_id==0 slots ignored by attention
- Output: CLS token → two heads:
  - Policy: Linear(d_model → action_size) → softmax at inference
  - Value: Linear(d_model → 1) → scalar

Default hyperparameters: d_model=64, n_heads=4, n_layers=2, lr=3e-4, batch_size=32, epochs=5, policy_weight=2.0.

### 6.4 MCTS Details

At each game step, MCTS runs `n_simulations` tree traversals:
1. **Select:** Follow UCB from root. UCB = Q_normalized + c_exploration * P(a) * sqrt(N_total) / (1 + N(a))
2. **Expand:** At leaf, query network → (policy, value)
3. **Rollout (optional):** Complete m random games from leaf, aggregate rewards
4. **Backup:** Update Q-values along path

**Q-normalization:** Q_norm = (Q - Q_min) / (Q_max - Q_min). When Q_min == Q_max (all values identical), defaults to 0.5 → pure exploration.

**Backup rules:**
- `mean`: Q = average of all values (standard)
- `max`: Q = best value seen (optimistic — good for synthesis where we want ONE good program)
- `topk`: Q = average of top-k values
- `softmax`: Smooth approximation of max

**Dirichlet noise:** At root, P_noisy = (1-ε)*P_network + ε*Dir(α), masked to valid actions.

**Rollout evaluation:** When enabled (rollout_n > 0), each MCTS leaf spawns m random game completions. The rollout value is blended with the network value: (1-blend)*rollout_value + blend*nn_value.

Current defaults: n_sims=80, c_exploration=1.5, dirichlet_alpha=0.25, dirichlet_epsilon=0.40, rollout_n=4, rollout_mode="max", rollout_blend=0.3, backup_rule="max".

### 6.5 Leaf Evaluation (Terminal Reward)

When a derivation game completes (no holes left), the finished program is evaluated:

```python
for each frozen_state:
    env = DoorsPDDLLiteEnv(D=3, ...)
    obs = env.reset(frozen_state)
    total_reward = 0
    while not done:
        action = program.execute(obs)   # reactive: first matching rule fires
        obs, reward, done = env.step(action)
        total_reward += reward
    record solve_rate, total_reward
```

**Metric computation (scalar returned to MCTS):**
```python
"weighted": alpha * solve_rate + (1-alpha) * avg_reward    # alpha=0.7
"avg_reward": avg_reward
"solve_rate": solve_rate
"penalized_reward": avg_reward - lambda * avg_ops / max_ops
"keys_progress": keys_picked/total_keys + 0.1 * avg_reward
```

Currently using `weighted` with alpha=0.7.

**Caching:** Programs are cached by their pretty-printed string. Same program = same score, no re-evaluation.

### 6.6 Reward Flow: Leaf → MCTS → Training

1. Terminal step: `reward = leaf_eval(program)` → scalar (e.g., -0.075 for garbage, +1.045 for solver)
2. Non-terminal steps: `reward = 0.0`
3. With `reward_discount = 1.0`, every step in the episode gets value target = terminal reward
4. Training examples: `(partial_ast_obs, mcts_policy, terminal_reward)` for every step

**Implication:** The value network is asked to predict, from a partial AST alone, what the eventual terminal reward will be. Since discount=1.0, all steps in one game share the same value target.

---

## 7. Current Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| **Problem** | D (rooms) | 3 |
| | M (locations) | 6 |
| | K (keys) | 2 |
| | n_sites | 11 |
| | n_actions | 9 |
| | budget (L) | 34 |
| | horizon | 25 |
| | step_penalty | 0.01 |
| | unlock_bonus | 0.10 |
| **Grammar** | program_budget_mode | "max" (allows early termination) |
| | allow_and | True |
| | allow_not | True |
| | one_hot_groups | [[0,1,2,3,4,5]] (location indices) |
| **Metric** | metric | "weighted" |
| | blend_alpha | 0.7 |
| **Network** | d_model | 64 |
| | n_heads | 4 |
| | n_layers | 2 |
| | lr | 3e-4 |
| | epochs | 5 |
| | batch_size | 32 |
| | policy_weight | 2.0 |
| **MCTS** | n_simulations | 80 |
| | c_exploration | 1.5 |
| | dirichlet_alpha | 0.25 |
| | dirichlet_epsilon | 0.40 |
| | rollout_n | 4 |
| | rollout_mode | "max" |
| | rollout_blend | 0.3 |
| | rollout_budget | 200 |
| | backup_rule | "max" |
| **Training** | n_games_per_train | 30 |
| | n_past_iterations | 20 |
| | n_procs | 8 |
| | reward_discount | 1.0 |
| **Evaluation** | n_games | 20 |
| | accept_threshold | 0.40 |
| | eval_temperature | 0.05 |
| **Run** | n_iterations | 30 |

---

## 8. What Has Been Tried and What Failed

### 8.1 D=2 Results: Consistently Solved

D=2 is reliably solved across all grammar modes:

| Experiment | Mode | Sims | Games/iter | Solved at iter | Programs explored |
|---|---|---|---|---|---|
| D2 flat (10 games/iter) | Flat | 100 | 10 | iter 8 | 6,977 |
| D2 flat (20 games/iter) | Flat | 100 | 20 | iter 7 | 4,487 |
| D2 factored+macro | Factored+macros | 80 | 30 | **iter 1** | 8,460 |

**Why D=2 is easy:**
- Program is small: 12 AST nodes (1 PickRule + 1 MoveRule + 1 Default)
- Only 1 key → only 1 conjunctive condition needed (`And(Not(IsZero(1)), Not(IsZero(6)))`)
- Budget L=18 → small program space
- The solver is common among random programs (~1 in 5K)
- Even flat grammar (90 actions) finds it within 5-7K programs

**D=2 optimal program found in every run:**
```
if And(Not(IsZero(1)), Not(IsZero(6))):   # at loc 1 AND key 0 available
    Flip(4)                                #   → PICK key 0
elif IsZero(5):                            # room 1 locked
    Flip(1)                                #   → MOVE to loc 1
else:
    Flip(3)                                #   → MOVE to goal (loc 3)
```

### 8.2 D=3 Results: Unreliable — The Critical Gap

**Flat grammar (modes 0, 1): NEVER solves D=3**

| Experiment | Sims | Games/iter | Iters | Programs | Best reward | Solved? |
|---|---|---|---|---|---|---|
| flat, 200 sims, 40 games | 200 | 40 | 21 | 125K | -0.15 | No |
| flat, 150 sims, 20 games | 150 | 20 | 20 | 66K | -0.15 | No |
| flat, 100 sims, 20 games | 100 | 20 | 20 | 45K | -0.15 | No |
| Grammar suite (13 runs) | 100 | 20 | 20 | ~44K each | -0.15 | No (0/13) |

Best programs from flat grammar are stuck at 1-key partial solutions (reward -0.15).

**Factored+macros grammar (mode 3): Solves SOME runs**

| Experiment | Sims | Games/iter | Solved? | At iter | Programs |
|---|---|---|---|---|---|
| 200 sims, 80 games, 50 iters | 200 | 80 | **Yes** | 14 | 396K |
| 100 sims, 80 games, 50 iters | 100 | 80 | No | -- | 203K |
| 80 sims, 30 games (run A) | 80 | 30 | **Yes** | 15 | 170K |
| 80 sims, 30 games (run B) | 80 | 30 | No | -- | 136K |
| 80 sims, 30 games (run C) | 80 | 30 | **Yes** | 11 | 228K |
| 50 sims, 30 games (run A) | 50 | 30 | **Yes** | 1 | 130K |
| 50 sims, 30 games (run B) | 50 | 30 | **Yes** | 30 | 350K |
| 50 sims, 50 games, 100 iters | 50 | 50 | No | -- | 688K (2 keys, -0.05) |
| 30 sims, 30 games | 30 | 30 | **Yes** | 1 | 85K |
| 20 sims, 30 games | 20 | 30 | **Yes** | 7 | 90K |
| 100 sims, 30 games (run A) | 100 | 30 | **Yes** | 1 | 221K |
| 100 sims, 30 games (run B) | 100 | 30 | **Yes** | 1 | 200K |
| 200 sims, 30 games | 200 | 30 | **Yes** | 1 | 345K |

**Key observations from D=3:**
1. **Factored+macros is necessary but not sufficient.** Flat grammar NEVER solves. Factored+macros sometimes solves (~60-70% of runs).
2. **Success is random, not learned.** Most successful runs find the solver at iteration 1 (before any training). The solver is found by brute-force exploration, not guided search.
3. **More compute doesn't guarantee success.** One run explored 688K programs over 100 iterations without solving. Another solved with 85K at iteration 1.
4. **The 2-key-but-not-solved trap.** Several failed runs find programs that pick 2 keys but can't reach the goal (reward -0.05 or weighted ~1.75). The system can't improve beyond this.
5. **Training loop contributes nothing.** In solved runs, the solver appears in early iterations via exploration. The AlphaZero learning cycle never actually guides search toward better programs.

### 8.3 What Changes Between D=2 and D=3

| Dimension | D=2 | D=3 | Impact |
|---|---|---|---|
| Keys to coordinate | 1 | 2 | Must find 2 PickRules + 2 MoveRules in correct structure |
| Optimal program nodes | 12 | 22 | 1.8x larger, many more possible ASTs |
| Budget (L) | 18 | 34 | 1.9x more headroom → more junk programs |
| Flat branching | ~90 | ~279 | 3x more actions per step |
| Factored branching | ~15 | ~39 | 2.6x more templates |
| Conjunctive conditions needed | 1 | 2 | Each requires specific 5-node And(Not(IsZero(a)), Not(IsZero(b))) |
| Solver frequency (factored+macros) | ~1 in 5K | ~1 in 100K-200K | 20-40x rarer |

**The scaling problem:** D=2 needs 1 PickRule (7 nodes with exact parameters). D=3 needs 2 PickRules (14 nodes with exact parameters from different observation indices). The probability of BOTH appearing correctly drops multiplicatively. Each additional key roughly squares the difficulty.

### 8.4 Pure MCTS Baseline (No Neural Network)

| Run | Mode | Rounds x Sims | Programs | Solved? |
|---|---|---|---|---|
| Baseline A | Flat, uniform prior | 400 x 100 | 110K | No |
| Baseline E | Factored+macros, uniform prior | 400 x 200 | 110K | No |

Pure MCTS with uniform priors explored 110K programs without finding a solver. But a randomly initialized Transformer found one in 9.4K programs. **Random Transformer priors create structured search bias that uniform priors cannot.**

### 8.5 Direct Play Comparison

| Experiment | Mode | Sims | Games/iter | Iters | Result |
|---|---|---|---|---|---|
| Direct RL (D=10!) | Direct play (no synthesis) | 100 | 50 | 5 | **Solved** |

In **direct play**, the neural network outputs actions directly (no program synthesis layer). The network's policy head maps raw Doors observations to actions, trained via AlphaZero self-play on the raw Doors environment. The same Doors environment with direct RL solves D=10 in 5 iterations because the environment has rich per-step reward structure. The synthesis layer collapses this into a sparse terminal signal.

### 8.6 The Core Failure: Reward Desert

**91-99% of random programs score identically** (-0.075 weighted metric). Programs that are "completely wrong" and programs that "almost solve but miss one PICK" get nearly the same score.

Consequences:
1. **Value head collapses:** All training targets ≈ -0.075. Network learns V(any_state) ≈ -0.075 (constant predictor). Value loss → ~0.0001 (noise floor).
2. **MCTS degenerates:** Q_min == Q_max → Q_normalized = 0.5 for all actions → UCB becomes pure exploration noise.
3. **Gate is useless:** New and old networks produce identical program distributions → score ≈ 0.50 → no selection pressure.
4. **Vicious cycle:** Random MCTS → garbage programs → constant value targets → blind MCTS → more garbage. Repeats indefinitely.

### 8.7 Reward Compression Analysis

Two layers compress the reward signal:

**Layer 1 — Environment:** unlock_bonus (0.1) is dwarfed by total step_penalty (25 × 0.01 = 0.25). Signal-to-noise ratio: 0.1/0.25 = 0.40.

**Layer 2 — Weighted metric:** When solve_rate=0, `weighted = 0.7*0 + 0.3*avg_reward = 0.3 * avg_reward`. The 0.3x multiplier compresses the already-narrow non-solving range by 3.3x.

Combined: non-solving range = 0.060 units. Value head cannot discriminate.

### 8.8 What Has NOT Been Tried Yet

| Idea | Status | Expected impact |
|---|---|---|
| Increase unlock_bonus 0.1 → 1.0 | Spec complete, not implemented | 10x wider non-solving range |
| Adaptive alpha (return raw avg_reward when solve_rate=0) | Spec complete, not implemented | Remove 0.3x compression |
| Per-key progress metric (keys_picked/total_keys) | Implemented in code, not tested | Uniform spacing (0, 0.5, 1.0 for D=3) |
| Running reward normalization (EMA) | Implemented in code, not tested | Auto-scale to any reward range |
| Grammar pruning (one-hot + action range) | Implemented in code | Modest branching reduction |
| D=2 experiments | Not tried | Simpler testbed for debugging |
| Alternative grammars | Not tried | Could fundamentally change search |

---

## 9. The Optimal Program (D=3)

For reference, this is what the system needs to find:

```
if And(Not(IsZero(1)), Not(IsZero(9))):      # at loc 1 AND key 0 available
    Flip(6)                                    #   → PICK key 0
elif IsZero(7):                                # room 1 locked
    Flip(1)                                    #   → MOVE to loc 1
elif And(Not(IsZero(3)), Not(IsZero(10))):    # at loc 3 AND key 1 available
    Flip(7)                                    #   → PICK key 1
elif IsZero(8):                                # room 2 locked
    Flip(3)                                    #   → MOVE to loc 3
else:
    Default(Flip(5))                           #   → MOVE to goal (loc 5)
```

**Structure:** Alternating PickRule/MoveRule pairs for each key, followed by a default move-to-goal.

**Node count:** 7 + 3 + 7 + 3 + 2 = 22 nodes (budget 34 gives 55% headroom).

**What makes it hard to find:**
- The PickRule conditions require `And(Not(IsZero(a)), Not(IsZero(b)))` — 5 nodes of specific structure with specific parameters.
- The rules must appear in a viable order (you can't pick a key you haven't navigated to).
- The program must be complete (every branch must terminate with a valid action).

---

## 10. D=2 as a Debugging Testbed

D=2 has never been systematically tested. Here are its parameters:

```
Room 0 (unlocked)     Room 1 (locked)
  loc 0 (start)         loc 2
  loc 1 [Key 0]         loc 3 (GOAL)

M=4, K=1, n_sites=7, n_actions=6 (4 moves + 1 pick + 1 noop)
Optimal steps: 3 (MOVE(1), PICK(0), MOVE(3))
Optimal nodes: 12 (1 PickRule + 1 MoveRule + 1 Default)
Budget: ~18-20
Horizon: 15
```

**Optimal D=2 program (12 nodes):**
```
if And(Not(IsZero(1)), Not(IsZero(6))):   # at loc 1 AND key 0 available
    Flip(4)                                #   → PICK key 0
elif IsZero(5):                            # room 1 locked
    Flip(1)                                #   → MOVE to loc 1
else:
    Default(Flip(3))                       #   → MOVE to goal (loc 3)
```

D=2 is easier because:
- Fewer keys to coordinate (1 vs 2)
- Smaller program (12 vs 22 nodes)
- Smaller budget (18 vs 34) → fewer possible programs
- Only 1 PickRule needed (fewer conjunctive conditions)

---

## 11. Known Structural Issues

### 11.1 The Synthesis Layer Destroys Environment Reward Structure

Direct RL on Doors D=10 solves in 5 iterations. The environment has rich, graded rewards (step penalties, unlock bonuses, goal reward). But the derivation game wraps this in a synthesis MDP where:

- 93% of derivation steps have reward = 0
- Only the terminal step gets the leaf evaluation score
- All steps in one game share the same value target (discount=1.0)
- The value network must predict terminal quality from partial AST alone

The rich per-step environment signal is collapsed into a single scalar per program.

### 11.2 The Value Network Has an Impossible Task

The value head is asked: "Given this partial AST with holes, what will the terminal reward be?"

At early derivation steps (e.g., just `Ite(CondHole(5), Flip(6), ProgramHole(15))`), the future reward depends entirely on how the remaining holes are filled — which is exactly what MCTS is searching for. The partial AST alone provides almost no information about eventual quality.

For the value prediction to be useful, the network would need to learn: "partial ASTs that start with PickRule-like patterns tend to produce better programs." But this requires seeing a mix of good and bad programs during training — which doesn't happen when 99% of programs score identically.

### 11.3 MCTS Explores Within One Game, Not Across Games

Each MCTS tree search completes ~80/15 ≈ 5 programs within its search tree. With 30 games per iteration, that's ~150 programs per iteration. These are heavily correlated (same network, similar search trajectories). The diversity of explored programs is low.

### 11.4 The Policy Network Learns Noise

When MCTS degenerates (Q-values uniform), visit counts are driven by Dirichlet noise. The policy targets become noisy distributions over valid actions. Training on these teaches the network to reproduce noise, not meaningful preferences.

### 11.5 The Derivation Game MDP Is Fundamentally Different from Board Games

AlphaZero was designed for two-player, zero-sum, perfect-information games where: (a) intermediate states have intrinsic value (board position quality), (b) the opponent creates natural curriculum (weak opponents early, strong later), (c) reward variance is high (win/loss/draw provides real signal), and (d) state space and action space are balanced (not extremely wide branching at every step).

The derivation game violates all four properties:
- Partial ASTs have no intrinsic value — all information is in the unfilled holes
- There is no opponent, so no curriculum — the problem difficulty is fixed
- Reward variance is near-zero (99% of programs score the same)
- Branching factor (39-279 actions) is high at every step, with no narrowing

This is not a tuning problem. It is a structural mismatch between the algorithm and the problem.

### 11.6 Exploration vs. Exploitation Balance

MCTS in AlphaZero was tuned for exploitation-heavy regimes (board games have strong value signals). In the derivation game:
- The value signal is flat, so MCTS cannot exploit
- c_exploration = 1.5 with flat values means pure exploration
- Dirichlet noise adds further randomness to an already random process
- The net effect is expensive random search with MCTS overhead

A simpler random sampling strategy (enumerate programs, evaluate each) might explore more programs per unit time, since MCTS adds overhead without adding guidance.

### 11.7 The Training Data Distribution Problem

The training loop collects examples from MCTS self-play. In board games, this produces a curriculum: early iterations play badly (diverse mistakes), later iterations play well (refined positions). In the derivation game:
- Early iterations produce 99% garbage programs (reward ~ -0.075)
- Later iterations ALSO produce 99% garbage programs (no improvement)
- The training distribution is essentially i.i.d. noise across all iterations
- The network converges to a constant predictor because the training data IS constant

---

## 12. Project Structure and Codebase Map

### 12.1 Directory Tree

```
AlphaZero_PP/
├── setup.py                              # Package definition (pip install -e .)
├── README.md
│
├── src/alphazeropp/                      # Main package
│   ├── __init__.py
│   ├── main.py                           # Stub entry point
│   │
│   ├── core/                             # Generic AlphaZero framework (domain-agnostic)
│   │   ├── config.py                     # TrainingConfig, MCTSConfig, EvalConfig dataclasses
│   │   ├── game.py                       # Abstract Game interface (getInitBoard, getNextState, etc.)
│   │   ├── mcts.py                       # MCTS: UCB selection, backup rules (mean/max/topk/softmax),
│   │   │                                 #   Dirichlet noise, Q-normalization, rollouts
│   │   ├── agent.py                      # Agent: plays games using MCTS + neural network
│   │   └── policy_value_net.py           # Abstract neural network interface (PyTorch)
│   │
│   ├── synthesis/                        # Domain-agnostic program synthesis layer
│   │   ├── ast_nodes.py                  # AST node types: Flip, IsZero, Not, And, Ite, Default
│   │   ├── budget_grammar.py             # Context-free grammar with budget constraints; program counting
│   │   ├── derivation.py                 # DerivationState: partial AST, production generation,
│   │   │                                 #   one-hot constraints, action masking
│   │   ├── derivation_game.py            # DerivationGame: wraps derivation as a core.Game (flat actions)
│   │   ├── factored_derivation_game.py   # FactoredDerivationGame: two-phase (structure/parameter) actions
│   │   ├── derivation_network.py         # Transformer encoder: partial AST obs → (policy, value)
│   │   ├── leaf_evaluator.py             # Evaluates complete programs on environment; caching; metrics
│   │   ├── interpreter.py                # Program interpreter (executes AST against environment)
│   │   └── protocols.py                  # DSLGameConfig Protocol (interface for domain configs)
│   │
│   ├── training/                         # Training loop components
│   │   ├── trainer.py                    # Self-play data collection, network training, replay buffer
│   │   ├── evaluator.py                  # Pits new network vs old network on fresh games
│   │   └── gated_trainer.py              # Acceptance gate: keep weights only if win_rate >= threshold
│   │
│   ├── utils/                            # Shared utilities
│   │   ├── checkpoint.py                 # Save/load model checkpoints
│   │   ├── common.py                     # Shared helpers (logging, path creation)
│   │   ├── derivation_utils.py           # Main training orchestration loop, logging, diagnostics
│   │   ├── interactive_config.py         # CLI-based config selection (mode, hyperparams)
│   │   ├── multiprocessing.py            # Multiprocessing helpers (MPS-safe)
│   │   ├── post_diagnostics.py           # Post-hoc analysis of experiment logs
│   │   └── statistics.py                 # Running statistics, EMA normalization
│   │
│   ├── benchmark/                        # Benchmark harness for comparing algorithms
│   │   ├── run.py                        # Benchmark runner
│   │   ├── sweep.py                      # Hyperparameter sweep orchestration
│   │   ├── eval_loop.py                  # Evaluation loop
│   │   ├── env_factory.py                # Environment factory
│   │   ├── solve_criterion.py            # What counts as "solved"
│   │   ├── result_schema.py              # Result data schema
│   │   ├── plotting.py, plot_cli.py      # Visualization and CLI
│   │   └── adapters/                     # Algorithm adapters
│   │       ├── base.py                   # Abstract adapter interface
│   │       ├── alphazero.py              # AlphaZero adapter
│   │       ├── oracle.py                 # Oracle (optimal) adapter
│   │       ├── random_agent.py           # Random baseline
│   │       ├── tabular_q.py              # Tabular Q-learning
│   │       └── sb3.py                    # Stable-Baselines3 adapter (PPO, etc.)
│   │
│   └── instances/                        # Domain-specific implementations
│       ├── bitstring/                    # Bitstring domain (simpler synthesis testbed)
│       │   ├── config.py, game.py, network.py, run.py
│       │   └── dsl/                      # Bitstring DSL (scan grammar)
│       ├── cartpole/                     # CartPole domain (direct RL only, no synthesis)
│       │   └── config.py, game.py, network.py, run.py
│       └── doors/                        # *** Doors domain (primary focus) ***
│           ├── doors_pddl_lite.py        # Gymnasium environment: step(), reset(), reward logic
│           ├── config.py                 # Direct-play config (DoorsDirectConfig)
│           ├── game.py                   # Direct-play game wrapper (DoorsDirectGame)
│           ├── network.py                # Direct-play MLP network
│           ├── oracle.py                 # Hand-coded optimal policy
│           ├── eval_ablations.py         # Ablation evaluation variants
│           ├── network_ablations.py      # Ablated network architectures
│           └── dsl/                      # *** Doors DSL for program synthesis ***
│               ├── doors_config.py       # DoorsGameConfig: derived parameters, env factory, frozen states
│               ├── derivation_config.py  # Four config classes (flat, no-and, factored, factored+macros)
│               └── doors_macros.py       # PickRule/MoveRule macro productions
│
├── scripts/                              # Experiment entry points
│   ├── run_doors_derivation.py           # PRIMARY: Doors synthesis training (mode selection)
│   ├── run_doors_derivation_suite.py     # Sweep multiple grammar modes
│   ├── run_doors_direct.py               # Direct RL on Doors (no synthesis)
│   ├── run_doors_mcts_baseline.py        # Pure MCTS baseline (no neural network)
│   ├── run_doors_baselines.py            # Benchmark baselines (oracle, random, Q-learning, SB3)
│   ├── run_ablation_sweep.py             # Ablation sweep runner
│   ├── run_bitstring.py                  # Bitstring direct RL
│   ├── run_bitstring_derivation.py       # Bitstring synthesis
│   ├── run_cartpole.py                   # CartPole direct RL
│   ├── enumerate_dsl.py                  # Enumerate all programs in DSL
│   ├── estimate_expressivity_gap.py      # DSL expressivity analysis
│   ├── sweep_doors_hyperparams.py        # Hyperparameter sweep
│   ├── plot_ablation_sweep.py            # Plot ablation results
│   ├── plot_d3_comparison.py             # Plot D=3 comparison
│   └── audit_doors_env.py               # Environment audit
│
├── tests/                                # Test suite (pytest, 27 modules)
│   ├── test_doors_pddl_lite.py           # Environment step/reward tests
│   ├── test_doors_macros.py              # Macro production tests
│   ├── test_doors_oracle.py              # Oracle policy tests
│   ├── test_derivation_game.py           # Flat derivation game tests
│   ├── test_factored_derivation_game.py  # Factored game tests
│   ├── test_dsl.py                       # DSL/AST/interpreter tests
│   ├── test_cfg_grammar.py              # Grammar counting tests
│   ├── test_grammar_pruning.py           # Pruning constraint tests
│   ├── test_leaf_evaluator_stats.py      # Evaluator statistics tests
│   ├── test_mcts_backup.py              # MCTS backup rule tests
│   ├── test_mcts_rollout.py             # Rollout tests
│   ├── test_max_mode.py                 # Budget max mode tests
│   ├── test_benchmark_smoke.py          # Benchmark smoke tests
│   └── ...                              # + tests for ablations, baselines, bitstring, cartpole
│
├── experiments/                          # Experiment results (timestamped directories)
│   ├── doors_derivation/                 # *** Doors synthesis runs (primary) ***
│   ├── doors_direct/                     # Doors direct RL runs + ablations
│   ├── doors_grammar_suite/              # Grammar mode comparison runs
│   ├── doors_mcts_baseline/              # Pure MCTS baseline runs
│   ├── doors_audit/                      # Environment audit results
│   ├── derivation/                       # Generic derivation experiments (bitstring)
│   ├── bitstring/                        # Bitstring experiments
│   └── stage3/                           # Benchmark comparison plots
│
├── specs/                                # Design specification documents
├── docs/                                 # Protocol/design documentation
└── notes/                                # Development notes
```

### 12.2 Architecture Layers

```
┌─────────────────────────────────────────────────┐
│ Scripts (scripts/)                               │
│ Entry points for training, evaluation, sweeps    │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────┐
│ Domain Instances (instances/)                    │
│ doors/, bitstring/, cartpole/                    │
│ Each provides: Game, Config, Network, DSL        │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────┐
│ Training & Benchmark (training/, benchmark/)     │
│ Trainer, Evaluator, GatedTrainer, Sweep, Adapters│
└─────────────┬───────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────┐
│ Core Engine (core/, synthesis/)                   │
│ MCTS, Agent, PolicyValueNet (core/)              │
│ AST, Grammar, Derivation, LeafEvaluator (synthesis/)│
└─────────────┬───────────────────────────────────┘
              │
┌─────────────┴───────────────────────────────────┐
│ Utilities (utils/)                               │
│ Checkpoint, Statistics, Multiprocessing          │
└─────────────────────────────────────────────────┘
```

### 12.3 Experiment Naming Convention

Experiment directories follow:
```
YYYYMMDD_HHMMSS_D<rooms>_<grammar>_N<sites>_L<budget>_<metric>_mcts<sims>_games<per_iter>_iter<iters>
```

Example: `20260304_190430_D3_and_factored_macro_N11_L34_max_weighted_mcts200_games80_iter50`

Each experiment directory contains:
- `config.json` — Full hyperparameter snapshot (reproducible)
- `train_stats.jsonl` — Per-iteration training metrics (loss, reward, programs explored)
- `eval_stats.jsonl` — Per-iteration evaluation metrics (win rate, accept/reject)
- `program_log.jsonl` — Log of all unique programs evaluated (pretty-print, score, iteration)
- `metrics_*.png` — Auto-generated training curve plot

### 12.4 Key Entry Points

| Task | Script |
|---|---|
| Doors synthesis (primary) | `scripts/run_doors_derivation.py` |
| Direct RL on Doors | `scripts/run_doors_direct.py` |
| Pure MCTS baseline | `scripts/run_doors_mcts_baseline.py` |
| Benchmark comparison | `scripts/run_doors_baselines.py` |
| Ablation sweep | `scripts/run_ablation_sweep.py` |
| Enumerate DSL programs | `scripts/enumerate_dsl.py` |

---

## 13. Diagnosis Summary and Open Questions

### 13.1 Summary of Confirmed Failures

1. **The AlphaZero learning loop contributes nothing.** In every successful D=3 run, the solver was found by exploration (MCTS noise + random network priors), not by learned guidance. The neural network never learns to prefer good partial ASTs over bad ones.

2. **The value head is a constant predictor.** With 99% of training targets at ~-0.075, the network learns V(s) = -0.075 for all states s. Value loss converges to noise floor (~0.0001). This makes MCTS degenerate to random search.

3. **The policy head learns noise.** MCTS visit counts, when driven by flat Q-values and Dirichlet noise, produce noisy training targets. The policy network reproduces this noise rather than learning meaningful preferences.

4. **The reward signal is too compressed.** The environment reward range for non-solving programs is 0.060 units wide. The weighted metric compresses this further to 0.018 units. The value head cannot discriminate within this range.

5. **Success depends entirely on grammar and luck.** Factored+macros grammar reduces the search space enough that random exploration occasionally finds the solver. Flat grammar never does. The learning component adds no value.

### 13.2 Root Cause Hypotheses

**Hypothesis A: Wrong reward function.** The synthesis MDP reward is too sparse and compressed. Even "almost correct" programs (picking 2 keys but failing to navigate to goal) score nearly the same as completely wrong programs. Fix: redesign the reward to give graded credit for partial progress (e.g., per-key progress, intermediate navigation milestones).

**Hypothesis B: Wrong algorithm.** AlphaZero is designed for domains where intermediate states have meaningful value. Program synthesis partial ASTs do not. A different search algorithm (e.g., evolutionary search, beam search, language-model guided sampling) might be more appropriate.

**Hypothesis C: Wrong problem decomposition.** Treating program synthesis as a single sequential game (fill holes left to right) makes the problem unnecessarily hard. Alternative decompositions (e.g., synthesize each rule independently then combine; hierarchical synthesis) could reduce effective search depth.

**Hypothesis D: Missing curriculum.** The system jumps directly to D=3. If it trained on D=1 and D=2 first and transferred learned network weights, the policy/value heads might have meaningful priors for D=3 partial ASTs.

### 13.3 Questions We Need Help With

1. **Is there a reward shaping that would make the value head useful?** Given the structure of the derivation game (partial AST → complete program → environment evaluation), what intermediate rewards could provide gradient to the value head? Ideas: predict number of keys the program will pick, predict whether the program will contain a PickRule, estimate structural similarity to known good programs.

2. **Is AlphaZero the right algorithm for this problem?** If the value function is fundamentally uninformative for partial ASTs, should we abandon AlphaZero entirely? What algorithms have succeeded in neural-guided program synthesis?

3. **Can the grammar be restructured to make search easier?** The current grammar builds programs left-to-right, node by node. Would a different traversal order (e.g., top-down by rule, bottom-up) make the value function more informative? Would a type-directed synthesis approach help?

4. **Is the observation encoding sufficient?** The partial AST is encoded as a flat vector of (type_id, param) pairs. Is there a richer encoding (e.g., tree-structured, graph neural network) that would help the network learn meaningful value predictions?

5. **Can we exploit the known structure of good programs?** We know the optimal program has alternating PickRule/MoveRule pairs. Can we build this structural knowledge into the grammar, the reward, or the search algorithm without completely hand-coding the solution?

6. **What can we learn from the D=2 success?** D=2 is reliably solved. Can we analyze the D=2 training dynamics to understand what (if anything) the network learns, and why that learning doesn't transfer to D=3?
