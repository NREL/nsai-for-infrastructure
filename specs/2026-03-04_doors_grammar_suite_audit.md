# Doors Grammar Suite Experiment Audit — 2026-03-04

## 1. Project Context

**AlphaZero_PP** is a research project that uses AlphaZero-style self-play (MCTS + neural network) to synthesize programs from a context-free grammar (CFG). Instead of playing a board game, the "game" is a derivation process: starting from a program hole, the agent applies grammar productions step by step to build a complete decision-list program. The completed program is then evaluated as a reactive policy on an environment and scored.

### Architecture

```
AlphaZero training loop:
  1. Self-play: MCTS explores the derivation tree, guided by a policy-value network
  2. Collect experience: (partial_derivation_state, mcts_visit_counts, game_outcome)
  3. Train network: predict visit distributions (policy head) and final reward (value head)
  4. Evaluate: pit new network against old via gated evaluation
  5. Repeat
```

### DSL (Domain-Specific Language)

Programs are decision lists over an observation vector:
```
if <condition>: <action>
elif <condition>: <action>
...
else: <default_action>
```

Where:
- **Conditions**: `IsZero(i)` (obs[i]==0), `Not(c)`, `And(c1, c2)`
- **Actions**: `Flip(j)` (select action j in the environment)
- **Program structure**: `Ite(cond, action, else_prog)` or `Default(action)`

Programs are built by expanding holes in an AST. A **budget** limits the total number of AST nodes.

### Derivation Game Variants

1. **Flat (DerivationGame)**: At each step, choose from ALL legal productions for the current hole. Action space = max productions across all possible holes.
2. **Factored (FactoredDerivationGame)**: Split each production choice into (1) structure template selection, (2) parameter selection. Reduces branching factor ~3-5×.

---

## 2. The Doors Environment

**DoorsPDDLLiteEnv** — a PDDL-faithful key-and-lock navigation puzzle.

### Layout (D=3, the configuration tested)
- **3 rooms** (0, 1, 2), each with 2 locations. 6 locations total.
- **2 keys**: Key 0 at location 1 (room 0), unlocks room 1. Key 1 at location 2 (room 1), unlocks room 2.
- **Start**: Location 0 (room 0). **Goal**: Location 5 (room 2).
- **Sequential dependency**: Must pick key 0 → unlock room 1 → pick key 1 → unlock room 2 → reach goal.

### Observation vector (11 features)
```
Index:   0  1  2  3  4  5   6   7   8   9  10
Meaning: at_loc[0..5]      unlocked[0..2]   key_avail[0..1]
Initial: 1  0  0  0  0  0   1   0   0    1   1
```

### Actions (9 valid)
```
0-5: MOVE_TO(location)  — succeeds only if target room is unlocked
6-7: PICK(key)          — succeeds only if at key's location AND key is available
  8: NOOP
```

### Rewards
- Each step: **-0.01** (step penalty)
- Unlock a room: **+0.1** (bonus)
- Reach goal: **+1.0** (terminal reward)
- Horizon: **25 steps** (truncated if not solved)

### Optimal solution
```
Step 1: MOVE_TO(1)   — go to key 0's location
Step 2: PICK(0)      — pick key 0, unlocks room 1 (+0.1)
Step 3: MOVE_TO(2)   — go to key 1's location (now accessible)
Step 4: PICK(1)      — pick key 1, unlocks room 2 (+0.1)
Step 5: MOVE_TO(5)   — reach goal (+1.0)
Total reward: 5×(-0.01) + 2×(0.1) + 1.0 = +1.15
```

### Optimal program (decision list)
```
if And(Not(IsZero(9)), Not(IsZero(1))):     Flip(6)   # at loc 1 & key 0 avail → PICK(0)
elif And(Not(IsZero(10)), Not(IsZero(2))):   Flip(7)   # at loc 2 & key 1 avail → PICK(1)
elif Not(IsZero(9)):                          Flip(1)   # key 0 avail → go to loc 1
elif And(Not(IsZero(10)), Not(IsZero(7))):   Flip(2)   # key 1 avail & room 1 open → go to loc 2
else:                                         Flip(5)   # go to goal
```
This requires ~27 AST nodes (4 Ite + 1 Default, each with conditions).

---

## 3. Experiment Configuration

**Date**: 2026-03-04
**Suite path**: `experiments/doors_grammar_suite/20260304_154856_rooms3_5/`

### Runs (10 total = 3 modes × 3 seeds + 1 macro run)

| Run | Mode | Grammar | Factored? | Seeds |
|-----|------|---------|-----------|-------|
| flat_and_D3 | Full grammar (And + Not) | Flat action space | No | 42, 137, 271 |
| flat_noand_D3 | No-And grammar | Flat action space | No | 42, 137, 271 |
| factored_D3 | Full grammar (And + Not) | Factored (struct/param) | Yes | 42, 137, 271 |
| factored_macro_D3 | Full grammar + macros | Factored + macro productions | Yes | 42 |

### Macro Grammar Mode (doors_d10_macro)

The `doors_d10_macro` mode adds two domain-expert **macro productions** on top of the factored game:

1. **PickRule macro** (7 AST nodes in one derivation step):
   ```
   Ite(And(Not(IsZero(key_loc[k])), Not(IsZero(key_avail_idx[k]))),
       Flip(pick_action[k]),
       <remaining_program>)
   ```
   Semantics: "If agent is at key k's location AND key k is available → PICK key k"

2. **MoveRule macro** (3 AST nodes in one derivation step):
   ```
   Ite(IsZero(unlocked_idx[k]),
       Flip(move_to_key_loc[k]),
       <remaining_program>)
   ```
   Semantics: "If target room is locked → move to key k's location"

Additionally, a **condition budget cap** (max 12 nodes) prunes excessively complex conditions.

**Effect on search space** (D=3):
- Derivation depth: ~119 steps → ~38 steps (3.1× reduction)
- Max structure templates: 135 → 15 (9× reduction)
- Macros encode the exact structure of an optimal solution — the agent only needs to pick the right macro sequence and parameters

### Shared hyperparameters
```json
{
  "budget": 34,
  "n_sites": 11,
  "program_budget_mode": "max",
  "horizon": 25,
  "step_penalty": 0.01,
  "unlock_bonus": 0.1,
  "metric": "weighted",
  "blend_alpha": 0.7,
  "penalty_lambda": 0.1,
  "n_simulations": 100,
  "temperature": 1.0,
  "c_exploration": 1.5,
  "dirichlet_alpha": 0.25,
  "dirichlet_epsilon": 0.4,
  "reward_discount": 1.0,
  "n_games_per_train": 20,
  "n_past_iterations_to_train": 20,
  "n_iterations": 20,
  "accept_threshold": 0.4,
  "network": { "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1,
               "epochs": 5, "batch_size": 32, "lr": 0.0003, "policy_weight": 2.0 }
}
```

### Evaluation metric
```
weighted = 0.7 × solve_rate + 0.3 × avg_env_reward
```

---

## 4. Results Summary

### All 10 runs failed identically

| Mode | Seed | Solve Rate | Best Reward (raw) | Unique Programs | Wall Clock |
|------|------|------------|-------------------|-----------------|------------|
| flat_and | 42 | 0% | -0.15 | 43,719 | 526s |
| flat_and | 137 | 0% | -0.15 | 48,188 | 741s |
| flat_and | 271 | 0% | -0.15 | 41,431 | 620s |
| flat_noand | 42 | 0% | -0.15 | 32,017 | 429s |
| flat_noand | 137 | 0% | -0.15 | 35,421 | 879s |
| flat_noand | 271 | 0% | -0.15 | 38,021 | 1288s |
| factored | 42 | 0% | -0.15 | 48,146 | 1592s |
| factored | 137 | 0% | -0.15 | 40,708 | 754s |
| factored | 271 | 0% | -0.15 | 43,161 | 952s |
| **factored_macro** | **42** | **0%** | **-0.15** | **37,845** | **1333s** |

Every run: 0% solve rate, best raw reward = -0.15 (= one room unlocked), never improved past iteration 1-2. The macro grammar — despite encoding domain-expert knowledge and reducing the search space by 9× — fails identically.

---

## 5. Detailed Data: factored_D3_seed271

### 5.1 Iteration log

| Iter | Unique Programs (cumulative) | New Programs | Gate Score | Avg Reward (weighted) | Wall Clock |
|------|------------------------------|-------------|------------|----------------------|------------|
| 1 | 373 | 373 | 0.500 | -0.075 | 14s |
| 2 | 2,590 | +2,217 | 0.500 | -0.075 | 15s |
| 3 | 6,280 | +3,690 | 0.500 | -0.075 | 20s |
| 5 | 12,016 | +3,071 | 0.500 | -0.075 | 22s |
| 10 | 25,543 | +1,823 | 0.550 | -0.074 | 39s |
| 15 | 35,591 | +2,815 | 0.525 | -0.075 | 122s |
| 17 | 38,534 | +494 | 0.525 | -0.074 | 50s |
| 20 | 43,161 | +130 | 0.500 | -0.075 | 34s |

### 5.2 Best program (found at iteration 2, never surpassed)

```
if And(IsZero(7), And(IsZero(2), IsZero(0))):
  Flip(6)    # PICK key 0
else:
  Flip(1)    # MOVE to location 1
```

Raw env reward: -0.15 (unlocks 1 room, then gets stuck in infinite loop for 23 steps).

### 5.3 Training metrics progression

| Iter | Policy Loss | Value Loss | Num Examples |
|------|------------|------------|-------------|
| 1 | 2.652 | 0.0522 | 214 |
| 5 | 1.922 | 0.0047 | 1,290 |
| 10 | 1.823 | 0.0006 | 3,054 |
| 15 | 1.778 | 0.00002 | 5,146 |
| 20 | 1.773 | 0.00002 | 6,699 |

### 5.4 Leaf value distribution (across 400 episodes)

| Weighted Metric | Count | Meaning |
|----------------|-------|---------|
| -0.075 | 399 | Program runs 25 steps, solves nothing |
| -0.045 | 1 | Program unlocks 1 room |

### 5.5 Derivation depth per iteration

| Iter | Avg Depth | Trivial (2-step) Episodes | Pattern |
|------|-----------|--------------------------|---------|
| 1 | 10.7 | 5/20 (25%) | Random exploration |
| 5 | 14.2 | 2/20 (10%) | Growing complexity |
| 10 | 18.2 | 2/20 (10%) | Exploring complex programs |
| 15 | 28.7 | 0/20 (0%) | Maximum complexity |
| 20 | 6.9 | 13/20 (65%) | Mode collapse to trivial programs |

---

## 5b. Detailed Data: factored_macro_D3_seed42

This run uses the `doors_d10_macro` mode: factored derivation game with macro productions (PickRule, MoveRule) and condition budget cap (12 nodes). Despite encoding domain-expert knowledge and reducing the search space ~9×, it fails identically.

### 5b.1 Iteration log

| Iter | Unique Programs (cumulative) | New Programs | Gate Score | Avg Reward (weighted) | Wall Clock |
|------|------------------------------|-------------|------------|----------------------|------------|
| 1 | 594 | 594 | 0.475 | -0.075 | 103s |
| 2 | 4,304 | +3,710 | 0.500 | -0.075 | 72s |
| 3 | 6,669 | +2,365 | 0.500 | -0.075 | 72s |
| 5 | 12,292 | +3,095 | 0.500 | -0.075 | 88s |
| 7 | 18,030 | +2,490 | 0.525 | -0.071 | 79s |
| 10 | 22,971 | +1,240 | 0.475 | -0.072 | 44s |
| 14 | 30,868 | +1,624 | 1.000 | -0.072 | 81s |
| 15 | 31,399 | +531 | 0.525 | -0.071 | 27s |
| 16 | 32,455 | +1,056 | 0.500 | -0.062 | 28s |
| 20 | 37,845 | +673 | 0.500 | -0.062 | 28s |

Notable: The macro grammar starts with more programs per iteration (594 vs 373 for factored) due to richer initial exploration, and the avg reward slightly improves to -0.062 by iter 16 (vs stuck at -0.075 for factored). But solve rate remains 0%.

### 5b.2 Best program (found at iteration 1, never surpassed)

```
if And(IsZero(7), IsZero(9)):
  Flip(8)    # NOOP
elif And(Not(IsZero(0)), And(And(IsZero(5), Not(IsZero(6))), IsZero(1))):
  Flip(1)    # MOVE_TO(1)
elif And(Not(IsZero(3)), Not(IsZero(10))):
  Flip(7)    # PICK(1)
else:
  Flip(6)    # PICK(0)
```

This is a **3-Ite + 1-Default** program (significantly more complex than factored's 1-Ite + 1-Default best). Raw env reward: -0.15 (unlocks 1 room, then gets stuck).

### 5b.3 Best program execution trace

For D=3: obs indices are at_loc[0-5], unlocked[6-8], key_avail[9-10]. Actions: 0-5=MOVE_TO(loc), 6=PICK(0), 7=PICK(1), 8=NOOP.

**Initial state**: `[1,0,0,0,0,0, 1,0,0, 1,1]` (at loc 0, room 0 unlocked, both keys available)

| Step | Key Obs | Branch Taken | Action | Effect |
|------|---------|-------------|--------|--------|
| 1 | obs[0]=1, obs[7]=0, obs[9]=1 | Branch 2: Not(IsZero(0))=T, IsZero(1)=T... | Flip(1)=MOVE_TO(1) | Moves to loc 1 (key 0's location) |
| 2 | obs[0]=0, obs[7]=0, obs[9]=1 | Branch 4 (else) | Flip(6)=PICK(0) | Picks key 0, unlocks room 1, +0.1 |
| 3-25 | obs[7]=1, obs[9]=0 | Branch 4 (else) | Flip(6)=PICK(0) | **Fails**: key 0 already used (key_avail[0]=0). NOOP × 23 steps |

**Why it gets stuck**: After picking key 0 (step 2), obs[9] becomes 0 (key no longer available). Branch 1 condition `And(IsZero(7), IsZero(9))` = `And(F, T)` = F. Branch 2 condition requires `Not(IsZero(0))` but agent is at loc 1, not loc 0. Branch 3 condition `And(Not(IsZero(3)), Not(IsZero(10)))` requires being at loc 3 — but agent never moves there. Falls through to else: `Flip(6)=PICK(0)` which silently fails.

**Raw reward**: 25 × (-0.01) + 0.1 = -0.15. Same ceiling as factored's best.

### 5b.4 Training metrics progression

| Iter | Policy Loss | Value Loss | Num Examples |
|------|------------|------------|-------------|
| 1 | 4.216 | 0.0339 | 590 |
| 5 | 3.411 | 0.0035 | 2,097 |
| 10 | 3.369 | 0.00003 | 4,158 |
| 15 | 3.319 | 0.00004 | 6,067 |
| 20 | 3.298 | 0.00005 | 7,277 |

**Key observations**:
- **Policy loss is ~2× higher** than factored (3.298 vs 1.773). This reflects the larger action space from macro productions — more templates to predict, harder to fit.
- **Value loss converges to same near-zero** (0.00005 vs 0.00002). Both learn the constant predictor.
- The value network is equally useless in both modes.

### 5b.5 Leaf value distribution (across 400 episodes)

| Weighted Metric | Count | Fraction | Meaning |
|----------------|-------|----------|---------|
| -0.075 | 348 | 87.0% | Program runs 25 steps, solves nothing |
| -0.045 | 52 | **13.0%** | Program unlocks 1 room |

**Comparison with factored_D3_seed271**:
| Metric | Factored | Factored+Macro | Improvement |
|--------|----------|----------------|-------------|
| Programs scoring -0.075 | 399/400 (99.75%) | 348/400 (87%) | 12.75% fewer failures |
| Programs scoring -0.045 | 1/400 (0.25%) | 52/400 (13%) | **52× more partial solutions** |
| Programs solving puzzle | 0/400 (0%) | 0/400 (0%) | No improvement |

The macro grammar dramatically increases the rate of finding partial solutions (1 room unlocked), but the gap from "1 room unlocked" to "puzzle solved" remains unbridged.

### 5b.6 Derivation depth per iteration

| Iter | Avg Depth | Trivial (≤3 step) | Min | Max | Pattern |
|------|-----------|-------------------|-----|-----|---------|
| 1 | 29.5 | 1/20 | 2 | 42 | Deep exploration (macros enable complex programs quickly) |
| 3 | 17.9 | 4/20 | 2 | 36 | Some settling |
| 5 | 22.2 | 2/20 | 2 | 36 | Sustained exploration |
| 7 | 21.9 | 1/20 | 2 | 40 | Sustained exploration |
| 10 | 13.7 | 4/20 | 2 | 28 | Slight decrease |
| 13 | 28.4 | 1/20 | 2 | 38 | Recovery to deep exploration |
| 15 | 10.8 | 3/20 | 2 | 28 | Oscillating |
| 17 | 11.3 | 5/20 | 2 | 35 | Oscillating |
| 20 | 9.3 | 1/20 | 2 | 29 | Gradual decrease, but NO sharp collapse |

**Comparison with factored_D3_seed271**:
| Metric | Factored | Factored+Macro |
|--------|----------|----------------|
| Iter 1 avg depth | 10.7 | 29.5 |
| Iter 15 avg depth | 28.7 | 10.8 |
| Iter 20 avg depth | **6.9** (mode collapse) | **9.3** (no sharp collapse) |
| Iter 20 trivial % | 65% | 5% |

The macro grammar **prevents mode collapse**. Derivation depth oscillates between 9-29 across all 20 iterations, rather than collapsing to trivial 2-step programs. However, sustained exploration still doesn't find a solving program — the search space is still too large relative to the reward signal.

---

## 6. Code Verification

The entire pipeline was audited for correctness. **No bugs were found.**

### Components verified
1. **DoorsPDDLLiteEnv** (`doors_pddl_lite.py`): PICK preconditions (at_loc AND key_available) correct. Unlock bonus applied correctly. Goal check correct. Observation layout verified against tests.
2. **Interpreter** (`interpreter.py`): eval_condition (IsZero, Not, And) correct. eval_program (decision list walk) correct. No off-by-one errors.
3. **LeafEvaluator** (`leaf_evaluator.py`): Weighted metric formula correct. is_solved callback properly passed. Caching works correctly. Frozen states are the canonical initial state.
4. **DerivationGame** (`derivation_game.py`): Terminal detection correct. Reward = 0 for non-terminal, leaf_value for terminal. Action masking correct. No dead ends occurred (0 out of 6,699 diagnostic entries).
5. **FactoredDerivationGame** (`factored_derivation_game.py`): Structure/parameter factorization correct. Phase transitions correct.
6. **MCTS** (`mcts.py`): Dirichlet noise masking correct. UCB formula correct. Value backup correct.
7. **Agent value targets** (`agent.py`): With gamma=1.0 and sparse terminal rewards, all steps get `leaf_value` as value target. This is the standard AlphaZero approach (game outcome assigned to all positions). Not a bug.
8. **Trainer** (`trainer.py`): Examples correctly packed as (obs, policy, value) tuples.

### Conclusion
The failure is **not caused by implementation bugs**. It is a structural problem with the learning setup.

---

## 7. Systematic Failure Analysis

### Failure 1: Flat Reward Landscape (Root Cause)

The reward metric is `weighted = 0.7 × solve_rate + 0.3 × avg_env_reward`.

| Outcome | Raw Reward | Weighted | Fraction |
|---------|-----------|----------|----------|
| Random program (25 steps × -0.01) | -0.25 | **-0.075** | **~99.75%** |
| Unlocks 1 room (+0.1 bonus) | -0.15 | -0.045 | ~0.25% |
| Solves puzzle (+1.0 goal) | +1.15 | +1.045 | **0%** |

**Evidence**: 399 out of 400 training episodes scored exactly -0.075. There is no gradient between "nearly correct" and "completely random" programs. Both MCTS and the value network require reward differentiation to guide search — without it, all derivation paths look equally (un)promising.

### Failure 2: Astronomical Search Space

The optimal program requires ~20-25 factored derivation steps with average branching factor ~8.

| Quantity | Value |
|----------|-------|
| Derivation depth for optimal program | ~20-25 steps |
| Average branching factor per step | ~8 |
| Effective search tree | ~8^20 ≈ **10^18 paths** |
| MCTS simulations per game | 100 |
| Total MCTS rollouts (20 iters × 20 games × 100 sims) | 40,000 |
| **Coverage** | **~10^-14** |

The MCTS explores 0.000000000001% of the search space. Random search over 10^18 paths with 40K samples has effectively zero probability of finding a solving program.

### Failure 3: All-or-Nothing Credit Assignment

The optimal program needs 4 Ite rules + 1 Default with the correct structure, conditions, AND action indices — all simultaneously. If any single derivation choice is wrong, the program scores -0.075 (same as random). There is no partial credit:
- 3 correct rules + 1 wrong → fails → -0.075
- Correct structure, wrong Flip indices → fails → -0.075
- Correct conditions, wrong else-branch ordering → fails → -0.075

MCTS must make ~20 correct sequential decisions and only receives a reward at the end. Individual decisions get no feedback.

### Failure 4: Value Network Predicts a Constant

```
Value loss progression: 0.052 → 0.005 → 0.0006 → 0.00002
```

The value network converges to predicting -0.075 for all partial derivation states. This is the optimal constant predictor (99.75% of outcomes are -0.075). With constant Q-values, MCTS UCB reduces to:
```
UCB(a) ≈ -0.075 + c × P(a) × sqrt(N) / (1 + n(a))
```
Selection depends only on the policy prior and visit counts. The value network provides zero search guidance.

### Failure 5: Policy Mode Collapse

Derivation depth dynamics show a three-phase trajectory:

```
Phase 1 (iter 1-5):  Random exploration, avg depth ~10-14
Phase 2 (iter 5-15): Self-reinforcing complexity growth, avg depth up to 28.7
Phase 3 (iter 16-20): MODE COLLAPSE, avg depth crashes to 6.9
                       65% of episodes produce trivial 2-step Default(Flip(x)) programs
```

**Mechanism**: Since complex programs score identically to trivial `Default(Flip(x))` programs (-0.075), the policy has no reason to prefer complex derivations. Once the policy slightly favors early termination, this gets reinforced through the self-play loop, causing rapid collapse to trivial programs. New programs per iteration drops from ~2K-3K to just 130.

### Failure 6: All Modes Fail Identically

All four grammar modes (flat_and, flat_noand, factored, factored_macro) across all seeds produce the same outcome: 0% solve rate, best reward -0.15. The factored game reduces branching ~3-5× and the macro grammar reduces it ~9× — neither helps because the core failures (flat rewards, vast search space, all-or-nothing credit assignment) are independent of action space representation.

### Failure 7: Macro Grammar — Necessary But Not Sufficient

The `doors_d10_macro` mode encodes domain-expert knowledge directly into the grammar via macro productions. This is the strongest grammar tested. Comparing factored_macro vs factored:

| Metric | Factored | Factored+Macro | Effect |
|--------|----------|----------------|--------|
| Max structure templates | 135 | 15 | 9× smaller action space |
| Derivation depth for optimal | ~25 steps | ~10 steps | 2.5× shorter derivation |
| Partial solutions found (1 room) | 1/400 (0.25%) | 52/400 (13%) | **52× more partial solutions** |
| Mode collapse at iter 20? | Yes (depth 6.9, 65% trivial) | No (depth 9.3, 5% trivial) | Macros prevent collapse |
| Policy loss | 1.773 | 3.298 | Higher (more templates) |
| Value loss | 0.00002 | 0.00005 | Both learn constant |
| **Solve rate** | **0%** | **0%** | **No improvement** |
| **Best reward** | **-0.15** | **-0.15** | **No improvement** |

**What macros fix**:
1. Search space: 9× fewer structure templates means MCTS concentrates on fewer, more promising derivations
2. Exploration: No mode collapse — the policy maintains program complexity across all 20 iterations
3. Partial credit: 13% of programs unlock a room (vs 0.25%) — macros encode the right "shape" for partial solutions

**What macros don't fix**:
1. **The reward cliff**: Even with 13% partial solutions, the gap from -0.045 (1 room) to +1.045 (solved) is enormous. The value network still sees 87% of outcomes at -0.075 and learns a constant.
2. **Sequential dependency**: Solving D=3 requires unlocking room 1, THEN room 2, THEN reaching goal. Even with PickRule and MoveRule macros, the agent must discover the correct 3-macro sequence with the right parameter bindings. Each macro choice alone is easy — but chaining 3+ macros with the right ordering is the hard part.
3. **No intermediate reward signal**: A program that correctly picks key 0 AND moves to key 1's location (but doesn't pick key 1) gets -0.045 — the same as a program that only picks key 0 and goes nowhere. There is no reward gradient for "closer to solving."

**Conclusion**: The macro grammar shows that reducing the action space and encoding domain knowledge are necessary conditions for tractability, but not sufficient. The fundamental barrier is the reward landscape — MCTS+NN needs a gradient to follow, and the doors domain provides none between "unlock 1 room" and "solve everything."

---

## 8. Best Program Analysis

The best program found across the 9 non-macro runs (always found by iteration 2, never surpassed):

```
if And(IsZero(7), And(IsZero(2), IsZero(0))):
  Flip(6)    # PICK key 0
else:
  Flip(1)    # MOVE to location 1
```

### Execution trace on initial state [1,0,0,0,0,0, 1,0,0, 1,1]

| Step | Obs[0] | Obs[7] | Obs[2] | Condition | Action | Effect |
|------|--------|--------|--------|-----------|--------|--------|
| 1 | 1 (at loc 0) | 0 (room 1 locked) | 0 (not at loc 2) | And(T, And(T, **F**))=**F** | Flip(1)=MOVE(1) | Agent moves to loc 1 |
| 2 | 0 | 0 | 0 | And(T, And(T, T))=**T** | Flip(6)=PICK(0) | Picks key 0, unlocks room 1 |
| 3-25 | 0 | 1 | 0 | And(**F**, ...)=**F** | Flip(1)=MOVE(1) | Moves to loc 1 (already there) → NOOP |

Wait — after step 2, obs[7] becomes 1 (room 1 unlocked), making IsZero(7)=FALSE. So the condition becomes FALSE for all remaining steps, and the agent keeps doing Flip(1)=MOVE(1) to location 1 forever. It gets stuck oscillating.

**Raw reward**: 25 × (-0.01) + 0.1 (unlock bonus) = -0.15.

**Why this is the ceiling**: The program has only 1 Ite branch. It can encode at most 1 conditional behavior. D=3 requires 4 conditional behaviors (2 moves + 2 picks). To improve beyond -0.15, the search must discover a program with 2+ correctly-sequenced Ite rules — but MCTS never finds one in 43K programs explored.

### 8b. Best Program: factored_macro_D3_seed42

The macro grammar finds a more complex best program at iteration 1 (never surpassed):

```
if And(IsZero(7), IsZero(9)):                                            Flip(8)  # NOOP
elif And(Not(IsZero(0)), And(And(IsZero(5), Not(IsZero(6))), IsZero(1))): Flip(1)  # MOVE_TO(1)
elif And(Not(IsZero(3)), Not(IsZero(10))):                                Flip(7)  # PICK(1)
else:                                                                     Flip(6)  # PICK(0)
```

This has 3 Ite + 1 Default — enough structure to potentially encode a multi-step plan. But the conditions and actions are wrong:

| Step | State | Branch | Action | Effect |
|------|-------|--------|--------|--------|
| 1 | At loc 0, rooms 1,2 locked, both keys avail | Branch 2 | MOVE_TO(1) | Go to key 0's location |
| 2 | At loc 1, key 0 available | Branch 4 (else) | PICK(0) | Pick key 0, unlock room 1 (+0.1) |
| 3-25 | At loc 1, key 0 used, key 1 avail | Branch 4 (else) | PICK(0) | **Fails**: key 0 used. Silent NOOP × 23 |

**Raw reward**: -0.15 (same as non-macro best). Despite having 3 branches, only 2 are ever triggered. Branch 3 (`Not(IsZero(3))` = "at location 3") is never reached because the program never moves to loc 3. The else branch locks the agent into repeated failed PICK(0) attempts.

**Significance**: The macro grammar enables more complex programs to be found (3 rules vs 1 rule), but without a reward signal to distinguish "almost correct 3-rule program" from "random 3-rule program," there is no evolutionary pressure toward the correct configuration.

---

## 9. Why Program Count Varies Between Iterations

The `unique_programs` count is cumulative. New programs per iteration:
```
Iter 1→2:   +2,217      Iter 13→14: +1,135      Iter 18→19: +2,209
Iter 2→3:   +3,690      Iter 14→15: +2,815      Iter 19→20: +130
Iter 3→4:   +2,665      Iter 15→16: +2,449
Iter 4→5:   +3,071      Iter 16→17: +494
```

Three mechanisms:

1. **Cache saturation**: The leaf evaluator caches all evaluated programs. As the cache grows (6K → 108K entries), more MCTS rollouts rediscover cached programs. Fewer rollouts reach novel programs.

2. **Policy network drift**: After training, the updated policy network shifts the MCTS prior. Some updates push toward unexplored derivation regions (e.g., iter 2→3: +3,690 new). Others reinforce known derivation paths (e.g., iter 16→17: +494 new). After mode collapse, the policy generates mostly trivial Default programs (+130 new at iter 20).

3. **Dirichlet noise stochasticity**: Each of the 20 games per iteration gets fresh Dirichlet noise at the MCTS root. Random noise samples sometimes push toward novel branches, sometimes toward well-explored ones. This creates per-iteration variance.

---

## 10. The Broken Learning Loop

```
           ┌─────────────────────────────────────────┐
           │                                         │
           ▼                                         │
  Programs evaluated ──→ 87-99.75% score -0.075      │
           │                                         │
           ▼                                         │
  Value network learns constant -0.075               │
           │                                         │
           ▼                                         │
  MCTS Q-values ≈ constant ──→ no search guidance    │
           │                                         │
           ▼                                         │
  Visit counts ≈ noise ──→ policy trains on noise    │
           │                                         │
           ▼                                         │
  Policy produces arbitrary programs ────────────────┘
           │
           ├──→ Without macros (after ~15 iters):
           │    Mode collapse to 2-step Default(Flip(x))
           │
           └──→ With macros:
                No mode collapse, but sustained exploration
                still fails to bridge the reward cliff
                (13% unlock 1 room, 0% solve puzzle)
```

**The macro grammar breaks the mode collapse sub-loop** but not the main loop. It proves that the fundamental bottleneck is not search space size or policy degeneration — it is the flat reward landscape. Even with perfect domain knowledge encoded in the grammar, the value network cannot learn anything useful from a landscape where 87% of outcomes are identical.

---