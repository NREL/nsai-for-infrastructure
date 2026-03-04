# Situation Audit: AlphaZero for Grammar-Guided Program Synthesis

_This document audits the current state of a research system that applies AlphaZero to program synthesis via grammar derivation. Its primary finding is a fundamental mismatch between the target problem (bitstring) and the grammar's expressiveness. It also catalogues the key technical design questions in the AlphaZero machinery that remain relevant for any future problem. Intended as context for an LLM or researcher planning the next iteration._

---

## Table of Contents

1. [System Summary](#1-system-summary)
2. [The Core Problem: Grammar-Problem Mismatch](#2-the-core-problem-grammar-problem-mismatch)
3. [Technical Design Questions](#3-technical-design-questions)
4. [Summary of Priorities](#4-summary-of-priorities)

---

## 1. System Summary

### What the system does

The system casts **program synthesis** as a **single-player game** and applies **AlphaZero** (MCTS + neural network) to search the space of grammar derivations:

- **State** = partial AST with "holes" (unfilled subtrees)
- **Action** = a grammar production that fills the leftmost hole
- **Terminal reward** = quality of the completed program (measured by running it on frozen evaluation states)
- **Neural network** = Transformer encoder (d=64, 2 layers, 4 heads) that reads the partial AST and outputs (policy over productions, value estimate)

### The two-level MDP

```
OUTER MDP (DerivationGame):              INNER MDP (BitStringGym):
  State: partial AST with holes            State: N-bit binary string
  Action: grammar production               Action: flip one bit
  Reward: 0 (non-terminal),               Reward: potential difference / N
          leaf_eval (terminal)
  Terminal: complete program               Terminal: all bits = 1

  The outer MDP produces a program.
  The inner MDP evaluates that program
  by running it as a policy on frozen
  initial states.
```

### Current target problem

The **BitString game**: an agent operates on an N-bit binary string (initially with `n_ones` bits set to 1). At each step, the agent flips one bit. The goal is all-ones. A potential function (onemax, leading_ones, or binval) provides shaped reward.

### The DSL

Decision-list programs of the form:

```
if IsZero(0): Flip(0)
elif IsZero(1): Flip(1)
elif And(IsZero(2), IsZero(3)): Flip(2)
else: Flip(5)
```

Primitives: `IsZero(j)`, `Not(c)`, `And(c1, c2)`, `Flip(j)`, `Ite(cond, action, else_prog)`, `Default(action)`.

### File map

| Component | Key files |
|-----------|-----------|
| Core AlphaZero | `core/game.py`, `core/mcts.py`, `core/agent.py`, `core/policy_value_net.py` |
| DSL & Grammar | `dsl/ast_nodes.py`, `dsl/budget_grammar.py`, `dsl/derivation.py`, `dsl/derivation_game.py` |
| Network | `dsl/derivation_network.py` |
| Evaluation | `dsl/leaf_evaluator.py`, `dsl/interpreter.py` |
| Training | `training/trainer.py`, `training/evaluator.py`, `training/gated_trainer.py` |
| Scan grammar | `dsl/scan_grammar.py`, `dsl/scan_derivation_game.py`, `dsl/scan_network.py` |
| Config | `dsl/derivation_config.py`, `dsl/game_config.py` |
| Entry point | `scripts/run_derivation.py` |

All paths relative to `src/alphazeropp/instances/bitstring/`.

### Key hyperparameters

| Parameter | Value | Component |
|-----------|-------|-----------|
| budget | 14 | Grammar (CFG mode) |
| n_sites (N) | 6 | Problem |
| n_ones | 2 | Problem |
| potential | onemax | Reward shaping |
| metric | avg_reward | Leaf evaluation |
| n_frozen_states | 1 | Leaf evaluation |
| d_model | 64 | Network |
| n_layers / n_heads | 2 / 4 | Network |
| learning_rate | 3e-4 | Training |
| batch_size | 32 | Training |
| policy_weight | 2.0 | Loss function |
| n_simulations | 200 | MCTS |
| temperature | 1.0 | MCTS |
| c_exploration | 1.5 | MCTS |
| dirichlet_alpha / epsilon | 0.25 / 0.40 | MCTS noise |
| n_games_per_train | 40 | Trainer |
| n_past_iterations | 20 | Replay buffer |
| eval n_games | 20 | Evaluator |
| accept_threshold | 0.55 | Gating |
| n_iterations | 30 | Run |

### Key quantities

```
Total programs (budget=14, N=6):     151,173,432
Canonical programs (after pruning):  37,463,688
Action space (max productions):      48
Derivation depth:                    ~9 steps
Programs evaluated per iteration:    ~360
Coverage per iteration:              ~0.0002% of program space
Replay buffer:                       ~7,200 examples (20 iters x ~360)
Network parameters:                  ~50K
```

---

## 2. The Core Problem: Grammar-Problem Mismatch

### 2.1 The bitstring problem does not require the grammar's expressiveness

The DSL can express compound conditions: `And(IsZero(0), IsZero(1))`, negations: `Not(IsZero(0))`, and arbitrarily nested condition trees. But the **optimal policy for all three bitstring variants never needs any of this**.

For each potential function, the optimal policy is a **flat permutation scan** — check bits in some priority order, flip the first zero:

| Variant | Optimal strategy | Optimal program structure |
|---------|-----------------|--------------------------|
| OneMax | Flip any zero bit | `if IsZero(σ(0)): Flip(σ(0)) elif IsZero(σ(1)): Flip(σ(1)) ... else: Flip(σ(N-1))` for **any** permutation σ |
| LeadingOnes | Flip leftmost zero | Same structure, σ = identity (0, 1, 2, ..., N-1) |
| BinVal | Flip highest-weight zero | Same structure, σ = identity (MSB-first) |

**Formal claim**: For all three BitString potentials, the optimal decision-list program uses only `IsZero(j)` conditions — never `Not`, `And`, or any compound condition. Every `Ite` branch tests a single bit and flips that same bit. The `Not` and `And` nodes in the grammar contribute zero expressive value; they only inflate the search space.

**Evidence**: The scan grammar (which generates only permutation-scan programs) achieves optimal results. For N=3, it solves in 1 iteration. For N=6, it converges to optimal avg_reward of 0.333. This proves the full CFG is unnecessary.

### 2.2 The Goldilocks failure: neither grammar demonstrates the paradigm

**CFG grammar** — too much irrelevant expressiveness:

- 151M programs at budget=14, N=6 (37M after canonical pruning)
- Action space of 48 productions per step, derivation depth ~9
- MCTS with 200 simulations explores ~360 programs per iteration (0.0002% coverage)
- The optimal programs exist somewhere in this space, but are buried under tens of millions of programs containing useless `Not(Not(...))`, `And(IsZero(0), IsZero(0))`, contradictions, and unreachable branches
- The grammar is a needle-in-a-haystack: 99.99%+ of the search space is programs with unnecessary compound conditions
- The neural network and MCTS must learn to *avoid* the grammar's own expressiveness — they must learn that `Not` and `And` are never useful

**Scan grammar** — too little search challenge:

- N! = 720 programs for N=6 (all are semantically distinct)
- Action space of N choices per step, derivation depth N-1
- MCTS trivially covers the space: 200 simulations in a tree with 720 leaves
- Solves N=3 in 1 iteration; converges to optimal for N=6 quickly
- But the "grammar" is just picking a permutation — there is no meaningful grammar structure, no condition synthesis, no tree-building. The AlphaZero machinery is overkill. A simple evolutionary search or beam search over permutations would work equally well.

**Neither demonstrates the paradigm**. The paradigm — AlphaZero guiding grammar-based program synthesis — requires a problem where the grammar's expressiveness is actually *needed* to solve the problem, and where the search space is large enough that intelligent guidance matters.

### 2.3 What a good problem requires

The paradigm requires a problem satisfying three conditions:

**Requirement 1: Grammar expressiveness is necessary**

The optimal policy must require compound conditions — `And`, `Not`, `Or`, or nested condition trees — not just flat single-variable checks. A flat decision list of the form `if cond_1: action_1 elif cond_2: action_2 ...` where each `cond_i` tests a single state variable must be *strictly worse* than a program using compound conditions.

*Validation test*: Write the best flat decision list (each branch tests one variable). Write a program using compound conditions. If the compound program achieves strictly higher reward on some class of initial states, Requirement 1 is met.

**Requirement 2: The search space is structured**

Partial derivations must carry a value signal — early grammar choices (e.g., the first `Ite` branch's condition structure) should meaningfully predict the quality of the completed program. This is what enables the value head to learn and MCTS to guide search.

*Validation test*: Consider two partial derivations at the same depth. One has committed to a structurally good first branch. The other is random. If the first has significantly higher expected terminal reward over all completions, the value head has something to learn.

**Requirement 3: The search space is non-trivial**

Random search must fail. The number of valid programs at a reasonable budget must be large enough that 200 MCTS simulations cannot cover a meaningful fraction, but not so large that no useful signal can be found.

*Validation test*: Estimate the program count at a reasonable budget. Compute the fraction that achieve near-optimal reward. If this fraction is < 0.01%, random search will almost certainly fail within 200 simulations. If the count is also > 10K, the search is non-trivial.

### 2.4 Why this matters

The entire AlphaZero machinery — MCTS tree search, Transformer policy-value network, self-play training loop, gated evaluation — cannot be properly evaluated on a problem that does not exercise the grammar. Specifically:

- **The value head** learns to predict program quality from partial ASTs. If compound conditions are never useful, the value head's predictions for partial ASTs containing `And` or `Not` holes are never tested in a meaningful way.
- **The policy head** learns which grammar productions to prefer. If the optimal production sequence never uses `Not` or `And`, the policy head must learn to *avoid* most of the action space — a degenerate learning problem.
- **MCTS exploration** is designed to balance breadth and depth in a structured search tree. If the structure (compound conditions) is irrelevant, MCTS is just doing random search with extra overhead.
- **Any improvements** to MCTS (Q-normalization, tree reuse), training (loss weighting, replay buffer), or architecture (Transformer depth, embedding) are confounded by the problem being wrong. We cannot know whether a change improves the *paradigm* or just happens to find bitstring permutations faster.

**Bottom line**: The first priority is finding a problem where the grammar's expressiveness is genuinely necessary. All other technical improvements are secondary.

---

## 3. Technical Design Questions

These remain relevant regardless of which problem is chosen. They describe the current state of the AlphaZero machinery and identify areas for investigation once the problem-grammar mismatch is resolved.

### 3.1 MCTS search

**Q-normalization via dynamic min-max**

Q-values are normalized to [0, 1] using the global min and max Q-values observed during the current `perform_simulations` call (reset each call):

```python
# mcts.py — UCB calculation
if self.q_max > self.q_min:
    q_normalized = (q - self.q_min) / (self.q_max - self.q_min)
else:
    q_normalized = 0.5
```

Concerns:
- In early simulations (1-2 Q-values exist), the min-max range is narrow — small differences are amplified to span [0, 1], potentially causing over-commitment to the first actions explored.
- A single outlier leaf evaluation can shift `q_max` or `q_min` dramatically, destabilizing the exploration-exploitation balance.
- MuZero uses running mean and std for normalization. Would that be more robust?

**Simulation budget vs search tree size**

With 200 simulations and branching factor ~48, the full search tree has ~48^9 ~ 10^15 leaf nodes. Only 200 paths are explored per step. The neural network prior must do the heavy lifting to focus search. If the prior is near-uniform (early training), MCTS is essentially random search.

**Sparse terminal-only reward and the value bootstrap**

All intermediate derivation steps return reward=0. Only terminal steps receive reward from `LeafEvaluator`. The value head at unexplored nodes is the *only* signal for deep tree branches. But the value head is trained on noisy targets (Section 3.3). This creates a chicken-and-egg: the value head needs good MCTS data, but good MCTS needs a good value head.

**Dirichlet noise**

Noise is injected once at the root with alpha=0.25 and epsilon=0.40. For 48 actions, this produces concentrated noise (most entries near-zero, a few large). The noise is fixed for all 200 simulations. This may provide insufficient exploration diversity for a large action space. The noise parameters are not annealed during training.

### 3.2 State representation and neural architecture

**Flat preorder encoding for tree-structured data**

The partial AST is encoded as a fixed-size float32 array via preorder traversal: each node becomes a `(type_id, parameter)` pair, padded to `2 * budget` floats. This loses explicit tree structure — depth, parent-child relationships, and sibling positions are not encoded. The Transformer must learn to reconstruct tree structure from positional embeddings alone.

**CLS pooling may lose positional information**

Both the policy and value heads are derived from the single [CLS] token. The policy needs position-specific information (which production to apply at the leftmost hole), but CLS summarizes the entire tree. Information about the specific hole being expanded may be diluted.

**Value head expressiveness**

The value head is a single `Linear(64, 1)` layer. For predicting program quality from a 64-dim representation of a partial AST, this may be too simple. A small MLP (64 -> 64 -> 1 with ReLU) could help.

**Device transfer overhead**

`predict()` calls `self.model.cpu()` on every invocation. With ~72,000 predict calls per iteration (200 sims x 9 steps x 40 games), this causes unnecessary CPU<->CUDA transfers if the model was on CUDA during training.

### 3.3 Reward signal and value targets

**Uniform value targets across episode steps**

With `reward_discount = 1.0` and intermediate rewards all 0, every step in an episode has the same value target = R_terminal:

```
Episode rewards: [0, 0, 0, 0, 0, 0, 0, 0, R]
Discounted returns: [R, R, R, R, R, R, R, R, R]
```

The value head sees `ProgramHole(14)` at step 0 and a nearly-complete AST at step 7, both with the same target. This makes it difficult to learn that "closer to completion" is more informative.

Options:
- Use `reward_discount < 1.0` (e.g., 0.95) to create a temporal gradient
- Use MCTS root value estimates as training targets instead of episode returns
- Add small intermediate rewards for progress (e.g., filling holes)

**Single frozen evaluation state**

The default `n_frozen_states=1` means every program is evaluated on exactly one initial state. For OneMax (symmetric in bit positions), this may not cause overfitting. But for leading_ones or binval, a single evaluation state is not representative. Increasing `n_frozen_states` would improve robustness at the cost of slower leaf evaluation.

### 3.4 Training loop

**Policy loss weighted 2x over value loss**

```python
loss = loss_value + 2.0 * loss_policy   # policy_weight = 2.0
```

In standard AlphaZero, the weighting is typically 1:1. Given that the value signal is already sparse (uniform targets per episode), a 2x policy weight may further suppress value learning. The policy targets (MCTS visit distributions) from a mostly-untrained network may not warrant this emphasis.

**No gradient clipping**

`loss.backward()` followed directly by `optimizer.step()` with no `clip_grad_norm_`. With small batch sizes (32) and noisy targets, gradient spikes are possible.

**Small dataset relative to model capacity**

~7,200 training examples for ~50K parameters gives ~7 parameters per example. This is high compared to typical deep learning (where 0.1-1 parameters per example is more common), raising overfitting risk. There is no validation loss tracking to detect this.

**No learning rate scheduling**

The learning rate stays at 3e-4 for all 30 iterations. Decay (cosine annealing, step decay) could improve convergence in later iterations.

### 3.5 Evaluation and gating

**Statistical power of 20 evaluation games**

With 20 games, each game contributes 5% (win) or 2.5% (tie) to the score. The acceptance threshold is 0.55. A marginally worse network could pass by chance. Empirically, the gate has accepted 8/9 iterations and rejected 1 (score 0.50), so it is not a no-op.

**Near-deterministic evaluation**

Evaluation uses temperature 0.05 (near-greedy) with no Dirichlet noise. Outcomes are largely determined by the MCTS random seed. Whether the 20 different seeds provide sufficient variation is unclear.

**Program synthesis evaluation via game comparison**

Two agents with different neural policies may find the *same* final program through different derivation paths, resulting in a tie even when one agent has a genuinely better search strategy. The gate measures program *outcome* quality, not search *efficiency*.

### 3.6 Action space design

**Mixed structural and parametric decisions**

Each action is a single integer indexing a production that simultaneously encodes the structural template (Default vs Ite, condition budget) and the parameter (which Flip index, which IsZero index). For example:

```
Action 0:  P(14) -> Ite(C(1), Flip(0), P(11))   <- structure: i=1, param: j=0
Action 1:  P(14) -> Ite(C(1), Flip(1), P(11))   <- structure: i=1, param: j=1
...
Action 6:  P(14) -> Ite(C(2), Flip(0), P(10))   <- structure: i=2, param: j=0
```

A hierarchical decomposition (first decide the structure template, then decide parameters) would reduce the effective branching factor at each level.

**Budget-exact constraint**

Every derivation must use exactly `budget` nodes. A program that solves the problem in 5 nodes cannot be discovered if the budget is 14. This forces the learner to "waste" budget on unnecessary `Not` wrappings or compound conditions. A "max-budget" mode exists as an alternative but increases the action space from 48 to 66.

**Missing primitives**

The condition language has `{IsZero, Not, And}` but no `IsOne` or `Or`. Expressing "bit j is 1" costs 2 nodes (`Not(IsZero(j))`). Expressing "a or b" costs 5 nodes via De Morgan's law. These gaps consume budget that could be used for more branches. Whether this matters depends on the target problem — for bitstring, it doesn't (optimal programs never need `Or`), but for a future problem it might.

### 3.7 Previously resolved items

The following issues from earlier audits have been resolved and are noted here for completeness:

- **Dead-end grammar productions** (budgets 3 and 4): Eliminated by filtering on `count_programs() == 0` in `_program_productions()`. Action space reduced 60 -> 48.
- **Double-negation redundancy**: Banned via `parent_is_not` flag on `ConditionHole`.
- **And commutativity redundancy**: Enforced via canonical ordering `i <= k-1-i` in the And production loop.
- **MCTS tree reuse**: Implemented in `perform_simulations_reuse()` with correct Dirichlet noise re-injection via `nn_policy_original`.

---

## 4. Summary of Priorities

### Priority 1: Find the right problem

The bitstring problem does not exercise the grammar's expressiveness. The optimal policy is a flat permutation scan that never uses `Not`, `And`, or compound conditions. Any conclusions drawn from the current system are confounded by this mismatch.

**Action**: Select a new target problem where:
- Compound conditions are necessary for optimal or near-optimal policies
- The search space at reasonable budgets is large enough for MCTS to matter (~10K-1M programs)
- Partial derivations carry value signal (early choices predict final quality)

Candidate domains from PDDL: Sokoban, Blocks World, Keys and Doors, Towers of Hanoi, Gripper, etc. A separate analysis is needed to evaluate these candidates.

### Priority 2: Address technical issues in the AlphaZero machinery

Once a suitable problem is selected, the following technical questions should be investigated:

| Issue | Section | Impact estimate |
|-------|---------|----------------|
| Uniform value targets (all steps = R_terminal) | 3.3 | High — value head may not learn useful gradients |
| Simulation budget vs tree size (200 in 10^15) | 3.1 | High — depends on neural prior quality |
| Mixed structural/parametric action space | 3.6 | Medium — hierarchical decomposition could help |
| Policy weight 2x suppressing value learning | 3.4 | Medium — easy to experiment with |
| Q-normalization stability | 3.1 | Medium — could cause early over-commitment |
| Single frozen evaluation state | 3.3 | Low-medium — depends on problem symmetry |
| CLS pooling losing positional info | 3.2 | Low-medium — may need conditioned policy head |
| No gradient clipping or LR schedule | 3.4 | Low — standard engineering improvements |

### Priority 3: Scale testing

Once problem-grammar alignment is achieved and technical issues are addressed, test scaling behavior: larger budgets, larger state spaces, longer derivation sequences. The current system has only been validated at small scales (N=3 solves trivially, N=6 converges slowly).
