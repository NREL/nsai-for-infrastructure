# Grammar-Guided Program Synthesis: Search Algorithm Comparison

> **Purpose**: Self-contained specification for comparing search/optimization algorithms within the grammar-guided derivation game for program synthesis. All algorithms operate on the same game formulation (leftmost-hole AST expansion). Includes full problem specification, failure diagnosis of the current AlphaZero approach, algorithm designs, and experiment plan. Intended to be fed into an LLM for experiment roadmap generation.

---

## 1. The Derivation Game

### Overview

Program construction is formalized as a single-player sequential decision game. The agent builds an Abstract Syntax Tree (AST) one grammar production at a time. At each step, it expands the **leftmost hole** in a partial AST by choosing a production rule. When all holes are filled, the complete program is executed and scored.

### Goal

Synthesize a reactive control policy (as an interpretable if-then-else program) for the Doors grid-world navigation task. The synthesized program takes an observation vector and outputs an action.

### States = Partial ASTs

A state is a partial AST where some leaves are **holes** (unfilled subtrees). Each hole has a **type** (ProgramHole or ConditionHole) and a **budget** (maximum remaining nodes).

```
Initial state:    ProgramHole(budget=24)

After 1 step:     Ite(ConditionHole(5), Flip(?), ProgramHole(17))

After 3 steps:    Ite(And(Not(IsZero(1)), ConditionHole(1)), Flip(6), ProgramHole(17))

Terminal:          Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6),
                    Ite(IsZero(7), Flip(3), Default(Flip(1))))
```

### Actions = Grammar Productions

At each step, the agent chooses a production rule to expand the leftmost hole. Legal productions depend on hole type, remaining budget, and grammar constraints.

**For ProgramHole(k):**
- `Default(Flip(j))` for each action j -- costs 2 nodes, requires k >= 2
- `Ite(ConditionHole(i), Flip(j), ProgramHole(k-2-i))` for each condition budget i and action j -- costs 2+i nodes, requires k >= 5

**For ConditionHole(k):**
- `IsZero(j)` for each observation index j -- costs 1 node, requires k >= 1
- `Not(ConditionHole(k-1))` -- costs k nodes, requires k >= 2, banned when parent is Not
- `And(ConditionHole(i), ConditionHole(k-1-i))` for each split i -- costs k nodes, requires k >= 3

**Branching factor**: Varies per step. Typically 20-150 legal productions depending on hole type and budget. Maximum precomputed as `max_productions` across all possible hole states.

### Observation Encoding

Fixed-size float32 vector of shape `(2 * budget,)`. Preorder traversal of partial AST encoded as `(node_type_id, parameter)` pairs:

| Type ID | Node | Parameter |
|---------|------|-----------|
| 0 | PAD | 0 (unused slots) |
| 1 | Flip | action index j |
| 2 | IsZero | observation index j |
| 3 | Not | 0 |
| 4 | And | 0 |
| 5 | Ite | 0 |
| 6 | Default | 0 |
| 7 | ProgramHole | remaining budget |
| 8 | ConditionHole | remaining budget |

Observation: `obs[2*i] = type_id, obs[2*i+1] = parameter` for each node i in preorder.

### Reward Pipeline

```
Non-terminal steps:  reward = 0.0
Terminal step:       reward = LeafEvaluator(complete_program)
Dead ends:           reward = 0.0, episode truncated
```

The LeafEvaluator:
1. Compiles the complete AST into an executable program
2. Runs the program on frozen initial Doors states (deterministic evaluation)
3. Computes metrics: solve_rate, avg_reward, keys_progress
4. Applies selected metric function to produce a scalar reward

**Available metrics:**
- `avg_reward`: Mean cumulative reward across evaluation episodes
- `solve_rate`: Fraction of episodes where goal is reached
- `weighted`: `alpha * solve_rate + (1-alpha) * avg_reward` when solve_rate > 0, else raw avg_reward
- `keys_progress`: `keys_picked/K + 0.1 * avg_reward` (domain-specific milestone metric)
- `penalized_reward`: `avg_reward - lambda * (interp_ops / max_ops)`

**Caching**: Programs with identical AST structure return cached values. Avoids re-evaluation of equivalent derivation paths.

### Episode Structure

- Episode length: ~10-20 steps (depends on budget and grammar choices)
- Average branching factor: ~50-100 productions per step
- Each complete derivation produces exactly one program
- Programs are deterministically evaluated (no stochastic reward noise)

---

## 2. The Doors Domain (Target Task)

### Environment

D rooms connected by locked doors. Agent navigates from Room 0 to goal in Room D-1, collecting keys along the way.

**State vector** (size `n_sites = M + 2D - 1`, where `M = D * locs_per_room`):
- `[0:M]`: Agent location (one-hot)
- `[M:M+D]`: Room unlock status (binary)
- `[M+D:M+2D-1]`: Key availability (binary)

**Actions** (size `M + K + 1` where `K = D - 1`):
- `MOVE_TO(l)`: Move to location l (requires room unlocked)
- `PICK(k)`: Pick key k (requires at key location AND key available)
- `NOOP`: Do nothing

**Per-step rewards during program execution**:
- `-0.01` step penalty
- `+0.1` unlock bonus
- `+1.0` goal bonus (terminates)

### Optimal Program (D=3, 22 nodes)

```
if And(Not(IsZero(1)), Not(IsZero(9))):    # At key0 loc AND key0 available
    Flip(6)                                  # PICK key0
elif And(Not(IsZero(3)), Not(IsZero(10))):  # At key1 loc AND key1 available
    Flip(7)                                  # PICK key1
elif IsZero(7):                              # Room 1 locked
    Flip(3)                                  # MOVE to key1 loc
elif IsZero(8):                              # Room 2 locked
    Flip(5)                                  # MOVE to goal loc
else:
    Flip(1)                                  # MOVE to key0 loc
```

**Optimal return**: 1.0 (goal) + 2*0.1 (unlocks) - 5*0.01 (steps) = 1.15

### Optimal Node Counts and Budget

```
optimal_nodes = 10*(D-1) + 2
current_budget = ceil(1.5 * optimal_nodes) rounded to even
```

| D | n_sites | Optimal nodes | Current budget | Recommended budget |
|---|---------|--------------|----------------|--------------------|
| 2 | 7 | 12 | 18 | 14 |
| 3 | 11 | 22 | 34 | 24 |
| 5 | 19 | 42 | 64 | 44 |
| 10 | 39 | 92 | 138 | 94 |
| 20 | 79 | 192 | 288 | 194 |

### Search Space Size

Number of syntactically valid programs grows super-exponentially with budget:

| Budget | Programs (D=3, n_sites=11) | Notes |
|--------|---------------------------|-------|
| 5 | ~1,342 | Single conditional |
| 10 | ~6.7 x 10^6 | Small chains |
| 14 | ~1.6 x 10^10 | Feasible exhaustive |
| 22 | ~9.9 x 10^16 | Minimum budget for D=3 |
| 24 | ~5.1 x 10^18 | +2 headroom |
| 34 | ~2.1 x 10^30 | Current budget (D=3) |

Each +2 budget multiplies search space by ~50x. Each +1 node adds ~7x.

---

## 3. Current System Diagnosis (AlphaZero)

### Configuration (D=3 experiment)

- Budget: 34, metric: weighted (alpha=0.7)
- MCTS: 80 simulations, c_exploration=1.5, backup_rule=max
- Rollouts: 4 per leaf, mode=max, blend=0.3 (70% rollout + 30% network)
- Training: 30 iterations, 30 games/iter, 20-iteration replay window
- Network: Transformer, d_model=64, 2 layers, 4 heads

### Empirical Results (100-iteration extended run)

| Metric | Iteration 1 | Iteration 100 | Change |
|--------|-------------|---------------|--------|
| best_avg_reward | 1.75 | 1.75 | 0.0 |
| best_solve_rate | 0% | 0% | 0.0 |
| unique_programs | 16,735 | 688,224 | +41x |
| policy_loss | 2.10 | 0.62 | -70% |
| value_loss | 0.18 | 0.22 | +22% |

**Best program found at iteration 1** (by MCTS with a random network) and never surpassed.

### Diagnosed Bottlenecks

**B1: Undifferentiated value targets.** Terminal-only reward with gamma=1.0 means every step in an episode gets the same value target `z_t = terminal_reward`. The value head learns V(s) ~= 1.3 for all partial ASTs -- a useless constant. No credit assignment to individual derivation decisions.

**B2: Sparse reward in vast space.** Budget=34 gives 2.1 x 10^30 programs. MCTS with 80 sims explores a fraction ~10^-29 per decision. Reward observed only at complete programs. MCTS degenerates to ~3 random completions per step.

**B3: Flat reward landscape.** When solve_rate=0, the metric collapses to avg_reward. Many random programs score 1.0-2.0 by accidentally collecting keys. Mediocre programs are indistinguishable from near-solutions.

**B4: Max backup noise.** Backup_rule=max means a single lucky rollout makes a branch permanently look good. Noisy Q-values, reduced exploration, bad value head training signal.

**B5: Budget bloat.** Budget=34 gives 12 extra nodes beyond the 22-node optimum. Search space bloat = 7^12 ~= 10^13 x beyond minimum budget.

### Root Cause

The AlphaZero self-improvement loop cannot close:
1. Value function learns V(s) ~= constant (useless)
2. MCTS gets no guidance -> search is effectively random
3. Random search in 10^30 space finds nothing better
4. Training data quality plateaus -> network doesn't improve -> cycle stagnates

---

## 4. Algorithm Catalog

### 4.1 Random Search (Baseline)

**What it does**: Repeatedly sample complete programs by making random production choices at each derivation step. Return the best program found.

**Design**:
- At each step, uniformly sample from legal productions
- Complete derivation -> evaluate program -> record if best
- No learning, no state, purely sampling

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total samples | 10K / 100K / 1M programs | Budget tiers |
| Sampling | Uniform over legal productions | Unbiased baseline |

**Strengths**: Zero overhead. Embarrassingly parallel. Establishes a floor: any learning algorithm must beat this.

**Weaknesses**: No learning. Probability of sampling the optimal program = 1/|search_space|. For budget=24 with 5.1 x 10^18 programs, expected 10^18 samples to find optimum.

**Expected**: Finds mediocre programs (reward 1.0-2.0) quickly. Never finds optimal program at budget >= 22. Performance ceiling depends on budget: at budget=14 (~10^10 programs), 1M samples covers 10^-4 of space. At budget=24, covers 10^-12.

---

### 4.2 Pure MCTS (No Neural Network)

**What it does**: Standard MCTS with random rollouts for leaf evaluation (no learned network). Builds a search tree over derivation steps, uses UCB for selection, random completions for value estimation.

**Design**:
- Selection: UCB with uniform prior (no policy network)
- Expansion: Expand one child at a time
- Evaluation: Random rollout to terminal, evaluate complete program
- Backup: Mean of rollout values

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Simulations/episode | 500, 2000, 10000 | Need many sims without network guidance |
| c_exploration | 1.0-2.0 | Standard UCB |
| Rollout mode | max over 4-8 rollouts | Best of several random completions |
| Backup rule | mean | Stable estimates |
| Total episodes | 100-1000 | Each episode = one game from root |
| Temperature | 0.5 | Slightly greedy action selection |

**Strengths**: No training required. MCTS provides structured exploration with UCB. Random rollouts give unbiased value estimates. More focused than pure random search.

**Weaknesses**: No learning across episodes. Each episode starts from scratch. Rollout quality degrades with budget (longer random programs are less likely to be good). Without a prior policy, UCB explores uniformly across all productions.

**Expected**: Better than random search (MCTS focuses on promising branches within each episode). Worse than AlphaZero at early iterations (no prior) but avoids AlphaZero's stagnation problem (no self-reinforcing bad learning). May find good programs if given enough simulations per episode.

**Key experiment**: Compare pure MCTS with N=10000 sims against AlphaZero with N=80 sims over 30 iterations. Does raw search power beat learned-but-stagnant guidance?

---

### 4.3 AlphaZero (Current System)

**What it does**: MCTS guided by a policy-value network. Network trained iteratively from MCTS-generated search targets.

Fully specified in Section 3 above. Current system fails to improve beyond iteration 1.

**Variants to test** (same AlphaZero algorithm, different configurations):

| Variant | Change | Expected Impact |
|---------|--------|-----------------|
| AZ-budget24 | budget 34 -> 24 | Search space reduced ~10^13 x |
| AZ-mean-backup | backup_rule: max -> mean | Smoother Q-values, better value learning |
| AZ-keys-progress | metric: weighted -> keys_progress | Better reward gradient |
| AZ-dead-end-penalty | dead_end reward: 0 -> -1 | Network learns to avoid dead ends |
| AZ-combined | All above together | Addresses all diagnosed bottlenecks |
| AZ-more-sims | n_simulations: 80 -> 400 | More search per step (expensive) |

---

### 4.4 REINFORCE / Policy Gradient (No Search)

**What it does**: Train a policy network `pi(production | partial_AST)` directly using policy gradient (REINFORCE). No MCTS. At each step, sample a production from the network's distribution, complete the derivation, compute return, update policy with `grad = return * grad(log pi)`.

**Design**:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Network | Transformer (match AlphaZero architecture) | Fair comparison |
| Learning rate | 1e-4 to 3e-4 | Standard for PG |
| Baseline | Moving average of returns | Variance reduction |
| Entropy bonus | 0.01-0.05 | Prevent premature collapse |
| Batch size | 50-100 programs per update | Reduce gradient variance |
| Total programs | 100K-1M | Needs many samples |
| Temperature | 1.0 (training), 0.1 (eval) | Explore during training |

**Key difference from AlphaZero**: No search at decision time. The policy network must learn everything from gradient signal. Much cheaper per program (~1 forward pass per step vs 80 MCTS sims), but noisier learning signal.

**Strengths**: Simple. Very cheap per program (orders of magnitude faster than MCTS). Can evaluate 100x more programs per wall-clock hour. Policy gradient directly optimizes expected return.

**Weaknesses**: High variance with terminal-only reward (same z_t = terminal_reward for all steps -- the REINFORCE baseline can't fix this fundamental issue). No lookahead. Likely collapses to a single program mode quickly (despite entropy bonus). Gradient is noisy because credit assignment through 15-20 steps with sparse reward is hard.

**Expected**: Similar stagnation to AlphaZero's value head problem. The policy receives the same undifferentiated signal. However, exploring 100x more programs per wall-clock hour might find better programs by volume alone. The interesting comparison: does REINFORCE with 100K programs beat AlphaZero with 1K MCTS-guided programs?

**Variant -- REINFORCE + reward shaping**: Add intermediate rewards (e.g., from random completion estimates at each step). This directly addresses the sparse reward problem and may make REINFORCE competitive.

---

### 4.5 Neural Beam Search

**What it does**: Train a policy network to score partial programs. At each step, maintain a beam of top-k partial ASTs, expand each with all legal productions, score with the network, keep top-k. Return the best complete program.

**Design**:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Beam width | 50, 200, 1000 | Trade compute for coverage |
| Scoring | Policy network (log-probability of derivation path) | Likelihood-based ranking |
| Training | Train policy on best programs found (supervised on successful derivation sequences) | Bootstrap from random search |
| Retraining | Every N programs, retrain on top-k programs found so far | Iterative improvement |
| Temperature | 0.5-1.0 | Diversity within beam |
| Total iterations | 50-100 | Training cycles |

**How training works**:
1. Phase 1: Random search to find initial good programs
2. Rank programs by reward. Take top-100.
3. Extract their derivation sequences (action sequences that produced them)
4. Train policy network via supervised learning (cross-entropy on derivation actions)
5. Use trained network for beam search -> find new programs
6. Add best new programs to training set. Repeat.

**Strengths**: Deterministic search (no stochasticity at eval time). Network directly learns "what does a good derivation look like?" from examples. Beam width provides controlled coverage. Much cheaper than MCTS per program (1 forward pass per step per beam element, no rollouts).

**Weaknesses**: Beam search can miss globally good programs if they look bad at intermediate steps (the "narrow beam" problem). Training is supervised -- requires initial good programs from another method. Risk of mode collapse (always producing similar programs).

**Expected**: Strong if initial programs are diverse enough. May outperform AlphaZero by avoiding the value head problem entirely (scores derivation likelihood, not state value). The beam provides a form of search that's cheaper than MCTS.

---

### 4.6 Evolutionary / Genetic Programming

**What it does**: Maintain a population of complete programs. Each generation: evaluate fitness, select parents, create offspring via mutation/crossover, replace worst individuals.

**Design**:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Population size | 200-1000 | Diversity vs compute |
| Generations | 500-5000 | Enough for convergence |
| Selection | Tournament (k=5) | Standard GP |
| Mutation rate | 0.3-0.5 per individual | High: exploration is critical |
| Crossover rate | 0.5-0.7 | Exchange subtrees |
| Elitism | Top 5-10% preserved | Prevent regression |
| Initial population | Random complete derivations | Cold start |
| Fitness | LeafEvaluator score | Same metric as other algorithms |

**Mutation operators** (within grammar constraints):
- **Subtree mutation**: Replace a random subtree with a fresh random derivation of the same type and budget
- **Point mutation**: Change a terminal's parameter (e.g., Flip(3) -> Flip(5), IsZero(2) -> IsZero(7))
- **Grow/shrink**: Replace `Default(Flip(j))` with `Ite(cond, Flip(j'), ProgramHole)` or vice versa
- **Budget-aware**: All mutations must respect budget constraints

**Crossover operators**:
- **Subtree crossover**: Swap compatible subtrees (same type, compatible budget) between two programs
- **Derivation crossover**: Cross derivation action sequences at aligned points

**Strengths**: Population maintains diversity (avoids mode collapse). No gradient computation. Naturally handles sparse reward (selection pressure is all you need). No credit assignment problem (fitness is per-program, not per-step). Proven effective for program synthesis (GP is the classic approach).

**Weaknesses**: No gradient = slow convergence per evaluation. Mutation operators must respect grammar constraints (extra implementation complexity). Crossover may disrupt functional programs (the "competing conventions" problem). Large populations needed for high-dimensional search spaces.

**Expected**: Likely the most competitive alternative to AlphaZero for this problem. GP avoids all five diagnosed bottlenecks: no value function, no per-step credit assignment needed, population diversity prevents plateaus, no backup rule issues, and fitness-proportional selection provides smooth gradient across the reward landscape. The question is whether GP's per-program evaluation overhead (cheap) times the number of evaluations needed exceeds AlphaZero's per-program overhead (expensive) times its smaller count.

---

### 4.7 Simulated Annealing / MCMC

**What it does**: Start with a random complete program. Propose modifications (mutations). Accept improvements always; accept worse programs with probability `exp(-delta/T)` where T decreases over time.

**Design**:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Initial temperature | 1.0 | Accept most proposals initially |
| Cooling schedule | Geometric: T *= 0.999 per step | Standard SA |
| Final temperature | 0.01 | Nearly greedy at end |
| Steps | 100K - 1M | Many proposals needed |
| Proposal distribution | Subtree re-derivation | Grammar-respecting mutations |
| Restarts | 10-50 | Escape local optima |

**Proposal operators** (same as evolutionary mutations):
- Subtree re-derivation: pick random subtree, delete it, re-derive randomly within its budget
- Point mutation: change terminal parameters
- Grow/shrink: adjust program depth

**Strengths**: Very simple to implement. Naturally handles grammar constraints via proposal operators. Good at local optimization around promising programs. Low memory (single program state).

**Weaknesses**: Single-chain = no diversity. Gets stuck in local optima without restarts. Proposal distribution matters enormously -- bad proposals waste compute. No population-level information sharing.

**Expected**: Good for local refinement (finding the best program "near" a known good one). Unlikely to discover the globally optimal program from scratch in large search spaces. Best used as a post-processing step on top of programs found by other methods.

**Variant -- Parallel tempering**: Run multiple chains at different temperatures, periodically swap states. Combines exploration (high T) with exploitation (low T). More robust than single-chain SA.

---

### 4.8 GFlowNet

**What it does**: Learn a policy that samples complete programs with probability proportional to their reward. Unlike REINFORCE (which collapses to the mode), GFlowNet maintains diversity by targeting the full reward distribution.

**Design**:

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Network | Transformer (match AlphaZero) | Fair comparison |
| Training objective | Trajectory balance (TB) | Most stable GFlowNet objective |
| Learning rate | 1e-4 | Sensitive to LR |
| Replay buffer | 50K derivation trajectories | Off-policy training |
| Exploration | Epsilon=0.1 mixed with policy | Maintain coverage |
| Batch size | 32-64 | Per-update |
| Total trajectories | 100K-500K | Needs many samples |
| Reward transform | R^beta, beta=2-4 | Sharpen reward landscape |

**How it works**:
- The policy `pi(production | partial_AST)` defines a flow over derivation paths
- Training objective (trajectory balance): `log Z + sum(log pi_forward) = log R + sum(log pi_backward)` where Z is the partition function (learnable), R is the program's reward, and pi_backward is a learned backward policy
- At convergence: sampling probability proportional to reward

**Strengths**: Learns to sample diverse high-reward programs (not just the single best). Naturally handles multi-modal reward landscapes. Off-policy training via replay buffer. Provides a distribution over good programs, not a single answer.

**Weaknesses**: GFlowNet training can be unstable, especially with sparse rewards. Reward transform (R^beta) is critical -- too flat and it doesn't focus, too sharp and it collapses. Relatively new method, less understood than GP or MCTS. Requires reward > 0 for all programs (may need reward transformation).

**Expected**: Promising for this problem because:
1. Diversity = explores many distinct program structures (avoids AlphaZero's mode collapse)
2. No value function needed (directly optimizes sampling probability)
3. Intermediate states get trained via trajectory balance (better than terminal-only REINFORCE)

The key question: does GFlowNet's trajectory balance objective provide better per-step credit assignment than AlphaZero's value head?

---

### 4.9 Enumerative Search (Bottom-Up)

**What it does**: Systematically enumerate all syntactically valid programs up to a given budget, evaluate each, return the best.

**Design**:
- Generate all derivation sequences via depth-first or breadth-first enumeration
- Prune branches where budget is exhausted
- Evaluate each complete program
- Guaranteed to find the optimum within budget

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Budget | Start at 4, increment by 2 | Iterative deepening |
| Pruning | Dead-end detection, one-hot contradiction | Reduce enumeration |
| Parallelism | Embarrassingly parallel | Split at first production choice |

**Strengths**: Guaranteed optimal within budget. No hyperparameters. No learning. Complete coverage.

**Weaknesses**: Exponential in budget. Feasible only for small budgets.

| Budget | Programs | Enumeration time (est.) |
|--------|----------|------------------------|
| 10 | ~6.7M | Minutes |
| 14 | ~16B | Hours-days |
| 18 | ~10^12 | Infeasible |
| 22+ | ~10^16+ | Impossible |

**Expected**: Solves D=2 trivially (budget=12-14 is enumerable). Finds D=3 optimum only if budget=22 is reached, which is infeasible for full enumeration. Useful as a gold standard for small budgets and as an initialization method (enumerate budget<=14 to seed other algorithms).

**Variant -- Iterative deepening with pruning**: Enumerate budget=4, 6, 8, ... stopping when a "good enough" program is found. Combined with semantic pruning (skip programs provably equivalent to already-evaluated ones), can push the feasibility boundary.

---

## 5. Systematic Comparison

### 5.1 Properties Table

| Property | Random | Pure MCTS | AlphaZero | REINFORCE | Beam Search | Evolutionary | SA/MCMC | GFlowNet | Enumerative |
|----------|--------|-----------|-----------|-----------|-------------|-------------|---------|----------|-------------|
| **Category** | Baseline | Tree search | Search+Learning | Policy gradient | Guided enum | Population | Local search | Flow-based | Exhaustive |
| **Learns across episodes** | No | No | Yes | Yes | Yes | Implicitly | No | Yes | No |
| **Search at generation time** | None | MCTS tree | MCTS tree | None | Beam | None | Local walk | None | Full enum |
| **Per-program cost** | Very low | High (MCTS) | Very high (MCTS+NN) | Low (1 fwd/step) | Low (1 fwd/step) | Very low | Very low | Low (1 fwd/step) | Very low |
| **Programs/hour (est.)** | 100K+ | 500-2K | 100-500 | 50K+ | 10K-50K | 50K+ | 100K+ | 20K-50K | Depends on budget |
| **Handles sparse reward** | N/A | Via rollouts | Poorly (diagnosed) | Poorly | Via initial seed | Well (selection) | Via acceptance | Moderately | N/A (exact) |
| **Credit assignment** | None | Rollout-based | Value head (broken) | Return-weighted | Supervised | Per-program fitness | Per-program | Trajectory balance | None needed |
| **Diversity** | High (random) | Low (single tree) | Low (mode collapse) | Low (policy collapse) | Medium (beam) | High (population) | Low (single chain) | High (by design) | Complete |
| **Exploits good solutions** | No | Within episode | Across iterations | Gradient | Supervised retraining | Crossover/mutation | Local refinement | Proportional sampling | N/A |
| **Implementation effort** | Trivial | Low (exists) | Exists | Low (~100 lines) | Medium (~200 lines) | Medium (~300 lines) | Low (~100 lines) | Medium-High (~400 lines) | Medium (~200 lines) |

### 5.2 How Each Algorithm Addresses the Five Bottlenecks

| Bottleneck | Random | Pure MCTS | AlphaZero | REINFORCE | Beam | Evolutionary | SA | GFlowNet | Enum |
|------------|--------|-----------|-----------|-----------|------|-------------|----|---------|----|
| **B1: Undiff. value targets** | N/A | N/A (no learning) | FAILS | Same problem | Avoids (supervised) | Avoids (no value fn) | N/A | Trajectory balance (partial fix) | N/A |
| **B2: Sparse reward in vast space** | Brute force | MCTS focuses | MCTS focuses (weakly) | Volume (many samples) | Beam focuses | Population covers | Local coverage | Learned sampling | Exact coverage |
| **B3: Flat reward landscape** | Can't distinguish | Rollouts average out | Same problem | Same problem | Same problem | Selection pressure helps | Same problem | Reward transform helps | Full evaluation |
| **B4: Max backup noise** | N/A | Can use mean | Can fix (use mean) | N/A | N/A | N/A | N/A | N/A | N/A |
| **B5: Budget bloat** | Directly affected | Directly affected | Directly affected | Directly affected | Directly affected | Directly affected | Directly affected | Directly affected | Directly affected |

**Key insight**: Budget reduction (B5) helps ALL algorithms equally. It should be applied universally. The algorithms differ primarily in how they handle B1 (credit assignment) and B2 (coverage of large space).

### 5.3 Which Algorithms to Try First (Priority Order)

| Priority | Algorithm | Why |
|----------|-----------|-----|
| 1 | **AlphaZero-combined** (budget=24 + mean backup + keys_progress + dead-end penalty) | Cheapest: config changes only. Tests whether fixing diagnosed issues is sufficient. |
| 2 | **Random Search** (budget=24, 1M programs) | Establishes baseline. If random search at budget=24 beats AlphaZero at budget=34, the budget is the primary problem. |
| 3 | **Evolutionary GP** (budget=24, population=500) | Avoids all learning-related bottlenecks. Population diversity is exactly what's needed. Proven for program synthesis. |
| 4 | **Pure MCTS** (budget=24, 5000 sims, no network) | Tests whether the neural network is actually helping or hurting. |
| 5 | **REINFORCE** (budget=24, 100K programs) | Tests whether volume (cheap programs) beats quality (expensive MCTS programs). |
| 6 | **Neural Beam Search** (budget=24, beam=200) | Tests supervised alternative to RL-style learning. |
| 7 | **GFlowNet** (budget=24) | Most novel. Tests diversity-focused generation. |
| 8 | **Enumerative** (budget=14, then 16, 18...) | Ground truth for small budgets. Seeds other algorithms. |

---

## 6. Scaling Analysis

### 6.1 Budget Scaling (Fixed D=3, n_sites=11)

| Budget | Search space | Random (1M samples) | Pure MCTS (5K sims) | AlphaZero (30 iter) | GP (500 pop, 1K gen) | Enum |
|--------|-------------|---------------------|---------------------|--------------------|--------------------|------|
| 14 | ~10^10 | Unlikely | May find decent | Should learn | Likely finds good | Feasible (hours) |
| 22 | ~10^17 | No chance | Unlikely optimal | Unknown | Possible with luck | Infeasible |
| 24 | ~10^18 | No chance | Very unlikely | Better chance (B5 fix) | Possible | Infeasible |
| 34 | ~10^30 | No chance | No chance | Failed (empirically) | Very unlikely | Infeasible |

### 6.2 D Scaling (Budget = optimal + 2)

| D | Budget | Search space | Derivation steps | Branching factor | Difficulty |
|---|--------|-------------|-----------------|-----------------|------------|
| 2 | 14 | ~10^10 | ~6-8 | ~20-40 | Low |
| 3 | 24 | ~10^18 | ~12-16 | ~30-80 | Medium |
| 5 | 44 | ~10^34 | ~22-28 | ~50-120 | High |
| 10 | 94 | ~10^73 | ~47-58 | ~80-200 | Very High |
| 20 | 194 | ~10^150 | ~97-120 | ~100-300 | Extreme |

**Observation**: Even at recommended budget (optimal+2), the search space grows super-exponentially with D. No algorithm can exhaustively search beyond D=2. Success at D>=3 requires either (a) very strong search heuristics or (b) compositional/modular structure exploitation.

### 6.3 Expected Algorithm Viability Per Scale

| Algorithm | D=2 (budget=14) | D=3 (budget=24) | D=5 (budget=44) | D=10 (budget=94) | D=20 (budget=194) |
|-----------|-----------------|-----------------|-----------------|------------------|-------------------|
| Random Search | Good programs | Mediocre | Poor | Useless | Useless |
| Pure MCTS | Optimal | Good | Mediocre | Poor | Useless |
| AlphaZero (fixed) | Optimal | Unknown (test!) | Unlikely | Very unlikely | No |
| REINFORCE | Good | Mediocre-Good | Unknown | Poor | Useless |
| Beam Search | Optimal (if seeded) | Good (if seeded) | Unknown | Unknown | Unknown |
| Evolutionary GP | Optimal | Good-Optimal | Possible | Unknown | Needs macro/modular |
| SA/MCMC | Good (local opt) | Good (local opt) | Mediocre | Poor | Useless |
| GFlowNet | Good | Unknown (test!) | Unknown | Unknown | Unknown |
| Enumerative | Optimal (exact) | Infeasible | Infeasible | Infeasible | Infeasible |

---

## 7. Experiment Design

### 7.1 Independent Variables

- **Algorithm**: All 9 listed above (prioritized order from Section 5.3)
- **Budget**: 14, 22, 24, 34 (for D=3); optimal+2 (for other D)
- **D**: 2, 3, 5, 10, 20
- **Metric**: keys_progress (recommended), weighted, avg_reward
- **Compute budget**: Normalize by total programs evaluated (not wall-clock, since per-program cost varies wildly)

### 7.2 Dependent Variables (Metrics)

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Best reward** | Highest LeafEvaluator score found | Quality ceiling |
| **Best solve rate** | Highest solve_rate of any program found | Did it solve the task? |
| **Programs to best** | Number of programs evaluated before finding the best | Sample efficiency |
| **Unique programs** | Number of distinct programs evaluated | Coverage |
| **Reward distribution** | Histogram of program rewards | Landscape exploration |
| **Best program** | Pretty-print of highest-scoring program | Interpretability check |
| **Wall-clock time** | Total time to best result | Practical efficiency |

### 7.3 Experiment Phases

#### Phase 0: Ground Truth (1 day)
- Enumerative search at budget=10, 12, 14 for D=2 and D=3
- Establishes optimal programs at small budgets
- Provides seed programs for beam search and evolutionary approaches

#### Phase 1: Budget Matters (1-2 days)
- Random search at budget=14, 22, 24, 34 for D=3, 1M programs each
- AlphaZero-combined (all fixes) at budget=24 vs budget=34 for D=3
- **Key question**: How much does budget reduction help? Is budget=24 sufficient?

#### Phase 2: Algorithm Comparison at D=3, Budget=24 (3-5 days)
- All algorithms on D=3 with budget=24
- Normalize comparison by total programs evaluated
- Plot best_reward vs programs_evaluated for each algorithm
- **Key question**: Which search strategy is best at fixed budget?

#### Phase 3: D Scaling (1 week)
- Top 3 algorithms from Phase 2, run at D=2, 3, 5, 10
- Budget = optimal+2 for each D
- **Key question**: Which algorithm degrades most gracefully?

#### Phase 4: Large D (optional, 1 week)
- Top 1-2 algorithms at D=20 with budget=optimal+2
- Add macro productions if not already used
- Test curriculum: pre-train on D=5, transfer to D=10, transfer to D=20
- **Key question**: Is any algorithm viable at D=20?

### 7.4 Implementation Notes

**All algorithms share**:
- Same `DerivationGame` (or `FactoredDerivationGame`) interface
- Same `LeafEvaluator` for program scoring
- Same grammar, budget, metric settings
- Same `DoorsPDDLLiteEnv` for program execution

**What needs building per algorithm**:

| Algorithm | New code | Uses existing |
|-----------|----------|--------------|
| Random Search | ~30 lines (loop + random derivation) | DerivationGame, LeafEvaluator |
| Pure MCTS | ~10 lines (disable network in existing MCTS) | MCTS, DerivationGame |
| AlphaZero variants | Config changes only | Everything |
| REINFORCE | ~150 lines (PG training loop) | Network architecture, DerivationGame |
| Beam Search | ~200 lines (beam loop + supervised training) | Network architecture, DerivationGame |
| Evolutionary | ~300 lines (population, mutation, crossover) | DerivationGame, LeafEvaluator |
| SA/MCMC | ~100 lines (proposal + acceptance) | DerivationGame, LeafEvaluator |
| GFlowNet | ~400 lines (TB loss, backward policy) | Network architecture, DerivationGame |
| Enumerative | ~200 lines (systematic enumeration) | DerivationGame, LeafEvaluator |

### 7.5 Fair Comparison Framework

To compare algorithms fairly, normalize on **total programs evaluated** (not wall-clock or iterations):

```
Efficiency = best_reward / programs_evaluated
```

Plot learning curves as: `best_reward_so_far vs cumulative_programs_evaluated`

This removes the confound of per-program cost (AlphaZero evaluates fewer but with expensive MCTS; random search evaluates many cheaply). Both axes are meaningful: x = compute spent, y = quality achieved.

Additionally, report wall-clock time for practical relevance, and note that some algorithms (random, GP, SA) are embarrassingly parallel while others (AlphaZero, beam search) have sequential dependencies.

---

## 8. Key Questions This Comparison Answers

1. **Is the budget the primary bottleneck?** If random search at budget=24 matches AlphaZero at budget=34, then search algorithm barely matters -- budget is everything.

2. **Does learning help at all?** Compare learning algorithms (AlphaZero, REINFORCE, GFlowNet) against non-learning ones (random, pure MCTS, GP, SA) at fixed budget. If non-learning algorithms win, the learning framework is actively hurting.

3. **Is the value function the problem, or is it deeper?** Compare AlphaZero (value + policy) vs REINFORCE (policy only) vs GP (no neural network). If GP wins, the neural approach is wrong. If REINFORCE matches AlphaZero, the MCTS overhead is wasted. If AlphaZero-fixed wins, the diagnosed bottlenecks were the real issue.

4. **What is the compute-optimal algorithm?** At a fixed compute budget (e.g., "evaluate 100K programs"), which algorithm finds the best program? This determines what you should actually use.

5. **Can any algorithm scale to D=10+?** The search space at D=10 is ~10^73. No brute-force approach works. Only algorithms that exploit problem structure (compositionality, modularity, curriculum) have a chance. This reveals whether the derivation game formulation itself is viable at scale.

6. **Is diversity or focus more valuable?** GP and GFlowNet maintain diversity. MCTS and beam search focus. Which strategy wins in this reward landscape? If diversity wins, the flat reward landscape (B3) is the binding constraint. If focus wins, the search space size (B2) is.
