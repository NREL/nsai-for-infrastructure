# Algorithm Audit: AlphaZero for Grammar-Guided Program Synthesis

_This document is a systematic audit of a research codebase that applies AlphaZero to program synthesis via grammar derivation. It catalogues the design choices, highlights areas of concern, and poses open questions for deeper analysis. It is intended to be consumed by an LLM or researcher for further investigation._

---

## Table of Contents

1. [Problem Statement & Context](#1-problem-statement--context)
2. [System Architecture & File Map](#2-system-architecture--file-map)
3. [Algorithm Flow Diagram](#3-algorithm-flow-diagram)
4. [Audit Scope Table](#4-audit-scope-table)
5. [A: Grammar & DSL Design](#a-grammar--dsl-design)
6. [B: MCTS Search](#b-mcts-search)
7. [C: State Representation & Observation Encoding](#c-state-representation--observation-encoding)
8. [D: Neural Network Architecture](#d-neural-network-architecture)
9. [E: Reward Signal & Value Targets](#e-reward-signal--value-targets)
10. [F: Training Loop & Optimization](#f-training-loop--optimization)
11. [G: Evaluation & Gating Mechanism](#g-evaluation--gating-mechanism)
12. [H: Scalability Questions](#h-scalability-questions)
13. [I: Correctness & Edge Cases](#i-correctness--edge-cases)

---

## 1. Problem Statement & Context

### What is the BitString problem?

The environment is a **BitString game**: an agent operates on an N-bit binary string (initially with `n_ones` bits set to 1, the rest 0). At each step, the agent selects one bit to flip. The goal is to reach the all-ones state. A **potential function** (onemax, leading_ones, or binval) provides shaped reward: `r_t = (phi(s_{t+1}) - phi(s_t)) / N`.

### What is the program synthesis task?

Instead of learning a neural-network policy directly, the system tries to **synthesize an explicit decision-list program** in a small DSL. A decision-list program has the form:

```
if IsZero(0): Flip(0)
elif IsZero(1): Flip(1)
elif And(IsZero(2), IsZero(3)): Flip(2)
else: Flip(5)
```

Each program is an AST with a fixed node count ("budget"). The system must find the program that achieves the highest average reward across all frozen initial states.

### What strategy was chosen?

The approach casts program synthesis as a **single-player game** and applies **AlphaZero** (MCTS + neural network) to search the space of grammar derivations:

- **State** = partial AST with "holes" (unfilled subtrees)
- **Action** = a grammar production that fills the leftmost hole
- **Terminal reward** = quality of the completed program (measured by running it on frozen bitstring states)
- **Neural network** = Transformer encoder that reads the partial AST and outputs (policy over productions, value estimate)

The system was previously built in stages: first exhaustive enumeration (for small budgets), then pure MCTS with uniform policy (no learning), and now the full AlphaZero loop with a learned Transformer.

### What is the current status?

The system is functional and has been validated with experiment data. A training run with tuned hyperparameters (budget=14, N=6, 200 MCTS simulations, 40 games/iteration) demonstrates that the algorithm **learns effectively**:

**Experiment: `20260301_213808_N6_L14_avg_reward_mcts200_games40_iter10`**

| Metric | Start (iter 1) | End (iter 9) |
|--------|---------------|--------------|
| Eval reward (new net) | 0.067 | 0.333 (optimal for avg_reward) |
| Training loss | 5.32 | 3.12 |
| Policy loss | 2.63 | 1.55 |
| Value loss | 0.053 | 0.026 |
| Unique programs explored | 3,438 | 54,913 (cumulative) |

- Eval reward converged to the optimal 1/3 by iteration 7 and remained stable.
- Gate accepted 8 of 9 iterations (rejected once at iteration 5 with score 0.50 < threshold 0.55).
- At iteration 9, the best program achieved **100% solve rate** and **0.667 avg_reward**:

```
if And(IsZero(3), IsZero(2)):
  Flip(3)
elif IsZero(2):
  Flip(2)
elif Not(IsZero(4)):
  Flip(5)
else:
  Flip(4)
```

Key improvements since the initial implementation: dead-end grammar productions eliminated (action space 60→48), canonical grammar pruning (double-negation ban + And commutativity), and hyperparameter tuning (simulations 50→200, games 20→40, eval games 2→20).

---

## 2. System Architecture & File Map

### Core AlphaZero Framework (domain-agnostic)

| File | Role | Key interfaces |
|------|------|---------------|
| `src/alphazeropp/core/game.py` | Abstract Game base class | `reset()`, `step()`, `get_action_mask()`, `hashable_obs`, `stash_state()`, `clone()` |
| `src/alphazeropp/core/mcts.py` | MCTS search engine | `perform_simulations()`, `search()` (recursive), `calc_masked_ucbs()`, `update_edge()` |
| `src/alphazeropp/core/agent.py` | Agent: wraps MCTS + network | `policy()` → move probabilities; `play_one_round()` → training examples |
| `src/alphazeropp/core/policy_value_net.py` | Abstract PolicyValueNet | `predict(state) → (policy, value)`; `train(examples)` |

### BitString DSL & Grammar

| File | Role | Key contents |
|------|------|-------------|
| `src/.../dsl/ast_nodes.py` | AST node definitions | `Flip`, `IsZero`, `Not`, `And`, `Ite`, `Default` — all frozen dataclasses |
| `src/.../dsl/budget_grammar.py` | Budget-constrained CFG | `ProgramHole`, `ConditionHole`, `enumerate_programs()`, `count_programs()` |
| `src/.../dsl/derivation.py` | Derivation engine | `DerivationState`, `Production`, `_program_productions()`, `_condition_productions()` |
| `src/.../dsl/derivation_game.py` | Game interface for MCTS | `DerivationGame(Game)`, observation encoding, action masking |
| `src/.../dsl/derivation_network.py` | Transformer policy-value net | `DerivationTransformerModel`, `DerivationPolicyValueNet` |
| `src/.../dsl/leaf_evaluator.py` | Terminal program evaluator | Runs completed programs on frozen states, returns scalar metric |
| `src/.../dsl/interpreter.py` | Program interpreter | `eval_program()`, `interp_ops()`, `run_policy_episode()` |
| `src/.../dsl/derivation_config.py` | Configuration & wiring | `DerivationConfig` — builds all components with default hyperparameters |
| `src/.../dsl/game_config.py` | Env config for leaf eval | `GameConfig`, `all_initial_states()` |

### Training Infrastructure

| File | Role |
|------|------|
| `src/.../training/trainer.py` | Self-play data collection + network training |
| `src/.../training/evaluator.py` | Pitting new vs old network |
| `src/.../training/gated_trainer.py` | Accept/reject gating around trainer + evaluator |

### Environment & Reward Shaping

| File | Role |
|------|------|
| `src/.../bitstring/game.py` | `BitStringGym` — the base Gymnasium environment |
| `src/.../bitstring/shaped_env.py` | `ShapedBitStringGym` — potential-based reward shaping wrapper |
| `src/.../bitstring/potentials.py` | `onemax`, `leading_ones`, `binval` potential functions |

### Entry Point

| File | Role |
|------|------|
| `scripts/run_derivation.py` | Main training script — interactive config, training loop, plotting |

---

## 3. Algorithm Flow Diagram

```
OUTER LOOP: 30 training iterations
│
├── STEP 1: SELF-PLAY (40 games)
│     │
│     │  Each game starts from ProgramHole(14)
│     │  and proceeds for ~9 derivation steps:
│     │
│     │  ┌──────────────────────────────────────────────┐
│     │  │  For each derivation step:                    │
│     │  │                                               │
│     │  │   partial AST ──► Transformer(d=64, L=2)      │
│     │  │                   ├── policy: softmax logits   │
│     │  │                   └── value: scalar estimate   │
│     │  │                                               │
│     │  │   policy + value ──► MCTS(200 simulations)    │
│     │  │                      UCB = Q_norm + c·P·√N    │
│     │  │                                /(1+n)         │
│     │  │                                               │
│     │  │   MCTS visit counts ──► pi_MCTS (softmax)     │
│     │  │   Sample action ~ pi_MCTS                     │
│     │  │   Apply grammar production to leftmost hole   │
│     │  │                                               │
│     │  │   SAVE: (observation, pi_MCTS, value_target)  │
│     │  └──────────────────────────────────────────────┘
│     │
│     │  At terminal step: run completed program on
│     │  frozen bitstring states via LeafEvaluator (1 by default)
│     │  → scalar reward (avg shaped reward)
│     │
│     │  Value targets for ALL steps = terminal reward
│     │  (discount=1.0, intermediate rewards=0)
│     │
├── STEP 2: TRAIN TRANSFORMER
│     │  Replay buffer: last 20 iterations (~7200 examples)
│     │  Loss = MSE(v_predicted, v_target) + 2.0 * CE(pi_predicted, pi_MCTS)
│     │  Adam optimizer, lr=3e-4, 5 epochs, batch_size=32
│     │
├── STEP 3: GATE (accept/reject)
│     │  Deep-copy old agent
│     │  Pit new net vs old net: 20 evaluation games
│     │  Accept if win_rate >= 55%, else restore old weights
│     │
└── Repeat
```

### The Two-Level MDP Structure

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

### Grammar Productions (Budget-Constrained CFG)

```
Program productions:
  P(2) → Default(Flip(j))                      j ∈ [0, N)
  P(k) → Ite(C(i), Flip(j), P(k-2-i))         k ≥ 5,  i ∈ [1, k-4],  j ∈ [0, N)

  P(3) → ∅     (no valid production — budget gap)
  P(4) → ∅     (no valid production — budget gap)

Condition productions:
  C(1) → IsZero(j)                             j ∈ [0, N)
  C(k) → Not(C(k-1))                           k ≥ 2
  C(k) → And(C(i), C(k-1-i))                   k ≥ 3,  i ∈ [1, k-2]

Budget accounting:
  Ite:     1 (Ite) + |cond| + 1 (Flip) + |else_prog| = k
  Default: 1 (Default) + 1 (Flip) = 2
  Not:     1 (Not) + |child| = k
  And:     1 (And) + |left| + |right| = k
  IsZero:  1
  Flip:    1
```

### Configuration & Wiring (from `derivation_config.py`)

The `DerivationConfig` class sets all hyperparameters and `build()` wires every component together:

```python
# derivation_config.py — DerivationConfig (lines 29-167)
class DerivationConfig(MetaConfig):
    def __init__(self):
        super().__init__()
        self.game = CoreGameConfig(game_cls=DerivationGame, kwargs={
            "budget": 14, "n_sites": 6, "n_ones": 2,
            "n_frozen_states": 1,
            "bit_flip": True, "sparse_reward": False,
            "potential_name": "onemax", "metric": "avg_reward",
            "penalty_lambda": 0.1, "blend_alpha": 0.5,
        })
        self.net = NetConfig(net_cls=DerivationPolicyValueNet, kwargs={
            "budget": 14, "n_sites": 6,
            "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1,
            "training_params": {
                "epochs": 5, "batch_size": 32, "learning_rate": 3e-4,
                "weight_decay": 1e-4, "policy_weight": 2.0,
            },
        })
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 200, "temperature": 1.0,
                "c_exploration": 1.5, "dirichlet_alpha": 0.25, "dirichlet_epsilon": 0.40,
            },
            reward_discount=1.0,
            random_seeds={"mcts": 43, "train": 47, "eval": 23, "external_policy": 68},
        )
        self.trainer = TrainerConfig(
            n_games_per_train=40, n_past_iterations_to_train=20, n_procs=-1,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(n_games=20, n_procs=-1)
        self.run = RunConfig(
            n_iterations=30, accept_threshold=0.55, plot_every=5,
        )

    def build(self):
        """Build all components. Dependency chain:
        DSLGameConfig → frozen_states → LeafEvaluator → DerivationGame → action_size → Network → Agent → Trainer → Evaluator
        """
        gk = dict(self.game.kwargs)
        budget, n_sites, n_ones = gk["budget"], gk["n_sites"], gk["n_ones"]

        # 1. LeafEvaluator (needs frozen states + DSL game config)
        dsl_cfg = DSLGameConfig(bit_flip=gk["bit_flip"], sparse_reward=gk["sparse_reward"],
                                n_ones=n_ones, potential_name=gk["potential_name"])
        frozen_states = all_initial_states(n_sites, n_ones)    # C(6,2) = 15 total
        n_frozen = gk.get("n_frozen_states", len(frozen_states))
        frozen_states = frozen_states[:n_frozen]               # default: 1 state
        leaf_eval = LeafEvaluator(n_sites, frozen_states, dsl_cfg, metric=gk["metric"])

        # 2. DerivationGame
        game = DerivationGame(budget, n_sites, leaf_eval)

        # 3. Network (action_size derived from game's max productions)
        action_size = game.action_space.n                       # 48
        net = DerivationPolicyValueNet(budget=budget, n_sites=n_sites, action_size=action_size, ...)

        # 4. Agent, Trainer, Evaluator
        agent = Agent(game=game, net=net, mcts_params=self.agent.mcts_params,
                      reward_discount=self.agent.reward_discount, ...)
        trainer = Trainer(agent=agent, net=net, game=game,
                         n_games_per_train=self.trainer.n_games_per_train, ...)
        evaluator = Evaluator(n_games=self.evaluator.n_games, ...)
        return game, net, agent, trainer, evaluator
```

### Default Hyperparameters (from `derivation_config.py`)

| Parameter | Value | Component |
|-----------|-------|-----------|
| budget | 14 | Grammar |
| n_sites | 6 | Problem |
| n_ones | 2 | Problem |
| potential | onemax | Reward shaping |
| metric | avg_reward | Leaf evaluation |
| d_model | 64 | Network |
| n_heads | 4 | Network |
| n_layers | 2 | Network |
| learning_rate | 3e-4 | Training |
| batch_size | 32 | Training |
| policy_weight | 2.0 | Loss function |
| epochs | 5 | Training |
| n_simulations | 200 | MCTS |
| temperature | 1.0 | MCTS |
| c_exploration | 1.5 | MCTS |
| dirichlet_alpha | 0.25 | MCTS noise |
| dirichlet_epsilon | 0.40 | MCTS noise |
| reward_discount | 1.0 | Agent |
| n_games_per_train | 40 | Trainer |
| n_past_iterations | 20 | Replay buffer |
| n_frozen_states | 1 | Leaf evaluation |
| eval n_games | 20 | Evaluator |
| accept_threshold | 0.55 | Gating |
| n_iterations | 30 | Run |

### Key Quantities for Default Config

```
Total programs at budget=14, N=6:    151,173,432
Action space (max productions):      48
Derivation depth (steps to terminal): ~9
Frozen evaluation states:            1 (of C(6,2) = 15 possible)
Programs evaluated per iteration:    ~360 (40 games × ~9 steps, some cached)
Coverage per iteration:              ~0.0002% of program space
Replay buffer size:                  ~7200 (20 iterations × ~360 examples)
```

---

## 4. Audit Scope Table

| # | Dimension | What to examine | Key questions |
|---|-----------|----------------|---------------|
| **A** | Grammar & DSL | Production rules, expressiveness, redundancy, action space | Is this grammar nearly by-construction inefficient? Does it enumerate far more programs than necessary? |
| **B** | MCTS Search | UCB formula, Q-normalization, tree lifecycle, exploration | Is the search algorithm sound? Is computation being wasted? |
| **C** | State Representation | Observation encoding, information content | Does the Transformer see enough to make good decisions? |
| **D** | Neural Architecture | Transformer design, policy/value heads | Is a flat Transformer the right choice for tree-structured data? |
| **E** | Reward Signal & Value Targets | Reward sparsity, value target construction | Can the value head learn anything useful from the training signal it receives? |
| **F** | Training Loop | Loss function, replay buffer, optimization | Are there training stability or signal quality issues? |
| **G** | Evaluation & Gating | Statistical power of network comparison | Does the gating mechanism actually discriminate? |
| **H** | Scalability | Growth rates, sample efficiency | Can this approach work at larger scales? |
| **I** | Correctness | Bugs, edge cases, numerical issues | Are there logical errors that silently degrade performance? |

---

## A. Grammar & DSL Design

_Key files: `budget_grammar.py`, `derivation.py`, `ast_nodes.py`_

### A1. [RESOLVED] The grammar had a structural dead zone at budgets 3 and 4

The smallest `Default` program uses 2 nodes. The smallest `Ite` program uses 5 nodes. No production exists for budgets 3 or 4:

```python
# budget_grammar.py lines 80-96
def count_programs(n_sites, budget):
    if budget < 2: return 0
    if budget == 2: total += n_sites              # Default(Flip(j))
    if budget >= 5:                                # Ite(C(i), Flip(j), P(k-2-i))
        for i in range(1, budget - 3):
            ...
```

**Resolution:** Dead-end productions are now filtered out in `_program_productions()` (`derivation.py` lines 173-174):

```python
# derivation.py lines 170-174
if budget >= 5:
    for i in range(1, budget - 3):
        else_budget = budget - 2 - i
        if count_programs(n_sites, else_budget) == 0:
            continue  # Skip dead-end budgets (e.g., 3 and 4)
```

This reduced the action space from 60 to 48 (for budget=14, N=6). No valid programs were removed — only unreachable dead-end paths. See `specs/dead_end_fix.md` for the full design document and test plan.

**Code reference — dead-end handling (still present as a safety net):**
```python
# derivation_game.py lines 154-168
is_dead_end = not is_complete and len(self._current_productions) == 0
truncated = is_dead_end
if is_dead_end:
    reward = 0.0
```

### A2. Semantic redundancy — how much of the search space is duplicated?

The grammar generates structurally distinct programs that are semantically equivalent. Five types of redundancy were identified:

1. **[RESOLVED] Double negation:** `Not(Not(c))` ≡ `c`, but costs 2 extra nodes. Now banned via the `parent_is_not` parameter on `ConditionHole`.

2. **[RESOLVED] And commutativity:** `And(c1, c2)` ≡ `And(c2, c1)`. Now enforced via canonical ordering `i ≤ k-1-i` in the And production loop.

3. **And idempotence:** `And(c, c)` ≡ `c`, but is still enumerated as a valid program.

4. **Contradictions:** `And(IsZero(j), Not(IsZero(j)))` is always false. Any `Ite` branch guarded by a contradiction is dead code.

5. **Unreachable branches:** In `Ite(c, Flip(j), Ite(c, Flip(k), rest))`, the inner Ite's condition `c` is re-evaluated on the same state — if `c` was false to reach the else branch, it will be false again (assuming the Flip in the first branch wasn't taken). So the second `c` is guaranteed false, making `Flip(k)` unreachable.

**Remaining question:** Items 3-5 are still open. For budget=14 and N=6, there are 151 million programs. With canonical pruning (items 1-2 resolved), the canonical count is computed by `count_canonical_programs()` in `budget_grammar.py`. How many of the remaining canonical programs are semantically unique when considering idempotence, contradictions, and unreachable branches?

**Current `_condition_productions()` — with canonical pruning:**

```python
# derivation.py lines 186-230
def _condition_productions(
    budget: int, n_sites: int, parent_is_not: bool = False,
) -> list[Production]:
    """Generate canonical productions for a ConditionHole with given budget.

    Canonicalization rules:
      1. **Double-negation ban**: When *parent_is_not* is True (this hole is
         the child of a ``Not``), suppress the ``Not(C(k-1))`` production.
      2. **And commutativity**: For ``And(C(i), C(j))``, restrict to
         ``i <= j`` (left budget <= right budget).
    """
    prods: list[Production] = []

    # C(1) → IsZero(j)
    if budget == 1:
        for j in range(n_sites):
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=IsZero(j),
                label=f"C({budget}) -> IsZero({j})",
            ))

    # C(k) → Not(C(k-1))  — only if parent is NOT a Not, and child
    # has canonical non-Not completions (prevents dead-end holes).
    if budget >= 2 and not parent_is_not and _ccnn(n_sites, budget - 1) > 0:
        child_budget = budget - 1
        result = Not(ConditionHole(child_budget, parent_is_not=True))
        prods.append(Production(
            hole_kind="C", hole_budget=budget,
            result=result,
            label=f"C({budget}) -> Not(C({child_budget}))",
        ))

    # C(k) → And(C(i), C(k-1-i))  — canonical: i <= k-1-i
    if budget >= 3:
        for i in range(1, (budget - 1) // 2 + 1):
            right_budget = budget - 1 - i
            result = And(ConditionHole(i), ConditionHole(right_budget))
            prods.append(Production(
                hole_kind="C", hole_budget=budget,
                result=result,
                label=f"C({budget}) -> And(C({i}), C({right_budget}))",
            ))

    return prods
```

**Implementation details:**
- `ConditionHole` now has a `parent_is_not: bool = False` field (`budget_grammar.py` line 50) to propagate context.
- `_ccnn(n_sites, budget)` counts canonical conditions that are **not** `Not`-prefixed (needed to prevent dead-end holes when the double-negation ban removes all completions).
- Supporting functions in `budget_grammar.py`: `count_canonical_conditions()`, `count_canonical_programs()`, `enumerate_canonical_conditions()`, `enumerate_canonical_programs()`, `format_canonical_reduction_report()`.

**Remaining interventions:**
- Add a semantic equivalence check to eliminate idempotent And, contradictions, and unreachable branches

### A3. Missing expressive primitives — is the grammar undersized?

The condition language provides `{IsZero, Not, And}`. Notable omissions:

| Pattern | Current encoding | Budget cost | With primitive | Budget cost |
|---------|-----------------|-------------|---------------|-------------|
| "bit j is 1" | `Not(IsZero(j))` | 2 nodes | `IsOne(j)` | 1 node |
| "a or b" | `Not(And(Not(a), Not(b)))` | 5 nodes | `Or(a, b)` | 3 nodes |

**Question:** For the OneMax problem (goal: all bits = 1), the optimal strategy is "find a zero bit and flip it." Expressing "is bit j zero?" is 1 node (`IsZero(j)`), but the symmetrically common pattern "is bit j already one?" costs 2 nodes. At budget=14, this overhead means one fewer `Ite` branch can be expressed. Does this bias the search toward programs that check for zeros but not ones? Would adding `IsOne` as a primitive meaningfully expand the reachable set of useful programs?

Similarly: `Or` must be encoded via De Morgan's law at 5 nodes. In a decision list, an `Or` condition is common ("flip bit 2 if either bit 0 or bit 1 is zero"). Is the grammar forcing the learner to spend budget on encoding patterns that should be cheap?

### A4. The action space mixes structural and parameter decisions

At each derivation step, the action is a single integer indexing into a list of productions. These productions mix two kinds of decisions:

1. **Structural:** Which template? (`Default` vs `Ite`; condition budget `i`)
2. **Parametric:** Which site index `j`? (for `Flip(j)` and `IsZero(j)`)

For example, at `ProgramHole(14)`, the productions include:
```
Action 0:  P(14) → Ite(C(1),  Flip(0), P(11))    ← structure: i=1, param: j=0
Action 1:  P(14) → Ite(C(1),  Flip(1), P(11))    ← structure: i=1, param: j=1
Action 2:  P(14) → Ite(C(1),  Flip(2), P(11))    ← structure: i=1, param: j=2
...
Action 6:  P(14) → Ite(C(2),  Flip(0), P(10))    ← structure: i=2, param: j=0
...
Action 47: (some other production)
```

**Question:** The network must learn a single 48-dimensional policy that simultaneously decides the structural template AND the parameter. Would a **hierarchical decomposition** (first decide the structure template, then decide parameters) be more sample-efficient? This would reduce the effective branching factor at each level.

**Code reference:**
```python
# derivation.py lines 155-183
def _program_productions(budget, n_sites):
    prods = []
    if budget == 2:
        for j in range(n_sites):                           # N productions
            prods.append(Production(..., Default(Flip(j))))
    if budget >= 5:
        for i in range(1, budget - 3):                     # (k-4) structural choices
            else_budget = budget - 2 - i
            if count_programs(n_sites, else_budget) == 0:
                continue                                    # Skip dead-end budgets
            for j in range(n_sites):                       # × N parametric choices
                prods.append(Production(..., Ite(ConditionHole(i), Flip(j), ProgramHole(else_budget))))
    return prods
```

### A5. Budget-exact constraint — are simple programs unreachable?

Every derivation must use **exactly** `budget` nodes. A program that solves the problem in 5 nodes cannot be discovered if the budget is 14. The grammar does not allow "padding" or early termination.

**Question:** Is this a fundamental limitation? For the BitString OneMax problem with N=6, the optimal program might be:
```
if IsZero(0): Flip(0)
elif IsZero(1): Flip(1)
...
elif IsZero(5): Flip(5)
else: Flip(0)
```
This uses `5 × 3 + 2 = 17` nodes (5 Ite branches with IsZero conditions + 1 Default). At budget=14, this program is **unreachable**. The system can only find the best 14-node program, which may be substantially worse.

What if the optimal program at budget=14 is actually a padded version of a simpler program, with wasted `Not(Not(...))` wrappings? The budget-exact constraint may force the learner to discover these wasteful patterns as a way to "spend" remaining budget.

---

## B. MCTS Search

_Key file: `mcts.py` (287 lines)_

### B1. Q-normalization via dynamic min-max — is this stable?

```python
# mcts.py lines 62-64 — reset per search
self.q_min = float('inf')
self.q_max = float('-inf')

# mcts.py lines 74-76 — reset AGAIN at start of perform_simulations
self.q_min = float('inf')
self.q_max = float('-inf')

# mcts.py lines 251-258 — applied during UCB calculation
if self.q_min == float('inf') or self.q_max == float('-inf'):
    q_normalized = 0.0
elif self.q_max > self.q_min:
    q_normalized = (q - self.q_min) / (self.q_max - self.q_min)
else:
    q_normalized = 0.5
```

The Q values are normalized to [0, 1] using the **global min and max Q values seen during the current search** (reset per `perform_simulations` call). The UCB formula is:

```
UCB(a) = Q_normalized(a) + c_exploration × P(a) × √(N_parent + eps) / (1 + N_a)
```

**Questions:**
- In the first few simulations, only 1-2 Q values exist. The min-max range is narrow, so small differences are amplified to span [0, 1]. Does this cause the algorithm to over-commit to the first few actions explored?
- As more simulations accumulate and the range widens, previously-decisive Q differences shrink in the normalized scale. Could this cause the algorithm to "forget" earlier preferences and re-explore?
- A single outlier leaf evaluation (an unusually good or bad program) could shift `q_max` or `q_min` dramatically. Does this destabilize the exploration-exploitation balance?
- MuZero uses a **running mean and std** for normalization rather than min-max. Would that be more robust here?
- The original AlphaZero paper for Go does not normalize Q at all — it uses raw Q values with a carefully tuned `c_puct`. Is normalization necessary for this domain, where rewards are in a bounded range?

**Code reference — Q update:**
```python
# mcts.py lines 273-287
def update_edge(self, mynode, action, reward):
    mynode.action_Q[action] = (mynode.action_N[action] * mynode.action_Q[action] + reward) / (1 + mynode.action_N[action])
    mynode.action_N[action] += 1
    new_q = mynode.action_Q[action]
    if new_q < self.q_min: self.q_min = new_q
    if new_q > self.q_max: self.q_max = new_q
```

### Full MCTS.search() method — the core recursive algorithm

The following is the complete `search()` method. Every audit concern in Section B references this code. Note the three base cases (terminal → 0.0, unexpanded → nn_value, recursive → UCB + descend) and the Bellman update that propagates leaf values to the root.

```python
# mcts.py — MCTS.search() (complete method, lines 150-209)
def search(self, msg) -> float:
    mystate = self.game.hashable_obs

    # Initialize node if we've not been here before
    if mystate not in self.nodes:
        reward: float = self.game.reward
        is_terminal: bool = self.game.terminated or self.game.truncated
        self.nodes[mystate] = MCTSTreeNode(reward, is_terminal)
    mynode = self.nodes[mystate]

    # Base case 1: terminal state → future value is 0
    if mynode.is_terminal_state:
        return 0.0

    # Base case 2: unexpanded node → query network, return value estimate
    if mynode.nn_policy is None:
        assert mynode.nn_value is None
        mypolicy, myvalue, myaction_mask = self.query_net_masked(msg)
        mynode.nn_policy = mypolicy
        mynode.nn_value = myvalue
        mynode.action_mask = myaction_mask
        return myvalue

    # Recursive case: select best action via UCB, descend, backup
    ucbs = self.calc_masked_ucbs(mynode, entab(msg, " ucb"))
    best_action = np.unravel_index(np.argmax(ucbs), ucbs.shape)

    to_step = best_action
    if len(self.game.action_space.shape) == 0: to_step, = to_step
    assert to_step in self.game.action_space
    self.game.step_wrapper(to_step)

    immediate_reward = self.game.reward          # 0.0 at non-terminal derivation steps
    future_value = self.search(entab(msg, " recurse"))   # recursive descent

    # Bellman: Q(s,a) = R(s,a) + gamma * V(s')
    # gamma is implicitly 1.0 — not configurable in MCTS
    total_reward = immediate_reward + future_value

    self.update_edge(mynode, best_action, total_reward)
    mynode.total_N += 1

    return total_reward
```

Also relevant — the simulation loop and Dirichlet noise injection:

```python
# mcts.py — MCTS.perform_simulations() (lines 66-148, simplified)
def perform_simulations(self, msg, add_noise=False):
    mystate = self.game.hashable_obs
    self.q_min = float('inf')    # Reset min-max Q stats per call
    self.q_max = float('-inf')

    # Expand root if not seen before
    if mystate not in self.nodes:
        old_game_state = self.game.stash_state()
        self.search(entab(msg, ", root expand"))
        self.game = self.game.unstash_state(old_game_state)

    mynode = self.nodes[mystate]

    # Add Dirichlet noise (only if fresh root, total_N == 0)
    if add_noise and mynode.total_N == 0:
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))
        mask = mynode.action_mask
        masked_noise = noise * mask
        sum_noise = masked_noise.sum()
        if sum_noise > 0:
            masked_noise /= sum_noise
            mynode.nn_policy = (1 - self.dirichlet_epsilon) * mynode.nn_policy \
                               + self.dirichlet_epsilon * masked_noise

    # Run n_simulations searches, restoring game state after each
    for i in range(self.n_simulations):
        old_game_state = self.game.stash_state()
        self.search(entab(msg, f", simulation {i+1}/{self.n_simulations}"))
        self.game = self.game.unstash_state(old_game_state)

    # Convert visit counts to probabilities via temperature
    mynode = self.nodes[mystate]
    counts = np.zeros_like(mynode.nn_policy)
    for action, count in mynode.action_N.items():
        counts[action] = count

    # Numerically stable: counts^(1/T) = exp(log(counts) / T)
    nonzero = counts > 0
    if nonzero.any():
        log_counts = np.full_like(counts, -np.inf)
        log_counts[nonzero] = np.log(counts[nonzero]) / self.temperature
        log_counts -= log_counts.max()
        probs = np.exp(log_counts)
        probs /= probs.sum()
    else:
        probs = counts  # all zeros — caller will handle

    return probs
```

### B2. MCTS tree is rebuilt from scratch at every derivation step

```python
# agent.py lines 93-101
def policy(self, state, ...):
    current_game_state = state.clone()
    mcts = MCTS(current_game_state, self.net, **self.mcts_params)  # NEW tree every call
    move_probs = mcts.perform_simulations("", add_noise=add_noise)
    return move_probs
```

A fresh `MCTS` instance (with an empty tree) is created for every `policy()` call. In a single derivation game of ~9 steps, the tree is built and discarded 9 times.

**Questions:**
- In standard AlphaZero for Go/Chess, after selecting a move, the MCTS tree is **reused** by descending to the child node. This preserves the search work from the previous step. Why is tree reuse not implemented here?
- During derivation step k, the MCTS tree built at step k-1 already contains partial evaluations of production sequences. After choosing production `a` at step k-1, the subtree rooted at `(state_k-1, a)` is exactly the tree we need for step k. Discarding it means re-doing ~200 simulations of work.
- With 200 simulations per step and 9 steps, that's 1800 total simulations. With tree reuse, the first step would still do 200 sims, but subsequent steps would start from a pre-populated tree, needing fewer new simulations to reach the same quality. Is the current approach wasting ~80% of search computation?
- Is there a technical reason tree reuse is not implemented? (e.g., the `hashable_obs` changes between steps, so old nodes can't be found?) The `stash_state/unstash_state` mechanism restores the game exactly, so tree nodes should be reusable.

### B3. How does sparse terminal-only reward interact with MCTS?

All intermediate derivation steps return reward=0. Only the terminal step (completed program) receives a reward from `LeafEvaluator`. In the MCTS backup:

```python
# mcts.py lines 193-209
immediate_reward = self.game.reward               # 0.0 at non-terminal
future_value = self.search(...)                    # recursive
total_reward = immediate_reward + future_value     # gamma=1.0 implicit
self.update_edge(mynode, best_action, total_reward)
```

At non-terminal nodes, `immediate_reward = 0`, so `total_reward = future_value`. At terminal nodes, `search()` returns `0.0` (line 169). The parent of a terminal node gets `total_reward = leaf_value + 0 = leaf_value`, and this propagates unchanged all the way to the root.

**Questions:**
- With 200 simulations and ~48 possible actions, each action gets on average ~4 visits at the root. Deeper nodes remain mostly unexplored. The neural network value estimate at unexplored nodes is the **only** signal for most of the deeper tree. But the value head is trained on very noisy targets (see Section E). Is there a chicken-and-egg problem: the value head needs good data to learn, but good data requires a good value head to guide MCTS?
- Would adding a small intermediate reward help? For example: `+0.01` for each hole filled (encouraging progress), or a heuristic value for partial programs based on their completed branches?
- The spec document (`derivation_game_mcts_plan.md`, Section 7.1) argues that "MCTS handles sparse terminal-only rewards well because it performs full rollouts." But this was written for the pure-MCTS case (uniform policy, no learning). Does the argument still hold when the value head is being trained simultaneously?
- With a derivation depth of ~9 and branching factor ~48, the full search tree has ~48^9 ≈ 10^15 nodes. With 200 simulations, only 200 paths through this tree are explored. Is this sufficient for the Q-values at the root to be meaningful?

### B4. Dirichlet noise injection — is it working correctly?

```python
# mcts.py lines 84-117
if mystate not in self.nodes:
    self.search(...)  # expand root node — sets nn_policy, but total_N stays 0

mynode = self.nodes[mystate]
if add_noise and mynode.total_N == 0:
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))
    mask = mynode.action_mask
    masked_noise = noise * mask
    masked_noise /= masked_noise.sum()
    mynode.nn_policy = (1 - eps) * mynode.nn_policy + eps * masked_noise
```

**Questions:**
- The root expansion on line 89 calls `self.search()`, which finds the node is new, queries the network, and returns `myvalue`. This is the "unexpanded node" base case (line 172-180), which does NOT increment `total_N` (line 207 is only in the recursive case). So `total_N == 0` is true after expansion, and noise IS added. Is this analysis correct?
- The Dirichlet alpha is 0.25 (moderately concentrated). For 48 actions, this produces noise vectors where most entries are near-zero and a few are large. Is this appropriate for a grammar-derivation setting where many productions lead to similar outcomes? Would a higher alpha (more uniform noise) provide better exploration?
- Noise is applied once at the root. During the 200 simulations, the root's nn_policy is fixed (with noise baked in). Does this provide sufficient exploration diversity for a 48-action space?

---

## C. State Representation & Observation Encoding

_Key file: `derivation_game.py` lines 40-77, 236-249_

### Full DerivationGame — the game interface that MCTS interacts with

This is the glue between MCTS and the grammar. Note how `reset()` always starts from `ProgramHole(budget)`, how `step()` applies a production and detects dead ends, and how `get_action_mask()` uses simple prefix-based masking.

```python
# derivation_game.py — DerivationGame (lines 102-249, key methods)
class DerivationGame(Game):
    """Single-player game where actions are grammar productions."""

    def __init__(self, budget, n_sites, leaf_evaluator):
        super().__init__()
        self.budget = budget
        self.n_sites = n_sites
        self.leaf_evaluator = leaf_evaluator
        self._max_productions = compute_max_productions(budget, n_sites)
        self.action_space = spaces.Discrete(self._max_productions)  # 48 for budget=14, N=6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * budget,), dtype=np.float32,    # 28 floats
        )
        self._deriv_state = None
        self._current_productions = []

    def reset(self, **kwargs):
        self._deriv_state = DerivationState.initial(self.budget)   # Always ProgramHole(budget)
        self._current_productions = self._deriv_state.legal_productions(self.n_sites)
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int):
        prod = self._current_productions[action]
        self._deriv_state = self._deriv_state.apply(prod)
        self._current_productions = self._deriv_state.legal_productions(self.n_sites)

        is_complete = self._deriv_state.is_terminal()
        # Dead end: holes remain but no legal productions (budget 3 or 4)
        is_dead_end = not is_complete and len(self._current_productions) == 0

        terminated = is_complete
        truncated = is_dead_end
        info = {"production": prod}

        if is_complete:
            program = self._deriv_state.to_program()
            reward = self.leaf_evaluator(program)           # ← Terminal reward
            info["program"] = program
        elif is_dead_end:
            reward = 0.0                                    # ← Dead-end: zero reward
            info["dead_end"] = True
        else:
            reward = 0.0                                    # ← Non-terminal: zero reward

        obs = self._encode_obs()
        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        mask = np.zeros(self._max_productions, dtype=bool)
        mask[:len(self._current_productions)] = True         # Prefix-based masking
        return mask

    @property
    def hashable_obs(self) -> str:
        return self._deriv_state.pretty()                    # e.g. "Ite([C:3], Flip(0), [P:5])"

    def stash_state(self):
        """Save game state as a lightweight tuple (no deepcopy — AST nodes are frozen)."""
        return (self._deriv_state, self._current_productions,
                self.obs, self.reward, self.terminated, self.truncated, self.info, self.step_count)

    def unstash_state(self, state):
        (self._deriv_state, self._current_productions,
         self.obs, self.reward, self.terminated, self.truncated, self.info, self.step_count) = state
        return self

    def clone(self):
        new = DerivationGame(self.budget, self.n_sites, self.leaf_evaluator)
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
```

### Encoding scheme

The partial AST is encoded as a fixed-size float32 array via preorder traversal:

```python
# derivation_game.py lines 236-249
def _encode_obs(self):
    obs = np.zeros(2 * self.budget, dtype=np.float32)  # 28 floats for budget=14
    items = _preorder_items(self._deriv_state.root)
    for i, (type_id, param) in enumerate(items):
        if i >= self.budget: break
        obs[2*i] = type_id          # integer 0-8
        obs[2*i + 1] = param        # float (index or budget)
    return obs
```

Each AST node becomes a `(type_id, parameter)` pair:

| Node type | type_id | parameter |
|-----------|---------|-----------|
| PAD (unused) | 0 | 0.0 |
| Flip(j) | 1 | j |
| IsZero(j) | 2 | j |
| Not | 3 | 0.0 |
| And | 4 | 0.0 |
| Ite | 5 | 0.0 |
| Default | 6 | 0.0 |
| ProgramHole(b) | 7 | b (budget) |
| ConditionHole(b) | 8 | b (budget) |

**Questions:**
- The encoding uses preorder traversal, which is sufficient to uniquely reconstruct the tree. But does the **Transformer** implicitly learn to parse tree structure from a flat sequence? Preorder doesn't explicitly encode depth, parent-child relationships, or sibling position. Does this matter for a small tree (≤14 nodes)?
- Padding uses zeros for both type_id and param. Since PAD has type_id=0, the Transformer's padding mask correctly identifies these as invalid positions (line 79 of `derivation_network.py`: `pad_mask_tokens = (type_ids == 0)`). But what if a valid node coincidentally has a zero parameter? (This can't happen since type_id=0 is reserved for PAD, so this is fine.)
- Internal nodes (Not, And, Ite, Default) all have `param = 0.0`. The network must distinguish their roles purely from type_id and positional context. Is this sufficient?
- Holes encode their remaining budget as the parameter. Does the network learn to use this information? Is there evidence that the value head correlates with the budget distribution of remaining holes?

---

## D. Neural Network Architecture

_Key file: `derivation_network.py`_

### Architecture details

```python
# derivation_network.py lines 26-104
class DerivationTransformerModel(nn.Module):
    # Embedding
    type_emb = nn.Embedding(9, d_model)         # 9 node types → d_model
    param_proj = nn.Linear(1, d_model)            # scalar param → d_model
    pos_emb = nn.Embedding(seq_len+1, d_model)   # positional (budget+1 slots)
    cls_emb = nn.Parameter(torch.zeros(d_model))  # learnable [CLS] token

    # Transformer
    encoder = TransformerEncoder(
        TransformerEncoderLayer(d_model=64, nhead=4, dim_ff=256, dropout=0.1, norm_first=True),
        num_layers=2
    )

    # Heads (both from [CLS] token)
    policy_head = nn.Linear(d_model, action_size)  # 64 → 48
    value_head = nn.Linear(d_model, 1)             # 64 → 1
```

**Questions:**
- The policy head always outputs 48 logits, regardless of the current hole type. But the semantics of action indices change depending on whether the leftmost hole is a ProgramHole or ConditionHole, and what its budget is. Does the network have enough information (from the observation) to distinguish these contexts? Or would a **conditioned policy head** (that takes hole type/budget as auxiliary input) be more sample-efficient?
- Both policy and value are derived from the single [CLS] token. The policy needs **position-specific** information (which production to apply at the leftmost hole), but [CLS] pooling summarizes the entire tree. Is information about the specific hole being expanded lost in the pooling?
- The value head is `Linear(64, 1)` — a single linear layer. Is this expressive enough to predict program quality from a 64-dimensional representation? Would a small MLP (e.g., `Linear(64, 64) → ReLU → Linear(64, 1)`) help?
- Token embeddings add `type_emb + param_proj`. Since type_emb is discrete and param_proj is continuous, the addition mixes very different signals. Would concatenation + projection be more appropriate?

### Full predict() method — inference pipeline

```python
# derivation_network.py — DerivationPolicyValueNet.predict() (lines 165-179)
def predict(self, state):
    self.model.cpu()                                            # ← Moves to CPU EVERY call
    nn_input = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        policy_logits, value = self.model(nn_input)
        policy_prob = F.softmax(policy_logits, dim=-1)

    policy_prob = policy_prob.numpy().squeeze(0)
    value = value.numpy().squeeze(0)

    assert policy_prob.shape == (self.action_size,)
    assert value.shape == ()
    return policy_prob, value
```

**Question:** If `train()` moves the model to CUDA (line 185: `model.to(self.DEVICE)`), then every `predict()` call moves it back to CPU. With ~72,000 predict calls per iteration (200 sims × 9 steps × 40 games), this could cause repeated CPU↔CUDA transfers. Is inference always on CPU by design, or is this an oversight?

---

## E. Reward Signal & Value Targets

### E1. Terminal-only reward creates uniform value targets

The DerivationGame returns reward=0 at all non-terminal steps. With `reward_discount = 1.0`:

```python
# agent.py lines 140-145
cumulative_reward = 0.0
for reward in reversed(collected_rewards):
    cumulative_reward = reward + self.reward_discount * cumulative_reward  # discount = 1.0
    discounted_rewards.append(cumulative_reward)
```

For a 9-step episode with rewards `[0, 0, 0, 0, 0, 0, 0, 0, R_terminal]`:

```
discounted_rewards = [R, R, R, R, R, R, R, R, R]
```

**Every step in the episode has the same value target = R_terminal.**

**Questions:**
- The value head sees `ProgramHole(14)` (step 0) and `Ite(IsZero(0), Flip(0), Ite(..., [P:2]))` (step 7) with the **same target value**. How can it learn that step 7 is "closer to done" than step 0?
- In standard AlphaZero for board games, the value target is the game outcome (win=+1, loss=-1), which is also uniform across steps. But in board games, the **state** varies enormously across steps, providing rich input variance. Here, partial ASTs at different derivation stages DO look different — but does the network learn to extract useful value estimates when the targets are constant?
- Would `reward_discount < 1.0` (e.g., 0.95) help? At step 0, the target would be `0.95^8 × R ≈ 0.66R`, and at step 8, it would be `R`. This creates a temporal gradient: later steps have higher value targets, encouraging the network to learn that progress is valuable.
- Alternatively, could the MCTS root value estimate be used as the training target (instead of the episode return)? The root value aggregates information from all 200 simulations and may be a better estimate than the single observed outcome.

### Full LeafEvaluator — how terminal reward is computed

```python
# leaf_evaluator.py — LeafEvaluator (lines 29-155)
class LeafEvaluator:
    """Evaluates complete DSL programs on frozen BitString initial states."""

    def __init__(self, n_sites, frozen_states, game_config, metric="avg_reward",
                 penalty_lambda=0.1, blend_alpha=0.5):
        self.n_sites = n_sites
        self.frozen_states = list(frozen_states)
        self.game_config = game_config
        self.metric = metric        # "avg_reward" by default
        self.penalty_lambda = penalty_lambda
        self.blend_alpha = blend_alpha
        self._max_ops = n_sites * game_config.max_steps(n_sites)
        self._cache: dict[str, float] = {}         # key = program.pretty()
        self._full_cache: dict[str, dict] = {}

    def __call__(self, program: Program) -> float:
        """Evaluate program, returning cached result if available."""
        key = program.pretty()
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        metrics = self._evaluate(program)
        value = self._compute_metric(metrics)
        self._cache[key] = value
        self._full_cache[key] = metrics
        return value

    def _evaluate(self, program: Program) -> dict:
        """Run the program on ALL frozen states and collect raw metrics."""
        solved_count = 0
        total_steps = 0
        total_ops = 0
        total_reward = 0.0

        for x0 in self.frozen_states:      # 1 frozen state by default (n_frozen_states=1)
            env = self.game_config.make_env(self.n_sites, frozen_states=[x0])
            env.reset()
            result = run_policy_episode(env, program)
            if result.solved:
                solved_count += 1
            total_steps += result.total_env_steps
            total_ops += result.total_interp_ops
            total_reward += result.cumulative_reward

        n = len(self.frozen_states)
        return {
            "solve_rate": solved_count / n,
            "avg_reward": total_reward / n,        # ← This is the default metric
            "avg_steps": total_steps / n,
            "avg_ops": total_ops / n,
        }

    def _compute_metric(self, metrics: dict) -> float:
        if self.metric == "avg_reward":
            return metrics["avg_reward"]
        elif self.metric == "solve_rate":
            return metrics["solve_rate"]
        elif self.metric == "penalized_reward":
            penalty = self.penalty_lambda * metrics["avg_ops"] / max(self._max_ops, 1)
            return metrics["avg_reward"] - penalty
        elif self.metric == "weighted":
            return (self.blend_alpha * metrics["solve_rate"]
                    + (1 - self.blend_alpha) * metrics["avg_reward"])
```

### E2. Reward distribution — is avg_reward discriminative?

The leaf evaluator returns `avg_reward` = mean potential-based shaped reward across frozen states (1 by default, configurable via `n_frozen_states`). With onemax potential and N=6:

- The reward for each flip is `+1/6 ≈ 0.167` (if flipping a zero to one) or `-1/6` (if flipping a one to zero)
- Episodes terminate after N steps (6 steps)
- Cumulative reward per state ranges from about `-1.0` to `+0.67`
- Average across 15 states: most programs cluster near `0.0 ± 0.2`

**Question:** Is the reward signal discriminative enough? If most programs score between `-0.1` and `+0.3`, the value head must learn to predict within this narrow range. With only ~360 training examples per iteration (many with the same target), is the signal-to-noise ratio sufficient for learning?

### E3. Fixed frozen evaluation states — is there overfitting risk?

```python
# game_config.py
def all_initial_states(n_sites, n_ones):
    """All C(n_sites, n_ones) initial states."""
```

For N=6, n_ones=2: there are C(6,2) = 15 possible initial states. The default configuration now uses only **1 frozen state** (`n_frozen_states=1`), selected as the first of the 15. Every program is evaluated on this single state, every time. There is no randomization, no train/test split.

**Question:** With only 1 frozen evaluation state, the overfitting risk is more pronounced than with all 15 states. Could a program achieve a high `avg_reward` on this specific state while being poor on others? For OneMax, the potential is symmetric in bit positions, so this may not matter much. But for `leading_ones` or `binval`, a single evaluation state would not be representative. Should `n_frozen_states` be increased for more robust program evaluation?

---

## F. Training Loop & Optimization

_Key files: `trainer.py`, `gated_trainer.py`, `derivation_network.py`_

### F1. Loss function weighting — is policy over-emphasized?

```python
# derivation_network.py lines 234-236
loss_value = criterion_value(outputs_value, targets_value)    # MSE
loss_policy = criterion_policy(outputs_policy, targets_policy) # CrossEntropy with soft targets
loss = loss_value + policy_weight * loss_policy                # policy_weight = 2.0
```

**Questions:**
- Policy loss is weighted 2x relative to value loss. In AlphaZero for Go, the standard is equal weighting. Given that the value signal is already very sparse (uniform targets per episode, see E1), does the 2x policy weight further suppress value learning?
- The policy targets (MCTS visit distributions) are derived from 200 simulations with a mostly-untrained network. Are these targets high-quality enough to warrant 2x weight? Or are we training the policy head to match noisy, uninformative visit counts?
- Has the sensitivity to `policy_weight` been explored? What happens at 0.5 or 1.0?

### Full DerivationPolicyValueNet.train() — the training loop

```python
# derivation_network.py — DerivationPolicyValueNet.train() (lines 183-261)
def train(self, examples, needs_reshape=True, print_all_epochs=False):
    model = self.model
    model.to(self.DEVICE)                              # Move to CUDA for training
    tp = self.training_params
    policy_weight = tp["policy_weight"]                # 2.0

    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()           # Supports soft targets (MCTS visit dist)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tp["learning_rate"],                        # 3e-4
        weight_decay=tp["weight_decay"],               # 1e-4
    )

    if needs_reshape:
        states = torch.from_numpy(np.array([s for s, _, _ in examples], dtype=np.float32))
        policies = torch.from_numpy(np.array([p for _, p, _ in examples], dtype=np.float32))
        values = torch.from_numpy(np.array([v for _, _, v in examples], dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(states, policies, values)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=tp["batch_size"], shuffle=True   # batch_size = 32
    )

    for epoch in range(tp["epochs"]):                  # 5 epochs
        model.train()
        for inputs, targets_policy, targets_value in train_loader:
            inputs = inputs.to(self.DEVICE)
            targets_policy = targets_policy.to(self.DEVICE)
            targets_value = targets_value.to(self.DEVICE)

            optimizer.zero_grad()
            outputs_policy, outputs_value = model(inputs)

            loss_value = criterion_value(outputs_value, targets_value)
            loss_policy = criterion_policy(outputs_policy, targets_policy)
            loss = loss_value + policy_weight * loss_policy     # 2.0x policy weight

            loss.backward()
            optimizer.step()            # ← No gradient clipping

    return model, train_batch_losses, train_losses, policy_losses, value_losses
```

### F2. Training stabilization — gradient clipping and monitoring

```python
# derivation_network.py lines 238-239
loss.backward()
optimizer.step()
```

**Questions:**
- There is no gradient clipping (`torch.nn.utils.clip_grad_norm_`). With small batch sizes (32), noisy value targets (all identical per episode), and MSE loss, gradients could spike. Is this a concern?
- There is no validation loss tracking. Training always runs for exactly 5 epochs. Could the network be overfitting within those 5 epochs given the small dataset (~7200 examples for ~50K parameters)?
- There is no learning rate scheduling. After 30 iterations, the learning rate is still 3e-4. Would decay (cosine annealing, step decay) help in later iterations?

### Full Trainer.train_iteration() — orchestration of self-play + training

```python
# trainer.py — Trainer (key methods, lines 68-176)
def _collect_training_examples(self) -> list:
    """Collect training examples by playing games."""
    multiprocessing_function = partial(self.agent.play_for_experience, self.game)
    arg_tuples = [
        (i, self.agent._randseed("train"), self.agent._randseed("mcts"))
        for i in range(self.n_games_per_train)          # 40 games
    ]
    # Sequential mode (n_procs=-1): run games one at a time
    train_example_sets = []
    for j, args in enumerate(arg_tuples):
        result = multiprocessing_function(*args)
        train_example_sets.append(result)
    return train_example_sets

def _process_training_examples(self, new_train_examples: list) -> list:
    """Accumulate examples, keep last N iterations, flatten."""
    self.all_training_examples.append(new_train_examples)
    if self.n_past_iterations_to_train is not None and \
        len(self.all_training_examples) > self.n_past_iterations_to_train:
        self.all_training_examples.pop(0)               # Hard truncation
    flat_examples = list(itertools.chain.from_iterable(
        itertools.chain.from_iterable(self.all_training_examples)
    ))
    return flat_examples

def train_iteration(self) -> None:
    """Run a single training iteration: collect examples and train."""
    # Step 1: Self-play → collect (obs, pi_MCTS, discounted_reward) triples
    train_example_sets = self._collect_training_examples()
    experience = [example_set[0] for example_set in train_example_sets]  # strip cumulative return

    # Step 2: Flatten and buffer
    flat_examples = self._process_training_examples(experience)

    # Step 3: Train network
    self.net.train(flat_examples)
```

Also relevant — how `play_for_experience` calls `play_one_round`:

```python
# agent.py — Agent.play_for_experience() (lines 152-163)
def play_for_experience(self, game, id, reset_seed, interaction_seed, ...):
    current_game_state = game.clone()
    current_game_state.reset_wrapper(seed=reset_seed)
    return self.play_one_round(current_game_state, random_seed=interaction_seed)

# agent.py — Agent.play_one_round() (lines 106-150, key logic)
def play_one_round(self, game, max_moves=10_000, random_seed=None, ...):
    current_game_state = game.clone()
    rng = np.random.default_rng(random_seed)

    collected_experience = []
    collected_rewards = []
    for i in range(max_moves):
        move_probs = self.policy(current_game_state, ...)   # ← Creates fresh MCTS each time
        action_idx = rng.choice(len(move_probs), p=move_probs)
        collected_experience.append((current_game_state.obs.copy(), move_probs))
        _, reward, terminated, truncated, _ = current_game_state.step_wrapper(action_idx)
        collected_rewards.append(reward)
        if terminated or truncated:
            break

    # Calculate discounted rewards (with discount=1.0, all equal terminal reward)
    discounted_rewards = []
    cumulative_reward = 0.0
    for reward in reversed(collected_rewards):
        cumulative_reward = reward + self.reward_discount * cumulative_reward
        discounted_rewards.append(cumulative_reward)
    discounted_rewards.reverse()

    return [(obs, pi, v) for (obs, pi), v in zip(collected_experience, discounted_rewards)], cumulative_reward
```

### F3. Replay buffer — hard truncation

```python
# trainer.py lines 125-127
if len(self.all_training_examples) > self.n_past_iterations_to_train:
    self.all_training_examples.pop(0)  # drop oldest iteration entirely
```

**Question:** When the oldest iteration is dropped, the training data composition shifts suddenly. With 20 iterations retained and 40 games per iteration, dropping one iteration removes ~360 examples (~5% of the buffer). Is this significant enough to cause training instability? Would gradual downweighting or prioritized sampling be more stable?

### F4. Small training dataset

Per iteration: 40 games × ~9 steps = ~360 new examples.
Total buffer: 20 iterations × 360 = ~7200 examples.
Network parameters: ~50K (Transformer with d=64, 2 layers).
Training: 5 epochs, batch_size=32 → ~225 batches/epoch → ~1125 gradient updates.

**Question:** Is ~7200 examples sufficient for a ~50K parameter network? The ratio is ~7 parameters per example (improved from ~18 with previous hyperparameters, but still high compared to typical deep learning ratios of 0.1-1). Is the network memorizing rather than generalizing?

---

## G. Evaluation & Gating Mechanism

_Key files: `evaluator.py`, `gated_trainer.py`_

### Full GatedTrainer.train_iteration() — the accept/reject flow

```python
# gated_trainer.py — GatedTrainer.train_iteration() (complete method, lines 30-74)
def train_iteration(self) -> tuple[float, bool]:
    """Run one gated training iteration."""
    # 1. Snapshot old agent (full deepcopy)
    old_agent = copy.deepcopy(self.trainer.agent)

    # 2. Train (modifies net weights in-place)
    self.trainer.train_iteration()

    # 3. Snapshot new agent for pit
    new_agent = copy.deepcopy(self.trainer.agent)

    # 4. Pit new vs old
    score = self.evaluator.pit(new_agent=new_agent, old_agent=old_agent)

    # 5. Gate decision
    accepted = score >= self.acceptance_threshold    # threshold = 0.55
    if not accepted:
        # Restore old weights IN-PLACE via load_state_dict
        old_state_dict = old_agent.net.model.state_dict()
        self.trainer.net.model.load_state_dict(old_state_dict)

    return score, accepted
```

### Full Evaluator.pit() — how agents are compared

```python
# evaluator.py — Evaluator.pit() (complete method, lines 70-128)
EVAL_TEMPERATURE = 0.05   # Near-greedy action selection

def pit(self, new_agent, old_agent, try_without_mcts=False) -> float:
    """Compare agent_new vs agent_old and return win rate of new agent."""
    mp_manager = MultiprocessingManager(new_agent.net, old_agent.net, self)
    mp_manager.push()
    try:
        arg_tuples = [
            (old_agent._randseed("eval"), old_agent._randseed("mcts"),
             new_agent, old_agent, try_without_mcts)
            for i in range(self.n_games)                       # n_games = 20
        ]
        eval_results = MultiprocessingManager.starmap(
            self._play_for_eval, arg_tuples, self.n_procs
        )
    finally:
        mp_manager.pop()

    old_rewards = np.array([r["old_net"] for r in eval_results])
    new_rewards = np.array([r["new_net"] for r in eval_results])

    wins = np.sum(new_rewards > old_rewards)
    ties = np.sum(np.isclose(new_rewards, old_rewards))
    losses = np.sum(new_rewards < old_rewards)
    score = (wins + ties / 2) / self.n_games

    return score

# evaluator.py — _play_for_eval() (lines 34-68)
def _play_for_eval(self, reset_seed, mcts_seed, new_agent, old_agent, ...):
    """Play one eval game for each agent and return rewards."""
    base_game = new_agent.game.clone()
    base_game.reset_wrapper(seed=reset_seed)
    old_game = base_game.clone()      # Both agents play from SAME initial state
    new_game = base_game.clone()

    old_trajectory, old_cumulative_reward = old_agent.play_one_round(
        game=old_game, random_seed=mcts_seed,
        add_noise=False, temperature_override=EVAL_TEMPERATURE,    # 0.05
    )
    new_trajectory, new_cumulative_reward = new_agent.play_one_round(
        game=new_game, random_seed=mcts_seed,
        add_noise=False, temperature_override=EVAL_TEMPERATURE,
    )
    return {"old_net": old_cumulative_reward, "new_net": new_cumulative_reward}
```

### G1. Twenty evaluation games — statistical power

```python
# derivation_config.py line 92
self.evaluator = EvaluatorConfig(n_games=20, ...)
```

With 20 evaluation games, the win-rate score has granularity of 2.5% (each game contributes 1/20 = 5% for a win, 2.5% for a tie). The acceptance threshold is 0.55, meaning the new network must win more than it loses by a meaningful margin.

**Empirical evidence:** In the `20260301_213808` experiment (9 iterations), the gate accepted 8 iterations and **rejected 1** (iteration 5, score=0.50 < threshold 0.55). The gate scores ranged from 0.55 to 0.725, showing genuine variance across iterations. This confirms the gate is not a no-op.

```python
# evaluator.py lines 115-118
wins = np.sum(new_rewards > old_rewards)
ties = np.sum(np.isclose(new_rewards, old_rewards))
losses = np.sum(new_rewards < old_rewards)
score = (wins + ties / 2) / self.n_games
```

**Remaining questions:**
- With 20 games, is the gate statistically powerful enough to reject a marginally worse network? A network that wins 9/20 and ties 2/20 gets score = (9 + 1) / 20 = 0.50, which is correctly rejected. But a network that is only slightly worse might still pass the threshold by chance.
- Since both agents start from `ProgramHole(14)` with near-greedy temperature (0.05) and no Dirichlet noise, outcomes per game are largely deterministic given the MCTS random seed. The 20 games use different seeds (via `_randseed("eval")`), providing variation. But is the seed-based variation sufficient, or would additional stochasticity (e.g., moderate temperature) improve the gate's discriminatory power?
- Is there a fundamental issue with evaluating **program synthesis** agents via game-by-game comparison? Two agents with different policies may converge to the same final program through different derivation paths, resulting in ties even when one agent has a genuinely better search strategy.

### G2. Deepcopy overhead

```python
# gated_trainer.py lines 38, 44
old_agent = copy.deepcopy(self.trainer.agent)  # before training
new_agent = copy.deepcopy(self.trainer.agent)  # after training
```

**Question:** Two `deepcopy` calls per iteration copy the entire agent (including the game, network, and all internal state). For the current small network (~50K params), this is fast. But the LeafEvaluator inside the game has an unbounded cache (`_cache` and `_full_cache`). After many iterations, this cache could become large. Is it being deep-copied as well?

---

## H. Scalability Questions

### H1. Program count growth

| Budget | Programs (N=6) | Derivation depth | Max productions at any step |
|--------|---------------|------------------|-----------------------------|
| 2      | 6             | 1                | 6                           |
| 5      | 216           | 3                | 66                          |
| 7      | 7,776         | 4-5              | 66                          |
| 9      | 419,904       | 5-6              | 66                          |
| 11     | 27,060,480    | 6-7              | 66                          |
| 14     | 151,173,432   | ~9               | 48                          |

**Questions:**
- The program count grows super-exponentially. At budget=14, there are 151M programs but the system evaluates ~360 per iteration (0.0002%). Is AlphaZero the right approach for such a large, discrete search space?
- Standard AlphaZero works well for Go (250^361 possible games) because the neural network **generalizes** across board positions via spatial symmetries. What symmetries exist in the derivation space? Do partial ASTs with similar structure lead to similar programs?
- At what budget does the approach break down? Is there a phase transition where MCTS-guided search stops finding better programs than random search?

### H2. MCTS simulation budget vs. search tree size

With 200 simulations and branching factor ~48, each simulation is a single path through the tree. The tree has ~48^9 ≈ 10^15 leaf nodes.

**Question:** 200 paths through a 10^15-leaf tree — is the neural network prior doing enough work to focus the search? If the prior is near-uniform (which it would be in early training), MCTS essentially does random search. How many iterations of training are needed before the prior becomes informative enough to meaningfully guide MCTS?

---

## I. Correctness & Edge Cases

### I1. Duplicate `action_Q` initialization

```python
# mcts.py lines 33-34
self.action_Q = {}
self.action_Q = {}  # duplicate — harmless but suspicious
```

### I2. `q_u_history` grows without bound

```python
# mcts.py line 265
mynode.q_u_history.append((q, u_val))
```

This list is appended on every UCB calculation but never cleared or read. Over 200 simulations × up to 48 actions per node, this accumulates thousands of entries per episode. With 40 games per iteration, this could consume non-trivial memory.

### I3. `_replace_leftmost_hole` uses `_count_holes` for branching

```python
# derivation.py lines 98-110
if isinstance(node, Ite):
    if _count_holes(node.cond) > 0:        # Full subtree traversal
        new_cond = _replace_leftmost_hole(node.cond, replacement)
        return Ite(new_cond, node.action, node.else_prog)
    if _count_holes(node.action) > 0:      # Full subtree traversal
        ...
    if _count_holes(node.else_prog) > 0:   # Full subtree traversal
        ...
```

Each `_count_holes` call traverses the entire subtree. The function could use `_find_leftmost_hole` (which returns early on first hit) for cheaper branching. Not a correctness issue, but `_replace_leftmost_hole` is called on every derivation step.

### I4. `model.cpu()` on every predict call

```python
# derivation_network.py line 166
def predict(self, state):
    self.model.cpu()  # Move to CPU every single call
    ...
```

**Question:** If the model was on CUDA during training (line 185: `model.to(self.DEVICE)`), then every `predict()` call moves it back to CPU. With ~72,000 predict calls per iteration (200 sims × 9 steps × 40 games), is this causing unnecessary device transfers?

### I5. The interpreter's cost model disagrees with the actual interpreter

```python
# interpreter.py lines 38-39 — actual evaluation
elif isinstance(cond, And):
    return eval_condition(cond.left, state) and eval_condition(cond.right, state)  # SHORT-CIRCUITS

# interpreter.py lines 75-76 — cost model
elif isinstance(cond, And):
    return 1 + _condition_ops(cond.left) + _condition_ops(cond.right)  # NO short-circuit
```

The actual Python `and` operator short-circuits: if the left operand is false, the right is never evaluated. But the cost model always charges both sides. This is documented as a deliberate design choice, but it means the `penalized_reward` metric penalizes And conditions more than their actual runtime cost.

**Question:** Does this discrepancy affect program quality? If the learner uses `penalized_reward`, it may avoid And conditions (perceived as expensive) even when they would be efficient at runtime (short-circuiting on the first false condition).

---

## Summary of Open Questions

For each concern, the core question is: **does this meaningfully affect the algorithm's ability to learn?**

### Resolved
- **[A1] Dead-end grammar productions** — eliminated by filtering on `count_programs() == 0`. Action space reduced 60→48.
- **[A2] Double-negation and And commutativity** — eliminated by canonical grammar pruning (`parent_is_not` flag, canonical And ordering).
- **[G1] Statistical power of gating** — increased from 2 to 20 evaluation games with threshold 0.55. Empirically validated: 1 rejection in 9 iterations.

### Grammar Design
- How much of the 151M search space is semantically redundant even after canonical pruning (And idempotence, contradictions, unreachable branches)?
- Are the missing primitives (IsOne, Or) limiting the quality of reachable programs?
- Does the budget-exact constraint prevent discovering simpler optimal programs?

### Search
- Is dynamic Q-normalization causing exploration instability?
- How much computation is wasted by rebuilding the MCTS tree from scratch each step?
- Is 200 simulations in a 10^15-leaf tree enough for meaningful Q-estimates?

### Learning
- Can the value head learn from uniform-across-episode targets?
- Is the 2.0x policy weight suppressing value learning?
- Is 7200 examples enough for a 50K-parameter network?

### Evaluation
- With only 1 frozen evaluation state (`n_frozen_states=1`), is there overfitting risk? Would increasing to more states improve program robustness?
- Is the gate's seed-based variation sufficient, or would additional stochasticity improve discriminatory power?

### Fundamental Framing
- Is AlphaZero the right approach for grammar-guided program synthesis, given the massive discrete search space and sparse rewards?
- Would alternative approaches (beam search, evolutionary methods, exhaustive enumeration at smaller budgets) be competitive or superior?
- Is the two-level MDP (grammar derivation → program evaluation) introducing unnecessary complexity compared to directly searching the program space?
