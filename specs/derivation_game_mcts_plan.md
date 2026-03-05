# MCTS-Guided Program Synthesis via DerivationGame

## 1. Context & Motivation

We have two systems that currently operate independently:

1. **AlphaZero framework** (`core/`) -- MCTS + neural network for single-player game solving
2. **Decision-list DSL** (`instances/bitstring/dsl/`) -- grammar-based enumeration of interpretable BitString policies

Currently, discovering good DSL programs requires **exhaustive enumeration**: DFS over all
programs of a given AST budget L. At N=3, L=8, this is 513 programs (tractable). At N=5,
L=11, it explodes to ~60,000+ (slow). At N=10, L=14+, it becomes infeasible.

**The idea:** Cast program synthesis as a single-player game where each "move" is a grammar
production that fills the leftmost hole in a partial AST. MCTS then searches this derivation
space, guided by leaf evaluation (running completed programs on the BitString environment).
This reuses the existing MCTS/Game infrastructure without modification.

```
                    Exhaustive Enumeration              MCTS-Guided Search
                    ────────────────────────            ────────────────────
                    Explores ALL 513 programs           Explores ~87 programs (17%)
                    Finds best by brute force           Finds best by guided search
                    O(|programs|) cost                  O(simulations) cost
                    Fails at large N, L                 Scales with more sims
```

---

## 2. Architecture Tree

```
AlphaZero_PP/
│
├── src/alphazeropp/
│   │
│   ├── core/                              ── EXISTING (do NOT modify) ──────────
│   │   ├── game.py                        Game[ObsType, ActType] base class
│   │   │   ├── reset() -> (obs, info)       │ DerivationGame subclasses this
│   │   │   ├── step(action) -> (o,r,t,t,i)  │ directly (NOT EnvGame)
│   │   │   ├── get_action_mask() -> array    │
│   │   │   ├── hashable_obs -> Hashable      │
│   │   │   ├── stash_state() -> snapshot     │ custom impl for performance
│   │   │   └── unstash_state(snap) -> self   │
│   │   │                                     │
│   │   ├── mcts.py                        MCTS search engine
│   │   │   ├── perform_simulations()        │ consumes DerivationGame
│   │   │   ├── search() [recursive]         │ via Game interface
│   │   │   ├── query_net_masked()           │ uses action mask + net.predict()
│   │   │   └── calc_masked_ucbs()           │ PUCT with Q-normalization
│   │   │                                     │
│   │   ├── policy_value_net.py            PolicyValueNet ABC
│   │   │   ├── predict(state) -> (pi, v)    │ UniformPolicyValueNet
│   │   │   └── train(examples)              │ implements this
│   │   │                                     │
│   │   ├── agent.py                       Agent (MCTS + net orchestrator)
│   │   └── config.py                      MetaConfig hierarchy
│   │
│   ├── instances/bitstring/
│   │   ├── game.py                        ── EXISTING ─────────────────────────
│   │   │   ├── BitStringGym               Gymnasium env: flip bits to all-ones
│   │   │   └── BitStringGame(EnvGame)     AlphaZero wrapper
│   │   │
│   │   ├── shaped_env.py                  ShapedBitStringGym (reward shaping)
│   │   ├── potentials.py                  onemax, leading_ones, binval
│   │   │
│   │   └── dsl/                           ── EXISTING + NEW ──────────────────
│   │       ├── __init__.py                MODIFY: add exports
│   │       ├── ast_nodes.py               EXISTING: Flip, IsZero, Not, And, Ite, Default
│   │       ├── interpreter.py             EXISTING: eval_program, interp_ops, run_policy_episode
│   │       ├── budget_grammar.py          EXISTING: enumerate/count programs & conditions
│   │       ├── derivation.py              EXISTING: DerivationState, Production, leftmost expansion
│   │       │
│   │       ├── game_config.py             NEW (Step 1): GameConfig + all_initial_states
│   │       ├── leaf_evaluator.py          NEW (Step 2): LeafEvaluator with caching
│   │       └── derivation_game.py         NEW (Step 3+4): DerivationGame + UniformPolicyValueNet
│   │
│   ├── training/                          EXISTING (not used initially)
│   │   ├── trainer.py                     Could train policy on derivation self-play (future)
│   │   └── evaluator.py
│   │
│   └── utils/
│       ├── statistics.py                  EXISTING: StatisticsManager (JSONL logging)
│       └── ...
│
├── scripts/
│   ├── enumerate_dsl.py                   MODIFY: import GameConfig from dsl/game_config.py
│   └── run_derivation_mcts.py             NEW (Step 5): experiment runner + plots
│
├── tests/
│   ├── test_cfg_grammar.py                EXISTING: grammar/derivation tests
│   └── test_derivation_game.py            NEW (Step 6): game + MCTS integration tests
│
├── results/
│   └── derivation_mcts/                   NEW: output directory for plots + JSONL
│
└── specs/
    ├── bitstring_cfg_grammar.md           EXISTING: grammar specification
    └── derivation_game_mcts_plan.md       THIS FILE
```

---

## 3. The Two-Level MDP Structure

This design involves two nested MDPs. Understanding their interaction is critical.

```
┌─────────────────────────────────────────────────────────────────────┐
│  OUTER MDP: DerivationGame                                         │
│  State:  partial AST with holes                                    │
│  Action: grammar production (fill leftmost hole)                   │
│  Reward: 0 at non-terminal, leaf_value at terminal                 │
│  Terminal: all holes filled (complete program)                     │
│                                                                     │
│  [P:8] ──prod──> Ite([C:1],Flip(0),[P:5]) ──prod──> ... ──> prog  │
│   s0                    s1                              s_T         │
│   r=0                   r=0                         r=leaf_val      │
│                                                         │           │
│                                                         ▼           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  INNER MDP: BitStringGym (frozen evaluation)                 │   │
│  │  The completed program is run as a policy on each            │   │
│  │  frozen initial state. Returns solve_rate / avg_reward.      │   │
│  │                                                               │   │
│  │  x0=[0,1,0]  ──Flip(0)──> [1,1,0] ──Flip(2)──> [1,1,1] OK  │   │
│  │  x0=[0,0,1]  ──Flip(0)──> [1,0,1] ──Flip(1)──> [1,1,1] OK  │   │
│  │  x0=[1,0,0]  ──Flip(1)──> [1,1,0] ──Flip(2)──> [1,1,1] OK  │   │
│  │                                                               │   │
│  │  leaf_value = avg_reward([+0.67, +0.67, +0.67]) = +0.67     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why this works with MCTS

MCTS handles sparse terminal-only rewards well because it performs **full rollouts** from
root to leaf, backing up the leaf value through the entire path. Unlike policy gradient
methods (which need dense reward signals), MCTS accumulates:

```
Q(s0, prod_0) = r0 + Q(s1, prod_1) = 0 + 0 + ... + 0 + leaf_value
```

The PUCT formula then directs exploration toward productions whose subtrees lead to
high-value programs. With enough simulations, MCTS discovers which early production
choices (e.g., "start with Ite(IsZero(0), Flip(0), ...)" vs "start with
Ite(IsZero(2), Flip(1), ...)") lead to better final programs.

---

## 4. Data Flow Through the System

```
                        ┌──────────────┐
                        │  MCTS Engine │
                        │  (core/mcts) │
                        └──────┬───────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐
    │ DerivationGame│   │ UniformPVNet │   │  Action Mask      │
    │  .step()     │   │  .predict()  │   │  .get_action_mask()│
    │  .reset()    │   │  -> (pi, v)  │   │  -> bool[K]       │
    └──────┬───────┘   └──────────────┘   └───────────────────┘
           │
           │ on terminal step
           ▼
    ┌──────────────┐
    │ LeafEvaluator│
    │  cache lookup│──── hit ──> return cached value
    │  cache miss  │
    └──────┬───────┘
           │ evaluate program
           ▼
    ┌──────────────────┐
    │  run_policy_episode()  │  (runs N_frozen times)
    │  on ShapedBitStringGym │
    │  -> EpisodeResult      │
    └────────────────────────┘
           │
           ▼
    scalar leaf_value (avg_reward / solve_rate / etc.)
```

---

## 5. Detailed Component Design

### 5.1 GameConfig + all_initial_states (Step 1)

**File:** `src/alphazeropp/instances/bitstring/dsl/game_config.py`

Extracted from `scripts/enumerate_dsl.py` lines 67-96 and 258-265. No new logic.

```python
@dataclass
class GameConfig:
    bit_flip: bool = True
    sparse_reward: bool = False
    n_ones: int = 2
    potential_fn: Callable[[np.ndarray], int] = None
    potential_name: str = "onemax"

    def max_steps(self, n_sites: int) -> int
    def make_env(self, n_sites: int, frozen_states=None) -> ShapedBitStringGym

def all_initial_states(n_sites: int, n_ones: int) -> list[np.ndarray]
```

**Reuses:** `BitStringGym`, `ShapedBitStringGym`, `POTENTIAL_REGISTRY`

---

### 5.2 LeafEvaluator (Step 2)

**File:** `src/alphazeropp/instances/bitstring/dsl/leaf_evaluator.py`

#### Metric Design Space

The `metric` parameter controls what scalar MCTS optimizes. Configurable at runtime.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Leaf Evaluation Metric Landscape                     │
│                                                                          │
│  Signal richness (gradient quality for MCTS)                             │
│  ▲                                                                       │
│  │                                                                       │
│  │  avg_reward ●──── best gradient, continuous, potential-shaped          │
│  │               \                                                       │
│  │  penalized    ●── continuous, adds efficiency pressure                │
│  │  _reward       \                                                      │
│  │                 \                                                     │
│  │  weighted ●──────── blends solve_rate + avg_reward                    │
│  │                                                                       │
│  │  solve_rate ●──── discrete (k/n), sparse but objective-aligned        │
│  │                                                                       │
│  └──────────────────────────────────────────────────────────────────── ▶  │
│                    Alignment with true objective                          │
│                                                                          │
│  Interaction with potential function:                                     │
│  ┌─────────────┬──────────────────────────────────────────────────┐      │
│  │ potential    │ what avg_reward measures                         │      │
│  ├─────────────┼──────────────────────────────────────────────────┤      │
│  │ onemax      │ fraction of bits set to 1 (order-agnostic)      │      │
│  │ leading_ones│ length of leading all-1s prefix (order-aware)   │      │
│  │ binval      │ binary value (bit 0 = MSB, exponential weights) │      │
│  │ sparse_only │ collapses to solve_rate (no intermediate signal)│      │
│  └─────────────┴──────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

| Metric | Formula | Signal | Best for |
|--------|---------|--------|----------|
| `"avg_reward"` | E[shaped_reward] over frozen states | Continuous | Default. Richest gradient. |
| `"solve_rate"` | fraction where all bits=1 at end | Discrete k/n | Correctness-only objectives |
| `"penalized_reward"` | avg_reward - lambda * avg_ops/max_ops | Continuous | Favoring efficient programs |
| `"weighted"` | alpha * solve_rate + (1-alpha) * avg_reward | Continuous | Blending both signals |

**Recommendation:** Start with `"avg_reward"` + `onemax`. Richest gradient. Compare
with `"solve_rate"` to measure how much sparser signal hurts sample efficiency.

#### Class Design

```python
class LeafEvaluator:
    METRICS = ("avg_reward", "solve_rate", "penalized_reward", "weighted")

    def __init__(self, n_sites, frozen_states, game_config,
                 metric="avg_reward", penalty_lambda=0.1, blend_alpha=0.5)

    def __call__(self, program: Program) -> float
    def get_all_metrics(self, program: Program) -> dict
    def stats(self) -> dict
```

**Cache:** keyed by `program.pretty()`. `_full_cache` stores all metrics per program so
we can report solve_rate even when optimizing avg_reward. Cache is essential because
MCTS visits the same terminal many times across rounds.

**Reuses:** `run_policy_episode()` from `interpreter.py`, `GameConfig.make_env()`

---

### 5.3 DerivationGame (Step 3)

**File:** `src/alphazeropp/instances/bitstring/dsl/derivation_game.py`

Subclasses `Game` directly (NOT `EnvGame`) -- there is no Gymnasium env.

```python
class DerivationGame(Game[np.ndarray, int]):

    def __init__(self, n_sites: int, budget: int, leaf_evaluator: LeafEvaluator = None)
    def reset(self) -> tuple[np.ndarray, dict]
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]
    def get_action_mask(self) -> np.ndarray
    def stash_state(self) -> tuple       # custom: avoids deepcopy
    def unstash_state(self, state) -> self
    @property
    def hashable_obs(self) -> str         # = DerivationState.pretty()
```

#### Observation encoding

Fixed-size 1D `float32[2*budget]`. Preorder traversal of partial AST, each node encoded
as (type_id, parameter):

```
Node type IDs: PAD=0  Flip=1  IsZero=2  Not=3  And=4  Ite=5  Default=6  PHole=7  CHole=8
Parameter:     -      index   index     -      -      -      -          budget   budget

Example (budget=5, after 1st production):
  Ite([C:1], Flip(0), [P:2])
  -> [(5,0), (8,1), (1,0), (7,2), (0,0)]  # 5 pairs in 10 floats
```

#### Action space

`Discrete(max_productions)` where `max_productions` is precomputed across all hole
types and budgets that can appear during derivation:

```python
def compute_max_productions(n_sites, budget):
    max_p = 0
    for b in range(2, budget + 1):
        max_p = max(max_p, len(_program_productions(b, n_sites)))
    for b in range(1, budget - 1):
        max_p = max(max_p, len(_condition_productions(b, n_sites)))
    return max_p
```

Action `i` maps to `legal_productions(n_sites)[i]`. Mask: first `len(legal)` True,
rest False.

#### Reward model

```
step 0:  [P:8] ──prod──> partial_AST     reward = 0.0   terminated = False
step 1:  partial ──prod──> partial        reward = 0.0   terminated = False
  ...
step k:  partial ──prod──> complete_prog  reward = leaf_evaluator(prog)  terminated = True
```

#### MCTS reward propagation (verified against mcts.py:140-199)

```
Terminal node:    direct_reward = leaf_value, returns 0.0 (future value)
Parent of term:  immediate_reward = leaf_value (from step), future = 0.0
                 total_reward = leaf_value + 0.0 = leaf_value  ✓
Grandparent:     immediate_reward = 0.0, future = leaf_value
                 total_reward = 0.0 + leaf_value = leaf_value  ✓
  ... (propagates unchanged to root)
```

#### Stash / Unstash

Custom implementation (avoids full deepcopy):

```python
def stash_state(self):
    return (self._deriv_state, self._current_productions,
            self.obs, self.reward, self.terminated, self.truncated,
            self.info, self.step_count)

def unstash_state(self, state):
    (self._deriv_state, self._current_productions,
     self.obs, self.reward, self.terminated, self.truncated,
     self.info, self.step_count) = state
    return self
```

Safe because all AST nodes and Productions are frozen dataclasses (immutable).

#### MCTS compatibility checklist

| MCTS code path | Requirement | DerivationGame |
|---|---|---|
| `mcts.py:73` `self.game.hashable_obs` | Hashable, stable | `pretty()` string |
| `mcts.py:148` `self.game.reward` | float after step, None after reset | yes |
| `mcts.py:149` `self.game.terminated or truncated` | bool after step, None after reset | yes |
| `mcts.py:166` `query_net_masked()` | action mask is numpy bool array | `bool[max_prod]` |
| `mcts.py:174` `np.unravel_index(argmax, shape)` | policy shape matches mask shape | `(max_prod,)` |
| `mcts.py:179` `len(action_space.shape)==0` | Discrete -> shape=() | yes |
| `mcts.py:180` `to_step in action_space` | 0 <= action < max_prod | yes |
| `mcts.py:88-90` stash/unstash | hashable_obs unchanged | yes (immutable AST) |
| `mcts.py:123` assert hashable_obs | stable after stash+search+unstash | yes |

---

### 5.4 UniformPolicyValueNet (Step 4)

**In same file:** `derivation_game.py`

```python
class UniformPolicyValueNet(PolicyValueNet):
    """Uniform policy (all productions equally likely), value=0.

    MCTS with uniform policy = pure exploration guided only by leaf values.
    Any structure in the search comes from backed-up leaf evaluations,
    not from policy prior. This is the correct baseline -- a learned
    policy would accelerate convergence but is not needed for correctness.
    """
    def predict(self, state) -> tuple[np.ndarray, np.ndarray]:
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        value = np.array(0.0, dtype=np.float32)
        return policy, value
```

**RL perspective on value=0:** Returning 0 at non-terminal states means MCTS gets NO
intermediate value estimates. It relies entirely on backed-up terminal rewards. This is
analogous to MCTS without a value network (pure rollout-based). For shallow derivation
trees (depth 3-5 at budget 5-8), this is fine. For deeper trees (budget 11+), a learned
value network would significantly help by providing "how promising is this partial AST?"
estimates.

---

## 6. Derivation Game as a Search Tree

Concrete example: N=3, budget=5. The derivation tree has this structure:

```
                              [P:5]                          (root: 1 ProgramHole)
                                │
              ┌─────────────────┼─────────────────┐
        P(5)->Ite(C(1),      P(5)->Ite(C(1),    P(5)->Ite(C(1),
        Flip(0),P(2))        Flip(1),P(2))       Flip(2),P(2))
              │                     │                   │
     Ite([C:1],Flip(0),[P:2])      ...                 ...
              │
     ┌────────┼────────┐
  C(1)->    C(1)->    C(1)->
  IsZero(0) IsZero(1) IsZero(2)
     │
  Ite(IsZero(0),Flip(0),[P:2])
     │
     ┌────────┼────────┐
  P(2)->    P(2)->    P(2)->
  Def(F(0)) Def(F(1)) Def(F(2))
     │
  ┌──────────────────────────────────┐
  │ TERMINAL: node_count=5           │
  │ Program: if IsZero(0): Flip(0)   │
  │          else: Flip(0)           │
  │                                  │
  │ leaf_eval -> run on 3 states     │
  │ solve_rate = 1/3                 │
  │ avg_reward = +0.22               │
  └──────────────────────────────────┘

Total programs at this budget: 27  (3 cond_choices x 3 flip_choices x 3 default_choices)
Derivation depth: always 3 steps  (ProgramHole -> ConditionHole -> ProgramHole -> done)
Branching factor: 3 at each step  (for N=3)
MCTS with 100 sims explores ~12-15 unique terminal programs
```

For budget=8 (two Ite rules + one Default):

```
                              [P:8]
                                │
              ┌─────────────────┼────────── ... ──┐
        P(8)->Ite(C(1),      P(8)->Ite(C(2),    P(8)->Ite(C(1),
        Flip(0),P(5))        Flip(0),P(4))       Flip(2),P(5))
              │                  │                    │
             ...              P(4)=0!              deeper...
                              (impossible)
              │
     Ite([C:1],Flip(0),[P:5])      depth: 3-5 steps
              │                     branching: 3-9 per step
             ...                    programs: 513 total
              │                     MCTS explores: ~87 (17%)
           TERMINAL
```

---

## 7. RL Soundness Analysis

### 7.1 Is the reward signal sufficient?

**Concern:** DerivationGame has reward=0 for all non-terminal steps. Only the terminal
step carries the leaf evaluation. This is extremely sparse in the outer MDP.

**Why it's OK:**
- MCTS performs full tree search to leaves, backing up values. Unlike policy gradient
  (which needs per-step rewards), MCTS naturally handles terminal-only rewards.
- Derivation depth is shallow (3-5 steps for budget 5-8). MCTS can exhaustively
  explore shallow trees.
- The backed-up Q-values at the root encode "which first production leads to the
  best programs on average?" -- exactly what we want.

**When it breaks down:** For budget >= 11, derivation depth exceeds 6-8 steps and the
branching factor grows. At N=5, budget=11: ~35 productions per step, depth ~6,
search space ~35^6 ~ 1.8 billion paths. Pure MCTS with uniform policy would need
millions of simulations. A learned policy prior would be essential.

### 7.2 Does MCTS correctly propagate the leaf value?

**Verified by tracing mcts.py lines 140-199:**

```
search(root_state):
  step(prod_0) -> reward=0, new_state_1
  total_reward = 0 + search(state_1)

  search(state_1):
    step(prod_1) -> reward=0, new_state_2
    total_reward = 0 + search(state_2)

    search(state_2):
      step(prod_2) -> reward=leaf_val, terminal_state
      total_reward = leaf_val + search(terminal)

      search(terminal):
        return 0.0   # line 159: terminal future value = 0

      total_reward = leaf_val + 0.0 = leaf_val  ← backed up to state_2

    total_reward = 0 + leaf_val = leaf_val  ← backed up to state_1

  total_reward = 0 + leaf_val = leaf_val  ← backed up to root
```

The leaf_val is stored in `Q(root, prod_0)` via `update_edge()`. Across multiple
simulations, `Q(root, prod_0)` converges to the **average** leaf value of all
programs reachable from `prod_0`. MCTS then selects the production with highest Q.

### 7.3 Discount factor

MCTS uses gamma=1.0 (line 194: `total_reward = immediate_reward + future_value`,
no gamma). This is correct for derivation: we want the full terminal reward to
propagate equally to all earlier decisions. Discounting would penalize deeper
derivations, which has no semantic justification here.

### 7.4 Exploration vs exploitation

With uniform policy + PUCT:
- **Early simulations:** uniform prior -> equal exploration of all productions
- **Later simulations:** Q-values differentiate good from bad productions, PUCT
  exploits high-Q subtrees while still exploring low-visit alternatives
- **Q-normalization** (mcts.py:242-248): normalizes Q to [0,1] range, preventing
  the exploration term from being dominated by raw Q-value scale

### 7.5 Canonical derivation prevents duplicate programs

Each complete program has exactly ONE derivation path (canonical leftmost expansion).
This means:
- No wasted simulations exploring different paths to the same program
- MCTS tree has no aliasing -- each terminal node is a unique program
- The number of leaf nodes in the full tree = `count_programs(n_sites, budget)`

### 7.6 Sample efficiency: MCTS vs. exhaustive enumeration

```
Budget │ Programs │ MCTS sims │ Coverage │ Finds optimum?
───────┼──────────┼───────────┼──────────┼────────────────
  5    │    27    │    100    │  ~55%    │ very likely
  8    │   513    │    200    │  ~17%    │ likely (needs ~500 for high confidence)
 11    │ ~14,000  │   2000    │  ~1.4%   │ possible (guided by Q-values)
 14    │ ~400,000 │   5000    │  ~0.1%   │ needs learned policy
```

### 7.7 Determinism requirements

For same `--seed` -> byte-identical JSONL:
- `np.random.seed(seed)` before each run (for Dirichlet noise in MCTS)
- Production ordering is already deterministic (defined by `_program_productions`,
  `_condition_productions` in `derivation.py`)
- `argmax` tie-breaking: numpy is deterministic for equal values
- LeafEvaluator: `run_policy_episode` is deterministic (no randomness in interpreter)
- Action selection: use `temperature=0.01` (near-greedy) or `argmax` for determinism

---

## 8. Experiment Runner Output

**File:** `scripts/run_derivation_mcts.py`

Six phases with comprehensive printing:

### Phase 1: Configuration
Print game parameters, grammar summary, metric explanation, frozen initial states.

### Phase 2: Ground Truth (small N only)
Exhaustive enumeration to establish the "ceiling" -- what's the best possible program?

### Phase 3: MCTS Search
Per-round: program found, its metrics, cache stats. Track best-so-far.

### Phase 4: Best Program Episode Trace
Show the best program running on each frozen initial state, step-by-step.
Reuse `format_trace()` from `interpreter.py`.

### Phase 5: Derivation Trace
Show how MCTS constructed the best program, production-by-production.
Reuse `format_derivation_trace()` from `derivation.py`.

### Phase 6: Summary Statistics
Total simulations, programs evaluated, cache hit rate, compute counters.

### Plots (static PNGs to `results/derivation_mcts/`)

| Plot | Shows | Why meaningful |
|------|-------|----------------|
| 1. MCTS Convergence | best solve_rate vs. cumulative sims | How quickly does MCTS find the optimum? |
| 2. Discovery Histogram | programs found by MCTS, bucketed by quality | Is MCTS biased toward good programs? |
| 3. Best vs Worst Execution | side-by-side bitstring state grids | What does a smart vs dumb policy look like? |
| 4. Derivation Depth Profile | program count and quality by derivation depth | Structure of the search space |

Plot 3 reuses `plot_comparative_evolution()` from `scripts/enumerate_dsl.py`.

### CLI Parameters

```
--n_sites N          Number of bits (default: 3)
--budget L           AST budget (default: 8)
--n_simulations S    MCTS sims per round (default: 200)
--n_rounds R         MCTS rounds (default: 10)
--metric M           avg_reward|solve_rate|penalized_reward|weighted
--penalty_lambda F   for penalized_reward (default: 0.1)
--blend_alpha F      for weighted (default: 0.5)
--c_exploration F    MCTS exploration constant (default: 1.5)
--temperature F      MCTS temperature (default: 0.1)
--seed S             Random seed (default: 42)
--potential P        onemax|leading_ones|binval
--n_ones K           Initial 1-bits (default: 2)
--no-interactive     Skip prompts
--skip-enumeration   Skip ground truth
```

---

## 9. Tests

**File:** `tests/test_derivation_game.py`

| Test Class | Tests | What's verified |
|------------|-------|-----------------|
| `TestDerivationGameBasics` | 8 tests | obs shape, step mechanics, mask, terminal, hashable_obs |
| `TestStashUnstash` | 2 tests | roundtrip restore, independence from mutations |
| `TestUniformPolicyValueNet` | 3 tests | uniform policy, zero value, correct shapes |
| `TestLeafEvaluator` | 4 tests | good/bad programs, caching, stats tracking |
| `TestMCTSIntegration` | 4 tests | single sim, determinism, finds optimal at L=5 and L=8 |

---

## 10. Consistency Invariants

1. **Budget conservation:** every terminal derivation -> `program.node_count() == budget`
2. **Action mask alignment:** `mask[i]=True` iff `i < len(legal_productions())`
3. **hashable_obs injectivity:** distinct partial ASTs -> distinct `pretty()` strings
4. **Reward propagation:** leaf_value reaches root Q-table with no distortion (gamma=1, no intermediate rewards)
5. **Stash/unstash round-trip:** `hashable_obs` identical before stash and after unstash

---

## 11. Implementation Order

```
Step 1: game_config.py          Extract from enumerate_dsl.py (pure refactor)
        └── pytest tests/test_cfg_grammar.py (regression check)

Step 2: leaf_evaluator.py       Depends on: interpreter.py, game_config.py
        └── pytest (add to test_derivation_game.py)

Step 3: derivation_game.py      Depends on: derivation.py, leaf_evaluator.py
        └── pytest (add game + stash tests)

Step 4: Update __init__.py      Add exports

Step 5: Update enumerate_dsl.py Import GameConfig from game_config.py
        └── pytest tests/test_cfg_grammar.py (regression check)

Step 6: test_derivation_game.py Full test suite including MCTS integration
        └── pytest tests/test_derivation_game.py -v

Step 7: run_derivation_mcts.py  Experiment runner with plots
        └── python scripts/run_derivation_mcts.py --no-interactive --seed 42
```

---

## 12. Verification Commands

```bash
# Unit + integration tests
pytest tests/test_derivation_game.py -v

# Regression tests (existing grammar tests still pass)
pytest tests/test_cfg_grammar.py -v

# Run experiment
python scripts/run_derivation_mcts.py \
  --n_sites 3 --budget 8 --n_simulations 200 --n_rounds 5 \
  --seed 42 --no-interactive

# Determinism check
python scripts/run_derivation_mcts.py --n_sites 3 --budget 5 --seed 42 --no-interactive
# Run again, diff JSONL output
```

---

## 13. Future Extensions (not in scope)

- **Learned policy:** Replace `UniformPolicyValueNet` with a trained network.
  The full AlphaZero loop (self-play -> train -> evaluate) would learn which
  productions are promising, dramatically improving sample efficiency at large budgets.
- **Persistent MCTS tree:** Reuse the search tree across rounds (same initial state).
  Currently each round builds a fresh tree.
- **Intermediate value estimates:** Train a value network to predict "how good is
  this partial AST?" without needing to complete and evaluate it.
- **Beam search:** Instead of MCTS, use beam search over derivations with the leaf
  evaluator as scoring function.
