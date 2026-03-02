# CFG Redesign: Domain-Informed Grammars for Grammar-Guided Program Synthesis on Bitstring Problems

_A comprehensive analysis of why the current context-free grammar fails to leverage domain knowledge, and rigorous proposals for alternative grammar designs. Intended as a self-contained research document for deep investigation._

---

## Table of Contents

- [Part 1: Self-Contained Problem Summary](#part-1-self-contained-problem-summary)
  - [1.1 The Bitstring Optimization Problem Family](#11-the-bitstring-optimization-problem-family)
  - [1.2 Two Approaches: Direct Play vs. Program Synthesis](#12-two-approaches-direct-play-vs-program-synthesis)
  - [1.3 The Decision-List DSL](#13-the-decision-list-dsl)
  - [1.4 The Size-Budget Grammar](#14-the-size-budget-grammar)
  - [1.5 Experimental Evidence](#15-experimental-evidence)
- [Part 2: Analysis of Current Grammar's Deficiencies](#part-2-analysis-of-current-grammars-deficiencies)
  - [2.1 Search Space Inflation Without Benefit](#21-search-space-inflation-without-benefit)
  - [2.2 The Budget-Exact Constraint Problem](#22-the-budget-exact-constraint-problem)
  - [2.3 Structural Overhead Consumes Budget](#23-structural-overhead-consumes-budget)
  - [2.4 Cost-Benefit Summary](#24-cost-benefit-summary)
- [Part 3: Alternative CFG Designs](#part-3-alternative-cfg-designs)
  - [3.0 Design Principles and Evaluation Criteria](#30-design-principles-and-evaluation-criteria)
  - [3.1 Proposal A: Priority-Scan Grammar](#31-proposal-a-priority-scan-grammar)
  - [3.2 Proposal B: Variable-Depth Decision-List Grammar](#32-proposal-b-variable-depth-decision-list-grammar)
  - [3.3 Proposal C: Decoupled Condition-Action Grammar](#33-proposal-c-decoupled-condition-action-grammar)
  - [3.4 Proposal D: Problem-Aware Atom Grammar](#34-proposal-d-problem-aware-atom-grammar)
  - [3.5 Proposal E: Hierarchical Strategy Grammar](#35-proposal-e-hierarchical-strategy-grammar)
  - [3.6 Proposal F: Grammar with Learned Abstractions](#36-proposal-f-grammar-with-learned-abstractions)
- [Part 4: Theoretical Analysis](#part-4-theoretical-analysis)
  - [4.1 Search Space Size Comparison](#41-search-space-size-comparison)
  - [4.2 MCTS Feasibility Analysis](#42-mcts-feasibility-analysis)
  - [4.3 Useful Program Density](#43-useful-program-density)
  - [4.4 Information-Theoretic Analysis](#44-information-theoretic-analysis)
- [Part 5: Research Questions for Deep Investigation](#part-5-research-questions-for-deep-investigation)

---

## Part 1: Self-Contained Problem Summary

### 1.1 The Bitstring Optimization Problem Family

#### Formal MDP Definition

The **BitString game** is a finite-horizon Markov Decision Process:

- **State space**: `S = {0, 1}^N` — binary vectors of length `N` (typically N = 5 to 10)
- **Initial state**: A random state with exactly `n_ones` bits set to 1 (default `n_ones = 2`), remaining bits 0
- **Action space**: `A = {0, 1, ..., N-1}` — choose a bit index to flip
- **Transition**: `s_{t+1}[i] = 1 - s_t[i]` if `i = action`, else `s_{t+1}[i] = s_t[i]` (toggle the selected bit)
- **Terminal condition**: All bits equal 1, OR `max_steps` reached
- **Max steps**: `2N` (dense reward mode) or `N - n_ones` (sparse mode)

#### Three Potential Functions

The reward is shaped by a **potential function** `Φ: {0,1}^N → Z`, yielding per-step reward:

```
r_t = (Φ(s_{t+1}) - Φ(s_t)) / N
```

The three potential functions define three problem variants with fundamentally different structure:

**1. OneMax** — `Φ(x) = Σ x[i]` (sum of all bits)

- Order-invariant: every zero-to-one flip yields `+1/N`, every one-to-zero flip yields `-1/N`
- The optimal strategy is: flip ANY zero bit. The ORDER does not matter.
- This is the easiest variant. The optimal policy is ANY permutation of "scan through bits, flip zeros."
- Optimal reward from initial state with `n_ones = 2`: `(N - 2)/N` (flip all `N - 2` zeros)

**2. LeadingOnes** — `Φ(x) = length of leading all-ones prefix`

- Example: `[1,1,1,0,1] → 3`, `[0,1,1,1,1] → 0`
- Order-dependent: only flipping the bit at position `leading_ones(x)` increases the potential
- The optimal strategy is: always flip the leftmost zero bit (bit at index `leading_ones(x)`)
- Highly deceptive: flipping bit 5 when bit 0 is zero yields zero potential change
- Medium difficulty

**3. BinVal** — `Φ(x) = Σ x[i] · 2^(N-1-i)` (binary value, x[0] is MSB)

- Example: `[1,0,1] → 5` (4 + 0 + 1)
- Weight-dependent: flipping bit `i` from 0→1 changes potential by `+2^(N-1-i)`
- The optimal strategy is: flip the highest-weight zero bit first (MSB-first ordering)
- Hardest variant: the reward signal is exponentially skewed toward high-order bits

#### What "Optimal" Means

For each variant, we can characterize the optimal decision-list program:

| Variant | Optimal Strategy | Description |
|---------|-----------------|-------------|
| OneMax | Any scan order | Check bits in any order, flip first zero found |
| LeadingOnes | Left-to-right scan | Check bit 0, then 1, then 2, ... |
| BinVal | MSB-first scan | Check bit 0 (MSB), then 1, then 2, ... |

**Key insight**: For ALL three variants, the optimal policy is a **priority ordering over bit indices** — a permutation `σ` of `{0, 1, ..., N-1}` where you scan bits in order `σ(0), σ(1), ...` and flip the first zero. The variants differ only in WHICH permutation is optimal:
- OneMax: any permutation
- LeadingOnes: identity permutation `(0, 1, 2, ..., N-1)`
- BinVal: identity permutation (MSB = index 0 has highest weight)

### 1.2 Two Approaches: Direct Play vs. Program Synthesis

The codebase implements two fundamentally different approaches to learning the BitString policy.

#### Approach 1: Direct BitString Play (`scripts/run_bitstring.py`)

AlphaZero learns a neural policy that directly maps bit-vector states to flip actions:

- **State representation**: The N-bit vector itself (N floats)
- **Action space**: `Discrete(N)` — which bit to flip
- **Neural network**: Small MLP (2 hidden layers of 128 units) → (policy: softmax over N actions, value: scalar)
- **MCTS**: 20 simulations per move
- **Reward**: Dense — `+1/N` for flipping a zero, `-1/N` for flipping a one, every step
- **Training**: 40 games per iteration, converges in ~10 iterations for N = 6–10

This approach is straightforward: the neural network sees the current bit-vector and learns which bit to flip. The action space IS the domain (N choices). MCTS with 20 simulations easily handles this.

#### Approach 2: Grammar-Guided Program Synthesis (`scripts/run_derivation.py`)

AlphaZero synthesizes an explicit decision-list **program** by constructing it one grammar production at a time:

- **State representation**: A partial AST (Abstract Syntax Tree) with "holes" — unfilled subtrees
- **Action space**: Grammar productions (up to 48 choices per step for budget=14, N=6)
- **Neural network**: Transformer encoder (d=64, 4 heads, 2 layers) → (policy: softmax over max_productions, value: scalar)
- **MCTS**: 200 simulations per derivation step
- **Reward**: Sparse — 0 at all non-terminal steps; at the terminal step (complete program), the program is run as a policy on frozen bitstring states and the resulting reward is returned
- **Training**: 40 games per iteration, requires 9+ iterations for N=6, stalls for N=7 at budget=14

This approach casts program synthesis as a **single-player game** (the "DerivationGame") and searches the space of grammar derivations with AlphaZero. A completed derivation yields a decision-list program that can be evaluated on the bitstring environment.

#### Performance Comparison

| Property | Direct Play | Program Synthesis |
|----------|------------|-------------------|
| State space | `{0,1}^N` (N floats) | Partial ASTs (2×budget floats) |
| Action space per step | N (e.g., 6) | Up to 48 (budget=14, N=6) |
| Derivation steps | N (one per bit flip) | ~9 (one per grammar production) |
| Total decisions per game | ~N | ~9 |
| Reward signal | Dense (every step) | Sparse (terminal only) |
| MCTS simulations | 20 | 200 |
| Iterations to solve N=6 | ~10 | 9+ (imperfect at L=14) |
| N=7 at budget=14 | N/A (trivial) | Stalls at 0.57 for 30 iterations |
| Search space | N^(max_steps) paths | 151M programs (L=14, N=6) |
| Branching factor | 6 | 48 |
| Output | Black-box neural policy | Interpretable decision-list program |

The program synthesis approach produces **interpretable programs** (a clear advantage) but is dramatically less sample-efficient than direct play. The central question of this document is: **is this inefficiency inherent to program synthesis, or is it caused by a poorly designed grammar?**

### 1.3 The Decision-List DSL

The Domain-Specific Language (DSL) defines the space of synthesizable programs. All programs are **decision lists** — nested if-elif-else chains that execute with **first-match semantics**.

#### AST Node Types

| Category | Node | Fields | Meaning | Node count |
|----------|------|--------|---------|------------|
| Action | `Flip(index)` | index ∈ [0, N) | Emit action: flip bit at index | 1 |
| Condition | `IsZero(index)` | index ∈ [0, N) | True iff state[index] == 0 | 1 |
| Condition | `Not(child)` | child: Condition | Logical negation | 1 + |child| |
| Condition | `And(left, right)` | left, right: Condition | Logical conjunction | 1 + |left| + |right| |
| Program | `Ite(cond, action, else_prog)` | cond: Condition, action: Flip, else_prog: Program | If cond then action, else continue | 1 + |cond| + |action| + |else_prog| |
| Program | `Default(action)` | action: Flip | Always return action (base case) | 1 + |action| = 2 |

#### Execution Semantics

A program is evaluated on a bitstring state `s ∈ {0,1}^N`:

```python
def eval_program(program, state):
    if isinstance(program, Default):
        return program.action.index        # Always fire
    if isinstance(program, Ite):
        if eval_condition(program.cond, state):
            return program.action.index    # First match fires
        return eval_program(program.else_prog, state)  # Try next rule

def eval_condition(cond, state):
    if isinstance(cond, IsZero):   return state[cond.index] == 0
    if isinstance(cond, Not):      return not eval_condition(cond.child, state)
    if isinstance(cond, And):      return eval_condition(cond.left, state) and eval_condition(cond.right, state)
```

**Properties**:
- **Totality**: Every program always returns an action (every path ends at `Default`)
- **Determinism**: Same state always produces same action
- **First-match**: Walk the if-elif chain top-down; first true condition fires
- **Interpretability**: Programs read as readable if-elif-else Python code

#### Example Program (N=6, budget=14)

```python
if IsZero(0):         # 3 nodes: Ite + IsZero + Flip
  Flip(0)
elif IsZero(1):       # 3 nodes
  Flip(1)
elif IsZero(2):       # 3 nodes
  Flip(2)
elif IsZero(3):       # 3 nodes
  Flip(3)
else:                 # 2 nodes: Default + Flip
  Flip(4)
```

Total: 4 rules × 3 nodes + 1 default × 2 nodes = **14 nodes**. This covers 5 of 6 bit positions.

### 1.4 The Size-Budget Grammar

The grammar generates programs by expanding "holes" (unfilled subtrees) one production at a time. Every production is indexed by its **exact AST node count** (the "budget").

#### Production Rules

**Program productions** (generate Program subtrees):
```
P(2) → Default(Flip(j))                        j ∈ [0, N)          — N productions
P(k) → Ite(C(i), Flip(j), P(k-2-i))           k ≥ 5, i ∈ [1, k-4], j ∈ [0, N)
                                                where count_programs(N, k-2-i) > 0
```

**Condition productions** (generate Condition subtrees):
```
C(1) → IsZero(j)                               j ∈ [0, N)          — N productions
C(k) → Not(C(k-1))                             k ≥ 2, if parent ≠ Not  — 1 production
C(k) → And(C(i), C(k-1-i))                     k ≥ 3, i ∈ [1, ⌊(k-1)/2⌋] — canonical ordering
```

**Budget accounting**:
- `Ite(C(i), Flip(j), P(e))`: 1 (Ite) + i (cond) + 1 (Flip) + e (else) = k, so e = k - 2 - i
- `Default(Flip(j))`: 1 + 1 = 2
- `Not(C(k-1))`: 1 + (k-1) = k
- `And(C(i), C(j))`: 1 + i + j = k, so j = k - 1 - i

**Dead budget zones**: P(1), P(3), P(4) produce zero programs (gap between Default=2 and minimum Ite=5).

**Canonicalization** (reduces redundancy):
1. **Double-negation ban**: `Not(Not(c))` ≡ `c`, prevented by tracking `parent_is_not` flag
2. **And commutativity**: `And(c1, c2)` ≡ `And(c2, c1)`, enforced by requiring left_budget ≤ right_budget

#### Concrete Numbers

Program counts at **exact** budget L (verified from codebase):

| N | L=14 | L=17 | L=20 |
|---|------|------|------|
| 5 | 40,597,625 | 4,708,244,500 | 581,752,731,375 |
| 6 | **151,173,432** | 22,176,553,464 | 3,458,336,859,720 |
| 7 | 468,094,501 | 84,149,530,444 | 16,051,342,254,885 |
| 10 | 6,826,261,000 | 1,997,340,661,000 | 617,787,672,351,000 |

Canonical program counts (after pruning double-negation and And commutativity):

| N | L=14 | L=17 | L=20 |
|---|------|------|------|
| 6 | **37,463,688** | 2,750,607,936 | 211,138,199,424 |
| 7 | 127,582,623 | 11,826,282,251 | 1,146,804,987,139 |

Even with canonical pruning, the search space is enormous: **37 million** canonical programs at L=14, N=6.

Maximum productions (branching factor) at any derivation step:

| N | L=14 | L=17 | L=20 |
|---|------|------|------|
| 6 | 48 | 66 | 84 |
| 7 | 56 | 77 | 98 |
| 10 | 80 | 110 | 140 |

Production counts by hole type (N=6, L=14 derivation):

| Hole | Productions | Context |
|------|------------|---------|
| P(14) | 48 | Root: first derivation step |
| P(11) | 30 | After choosing i=1 at P(14) |
| P(8) | 12 | After choosing i=1 at P(11) |
| P(5) | 6 | After choosing i=1 at P(8) |
| P(2) | 6 | Terminal program step |
| C(1) | 6 | Simple IsZero condition |
| C(2) | 1 | Not(C(1)) |
| C(3) | 1 | And(C(1), C(1)) |

Typical derivation depth: **~9 steps** (5 program-hole expansions + 4 condition-hole expansions).

#### The Derivation Process

Starting from `ProgramHole(14)`, the derivation proceeds by always expanding the **leftmost hole**:

```
Step 0: [P:14]
  → Apply P(14) → Ite(C(1), Flip(0), P(11))
Step 1: Ite([C:1], Flip(0), [P:11])
  → Apply C(1) → IsZero(0)
Step 2: Ite(IsZero(0), Flip(0), [P:11])
  → Apply P(11) → Ite(C(1), Flip(1), P(8))
...continues until all holes are filled...
Step 8: Complete program (no holes)
  → LeafEvaluator runs the program on frozen states → terminal reward
```

At each step, the MCTS agent must choose among all legal productions for the current leftmost hole. The Transformer network reads the partial AST (encoded as a flat sequence of (type_id, parameter) pairs) and outputs a policy distribution over productions plus a value estimate.

### 1.5 Experimental Evidence

#### N=6, Budget=14, OneMax (Successful Run)

Experiment `20260301_213808_N6_L14_avg_reward_mcts200_games40_iter10`:
- Configuration: N=6, L=14, OneMax, 200 MCTS sims, 40 games/iter, 1 frozen state
- **Result**: Found a perfect program (100% solve rate) at iteration 9

```
if And(IsZero(3), IsZero(2)):     ← complex condition: 3 nodes
  Flip(3)
elif IsZero(2):                   ← simple condition: 1 node
  Flip(2)
elif Not(IsZero(4)):              ← negated condition: 2 nodes
  Flip(5)
else:
  Flip(4)                         ← default
```

Total: 14 nodes. Covers positions {2, 3, 4, 5}. The initial frozen state has `n_ones=2`, so only 4 bits need flipping. With 1 frozen state, this suffices. But note: this program uses `Not(IsZero(4))` (= "bit 4 is 1") which wastes 2 budget nodes on a condition that checks the WRONG thing for OneMax (you want to flip zeros, not detect ones). The `And` condition costs 3 nodes for the same effect as two separate `IsZero` checks.

#### N=7, Budget=14, OneMax (Failed Run)

Experiment `20260301_224313_N7_L14_avg_reward_mcts200_games40_iter30`:
- Configuration: N=7, L=14, OneMax, 200 MCTS sims, 40 games/iter, 1 frozen state
- **Result**: Stalled at avg_reward=0.571 (4/7) for 25 iterations

```
if Not(IsZero(4)):    ← 2 budget nodes to check "bit 4 is 1" (useless for OneMax)
  Flip(5)
elif Not(IsZero(6)):  ← 2 budget nodes wasted
  Flip(4)
elif Not(IsZero(2)):  ← 2 budget nodes wasted
  Flip(6)
else:
  Flip(2)
```

Total: 14 nodes. The program uses `Not(IsZero(i))` conditions (checking "is bit i already one?"), which is the OPPOSITE of what OneMax needs. It covers 4 out of 5 zero positions. The budget is too small to express the optimal 7-bit program (needs 20 nodes = 6 rules + 1 default = 6×3 + 2).

**Critical observation**: After 30 iterations and evaluating **191,852 unique programs**, the system never found a program better than 0.571 avg_reward. The grammar's budget constraint makes the optimal program physically unreachable.

#### Summary of Performance Data

| Experiment | N | L | Best solve_rate | Best avg_reward | Unique programs | Budget sufficient? |
|-----------|---|---|----------------|----------------|-----------------|-------------------|
| N=5, L=14 | 5 | 14 | TBD | TBD | TBD | Yes (optimal = 14 nodes) |
| N=6, L=14 | 6 | 14 | 100% (1 state) | 0.667 | 54,913 | Partial (optimal = 17) |
| N=7, L=14 | 7 | 14 | 0% | 0.571 | 191,852 | No (optimal = 20) |
| N=7, L=20 | 7 | 20 | TBD | 0.286 (iter 1) | 3,170 | Yes (optimal = 20) |

---

## Part 2: Analysis of Current Grammar's Deficiencies

### 2.1 Search Space Inflation Without Benefit

The grammar's primary role should be to **structure the search** so that MCTS can efficiently navigate toward good programs. Instead, the current grammar inflates the search space by orders of magnitude relative to the useful program space.

#### Quantitative Analysis

**Derivation paths vs. programs:**

At budget=14, N=6, the grammar has:
- **48** max productions at the root
- **~9** derivation steps per complete derivation
- **48^9 ≈ 1.2 × 10^15** possible derivation paths (upper bound; actual is lower due to varying branching)
- **151,173,432** distinct programs (37M canonical)
- Derivation paths per program: **~8 million** on average

Even though each canonical program has exactly one derivation path (by construction), the MCTS agent does not know the canonical path a priori. It must search through the full derivation tree, encountering many paths that lead to non-canonical or dead-end states.

**Branching factor comparison:**

| Decision Problem | Branching Factor | Depth | Search Tree Size |
|-----------------|-----------------|-------|-----------------|
| Direct bitstring (which bit?) | N = 6 | ~6 steps | 6^6 ≈ 47K |
| Grammar derivation (which production?) | 48 | ~9 steps | 48^9 ≈ 1.2 × 10^15 |
| **Inflation factor** | **8x** | **1.5x** | **~25 billion x** |

The grammar multiplied the branching factor by 8× and the depth by 1.5×, yielding a search tree that is approximately **25 billion times larger** than the direct problem's search tree.

#### What the Grammar Encodes vs. What's Needed

**What the grammar encodes:**
- Syntactic correctness of decision-list programs
- Budget-exact constraint on program size
- Arbitrary boolean conditions (IsZero, Not, And)
- Arbitrary condition-action pairings (condition can check bit i, action can flip bit j)

**What domain knowledge would encode:**
- The optimal policy is a priority scan over bit indices
- The only useful condition for OneMax is `IsZero(i)` (is this bit still zero?)
- The condition bit and action bit should usually match (`if IsZero(i): Flip(i)`)
- `Not(IsZero(i))` is almost never useful for OneMax (checking if a bit is already 1)
- `And(...)` conditions waste budget without improving coverage in most cases
- The program should have as many rules as possible (more coverage = better)

**The fundamental disconnect**: The grammar generates programs that permute through condition types, action targets, and structural patterns in a way that is ORTHOGONAL to the actual problem structure. It's like searching for a word in a dictionary by trying random letter combinations instead of using alphabetical order.

### 2.2 The Budget-Exact Constraint Problem

The grammar requires every program to have EXACTLY `budget` AST nodes. This creates two critical problems.

#### Problem 1: Optimal Programs Are Unreachable

The minimum-node optimal program for OneMax with N bits uses:
- `N - 1` rules of `if IsZero(i): Flip(i)` — each costs 3 nodes (Ite + IsZero + Flip)
- 1 default of `Default(Flip(j))` — costs 2 nodes
- Total: **3(N-1) + 2 = 3N - 1** nodes

| N | Optimal nodes (3N-1) | Budget 14 sufficient? | Max rules at L=14 | Coverage |
|---|---------------------|----------------------|-------------------|----------|
| 5 | 14 | Yes (exact fit!) | 4 rules + default | 5/5 bits |
| 6 | 17 | **No** (3 nodes short) | 4 rules + default | 5/6 bits |
| 7 | 20 | **No** (6 nodes short) | 4 rules + default | 5/7 bits |
| 10 | 29 | **No** (15 nodes short) | 4 rules + default | 5/10 bits |

For N ≥ 6, the grammar at budget=14 **physically cannot express** the optimal OneMax program. The system is searching through 151 million programs, none of which can solve the problem optimally.

#### Problem 2: Budget Padding Forces Waste

When the budget is larger than needed for a simple program, the grammar has no "padding" mechanism. The budget must be spent, so the system is forced to generate complex conditions that consume nodes without adding value:

Example: At budget=14 with 4 rules, the system has 14 - (4×3 + 2) = 0 spare nodes if all conditions are `IsZero(i)`. But if the system "wants" only 3 rules, it must spend 14 - (3×3 + 2) = 3 spare nodes on making one condition more complex (e.g., `And(IsZero(i), IsZero(j))` instead of `IsZero(i)`).

This explains why the N=7 experiment's best program uses `Not(IsZero(i))` conditions — these cost 2 nodes each (vs. 1 for `IsZero`), which is a way to "spend" budget on conditions. The grammar forces the system to generate complex conditions not because they're useful, but because the budget demands it.

### 2.3 Structural Overhead Consumes Budget

Each AST node type has an inherent structural cost. For bitstring problems, most of this cost is wasted:

#### Per-Rule Budget Breakdown

| Component | Nodes | Purpose | Useful for OneMax? |
|-----------|-------|---------|-------------------|
| `Ite` | 1 | Structural (if-then-else) | Yes (needed for control flow) |
| `IsZero(i)` | 1 | Check if bit i is zero | Yes (core logic) |
| `Flip(j)` | 1 | Flip bit j | Yes (core action) |
| **Subtotal per rule** | **3** | | |

| Component | Nodes | Purpose | Useful for OneMax? |
|-----------|-------|---------|-------------------|
| `Not(IsZero(i))` | 2 | Check if bit i is one | **No** — checking for ones is useless in OneMax |
| `And(IsZero(i), IsZero(j))` | 3 | Check two bits simultaneously | **Rarely** — two separate rules cost the same (2×3=6) and cover the same bits |
| `Not(And(...))` | 4+ | De Morgan's OR | **No** — disjunctions are not useful for scan strategies |

**Budget efficiency metric**: For a budget of L nodes, the maximum number of useful bit-positions covered is:

```
max_coverage(L) = ⌊(L - 2) / 3⌋ + 1
```

where `(L-2)/3` rules each cover one bit, plus the default covers one more.

| Budget L | Max rules | Max coverage | Coverage/Budget ratio |
|----------|-----------|-------------|----------------------|
| 5 | 1 | 2 | 40% |
| 8 | 2 | 3 | 38% |
| 11 | 3 | 4 | 36% |
| 14 | 4 | 5 | 36% |
| 17 | 5 | 6 | 35% |
| 20 | 6 | 7 | 35% |

Only about **36% of the budget** goes toward useful coverage. The rest is structural overhead (Ite nodes, Flip nodes that duplicate the condition index). Any condition more complex than `IsZero(i)` further decreases this ratio.

### 2.4 Cost-Benefit Summary

#### What the Current Grammar Provides

1. **Syntactic correctness guarantee**: Every completed derivation is a valid, total decision-list program
2. **Systematic enumeration**: No duplicates (with canonical ordering)
3. **Interpretability**: Output is human-readable if-elif-else code
4. **Expressiveness**: Can represent conditions involving multiple bits (via And, Not)

#### What the Current Grammar Costs

1. **48x branching factor** vs. 6x for direct play (8x inflation)
2. **Sparse terminal-only reward** (vs. dense per-step in direct play)
3. **151 million programs** to search through at L=14, N=6 (37M canonical)
4. **~10^15 derivation paths** for 37M programs
5. **Budget constraint prevents optimal programs** for N ≥ 6 at L=14
6. **Condition primitives (Not, And) are mostly useless** for bitstring problems
7. **200 MCTS sims** needed per step (10x more than direct play) — still insufficient for the search space
8. **No domain knowledge**: The grammar structure is generic (works for any decision-list program over any predicate/action set) and encodes zero information about bitstring optimization

#### The Core Problem: Rephrasing Without Compressing

The current grammar takes the original problem (choose which bit to flip) and rephrases it as a **program construction problem** without compressing the search space. In fact, it EXPANDS the search space dramatically:

- Direct play: N choices per step, ~N steps = N^N ≈ 10^5 paths
- Grammar derivation: 48 choices per step, ~9 steps = 48^9 ≈ 10^15 paths

The grammar adds the intermediate layer of "constructing a program" but doesn't use this layer to inject any knowledge about which programs are good. A well-designed grammar should REDUCE the search space by constraining it to only programs that encode sensible strategies.

---

## Part 3: Alternative CFG Designs

### 3.0 Design Principles and Evaluation Criteria

Any alternative grammar should be evaluated on:

1. **Search space size**: Total number of programs generated by the grammar
2. **Useful program density**: Fraction of programs that achieve non-trivial performance
3. **Expressiveness**: Can the grammar express the optimal program for each bitstring variant?
4. **Branching factor**: Maximum number of productions at any derivation step
5. **Derivation depth**: Number of steps to complete a derivation
6. **Domain knowledge**: What structural properties of optimal strategies does the grammar encode?
7. **MCTS compatibility**: Is the grammar usable as a DerivationGame with sparse terminal reward?
8. **Generality**: Does the grammar transfer across OneMax, LeadingOnes, and BinVal?

The **ideal grammar** would:
- Express ALL optimal programs (for all three variants)
- Express FEW non-optimal programs
- Have small branching factor and shallow derivation depth
- Have high useful-program density

### 3.1 Proposal A: Priority-Scan Grammar

**Insight**: For all three bitstring variants, the optimal policy is a **priority ordering** over bit indices. Encode this directly.

#### Formal Grammar

```
Program   ::= Scan(σ)     where σ is a permutation of [0, N)
```

A `Scan(σ)` program translates to the decision list:
```
if IsZero(σ(0)): Flip(σ(0))
elif IsZero(σ(1)): Flip(σ(1))
...
elif IsZero(σ(N-2)): Flip(σ(N-2))
else: Flip(σ(N-1))
```

#### Derivation Process

Build the permutation one element at a time:

```
Step 0: Choose σ(0) from {0, 1, ..., N-1}     → N choices
Step 1: Choose σ(1) from {0, ..., N-1} \ {σ(0)} → N-1 choices
...
Step N-1: σ(N-1) is determined                 → 1 choice
```

Total steps: N-1 (the last element is forced).

#### Search Space Analysis

| N | Programs (N!) | Branching factor | Derivation depth | Current grammar (L=14) |
|---|-------------|-----------------|-----------------|----------------------|
| 5 | 120 | 5 → 1 | 4 | 40,597,625 |
| 6 | 720 | 6 → 1 | 5 | 151,173,432 |
| 7 | 5,040 | 7 → 1 | 6 | 468,094,501 |
| 10 | 3,628,800 | 10 → 1 | 9 | 6,826,261,000 |

**Reduction factor**: 210,000× for N=6. The grammar went from 151M programs to 720.

#### Domain Knowledge Encoded

- Every program checks `IsZero(i)` and flips `i` — condition and action always match
- Every program covers ALL N bits — no wasted budget
- The only degree of freedom is the CHECK ORDER — which is exactly what differs across problem variants
- For OneMax: all 720 permutations are optimal
- For LeadingOnes: exactly 1 permutation is optimal (identity)
- For BinVal: exactly 1 permutation is optimal (identity, since x[0] is MSB)

#### Trade-offs

**Pros:**
- 210,000× smaller search space
- 100% useful program density for OneMax (every program solves optimally)
- Branching factor = N (vs. 48), derivation depth = N-1 (vs. ~9)
- MCTS with even 10 simulations would be sufficient for N=6
- Trivially compatible with the DerivationGame interface

**Cons:**
- Cannot express programs where condition bit ≠ action bit (e.g., `if IsZero(0): Flip(5)`)
- Cannot express programs with complex conditions (And, Not)
- For variants beyond OneMax/LeadingOnes/BinVal, the restriction to "scan and flip matching zeros" may be too narrow
- Does not generalize to problems where the action depends on multiple bits

#### Implementation Sketch

```python
# New hole type
@dataclass(frozen=True)
class ScanHole:
    remaining_indices: frozenset[int]  # bits not yet assigned

# Productions for ScanHole with k remaining indices:
# For each i in remaining_indices:
#   ScanHole({...}) → (i, ScanHole({...} - {i}))
# Terminal when |remaining_indices| == 1: forced choice
```

The observation encoding would be: the partial permutation so far (which bits in which order) + the set of remaining bits. Much simpler than the current preorder-AST encoding.

### 3.2 Proposal B: Variable-Depth Decision-List Grammar

**Insight**: Remove the budget-exact constraint and restrict conditions to `IsZero(i)` only. Allow programs of any depth from 1 rule to N rules.

#### Formal Grammar

```
Program  ::= Default(Flip(j))                           j ∈ [0, N)
           | Ite(IsZero(i), Flip(j), Program)            i, j ∈ [0, N)
```

At each derivation step, the agent chooses:
1. **Terminate**: `Default(Flip(j))` — choose default action j
2. **Extend**: `Ite(IsZero(i), Flip(j), ?)` — add a rule checking bit i, flipping bit j, then continue

#### Derivation Process

```
Step 0: Choose "extend with (i, j)" or "terminate with j"
  If extend: → Ite(IsZero(i_0), Flip(j_0), ?)
Step 1: Choose "extend with (i, j)" or "terminate with j"
  If extend: → Ite(IsZero(i_0), Flip(j_0), Ite(IsZero(i_1), Flip(j_1), ?))
...
Step k: Choose "terminate with j"
  → Complete program with k rules + 1 default
```

#### Search Space Analysis

Programs with exactly d rules + 1 default: N² choices per rule × N for default = N^(2d+1).
Total programs up to D max rules: Σ_{d=0}^{D} N^(2d+1).

| N | D=4 (same depth as L=14) | D=N-1 (full coverage) | Current grammar L=14 |
|---|--------------------------|----------------------|---------------------|
| 6 | 10,365,630 | 373,162,686 | 151,173,432 |
| 7 | 41,194,307 | 2,018,521,050 | 468,094,501 |
| 10 | 1,010,101,010 | 101,010,101,010 | 6,826,261,000 |

For N=6 with D=4, this is 10.4M programs — smaller than the current 151M at the same depth. For D=N-1 (full coverage), the space is 373M — comparable to the current grammar. The key benefit is NOT space reduction but **removal of the budget-exact constraint** and elimination of useless Not/And conditions.

**With the restriction condition_bit = action_bit (scan semantics):**

Programs with exactly d rules (condition = action, no repeats): P(N, d) × N for default.

| N | D=N-1 (full coverage, scan) | Scan total | Current grammar L=14 |
|---|---------------------------|-----------|---------------------|
| 6 | 720 × 6 = 4,320 | 7,422 | 151,173,432 |
| 7 | 5,040 × 7 = 35,280 | 60,620 | 468,094,501 |
| 10 | 3,628,800 × 10 = 36.3M | 62,353,010 | 6,826,261,000 |

#### Domain Knowledge Encoded

- No budget-exact constraint: programs can be any length
- Conditions restricted to `IsZero(i)`: the only useful primitive for bitstring problems
- `Not` and `And` removed entirely: they consume budget without adding value
- The agent learns when to STOP adding rules (the "terminate" action)
- Condition-action decoupling allows `if IsZero(0): Flip(5)` patterns (useful for LeadingOnes where you might want to check a high-order bit and flip a different one)

#### Trade-offs

**Pros:**
- No budget waste: programs are exactly as large as needed
- ~20,000× smaller search space than current grammar (scan variant)
- Branching factor: N²+N at each step (extend: N² choices; terminate: N choices) = 42 for N=6 — similar to current, but with MUCH shallower depth
- For scan variant: branching factor = N+N = 12 for N=6
- Can express the optimal program for any N

**Cons:**
- Variable-length derivations make value estimation harder (different programs have different depths)
- Without the scan restriction, N² productions per step is still moderately large
- Does not express complex multi-bit conditions (but these are rarely useful)

### 3.3 Proposal C: Decoupled Condition-Action Grammar

**Insight**: The current grammar couples structural decisions (how many rules, condition complexity) with parametric decisions (which bit indices) in a single flat action space. Decouple them into a **two-phase derivation** per rule.

#### Formal Grammar

Each rule is built in two steps:

```
Step A (Structure): Choose rule template
  - Terminate(j)        → Default(Flip(j))
  - SimpleRule(i, j)    → Ite(IsZero(i), Flip(j), ?)
  - NegRule(i, j)       → Ite(Not(IsZero(i)), Flip(j), ?)
  - AndRule(i, k, j)    → Ite(And(IsZero(i), IsZero(k)), Flip(j), ?)

Step B (Continue): Recursively build the else-branch
```

But this can be further simplified. Since we're arguing that `Not` and `And` conditions are rarely useful for bitstring problems, the grammar can be:

```
Program  ::= Default(j)              j ∈ [0, N)
           | Rule(i, j, Program)     i, j ∈ [0, N)     — "if IsZero(i): Flip(j)"
```

Each derivation step: choose (type, params):
- **Terminate**: 1 structural choice × N parametric choices = N options
- **Extend**: 1 structural choice × N² parametric choices = N² options
- Total: N² + N options per step

This is identical to Proposal B. The key difference would be in the OBSERVATION ENCODING: instead of encoding the full partial AST, encode the **rule list so far** as a sequence of (condition_bit, action_bit) pairs. This is more natural for a Transformer that must decide the next rule.

### 3.4 Proposal D: Problem-Aware Atom Grammar

**Insight**: Design different condition atoms for each problem variant, encoding variant-specific domain knowledge directly into the grammar.

#### OneMax-Specific Grammar

For OneMax, the only thing that matters is "which zero bits exist?" The grammar:

```
Program ::= ScanOrder(σ)    — permutation, as in Proposal A
```

Total programs: N! (see Proposal A).

#### LeadingOnes-Specific Grammar

For LeadingOnes, the key domain knowledge is: "the next bit to flip is always the bit at the frontier (the current leading-ones count)." A specialized grammar:

```
Program  ::= FixedLeftToRight                         — the optimal strategy
           | PartialLeftToRight(k, FallbackStrategy)  — fix bits 0..k-1 in order, then switch
FallbackStrategy ::= ScanOrder(σ over remaining bits)
```

This encodes the knowledge that "bits should be processed left-to-right, at least initially." The search space is much smaller than the general decision-list space.

Alternatively, with the priority-scan grammar (Proposal A), the optimal program for LeadingOnes is simply `Scan(0, 1, 2, ..., N-1)` — the identity permutation. The MCTS search over N! permutations would need to find this specific one, which is feasible for small N (720 permutations for N=6) but may require many simulations for larger N.

#### BinVal-Specific Grammar

For BinVal, the domain knowledge is: "higher-weight bits should be prioritized." A weighted grammar:

```
Program ::= WeightedScan(σ, weights)    — σ is a permutation, weights guide priority
```

But again, within the priority-scan framework (Proposal A), the optimal BinVal program is `Scan(0, 1, 2, ..., N-1)` (MSB first = index 0 first). Same as LeadingOnes.

#### Universal Problem-Aware Grammar

Rather than designing a separate grammar per variant, we can parameterize the **reward signal** while keeping the grammar fixed:

```
Grammar: Proposal A (Priority-Scan)
Reward: LeafEvaluator with the appropriate potential function
```

The grammar generates all N! scan orderings. The reward signal (from LeafEvaluator with onemax, leading_ones, or binval potential) guides MCTS toward the correct permutation. This is elegant: the grammar encodes the STRUCTURAL knowledge ("optimal policies are scan orderings") while the reward encodes the VARIANT-SPECIFIC knowledge ("this particular ordering is best").

### 3.5 Proposal E: Hierarchical Strategy Grammar

**Insight**: For larger N, even N! becomes large (10! = 3.6M). Decompose the program into hierarchical sub-strategies.

#### Formal Grammar

```
Program     ::= Strategy(Phase_1, Phase_2, ..., Phase_m)
Phase       ::= ScanBlock(BitGroup, ScanOrder)
BitGroup    ::= subset of [0, N)
ScanOrder   ::= permutation of BitGroup
```

A `Strategy` executes phases in order. Each phase scans a group of bits in a specified order. The overall program is a decision list that first handles Phase_1's bits, then Phase_2's bits, etc.

#### Example: BinVal with N=10, m=2 phases

```
Phase 1: ScanBlock({0,1,2,3,4}, order=(0,1,2,3,4))   — high-weight bits first
Phase 2: ScanBlock({5,6,7,8,9}, order=(5,6,7,8,9))   — low-weight bits second
```

This encodes the domain knowledge that "high-order bits are more important" while allowing flexibility in the ordering within each group.

#### Search Space

The derivation has two levels:
1. **Partition**: Choose how to partition N bits into m groups (Stirling numbers)
2. **Order within groups**: Choose scan order for each group

For m=2 groups of sizes k and N-k: C(N, k) × k! × (N-k)! = N! (same as full permutation). So hierarchical decomposition doesn't help unless we restrict the partition structure.

**Restricted version**: Force groups to be contiguous ranges (e.g., {0,...,k-1} and {k,...,N-1}). Then there are N-1 partition points × k! × (N-k)! orderings. This is still N! in the worst case but the search is structured: first choose the split point, then order within groups.

#### Trade-offs

**Pros:**
- Natural decomposition for problems with hierarchical structure (BinVal, LeadingOnes)
- The split-point decision provides early signal about strategy quality
- Compatible with multi-scale MCTS (coarse search for partition, fine search for ordering)

**Cons:**
- Total search space is not reduced unless the grammar restricts partitions
- More complex derivation process
- May not help for OneMax (no hierarchical structure)
- Implementation complexity is higher

### 3.6 Proposal F: Grammar with Learned Abstractions

**Insight**: Rather than hand-designing the grammar, let the system discover useful abstractions from training data.

#### Approach: Library Learning (DreamCoder-style)

1. Start with a minimal grammar (e.g., Proposal B: variable-depth decision lists)
2. After each training epoch, analyze the top-performing programs
3. Identify common sub-patterns (e.g., `if IsZero(i): Flip(i)` appears frequently)
4. Add these as macro-productions to the grammar (e.g., `CheckFlip(i)` = `Ite(IsZero(i), Flip(i), ?)`)
5. Re-run training with the enriched grammar

This is related to **DreamCoder** (Ellis et al., 2021) and **Stitch** (Bowers et al., 2023), which perform library learning for program synthesis.

#### How It Would Work

After the first few iterations of training:
- The system discovers that `if IsZero(i): Flip(i)` is a common pattern in good programs
- The grammar adds `CheckFlip(i)` as a single production (cost: 1 derivation step instead of 3)
- The search space shrinks because common patterns are "compressed" into single steps

After more iterations:
- The system discovers that the best programs are scan orderings
- The grammar adds `ScanPrefix(i_1, ..., i_k)` as a macro
- Eventually, the grammar converges to something like Proposal A

#### Trade-offs

**Pros:**
- No need to hand-design domain-specific grammars
- Automatically adapts to the problem structure
- Could discover abstractions that humans wouldn't think of
- Generalizes to other domains beyond bitstrings

**Cons:**
- Computationally expensive (grammar induction + re-training)
- Requires careful design of the abstraction-discovery mechanism
- May converge slowly or to suboptimal abstractions
- Significant implementation complexity

---

## Part 4: Theoretical Analysis

### 4.1 Search Space Size Comparison

All numbers computed for the exact grammar specifications above.

| Grammar | N=6 | N=7 | N=10 | Max BF | Depth |
|---------|-----|-----|------|--------|-------|
| **Current (L=14)** | **151,173,432** | **468,094,501** | **6,826,261,000** | **48/56/80** | **~9** |
| Current canonical (L=14) | 37,463,688 | 127,582,623 | 2,316,316,500 | same | ~9 |
| A: Priority-Scan | 720 | 5,040 | 3,628,800 | 6/7/10 | 5/6/9 |
| B: Variable-depth (free cond-action, D=4) | 10,365,630 | 41,194,307 | 1,010,101,010 | 42/56/110 | 1–5 |
| B: Variable-depth (scan, cond=action) | 7,422 | 60,620 | 62,353,010 | 12/14/20 | 1–6 |
| D: Problem-aware (scan) | 720 | 5,040 | 3,628,800 | 6/7/10 | 5/6/9 |

**Key takeaway**: The Priority-Scan grammar (Proposal A) reduces the search space by a factor of **210,000×** for N=6 and **93,000×** for N=7 compared to the current grammar. Even the Variable-Depth Scan variant (Proposal B with cond=action) achieves a **20,000× reduction**.

### 4.2 MCTS Feasibility Analysis

How many MCTS simulations are needed to find the optimal program with each grammar?

**Lower bound**: MCTS must visit the optimal derivation path at least once. With uniform prior policy, the probability of reaching the optimal program in one random walk is:

```
P(optimal | uniform) = 1 / (number of programs)
```

Expected number of random walks to find optimal:

| Grammar | N=6 programs | Expected random walks |
|---------|-------------|----------------------|
| Current (L=14) | 151M | 151,173,432 |
| A: Priority-Scan | 720 | 720 (OneMax: 1, since all are optimal) |
| B: Variable-depth scan | 7,422 | 7,422 |

With MCTS (not uniform): the neural network prior guides search. The effective number of simulations needed scales roughly as:

```
sims_needed ≈ branching_factor × depth × log(programs)
```

| Grammar | BF × depth × log₂(programs) | Estimate |
|---------|----------------------------|----------|
| Current (L=14) | 48 × 9 × 27 = 11,664 | ~12,000 sims needed |
| A: Priority-Scan | 6 × 5 × 10 = 300 | ~300 sims needed |
| B: Variable-depth scan | 12 × 6 × 13 = 936 | ~1,000 sims needed |

The current setup uses 200 simulations per step — **far below** the ~12,000 needed for the current grammar, but **sufficient** for the Priority-Scan grammar.

### 4.3 Useful Program Density

What fraction of each grammar's programs achieve non-trivial performance?

#### OneMax

- **Current grammar (L=14, N=6)**: Most programs have near-zero avg_reward. The experiment evaluated 191K of 151M programs (0.13%) and the best achieved 0.571 avg_reward. The density of programs with avg_reward > 0.5 is likely < 0.01%.
- **Priority-Scan (N=6)**: ALL 720 programs achieve 100% solve rate and optimal avg_reward (for OneMax, any scan order works). **Useful density: 100%.**
- **Variable-depth scan (N=6)**: Programs with d = N-1 = 5 rules all achieve 100% solve rate (assuming distinct indices). Programs with fewer rules have partial coverage. Density depends on threshold.

#### LeadingOnes

- **Current grammar (L=14, N=6)**: Very few programs achieve high solve_rate for LeadingOnes, since only the left-to-right scan order works.
- **Priority-Scan (N=6)**: Exactly 1 of 720 programs is optimal (the identity permutation). **Useful density: 0.14%.** Still, 720 is a tractable search space.
- **Variable-depth scan (N=6)**: Similarly, very few programs are optimal.

#### Summary

| Grammar | OneMax density | LeadingOnes density |
|---------|---------------|-------------------|
| Current (L=14) | < 0.01% | << 0.01% |
| A: Priority-Scan | **100%** | 0.14% |
| B: Variable-depth scan | ~10-50% | ~0.1% |

### 4.4 Information-Theoretic Analysis

How much information does each grammar production reveal about the final program's quality?

#### Mutual Information Between Early Productions and Reward

In the **current grammar**: the first production at P(14) chooses between structural templates (Ite(C(i), Flip(j), P(k-2-i)) for various i and j). This determines:
- The condition budget of the first rule (i)
- The action of the first rule (j)
- The budget remaining for subsequent rules (k-2-i)

But the QUALITY of the final program depends on ALL rules, not just the first. So the first production has LOW mutual information with the reward. The MCTS value estimate at the root has high uncertainty, making the search inefficient.

In the **Priority-Scan grammar**: the first production chooses which bit to check first (σ(0)). For OneMax, this has zero mutual information (all choices are equally good). For LeadingOnes, this has MAXIMUM mutual information (only σ(0) = 0 is optimal — all others are suboptimal). The MCTS value estimate can immediately discriminate good from bad first choices.

This is a key advantage: grammars that reveal information about program quality early in the derivation enable MCTS to prune bad branches sooner, requiring fewer simulations.

---

## Part 5: Research Questions for Deep Investigation

### Grammar Design

1. **Minimum grammar complexity**: For the bitstring problem family, what is the smallest grammar (fewest productions) that can express all optimal programs? Is the Priority-Scan grammar (N productions per step) provably minimal?

2. **Grammar expressiveness vs. search efficiency**: Is there a formal trade-off (Pareto frontier) between grammar expressiveness and MCTS sample complexity? Can we derive bounds?

3. **Cross-variant universality**: Can a single grammar handle all three bitstring variants (OneMax, LeadingOnes, BinVal) effectively, with the reward signal alone determining which programs are selected? Or do different variants require different grammars?

4. **Beyond scan orderings**: Are there bitstring problem variants where the optimal policy is NOT a scan ordering? If so, what grammar structure would be needed? Consider variants like:
   - OneMax with non-uniform bit weights
   - Bitstring problems with state-dependent rewards
   - Problems where the optimal action depends on multiple bits simultaneously

### MCTS and Search

5. **MCTS simulation budget**: Given a grammar with branching factor B and depth D, what is the minimum number of MCTS simulations needed to find the optimal program with probability p? How does this depend on the neural network prior quality?

6. **Value estimation with grammar structure**: The current system trains a value head that predicts terminal reward from partial ASTs. For the Priority-Scan grammar, the partial state is a partial permutation. Is it easier or harder for a neural network to predict program quality from a partial permutation vs. a partial AST?

7. **Tree reuse across derivation steps**: The codebase already implements MCTS tree reuse. How does grammar choice affect the benefit of tree reuse? (In the Priority-Scan grammar, tree reuse is especially valuable because the search tree after choosing σ(0) directly contains information about all programs starting with σ(0).)

### Neural Architecture

8. **Observation encoding**: The current system encodes partial ASTs as flat sequences of (type_id, parameter) pairs. For the Priority-Scan grammar, the partial state is a partial permutation. What is the best observation encoding for permutations? Options include:
   - Sequence of chosen indices (variable-length)
   - Set of remaining indices (unordered)
   - Bitmap of chosen/remaining indices

9. **Policy head design**: For the Priority-Scan grammar, the policy head must output a distribution over remaining bit indices (a shrinking set). Should the architecture be a pointer network (attention over remaining candidates) or a masked softmax (fixed N-dimensional output with mask)?

### Scalability

10. **N scaling**: At what N does each proposed grammar become infeasible for MCTS? The Priority-Scan grammar has N! programs, which grows factorially. For N=10, that's 3.6M programs — still feasible. For N=20, that's 2.4 × 10^18 — infeasible. Can hierarchical decomposition (Proposal E) help?

11. **Grammar induction at scale**: Can DreamCoder-style library learning (Proposal F) discover the Priority-Scan abstraction automatically? If so, how many training iterations are needed before the abstraction emerges?

### Theoretical Foundations

12. **PAC-learning bounds**: Can we derive PAC-learning bounds for program synthesis over each grammar? I.e., how many programs must be evaluated to find an ε-optimal program with probability 1-δ?

13. **Grammar entropy**: Define the "entropy" of a grammar as the expected information content of a random derivation. How does grammar entropy relate to MCTS efficiency? Is lower entropy always better, or is there a sweet spot?

14. **Semantic equivalence classes**: For the current grammar, how many semantically distinct programs exist at budget=14? (I.e., programs that produce different actions on at least one state.) This is a strict lower bound on the "useful" search space.

### Implementation and Experimentation

15. **A/B testing protocol**: Design an experiment to compare the current grammar against Proposal A (Priority-Scan) on the same hardware, measuring:
    - Time to find optimal program
    - MCTS simulations per iteration
    - Number of unique programs evaluated
    - Convergence curve (avg_reward vs. iteration)

16. **Grammar transition**: Can we gradually transition from a general grammar (current) to a specialized grammar (Priority-Scan) during training? Start with the general grammar for exploration, then narrow to the scan grammar once the system has identified the "scan" pattern.

17. **Multi-grammar curriculum**: Train with progressively more restrictive grammars:
    - Phase 1: Full grammar (explore broadly)
    - Phase 2: Variable-depth with IsZero-only conditions (remove Not, And)
    - Phase 3: Scan grammar (restrict to condition = action)
    This curriculum could help the system discover the right grammar structure.

---

## Appendix: Code Architecture Reference

### Key Source Files

| File | Role |
|------|------|
| `src/alphazeropp/instances/bitstring/dsl/budget_grammar.py` | Current grammar productions, counting, enumeration |
| `src/alphazeropp/instances/bitstring/dsl/derivation.py` | DerivationState, production generation, tree manipulation |
| `src/alphazeropp/instances/bitstring/dsl/ast_nodes.py` | AST node definitions (Flip, IsZero, Not, And, Ite, Default) |
| `src/alphazeropp/instances/bitstring/dsl/interpreter.py` | Program execution, cost model, episode runner |
| `src/alphazeropp/instances/bitstring/dsl/derivation_game.py` | Game interface for MCTS (observation encoding, action masking) |
| `src/alphazeropp/instances/bitstring/dsl/derivation_network.py` | Transformer policy-value network |
| `src/alphazeropp/instances/bitstring/dsl/leaf_evaluator.py` | Terminal program evaluation on frozen states |
| `src/alphazeropp/instances/bitstring/dsl/derivation_config.py` | Configuration and component wiring |
| `src/alphazeropp/instances/bitstring/game.py` | BitString environment |
| `src/alphazeropp/instances/bitstring/potentials.py` | Potential functions (onemax, leading_ones, binval) |
| `src/alphazeropp/core/mcts.py` | MCTS search engine |
| `src/alphazeropp/core/agent.py` | Agent (wraps MCTS + network) |
| `scripts/run_derivation.py` | Main training script for program synthesis |
| `scripts/run_bitstring.py` | Main training script for direct play |

### Data Flow

```
Training Loop (AlphaZero for Program Synthesis):
│
├── For each iteration:
│   ├── STEP 1: Self-Play (40 games)
│   │   └── Each game:
│   │       ├── Start: ProgramHole(budget)
│   │       ├── For each derivation step:
│   │       │   ├── Transformer reads partial AST → (policy, value)
│   │       │   ├── MCTS(200 sims) refines policy using UCB + value backup
│   │       │   ├── Sample production from MCTS visit counts
│   │       │   └── Apply production to leftmost hole
│   │       └── Terminal: run completed program on frozen bitstring states
│   │           └── LeafEvaluator returns scalar reward
│   │
│   ├── STEP 2: Train Transformer
│   │   └── Loss = MSE(value) + 2.0 × CrossEntropy(policy)
│   │
│   └── STEP 3: Evaluate (pit new vs old network, accept if win_rate ≥ 55%)
│
└── Output: Best program found across all iterations
```

### Configuration Defaults (DerivationConfig)

| Parameter | Value | Notes |
|-----------|-------|-------|
| budget | 14 | Too small for optimal N≥6 program |
| n_sites | 6 | Bitstring length |
| n_ones | 2 | Initial ones |
| potential | onemax | Reward shaping |
| metric | avg_reward | Leaf evaluation |
| d_model | 64 | Transformer width |
| n_heads | 4 | Attention heads |
| n_layers | 2 | Transformer depth |
| n_simulations | 200 | MCTS rollouts per step |
| n_games_per_train | 40 | Self-play games per iteration |
| n_iterations | 30 | Training iterations |
| n_frozen_states | 1 | Evaluation states (of C(6,2)=15 possible) |
