# LLM Prompt: Doors Game — Direct Play vs Grammar Game Experiment Plan

You are a research advisor and systems engineer. Your task is to analyze a codebase audit, produce theoretical predictions, critique the audit for gaps, and output phased implementation plans.

## Your Inputs

1. This prompt (instructions + constraints)
2. The complete system audit: `specs/doors_game_audit.md` (attached below or provided separately)

## Core Research Question

> Given the same computational budget (MCTS simulations x training iterations x games per iteration), which approach solves the D=2 Doors environment faster — and what are the structural tradeoffs?
>
> - **Approach A (Direct Play):** AlphaZero agent observes the 7-dim environment state and selects from 7 env actions directly.
> - **Approach B (Grammar Game):** AlphaZero agent constructs a DSL program (decision list) by selecting grammar productions (~150 actions on a 36-dim AST state). The completed program is then executed as a reactive policy on the environment.

---

## Required Output

Produce exactly three sections in your response:

---

### OUTPUT A: Theoretical Breakdown

Analyze each approach and make concrete, falsifiable predictions. Ground every claim in numbers from the audit.

#### A.1 — MDP Complexity Analysis

For **Direct Play**:
- Enumerate the reachable state space. The state vector is 7-dim binary with constraints (at_loc is one-hot over 4 positions, unlocked[0] is always 1, key transitions are irreversible). Count the exact number of reachable states.
- Analyze the effective branching factor. 7 discrete actions, but many are no-ops due to precondition failures. What is the average effective branching factor across the reachable states?
- Episode length: max 15 steps, optimal 3 steps. What is the expected episode length under random play?

For **Grammar Game**:
- The action space is ~150 grammar productions, but `get_action_mask()` constrains this per state. What is the typical effective branching factor at each derivation step?
- A complete program is ~10-15 derivation steps. How many distinct programs can be synthesized within budget=18?
- The reward is sparse: 0.0 at every non-terminal step, then `leaf_evaluator(program)` at terminal. What fraction of random derivations produce a program that solves the environment?

#### A.2 — Reward Signal Density

Quantify the reward signal each approach receives per training iteration:

- **Direct Play**: 40 games x up to 15 steps/game = up to 600 reward signals per iteration. Rewards are shaped: -0.01/step, +0.1 unlock, +1.0 goal. The value function gets dense gradient signal.
- **Grammar Game**: 40 games x 1 terminal reward/game = 40 reward signals per iteration. Intermediate derivation steps get reward=0. The value function must learn from sparse, delayed signal.

What is the ratio of reward density? What does this predict about convergence speed?

#### A.3 — Compute Cost Per MCTS Simulation

- **Direct Play**: Each MCTS simulation steps the env (O(1) per step). Cost per simulation ≈ O(remaining_episode_length).
- **Grammar Game**: Each MCTS simulation that reaches a terminal derivation calls `leaf_evaluator(program)`, which executes `run_policy_episode` on the doors env (~15 env steps). Simulations that don't reach terminal are cheaper (just AST manipulation). Estimate: what fraction of MCTS simulations reach a complete program?

#### A.4 — Convergence Predictions

Make specific predictions:
- **Direct Play**: How many training iterations (out of 30) to reach solve_rate=1.0? Justify.
- **Grammar Game**: How many training iterations to synthesize the optimal 16-node program? What about a suboptimal but solving program?
- **Which converges first?** State your prediction and the key factor driving it.

#### A.5 — Where Grammar Game Wins

Analyze the structural advantages of the grammar game that don't show up in D=2 convergence speed:
- Interpretability: the output is a human-readable decision list
- Verifiability: the program can be formally checked against all reachable states
- Generalization: a synthesized program works for ANY initial state, not just the training distribution
- Scaling to D=3+: how does each approach's difficulty scale with num_rooms?

#### A.6 — Null Hypothesis

State a clear null hypothesis for the D=2 comparison, and what experimental result would reject it.

---

### OUTPUT B: Systematic TODO List

First, **critique the audit document** (`specs/doors_game_audit.md`). Identify:
- Gaps in the specification (missing details, ambiguous claims)
- Missing experimental controls
- Implicit assumptions that should be made explicit
- Anything that would block implementation

Then organize all required work into phased TODOs. Each TODO item must be:
- Actionable (a developer can execute it)
- Verifiable (there's a concrete check for "done")
- Dependency-ordered (phases build on each other)

#### Phase 0: Baselines (establish floor and ceiling)

Establish reference points before running any training:
- **Random agent**: solve rate and avg reward over 10,000 episodes
- **Optimal hardcoded agent**: manually code the 3-step solution, verify reward ≈ 1.07
- **Optimal DSL program**: manually construct the 16-node program from audit Section 2.3, run via `run_policy_episode`, verify solve_rate=1.0

#### Phase 1: Implement Direct Play Infrastructure (~120 lines total)

Build the missing components. For each file, state:
- Exact file path
- Which existing class to subclass or reuse
- What the implementation does

Components needed:
1. `DoorsDirectGame(EnvGame)` — wraps `DoorsPDDLLiteEnv`
2. `DoorsDirectNet(TorchPolicyValueNet)` — MLP using existing `PolicyValueNetModel`
3. `DoorsDirectConfig(MetaConfig)` — wires everything, matched MCTS params

#### Phase 2: Run Grammar Game Experiment

Already runnable. Execute and capture metrics:
```
python scripts/run_derivation.py  # select mode 2 (doors)
```
Log: per-iteration solve_rate, avg_reward, gate_score, accepted, wall-clock time, best program at each iteration.

#### Phase 3: Run Direct Play Experiment

Execute with matched hyperparameters. Same logging format as Phase 2.

#### Phase 4: Comparison Analysis

Side-by-side comparison:
- Convergence curves (solve_rate vs iteration, avg_reward vs iteration)
- Total env steps consumed
- Wall-clock time
- Final solve_rate and avg_reward
- For grammar game: best synthesized program (human-readable)

#### Phase 5: Ablations (optional, for deeper analysis)

- MCTS simulation budget sweep: 50, 100, 200, 400
- Scale to D=3 (6 locations, 2 keys, obs_size=11)
- Grammar ablation: And=True vs And=False
- Direct play: with/without action masking (precondition-aware mask vs all-True)

---

### OUTPUT C: Claude Code Plans

For each phase in the TODO list, produce a **Claude Code implementation plan** — a block that Claude Code can execute. Each plan must contain:

1. **Goal**: One sentence stating what this phase accomplishes
2. **Files to create/modify**: Exact paths
3. **Key imports & reusable components**: What existing code to use (with file paths from audit Section 10)
4. **Implementation sketch**: Pseudocode or key code fragments (not full implementations — enough for Claude Code to fill in the details)
5. **Verification command**: A shell command or test to confirm the phase works

Format each plan as:

```
## Claude Code Plan: Phase N — <Title>

**Goal:** <one sentence>

**Files:**
- CREATE: <path> — <description>
- MODIFY: <path> — <description>

**Reuse:**
- <ClassName> from <file_path> — <why>

**Implementation sketch:**
<pseudocode or key fragments>

**Verify:**
<shell command>
```

---

## Constraints

### Reuse existing infrastructure — do NOT reinvent

The codebase already provides all the building blocks. The prompt recipient must use:

| Component | Class | File |
|-----------|-------|------|
| Game wrapper | `EnvGame` | `src/alphazeropp/core/game.py` |
| Network base | `TorchPolicyValueNet` | `src/alphazeropp/core/policy_value_net.py` |
| Generic MLP | `PolicyValueNetModel` | `src/alphazeropp/core/policy_value_net.py` |
| Config base | `MetaConfig` | `src/alphazeropp/core/config.py` |
| Agent | `Agent` | `src/alphazeropp/core/agent.py` |
| MCTS | `MCTS` | `src/alphazeropp/core/mcts.py` |
| Trainer | `Trainer` | `src/alphazeropp/training/trainer.py` |
| Gated trainer | `GatedTrainer` | `src/alphazeropp/training/gated_trainer.py` |
| Evaluator | `Evaluator` | `src/alphazeropp/training/evaluator.py` |

### Existing templates to follow

- **DoorsDirectGame** should follow the pattern of `CartPoleGame` at `src/alphazeropp/instances/cartpole/game.py`
- **DoorsDirectNet** should follow the pattern of `CartPolePolicyValueNet` at `src/alphazeropp/instances/cartpole/network.py`
- **DoorsDirectConfig** should follow the pattern of `DoorsDerivationConfig` at `src/alphazeropp/instances/bitstring/dsl/derivation_config.py`

### Fairness controls (apples-to-apples)

Both arms must share these hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MCTS simulations/move | 200 | Match grammar game default |
| c_exploration | 1.5 | Match grammar game default |
| dirichlet_alpha | 0.25 | Match grammar game default |
| dirichlet_epsilon | 0.40 | Match grammar game default |
| Training iterations | 30 | Match grammar game default |
| Games per iteration | 40 | Match grammar game default |
| Evaluation games | 20 | Match grammar game default |
| Gate threshold | 0.55 | Match grammar game default |
| n_procs | -1 (sequential) | Reproducible timing |
| reward_discount | 1.0 | Match grammar game default |

Things that inherently differ (document but don't try to equalize):

| Parameter | Direct Play | Grammar Game |
|-----------|-------------|--------------|
| Action space | 7 | ~150 |
| Observation dim | 7 (binary) | 36 (float, AST encoding) |
| Episode length | up to 15 env steps | ~10-15 derivation steps |
| Reward structure | shaped (-0.01/step, +0.1 unlock, +1.0 goal) | sparse (0 until terminal, then leaf_eval) |
| Network | MLP (7→64→64→7+1) | Transformer (36→d64, 4 heads, 2 layers→150+1) |

---

## Audit Document Reference

The complete system audit is in `specs/doors_game_audit.md`. It contains:

1. **Section 1** — Doors environment: layout, state vector, actions, rewards, terminal conditions, optimal play, class interface
2. **Section 2** — DSL: AST nodes, interpreter, optimal program, DSL-env mapping
3. **Section 3** — AlphaZero pipeline: Game, PolicyValueNet, MCTS, Agent, Trainer, Evaluator, GatedTrainer, full training loop
4. **Section 4** — Grammar game: DerivationGame, LeafEvaluator, Transformer network, UniformPolicyValueNet
5. **Section 5** — Configs: DoorsGameConfig, DoorsDerivationConfig
6. **Section 6** — Existing scripts: run_derivation.py, estimate_expressivity_gap.py
7. **Section 7** — What's missing for direct play: DoorsDirectGame, DoorsDirectNet, DoorsDirectConfig
8. **Section 8** — Experiment proposals (5 experiments)
9. **Section 9** — Comparison framework: metrics table, fairness controls, key question
10. **Section 10** — File inventory (full tree)

Use this document as your sole source of truth. All code references, class names, and file paths come from it.
