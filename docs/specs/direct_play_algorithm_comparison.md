# Direct Play Algorithm Comparison for the Doors Domain

> **Purpose**: Self-contained specification for comparing RL algorithms that play the Doors environment directly (no program synthesis). Includes full environment specification, algorithm designs with hyperparameters, scaling analysis across D=2..20+, and experiment design. Intended to be fed into an LLM for experiment roadmap generation.

---

## 1. The Doors Environment

### Overview

A grid-world with D rooms connected by locked doors. The agent must navigate from Room 0 to the goal location in Room D-1, collecting keys along the way to unlock doors.

```
Room 0 (unlocked)     Room 1 (locked)       Room 2 (locked)
+---------------+    +---------------+     +---------------+
| loc_0   loc_1 | -- | loc_2   loc_3 | --- | loc_4   loc_5 |
|         KEY_0 |    |         KEY_1 |     |         GOAL  |
+---------------+    +---------------+     +---------------+
  Key 0 unlocks Room 1    Key 1 unlocks Room 2
```

### State Space

Float32 vector of size `n = M + 2D - 1` where `M = D * locs_per_room`:

| Indices | Meaning | Size | Encoding |
|---------|---------|------|----------|
| `[0, M)` | Agent location | M | One-hot |
| `[M, M+D)` | Room unlock status | D | Binary (room 0 always 1) |
| `[M+D, M+2D-1)` | Key availability | D-1 | Binary (1 = available) |

**Example D=3, locs_per_room=2**: `M=6`, `n=11`
- `[0:6]` = agent at one of 6 locations (one-hot)
- `[6:9]` = rooms 0,1,2 unlocked? (room 0 always 1)
- `[9:11]` = keys 0,1 available?

### Action Space

Discrete, size = `M + K + 1` real actions (padded to `n` for alignment):

| Action indices | Meaning | Count |
|---------------|---------|-------|
| `[0, M)` | MOVE_TO(location) | M |
| `[M, M+K)` | PICK(key) | K = D-1 |
| `M+K` | NOOP | 1 |
| `[M+K+1, n)` | Invalid (mapped to NOOP) | padding |

**Preconditions**:
- MOVE_TO(l): succeeds only if `unlocked[room_of(l)] == 1`
- PICK(k): succeeds only if `at_loc[key_loc[k]] == 1 AND key_available[k] == 1`
- Failed actions: state unchanged, step penalty still applies

### Reward Structure

Per-step rewards (dense):
- `-0.01`: step penalty (every step)
- `+0.1`: unlock bonus (when PICK succeeds and unlocks a new room)
- `+1.0`: goal bonus (reaching goal location, terminates episode)

**Optimal return for D rooms**: `1.0 + (D-1) * 0.1 - optimal_steps * 0.01`
- Optimal steps = `2*(D-1) + 1` (pick each key + move to each next key/goal)

| D | Optimal Steps | Optimal Return |
|---|--------------|----------------|
| 2 | 3 | 1.07 |
| 3 | 5 | 1.15 |
| 5 | 9 | 1.31 |
| 10 | 19 | 1.71 |
| 20 | 39 | 2.51 |

### Horizon (max episode length)

`horizon = max(15, 5 * optimal_steps)`

| D | Optimal Steps | Horizon |
|---|--------------|---------|
| 2 | 3 | 15 |
| 3 | 5 | 25 |
| 5 | 9 | 45 |
| 10 | 19 | 95 |
| 20 | 39 | 195 |

### Environment Layout

- Locations numbered `0..M-1`, grouped by room: `loc_room[l] = l // locs_per_room`
- Key `k` placed at location `k * locs_per_room + 1` (2nd location of room k)
- Key `k` unlocks room `k+1`
- Agent starts at location 0 (room 0)
- Goal at location `M-1` (last location of last room)

### Scaling Properties

| D | State dim | Actions (real) | Optimal steps | Search complexity |
|---|-----------|---------------|---------------|-------------------|
| 2 | 7 | 5 | 3 | Trivial |
| 3 | 11 | 8 | 5 | Easy |
| 5 | 19 | 14 | 9 | Moderate |
| 10 | 39 | 29 | 19 | Requires credit assignment |
| 20 | 79 | 59 | 39 | Long-horizon sequential |
| 50 | 199 | 149 | 99 | Very hard for most RL |

**Key scaling challenge**: The task has a strict sequential dependency chain. To reach room k, the agent must have picked key k-1, which requires being in room k-1, which requires key k-2, etc. This creates a **hard exploration problem** at large D: random exploration rarely stumbles into the correct sequence.

### Optimal Policy (D=3 example)

```
if at key_0 location AND key_0 available:  PICK(key_0)
elif at key_1 location AND key_1 available: PICK(key_1)
elif room_1 locked:                         MOVE_TO(key_0 location)
elif room_2 locked:                         MOVE_TO(key_1 location)
else:                                       MOVE_TO(goal location)
```

General pattern for D rooms: pick any available key you're standing on, otherwise move toward the first locked room's key (or goal if all unlocked). Requires `2*(D-1) + 1` steps.

---

## 2. Algorithms

### 2.1 Tabular Q-Learning

**What it learns**: Q-table `Q[s][a]` mapping discrete states to action values.

**How it works**: Temporal difference update `Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`. Epsilon-greedy exploration.

**State discretization**: The Doors state is already binary, so each unique observation maps to a unique table entry. Number of reachable states << 2^n because of constraints (one-hot location, monotonic key depletion).

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Alpha | 0.1 | Standard tabular learning rate |
| Gamma | 0.99 | Short episodes, mild discounting |
| Epsilon | 1.0 -> 0.01 linear over 10K episodes | Explore then exploit |
| Episodes | 50K | Conservative upper bound |

**Strengths**: Exact solution for small state spaces. No function approximation error. Guaranteed convergence.

**Weaknesses**: State table grows exponentially. Reachable states for D=10 is O(thousands) (manageable), but D=20 has O(millions+) and becomes slow. D=50 is infeasible.

**Scaling limit**: ~D=10-15 (depending on locs_per_room).

---

### 2.2 DQN (Deep Q-Network)

**What it learns**: Neural network `Q(s,a; theta)` approximating the optimal action-value function.

**How it works**: Off-policy learning with experience replay. Samples mini-batches from replay buffer, minimizes TD error. Target network stabilizes training.

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Network | MLP [n, 128, 128, n_actions] | 2-layer MLP, sufficient for small/medium D |
| Learning rate | 1e-3 | Standard for DQN |
| Replay buffer | 50K transitions | Small env, short episodes |
| Batch size | 64 | Standard |
| Gamma | 0.99 | |
| Epsilon start | 1.0 | Full exploration initially |
| Epsilon end | 0.05 | Retain some exploration |
| Epsilon decay | Linear over 10K steps | |
| Target net update | Hard copy every 500 steps | Stability |
| Double DQN | Yes | Reduces max overestimation |
| Dueling architecture | Optional | Separates V(s) from A(s,a) |
| Train frequency | Every 4 steps | Standard |
| Learning starts | 1000 steps | Fill replay buffer first |
| Total timesteps | 100K (D<=5), 500K (D=10), 2M (D=20) | Scale with difficulty |

**Action masking**: Apply structural mask (real actions only, no padding). Optionally apply precondition mask (only valid moves) for fair comparison with AlphaZero.

**Strengths**: Off-policy (sample efficient via replay). Scales to any state dimension. Well-understood, easy to implement (stable-baselines3).

**Weaknesses**: No lookahead search. Epsilon-greedy exploration may be slow for long sequential dependencies. Value function must generalize across states without search.

**Expected scaling**: Solves D=2,3,5 easily. D=10 likely works with tuning. D=20+ may need exploration aids (curiosity, HER). The sequential key-collection chain means random exploration rarely reaches later rooms at large D.

---

### 2.3 PPO (Proximal Policy Optimization)

**What it learns**: Policy `pi(a|s; theta)` and value function `V(s; phi)` jointly.

**How it works**: On-policy. Collects rollouts, computes GAE advantages, updates policy with clipped surrogate objective. Discards data after update.

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Network | MLP [n, 128, 128], shared trunk, separate heads | Match DQN capacity |
| Learning rate | 3e-4 | Standard PPO |
| Gamma | 0.99 | |
| GAE lambda | 0.95 | Bias-variance tradeoff |
| Clip range | 0.2 | Standard |
| Entropy coefficient | 0.01 (D<=5), 0.05 (D>=10) | More exploration for harder tasks |
| VF coefficient | 0.5 | Standard |
| N_steps | 256 | Rollout length before update |
| N_epochs | 4 | PPO update epochs per batch |
| Batch size | 64 | Mini-batch for PPO update |
| N_envs | 8 | Parallel environments |
| Total timesteps | 200K (D<=5), 1M (D=10), 5M (D=20) | On-policy needs more data |

**Action masking**: Mask logits for invalid actions before softmax (`MaskablePPO` from sb3-contrib).

**Strengths**: Stable training. Entropy bonus provides structured exploration. Policy gradient directly optimizes expected return. Scales well with parallel envs.

**Weaknesses**: On-policy = sample inefficient (discards experience). Needs more total timesteps than DQN. Credit assignment over long horizons relies on GAE, which can be slow to propagate.

**Expected scaling**: Solves D=2,3,5 easily. D=10 works but needs more steps. D=20+ likely needs increased entropy, larger N_steps, or curriculum. Sequential dependency is the bottleneck: GAE must propagate reward signal backward through 39+ steps.

---

### 2.4 AlphaZero (Current Implementation)

**What it learns**: Policy `pi(a|s)` and value `V(s)` network, trained from MCTS-generated targets.

**How it works**: At each step, runs MCTS (120 simulations) using the current network for leaf evaluation. Action selected proportional to MCTS visit counts. Network trained on (state, MCTS_policy, discounted_return) tuples from self-play.

| Parameter | Current Value | Purpose |
|-----------|--------------|---------|
| Network | MLP, hidden=max(64, 4*n) | Policy + value heads |
| MCTS simulations | 120 per step | Search depth/breadth |
| Temperature | 1.0 | Visit count -> probability |
| c_exploration | 1.5 | UCB exploration weight |
| Dirichlet alpha | 0.25 | Root noise concentration |
| Dirichlet epsilon | 0.40 | Noise mixing weight |
| Reward discount | 1.0 | No discounting |
| Games/iteration | 50 | Self-play batch |
| Training window | 20 iterations | Replay from recent history |
| Total iterations | 50 | Training loop |
| Workers | 8 | Parallel self-play |
| Learning rate | 3e-4 | Adam optimizer |
| Epochs per iter | 10 | Network training |
| Batch size | 32 | Training batch |
| Policy weight | 2.0 | loss = MSE(value) + 2*CE(policy) |

**Strengths**: MCTS provides structured lookahead search at every decision. Network learns from high-quality search-refined targets (not raw returns). Explores systematically within each episode via UCB.

**Weaknesses**: Extremely expensive per episode (120 forward passes per step * ~5-20 steps). The search is overkill for small D where a simple policy suffices. For large D, the branching factor (29+ actions at D=10) dilutes the 120 simulations. Self-play framework designed for two-player games; single-player adaptation is less natural.

**Expected scaling**: Solves D=2,3,5 reliably but slowly. D=10 with 120 sims may be thin (120 sims / 29 actions = ~4 visits per action on average). D=20+ needs significantly more simulations or a very strong prior policy.

---

### 2.5 SAC (Soft Actor-Critic)

**What it learns**: Policy `pi(a|s)`, two Q-networks `Q1(s,a)`, `Q2(s,a)`, and entropy temperature alpha.

**How it works**: Off-policy, maximum entropy RL. Learns a stochastic policy that maximizes both return and entropy. Uses twin Q-networks to reduce overestimation. Auto-tunes entropy coefficient.

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Network | MLP [n, 256, 256] per network | Slightly larger for SAC |
| Learning rate | 3e-4 | Standard |
| Replay buffer | 100K | Off-policy |
| Batch size | 256 | SAC benefits from larger batches |
| Gamma | 0.99 | |
| Tau (soft update) | 0.005 | Polyak averaging |
| Auto entropy | Yes | Learn alpha automatically |
| Target entropy | -n_actions (default) | Encourages exploration |
| Total timesteps | 100K (D<=5), 500K (D=10) | Similar to DQN |

**Note**: SAC is designed for continuous action spaces. For discrete Doors, use SAC-Discrete variant (available in cleanrl or custom implementation).

**Strengths**: Maximum entropy encourages diverse exploration (important for sequential key collection). Off-policy sample efficiency. Twin Q reduces overestimation.

**Weaknesses**: More complex than DQN. Discrete SAC variants less mature in standard libraries. May be overkill for this domain.

**Expected scaling**: Similar to DQN, potentially better exploration at large D due to entropy maximization.

---

## 3. Systematic Comparison

### 3.1 Properties Table

| Property | Tabular Q | DQN | PPO | AlphaZero | SAC |
|----------|-----------|-----|-----|-----------|-----|
| **Learning type** | Value (tabular) | Value (neural) | Policy+Value | Policy+Value+Search | Policy+Value (max-ent) |
| **On/Off policy** | Off | Off | On | On (self-play) | Off |
| **Replay buffer** | N/A (tabular) | Yes | No | Yes (recent iters) | Yes |
| **Search at test time** | No | No | No | Yes (MCTS) | No |
| **Compute per episode** | Negligible | Low | Low | Very High (120 sims/step) | Low |
| **Sample efficiency** | High (small D) | Medium-High | Low | High per-episode | Medium-High |
| **Wall-clock efficiency** | Best (small D) | Fast | Fast | Slow (MCTS) | Fast |
| **Exploration strategy** | Epsilon-greedy | Epsilon-greedy | Entropy bonus | MCTS + Dirichlet | Max-entropy |
| **Credit assignment** | TD(0) bootstrap | TD(0) bootstrap | GAE (multi-step) | MCTS returns | TD(0) twin-Q |
| **Handles action masking** | Trivial | Simple | Via logit masking | Built-in | Via logit masking |
| **Max D (expected)** | ~10-15 | ~20-30 | ~15-25 | ~10-15 (sim-limited) | ~20-30 |
| **Implementation** | Custom (~50 lines) | stable-baselines3 | stable-baselines3 | Already implemented | cleanrl or custom |

### 3.2 Scaling Bottlenecks Per Algorithm

| D range | Primary bottleneck | Hardest for | Easiest for |
|---------|-------------------|-------------|-------------|
| 2-5 | None (all solve) | AlphaZero (overkill overhead) | All others |
| 5-10 | Exploration (finding key chain) | PPO (on-policy), Tabular Q (table size) | DQN, SAC |
| 10-20 | Long credit assignment chain | PPO (GAE over 39 steps), AlphaZero (sim budget) | DQN + HER, SAC |
| 20-50 | Deep sequential exploration | All struggle without curriculum | None without augmentation |
| 50+ | Requires structured exploration | All standard algorithms | Curriculum / HER needed |

### 3.3 Key Differentiators for Doors Specifically

**Dense reward helps everyone**: Unlike the synthesis problem, direct Doors play gives step-level rewards (+0.1 per unlock, -0.01 per step). This is a massive advantage -- all algorithms get gradient signal from partial progress.

**Sequential dependency is the real challenge**: The key chain (key0 -> room1 -> key1 -> room2 -> ...) creates a strict ordering. At large D, an agent must discover this entire chain through exploration. This favors:
- Algorithms with good exploration (SAC's max-entropy, PPO's entropy bonus)
- Algorithms with efficient credit assignment (DQN with n-step returns)
- Curriculum learning (start with D=2, transfer to D=3, etc.)

**AlphaZero's MCTS is disproportionately expensive here**: For a 5-step optimal solution (D=3), spending 120 simulations per step is ~600 forward passes per episode. DQN/PPO use 5 forward passes. AlphaZero must provide 120x better signal per episode to justify the cost.

---

## 4. Experiment Design

### 4.1 Independent Variables

- **Algorithm**: Tabular Q, DQN, PPO, AlphaZero, SAC
- **D** (rooms): 2, 3, 5, 10, 20 (and 50 if any algorithm reaches D=20)
- **Action masking**: structural only vs. full precondition mask
- **locs_per_room**: 2 (default), optionally 3 for increased state space

### 4.2 Dependent Variables (Metrics)

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Episodes to solve** | First episode achieving >= 95% of optimal return | Sample efficiency |
| **Wall-clock to solve** | Time to first solve | Practical efficiency |
| **Final solve rate** | % of eval episodes reaching goal (after training) | Reliability |
| **Average return** | Mean eval return over last 100 episodes | Quality |
| **Learning curve** | Return vs episodes (plot) | Convergence behavior |
| **Compute cost** | Total forward passes / total gradient steps | Normalized cost |

### 4.3 Experiment Matrix

#### Phase 1: Sanity Check (D=2,3)
All algorithms on D=2 and D=3. Expect all to solve. Compare wall-clock and sample efficiency. Validates implementations.

#### Phase 2: Medium Scale (D=5,10)
All algorithms on D=5, D=10. Identify which algorithms start to struggle. Measure degradation curves.

#### Phase 3: Hard Scale (D=20)
Surviving algorithms on D=20. Add exploration augmentations:
- DQN + Hindsight Experience Replay (HER)
- PPO + increased entropy (0.05-0.1)
- AlphaZero + more simulations (500+)
- Curriculum: pre-train on D=10, fine-tune on D=20

#### Phase 4: Stress Test (D=50, optional)
Only if any algorithm solves D=20 reliably. Likely requires curriculum.

### 4.4 Expected Outcome Summary

| D | Winner (sample eff.) | Winner (wall-clock) | Expected failures |
|---|---------------------|--------------------|--------------------|
| 2 | Tabular Q | Tabular Q | None |
| 3 | Tabular Q / DQN | Tabular Q / DQN | None |
| 5 | DQN / SAC | DQN / PPO | None |
| 10 | DQN / SAC | DQN | Tabular Q (table size), AlphaZero (slow) |
| 20 | DQN+HER / SAC | DQN+HER | PPO (credit assignment), AlphaZero (sim budget), Tabular Q |
| 50 | Curriculum only | Curriculum only | Most without curriculum |

### 4.5 Implementation Notes

**stable-baselines3**: Use for DQN, PPO. Add `sb3-contrib` for `MaskablePPO`.
**AlphaZero**: Use existing `DoorsDirectConfig` with `run_doors_direct.py`.
**Tabular Q**: Write from scratch (~50 lines), use `DoorsPDDLLiteEnv` directly.
**SAC-Discrete**: Use cleanrl implementation or adapt sb3 SAC.

**Fair comparison**: All algorithms use the same `DoorsPDDLLiteEnv` with same D, locs_per_room, rewards. Log metrics in common format for plotting.

**Gymnasium wrapper**: `DoorsPDDLLiteEnv` already extends `gym.Env`. For sb3, may need to wrap in `gymnasium.wrappers.TimeLimit` (if horizon isn't handled internally -- it is, via `self.horizon`). Action space padding (obs_size > real actions) should be handled via action masking.

---

## 5. Key Questions This Comparison Answers

1. **Is MCTS search worth its cost for direct play?** Compare AlphaZero's solve rate and sample efficiency against DQN/PPO at equal wall-clock time.

2. **Which algorithm scales best with D?** Plot episodes-to-solve vs D for each algorithm. Identify the crossover point where simple algorithms fail.

3. **Where does exploration become the bottleneck?** At what D does epsilon-greedy (DQN) fail while entropy-based (SAC/PPO) still works?

4. **How much does action masking help?** Compare structural-only vs precondition masking. Precondition masking eliminates impossible actions -- this should help all algorithms but especially exploration-limited ones.

5. **Does curriculum learning change the picture?** If no algorithm solves D=20 from scratch, does pre-training on D=10 help? This tests transfer/generalization.

6. **What is the gap between direct RL and program synthesis?** The synthesis approach (AlphaZero on derivation game) fails at D=3 with 688K programs explored. If DQN solves D=3 in 1000 episodes, that quantifies the cost of the synthesis framing and motivates either fixing synthesis or using direct RL + distillation.
