# AlphaZero Component Ablation Plan: Isolating Network vs MCTS Contribution

*Written: 2026-03-09*

## 1. Problem Statement

Existing D=10 direct-play experiments show a suspicious pattern:

| Experiment | MCTS Sims | Solve Rate | When |
|-----------|-----------|------------|------|
| `20260303_230143` | 100 | 100% | By iteration 4 |
| `20260306_173721` (4 seeds) | 120 | 100% | By iteration 4–7 |
| `20260303_214517` | 20 | 0% | Never |

**After solving**: The network keeps learning — policy loss drops from 3.65 → 0.007 over 50 iterations — but performance never improves beyond the iteration 4–7 plateau.

**Core question**: Is the neural network doing real work, or is MCTS the sole solver at this scale?

**Three sub-questions**:
1. Can MCTS solve D=10 without any learned network? (Pure search baseline)
2. Can the trained network solve D=10 without MCTS? (Pure policy baseline)
3. Which network head matters more — policy prior or value estimate?

---

## 2. How AlphaZero Uses the Network (Code Trace)

Understanding the exact code paths is essential for designing clean ablations.

### 2.1 MCTS Network Query

In `src/alphazeropp/core/mcts.py`, the network is queried exactly once per new node expansion:

```
query_net_masked() (line ~382)
  → self.net.predict(self.game.obs)
  → returns (policy_prob, value)       # numpy arrays
  → applies action_mask to policy
  → stores in node.nn_policy, node.nn_value
```

**Policy prior usage** (UCB formula, line ~424):
```python
u = c_exploration * policy * sqrt(total_N + EPS) / (1 + n_arr)
ucbs = q_norm + u
action = argmax(ucbs)
```
- `policy` = `node.nn_policy` (the network's masked/normalized output)
- If policy is uniform → all actions get equal exploration bonus → search becomes visit-count-driven

**Value usage** (leaf evaluation, line ~335):
```python
leaf_value = myvalue  # from net.predict()
if self.rollout_n > 0:
    rv = _rollout_value(...)
    leaf_value = (1 - rollout_blend) * rv + rollout_blend * myvalue
```
- `myvalue` = network's value prediction for the leaf node
- This value is backpropagated up the tree to update Q-values
- If value is always 0.0 → Q-values only accumulate from direct rewards during rollouts (which are 0 in direct play since there are no rollouts by default)

### 2.2 Agent → MCTS Flow

In `src/alphazeropp/core/agent.py`:

```python
def policy(self, state, msg, add_noise, temperature_override):
    current_game_state = state.clone()
    mcts = MCTS(current_game_state, self.net, **self.mcts_params)  # line 98
    move_probs = mcts.perform_simulations("", add_noise=add_noise) # line 101
    return move_probs
```

- A **fresh MCTS tree** is created for every move decision
- `self.net` is passed to MCTS — this is where we inject wrappers
- `n_simulations < 0` → skips search, returns raw network policy (line 107–109 in mcts.py)
- `n_simulations = 0` → runs the expansion loop 0 times, so only root expansion happens (1 net query, no search)

### 2.3 Training Loop

In `src/alphazeropp/training/trainer.py`:

```python
def _train_network(self, flat_examples, avg_reward):
    _, batch_losses, train_losses, policy_losses, value_losses = self.net.train(flat_examples)
    # Records statistics...
```

- `self.net.train(flat_examples)` → calls `DoorsDirectNet.train()` in `network.py`
- Training targets come from MCTS self-play: `(obs, mcts_policy, discounted_return)`
- Returns 5-tuple: `(model, batch_losses, train_losses, policy_losses, value_losses)`

### 2.4 Network Architecture

In `src/alphazeropp/instances/doors/network.py`:

```
Input (obs_size=49 for D=10,L=3)
  → Linear(49, 196) + ReLU
  → Linear(196, 196) + ReLU
  → Policy Head: Linear(196, 49) → Softmax  (outputs action probabilities)
  → Value Head:  Linear(196, 1)             (outputs scalar value estimate)
```

- `predict(state)` returns `(policy_prob, value)` — both numpy
- `train(examples)` returns `(model, batch_losses, train_losses, policy_losses, value_losses)`

---

## 3. Code Changes

### 3.1 New File: `src/alphazeropp/instances/doors/network_ablations.py`

Three wrapper classes that intercept `predict()` and/or `train()` calls:

```python
"""Network wrappers for AlphaZero ablation experiments."""
import numpy as np


class UniformPolicyWrapper:
    """Replaces policy output with uniform distribution, keeps value.

    Tests: What does the value head alone contribute?
    MCTS gets no directional guidance from the policy prior.
    """
    def __init__(self, net):
        self._net = net

    def predict(self, state):
        policy, value = self._net.predict(state)
        uniform = np.ones_like(policy) / len(policy)
        return uniform, value

    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._net, name)


class ZeroValueWrapper:
    """Replaces value output with 0.0, keeps policy.

    Tests: What does the policy prior alone contribute?
    MCTS leaf nodes get no value signal.
    """
    def __init__(self, net):
        self._net = net

    def predict(self, state):
        policy, _ = self._net.predict(state)
        return policy, np.float32(0.0)

    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._net, name)


class FrozenWrapper:
    """Makes train() a no-op. Network stays at random initialization.

    Tests: Can MCTS solve the problem with a random (untrained) network?
    """
    def __init__(self, net):
        self._net = net

    def predict(self, state):
        return self._net.predict(state)

    def train(self, *args, **kwargs):
        # No-op: return dummy values matching expected 5-tuple signature
        return self._net.model, [0.0], [0.0], [0.0], [0.0]

    def __getattr__(self, name):
        return getattr(self._net, name)
```

**Why `__getattr__` delegation works**: The `Trainer` and `MultiprocessingManager` access attributes like `net.model`, `net.push_multiprocessing()`, `net.pop_multiprocessing()`, `net.save_checkpoint()`. The `__getattr__` fallback delegates all attribute lookups (except `predict`, `train`, `_net`) to the wrapped network, so these all work transparently.

**Why not subclass**: The wrappers need to intercept calls on an *existing* net instance. Subclassing would require changing the construction flow. Delegation is simpler and non-invasive.

### 3.2 Modifications to `scripts/run_doors_direct.py`

#### 3.2.1 New CLI Arguments

Add to `parse_args()` (currently at line 1219):

```python
parser.add_argument("--ablation", type=str, default=None,
                    choices=["frozen", "uniform-policy", "zero-value"],
                    help="Ablation mode: frozen=no training, "
                         "uniform-policy=uniform prior, zero-value=value=0")
parser.add_argument("--no-mcts", action="store_true",
                    help="Use raw network policy (n_simulations=-1), no MCTS search")
parser.add_argument("--log-analysis", action="store_true",
                    help="Log network prediction quality metrics each iteration")
```

#### 3.2.2 Apply Wrapper After `cfg.build()`

In `_run_single_seed()` (line ~1093), after `game, net, agent, trainer, evaluator = cfg.build()`:

```python
# Apply ablation wrapper if requested
if ablation == "frozen":
    from alphazeropp.instances.doors.network_ablations import FrozenWrapper
    net = FrozenWrapper(net)
elif ablation == "uniform-policy":
    from alphazeropp.instances.doors.network_ablations import UniformPolicyWrapper
    net = UniformPolicyWrapper(net)
elif ablation == "zero-value":
    from alphazeropp.instances.doors.network_ablations import ZeroValueWrapper
    net = ZeroValueWrapper(net)

# Replace net reference in both agent and trainer so they use the wrapper
agent.net = net
trainer.net = net
```

**Why replace in both**: The agent uses `self.net` to pass to MCTS (line 98 in agent.py), and the trainer uses `self.net.train()` (line 156 in trainer.py). Both must point to the same wrapper.

#### 3.2.3 Handle `--no-mcts` Flag

```python
if args.no_mcts:
    cfg.agent.mcts_params["n_simulations"] = -1
```

This triggers the existing code path in `mcts.py:107-109` that returns raw network policy without any tree search.

#### 3.2.4 Pass `ablation` Argument Through

`_run_single_seed()` currently takes `(cfg, exp_dir)`. Add `ablation=None` parameter. Thread it from `main()`.

#### 3.2.5 Update Experiment Directory Naming

In `setup_experiment_dir()`, append the ablation mode to the directory name so results don't mix:

```python
if ablation:
    dirname += f"_{ablation}"
if no_mcts:
    dirname += "_nomcts"
```

### 3.3 Analysis Logging (for `--log-analysis` flag)

Add a new function `compute_analysis_metrics()` to `run_doors_direct.py`:

```python
def compute_analysis_metrics(agent, n_episodes=5):
    """Compute network-vs-MCTS diagnostic metrics.

    Returns dict with:
      - policy_entropy: avg entropy of raw network policy (before MCTS)
      - mcts_kl: avg KL(network_policy || mcts_policy)
      - value_error: avg |v_net(s) - actual_return|
      - action_agreement: fraction of steps where argmax(net) == argmax(mcts)
    """
    entropies = []
    kl_divs = []
    value_errors = []
    agreements = []

    for i in range(n_episodes):
        game = agent.game.clone()
        game.reset_wrapper(seed=2000 + i)
        episode_return = 0.0
        net_values = []

        for _ in range(game.env.horizon):
            obs = game.obs.copy()

            # Raw network prediction (no MCTS)
            net_policy, net_value = agent.net.predict(obs)
            mask = game.get_action_mask()
            net_policy_masked = net_policy * mask
            s = net_policy_masked.sum()
            if s > 0:
                net_policy_masked /= s

            # MCTS prediction
            mcts_policy = agent.policy(game, "", add_noise=False,
                                        temperature_override=0.01)

            # Metrics
            # Entropy of raw network policy
            p = net_policy_masked[net_policy_masked > 0]
            entropies.append(float(-np.sum(p * np.log(p + 1e-10))))

            # KL divergence: sum p_net * log(p_net / p_mcts)
            valid = (net_policy_masked > 0) & (mcts_policy > 0)
            if valid.any():
                kl = np.sum(net_policy_masked[valid] *
                           np.log(net_policy_masked[valid] / mcts_policy[valid]))
                kl_divs.append(float(kl))

            # Action agreement
            net_action = int(np.argmax(net_policy_masked))
            mcts_action = int(np.argmax(mcts_policy))
            agreements.append(1.0 if net_action == mcts_action else 0.0)

            # Store value for later error computation
            net_values.append(float(net_value))

            # Take MCTS action
            action = mcts_action
            _, reward, terminated, truncated, _ = game.step_wrapper(action)
            episode_return += reward
            if terminated or truncated:
                break

        # Compute value errors using actual discounted returns
        # (simplified: just compare v(s0) to episode return)
        if net_values:
            value_errors.append(abs(net_values[0] - episode_return))

    return {
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mcts_kl": float(np.mean(kl_divs)) if kl_divs else 0.0,
        "value_error": float(np.mean(value_errors)) if value_errors else 0.0,
        "action_agreement": float(np.mean(agreements)) if agreements else 0.0,
    }
```

Call this at the end of each iteration in `_run_single_seed()` when `--log-analysis` is set, and append to `iteration_log` entries. Save to `analysis_log.jsonl`.

---

## 4. Experiments

### 4.1 Overview Table

All experiments use D=10, L=3 (obs_size=49, action_space=49, horizon=95).

| ID | Name | Ablation Flag | MCTS Sims | Network Trains? | What It Tests |
|----|------|--------------|-----------|----------------|---------------|
| E1 | Full AlphaZero | (none) | 120 | Yes | Baseline (already have data) |
| E2 | Frozen Network | `--ablation frozen` | 120 | No | Pure MCTS, random network |
| E3 | No MCTS | `--no-mcts` | -1 | Yes | Pure network policy, no search |
| E4 | Uniform Policy | `--ablation uniform-policy` | 120 | Yes | Value head only |
| E5 | Zero Value | `--ablation zero-value` | 120 | Yes | Policy prior only |
| E6a | Frozen + Low Sims | `--ablation frozen` | 20 | No | Pure MCTS, low budget |
| E6b | Frozen + High Sims | `--ablation frozen` | 500 | No | Pure MCTS, high budget |
| E7a | Full + Low Sims | (none) | 20 | Yes | AlphaZero, low budget |
| E7b | Full + Medium Sims | (none) | 50 | Yes | AlphaZero, medium budget |

### 4.2 Terminal Commands

Each command assumes you're in the project root directory.

**E1: Full AlphaZero Baseline** (already have data, but re-run with analysis logging)
```bash
python scripts/run_doors_direct.py \
    --seeds 41 42 43 \
    --non-interactive \
    --log-analysis
```

**E2: Frozen Network (pure MCTS, 120 sims)**
```bash
python scripts/run_doors_direct.py \
    --seeds 41 42 43 \
    --non-interactive \
    --ablation frozen
```

**E3: No MCTS (pure network policy)**
```bash
python scripts/run_doors_direct.py \
    --seeds 41 42 43 \
    --non-interactive \
    --no-mcts
```

**E4: Uniform Policy (value head only)**
```bash
python scripts/run_doors_direct.py \
    --seeds 41 42 43 \
    --non-interactive \
    --ablation uniform-policy
```

**E5: Zero Value (policy prior only)**
```bash
python scripts/run_doors_direct.py \
    --seeds 41 42 43 \
    --non-interactive \
    --ablation zero-value
```

**E6: Frozen Network Simulation Sweep**
```bash
# Run each simulation count separately (or write a loop)
for sims in 20 50 100 200 500; do
    python scripts/run_doors_direct.py \
        --seeds 41 42 43 \
        --non-interactive \
        --ablation frozen \
        --override n_simulations=$sims
done
```

Note: `--override n_simulations=$sims` requires adding a generic `--override KEY=VALUE` CLI arg (see Section 3.2 below), or just manually editing the config for each run. Alternatively, we can add a `--n-simulations` override flag specifically.

**E7: Full AlphaZero Simulation Sweep**
```bash
for sims in 20 50 100 200 500; do
    python scripts/run_doors_direct.py \
        --seeds 41 42 43 \
        --non-interactive \
        --override n_simulations=$sims
done
```

### 4.3 Additional CLI: `--override` for Hyperparameter Sweeps

Add to `parse_args()`:

```python
parser.add_argument("--override", nargs="*", default=[],
                    help="Override config values as KEY=VALUE pairs. "
                         "Supported: n_simulations, n_iterations, n_games_per_train")
```

In `main()`, after creating `cfg`:

```python
for item in args.override:
    key, val = item.split("=", 1)
    if key == "n_simulations":
        cfg.agent.mcts_params["n_simulations"] = int(val)
    elif key == "n_iterations":
        cfg.run.n_iterations = int(val)
    elif key == "n_games_per_train":
        cfg.trainer.n_games_per_train = int(val)
```

---

## 5. Plots

### 5.1 New File: `scripts/plot_ablation.py`

A standalone script that reads experiment output directories and generates comparison plots.

#### Plot A: Solve Rate vs Iteration (All Conditions)

**What**: 5 curves on one plot (E1–E5), showing solve rate over training iterations.

**Data source**: `iteration_log.jsonl` → field `best_solve_rate`

**Purpose**: Answers "does the network matter?" at a glance.

```
Y: best_solve_rate (0–1)
X: iteration (1–30)
Curves: Full, Frozen, NoMCTS, UniformPolicy, ZeroValue
Style: solid line = mean across seeds, shaded band = ±1 std
```

**Expected outcome**:
- If Frozen ≈ Full → network doesn't help at 120 sims
- If NoMCTS ≈ 0 → network can't solve alone
- If ZeroValue > UniformPolicy → policy prior matters more than value

#### Plot B: Solve Rate vs Simulation Count (Frozen vs Learning)

**What**: The "crossover plot" — where does learning start to matter?

**Data source**: E6 + E7 iteration_log.jsonl → final `best_solve_rate`

**Purpose**: Finds the simulation budget where a trained network gives an advantage.

```
Y: best_solve_rate at iteration 30 (0–1)
X: n_simulations (log scale: 20, 50, 100, 200, 500)
Curves: Frozen (red), Full AlphaZero (blue)
Points: individual seeds as scatter, line through means
```

**Expected outcome**:
- At high sims (200+): both converge → MCTS sufficient alone
- At low sims (20–50): Full > Frozen → network compensates for less search
- Crossover point: the sim count where the curves diverge

#### Plot C: Reward vs Iteration (All Conditions)

**What**: Same as Plot A but using `avg_eval_reward` instead of solve rate.

**Data source**: `iteration_log.jsonl` → field `avg_eval_reward`

```
Y: avg_eval_reward
X: iteration
Curves: Full, Frozen, NoMCTS, UniformPolicy, ZeroValue
Horizontal line: optimal reward (1.71 for D=10)
```

#### Plot D: Policy-MCTS KL Divergence Over Training (Analysis 1)

**What**: How much does MCTS override the network's policy?

**Data source**: `analysis_log.jsonl` → field `mcts_kl` (requires `--log-analysis`)

```
Y: KL(network || MCTS)
X: iteration
Single curve (Full AlphaZero), 3 seeds
```

**Interpretation**:
- KL shrinks → network is learning to match MCTS (distillation working)
- KL stays high → MCTS always corrects the network significantly

#### Plot E: Action Agreement Rate Over Training (Analysis 2)

**What**: How often does the raw network agree with MCTS's choice?

**Data source**: `analysis_log.jsonl` → field `action_agreement`

```
Y: agreement rate (0–1)
X: iteration
```

**Interpretation**:
- Agreement rises to ~1.0 → network has fully internalized the policy
- Agreement stays at ~0.3 → network is weak, MCTS is essential

#### Plot F: Network Prediction Quality (Composite)

**What**: 2×2 subplot combining entropy, KL, value error, agreement.

**Data source**: `analysis_log.jsonl`

```
(0,0): policy_entropy vs iteration — does the network become more confident?
(0,1): mcts_kl vs iteration — does MCTS correction shrink?
(1,0): value_error vs iteration — does value prediction improve?
(1,1): action_agreement vs iteration — does the network match MCTS?
```

### 5.2 Running the Plotting Script

```bash
# After all experiments are done:
python scripts/plot_ablation.py \
    --exp-dir experiments/doors_direct/ \
    --output-dir experiments/doors_direct/ablation_plots/
```

The script should:
1. Auto-discover experiment directories by their ablation suffix
2. Group by condition (full, frozen, uniform-policy, zero-value, nomcts)
3. Generate Plots A–F as PNG files

---

## 6. Execution Order

| Step | What | Time Estimate | Depends On |
|------|------|--------------|------------|
| 1 | Create `network_ablations.py` | 5 min | — |
| 2 | Add CLI flags to `run_doors_direct.py` | 15 min | — |
| 3 | Add analysis logging | 15 min | — |
| 4 | Run E2 (Frozen, 120 sims) | ~20 min | Steps 1–2 |
| 5 | Run E3 (No MCTS) | ~10 min | Step 2 |
| 6 | Run E4 (Uniform Policy) | ~20 min | Steps 1–2 |
| 7 | Run E5 (Zero Value) | ~20 min | Steps 1–2 |
| 8 | Run E1 with analysis logging | ~25 min | Step 3 |
| 9 | Run E6 (Frozen sim sweep) | ~1.5 hr | Steps 1–2 |
| 10 | Run E7 (Full sim sweep) | ~1.5 hr | Step 2 |
| 11 | Create `plot_ablation.py` | 30 min | — |
| 12 | Generate all plots | 2 min | Steps 4–11 |

Steps 4–7 can run in parallel on different terminals.
Steps 9–10 can run in parallel on different terminals.

---

## 7. Expected Outcomes & Decision Tree

```
E2 result (Frozen, 120 sims):
├── Solves D=10 (solve_rate ≈ 1.0)
│   → Network is DISPENSABLE at this scale
│   → Next: try D=20 or D=30 to find where network matters
│   └── E6 sweep tells us: minimum sims for pure MCTS
│
└── Does NOT solve D=10
    → Network IS contributing
    → E4 vs E5 tells us which head matters:
    ├── E4 (uniform-policy) fails, E5 (zero-value) works
    │   → Policy prior is the key contribution
    ├── E4 works, E5 fails
    │   → Value head is the key contribution
    └── Both fail
        → Both heads are needed jointly

E3 result (No MCTS):
├── Solves D=10
│   → Network learns a complete policy, MCTS is dispensable
│   → This would be surprising
└── Does NOT solve
    → Network alone is insufficient (expected)
    → But KL/agreement analysis shows if network is APPROACHING sufficiency
```

---

## 8. What This Tells Us About the Derivation Game

The direct-play ablation results will inform derivation game strategy:

1. **If MCTS alone suffices** (Frozen solves D=10): The derivation game's AlphaZero failures are NOT about the MCTS-network interaction — they're about the derivation game's combinatorial explosion. Focus on grammar/search space redesign.

2. **If the network is critical** (Frozen fails): The same MCTS-network synergy might be necessary for the derivation game too. Investigate why the derivation game's network isn't learning (sparse reward, large action space, poor credit assignment).

3. **Policy vs Value importance**: If the policy prior matters more than value estimates in direct play, then for the derivation game we should focus on learning good grammar production priors rather than accurate program value estimates.
