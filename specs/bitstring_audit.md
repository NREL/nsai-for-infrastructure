# BitString Game Audit: AlphaZero for the OneMax / BitString RL Problem

_This document is a systematic audit of the BitString game implementation and the AlphaZero algorithm applied to it. It catalogues the MDP structure, algorithm design, all tunable knobs, and relevant code. It is intended to be consumed by an LLM for a rigorous comparative analysis of how various RL algorithms would perform on this problem._

---

## Table of Contents

1. [Problem Statement & Formal MDP Definition](#1-problem-statement--formal-mdp-definition)
2. [Fitness Landscape Analysis](#2-fitness-landscape-analysis)
3. [System Architecture & File Map](#3-system-architecture--file-map)
4. [Algorithm Flow Diagram](#4-algorithm-flow-diagram)
5. [Environment Implementation](#5-environment-implementation)
6. [Reward Shaping via Potential Functions](#6-reward-shaping-via-potential-functions)
7. [Neural Network Architecture](#7-neural-network-architecture)
8. [MCTS Implementation](#8-mcts-implementation)
9. [Agent & Self-Play](#9-agent--self-play)
10. [Training Loop & Gating](#10-training-loop--gating)
11. [Configuration & Design Knobs](#11-configuration--design-knobs)
12. [Key Structural Properties of the BitString MDP](#12-key-structural-properties-of-the-bitstring-mdp)
13. [Open Questions for RL Algorithm Comparison](#13-open-questions-for-rl-algorithm-comparison)

---

## 1. Problem Statement & Formal MDP Definition

### Informal description

An agent operates on an N-bit binary string. Some bits start as 1, the rest as 0. At each step, the agent selects one bit index to flip (toggle 0↔1). The goal is to reach the all-ones state as efficiently as possible. A potential function provides shaped reward to guide learning.

### Formal MDP

| Component | Definition |
|-----------|-----------|
| **State space** | S = {0, 1}^N × {0, 1, ..., T_max}. A state is a pair (x, t) where x is the N-bit binary vector and t is the step count. |
| **Initial state** | x₀ has exactly `n_ones` bits set to 1 (randomly placed). t₀ = 0. |
| **Action space** | A = {0, 1, ..., N-1}. Action `a` selects bit index `a`. |
| **Transition** | x' = x with bit `a` flipped (x'[a] = 1 - x[a]). t' = t + 1. Deterministic. |
| **Terminal condition** | Episode ends when sum(x) = N (all ones) OR t ≥ T_max. |
| **T_max** | Dense mode: 2N. Sparse mode: N - n_ones (minimum steps to solve). |
| **Dense reward** | r(x, a) = +1/N if x[a]=0 (flipping 0→1), else -1/N (flipping 1→0). |
| **Sparse reward** | r = 0 at all non-terminal steps. At terminal: r = sum(x)/N. |
| **Shaped reward** | r(x, a) = (φ(x') - φ(x)) / N, where φ is a potential function (onemax, leading_ones, or binval). |
| **Discount factor** | γ = 1.0 (undiscounted episodic). |
| **Optimal policy (OneMax)** | Always flip a 0-bit to 1. Never flip a 1-bit. Solves in exactly N - n_ones steps. |
| **Optimal return (OneMax)** | (N - n_ones) / N. For N=10, n_ones=2: 0.8. |

### Key properties

- **Fully observable**: The agent sees the complete binary vector.
- **Deterministic transitions**: Given (state, action), the next state is deterministic.
- **Stochastic initial states**: The placement of initial 1-bits is random (unless frozen).
- **Symmetric action space**: All N actions are always legal (no invalid actions).
- **State space size**: |S| = 2^N × T_max. For N=10, T_max=20: ~20,000 states.
- **Episode length**: At most T_max steps (20 for dense, 8 for sparse with n_ones=2).

---

## 2. Fitness Landscape Analysis

Three potential functions define different reward-shaping landscapes. All map a bitstring x ∈ {0,1}^N to an integer.

### OneMax: φ(x) = sum(x)

```python
# potentials.py
def onemax(x: np.ndarray) -> int:
    """Sum of bits. Optimal value = len(x)."""
    return int(np.sum(x))
```

**Properties:**
- Monotonically increasing with number of 1-bits
- Every 0→1 flip yields reward +1/N; every 1→0 flip yields -1/N
- No plateaus, no local optima — pure gradient toward the goal
- The greedy policy (always flip a 0-bit) is optimal
- Difficulty: **Easy**. The reward signal perfectly aligns with progress.

**Optimal strategy:** Scan bits, find any 0, flip it. Order doesn't matter.

### LeadingOnes: φ(x) = length of leading all-1s prefix

```python
# potentials.py
def leading_ones(x: np.ndarray) -> int:
    """Length of the leading all-ones prefix.
    For x = [1, 1, 1, 0, 1], returns 3.
    For x = [0, 1, 1, 1, 1], returns 0.
    """
    n = len(x)
    for i in range(n):
        if x[i] != 1:
            return i
    return n
```

**Properties:**
- Only counts consecutive 1s from the left
- Flipping bit `i` to 1 only yields reward if bits 0..i-1 are already 1
- **Large plateaus**: flipping any bit past the first 0 yields zero potential change
- **Sequential dependency**: bits must be set left-to-right for reward signal
- Difficulty: **Hard**. The agent must discover the correct ordering without intermediate reward for out-of-order actions.

**Optimal strategy:** Flip bits strictly left-to-right: bit 0, then bit 1, then bit 2, etc.

**Deceptive property:** Flipping bit 5 to 1 when bit 2 is still 0 gives zero reward, even though it's "progress" toward all-ones. The agent gets no signal for this useful action.

### BinVal: φ(x) = binary value interpretation (x[0] is MSB)

```python
# potentials.py
def binval(x: np.ndarray) -> int:
    """Binary value with x[0] as the most significant bit.
    For x = [1, 0, 1], returns 5 (i.e. 1*4 + 0*2 + 1*1).
    """
    result = 0
    for bit in x:
        result = 2 * result + int(bit)
    return result
```

**Properties:**
- Exponentially weighted: bit 0 contributes 2^(N-1), bit N-1 contributes 1
- Flipping bit `i` from 0→1 yields reward 2^(N-1-i) / N
- **Extreme reward imbalance**: flipping bit 0 gives 2^(N-1)/N times more reward than flipping bit N-1
- For N=10: flipping bit 0 gives 512/10 = 51.2× more reward per step than bit 9
- Difficulty: **Medium-Hard**. The agent is strongly incentivized to set high-order bits first, but may neglect low-order bits.

**Optimal strategy:** Set bits in order of significance (MSB first), though any order that only flips 0→1 is equally efficient.

### Landscape comparison table

| Property | OneMax | LeadingOnes | BinVal |
|----------|--------|-------------|--------|
| Max potential | N | N | 2^N - 1 |
| Reward per useful flip | +1/N (uniform) | +1/N if sequential, else 0 | 2^(N-1-i)/N (exponential) |
| Plateaus | None | Large (N-k positions give 0 reward) | None (but extreme skew) |
| Local optima | None | None (but plateaus) | None |
| Greedy-optimal | Yes | Yes (if ordered) | Yes (if ordered) |
| Required ordering | Any | Strict left-to-right | Any (but MSB-first is greedy) |
| Exploration challenge | Low | High (must discover ordering) | Medium (must balance bit priority) |

---

## 3. System Architecture & File Map

### Core AlphaZero Framework (domain-agnostic)

| File | Role | Key interfaces |
|------|------|---------------|
| `src/alphazeropp/core/game.py` | Abstract Game base class | `reset()`, `step()`, `get_action_mask()`, `hashable_obs`, `stash_state()`, `clone()` |
| `src/alphazeropp/core/mcts.py` | MCTS search engine | `perform_simulations()`, `search()`, `calc_masked_ucbs()`, `update_edge()` |
| `src/alphazeropp/core/agent.py` | Agent: wraps MCTS + network | `policy()` → move probabilities; `play_one_round()` → training examples |
| `src/alphazeropp/core/policy_value_net.py` | Abstract PolicyValueNet + default MLP | `predict(state) → (policy, value)`; `train(examples)` |

### BitString Game Instance

| File | Role | Key contents |
|------|------|-------------|
| `src/.../bitstring/game.py` | Gymnasium environment + Game wrapper | `BitStringGym`, `BitStringGame` |
| `src/.../bitstring/shaped_env.py` | Potential-based reward shaping wrapper | `ShapedBitStringGym`, `make_initial_states()` |
| `src/.../bitstring/potentials.py` | Potential functions for reward shaping | `onemax`, `leading_ones`, `binval`, `POTENTIAL_REGISTRY` |
| `src/.../bitstring/network.py` | MLP policy-value network | `BitStringPolicyValueNet` |
| `src/.../bitstring/config.py` | Configuration & wiring | `BitStringConfig` — builds all components |

### Training Infrastructure

| File | Role |
|------|------|
| `src/.../training/trainer.py` | Self-play data collection + network training |
| `src/.../training/evaluator.py` | Pitting new vs old network |
| `src/.../training/gated_trainer.py` | Accept/reject gating around trainer + evaluator |

### Entry Point

| File | Role |
|------|------|
| `scripts/run_bitstring.py` | Main training script — interactive config, training loop, plotting |

---

## 4. Algorithm Flow Diagram

```
OUTER LOOP: n_iterations training iterations (default: 10)
│
├── STEP 1: SELF-PLAY (40 games, each ~10-20 steps for N=10)
│     │
│     │  Each game starts from a random N-bit string with n_ones=2 bits set.
│     │
│     │  ┌──────────────────────────────────────────────┐
│     │  │  For each step in the episode:                │
│     │  │                                               │
│     │  │   bitstring obs ──► MLP(128×2 hidden)         │
│     │  │                     ├── policy: softmax logits │
│     │  │                     └── value: scalar estimate │
│     │  │                                               │
│     │  │   policy + value ──► MCTS(20 simulations)     │
│     │  │                      UCB = Q_norm + c·P·√N    │
│     │  │                               /(1+n)          │
│     │  │                                               │
│     │  │   MCTS visit counts ──► π_MCTS (temperature)  │
│     │  │   Sample action ~ π_MCTS                      │
│     │  │   Flip bit[action] in bitstring               │
│     │  │                                               │
│     │  │   SAVE: (observation, π_MCTS, discounted_ret) │
│     │  └──────────────────────────────────────────────┘
│     │
│     │  At terminal: all bits=1 or step_count >= T_max
│     │  Reward at each step: (φ(x') - φ(x)) / N
│     │  Discounted return: z_t = Σ_{k≥t} γ^(k-t) r_k  (γ=1.0)
│     │
├── STEP 2: TRAIN NETWORK
│     │  Replay buffer: last 5 iterations (~2000 examples for N=10)
│     │  Loss = MSE(v_predicted, z_target) + 2.0 * CE(π_predicted, π_MCTS)
│     │  Adam optimizer, lr=1e-3, 10 epochs, batch_size=32
│     │
├── STEP 3: GATE (accept/reject)
│     │  Deep-copy old agent
│     │  Pit new net vs old net: 20 evaluation games
│     │  Accept if win_rate >= 55%, else restore old weights
│     │
└── Repeat
```

### The MDP at a glance

```
State:     [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]   (10-bit binary vector, n_ones=2 initially set)
Action:    flip bit index 2                     (any of 10 actions always legal)
Next:      [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]     (bit 2 flipped 0→1)
Reward:    (onemax([0,1,1,0,1,0,0,0,1,0]) - onemax([0,1,0,0,1,0,0,0,1,0])) / 10
         = (4 - 3) / 10 = +0.1

Terminal:  when sum(state) == 10 or step_count >= 20
```

---

## 5. Environment Implementation

### BitStringGym — the core Gymnasium environment

```python
# src/alphazeropp/instances/bitstring/game.py (complete)

class BitStringGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_sites=10, bit_flip=True, sparse_reward=True, n_ones=2):
        super().__init__()
        self.bit_flip = bit_flip
        self.sparse_reward = sparse_reward
        self.n_ones = n_ones

        self.n_sites = n_sites
        self.max_steps = 2 * self.n_sites if not self.sparse_reward else self.n_sites - self.n_ones
        self.observation_space = spaces.MultiBinary([self.n_sites])
        self.action_space = spaces.Discrete(self.n_sites)

        self.state = None
        self.step_count = 0

    def step(self, action):
        assert self.state is not None, "Environment must be reset before stepping."

        self.step_count += 1
        done = self.step_count >= self.max_steps
        r = -1.0 / self.n_sites

        if action == -1:
            return self.state.copy(), r, done, {}

        if self.state[action] == 0:
            r = 1.0 / self.n_sites

        if self.bit_flip:
            self.state[action] = 1 - self.state[action]  # Flip the bit
        else:
            self.state[action] = 1

        done = done or sum(self.state) == self.n_sites

        normalizer = self.n_sites
        if self.sparse_reward:
            if done:
                r = sum(self.state) / normalizer
            else:
                r = 0.0
        truncated = done

        return self.state.copy(), r, done, truncated, {}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        ones = np.random.choice(range(self.n_sites), self.n_ones, replace=False)
        self.state = np.zeros(self.n_sites, dtype=np.float32)
        self.state[ones] = 1
        self.step_count = 0

        return self.state.copy(), {}
```

**Design notes:**

- **`bit_flip=True`**: Action toggles the bit. The agent can undo its own actions, creating the possibility of cycling. If `bit_flip=False`, action sets bit to 1 (monotonic progress, cannot undo).
- **`sparse_reward=True`**: Reward only at terminal. If False, dense ±1/N reward per step.
- **`max_steps`**: With dense reward, T_max = 2N (generous budget, allows mistakes). With sparse reward, T_max = N - n_ones (tight budget, no room for error).
- **Default reward (dense mode)**: +1/N for 0→1 flip, -1/N for 1→0 flip. This is the base reward *before* any potential-based shaping is applied.
- **All actions always legal**: The action mask is always all-ones. The agent can flip any bit at any time, including bits that are already 1.

### BitStringGame — the AlphaZero-compatible wrapper

```python
# src/alphazeropp/instances/bitstring/game.py

class BitStringGame(EnvGame):
    def __init__(self, env=None, **kwargs):
        if env is None:
            env = BitStringGym(**kwargs)
        super().__init__(env)
        self.action_mask = np.ones(env.n_sites, dtype=bool)  # All actions always available

    def get_action_mask(self):
        return self.action_mask

    @property
    def hashable_obs(self) -> Hashable:
        return "".join([str(int(x)) for x in self.obs]) + " " + str(self.env.step_count)
```

**Design note — `hashable_obs` includes step count**: Two visits to the same bitstring at different time steps are treated as different MCTS tree nodes. This is correct for a time-limited MDP where the optimal policy may depend on remaining budget.

---

## 6. Reward Shaping via Potential Functions

### ShapedBitStringGym — the shaping wrapper

```python
# src/alphazeropp/instances/bitstring/shaped_env.py (complete)

class ShapedBitStringGym(gym.Wrapper):
    """Wraps BitStringGym to provide potential-based shaped rewards."""

    def __init__(self, env: BitStringGym, potential_fn: Callable[[np.ndarray], int],
                 reward_mode: str = "dense_potential",
                 frozen_states: Optional[Sequence[np.ndarray]] = None):
        super().__init__(env)
        assert reward_mode in ("dense_potential", "sparse_pm1")
        self.potential_fn = potential_fn
        self.reward_mode = reward_mode
        self.frozen_states = frozen_states
        self._frozen_idx = 0
        self._prev_potential: int = 0

    def __getattr__(self, name):
        """Proxy attribute access to the wrapped BitStringGym env."""
        if "env" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        if self.frozen_states is not None:
            obs, info = self.env.reset(**kwargs)
            state = self.frozen_states[self._frozen_idx % len(self.frozen_states)]
            self.env.state = state.copy()
            self.env.step_count = 0
            obs = state.copy()
            self._frozen_idx += 1
        else:
            obs, info = self.env.reset(**kwargs)
            info = {}
        self._prev_potential = self.potential_fn(obs)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        if self.reward_mode == "dense_potential":
            new_potential = self.potential_fn(obs)
            reward = (new_potential - self._prev_potential) / self.env.n_sites
            info["potential_prev"] = self._prev_potential
            info["potential_curr"] = new_potential
            self._prev_potential = new_potential
        else:
            reward = original_reward
        return obs, reward, terminated, truncated, info

    def reset_frozen_index(self):
        self._frozen_idx = 0
```

### Potential-based reward shaping theory

The shaped reward at step t is:

```
r_shaped(t) = (φ(s_{t+1}) - φ(s_t)) / N
```

By the **telescoping property**, the cumulative shaped reward over an episode equals:

```
Σ r_shaped(t) = (φ(s_T) - φ(s_0)) / N
```

This means the total return is determined solely by the initial and final states, regardless of the path taken. This is a well-known result from potential-based reward shaping (Ng et al., 1999) — it preserves the optimal policy.

**Implementation detail:** Potential functions return `int` (not float) to enable exact telescoping verification via integer arithmetic, avoiding floating-point accumulation error.

### Reward modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `dense_potential` | r_t = (φ(s_{t+1}) - φ(s_t)) / N | Default. Provides dense signal at every step. |
| `sparse_pm1` | Pass-through from BitStringGym | Sparse reward only at terminal state. |

### Frozen states

When `frozen_states` is provided, the environment cycles through a deterministic list of initial states rather than randomizing. Used for evaluation (deterministic comparison) and for the DSL leaf evaluator.

---

## 7. Neural Network Architecture

### PolicyValueNetModel — the shared-body MLP

```python
# src/alphazeropp/core/policy_value_net.py

class PolicyValueNetModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 n_hidden_layers: int = 2, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.body = nn.Sequential(
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
              for _ in range(n_hidden_layers)],
        )
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.body(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value
```

**Architecture diagram (for N=10, default config):**

```
Input: x ∈ R^10  (binary vector)
  │
  ├── Linear(10, 128) + ReLU    ← shared body layer 1
  ├── Linear(128, 128) + ReLU   ← shared body layer 2
  ├── Linear(128, 128) + ReLU   ← shared body layer 3
  │
  ├── Policy head: Linear(128, 10)  → logits → softmax → π(a|s)
  └── Value head:  Linear(128, 1)   → scalar  → v(s)
```

**Parameter count:** ~10×128 + 128×128×2 + 128×10 + 128×1 ≈ 36,000 parameters.

### BitStringPolicyValueNet — training and inference

```python
# src/alphazeropp/instances/bitstring/network.py (complete)

class BitStringPolicyValueNet(TorchPolicyValueNet):
    save_file_name = "bitstring_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 2.0,
    }

    def __init__(self, random_seed=None, n_sites=10, n_hidden_layers=2,
                 hidden_size=128, training_params={}, device=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        model = PolicyValueNetModel(input_size=n_sites, output_size=n_sites,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_size=hidden_size)
        self.n_sites = n_sites
        super().__init__(model)
        self.training_params = self.default_training_params | training_params
        self.DEVICE = get_device() if device is None else device

    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=tp["learning_rate"],
                                     weight_decay=tp["weight_decay"])

        if needs_reshape:
            states = torch.from_numpy(np.array([s for s, _, _ in examples], dtype=np.float32))
            policies = torch.from_numpy(np.array([p for _, p, _ in examples], dtype=np.float32))
            values = torch.from_numpy(np.array([v for _, _, v in examples], dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(states, policies, values)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=tp["batch_size"], shuffle=True)

        train_batch_losses, train_losses, policy_losses, value_losses = [], [], [], []

        for epoch in range(tp["epochs"]):
            model.train()
            train_loss, policy_loss, value_loss = 0.0, 0.0, 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs = inputs.to(self.DEVICE)
                targets_value = targets_value.to(self.DEVICE)
                targets_policy = targets_policy.to(self.DEVICE)

                optimizer.zero_grad()
                outputs_policy, outputs_value = model(inputs)

                loss_value = criterion_value(outputs_value, targets_value)
                loss_policy = criterion_policy(outputs_policy, targets_policy)
                loss = loss_value + policy_weight * loss_policy   # ← 2.0× policy weight

                loss.backward()
                optimizer.step()                                  # ← No gradient clipping

                train_batch_losses.append(loss.item())
                train_loss += loss.item()
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()

            train_losses.append(train_loss / len(train_loader))
            policy_losses.append(policy_loss / len(train_loader))
            value_losses.append(value_loss / len(train_loader))

        return model, train_batch_losses, train_losses, policy_losses, value_losses

    def predict(self, state):
        self.model.cpu()                    # ← Moves to CPU on EVERY predict call
        nn_input = torch.tensor(state).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)

        policy_prob = policy_prob.numpy().squeeze(0)
        value = value.numpy().squeeze(0)

        return policy_prob, value
```

### Loss function

```
L = MSE(v_predicted, z_target) + 2.0 × CrossEntropy(π_logits, π_MCTS) + 1e-4 × ||θ||²
    ─────────────────────────     ──────────────────────────────────────   ────────────
    value loss                    policy loss (weighted 2×)                weight decay
```

**Key observations:**
- `CrossEntropyLoss` with soft targets (π_MCTS is a probability distribution from MCTS visit counts)
- Policy loss is weighted 2× relative to value loss
- No gradient clipping is applied
- Training creates a fresh Adam optimizer each iteration (no momentum carry-over)
- `predict()` moves model to CPU on every call; `train()` moves it to CUDA. With ~8000+ predict calls per iteration (20 sims × 10 steps × 40 games), this causes repeated device transfers.

---

## 8. MCTS Implementation

### MCTSTreeNode — tree node data structure

```python
# src/alphazeropp/core/mcts.py

class MCTSTreeNode():
    def __init__(self, direct_reward: float, is_terminal_state: bool):
        self.direct_reward = direct_reward
        self.is_terminal_state = is_terminal_state
        self.nn_policy = None        # Network policy output
        self.nn_value = None         # Network value estimate
        self.action_mask = None      # Valid actions mask
        self.total_N = 0             # Total visit count
        self.action_Q = {}           # action → Q-value (running mean)
        self.action_N = {}           # action → visit count
        self.q_u_history = []        # Diagnostic log (grows unbounded)
```

### MCTS — the search engine

```python
# src/alphazeropp/core/mcts.py

EPS = 1e-8  # Prevents zeroing policy when total_N is 0

class MCTS():
    def __init__(self, game, net,
                 n_simulations=25, temperature=1.0,
                 c_exploration=1.0, dirichlet_alpha=0.3,
                 dirichlet_epsilon=0.25):
        self.game = game
        self.net = net
        self.nodes = {}
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.c_exploration = c_exploration
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.q_min = float('inf')
        self.q_max = float('-inf')
```

### perform_simulations — the main entry point

```python
    def perform_simulations(self, msg, add_noise=False):
        mystate = self.game.hashable_obs
        self.q_min = float('inf')     # Reset Q-normalization per call
        self.q_max = float('-inf')

        if self.n_simulations < 0:
            counts, _, _ = self.query_net_masked(msg)
        else:
            # Expand root if not yet in tree
            if mystate not in self.nodes:
                old_game_state = self.game.stash_state()
                self.search(entab(msg, ", root expand"))
                self.game = self.game.unstash_state(old_game_state)

            mynode = self.nodes[mystate]

            # Add Dirichlet noise at root (only on fresh root)
            if add_noise and mynode.total_N == 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(mynode.nn_policy))
                mask = mynode.action_mask
                masked_noise = noise * mask
                sum_noise = masked_noise.sum()
                if sum_noise > 0:
                    masked_noise /= sum_noise
                    mynode.nn_policy = ((1 - self.dirichlet_epsilon) * mynode.nn_policy
                                       + self.dirichlet_epsilon * masked_noise)

            # Run simulations, restoring game state after each
            for i in range(self.n_simulations):
                old_game_state = self.game.stash_state()
                self.search(entab(msg, f", simulation {i+1}/{self.n_simulations}"))
                self.game = self.game.unstash_state(old_game_state)

            mynode = self.nodes[mystate]
            counts = np.zeros_like(mynode.nn_policy)
            for action, count in mynode.action_N.items():
                counts[action] = count

        # Numerically stable temperature scaling: counts^(1/T) = exp(log(counts)/T)
        nonzero = counts > 0
        if nonzero.any():
            log_counts = np.full_like(counts, -np.inf)
            log_counts[nonzero] = np.log(counts[nonzero]) / self.temperature
            log_counts -= log_counts.max()
            probs = np.exp(log_counts)
            probs /= probs.sum()
        else:
            probs = counts

        return probs
```

### search — the recursive tree traversal

```python
    def search(self, msg) -> float:
        mystate = self.game.hashable_obs

        # Initialize node if first visit
        if mystate not in self.nodes:
            reward = self.game.reward
            is_terminal = self.game.terminated or self.game.truncated
            self.nodes[mystate] = MCTSTreeNode(reward, is_terminal)
        mynode = self.nodes[mystate]

        # Base case 1: terminal state → future value is 0
        if mynode.is_terminal_state:
            return 0.0

        # Base case 2: unexpanded node → query network, return value
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
        self.game.step_wrapper(to_step)

        immediate_reward = self.game.reward
        future_value = self.search(entab(msg, " recurse"))

        # Bellman: total_reward = R(s,a) + γ·V(s')  (γ=1.0 implicit)
        total_reward = immediate_reward + future_value

        self.update_edge(mynode, best_action, total_reward)
        mynode.total_N += 1

        return total_reward
```

### UCB calculation with Q-normalization

```python
    def calc_masked_ucbs(self, mynode, msg):
        valid_actions = list(zip(*np.nonzero(mynode.action_mask)))
        all_ucbs = np.full(mynode.nn_policy.shape, -np.inf)

        for action in valid_actions:
            q = mynode.action_Q.get(action, 0.0)
            n = mynode.action_N.get(action, 0)

            # Normalize Q to [0, 1] using global min-max
            if self.q_min == float('inf') or self.q_max == float('-inf'):
                q_normalized = 0.0
            elif self.q_max > self.q_min:
                q_normalized = (q - self.q_min) / (self.q_max - self.q_min)
            else:
                q_normalized = 0.5

            u_val = (self.c_exploration * mynode.nn_policy[action]
                     * np.sqrt(mynode.total_N + EPS) / (1 + n))
            ucb = q_normalized + u_val
            all_ucbs[action] = ucb
            mynode.q_u_history.append((q, u_val))

        return all_ucbs
```

**UCB formula:**

```
UCB(s, a) = Q_norm(s, a) + c_exploration × π_net(a) × √(N_total + ε) / (1 + N_a)

where:
  Q_norm(s, a) = (Q(s,a) - Q_min) / (Q_max - Q_min)    ∈ [0, 1]
  π_net(a) = masked, normalized network policy for action a
  N_total = total visits to state s
  N_a = visits to action a from state s
  c_exploration = 1.5 (default)
  ε = 1e-8
```

### Edge update (running mean Q-value)

```python
    def update_edge(self, mynode, action, reward):
        if action not in mynode.action_N:
            mynode.action_N[action] = 0
            mynode.action_Q[action] = 0.0

        mynode.action_Q[action] = (
            mynode.action_N[action] * mynode.action_Q[action] + reward
        ) / (1 + mynode.action_N[action])
        mynode.action_N[action] += 1

        new_q = mynode.action_Q[action]
        if new_q < self.q_min: self.q_min = new_q
        if new_q > self.q_max: self.q_max = new_q
```

---

## 9. Agent & Self-Play

### Agent.policy — how move probabilities are generated

```python
# src/alphazeropp/core/agent.py

class Agent:
    def policy(self, state: Game, msg=None,
               add_noise: bool = True,
               temperature_override: float | None = None) -> np.ndarray:
        current_game_state = state.clone()

        if self.external_policy is not None:
            move_probs = self.external_policy(current_game_state)
        else:
            mcts = MCTS(current_game_state, self.net, **self.mcts_params)  # ← Fresh tree each call
            if temperature_override is not None:
                mcts.temperature = temperature_override
            move_probs = mcts.perform_simulations("", add_noise=add_noise)

        return move_probs
```

**Key observation:** A fresh MCTS tree is created for every `policy()` call. The tree from the previous step is discarded. No tree reuse across steps.

### Agent.play_one_round — experience collection

```python
    def play_one_round(self, game: Game, max_moves: int = 10_000,
                       random_seed: int | None = None, msg="",
                       add_noise: bool = True,
                       temperature_override: float | None = None):
        current_game_state = game.clone()
        rng = np.random.default_rng(random_seed)

        collected_experience = []
        collected_rewards = []
        cumulative_reward = 0.0
        for i in range(max_moves):
            move_probs = self.policy(current_game_state, "",
                                     add_noise=add_noise,
                                     temperature_override=temperature_override)
            action_idx = rng.choice(len(move_probs), p=move_probs)
            collected_experience.append((current_game_state.obs.copy(), move_probs))

            _, reward, terminated, truncated, _ = current_game_state.step_wrapper(action_idx)

            collected_rewards.append(reward)
            cumulative_reward += reward
            if terminated or truncated:
                break

        # Discounted returns (γ=1.0 → z_t = Σ_{k≥t} r_k)
        discounted_rewards = []
        cumulative_reward = 0.0
        for reward in reversed(collected_rewards):
            cumulative_reward = reward + self.reward_discount * cumulative_reward
            discounted_rewards.append(cumulative_reward)
        discounted_rewards.reverse()

        collected_experience = [
            (obs, move_probs, discounted_reward)
            for ((obs, move_probs), discounted_reward)
            in zip(collected_experience, discounted_rewards)
        ]

        return collected_experience, cumulative_reward
```

**Discounted returns with γ=1.0 and dense potential-based shaping:**

For OneMax with a good episode (only flipping 0→1):
- Rewards: [+0.1, +0.1, +0.1, +0.1, +0.1, +0.1, +0.1, +0.1]
- Returns: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

The value target **decreases** monotonically as the episode progresses — this is correct and informative. Early states have high value (much reward remaining), late states have low value.

For OneMax with a bad episode (some 1→0 flips):
- Rewards: [+0.1, -0.1, +0.1, +0.1, -0.1, ...]
- Returns: [less than 0.8, less predictable]

The value head should learn that states with more 0-bits remaining have higher value under a good policy.

---

## 10. Training Loop & Gating

### Trainer — orchestrates self-play and training

```python
# src/alphazeropp/training/trainer.py

class Trainer:
    def __init__(self, agent, net, game,
                 n_games_per_train=100,
                 n_past_iterations_to_train=20,
                 n_procs=None, checkpoint_dir="checkpoints"):
        self.agent = agent
        self.net = net
        self.game = game
        self.n_games_per_train = n_games_per_train
        self.n_past_iterations_to_train = n_past_iterations_to_train
        self.n_procs = n_procs
        self.all_training_examples = []
        # ...

    def _collect_training_examples(self):
        """Play n_games_per_train games, collect (state, π_MCTS, return) triples."""
        mp_manager = MultiprocessingManager(self.agent.net, self)
        mp_manager.push()
        multiprocessing_function = partial(self.agent.play_for_experience, self.game)
        try:
            arg_tuples = [
                (i, self.agent._randseed("train"), self.agent._randseed("mcts"))
                for i in range(self.n_games_per_train)
            ]
            if self.n_procs is not None and self.n_procs < 0:
                # Sequential mode
                train_example_sets = []
                for j, args in enumerate(arg_tuples):
                    result = multiprocessing_function(*args)
                    train_example_sets.append(result)
            else:
                train_example_sets = MultiprocessingManager.starmap(
                    multiprocessing_function, arg_tuples, self.n_procs)
        finally:
            mp_manager.pop()
        return train_example_sets

    def _process_training_examples(self, new_train_examples):
        """Append to replay buffer, keep last N iterations, flatten."""
        self.all_training_examples.append(new_train_examples)
        if (self.n_past_iterations_to_train is not None and
            len(self.all_training_examples) > self.n_past_iterations_to_train):
            self.all_training_examples.pop(0)       # ← Hard truncation of oldest
        flat = list(itertools.chain.from_iterable(
            itertools.chain.from_iterable(self.all_training_examples)))
        return flat

    def train_iteration(self):
        train_example_sets = self._collect_training_examples()
        experience = [example_set[0] for example_set in train_example_sets]
        flat_examples = self._process_training_examples(experience)
        self._train_network(flat_examples)
```

### Evaluator — pitting new vs old network

```python
# src/alphazeropp/training/evaluator.py

EVAL_TEMPERATURE = 0.05   # Near-greedy action selection

class Evaluator:
    def __init__(self, n_games=20, n_procs=None):
        self.n_games = n_games
        self.n_procs = n_procs

    def _play_for_eval(self, reset_seed, mcts_seed, new_agent, old_agent,
                       try_without_mcts=False):
        base_game = new_agent.game.clone()
        base_game.reset_wrapper(seed=reset_seed)
        old_game = base_game.clone()       # Both play from SAME initial state
        new_game = base_game.clone()

        old_trajectory, old_reward = old_agent.play_one_round(
            game=old_game, random_seed=mcts_seed,
            add_noise=False, temperature_override=EVAL_TEMPERATURE)
        new_trajectory, new_reward = new_agent.play_one_round(
            game=new_game, random_seed=mcts_seed,
            add_noise=False, temperature_override=EVAL_TEMPERATURE)
        return {"old_net": old_reward, "new_net": new_reward}

    def pit(self, new_agent, old_agent, try_without_mcts=False):
        # ... multiprocessing setup ...
        arg_tuples = [
            (old_agent._randseed("eval"), old_agent._randseed("mcts"),
             new_agent, old_agent, try_without_mcts)
            for i in range(self.n_games)     # 20 games by default
        ]
        eval_results = MultiprocessingManager.starmap(
            self._play_for_eval, arg_tuples, self.n_procs)

        old_rewards = np.array([r["old_net"] for r in eval_results])
        new_rewards = np.array([r["new_net"] for r in eval_results])

        wins = np.sum(new_rewards > old_rewards)
        ties = np.sum(np.isclose(new_rewards, old_rewards))
        losses = np.sum(new_rewards < old_rewards)
        score = (wins + ties / 2) / self.n_games

        return score
```

### GatedTrainer — accept/reject gating

```python
# src/alphazeropp/training/gated_trainer.py

class GatedTrainer:
    def __init__(self, trainer, evaluator, acceptance_threshold=0.55):
        self.trainer = trainer
        self.evaluator = evaluator
        self.acceptance_threshold = acceptance_threshold

    def train_iteration(self):
        old_agent = copy.deepcopy(self.trainer.agent)       # Snapshot before training
        self.trainer.train_iteration()                       # Train (modifies weights in-place)
        new_agent = copy.deepcopy(self.trainer.agent)        # Snapshot after training

        score = self.evaluator.pit(new_agent=new_agent, old_agent=old_agent)

        accepted = score >= self.acceptance_threshold        # Default: >= 55%
        if not accepted:
            old_state_dict = old_agent.net.model.state_dict()
            self.trainer.net.model.load_state_dict(old_state_dict)  # Restore old weights

        return score, accepted
```

---

## 11. Configuration & Design Knobs

### Default Configuration (from `BitStringConfig`)

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Game** | `n_sites` | 10 | Length of binary vector (state/action dimensionality) |
| | `bit_flip` | True | If True, action flips bit; if False, sets to 1 |
| | `sparse_reward` | False | If True, reward only at terminal; if False, dense |
| | `n_ones` | 2 | Number of bits initially set to 1 |
| | `fitness_fn` | None | Potential function: None, "onemax", "leading_ones", "binval" |
| | `reward_mode` | "dense_potential" | "dense_potential" or "sparse_pm1" |
| **Network** | `n_hidden_layers` | 2 | Shared body depth |
| | `hidden_size` | 128 | Neurons per hidden layer |
| | `learning_rate` | 0.001 | Adam optimizer learning rate |
| | `batch_size` | 32 | Training mini-batch size |
| | `epochs` | 10 | Training epochs per iteration |
| | `weight_decay` | 1e-4 | L2 regularization |
| | `policy_weight` | 2.0 | Policy loss multiplier (relative to value loss) |
| **MCTS** | `n_simulations` | 20 | MCTS rollouts per action selection |
| | `temperature` | 1.0 | Visit count → probability conversion |
| | `c_exploration` | 1.5 | UCB exploration constant |
| | `dirichlet_alpha` | 0.3 | Dirichlet noise concentration |
| | `dirichlet_epsilon` | 0.25 | Dirichlet noise weight at root |
| **Agent** | `reward_discount` | 1.0 | Discount factor γ (undiscounted) |
| **Trainer** | `n_games_per_train` | 40 | Self-play games per iteration |
| | `n_past_iterations_to_train` | 5 | Replay buffer window (iterations) |
| | `n_procs` | 8 | Parallel workers for self-play |
| **Evaluator** | `n_games` | 20 | Pit games for gating |
| | `EVAL_TEMPERATURE` | 0.05 | Near-greedy evaluation |
| | `n_procs` | 8 | Parallel workers for evaluation |
| **Gating** | `accept_threshold` | 0.55 | Win rate to accept new network |
| **Run** | `n_iterations` | 10 | Total training iterations |
| | `plot_every` | 3 | Plot frequency |

### Alternative configuration (from `bitstring_config.json`)

| Parameter | Alternative | Default |
|-----------|-------------|---------|
| `sparse_reward` | True | False |
| `n_simulations` | 120 | 20 |
| `n_games_per_train` | 100 | 40 |
| `n_iterations` | 100 | 10 |

### Key quantities for default config (N=10, n_ones=2)

```
State space size:          2^10 = 1024 unique bitstrings
State space (with time):   1024 × 20 = 20,480 (state, step_count) pairs
Action space:              10 (always all legal)
Episode length:            T_max = 20 (dense mode)
Optimal episode length:    8 steps (flip 8 zeros to 1)
Optimal return (OneMax):   0.8
MCTS simulations/step:     20
NN forward passes/iter:    ~8000  (20 sims × 10 steps × 40 games)
Training examples/iter:    ~400   (40 games × ~10 steps)
Replay buffer size:        ~2000  (5 iterations × 400)
Network parameters:        ~36K
```

---

## 12. Key Structural Properties of the BitString MDP

This section characterizes the problem from an RL-theoretic perspective, providing the information needed to analyze how different algorithms would perform.

### 12.1. State space structure

- **Finite, small**: 2^N states (1024 for N=10). Tabular methods are feasible.
- **Bit-permutation symmetry (OneMax only)**: Under OneMax, states with the same Hamming weight are equivalent. There are only N+1 equivalence classes.
- **No symmetry (LeadingOnes, BinVal)**: Bit position matters. Full state representation needed.

### 12.2. Transition structure

- **Deterministic**: Given (state, action), next state is uniquely determined.
- **Reversible**: Every action can be undone by taking the same action again.
- **Self-loops impossible**: Flipping a bit always changes the state.
- **State graph**: The N-dimensional hypercube Q_N. Every state has exactly N neighbors.

### 12.3. Reward structure

| Property | OneMax Dense | OneMax Shaped | LeadingOnes Shaped | BinVal Shaped | Sparse |
|----------|-------------|---------------|-------------------|---------------|--------|
| Reward per step | ±1/N | ±1/N | 0 or ±1/N | ±2^k/N | 0 (terminal only) |
| Reward range | [-1/N, +1/N] | [-1/N, +1/N] | [-1/N, +1/N] | [-2^(N-1)/N, +2^(N-1)/N] | [0, 1] |
| Signal density | Every step | Every step | Sparse (many 0s) | Every step | Terminal only |
| Credit assignment | Easy | Easy | Hard | Medium | Very hard |

### 12.4. Optimal policy complexity

**OneMax:** The optimal policy is trivially simple: "flip any 0-bit." This can be represented as:
- A lookup table of size 2^N
- A linear function: π(a|s) ∝ (1 - s[a])
- A single-layer neural network

**LeadingOnes:** The optimal policy is more complex: "find the first 0-bit from the left, flip it." This requires:
- Detecting the position of the first 0 in a prefix scan
- Cannot be represented by a single linear layer
- Requires at least the capacity of a 1-hidden-layer network

**BinVal:** The greedy-optimal policy is "flip the highest-order 0-bit first." Under γ=1.0, any order of only-0→1 flips is equally optimal for total return.

### 12.5. Exploration requirements

- **OneMax (dense)**: Minimal exploration needed. The reward gradient points directly toward the goal. Even a random policy that slightly favors 0→1 flips will learn.
- **OneMax (sparse)**: Moderate exploration. Must complete the episode to get any signal. Random play with bit_flip=True may solve some instances by luck (probability decreases with N).
- **LeadingOnes**: High exploration needed. Many actions give zero reward. The agent must discover that bit order matters through trial and error.
- **BinVal**: Moderate exploration. The reward signal is present but wildly skewed toward high-order bits.

### 12.6. Value function properties

For OneMax with dense shaping and γ=1.0:
- V*(s) = (N - onemax(s)) / N = (number of remaining 0-bits) / N
- V* is a simple linear function of the state's Hamming weight
- A single-layer neural network can represent V* perfectly

For LeadingOnes:
- V*(s) depends on the positions of 0-bits, not just their count
- V* is not a linear function of the state
- Requires nonlinear function approximation

### 12.7. Sample complexity considerations

- **Tabular**: 1024 states × 10 actions = 10,240 (state, action) pairs. With enough exploration, tabular Q-learning should converge.
- **Function approximation**: The state is a 10-dimensional binary vector. A small MLP should easily represent the value function and optimal policy.
- **AlphaZero overhead**: MCTS adds 20 simulations per action selection. For N=10, this means ~200 network queries per episode, versus 1 for vanilla policy gradient.

---

## 13. Open Questions for RL Algorithm Comparison

The following questions are designed to guide a rigorous analysis of how various RL algorithms would perform on the BitString game. Each question identifies a specific algorithmic property and its interaction with the problem structure.

### Algorithmic suitability

1. **Is MCTS necessary for this problem?** The state space is small (1024 states), the action space is small (10 actions), and the transitions are deterministic. Standard RL algorithms (Q-learning, REINFORCE, PPO, SAC) can solve much larger problems. What does MCTS add here that a simpler algorithm cannot provide?

2. **Is the network architecture appropriate?** The optimal policy for OneMax is linear in the state (π(a) ∝ 1 - s[a]). A 36K-parameter MLP with 3 hidden layers is massively overparameterized. Would a simpler network (even a linear policy) converge faster?

3. **Is the self-play loop necessary?** AlphaZero's self-play loop is designed for competitive two-player games where the opponent improves over time. In a single-player optimization problem, there is no opponent. Is the self-play framing introducing unnecessary complexity?

### Comparative analysis dimensions

For each candidate algorithm, consider:

| Dimension | Question |
|-----------|----------|
| **Sample efficiency** | How many environment steps to learn the optimal policy? |
| **Computational cost** | Wall-clock time per training step (considering MCTS overhead)? |
| **Reward signal** | Can the algorithm work with sparse/dense/shaped rewards? |
| **Exploration** | How does the algorithm handle the exploration challenge for each potential function? |
| **Scalability** | How does performance change with N (state space grows as 2^N)? |
| **Optimality** | Does the algorithm converge to the true optimal policy? |
| **Simplicity** | Implementation complexity and number of hyperparameters? |

### Candidate algorithms to compare

| Algorithm | Category | Key properties for this problem |
|-----------|----------|-------------------------------|
| **Tabular Q-learning** | Model-free, value-based | Exact; no function approx needed for small N. O(2^N × N) table. |
| **DQN** | Model-free, value-based | Neural network Q-function. Experience replay. Handles the binary observation easily. |
| **REINFORCE** | Model-free, policy gradient | Direct policy optimization. High variance for sparse reward. Works well with dense shaping. |
| **PPO** | Model-free, policy gradient | Clipped surrogate objective. Stable training. Widely used baseline. |
| **SAC** (discrete) | Model-free, actor-critic | Entropy-regularized. Automatic exploration. May be overkill for discrete actions. |
| **AlphaZero (current)** | MCTS + neural network | Combines tree search with learned prior/value. High computational cost per step. |
| **Pure MCTS** | Tree search, no learning | Uniform policy, no neural network. Relies on leaf evaluations. Already implemented as a baseline. |
| **Random search** | Brute force | Sample random policies and evaluate. Baseline for comparison. |
| **Greedy oracle** | Hand-coded | Flip the first 0-bit found. Optimal for OneMax. Useful as an upper bound. |

### Specific questions for the LLM analysis

1. **For OneMax with dense shaping**: What is the simplest algorithm that achieves optimal performance? Is tabular Q-learning sufficient? Does it converge faster than AlphaZero?

2. **For LeadingOnes with dense shaping**: How do different algorithms handle the plateau problem? Does PPO's entropy bonus help exploration compared to MCTS's Dirichlet noise? Would curiosity-driven exploration (ICM, RND) help?

3. **For sparse reward**: Which algorithms can learn at all without reward shaping? Does AlphaZero's tree search provide any advantage for credit assignment over temporal-difference methods?

4. **For scaling to larger N**: At what N does tabular Q-learning become infeasible? Does AlphaZero's sample efficiency advantage (if any) increase with N? At what N does the MCTS simulation budget (20) become insufficient?

5. **For the specific AlphaZero implementation**:
   - Is the 2× policy weight hurting value learning?
   - Is 20 simulations per step sufficient for meaningful search in a 10-action space?
   - Does the gating mechanism (55% threshold over 20 games) provide meaningful quality control?
   - Would removing MCTS and training the network directly via policy gradient be faster and equally effective?

6. **Hyperparameter sensitivity**: Which hyperparameters matter most for each algorithm? Is the current AlphaZero configuration well-tuned, or is performance sensitive to `n_simulations`, `c_exploration`, `learning_rate`, etc.?

7. **Theoretical analysis**: For the OneMax problem with dense shaping, can we derive the expected convergence rate for tabular Q-learning? For REINFORCE? How does the AlphaZero convergence rate compare?

---

## Appendix: Configuration Wiring (how components connect)

```python
# src/alphazeropp/instances/bitstring/config.py — BitStringConfig.build()

def build(self):
    game_kwargs = dict(self.game.kwargs)
    fitness_fn_name = game_kwargs.pop("fitness_fn", None)
    reward_mode = game_kwargs.pop("reward_mode", "dense_potential")

    if fitness_fn_name is not None:
        base_env = BitStringGym(**game_kwargs)
        shaped_env = ShapedBitStringGym(
            base_env, POTENTIAL_REGISTRY[fitness_fn_name], reward_mode)
        game = BitStringGame(env=shaped_env)
    else:
        game = BitStringGame(**game_kwargs)

    net = BitStringPolicyValueNet(**self.net.kwargs)
    agent = Agent(game=game, net=net, mcts_params=self.agent.mcts_params,
                  reward_discount=self.agent.reward_discount,
                  random_seeds=self.agent.random_seeds)
    trainer = Trainer(agent=agent, net=net, game=game,
                      n_games_per_train=self.trainer.n_games_per_train,
                      n_past_iterations_to_train=self.trainer.n_past_iterations_to_train,
                      n_procs=self.trainer.n_procs)
    evaluator = Evaluator(n_games=self.evaluator.n_games, n_procs=self.evaluator.n_procs)
    return game, net, agent, trainer, evaluator
```

**Dependency chain:**
```
BitStringGym → [ShapedBitStringGym] → BitStringGame → action_size(=N)
                                                     → PolicyValueNetModel(N, N)
                                                     → Agent(game, net, mcts_params)
                                                     → Trainer(agent, net, game)
                                                     → Evaluator()
                                                     → GatedTrainer(trainer, evaluator)
```
