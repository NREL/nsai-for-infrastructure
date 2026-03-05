# Doors Game Audit: Direct Play vs Grammar Game

**Date:** 2026-03-03
**Purpose:** Self-contained reference for designing experiments that compare two approaches to solving the Doors environment:
1. **Direct Play** — An RL agent observes the environment state and selects actions directly
2. **Grammar Game** — AlphaZero synthesizes a DSL program (decision list) that acts as a deterministic policy

This document contains all environment specs, code interfaces, existing infrastructure, gaps, and experiment proposals. An LLM can use this document alone to draft minimal working experiments.

---

## 1. The Doors Environment

### 1.1 PDDL Domain (Simplified)

The environment models rooms gated by locked doors. An agent must collect keys to unlock doors and navigate to a goal location.

- **Rooms** r=0..D-1. Room 0 is always unlocked; others start locked.
- **Locations** l=0..M-1. Each location belongs to a room (`loc_room[l]`).
- **Keys** k=0..D-2. Key k is at `key_loc[k]` and unlocks room `key_unlocks[k]`.
- **MOVE_TO(l)**: succeeds iff `unlocked[loc_room[l]]`. Sets agent location to l.
- **PICK(k)**: succeeds iff `at_loc[key_loc[k]] AND key_available[k]`. Consumes key, unlocks room.

### 1.2 Default Layout (D=2, M=4)

```python
_LAYOUT_D2 = dict(
    num_rooms=2,
    loc_room=[0, 0, 1, 1],   # 4 locations: locs 0,1 in room 0; locs 2,3 in room 1
    key_loc=[1],              # key 0 is at location 1
    key_unlocks=[1],          # key 0 unlocks room 1
    start_loc=0,              # agent starts at location 0
    goal_loc=3,               # goal is location 3
    horizon=15,               # max 15 steps per episode
)
```

```
Room 0 (unlocked)          Room 1 (locked)
┌──────────────────┐       ┌──────────────────┐
│  loc 0 (start)   │       │  loc 2           │
│  loc 1 (key_0)   │  ──→  │  loc 3 (goal)    │
└──────────────────┘       └──────────────────┘
        pick key_0 at loc 1 → unlocks room 1
```

### 1.3 State Vector (7 floats, float32)

```
Index  Name          Meaning                     Initial Value
─────  ────────────  ──────────────────────────  ─────────────
0      at_loc[0]     Agent at location 0          1.0 (start)
1      at_loc[1]     Agent at location 1          0.0
2      at_loc[2]     Agent at location 2          0.0
3      at_loc[3]     Agent at location 3 (goal)   0.0
4      unlocked[0]   Room 0 lock status           1.0 (always)
5      unlocked[1]   Room 1 lock status           0.0 (locked)
6      key_avail[0]  Key 0 availability           1.0 (available)
```

**Initial state vector:** `[1, 0, 0, 0, 1, 0, 1]`

The `at_loc` bits are one-hot (exactly one is 1 at any time). The formula `obs_size = M + 2D - 1` gives 4 + 4 - 1 = 7.

### 1.4 Action Space (7 discrete)

```
Index  Action      Precondition                          Effect
─────  ──────────  ────────────────────────────────────  ──────────────────
0      MOVE_TO(0)  unlocked[loc_room[0]] = unlocked[0]   Set at_loc one-hot to 0
1      MOVE_TO(1)  unlocked[loc_room[1]] = unlocked[0]   Set at_loc one-hot to 1
2      MOVE_TO(2)  unlocked[loc_room[2]] = unlocked[1]   Set at_loc one-hot to 2
3      MOVE_TO(3)  unlocked[loc_room[3]] = unlocked[1]   Set at_loc one-hot to 3
4      PICK(0)     at_loc[1]==1 AND key_avail[0]==1       key_avail[0]=0, unlocked[1]=1
5      NOOP        always                                 No change
6      (invalid)   always                                 Treated as NOOP
```

Failed preconditions → no-op (no exception). The action space is padded to `obs_size` for alignment with the DSL grammar (`Flip(i)` uses indices 0..n_sites-1).

### 1.5 Rewards

| Event | Reward |
|-------|--------|
| Each step | -0.01 (step penalty) |
| Unlock a new room (0→1 flip) | +0.1 |
| Reach goal location | +1.0 |

### 1.6 Terminal Conditions

- **Solved:** `at_loc[goal_loc] == 1` → `terminated=True`
- **Timeout:** `step_count >= horizon` → `truncated=True`

### 1.7 Optimal Play

```
Step 1: MOVE_TO(1)  — move to key location  → reward = -0.01
Step 2: PICK(0)     — pick key, unlock room 1 → reward = -0.01 + 0.1 = +0.09
Step 3: MOVE_TO(3)  — move to goal           → reward = -0.01 + 1.0 = +0.99
                                         Total reward = 1.07
```

3 steps, cumulative reward ≈ 1.07.

### 1.8 DoorsPDDLLiteEnv Class Interface

**File:** `src/alphazeropp/instances/bitstring/envs/doors_pddl_lite.py`

```python
class DoorsPDDLLiteEnv(gym.Env):
    """PDDL-faithful doors environment with fixed-size numpy observations."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_rooms=2, loc_room=None, key_loc=None,
                 key_unlocks=None, start_loc=0, goal_loc=None,
                 horizon=20, step_penalty=0.01, unlock_bonus=0.1,
                 frozen_states=None):
        self.D = num_rooms
        self.M = len(loc_room)          # number of locations
        self.K = self.D - 1             # number of keys
        self._obs_size = self.M + 2 * self.D - 1
        self.observation_space = spaces.MultiBinary(self._obs_size)
        self.action_space = spaces.Discrete(self._obs_size)

    @property
    def n_sites(self) -> int:
        return self._obs_size           # 7 for D=2

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy().astype(np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset to initial state (or next frozen state if cycling)."""
        # Agent at start_loc, room 0 unlocked, all keys available
        # Returns (obs, {})

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with PDDL-faithful transitions.
        Returns (obs, reward, terminated, truncated, info)"""

    def is_solved(self, obs: np.ndarray) -> bool:
        return bool(obs[self.goal_loc] == 1.0)

    def reset_frozen_index(self):
        """Reset frozen-state cycling index to 0."""

    @property
    def feature_spec(self) -> list[FeatureSpec]:
        """Semantic mapping: obs index → predicate name."""

    @property
    def action_spec(self) -> list[ActionSpec]:
        """Semantic mapping: action index → action name."""

    @classmethod
    def make_d2(cls, **kwargs) -> DoorsPDDLLiteEnv:
        """D=2, M=4 preset."""

    @classmethod
    def make_d3(cls, **kwargs) -> DoorsPDDLLiteEnv:
        """D=3, M=6 preset."""
```

---

## 2. The DSL (Domain-Specific Language)

### 2.1 AST Nodes

**File:** `src/alphazeropp/instances/bitstring/dsl/ast_nodes.py`

Programs are decision lists with first-match semantics. Every program terminates with a `Default` node.

```python
from dataclasses import dataclass
from typing import Union

Condition = Union["IsZero", "Not", "And"]
Program = Union["Ite", "Default"]

# --- Actions ---
@dataclass(frozen=True)
class Flip:
    index: int    # 0..n_sites-1, maps directly to env action index
    def node_count(self) -> int: return 1
    def pretty(self) -> str: return f"Flip({self.index})"

# --- Conditions ---
@dataclass(frozen=True)
class IsZero:
    index: int    # True iff state[index] == 0
    def node_count(self) -> int: return 1

@dataclass(frozen=True)
class Not:
    child: Condition    # logical negation
    def node_count(self) -> int: return 1 + self.child.node_count()

@dataclass(frozen=True)
class And:
    left: Condition
    right: Condition    # conjunction
    def node_count(self) -> int: return 1 + self.left.node_count() + self.right.node_count()

# --- Programs ---
@dataclass(frozen=True)
class Ite:
    cond: Condition     # if cond matches → return action
    action: Flip
    else_prog: Program  # otherwise continue to next rule
    def node_count(self) -> int:
        return 1 + self.cond.node_count() + self.action.node_count() + self.else_prog.node_count()

@dataclass(frozen=True)
class Default:
    action: Flip        # always-fire fallback (terminal)
    def node_count(self) -> int: return 1 + self.action.node_count()
```

### 2.2 Interpreter

**File:** `src/alphazeropp/instances/bitstring/dsl/interpreter.py`

```python
def eval_condition(cond: Condition, state: np.ndarray) -> bool:
    """Evaluate a condition on a state vector."""
    if isinstance(cond, IsZero):
        return bool(state[cond.index] == 0)
    elif isinstance(cond, Not):
        return not eval_condition(cond.child, state)
    elif isinstance(cond, And):
        return eval_condition(cond.left, state) and eval_condition(cond.right, state)

def eval_program(program: Program, state: np.ndarray) -> int:
    """Walk decision list, return first matching action index."""
    if isinstance(program, Default):
        return program.action.index
    elif isinstance(program, Ite):
        if eval_condition(program.cond, state):
            return program.action.index
        return eval_program(program.else_prog, state)

@dataclass
class EpisodeResult:
    steps: list[StepRecord]
    total_env_steps: int
    total_interp_ops: int
    final_state: np.ndarray
    cumulative_reward: float
    solved: bool

def run_policy_episode(env, program, x0=None, verbose=False,
                       is_solved=None) -> EpisodeResult:
    """Execute program as a reactive policy on the environment.
    Each step: eval_program(program, obs) → action → env.step(action).
    Loops until terminated or truncated.
    solved = is_solved(obs) if provided, else np.all(obs == 1.0)."""
    obs, _ = env.reset()
    if x0 is not None:
        env.state = x0.copy()
        obs = x0.copy()
    steps, total_interp, cumulative_reward = [], 0, 0.0
    done = False
    while not done:
        action = eval_program(program, obs)
        ops = interp_ops(program, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_interp += ops
        cumulative_reward += reward
        done = terminated or truncated
    return EpisodeResult(steps, step_num, total_interp, obs.copy(),
                         cumulative_reward,
                         is_solved(obs) if is_solved else bool(np.all(obs == 1.0)))
```

### 2.3 Optimal D=2 Doors Program

This 16-node decision list solves D=2 in 3 steps:

```
if And(Not(IsZero(1)), Not(IsZero(6))):   # at key loc (1=1) AND key available (6=1)
  Flip(4)                                  # → PICK(0)
elif Not(IsZero(6)):                       # key still available (6=1)
  Flip(1)                                  # → MOVE_TO(1) go to key location
elif IsZero(3):                            # not at goal (3=0)
  Flip(3)                                  # → MOVE_TO(3) go to goal
else:
  Flip(5)                                  # → NOOP
```

**Why `And` is required:** Without `And`, a rule like `if Not(IsZero(1)): Flip(4)` fires whenever the agent is at location 1 — even after the key has been consumed. The agent gets stuck calling PICK(0) as a no-op forever. With `And(Not(IsZero(1)), Not(IsZero(6)))`, the rule only fires when BOTH the agent is at the key location AND the key is still available.

### 2.4 DSL ↔ Environment Mapping

| DSL Construct | Environment Meaning |
|---------------|---------------------|
| `IsZero(i)` | `state[i] == 0` (check any obs bit) |
| `Not(IsZero(i))` | `state[i] == 1` |
| `And(c1, c2)` | Both conditions true |
| `Flip(0..3)` | `MOVE_TO(location)` |
| `Flip(4)` | `PICK(0)` |
| `Flip(5)` | `NOOP` |
| `Flip(6)` | Invalid → NOOP |

---

## 3. The AlphaZero Pipeline

The codebase implements a complete AlphaZero training loop that can be applied to any single-player game. Here are all the components.

### 3.1 Game Interface

**File:** `src/alphazeropp/core/game.py`

```python
class Game(ABC):
    """Abstract base class for single-player games."""
    obs: ObsType | None
    reward: float | None
    terminated: bool | None
    truncated: bool | None
    info: dict | None
    step_count: int | None

    action_space: Space       # gymnasium.spaces.Discrete
    observation_space: Space  # gymnasium.spaces.Box or MultiBinary

    @abstractmethod
    def step(self, action) -> (obs, reward, terminated, truncated, info): ...

    @abstractmethod
    def reset(self) -> (obs, info): ...

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """Boolean array of shape (action_space.n,). True = valid action."""

    def reset_wrapper(self, **kwargs):
        """Calls reset() and updates self.obs, self.step_count, etc."""

    def step_wrapper(self, action):
        """Calls step() and updates self.obs, self.reward, etc."""

    def clone(self) -> Game:
        """Independent copy. Default: deepcopy."""

    def stash_state(self) -> Any:
        """Snapshot for MCTS rollback. Default: deepcopy(self)."""

    def unstash_state(self, state) -> Game:
        """Restore from snapshot. Returns the restored game object."""

    @property
    def hashable_obs(self) -> Hashable:
        """Hashable key for MCTS tree node lookup. Default: obs.tobytes()."""

class EnvGame(Game):
    """Wraps a Gymnasium Env as a Game. Delegates step/reset to env.
    Requires env to implement get_action_mask()."""
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    def step(self, action): return self.env.step(action)
    def reset(self, **kw): return self.env.reset(**kw)
    def get_action_mask(self): return self.env.get_action_mask()
```

### 3.2 PolicyValueNet Interface

**File:** `src/alphazeropp/core/policy_value_net.py`

```python
class PolicyValueNet(ABC):
    @abstractmethod
    def predict(self, state) -> tuple[np.ndarray, float]:
        """Given observation, return (policy_logits, value_scalar).
        Policy array has shape (action_space.n,)."""

    @abstractmethod
    def train(self, examples: list[tuple]):
        """Train on list of (obs, policy_probs, discounted_reward) tuples.
        Returns: (model, batch_losses, epoch_losses, policy_losses, value_losses)"""

    @abstractmethod
    def save_checkpoint(self, save_dir): ...
    @abstractmethod
    def load_checkpoint(self, save_dir): ...
    @abstractmethod
    def push_multiprocessing(self): ...     # Move tensors GPU → CPU
    @abstractmethod
    def pop_multiprocessing(self, *a): ...  # Restore GPU after
```

### 3.3 MCTS

**File:** `src/alphazeropp/core/mcts.py`

```python
class MCTSTreeNode:
    direct_reward: float
    is_terminal_state: bool
    nn_policy: np.ndarray         # Policy prior (possibly noise-injected)
    nn_policy_original: np.ndarray
    nn_value: float
    action_mask: np.ndarray       # Bool mask from game.get_action_mask()
    total_N: int                  # Total visits
    action_Q: dict[tuple, float]  # Q-value per action
    action_N: dict[tuple, int]    # Visit count per action

class MCTS:
    def __init__(self, game: Game, net: PolicyValueNet,
                 n_simulations=25, temperature=1.0,
                 c_exploration=1.0, dirichlet_alpha=0.3,
                 dirichlet_epsilon=0.25):
        self.nodes = {}  # maps game.hashable_obs → MCTSTreeNode

    def perform_simulations(self, msg="", add_noise=False) -> np.ndarray:
        """Run n_simulations from current game state.
        Each simulation:
          1. Stash game state
          2. Recursive UCB descent (select, expand, backup)
          3. Unstash to restore root
        After all sims: convert visit counts → action probs via temperature.
        Returns: action_probs array of shape (action_space.n,)."""

    def perform_simulations_reuse(self, msg="", add_noise=False) -> np.ndarray:
        """Same but reuses existing tree (for tree-reuse across moves)."""

    def advance_to(self, action):
        """Step game forward by action. Shift tree root to child node."""

    def search(self, msg="") -> float:
        """Recursive: UCB action selection → step → search(child) → backup."""
        # At leaf: query net.predict(obs) → expand node → return value
        # At internal: UCB = Q(s,a) + c * P(s,a) * sqrt(N) / (1 + N(s,a))
        # Select best valid action (invalid actions get -inf)
```

### 3.4 Agent

**File:** `src/alphazeropp/core/agent.py`

```python
class Agent:
    def __init__(self, game: Game, net: PolicyValueNet,
                 mcts_params: dict, reward_discount=1.0,
                 random_seeds={}):
        self.game = game
        self.net = net
        self.mcts_params = mcts_params
        self.reward_discount = reward_discount

    def policy(self, game_state, msg="", add_noise=True,
               temperature_override=None) -> np.ndarray:
        """Create fresh MCTS for game_state, run simulations, return probs."""
        mcts = MCTS(game_state.clone(), self.net, **self.mcts_params)
        if temperature_override: mcts.temperature = temperature_override
        return mcts.perform_simulations(msg, add_noise=add_noise)

    def play_one_round(self, game, max_moves=10000, random_seed=None,
                       add_noise=True, temperature_override=None):
        """Play one full episode, collecting training data.
        Returns: (experience, cumulative_reward)
        experience = [(obs, move_probs, discounted_reward), ...]

        Algorithm:
          1. Clone game
          2. Loop until done:
             a. move_probs = self.policy(game_state)
             b. action = sample from move_probs
             c. Collect (obs, move_probs)
             d. game.step_wrapper(action)
             e. Collect reward
          3. Calculate discounted rewards backwards:
             for reward in reversed(rewards):
               G = reward + discount * G
          4. Zip into [(obs, probs, G), ...]
        """

    def play_one_round_reuse_tree(self, game, ...):
        """Same but creates ONE MCTS tree, reuses across all moves.
        Uses mcts.perform_simulations_reuse() + mcts.advance_to(action)."""

    def play_for_experience(self, game, id, reset_seed, interaction_seed, ...):
        """Multiprocessing-safe: clone game, reset with seed, play_one_round."""
```

### 3.5 Trainer

**File:** `src/alphazeropp/training/trainer.py`

```python
class Trainer:
    def __init__(self, agent: Agent, net: PolicyValueNet, game: Game,
                 n_games_per_train=100,
                 n_past_iterations_to_train=20,
                 n_procs=None, use_tree_reuse=False):
        self.all_training_examples = []  # rolling buffer of past iterations

    def train_iteration(self):
        """One training iteration:
        1. Collect n_games_per_train episodes via self-play
           - Sequential (n_procs=-1) or multiprocessing
           - Each game → list of (obs, move_probs, discounted_reward)
        2. Accumulate examples in rolling buffer
           - Keep last n_past_iterations_to_train iterations
        3. Flatten all examples → net.train(flat_examples)
        """

    def train_multiple(self, n_iterations, start_at=0, checkpoint_every=None):
        """Run multiple train_iteration() calls with optional checkpointing."""
```

### 3.6 Evaluator

**File:** `src/alphazeropp/training/evaluator.py`

```python
EVAL_TEMPERATURE = 0.05  # Near-greedy for evaluation

class Evaluator:
    def __init__(self, n_games=20, n_procs=None): ...

    def pit(self, new_agent: Agent, old_agent: Agent) -> float:
        """Compare agents: play n_games from identical starts.
        Both agents play from the same reset seed, near-greedy (temp=0.05).
        Returns: win_rate = (wins + ties/2) / n_games."""
```

### 3.7 GatedTrainer

**File:** `src/alphazeropp/training/gated_trainer.py`

```python
class GatedTrainer:
    def __init__(self, trainer: Trainer, evaluator: Evaluator,
                 acceptance_threshold=0.55):
        ...

    def train_iteration(self) -> tuple[float, bool]:
        """Gated training iteration:
        1. Snapshot old agent (deepcopy)
        2. trainer.train_iteration() — modifies network weights in-place
        3. Snapshot new agent (deepcopy)
        4. score = evaluator.pit(new_agent, old_agent)
        5. If score < acceptance_threshold:
           Restore old weights via model.load_state_dict()
        Returns: (score, accepted)"""
```

### 3.8 Full Training Loop (as run in `scripts/run_derivation.py`)

```python
# Pseudocode of what happens when you run:
#   python scripts/run_derivation.py  (select mode 2 = doors)

cfg = DoorsDerivationConfig()
game, net, agent, trainer, evaluator = cfg.build()
gated_trainer = GatedTrainer(trainer, evaluator, acceptance_threshold=0.55)

for iteration in range(30):
    # 1. Self-play: 40 games, each using MCTS (200 sims/move)
    #    Agent plays DerivationGame: expand grammar holes → complete program
    #    Terminal reward = LeafEvaluator(program) = run program on doors env
    # 2. Train Transformer network on collected (obs, policy, value) tuples
    # 3. Pit new network vs old network (20 games)
    # 4. Accept if win_rate >= 55%, else restore old weights
    score, accepted = gated_trainer.train_iteration()
```

---

## 4. The Grammar Game (Approach B) — Complete Detail

### 4.1 DerivationGame

**File:** `src/alphazeropp/instances/bitstring/dsl/derivation_game.py`

The grammar game casts program synthesis as a single-player game where:
- **State** = partial AST with "holes" (unexpanded non-terminals)
- **Action** = grammar production (expand the leftmost hole)
- **Terminal** = complete program (no holes remain)
- **Reward** = 0 at non-terminals; `leaf_evaluator(program)` at terminal

```python
NODE_TYPE_IDS = {
    "PAD": 0, "Flip": 1, "IsZero": 2, "Not": 3, "And": 4,
    "Ite": 5, "Default": 6, "ProgramHole": 7, "ConditionHole": 8,
}

class DerivationGame(Game):
    def __init__(self, budget: int, n_sites: int,
                 leaf_evaluator: LeafEvaluator,
                 program_budget_mode="exact",
                 allow_and=True, allow_not=True):
        # Observation: preorder AST traversal as (type_id, param) pairs
        self.observation_space = Box(shape=(2 * budget,), dtype=float32)
        # Actions: legal grammar productions (varies per state)
        self._max_productions = compute_max_productions(budget, n_sites, ...)
        self.action_space = Discrete(self._max_productions)  # ~150 for budget=18

    def reset(self):
        self._deriv_state = DerivationState.initial(self.budget)  # ProgramHole(18)
        self._current_productions = self._deriv_state.legal_productions(
            self.n_sites, mode=self._mode,
            allow_and=self._allow_and, allow_not=self._allow_not)
        return self._encode_obs(), {}

    def step(self, action: int):
        prod = self._current_productions[action]
        self._deriv_state = self._deriv_state.apply(prod)
        self._current_productions = self._deriv_state.legal_productions(...)

        is_complete = self._deriv_state.is_terminal()
        is_dead_end = not is_complete and len(self._current_productions) == 0

        if is_complete:
            program = self._deriv_state.to_program()
            reward = self.leaf_evaluator(program)  # ← runs program on doors env!
            return obs, reward, True, False, {"program": program}
        elif is_dead_end:
            return obs, 0.0, False, True, {"dead_end": True}
        else:
            return obs, 0.0, False, False, {}

    def get_action_mask(self):
        mask = np.zeros(self._max_productions, dtype=bool)
        mask[:len(self._current_productions)] = True
        return mask

    def _encode_obs(self) -> np.ndarray:
        """Encode partial AST as fixed-size float32 array.
        Layout: (type_id, parameter) pairs in preorder, padded to 2*budget.
        Example: ProgramHole(18) → [7.0, 18.0, 0.0, 0.0, ..., 0.0]"""
        obs = np.zeros(2 * self.budget, dtype=np.float32)
        items = _preorder_items(self._deriv_state.root)
        for i, (type_id, param) in enumerate(items):
            if i >= self.budget: break
            obs[2 * i] = type_id
            obs[2 * i + 1] = param
        return obs

    # Efficient state management (no deepcopy needed — frozen dataclasses)
    def stash_state(self) -> tuple:
        return (self._deriv_state, self._current_productions,
                self.obs, self.reward, self.terminated, self.truncated,
                self.info, self.step_count)

    def unstash_state(self, state: tuple):
        (self._deriv_state, self._current_productions,
         self.obs, self.reward, self.terminated, self.truncated,
         self.info, self.step_count) = state
        return self

    def clone(self):
        new = DerivationGame(self.budget, self.n_sites, self.leaf_evaluator,
                             program_budget_mode=self._mode,
                             allow_and=self._allow_and, allow_not=self._allow_not)
        new.unstash_state(self.stash_state())
        if self.obs is not None: new.obs = self.obs.copy()
        return new
```

### 4.2 LeafEvaluator

**File:** `src/alphazeropp/instances/bitstring/dsl/leaf_evaluator.py`

Scores completed programs by executing them as policies on frozen initial states.

```python
VALID_METRICS = ("avg_reward", "solve_rate", "penalized_reward", "weighted")

class LeafEvaluator:
    def __init__(self, n_sites: int, frozen_states: list[np.ndarray],
                 game_config,    # DoorsGameConfig (duck-typed)
                 metric="avg_reward", penalty_lambda=0.1,
                 blend_alpha=0.5, is_solved=None):
        self._cache = {}           # program.pretty() → scalar value
        self._full_cache = {}      # program.pretty() → metrics dict
        self._program_cache = {}   # program.pretty() → Program AST

    def __call__(self, program: Program) -> float:
        """Evaluate program. Returns cached result if available."""
        key = program.pretty()
        if key in self._cache: return self._cache[key]
        metrics = self._evaluate(program)
        value = self._compute_metric(metrics)
        self._cache[key] = value
        self._full_cache[key] = metrics
        self._program_cache[key] = program
        return value

    def _evaluate(self, program) -> dict:
        """Run program on each frozen state, collect metrics."""
        for x0 in self.frozen_states:
            env = self.game_config.make_env(self.n_sites, frozen_states=[x0])
            env.reset()
            result = run_policy_episode(env, program, is_solved=self.is_solved)
            # Accumulate: solved_count, total_steps, total_ops, total_reward
        return {"solve_rate": solved/n, "avg_reward": reward/n,
                "avg_steps": steps/n, "avg_ops": ops/n}

    def _compute_metric(self, metrics) -> float:
        if self.metric == "avg_reward": return metrics["avg_reward"]
        if self.metric == "solve_rate": return metrics["solve_rate"]
        if self.metric == "penalized_reward":
            return metrics["avg_reward"] - self.penalty_lambda * metrics["avg_ops"] / self._max_ops
        if self.metric == "weighted":
            return self.blend_alpha * metrics["solve_rate"] + (1-self.blend_alpha) * metrics["avg_reward"]

    def get_all_metrics(self, program) -> dict: ...
    def stats(self) -> dict:
        return {"eval_count": ..., "cache_hits": ..., "unique_programs": ...,
                "total_env_steps": ..., "total_interp_ops": ...}
```

### 4.3 Transformer Network

**File:** `src/alphazeropp/instances/bitstring/dsl/derivation_network.py`

```python
class DerivationTransformerModel(nn.Module):
    """Reads DerivationGame observations, outputs policy + value."""

    def __init__(self, budget=18, n_sites=7, action_size=150,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        # Embeddings
        self.type_emb = nn.Embedding(9, d_model)          # 9 node type IDs
        self.param_proj = nn.Linear(1, d_model)            # parameter value
        self.pos_emb = nn.Embedding(budget + 1, d_model)   # positional
        self.cls_emb = nn.Parameter(torch.zeros(d_model))  # learned [CLS]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Output heads
        self.policy_head = nn.Linear(d_model, action_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, 2*budget)
        type_ids = x[:, 0::2].long()   # (B, budget)
        params = x[:, 1::2]            # (B, budget)
        # 1. Embed type + param, add positional
        # 2. Prepend [CLS] token
        # 3. Create padding mask (type_id==0 → masked)
        # 4. Transformer encoder
        # 5. Extract CLS output
        # 6. policy_logits = policy_head(cls), value = value_head(cls)
        return policy_logits, value
```

### 4.4 UniformPolicyValueNet (Pure MCTS Baseline)

```python
class UniformPolicyValueNet(PolicyValueNet):
    """Uniform-random policy, constant value=0.
    MCTS relies entirely on backed-up leaf values — no policy prior guidance."""

    def __init__(self, action_size: int):
        self.action_size = action_size

    def predict(self, state):
        policy = np.ones(self.action_size, dtype=np.float32) / self.action_size
        value = np.array(0.0, dtype=np.float32)
        return policy, value

    def train(self, examples): pass
    def save_checkpoint(self, d): pass
    def load_checkpoint(self, d): pass
    def push_multiprocessing(self): pass
    def pop_multiprocessing(self, *a): pass
```

---

## 5. Configuration Classes

### 5.1 DoorsGameConfig

**File:** `src/alphazeropp/instances/bitstring/dsl/doors_config.py`

Duck-typed interface matching `GameConfig` so `LeafEvaluator` works without modification.

```python
@dataclass
class DoorsGameConfig:
    num_rooms: int = 2
    loc_room: list[int] | None = None      # defaults to [0,0,1,1]
    key_loc: list[int] | None = None       # defaults to [1]
    key_unlocks: list[int] | None = None   # defaults to [1]
    start_loc: int = 0
    goal_loc: int | None = None            # defaults to M-1 = 3
    horizon: int = 15
    step_penalty: float = 0.01
    unlock_bonus: float = 0.1

    @property
    def M(self) -> int: return len(self.loc_room)    # 4
    @property
    def D(self) -> int: return self.num_rooms         # 2
    @property
    def K(self) -> int: return self.D - 1             # 1

    def obs_size(self) -> int: return self.M + 2*self.D - 1  # 7
    def max_steps(self, n_sites) -> int: return self.horizon  # 15
    def is_solved(self, obs) -> bool: return bool(obs[self.goal_loc] == 1.0)

    def make_env(self, n_sites, frozen_states=None) -> DoorsPDDLLiteEnv:
        return DoorsPDDLLiteEnv(
            num_rooms=self.num_rooms, loc_room=self.loc_room,
            key_loc=self.key_loc, key_unlocks=self.key_unlocks,
            start_loc=self.start_loc, goal_loc=self.goal_loc,
            horizon=self.horizon, step_penalty=self.step_penalty,
            unlock_bonus=self.unlock_bonus, frozen_states=frozen_states)

def doors_initial_state(config: DoorsGameConfig) -> np.ndarray:
    """Canonical initial state: [1,0,0,0, 1,0, 1]"""
    state = np.zeros(config.obs_size(), dtype=np.float32)
    state[config.start_loc] = 1.0           # at start location
    state[config.M] = 1.0                   # room 0 unlocked
    for k in range(config.K):
        state[config.M + config.D + k] = 1.0  # all keys available
    return state
```

### 5.2 DoorsDerivationConfig (Grammar Game)

**File:** `src/alphazeropp/instances/bitstring/dsl/derivation_config.py`

```python
class DoorsDerivationConfig(MetaConfig):
    """Full config for grammar game on doors environment."""

    def __init__(self):
        self.game = CoreGameConfig(
            game_cls=DerivationGame,
            kwargs={
                "budget": 18,                   # max AST nodes
                "n_sites": 7,                   # obs_size = M + 2D - 1
                "program_budget_mode": "max",   # programs use <= budget nodes
                "allow_and": True,              # And() enabled in grammar
                "allow_not": True,
                "num_rooms": 2, "horizon": 15,
                "step_penalty": 0.01, "unlock_bonus": 0.1,
                "metric": "weighted",           # 0.7*solve_rate + 0.3*avg_reward
                "blend_alpha": 0.7,
            })
        self.net = NetConfig(
            net_cls=DerivationPolicyValueNet,
            kwargs={
                "budget": 18, "n_sites": 7,
                "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1,
                "training_params": {
                    "epochs": 5, "batch_size": 32,
                    "learning_rate": 3e-4, "weight_decay": 1e-4,
                    "policy_weight": 2.0,
                }})
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 200, "temperature": 1.0,
                "c_exploration": 1.5, "dirichlet_alpha": 0.25,
                "dirichlet_epsilon": 0.40,
            },
            reward_discount=1.0)
        self.trainer = TrainerConfig(
            n_games_per_train=40,
            n_past_iterations_to_train=20,
            n_procs=-1)                  # sequential
        self.evaluator = EvaluatorConfig(n_games=20, n_procs=-1)
        self.run = RunConfig(n_iterations=30, accept_threshold=0.55)

    def build(self):
        # 1. DoorsGameConfig → doors_initial_state() → LeafEvaluator
        doors_cfg = DoorsGameConfig(num_rooms=2, horizon=15, ...)
        frozen_states = [doors_initial_state(doors_cfg)]
        leaf_eval = LeafEvaluator(7, frozen_states, doors_cfg,
                                   metric="weighted", blend_alpha=0.7,
                                   is_solved=doors_cfg.is_solved)
        # 2. DerivationGame
        game = DerivationGame(18, 7, leaf_eval,
                              program_budget_mode="max", allow_and=True)
        # 3. Network
        net = DerivationPolicyValueNet(budget=18, n_sites=7,
                                        action_size=game.action_space.n, ...)
        # 4. Agent, Trainer, Evaluator
        agent = Agent(game, net, mcts_params={...})
        trainer = Trainer(agent, net, game, n_games_per_train=40, ...)
        evaluator = Evaluator(n_games=20, ...)
        return game, net, agent, trainer, evaluator

class DoorsDerivationConfigNoAnd(DoorsDerivationConfig):
    """Baseline: same config but allow_and=False."""
    def __init__(self):
        super().__init__()
        self.game.kwargs["allow_and"] = False
```

---

## 6. Existing Scripts

### 6.1 Grammar Game Training

**File:** `scripts/run_derivation.py`

```bash
python scripts/run_derivation.py
# Interactive menu:
#   0) scan         — Priority-scan grammar
#   1) cfg          — Size-budget CFG grammar
#   2) doors        — Doors PDDL environment (And enabled)
#   3) doors_no_and — Doors PDDL environment (And disabled)
```

Select mode 2 (doors). The script:
1. Creates `DoorsDerivationConfig`
2. Interactive config editing (change any hyperparameter)
3. Builds game, net, agent, trainer, evaluator
4. Runs 30 iterations of gated AlphaZero training
5. Outputs to `experiments/derivation/<timestamp>_doors_*/`:
   - `config.json`, `train_stats.jsonl`, `eval_stats.jsonl`
   - `program_log.jsonl`, `metrics_*.png`

### 6.2 Expressivity Gap Estimation

**File:** `scripts/estimate_expressivity_gap.py`

Pure MCTS search (no neural network training) comparing `allow_and=True` vs `False`.

```python
def build_search_agent(allow_and, budget=18, n_simulations=200, num_rooms=2):
    doors_cfg = DoorsGameConfig(num_rooms=num_rooms)
    n_sites = doors_cfg.obs_size()   # 7
    frozen_states = [doors_initial_state(doors_cfg)]
    leaf_eval = LeafEvaluator(n_sites, frozen_states, doors_cfg,
                               metric="weighted", blend_alpha=0.7,
                               is_solved=doors_cfg.is_solved)
    game = DerivationGame(budget, n_sites, leaf_eval,
                          program_budget_mode="max", allow_and=allow_and)
    net = UniformPolicyValueNet(action_size=game.action_space.n)
    agent = Agent(game=game, net=net,
                  mcts_params={"n_simulations": n_simulations,
                               "temperature": 1.0, "c_exploration": 1.5,
                               "dirichlet_alpha": 0.25, "dirichlet_epsilon": 0.40})
    return agent, game, leaf_eval
```

```bash
python scripts/estimate_expressivity_gap.py --rounds 20 --sims 200
# Outputs: comparison table (solve_rate, avg_reward, unique_programs, time)
```

---

## 7. What's Missing for Direct Play

The codebase has **no direct-play agent** for DoorsPDDLLiteEnv. To compare "direct play vs grammar game", the following components are needed:

### 7.1 DoorsDirectGame

`DoorsPDDLLiteEnv` is a Gymnasium env but lacks `get_action_mask()`. The simplest approach:

```python
class DoorsDirectGame(EnvGame):
    """Wrap DoorsPDDLLiteEnv as an AlphaZero Game."""
    def __init__(self, env: DoorsPDDLLiteEnv):
        super().__init__(env)

    def get_action_mask(self):
        # All actions valid (env handles failed preconditions as noop)
        return np.ones(self.action_space.n, dtype=bool)
```

Or, for smarter masking, compute preconditions from current state.

### 7.2 DoorsDirectNet

A simple MLP for the 7-dim binary observation:

```python
class DoorsDirectNet(TorchPolicyValueNet):
    """MLP: obs(7) → hidden(64) → hidden(64) → policy(7) + value(1)"""
```

### 7.3 DoorsDirectConfig

Wire game + net + agent + trainer with matching MCTS parameters for fair comparison.

### 7.4 Experiment Harness

A script that runs both approaches and logs comparable metrics.

---

## 8. Experiment Proposals

### Experiment 1: Grammar Game with Training (RUNNABLE NOW)

```bash
python scripts/run_derivation.py  # select mode 2 (doors)
# 30 iterations, 40 games/iter, 200 MCTS sims/move
# Measure: best program solve_rate, avg_reward, unique programs, wall-clock time
```

**What to look for:** Does AlphaZero learn to synthesize the optimal 16-node program? How many iterations to reach solve_rate=1.0?

### Experiment 2: Grammar Game Pure Search (RUNNABLE NOW)

```bash
python scripts/estimate_expressivity_gap.py --rounds 20 --sims 200
# Pure MCTS with uniform policy (no learning)
# Compares And=True vs And=False
```

**What to look for:** Gap in best solve_rate between And variants. Does pure MCTS find solving programs?

### Experiment 3: Direct Play with AlphaZero (NEEDS ~150 LINES)

Create `DoorsDirectGame`, `DoorsDirectNet`, run same AlphaZero pipeline.

**Key comparison:** Same MCTS budget (200 sims/move), same number of training iterations (30), same number of games per iteration (40). Measure cumulative env steps and solve rate.

### Experiment 4: Tabular Q-Learning (NEEDS ~50 LINES)

The reachable state space is small (~20 states). Epsilon-greedy Q-learning:

```python
# Pseudocode
Q = defaultdict(lambda: np.zeros(7))
for episode in range(10000):
    obs = env.reset()
    while not done:
        if random() < epsilon: action = random_action()
        else: action = argmax(Q[obs.tobytes()])
        next_obs, reward, done, _, _ = env.step(action)
        Q[obs.tobytes()][action] += alpha * (reward + gamma * max(Q[next_obs.tobytes()]) - Q[obs.tobytes()][action])
        obs = next_obs
```

**What to look for:** How many episodes to converge? What's the converged solve rate?

### Experiment 5: Random Baselines (TRIVIAL)

```python
from alphazeropp.instances.bitstring.envs.doors_pddl_lite import DoorsPDDLLiteEnv

# Random action baseline
env = DoorsPDDLLiteEnv.make_d2()
solved = 0
for _ in range(10000):
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if env.is_solved(obs):
        solved += 1
print(f"Random action solve rate: {solved/10000:.1%}")
```

---

## 9. Comparison Framework

### 9.1 Metrics

| Metric | How to Measure | Direct Play | Grammar Game |
|--------|----------------|-------------|--------------|
| **Solve rate** | Fraction of episodes reaching goal | eval_games / total | leaf_eval solve_rate |
| **Avg reward** | Mean cumulative reward | from evaluator | from leaf_eval |
| **Sample efficiency** | Env steps to reach solve_rate ≥ X | count env steps in training | count env steps in leaf_eval |
| **Programs evaluated** | N/A for direct play | N/A | leaf_eval.stats()["unique_programs"] |
| **Wall-clock time** | time.time() around training loop | direct | direct |
| **Interpretability** | Can human verify policy? | No (NN weights) | Yes (DSL program) |
| **Generalization** | Test on unseen initial states | reset with different seeds | run_policy_episode with new x0 |

### 9.2 Apples-to-Apples Controls

For a fair comparison, hold constant:
- **MCTS simulations per move:** 200
- **Training iterations:** 30
- **Games per iteration:** 40
- **Evaluation games:** 20
- **Acceptance threshold:** 0.55
- **Environment:** DoorsPDDLLiteEnv D=2 with identical layout and rewards

Things that differ inherently:
- **Action space:** 7 (direct) vs ~150 (grammar)
- **Observation:** 7-dim binary (direct) vs 36-dim AST encoding (grammar)
- **Episode length:** up to 15 env steps (direct) vs ~10-15 derivation steps (grammar)
- **Reward structure:** shaped per-step (direct) vs sparse terminal (grammar)

### 9.3 Key Question

> Given the same computational budget (MCTS simulations × training iterations × games), which approach reaches solve_rate=1.0 faster?

The grammar game has a harder search problem (larger action space, sparse reward) but produces an interpretable, verifiable program. Direct play has a smaller action space with shaped rewards but produces opaque weights.

---

## 10. File Inventory

```
src/alphazeropp/
├── core/
│   ├── game.py                 # Game, EnvGame base classes
│   ├── agent.py                # Agent (MCTS + self-play)
│   ├── mcts.py                 # MCTS tree search
│   ├── policy_value_net.py     # PolicyValueNet ABC + TorchPolicyValueNet
│   └── config.py               # MetaConfig, GameConfig, NetConfig, etc.
├── training/
│   ├── trainer.py              # Trainer (collect + train loop)
│   ├── evaluator.py            # Evaluator (pit new vs old)
│   └── gated_trainer.py        # GatedTrainer (accept/reject gate)
├── instances/bitstring/
│   ├── envs/
│   │   └── doors_pddl_lite.py  # DoorsPDDLLiteEnv
│   └── dsl/
│       ├── ast_nodes.py        # Flip, IsZero, Not, And, Ite, Default
│       ├── interpreter.py      # eval_program, run_policy_episode
│       ├── derivation.py       # DerivationState, grammar productions
│       ├── derivation_game.py  # DerivationGame, UniformPolicyValueNet
│       ├── derivation_network.py # DerivationPolicyValueNet (Transformer)
│       ├── leaf_evaluator.py   # LeafEvaluator (scores programs)
│       ├── doors_config.py     # DoorsGameConfig, doors_initial_state
│       ├── derivation_config.py # DoorsDerivationConfig, DoorsDerivationConfigNoAnd
│       └── budget_grammar.py   # Grammar utilities, count_programs
└── utils/
    ├── interactive_config.py   # Interactive config editing
    ├── multiprocessing.py      # MultiprocessingManager
    ├── checkpoint.py           # CheckpointManager
    └── statistics.py           # StatisticsManager

scripts/
├── run_derivation.py           # Training entry point (modes: scan, cfg, doors, doors_no_and)
└── estimate_expressivity_gap.py # Pure MCTS gap estimation

tests/
└── test_doors_pddl_lite.py     # 23 tests for doors environment + grammar
```
