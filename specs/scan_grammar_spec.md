# Priority-Scan Grammar for the Derivation Game

## Context

The current size-budget CFG grammar searches through 151M+ programs (at L=14, N=6) where the branching factor is 48 and optimal programs are unreachable for N>=7. The analysis in `specs/cfg_redesign.md` Proposal A demonstrates that for all three bitstring variants (OneMax, LeadingOnes, BinVal), the optimal policy is a **priority ordering** over bit indices. This motivates replacing the grammar with one that constructs permutations directly, reducing the search space to N! (720 for N=6) with branching factor N.

The goal: add a new "scan" grammar as the **default grammar**, alongside the existing CFG grammar which remains available as an opt-in alternative. No existing code is deleted or renamed.

---

## Critical Evaluation of the Original Plan

The original plan had several issues against the actual codebase:

1. **Wrong paths**: References `src/nsai_experiments/` — the project is at `src/alphazeropp/`
2. **Unnecessary "rename legacy" step**: The existing code doesn't need renaming. The new grammar is a separate module that coexists naturally.
3. **Missing Program conversion**: The existing `LeafEvaluator` accepts `Program` AST objects. The plan never addresses how a permutation becomes a `Program`. Solution: a `permutation_to_program(perm, n_sites) -> Program` function that converts to the same AST types (Ite/IsZero/Flip/Default), allowing full LeafEvaluator reuse.
4. **"Intermediate value estimate" is a red herring**: The framework doesn't use `info` dict for value estimation — the value head does that. Skip this.
5. **"M random initial states" contradicts existing design**: The existing system uses deterministic frozen states (all C(n,k) initial states), not random samples. Reuse the same mechanism.
6. **Overcomplicated state**: `remaining_mask` is redundant with `get_action_mask()`. State is just the prefix.
7. **Deterministic completion (`comp()`) is unnecessary for terminal reward**: The framework already gives reward=0 at non-terminal steps and evaluates only at terminal. No need for intermediate evaluation via `comp()`. The value head learns to estimate partial-permutation value.

---

## Implementation Plan

### File Map (existing, to be read but not modified)

| File | Role |
|------|------|
| `src/alphazeropp/core/game.py` | Game base class (interface contract) |
| `src/alphazeropp/core/mcts.py` | MCTS with action masking, tree reuse |
| `src/alphazeropp/core/agent.py` | Agent.play_one_round, experience collection |
| `src/alphazeropp/core/policy_value_net.py` | PolicyValueNet / TorchPolicyValueNet base |
| `src/alphazeropp/instances/bitstring/dsl/derivation_game.py` | Existing DerivationGame |
| `src/alphazeropp/instances/bitstring/dsl/derivation.py` | Existing grammar (Production, DerivationState) |
| `src/alphazeropp/instances/bitstring/dsl/ast_nodes.py` | Ite, Default, Flip, IsZero, etc. |
| `src/alphazeropp/instances/bitstring/dsl/leaf_evaluator.py` | LeafEvaluator (evaluates Program on frozen states) |
| `src/alphazeropp/instances/bitstring/dsl/derivation_config.py` | DerivationConfig |
| `src/alphazeropp/instances/bitstring/dsl/derivation_network.py` | DerivationPolicyValueNet (Transformer) |
| `scripts/run_derivation.py` | Training script entrypoint |

### New Files

#### 1. `src/alphazeropp/instances/bitstring/dsl/scan_grammar.py`

Core scan grammar module. Contains:

**`permutation_to_program(perm: list[int]) -> Program`**
- Converts a permutation `[σ(0), σ(1), ..., σ(N-1)]` to an equivalent Program AST:
  ```
  Ite(IsZero(σ(0)), Flip(σ(0)),
    Ite(IsZero(σ(1)), Flip(σ(1)),
      ...
        Ite(IsZero(σ(N-2)), Flip(σ(N-2)),
          Default(Flip(σ(N-1))))))
  ```
- This is the key bridge: same AST type → existing LeafEvaluator works unchanged.

**`ScanState`** (dataclass)
- `prefix: tuple[int, ...]` — chosen indices so far (immutable for stash/unstash)
- `n_sites: int`
- `remaining: frozenset[int]` — indices not yet chosen
- Methods: `is_terminal()`, `legal_actions() -> list[int]`, `apply(action) -> ScanState`, `to_program() -> Program` (calls `permutation_to_program`), `pretty() -> str`
- Terminal when `len(prefix) == n_sites` (or `n_sites - 1` if we force the last element)

The last index is forced (only 1 remaining), so the game has `N-1` real decision steps with branching factor `N, N-1, ..., 2`.

#### 2. `src/alphazeropp/instances/bitstring/dsl/scan_derivation_game.py`

Implements `Game` interface. Similar structure to the existing `DerivationGame`:

```python
class ScanDerivationGame(Game):
    def __init__(self, n_sites, leaf_evaluator):
        self.action_space = spaces.Discrete(n_sites)  # actions are bit indices [0, N)
        self.observation_space = spaces.Box(shape=(2 * n_sites,), ...)

    def reset(self) -> (obs, info): ...
    def step(self, action) -> (obs, reward, terminated, truncated, info): ...
    def get_action_mask(self) -> np.ndarray:  # True for indices in remaining set
    def hashable_obs -> str:  # e.g. "Scan(3,1,_,_,_,_)"
    def stash_state() / unstash_state(): ...  # lightweight, ScanState is immutable
    def clone(): ...
```

**Action semantics**: action `i` means "pick bit index `i` as the next priority". The action mask has `True` at positions in `remaining`.

**Observation encoding**: flat array of length `2 * n_sites`:
- First N slots: prefix padded with -1 (e.g., `[3, 1, -1, -1, -1, -1]`)
- Next N slots: remaining mask as 0/1 (e.g., `[1, 0, 1, 0, 1, 1]`)

**Terminal**: when all indices are chosen, convert to Program via `permutation_to_program()`, evaluate with `leaf_evaluator(program)`, return reward.

**No dead ends**: every non-terminal state has at least 1 legal action (remaining is non-empty).

#### 3. `src/alphazeropp/instances/bitstring/dsl/scan_network.py`

Simple MLP-based PolicyValueNet (no Transformer needed — observation is a flat fixed-size vector):

```python
class ScanPolicyValueNet(TorchPolicyValueNet):
    def __init__(self, n_sites, d_hidden=128, n_hidden_layers=2, training_params={}): ...
```

Uses the existing `PolicyValueNetModel` pattern from `core/policy_value_net.py`:
- Input: `2 * n_sites` floats
- Body: MLP with ReLU
- Policy head: `Linear(d_hidden, n_sites)` — logits over bit indices
- Value head: `Linear(d_hidden, 1)` — scalar value

### Modified Files

#### 4. `src/alphazeropp/instances/bitstring/dsl/derivation_config.py`

Add a new `ScanDerivationConfig` class (parallel to `DerivationConfig`, not modifying it):

```python
@dataclass
class ScanDerivationConfig(MetaConfig):
    def __init__(self):
        # Similar structure but uses ScanDerivationGame + ScanPolicyValueNet
        # Simpler config: just n_sites, no budget needed
        # MCTS sims can be much lower (e.g., 50 vs 200)
        ...
    def build(self):
        # Reuses same LeafEvaluator construction
        # Instantiates ScanDerivationGame instead of DerivationGame
        ...
```

#### 5. `scripts/run_derivation.py`

Add a `derivation_mode` parameter (e.g., `"scan"` vs `"cfg"`) at the top of the interactive config. **Default is `"scan"`**. When `"scan"`:
- Use `ScanDerivationConfig` instead of `DerivationConfig`
- Hide budget-related parameters (irrelevant for scan)
- Adjust default hyperparameters (fewer MCTS sims, etc.)

When `"cfg"`: use existing `DerivationConfig` (legacy behavior, unchanged).

### New Test File

#### 6. `tests/test_scan_grammar.py`

- `permutation_to_program()` produces valid Programs with correct node counts
- `ScanState` transitions: remaining shrinks, prefix grows, terminal detection
- Action mask correctness: exactly the remaining indices
- No dead ends: random episodes always complete
- No duplicate indices in final permutation
- Evaluation: for small N (e.g., N=4), compare `permutation_to_program([0,1,2,3])` evaluated on all initial states against brute-force

---

## Why This Is Compatible and Modular

1. **LeafEvaluator reuse**: `permutation_to_program()` outputs the same `Program` AST → LeafEvaluator works unchanged, including caching by `program.pretty()`
2. **Game interface compliance**: `ScanDerivationGame` implements the same `Game` base class → MCTS, Agent, Trainer, Evaluator all work unchanged
3. **No existing code modified**: The new grammar is additive. `DerivationGame` and its grammar continue to work as-is.
4. **Config pattern**: `ScanDerivationConfig` follows the same `MetaConfig.build()` pattern → `run_derivation.py` switches between modes via a single config choice
5. **Frozen state reuse**: Same `all_initial_states()` and `DSLGameConfig` → identical evaluation conditions for fair comparison

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Force last element? | Yes (N-1 steps, not N) | Last step has only 1 choice — no decision to make |
| Observation encoding | `[prefix..., remaining_mask...]` flat vector | Simple, fixed-size, no padding ambiguity |
| Network architecture | MLP (not Transformer) | Observation is flat fixed-size, no sequential structure to exploit |
| Action space | `Discrete(N)` with masking | Natural: actions ARE bit indices. Masking handled by existing MCTS code |
| Intermediate reward | None (terminal only) | Consistent with existing framework. Value head learns partial estimates. |

---

## Search Space Comparison

| Property | Current CFG (L=14, N=6) | Scan Grammar (N=6) |
|----------|------------------------|-------------------|
| Programs | 151,173,432 | 720 |
| Canonical programs | 37,463,688 | 720 |
| Max branching factor | 48 | 6 |
| Derivation depth | ~9 | 5 |
| Search tree size | ~10^15 | ~720 |
| Dead ends possible | Yes | No |
| Optimal program reachable (N=7) | No (budget too small) | Yes (always) |

---

## Verification Plan

1. **Unit tests** (`test_scan_grammar.py`): grammar correctness, state transitions, action masks, permutation validity, program evaluation
2. **Smoke test**: run `scripts/run_derivation.py` with `derivation_mode="scan"`, N=6, ~5 iterations, verify learning curve improves
3. **Comparison**: run both `cfg` and `scan` modes on N=6 and N=7 with same frozen states, compare best reward and convergence speed
4. **Regression**: existing `cfg` mode tests still pass (no code changed)
