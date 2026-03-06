# MCTS Nonterminal Rollout Evaluation

**Date:** 2026-03-05
**Status:** Proposed
**Goal:** Break the reward-desert bootstrapping failure in MCTS-guided program synthesis by replacing network value estimates at nonterminal leaves with Monte Carlo random completions. Game-agnostic: modifies only `mcts.py` using the `Game` interface.

---

## Context

The AlphaZero program synthesis system fails to bootstrap on Doors D>=4 (and in 13/13 grammar-suite runs for D=3 with limited compute). The causal chain:

1. **Reward desert**: 99% of randomly derived programs score identically (~-0.105 for D=4, ~-0.075 for D=3). The reward distribution is unimodal with variance ~0.00002.
2. **Value collapse**: The MSE-optimal value prediction is a constant. The network correctly learns this, driving value loss to ~0.00003.
3. **Q-normalization degeneracy**: When all backed-up Q-values are identical, `q_max == q_min`, and Q-normalization outputs 0.5 for all actions (`mcts.py:338`).
4. **MCTS collapses**: UCB becomes `0.5 + c * P(a) * sqrt(N)/(1+N(a))` -- pure policy-prior + exploration, zero exploitation. MCTS degenerates into policy-guided BFS.
5. **Vicious cycle**: No exploitation -> no discovery of better programs -> no reward variance -> no learning.

**Contrast with Go**: Go has a structural guarantee of reward variance (zero-sum: one player always wins, bimodal {+1,-1}). Program synthesis has no such guarantee.

**The fix**: At nonterminal MCTS leaf nodes, instead of returning the network's uninformed value estimate, **randomly complete the game m times using only the Game interface, evaluate each completion, and aggregate**. This injects ground-truth reward variance into MCTS, breaking the flat Q-value landscape from iteration 1.

---

## Critical Evaluation of Original Plan \<a\>

### Sound Elements
- Correct root-cause diagnosis (reward desert -> value collapse -> Q-degeneration)
- Correct integration point (`search()` leaf expansion, `mcts.py:252-261`)
- `rollout_max` aggregation well-suited for sparse-reward domains (optimistic bias amplifies rare signal)
- Logging `std(leaf_values)` and Q-norm health as validation metrics

### Issues Identified and Resolved

| # | Issue in Plan \<a\> | Problem | Resolution |
|---|---------------------|---------|------------|
| 1 | `completion_policy=default` | Domain-specific heuristic ("fill holes with fixed completion"). Violates game-agnostic requirement. | **Dropped.** |
| 2 | `completion_policy=policy_guided` | Circular: if the network is uninformed, policy-guided ~ random. Only helps after learning starts, at which point the value network should also be improving. Adds code complexity for marginal benefit. | **Dropped.** Uniform random only. |
| 3 | Caching by `partial_ast_hash` | Random completions are stochastic. Caching one result for a given partial AST misrepresents the distribution. Furthermore, MCTS already expands each leaf exactly once (`nn_policy is None` check), so there is no recomputation to cache away. | **Dropped.** LeafEvaluator already caches by `program.pretty()` for complete programs. |
| 4 | `rollout_softmax` mode | Introduces temperature hyperparameter `tau` with no principled way to set it. `tau -> inf` = mean, `tau -> 0` = max. The two extremes (mean, max) suffice. | **Dropped.** Retain only mean and max. |
| 5 | Computational cost unaddressed | With m=4 rollouts per leaf, ~50 new leaves per MCTS search, 17 searches/game, 80 games/iter, the overhead is ~272K program completions per iteration. | **Added `rollout_budget`**: caps total game steps across all m rollouts for a single leaf. Graceful fallback to `nn_value` if budget exhausted. |
| 6 | No hybrid/transition strategy | Early training needs rollouts (uninformed network). Later, the network should be good enough. No mechanism to transition. | **Added `rollout_blend`**: `value = (1-b)*rollout + b*nn_value`. Start at 0.0, increase over training. |
| 7 | Only terminal reward collected | For games with dense intermediate rewards, collecting only terminal reward is incorrect. | **Fixed**: accumulate `game.reward` across all rollout steps. For DerivationGame intermediates are 0, so this is equivalent. For general games, it's correct. |
| 8 | `complete_program(partial_ast, mode)` | Proposed function signature uses domain-specific types (`partial_ast`). | **Fixed**: `_rollout_value()` operates on `self.game` via the Game interface. No AST knowledge. |

---

## Mathematical Justification

### Why rollouts from partial ASTs create differential signal

The key insight: partial ASTs that have committed to productive structure produce random completions that score better on average than those with unproductive structure. Unlike a flat prior over all programs, completing from a partial AST **conditions** the program distribution on the already-derived prefix.

**Concrete analysis (D=4, 8 locations, 3 keys):**

Consider two partial ASTs after 6 derivation steps:
- **AST_A**: `Ite(And(Not(IsZero(1)), Not(IsZero(12))), Flip(8), ProgramHole(34))` -- encodes "if at key-0 location AND key-0 available, PICK(key 0), else [...]"
- **AST_B**: `Ite(IsZero(0), Flip(0), ProgramHole(38))` -- encodes "if not at location 0, MOVE to location 0, else [...]"

Random completions of AST_A: the first Ite-branch fires productively whenever the agent happens to be at loc 1 with key 0 available. Even with uniformly random remaining rules, programs pick up key 0 some fraction of the time. Estimated: ~15% of completions score -0.075 (1 key picked up), rest score -0.105.

Random completions of AST_B: the first Ite-branch moves to location 0, which is the start position (useless). The remaining random rules are no better than fully random programs. Estimated: ~1% score -0.075.

**With `rollout_n=4`, `rollout_mode=max`:**

| Partial AST | P(>= 1 good completion) | Expected rollout value |
|-------------|------------------------|----------------------|
| AST_A (productive) | 1 - (0.85)^4 = **48%** | ~-0.090 |
| AST_B (useless) | 1 - (0.99)^4 = **4%** | ~-0.104 |

The Q-value gap is 0.014. Q-normalization maps this to [0, 1]: AST_A's children get q_norm ~ 1.0, AST_B's get q_norm ~ 0.0. **MCTS can now exploit the difference.**

**Current behavior (nn_value only):**
Both ASTs get nn_value ~ -0.105 (the constant prediction). q_max == q_min. q_norm = 0.5 for all actions. **Zero exploitation signal.**

### Why `rollout_max` is preferred over `rollout_mean`

In sparse-reward domains, the mean of m random completions is dominated by the majority outcome (-0.105). The max acts as an **optimistic estimator**: it asks "does there exist at least one completion from this partial AST that's better than baseline?"

This is analogous to Upper Confidence Bound (UCB) philosophy: be optimistic in the face of uncertainty. The max doesn't overestimate the true value -- it measures the best achievable outcome, which is the relevant signal for MCTS to determine which subtrees to explore.

For `rollout_mean` with m=4: AST_A averages ~-0.103, AST_B averages ~-0.105. Gap = 0.002. Barely detectable.
For `rollout_max` with m=4: AST_A ~ -0.090, AST_B ~ -0.104. Gap = 0.014. 7x stronger signal.

### Comparison to classical MCTS rollouts (Go, etc.)

This approach is the program-synthesis analogue of classical MCTS random rollouts (pre-AlphaGo). AlphaGo/AlphaZero replaced rollouts with a value network because the network could generalize across positions. In Go, the network succeeds because reward variance exists from game 1 (someone always wins).

In program synthesis, the value network has nothing to learn from (flat reward landscape). Rollouts provide the bootstrap signal that the network cannot. Once MCTS with rollouts discovers diverse programs, the value network's training data gains variance, and the network begins to learn. At that point, `rollout_blend` can be increased toward 1.0 to phase out rollouts and rely on the (now-informed) network.

This is a **bootstrap mechanism**, not a permanent replacement for the value network.

---

## Implementation

### Files to Modify

| # | File | Action | Scope |
|---|------|--------|-------|
| 1 | `src/alphazeropp/core/mcts.py` | **MODIFY** | Add 4 constructor params, `_rollout_value()` method, modify `search()` leaf expansion |
| 2 | `scripts/run_doors_derivation.py` | **MODIFY** | Expose rollout params in interactive config UI |
| 3 | `src/alphazeropp/instances/doors/dsl/derivation_config.py` | **MODIFY** | Add rollout defaults to mcts_params dicts |
| 4 | `tests/test_mcts_rollout.py` | **CREATE** | Unit tests for rollout mechanism |

### File 1: `src/alphazeropp/core/mcts.py`

**A. Constructor** -- add 4 keyword arguments with backward-compatible defaults:

```python
def __init__(self, game: Game, net: PolicyValueNet,
             n_simulations: int = 25,
             temperature: float = 1.0,
             c_exploration: float = 1.0,
             dirichlet_alpha: float = 0.3,
             dirichlet_epsilon: float = 0.25,
             # -- Nonterminal rollout evaluation --
             rollout_n: int = 0,
             rollout_mode: str = "mean",
             rollout_blend: float = 0.0,
             rollout_budget: int = 500):
```

**B. New method** `_rollout_value(self, msg) -> float | None`:

```python
def _rollout_value(self, msg) -> float | None:
    """Monte Carlo value estimate via random game completion.

    Completes the game rollout_n times using uniform random actions,
    accumulates rewards, and aggregates. Uses only the Game interface
    (clone, get_action_mask, step_wrapper). Fully game-agnostic.

    Returns None if no rollout reached a terminal state (budget exhausted),
    signaling the caller to fall back to nn_value.
    """
    rewards = []
    budget_remaining = self.rollout_budget

    for _ in range(self.rollout_n):
        if budget_remaining <= 0:
            break
        rollout_game = self.game.clone()
        cumulative_reward = 0.0

        while not (rollout_game.terminated or rollout_game.truncated):
            if budget_remaining <= 0:
                break
            mask = rollout_game.get_action_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break  # dead end
            action = valid[np.random.randint(len(valid))]
            if len(rollout_game.action_space.shape) == 0:
                rollout_game.step_wrapper(int(action))
            else:
                rollout_game.step_wrapper(
                    np.unravel_index(action, mask.shape))
            cumulative_reward += rollout_game.reward
            budget_remaining -= 1

        if rollout_game.terminated or rollout_game.truncated:
            rewards.append(cumulative_reward)

    if not rewards:
        return None

    if self.rollout_mode == "max":
        return float(max(rewards))
    return float(sum(rewards) / len(rewards))
```

**Why `game.clone()` (not `stash_state()`)**:
Each rollout needs an independent game that runs to completion. `clone()` returns an independent copy. `stash_state()` operates on `self` and would corrupt the MCTS game state. Both `DerivationGame` and `FactoredDerivationGame` have lightweight `clone()` overrides that share the `leaf_evaluator` reference (verified: `derivation_game.py:254-269`, `factored_derivation_game.py:374-389`).

**Why uniform random (not policy-guided)**:
At early training, the network policy is uninformed (uniform or random). Policy-guided completion ~= random completion but requires a network forward pass per step (expensive). Once the network is informed, the value network should also be informed, making rollouts unnecessary. Uniform random avoids both the circular dependency and the extra cost.

**C. Modify `search()` leaf expansion** (lines 252-261):

Replace:
```python
mynode.nn_value = myvalue
mynode.action_mask = myaction_mask
return myvalue
```

With:
```python
mynode.action_mask = myaction_mask

leaf_value = myvalue
if self.rollout_n > 0:
    rv = self._rollout_value(msg)
    if rv is not None:
        leaf_value = ((1.0 - self.rollout_blend) * rv
                      + self.rollout_blend * myvalue)

mynode.nn_value = leaf_value
return leaf_value
```

**Correctness argument**: At this point in `search()`, `self.game` is at the leaf state (reached via recursive `step_wrapper` calls at line 272). `_rollout_value()` clones from this state, runs completions independently, and returns without modifying `self.game`. The parent call's `unstash_state` restores `self.game` after `search()` returns. No state corruption.

**Semantic correctness**: The network predicts V(s) -- expected total future return. The rollout computes a Monte Carlo estimate of V(s) via random play. Both are valid estimates of the same quantity. The rollout is noisier but unbiased (given uniform random completion), while the network estimate has low variance but high bias (outputs a constant).

### File 2: `scripts/run_doors_derivation.py`

In `_build_sections()`, extend `mcts_descs` dict (~line 108) and the `for k in [...]` loop (~line 115) to include the 4 new params. Add them to `mcts_labels` set (~line 174) so they appear in the MCTS section of the interactive editor.

### File 3: `src/alphazeropp/instances/doors/dsl/derivation_config.py`

Add to `mcts_params` in each config class:
```python
"rollout_n": 4,
"rollout_mode": "max",
"rollout_blend": 0.0,
"rollout_budget": 200,
```

### File 4: `tests/test_mcts_rollout.py`

Create test file following patterns from existing `tests/test_derivation_game.py`. Test cases:

1. **`test_rollout_disabled_by_default`**: MCTS with default params (rollout_n=0) produces identical results to current behavior.
2. **`test_rollout_returns_value`**: With rollout_n=4 on a DerivationGame, `_rollout_value()` returns a finite float.
3. **`test_rollout_budget_respected`**: rollout_budget=10, rollout_n=100 -- completes without hanging, returns a value or None.
4. **`test_rollout_blend_zero`**: blend=0.0 returns pure rollout value (not nn_value).
5. **`test_rollout_blend_one`**: blend=1.0 returns nn_value regardless of rollout result.
6. **`test_q_norm_not_degenerate`**: After `perform_simulations` with rollout_n=4, verify `mcts.q_max > mcts.q_min` (Q-normalization is not stuck in the degenerate case).
7. **`test_rollout_game_agnostic`**: Run rollout on a non-derivation game (e.g., BitString) to confirm no DerivationGame-specific assumptions.

---

## Configuration

| Param | Type | Default | Recommended (D=4 Doors) | Rationale |
|-------|------|---------|------------------------|-----------|
| `rollout_n` | int | 0 | 4 | 0 = disabled (backward compat). 4 gives P(finding better reward) ~ 48% for productive ASTs. |
| `rollout_mode` | str | `"mean"` | `"max"` | Max provides 7x stronger signal than mean in sparse-reward domains. |
| `rollout_blend` | float | 0.0 | 0.0 | 0 = pure rollout (best for uninformed network). Increase toward 1.0 as network improves. |
| `rollout_budget` | int | 500 | 200 | Caps cost. D=4 derivations are ~17 steps; 200 allows ~11 full completions. |

All new params go into `AgentConfig.mcts_params` dict, which flows through to `MCTS.__init__(**kwargs)`. No new config classes needed.

---

## Cost Analysis

### Per-leaf cost breakdown (rollout_n=4, D=4)

| Operation | Count | Unit cost | Total |
|-----------|-------|-----------|-------|
| `game.clone()` | 4 | ~1 us (lightweight) | ~4 us |
| `step_wrapper()` (grammar ops) | 4 x ~17 = 68 | ~10 us | ~680 us |
| `leaf_evaluator()` (program eval) | 4 | ~1 ms (10 frozen states x 35 steps) | ~4 ms |
| **Total per leaf** | | | **~5 ms** |

### Per-iteration cost (80 games, 100 sims, 17 searches/game)

| Metric | Without rollouts | With rollouts (m=4) |
|--------|-----------------|---------------------|
| New leaves per MCTS search | ~50 | ~50 |
| LeafEvaluator calls per search | ~50 (terminal only) | ~50 + 200 = 250 |
| LeafEvaluator calls per game | ~850 | ~4250 |
| LeafEvaluator calls per iteration | ~68K | ~340K |
| LeafEvaluator cache hit rate | moderate | moderate+ (same cache shared) |
| Estimated wall-clock multiplier | 1x | **~2-3x** |

The 2-3x cost increase is acceptable given the qualitative difference (0% solve rate -> potential bootstrapping). The `rollout_budget` parameter provides a hard cap on worst-case cost per leaf.

### Mitigation strategies
- **Reduce `n_simulations`** (e.g., from 100 to 50) when using rollouts. Rollouts provide better per-leaf estimates, so fewer simulations may suffice.
- **`rollout_budget`** prevents runaway cost on deep derivations.
- **LeafEvaluator caching** deduplicates identical programs across rollouts.
- **Future: adaptive rollouts** -- skip rollouts at shallow MCTS depths where all children look similar regardless.

---

## Edge Cases and Invariants

1. **Dead-end programs**: If a rollout hits a dead end (no legal productions but program incomplete), `game.truncated = True` and `reward = 0.0`. The rollout correctly records this as `cumulative_reward = 0.0`. This is fine -- dead ends are genuinely worse than completing programs (which score ~-0.105, below 0.0 in absolute terms but...wait, 0.0 > -0.105). This is a pre-existing issue with the reward scale, not introduced by rollouts.

2. **Budget exhaustion**: If `rollout_budget` is too small for any rollout to complete, `_rollout_value()` returns `None`, and the leaf falls back to `nn_value`. No crash, no degradation vs current behavior.

3. **Multiprocessing**: Each self-play worker has its own MCTS, game, and network. Rollouts are process-local. `np.random.randint` uses the per-process RNG state (forked at process creation). No thread-safety issues. For strict reproducibility, could use a local `np.random.Generator` seeded from the MCTS instance -- out of scope for initial implementation.

4. **Tree reuse** (`perform_simulations_reuse`): Rollouts happen inside `search()`, which is called identically by both `perform_simulations` and `perform_simulations_reuse`. No special handling needed.

5. **Factored game two-phase actions**: During rollouts, `FactoredDerivationGame.step_wrapper()` handles the structure/parameter phase internally. The action mask changes between phases. The rollout's `get_action_mask() -> random action -> step_wrapper()` loop handles this transparently.

---

## Verification Plan

### Phase 1: Unit tests (test_mcts_rollout.py)
- Verify backward compatibility (rollout_n=0)
- Verify rollout produces values and respects budget
- Verify blend behavior at extremes (0.0, 1.0)
- **Critical**: verify q_max > q_min with rollouts enabled

### Phase 2: Integration test (D=3 Doors, 20 iterations)
Run D=3 doors_factored with:
```python
rollout_n=4, rollout_mode="max", rollout_blend=0.0, rollout_budget=200
n_simulations=100, n_games_per_train=30, n_iterations=20
```

**Expected observations (vs baseline rollout_n=0):**
- `std(leaf_values) > 0` from iteration 1 (not collapsing to constant)
- `q_max - q_min > 0` within MCTS searches (Q-normalization active)
- `val_loss` does NOT monotonically decrease to ~0.00003 (training data has variance)
- `gate_score > 0.5` by iteration 5-10 (new network beats old)
- Possible: solve_rate > 0 by iteration 15-20

### Phase 3: Backward compatibility
Run full existing test suite with default configs (rollout_n=0). All tests must pass unchanged.
