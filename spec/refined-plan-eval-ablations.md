# Eval-Time Ablation: Refined Implementation Plan

**Date**: 2026-03-09

## Context

We want to isolate what the trained network contributes at inference time. The plan adds an `--eval-ablation` flag to `scripts/run_doors_direct.py` that modifies ONLY evaluation behavior — training remains unchanged. Each iteration runs the standard training loop, then runs a modified evaluation under the chosen ablation mode.

Prior work already completed:
- `docs/ablation_semantics.md` — 264-line verified semantic map of MCTS/agent/trainer interactions
- `tests/test_ablation_semantics.py` — 11 test classes verifying wrapper delegation, masking, pickle, etc.
- `src/alphazeropp/instances/doors/network_ablations.py` — 3 wrappers (Frozen, UniformPolicy, ZeroValue) + `_safe_getattr`
- `src/alphazeropp/benchmark/` — full harness with `eval_loop.py`, `result_schema.py`, `ResultWriter`

---

## Codebase Mapping

| Plan Item | Existing Code | Classification |
|---|---|---|
| `--eval-ablation` CLI flag | `parse_args()` at `run_doors_direct.py:1214` — only has `--seeds`, `--non-interactive` | **Modify** |
| `full` mode | `compute_solve_rate()` at `run_doors_direct.py:522` | **Exists** — no change |
| `policy-only` mode | `n_simulations=-1` path in `mcts.py:107-109` — already works with legal masking | **Reuse** existing path |
| `value-only-1step` | Nothing exists — custom 1-step lookahead needed | **New** |
| `uniform-prior` wrapper | `UniformPolicyWrapper` in `network_ablations.py:22` | **Reuse** — wrap at eval only |
| `zero-value` wrapper | `ZeroValueWrapper` in `network_ablations.py:43` | **Reuse** — wrap at eval only |
| `unguided-search` | No combined wrapper | **New** — compose both existing wrappers |
| `frozen-random` | `FrozenWrapper` changes training — NOT eval-only | **Conflicts** — redefined as `random-net` |
| `eval_summary.csv` output | `ResultWriter` in `benchmark/result_schema.py` exists but is for benchmark harness | **New** — lightweight eval CSV for ablation |
| `eval_manifest.json` | Nothing exists | **New** |
| Network-call counter | Nothing exists | **Deferred** — trivial to add later via wrapper |

### Key Assumptions Verified

| Assumption | Status | Evidence |
|---|---|---|
| `compute_solve_rate` uses `add_noise=False` | **Confirmed** | `run_doors_direct.py:533` |
| `agent.policy()` clones game internally | **Confirmed** | `agent.py:93` |
| Agent constructor doesn't mutate game/net | **Confirmed** | `agent.py:40-56` |
| `model.input_size`, `model.output_size` available | **Confirmed** | `policy_value_net.py:162-163` |
| Trainer + agent share same net object | **Confirmed** | `config.py:106-122`, `gated_trainer.py:53-54` |
| `n_simulations=0` is broken (returns all zeros) | **Confirmed** | `docs/ablation_semantics.md` Q1 |

### Design Decision: `frozen-random` → `random-net`

The original plan's `frozen-random` mode uses `FrozenWrapper` which disables `train()` — this is a training-time ablation, not eval-only. Redefined as `random-net`: at eval time, create a freshly initialized (random-weight) network and use it for evaluation. Training is untouched.

---

## Execution Plan

### Day 1: Core ablation dispatch + value-only-1step

#### Step 1: Add `value_only_1step_policy()` to `network_ablations.py`

**File**: `src/alphazeropp/instances/doors/network_ablations.py`
**Action**: Add function after existing wrappers

```python
def value_only_1step_policy(game, net, gamma=1.0):
    """One-step lookahead using only the value head.

    For each legal action: clone state, step, compute r + γ*V(s').
    Returns one-hot probability on argmax action.
    """
```

Logic:
1. `mask = game.get_action_mask()` — get legal actions
2. For each legal action `a`: `g = game.clone(); obs, r, term, trunc, _ = g.step_wrapper(a)`
3. If terminated: `q[a] = r`. Else: `_, v = net.predict(g.obs); q[a] = r + gamma * v`
4. Return one-hot on `argmax(q)` over legal actions

**Justification**: Belongs with ablation code. Uses existing `game.clone()`, `game.step_wrapper()`, `net.predict()`.

#### Step 2: Add `RandomNetWrapper` to `network_ablations.py`

**File**: `src/alphazeropp/instances/doors/network_ablations.py`
**Action**: Add class after `FrozenWrapper`

```python
class RandomNetWrapper:
    """Eval-only: predict() uses fresh random net, train() delegates normally."""
    def __init__(self, net):
        self._net = net
        from alphazeropp.instances.doors.network import DoorsDirectNet
        self._random_net = DoorsDirectNet(
            input_size=net.model.input_size,
            output_size=net.model.output_size,
            device=str(net.DEVICE),
        )
    def predict(self, state):
        return self._random_net.predict(state)
    def train(self, *args, **kwargs):
        return self._net.train(*args, **kwargs)
    __getattr__ = _safe_getattr
```

**Justification**: `model.input_size` and `model.output_size` confirmed available at `policy_value_net.py:162-163`. No `random_seed` → torch uses default random init.

#### Step 3: Create `eval_ablations.py` — dispatch module

**File**: `src/alphazeropp/instances/doors/eval_ablations.py` (NEW)

Contains:
- `ABLATION_MODES` list (for CLI choices)
- `compute_solve_rate_ablated(agent, mode, n_episodes, gamma)` → `(solve_rate, avg_reward)`

This function:
1. If `mode == "full"`: calls existing `compute_solve_rate(agent)` unchanged
2. If `mode == "policy-only"`: creates temp Agent with `n_simulations=-1` in mcts_params, calls compute_solve_rate with temp agent
3. If `mode == "value-only-1step"`: runs episodes directly using `value_only_1step_policy` per step
4. If `mode in ("uniform-prior", "zero-value", "unguided-search", "random-net")`: wraps `agent.net` in appropriate wrapper(s), creates temp Agent, calls compute_solve_rate

Temp agent construction:
```python
Agent(game=agent.game, net=wrapped_net, mcts_params=agent.mcts_params.copy(),
      reward_discount=agent.reward_discount, random_seeds=agent.random_seeds)
```

This is safe because `Agent.__init__` doesn't mutate game/net (`agent.py:40-56`), and `agent.policy()` clones game internally (`agent.py:93`).

**Justification**: Keeps `run_doors_direct.py` clean. Single dispatch point for all modes.

### Day 2: CLI integration + output files

#### Step 4: Add `--eval-ablation` to `parse_args()`

**File**: `scripts/run_doors_direct.py` line 1214
**Action**: Add argument

```python
parser.add_argument("--eval-ablation", type=str, default=None,
                    choices=["full", "policy-only", "value-only-1step",
                             "uniform-prior", "zero-value", "unguided-search", "random-net"],
                    help="Eval-only ablation mode (training unchanged)")
```

#### Step 5: Thread ablation through `main()` → `_run_single_seed()`

**File**: `scripts/run_doors_direct.py`
**Actions**:
- `_run_single_seed(cfg, exp_dir)` → `_run_single_seed(cfg, exp_dir, eval_ablation=None)`
- After the existing `compute_solve_rate` call at line 1121, add ablation eval:
  ```python
  if eval_ablation and eval_ablation != "full":
      abl_solve, abl_reward = compute_solve_rate_ablated(
          agent, eval_ablation, n_episodes=20, gamma=cfg.agent.reward_discount)
      entry[f"abl_solve_rate"] = abl_solve
      entry[f"abl_avg_reward"] = abl_reward
  ```
- In `main()`: pass `args.eval_ablation` to `_run_single_seed`

#### Step 6: Update `setup_experiment_dir()` to encode ablation mode

**File**: `scripts/run_doors_direct.py` line 150
**Action**: Append `_abl-{mode}` to dirname when `eval_ablation` is set

```python
if eval_ablation:
    dirname += f"_abl-{eval_ablation}"
```

#### Step 7: Add `eval_summary.csv` and `eval_manifest.json` output

**File**: `scripts/run_doors_direct.py`, end of `_run_single_seed()`
**Action**: After the training loop, write:

1. `eval_summary.csv` — one row per iteration with columns: `iteration, seed, eval_ablation, solve_rate, avg_reward, abl_solve_rate, abl_avg_reward, wall_clock_s`
2. `eval_manifest.json` — full config dict + ablation mode + git hash

**Justification**: Lightweight output format matching plan requirements. Does not duplicate `iteration_log.jsonl` (which still has full training metrics).

### Day 3: Tests

#### Step 8: Create `tests/test_eval_ablations.py`

**File**: `tests/test_eval_ablations.py` (NEW)

Test cases using D=2 game (fast):

1. **`test_full_matches_baseline`** — `compute_solve_rate_ablated(agent, "full")` equals `compute_solve_rate(agent)`
2. **`test_policy_only_legal_actions`** — policy-only mode returns valid probabilities (sum=1, zeros on illegal)
3. **`test_value_only_1step_deterministic`** — given a known state where one action leads to reward and others don't, verify correct action chosen
4. **`test_value_only_1step_terminal_handling`** — if action leads to terminal state, use reward only (no V)
5. **`test_uniform_prior_uses_wrapper`** — verify the temp agent's net is `UniformPolicyWrapper` instance
6. **`test_unguided_ignores_learned`** — verify both policy and value are overridden
7. **`test_random_net_differs_from_trained`** — random-net predictions differ from trained net
8. **`test_training_net_unchanged`** — after any ablation eval, `agent.net.model.state_dict()` is byte-identical to before

---

## Files Summary

| File | Action | Lines Changed (est.) |
|---|---|---|
| `src/alphazeropp/instances/doors/network_ablations.py` | **Modify** | +50 (function + class) |
| `src/alphazeropp/instances/doors/eval_ablations.py` | **Create** | ~100 |
| `scripts/run_doors_direct.py` | **Modify** | ~30 (CLI, threading, output) |
| `tests/test_eval_ablations.py` | **Create** | ~150 |

**No modifications to**: `mcts.py`, `agent.py`, `trainer.py`, `network.py`, `evaluator.py`

---

## Verification

```bash
# 1. Run existing ablation semantics tests (regression check)
pytest tests/test_ablation_semantics.py -v

# 2. Run new eval ablation tests
pytest tests/test_eval_ablations.py -v

# 3. Run existing doors tests (regression check)
pytest tests/test_doors_direct.py -v

# 4. Smoke test: train 3 iterations with each ablation mode (D=2, fast)
for mode in full policy-only value-only-1step uniform-prior zero-value unguided-search random-net; do
  python scripts/run_doors_direct.py --seeds 42 --non-interactive --eval-ablation $mode
done

# 5. Verify output files exist
ls experiments/doors_direct/*_abl-*/eval_summary.csv
ls experiments/doors_direct/*_abl-*/eval_manifest.json

# 6. Compare solve rates across modes (quick eyeball)
for d in experiments/doors_direct/*_abl-*/; do
  echo "=== $(basename $d) ==="
  tail -1 "$d/eval_summary.csv"
done
```
