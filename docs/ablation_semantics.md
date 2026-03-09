# Ablation Semantics: Verified Code Reference Map

**Date:** 2026-03-09
**Purpose:** Exact semantic map of network query paths, reference sharing, and edge cases — verified against source code before implementing ablation wrappers.

---

## Q1: What happens for n_simulations = -1, 0, 1?

| n_sims | Behavior | Net queried? | Returns |
|--------|----------|-------------|---------|
| **-1** | `mcts.py:107-109`: Direct `query_net_masked()` call. Returns masked policy as "counts". Temperature scaling applied at `mcts.py:160-175`. | Yes (1 call) | Temperature-scaled network policy |
| **0** | `mcts.py:112-117`: Root expanded via `search()` → 1 net query. `mcts.py:146`: `range(0)` loop runs 0 times. `mcts.py:154-157`: Counts from visit counts = all zeros. `mcts.py:163-171`: `nonzero.any()` is False → returns all-zero probs. | Yes (1 call) | **All zeros (broken!)** |
| **1** | Root expanded via `search()` + 1 simulation loop iteration. Visit counts reflect 1 search path. | Yes (1-2 calls) | Visit-count-based probs |

### Detail: n_simulations = -1

```python
# mcts.py:107-109
if self.n_simulations < 0:
    if msg: print(msg, "n_simulations < 0, directly querying policy net")
    counts, _, _ = self.query_net_masked(msg)
```

The masked policy probabilities are used as "counts" and then temperature-scaled at `mcts.py:160-175`. With `temperature=1.0`, this is equivalent to returning the raw network policy (log-exp round-trip preserves values). With low temperature, the policy is sharpened.

### Detail: n_simulations = 0

```python
# mcts.py:112-117 — root expansion happens
if mystate not in self.nodes:
    old_game_state = self.game.stash_state()
    self.search(entab(msg, ", root expand"))
    self.game = self.game.unstash_state(old_game_state)

# mcts.py:146 — loop runs 0 times
for i in range(self.n_simulations):  # range(0) = empty
    ...

# mcts.py:154-157 — counts are all zero
counts = np.zeros_like(mynode.nn_policy)
for action, count in mynode.action_N.items():  # empty dict
    counts[action] = count

# mcts.py:163-171 — returns all zeros
nonzero = counts > 0  # all False
if nonzero.any():     # False
    ...
else:
    probs = counts    # all zeros
```

**Conclusion:** `n_simulations=0` is **not usable** as an experiment condition. The net is queried but the result is discarded.

---

## Q2: Does n_simulations=0 still query the net at the root?

**Yes.** The root expansion at `mcts.py:112-117` calls `self.search()`, which reaches the unexpanded root node and calls `query_net_masked()` at `mcts.py:329`. The policy and value are stored in the node (`mcts.py:330-332`), but since no simulations run, visit counts remain zero and the stored values are never used for the returned probabilities.

---

## Q3: Where exactly is legal-action masking applied?

### 3a. Network policy masking — `query_net_masked()` at `mcts.py:382-398`

```python
def query_net_masked(self, msg):
    mypolicy, myvalue = self.query_net(msg)          # mcts.py:384
    myaction_mask = self.game.get_action_mask()       # mcts.py:386
    mypolicy *= myaction_mask                         # mcts.py:390 — MASKING
    sum_policy = mypolicy.sum()
    if sum_policy > 0:
        mypolicy /= sum_policy                       # renormalize
    else:
        mypolicy += myaction_mask / sum_mask          # fallback: uniform over legal
    return mypolicy, myvalue, myaction_mask
```

### 3b. UCB masking — `calc_masked_ucbs()` at `mcts.py:422-446`

- **1D** (`mcts.py:428-430`): Invalid actions set to `-inf` via `ucbs[~mask] = -np.inf`
- **Multi-dim** (`mcts.py:434-446`): Only valid actions from `np.nonzero(mask)` are processed; rest default to `-inf`

### 3c. Dirichlet noise masking — `perform_simulations()` at `mcts.py:136-140`

```python
mask = mynode.action_mask
masked_noise = noise * mask          # mcts.py:137
sum_noise = masked_noise.sum()
if sum_noise > 0:
    masked_noise /= sum_noise        # mcts.py:140
    mynode.nn_policy = (1 - eps) * mynode.nn_policy + eps * masked_noise
```

---

## Q4: Where exactly are Dirichlet noise and temperature applied?

### Dirichlet noise — `perform_simulations()` at `mcts.py:121-144`

Applied **once** at the root when:
- `add_noise=True` (parameter)
- `mynode.total_N == 0` (fresh, unexpanded root)

Formula: `policy_new = (1 - epsilon) * policy + epsilon * Dir(alpha) * mask`

Parameters: `self.dirichlet_alpha` and `self.dirichlet_epsilon` from MCTS constructor.

Also applied in `perform_simulations_reuse()` at `mcts.py:206-221` with the same logic but using `nn_policy_original` as the base.

### Temperature — `mcts.py:160-175`

Applied to visit counts (or raw policy for n_sims < 0) via numerically stable log-space:

```python
log_counts[nonzero] = np.log(counts[nonzero]) / self.temperature  # mcts.py:166
log_counts -= log_counts.max()
probs = np.exp(log_counts)
probs /= probs.sum()
```

Temperature can be overridden per-call via `agent.policy(temperature_override=...)` at `agent.py:99-100`.

---

## Q5: During evaluation, can noise be fully disabled?

**Yes.** Pass `add_noise=False` to `agent.policy()` (`agent.py:76`). This propagates to `mcts.perform_simulations(..., add_noise=False)` at `agent.py:101`. The noise block at `mcts.py:122` checks `add_noise` as the first condition, so noise is never injected.

For near-greedy evaluation, also pass `temperature_override=0.05` (low temperature sharpens visit-count distribution toward argmax).

---

## Q6: Does trainer hold its own net reference separate from agent?

**No — same Python object.** In `instances/doors/config.py:106-122`:

```python
net = DoorsDirectNet(**self.net.kwargs)            # config.py:106 — ONE net created
agent = Agent(game=game, net=net, ...)             # config.py:109 — same net
trainer = Trainer(agent=agent, net=net, game=game, ...)  # config.py:116 — same net
```

`trainer.net`, `agent.net`, and `trainer.agent.net` all point to the **same object**. Confirmed by comment in `gated_trainer.py:53-54`: *"trainer.net and trainer.agent.net are the same object, so one load_state_dict updates both references."*

### Implication for ablation wrappers

After `cfg.build()`, wrapping must update ALL references:

```python
wrapped = SomeWrapper(net)
agent.net = wrapped
trainer.net = wrapped
# trainer.agent.net is agent.net (same Agent object), so already updated
```

---

## Q7: Does evaluator hold its own agent/net reference?

**No.** `Evaluator.__init__()` takes only `n_games` and `n_procs` (`evaluator.py` constructor). Agents are received as parameters to `pit()` at `evaluator.py:70-75`.

Inside `pit()`, nets are extracted from agents for multiprocessing:

```python
mp_manager = MultiprocessingManager(new_agent.net, old_agent.net, self)  # evaluator.py:81
```

The agents passed to `pit()` are **deepcopies** created by GatedTrainer (`gated_trainer.py:38,44`). If the original agent's net is wrapped, the deepcopy preserves the wrapper.

---

## Q8: Are rollouts active by default? If yes, where?

**No.** Default `rollout_n=0` at `mcts.py:54`. Rollout evaluation at `mcts.py:336-340` is gated:

```python
if self.rollout_n > 0:  # mcts.py:336 — False by default
    rv = self._rollout_value(msg)
    ...
```

No interference with ablation experiments.

---

## Q9: What tuple signature does net.train(examples) have?

### Signature (`network.py:54`)

```python
def train(self, examples, needs_reshape=True, print_all_epochs=False):
```

### Return (`network.py:126`)

```python
return model, train_batch_losses, train_losses, policy_losses, value_losses
```

- `model`: PyTorch `nn.Module` (the network's `self.model`)
- `train_batch_losses`: `list[float]` — per-batch total losses
- `train_losses`: `list[float]` — per-epoch average total losses
- `policy_losses`: `list[float]` — per-epoch average policy losses
- `value_losses`: `list[float]` — per-epoch average value losses

### Consumer (`trainer.py:156`)

```python
_, train_batch_losses, train_losses, policy_losses, value_losses = self.net.train(flat_examples)
```

First element (model) is discarded. Remaining 4 lists are used for logging.

### FrozenWrapper requirement

Must return: `(self._net.model, [0.0], [0.0], [0.0], [0.0])` — the real model object + 4 single-element lists.

---

## Discrepancies with Ablation Plan

### 1. n_simulations=0 is broken

The ablation plan (Section 2.2) states: *"n_simulations = 0 → runs the expansion loop 0 times, so only root expansion happens (1 net query, no search)"* — implying it returns useful policy output. **In reality, it returns all-zero probabilities.** Do NOT use `n_simulations=0` as an experiment condition. Use `n_simulations=1` as the minimum useful count.

### 2. n_simulations=-1 temperature interaction

When `--no-mcts` sets `n_simulations=-1`, the network policy values pass through temperature scaling (`mcts.py:160-175`). The `--no-mcts` flag should **also set `temperature=1.0`** to get the raw network policy. Otherwise, evaluation with low temperature (e.g., 0.05) would artificially sharpen the "pure network" output.

### 3. Wrapper injection must update both references

The ablation plan (Section 3.2.2) says to replace `agent.net` and `trainer.net`. Since `trainer.agent` IS `agent` (same object from `config.py:107-113,114-122`), setting `agent.net = wrapped` automatically updates `trainer.agent.net`. But `trainer.net` must ALSO be set explicitly because it's a direct attribute, not derived from `trainer.agent.net`.

### 4. GatedTrainer deepcopy preserves wrappers

`copy.deepcopy(self.trainer.agent)` at `gated_trainer.py:38` copies the wrapper and inner net. Weight restoration at `gated_trainer.py:53-54` does `self.trainer.net.model.load_state_dict(...)`. With `__getattr__` delegation, `wrapper.model` returns `wrapper._net.model` — correct. The `train()` return's first element (`self._net.model`) is the same object, maintaining consistency.

### 5. `__getattr__` + pickle = infinite recursion (FOUND AND FIXED)

During pickle unpickling, Python calls `__getattr__('_net')` before `__dict__` is populated. A naive `__getattr__` that does `return getattr(self._net, name)` causes infinite recursion because `self._net` triggers another `__getattr__` call.

**Fix:** Use `object.__getattribute__(self, '_net')` with an `AttributeError` guard. Implemented as `_safe_getattr()` in `network_ablations.py`. All 3 wrappers use this pattern.

---

## Network Attribute Access Paths

For wrapper `__getattr__` delegation to work, these access patterns must be verified:

| Access Pattern | Where Used | Wrapper Behavior |
|---------------|-----------|-----------------|
| `net.predict(obs)` | `mcts.py:376` via `query_net()` | Direct method on wrapper |
| `net.train(examples)` | `trainer.py:156` | Direct method on wrapper |
| `net.model` | `gated_trainer.py:53-54` | `__getattr__` → `_net.model` |
| `net.model.state_dict()` | `gated_trainer.py:53` | `__getattr__` → `_net.model.state_dict()` |
| `net.model.load_state_dict()` | `gated_trainer.py:54` | `__getattr__` → `_net.model.load_state_dict()` |
| `net.push_multiprocessing()` | `evaluator.py:81` via `MultiprocessingManager` | `__getattr__` → `_net.push_multiprocessing()` |
| `net.pop_multiprocessing()` | `evaluator.py:81` via `MultiprocessingManager` | `__getattr__` → `_net.pop_multiprocessing()` |
| `net.save_checkpoint()` | training loop | `__getattr__` → `_net.save_checkpoint()` |
| `net.DEVICE` | `network.py:44` | `__getattr__` → `_net.DEVICE` |
| `net.optimizer` | `network.py:48` | `__getattr__` → `_net.optimizer` |
