# BitString Fitness Functions

## Overview

The BitString game can now be trained with different fitness functions that control
how per-step rewards are computed. This lets you run experiments where the agent
optimizes different objectives (onemax vs leading_ones vs binval), each presenting
a different difficulty landscape while sharing the same state space and action space.

## Architecture

```
BitStringConfig.build()
│
├─ fitness_fn = None (default)
│   └─ BitStringGame(BitStringGym(...))
│      Reward: ±1/N per step (flip 0→1 = +1/N, flip 1→0 = -1/N)
│
└─ fitness_fn = "onemax" | "leading_ones" | "binval"
    └─ BitStringGame(ShapedBitStringGym(BitStringGym(...)))
       Reward: (f(s_next) - f(s_prev)) / N  per step
```

Both paths terminate when all bits = 1 OR max_steps is reached.
The only difference is the **reward signal** the agent sees.

## Fitness Functions

Defined in `src/alphazeropp/instances/bitstring/potentials.py`.
All return `int` (not float) for exact arithmetic.

### onemax(x) → int
Sum of bits. Every 0→1 flip contributes equally.

```
x = [1, 0, 1, 0, 1] → onemax = 3
x = [1, 1, 1, 1, 1] → onemax = 5 (optimal)
```

**Difficulty:** Easiest. Any 0→1 flip gives +1/N reward. No deceptive gradients.

### leading_ones(x) → int
Length of the leading all-ones prefix. Only the leftmost gap matters.

```
x = [1, 1, 1, 0, 1] → leading_ones = 3
x = [0, 1, 1, 1, 1] → leading_ones = 0
x = [1, 1, 1, 1, 1] → leading_ones = 5 (optimal)
```

**Difficulty:** Harder. Flipping a bit at position 4 gives zero reward if position 3
is still 0. The agent must learn to fix bits left-to-right.

### binval(x) → int
Binary value with x[0] as the most significant bit.

```
x = [1, 0, 1] → binval = 5  (1×4 + 0×2 + 1×1)
x = [0, 0, 1] → binval = 1  (0×4 + 0×2 + 1×1)
x = [1, 1, 1] → binval = 7  (optimal)
```

**Difficulty:** Hardest. Flipping x[0] gives ±2^(N-1)/N reward, while flipping x[N-1]
gives ±1/N. The agent must learn to prioritize high-order bits.

## How It Works

### Reward Shaping via ShapedBitStringGym

`src/alphazeropp/instances/bitstring/shaped_env.py`

This is a `gym.Wrapper` around `BitStringGym` that replaces the reward:

```
Original BitStringGym reward:    ±1/N  (always same magnitude)
ShapedBitStringGym reward:       (f(s_next) - f(s_prev)) / N
```

The shaped reward is a **potential-based** shaping (Ng et al., 1999), meaning:
- The optimal policy is preserved (same as unshped MDP)
- The reward telescopes: sum of rewards = (f(s_T) - f(s_0)) / N exactly

### Integration into Training Pipeline

Three files were modified to connect the wrapper:

**1. `src/alphazeropp/instances/bitstring/game.py`** — `BitStringGame.__init__` now
accepts an optional `env` parameter. When provided, it uses that env instead of
creating its own `BitStringGym`:

```python
class BitStringGame(EnvGame):
    def __init__(self, env=None, **kwargs):
        if env is None:
            env = BitStringGym(**kwargs)   # Default: same as before
        super().__init__(env)
        self.action_mask = np.ones(env.n_sites, dtype=bool)
```

**2. `src/alphazeropp/instances/bitstring/config.py`** — `BitStringConfig` has two
new kwargs: `fitness_fn` and `reward_mode`. The `build()` method pops them before
creating `BitStringGym` (to avoid unexpected keyword errors), then conditionally
wraps with `ShapedBitStringGym`:

```python
game_kwargs = dict(self.game.kwargs)
fitness_fn_name = game_kwargs.pop("fitness_fn", None)
reward_mode = game_kwargs.pop("reward_mode", "dense_potential")

if fitness_fn_name is not None:
    base_env = BitStringGym(**game_kwargs)
    shaped_env = ShapedBitStringGym(base_env, POTENTIAL_REGISTRY[fitness_fn_name], reward_mode)
    game = BitStringGame(env=shaped_env)
else:
    game = BitStringGame(**game_kwargs)
```

**3. `scripts/run_bitstring.py`** — `fitness_fn` and `reward_mode` appear in the
interactive config menu, experiment directory naming, and startup banner.

## Usage

### Training with a fitness function

```bash
python scripts/run_bitstring.py
```

In the interactive config menu, set:
```
fitness_fn = onemax        # or leading_ones, binval, None
reward_mode = dense_potential
```

Then type `run` to start training.

### Sanity check (standalone)

```bash
python scripts/e1_sanity.py --task onemax --N 10 --H 200 --seed 0
# Telescoping check: max_abs_error=0.0
# Greedy oracle solve_rate=1.0
```

### Tests

```bash
pytest tests/test_bitstring_shaped.py -v --noconftest
# 41 tests: potentials, telescoping identity, greedy oracle, frozen states
```

## File Index

| File | Role |
|---|---|
| `src/alphazeropp/instances/bitstring/potentials.py` | Fitness functions + registry |
| `src/alphazeropp/instances/bitstring/shaped_env.py` | `ShapedBitStringGym` wrapper + `make_initial_states()` |
| `src/alphazeropp/instances/bitstring/game.py` | Modified: `BitStringGame(env=None, **kwargs)` |
| `src/alphazeropp/instances/bitstring/config.py` | Modified: `fitness_fn`/`reward_mode` in kwargs + `build()` |
| `scripts/run_bitstring.py` | Modified: interactive config, dir naming, banner |
| `tests/test_bitstring_shaped.py` | 41 tests |
| `scripts/e1_sanity.py` | Standalone sanity check |
