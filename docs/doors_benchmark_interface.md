# Doors Benchmark Interface

## Action Space

The `DoorsPDDLLiteEnv` action space contains only semantically real actions:

```
action_space = Discrete(M + K + 1)

  [0..M-1]       MOVE_TO(l)   Teleport to location l (noop if room locked)
  [M..M+K-1]     PICK(k)      Pick key k (requires at key_loc AND key available)
  [M+K]          NOOP         Do nothing
```

Where `M = num_rooms * locs_per_room` and `K = num_rooms - 1`.

### Old interface (before 2026-03-09)

The action space was padded to `Discrete(obs_size)` so that `output_size == input_size` in the network. Actions at indices `[M+K+1, obs_size-1]` were invalid padding, always masked out by MCTS. This conflated network architecture with environment semantics.

### Why padding was removed

1. `action_space.n` should reflect the true branching factor for fair algorithm comparison.
2. RL baselines without masking (e.g., vanilla PPO) wasted exploration on dead actions.
3. The AlphaZero core is fully generic (`len(nn_policy)` for sizing) and works with any `output_size`.

## Helpers

```python
env.encode_action("move", loc)   # -> int
env.encode_action("pick", key)   # -> int
env.encode_action("noop")        # -> int

env.decode_action(action_id)     # -> (type_str, param)
```

## Masking

```python
env.action_masks("none")          # all-True (invalid semantics become noop)
env.action_masks("precondition")  # only currently legal actions
```

The `DoorsDirectGame` wrapper also exposes `get_action_mask()` for the AlphaZero pipeline, with modes controlled by `use_precondition_mask`.
