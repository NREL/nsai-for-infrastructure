# Doors Benchmark Protocol

## What This Benchmark Measures

This benchmark evaluates how efficiently different algorithms learn to solve the Doors environment — a deterministic finite-horizon task where the agent must navigate through D linearly connected rooms by picking up keys in sequence.

The benchmark compares algorithms on:
- **Sample efficiency**: how many environment steps to reach optimal or near-optimal performance
- **Final performance**: return and success rate at convergence
- **Wall-clock time**: real-time to solve
- **Invalid action rate**: how quickly the agent learns preconditions

## Train/Eval Separation

Training and evaluation use **separate environment instances** created by `env_factory.make_env_factory()`. This ensures:
- Evaluation rewards are unmodified by any training wrappers
- Evaluation episodes use independent seeds (`seed + i` for episode `i`)
- Training exploration noise does not contaminate evaluation metrics

Evaluation uses the **greedy** policy: for AlphaZero, the network's masked argmax (no MCTS at eval time); for SB3 algorithms, `deterministic=True`.

## Logged Metrics

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | str | Algorithm name (e.g., "alphazero", "oracle", "random") |
| `seed` | int | Random seed for this run |
| `D` | int | Number of rooms |
| `locs_per_room` | int | Locations per room |
| `mask_mode` | str | "none" or "precondition" |
| `env_steps` | int | Cumulative training environment steps |
| `train_episodes` | int | Cumulative training episodes |
| `learner_updates` | int | Number of gradient updates (or iterations for AlphaZero) |
| `eval_checkpoint_idx` | int | Checkpoint sequence number |
| `eval_return_mean` | float | Mean episodic return over eval episodes |
| `eval_return_std` | float | Std of episodic return over eval episodes |
| `eval_success_rate` | float | Fraction of eval episodes that reached the goal |
| `eval_episode_length_mean` | float | Mean episode length |
| `max_room_reached_mean` | float | Mean maximum room index reached |
| `keys_picked_mean` | float | Mean number of keys picked |
| `invalid_action_rate` | float | Fraction of actions violating preconditions |
| `wall_clock_sec` | float | Cumulative wall-clock time |
| `solved_flag` | bool | Whether sustained solve criterion is met |
| `solve_env_steps` | int? | Env steps at which solve criterion was first met |
| `solve_wall_clock_sec` | float? | Wall-clock at which solve criterion was first met |

## Solve Criterion

A run is marked as **solved** when **all three conditions** are met for **3 consecutive** evaluation checkpoints:

1. `eval_success_rate >= 0.95`
2. `eval_return_mean >= 0.95 * oracle_return(D)`

where `oracle_return(D) = 1.0 + (D-1)*0.1 - (2*(D-1)+1)*0.01`.

The `solve_env_steps` and `solve_wall_clock_sec` record the earliest point where this sustained criterion is first satisfied.

## Reproducing a Single Run

```bash
# Oracle baseline (instant, no training)
python -m alphazeropp.benchmark.run --algo oracle --D 3 --seed 42

# Random baseline
python -m alphazeropp.benchmark.run --algo random --D 3 --seed 42

# AlphaZero (D=3, 10 iterations, sequential)
python -m alphazeropp.benchmark.run --algo alphazero --D 3 --seed 42 \
    --total-iterations 10 --n-procs -1 --output-dir experiments/benchmark
```

## Reproducing a Sweep

```bash
for seed in 1 2 3; do
  for algo in oracle random alphazero; do
    python -m alphazeropp.benchmark.run \
        --algo $algo --D 5 --seed $seed \
        --total-iterations 50 --eval-episodes 100 \
        --output-dir experiments/benchmark/D5_sweep
  done
done
```

## Output Format

Each run produces two files in the output directory:
- `{algo}_D{D}_seed{seed}.csv` — tabular format for pandas analysis
- `{algo}_D{D}_seed{seed}.jsonl` — streaming JSON lines, one per checkpoint

### Example CSV Row

```
algorithm,seed,D,locs_per_room,mask_mode,env_steps,train_episodes,learner_updates,eval_checkpoint_idx,eval_return_mean,eval_return_std,eval_success_rate,eval_episode_length_mean,max_room_reached_mean,keys_picked_mean,invalid_action_rate,wall_clock_sec,solved_flag,solve_env_steps,solve_wall_clock_sec
oracle,42,3,2,none,0,0,0,0,1.15,0.0,1.0,5.0,2.0,2.0,0.0,0.01,True,0,0.0
```
