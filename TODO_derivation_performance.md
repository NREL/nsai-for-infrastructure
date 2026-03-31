# DerivationGame Performance TODO

## Problem

With current defaults (budget=14, N=6), training hangs at iteration 1/30
with no output. The program is not deadlocked — it's doing ~200K sequential
neural net forward passes per iteration with zero progress logging.

## Cost Breakdown Per Iteration

| Phase              | Games | Steps/game | MCTS sims/step | Total predict() calls |
|--------------------|-------|------------|----------------|-----------------------|
| Self-play          | 50    | ~9         | 250            | 112,500               |
| Eval (new agent)   | 20    | ~9         | 250            | 45,000                |
| Eval (old agent)   | 20    | ~9         | 250            | 45,000                |
| **Total**          |       |            |                | **~202,500**          |

Each predict() is a Transformer forward pass (d=64, 2 layers) on CPU at ~2ms.
Estimated: ~7 min pure inference + overhead = **10-15 min per iteration**.
30 iterations = **5-7.5 hours**.

## Root Causes

1. **n_simulations=250 is too high** for initial experimentation. This is the
   dominant cost — 250 MCTS sims × 9 derivation steps × (50+40) games = 202K
   forward passes per iteration.

2. **No progress output during self-play**. Between `print_iteration_header`
   and `print_iteration_summary` in `run_derivation.py`, there is zero output.
   The user sees "--- Iteration 1/30 ---" then nothing for 10+ minutes.

3. **GatedTrainer evaluation doubles the work**. After every training
   iteration, the evaluator pits new vs old agent — both play 20 games each
   with full MCTS (250 sims/step), adding ~90K more forward passes.

4. **Leaf evaluator overhead in MCTS search**. Each time MCTS reaches a
   complete program, `leaf_evaluator._evaluate()` creates 15 fresh
   ShapedBitStringGym environments and runs full episodes. Not the bottleneck,
   but adds up.

## Proposed Fixes

### Quick fix: Reduce hyperparameters for tractability

In `derivation_config.py`:

| Param               | Current | Proposed | Impact                          |
|----------------------|---------|----------|---------------------------------|
| `n_simulations`      | 250     | 50       | 5x fewer forward passes         |
| `n_games_per_train`  | 50      | 20       | 2.5x fewer self-play games      |
| `n_games` (eval)     | 20      | 10       | 2x fewer eval games             |

Revised total: ~18K predict() calls per iteration → ~36 sec → **~18 min for 30 iterations** (20x speedup).

### Structural fix: Add progress logging

In `trainer.py` `_collect_training_examples()`, add per-game progress output
when running sequentially (n_procs < 0), e.g.:
```
[TRAIN] Game 1/20 complete (9 steps, 3.2s)
[TRAIN] Game 2/20 complete (9 steps, 2.8s)
```

## Theoretical Targets for Budget=14, N=6

```
Action space:         60
Total programs:       151,173,432
Derivation depth:     9 steps
Frozen eval states:   15 (C(6,2))
Max avg_reward:       0.4444 (budget manages 5/6 bits)
Max solve_rate:       33%
```
