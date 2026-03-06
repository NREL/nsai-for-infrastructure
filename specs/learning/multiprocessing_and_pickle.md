# Foundational Concepts: Multiprocessing, Pickle, and Caching

**Date:** 2026-03-06
**Context:** Notes from debugging the LeafEvaluator stats inflation bug.

---

## 1. Workers and Pools

### The problem: one CPU core is slow

A Python program normally runs on **one CPU core**. If you need to play 200 self-play games and each takes 30 seconds, that's 200 x 30 = 6,000 seconds = 100 minutes.

### The solution: parallel workers

Your computer has multiple CPU cores. A **worker** is a separate Python process running on a different core. A **pool** is Python's mechanism for managing a group of workers.

```
Analogy: Restaurant Kitchen

  Head chef (main process)
    "I have 200 dishes to prepare."

  4 line cooks (workers in a pool)
    Each cook prepares ~50 dishes independently.
    They share the same recipe book (neural network weights)
    but work on separate dishes at their own stations.

  Result: 200 dishes done in 1/4 the time.
```

In code:

```python
from multiprocessing import Pool

pool = Pool(processes=4)           # Hire 4 cooks
results = pool.starmap(fn, tasks)  # Hand out 200 tasks
#                      ↑    ↑
#         function to run    list of 200 (game_id, seed) tuples
```

The pool distributes tasks across workers automatically. When a worker finishes one task, the pool gives it the next one.

### What CAN be parallelized?

Only work where tasks are **independent**. In AlphaZero:

```
SELF-PLAY GAMES: Independent (parallelizable)

  Game 1: "Build a program using MCTS"  ←─ doesn't need
  Game 2: "Build a program using MCTS"  ←─ to know about
  Game 3: "Build a program using MCTS"  ←─ other games

  Each game uses the same neural network (read-only),
  but explores its own search tree independently.


NETWORK TRAINING: Sequential (not parallelized here)

  Must wait for ALL games to finish before training,
  because the network needs all the data at once.
```

This is the standard AlphaZero approach. DeepMind's original paper used 5,000 TPUs for self-play in parallel.

---

## 2. Pickle: Sending Objects Between Processes

### The problem: processes don't share memory

Each worker is a separate Python process with its own memory. It cannot see the main process's variables. So how does a worker get the game object, the neural network, and the leaf evaluator?

### The solution: serialize, send, deserialize

**Pickle** is Python's built-in serialization protocol. It converts a Python object into a stream of bytes (like saving to a file), which can be sent to another process and reconstructed there.

```
MAIN PROCESS                              WORKER PROCESS

game = DerivationGame(...)
game.leaf_evaluator._eval_count = 500
game.leaf_evaluator._cache = {"prog_A": 0.3}
        │
        │  pickle.dumps(game)
        ▼
  [bytes: b'\x80\x05\x95...' ]     ──── send via pipe ────>
                                                │
                                                │  pickle.loads(bytes)
                                                ▼
                                    game_copy = DerivationGame(...)
                                    game_copy.leaf_evaluator._eval_count = 500
                                    game_copy.leaf_evaluator._cache = {"prog_A": 0.3}
```

Key properties:
- The worker gets an **independent copy** (a clone, not a reference)
- Changes in the worker do NOT affect the original
- Changes in the main do NOT affect the worker's copy
- **Every field** is copied, including internal counters like `_eval_count`

### Pickle size matters

Pickle must convert every field into bytes. Bigger objects = more bytes = slower:

```
Small object:   game config, a few numbers     →  ~1 KB   → instant
Medium object:  cache with 100K entries         →  ~20 MB  → noticeable
Large object:   cache with 1M entries           →  ~200 MB → slow
```

When you pickle the same object 200 times (once per task), the total serialization work is:

```
200 tasks × 200 MB per pickle = 40 GB of serialization work
```

This is pure overhead — it's not doing any useful computation, just copying data.

### Customizing pickle with `__getstate__`

Python lets you customize what gets pickled by defining `__getstate__`:

```python
class LeafEvaluator:
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't send the cache to workers — they don't need it
        state['_cache'] = {}
        state['_eval_count'] = 0
        return state
```

Now when Python pickles a LeafEvaluator, it calls `__getstate__()` instead of copying `__dict__` directly. The worker receives a lightweight version with empty caches.

---

## 3. Caching: Remembering Previous Results

### The concept

A **cache** is a lookup table that stores results of expensive computations so you don't redo them.

```
Analogy: Teacher grading essays

  Without cache:
    Student submits "Essay A" → teacher reads it → grade: B+
    Student submits "Essay A" again → teacher reads it AGAIN → grade: B+
    (Wasted 30 minutes re-reading)

  With cache:
    Student submits "Essay A" → teacher reads it → grade: B+ → writes in gradebook
    Student submits "Essay A" again → teacher checks gradebook → B+!
    (Saved 30 minutes)
```

### In our code

The `LeafEvaluator` evaluates complete programs by running them on test puzzles. This is expensive (many environment steps). The cache avoids re-evaluating the same program:

```python
def __call__(self, program):
    key = program.pretty()          # "if IsZero(0): Flip(1) else: Flip(0)"

    if key in self._cache:          # Already graded this essay?
        self._cache_hits += 1       # Yes — record the shortcut
        return self._cache[key]     # Return stored grade

    metrics = self._evaluate(program)  # No — actually run the program (expensive)
    value = self._compute_metric(metrics)
    self._cache[key] = value        # Store for next time
    return value
```

### Cache hit vs cache miss

```
MCTS simulation 1: Reaches complete program "if IsZero(0): Flip(1) ..."
  → Cache MISS (first time seeing this program)
  → Run it on test puzzles → score = 0.35
  → Store in cache: {"if IsZero(0): Flip(1) ...": 0.35}
  → _eval_count: 0 → 1

MCTS simulation 47: Reaches the SAME program again
  → Cache HIT (already in the lookup table)
  → Return 0.35 immediately, no computation needed
  → _cache_hits: 0 → 1
  → _eval_count stays at 1 (no new work done)
```

### Why workers don't benefit from the main's cache

Each worker explores different random paths through the search tree (different random seeds, different Dirichlet noise). They almost never produce the same programs as each other or as previous iterations. This is confirmed by the data:

```
cache_hits = 0  across all workers
```

The cache IS useful within a single game (MCTS simulations within one game often revisit the same programs), but NOT useful across games or across iterations.

---

## 4. Putting It All Together: The Full Training Loop

### Baby example: 3 games, 1 worker (sequential)

```
ITERATION 1: "Play games and learn"

  leaf_evaluator starts with: _eval_count = 0, cache = {}

  Game 1:
    MCTS runs 200 simulations per move.
    Some simulations complete full programs → leaf_evaluator scores them.
    8 unique programs evaluated.
    _eval_count: 0 → 8,   cache: {} → {prog_A: 0.1, prog_B: -0.3, ...}

  Game 2:
    MCTS explores different paths, finds 7 new programs.
    3 programs match ones from Game 1 → cache hits (free!).
    _eval_count: 8 → 15,  cache grows to 15 entries

  Game 3:
    6 new programs, 2 cache hits.
    _eval_count: 15 → 21

  Total: _eval_count = 21 (correct)

  Now train the neural network on data from all 3 games.

ITERATION 2: "Play more games with improved network"

  leaf_evaluator continues with: _eval_count = 21, cache = {21 entries}

  Game 4: 9 new programs.  _eval_count: 21 → 30
  Game 5: 5 new programs.  _eval_count: 30 → 35
  Game 6: 7 new programs.  _eval_count: 35 → 42

  Total: _eval_count = 42 (correct, linear growth)

  Train network again...
```

With one worker, everything is simple and correct. The counter grows linearly: 21, 42, 63, 84...

### The same example with 3 workers (parallel) — THE BUG

```
ITERATION 1: Start with _eval_count = 0

  Main pickles game (with _eval_count = 0) for each task.
  Each worker gets an independent copy starting at 0.

  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │   Worker 1    │  │   Worker 2    │  │   Worker 3    │
  │   start: 0    │  │   start: 0    │  │   start: 0    │
  │   +7 evals    │  │   +5 evals    │  │   +9 evals    │
  │   end: 7      │  │   end: 5      │  │   end: 9      │
  │   exports: 7  │  │   exports: 5  │  │   exports: 9  │
  └───────────────┘  └───────────────┘  └───────────────┘

  Main merges: 0 + 7 + 5 + 9 = 21   CORRECT


ITERATION 2: Start with _eval_count = 21

  Main pickles game (with _eval_count = 21) for each task.
  Each worker gets a copy starting at 21.        ← THE PROBLEM

  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │   Worker 1    │  │   Worker 2    │  │   Worker 3    │
  │   start: 21   │  │   start: 21   │  │   start: 21   │
  │   +8 evals    │  │   +6 evals    │  │   +7 evals    │
  │   end: 29     │  │   end: 27     │  │   end: 28     │
  │   exports: 29 │  │   exports: 27 │  │   exports: 28 │
  └───────────────┘  └───────────────┘  └───────────────┘

  Each worker reported 21 (inherited) + their own work.
  The "21" is NOT work they did — it's a copy of the main's counter.

  Main merges: 21 + 29 + 27 + 28
             = 21 + (21+8) + (21+6) + (21+7)
             = 21 + 3*21 + (8+6+7)
             = 4*21 + 21
             = 105

  CORRECT answer: 21 + 8 + 6 + 7 = 42
  BUG answer:     105              (2.5x inflated)
```

The multiplier = `1 + number_of_tasks`. With 3 tasks: 1 + 3 = 4x.

### Your experiment: 200 tasks

```
multiplier = 1 + 200 = 201

ITERATION 1:  E = 0
  201 * 0 + W = W = 1,739,000
  (No inflation because 201 * 0 = 0)

ITERATION 2:  E = 1,739,000
  201 * 1,739,000 + W ≈ 351,000,000
  (Should be ~3,478,000 — inflated 101x)

ITERATION 3:  E = 351,000,000
  201 * 351,000,000 + W ≈ 70,600,000,000
  (Should be ~5,217,000 — inflated 13,500x)

ITERATION 4:  E = 70,600,000,000
  201 * 70,600,000,000 + W ≈ 14,200,000,000,000
  (Should be ~6,956,000 — inflated 2,041,000x)
```

### The formula: why 201 * E

```
E_new = E                    (main keeps its own counter)
      + task_0_export         (= E + w_0)
      + task_1_export         (= E + w_1)
      + ...
      + task_199_export       (= E + w_199)

     = E + 200*E + (w_0 + w_1 + ... + w_199)
     = 201*E + W
       ↑
       This is the bug: 200 copies of E
       that shouldn't be there
```

### After the fix

Workers start with `_eval_count = 0` (via `__getstate__`), so exports are pure deltas:

```
ITERATION 2 (fixed):  E = 21

  Workers start at 0, do work, export just their work:
  Worker 1 exports: 8    (not 29)
  Worker 2 exports: 6    (not 27)
  Worker 3 exports: 7    (not 28)

  Main merges: 21 + 8 + 6 + 7 = 42   CORRECT
```

---

## 5. Key Takeaways

| Concept | What it is | Analogy |
|---------|-----------|---------|
| **Worker** | Separate Python process on another CPU core | Line cook in a kitchen |
| **Pool** | Manager that distributes tasks to workers | Head chef assigning dishes |
| **Pickle** | Converting objects to bytes to send between processes | Photocopying a recipe to hand to a cook |
| **Cache** | Lookup table storing previous results | Gradebook remembering essay scores |
| **`__getstate__`** | Hook to customize what gets pickled | Giving cooks a blank notepad instead of your full gradebook |

### The bug in one sentence

Workers inherited the main's cumulative counter, reported it back as their own work, and the main added it 200 times — turning linear growth into exponential (201^N).

### The fix in one sentence

Workers start with zeroed counters (via `__getstate__`), so they only report new work done, and the main accumulates correctly.
