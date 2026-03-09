# Doors Environment Audit

Audit date: 2026-03-09

## Environment Specification

**Class**: `DoorsPDDLLiteEnv` in `src/alphazeropp/instances/doors/doors_pddl_lite.py`

| Parameter | Formula |
|-----------|---------|
| Locations (M) | `D * locs_per_room` |
| Keys (K) | `D - 1` |
| Observation size | `M + 2D - 1` |
| Action count | `M + K + 1` (MOVE + PICK + NOOP) |
| Optimal steps | `2*(D-1) + 1` |
| Optimal return | `1.0 + K*0.1 - optimal_steps*0.01` |

## Verified Properties

1. **Monotone unlocking**: Key k is in room k, unlocks room k+1. Keys must be picked in order 0..K-1.
2. **Reachable states**: `locs_per_room * D * (D+1) / 2` — verified by BFS enumeration for D in {2, 3, 5}.
3. **Oracle optimality**: General oracle solves all tested D values in exactly `2*(D-1)+1` steps.
4. **Reward accounting**: Oracle cumulative reward matches the closed-form formula for all tested D.

## Audit Results (locs_per_room=2)

|  D | M   | K  | Actions | Reachable | Opt Steps | Opt Return |
|---:|----:|---:|--------:|----------:|----------:|-----------:|
|  2 |   4 |  1 |       6 |         6 |         3 |     1.0700 |
|  3 |   6 |  2 |       9 |        12 |         5 |     1.1500 |
|  5 |  10 |  4 |      15 |        30 |         9 |     1.3100 |
| 10 |  20 |  9 |      30 |       110 |        19 |     1.7100 |
| 20 |  40 | 19 |      60 |       420 |        39 |     2.5100 |
| 50 | 100 | 49 |     150 |      2550 |        99 |     4.9100 |

## How to Run

```bash
# Audit script (prints table above)
python scripts/audit_doors_env.py

# Tests (oracle, state enumeration, reward, horizon, sb3 check)
pytest tests/test_doors_oracle.py -v
```
