# Refined Plan: Ablation Semantic Audit

**Date:** 2026-03-09
**Status:** COMPLETE — all tests pass, semantic map verified

---

## What Was Done

1. **Created `docs/ablation_semantics.md`** — Verified semantic map answering 9 questions about MCTS/network/trainer interactions with exact file:line references.

2. **Created `src/alphazeropp/instances/doors/network_ablations.py`** — Three ablation wrapper classes (FrozenWrapper, UniformPolicyWrapper, ZeroValueWrapper) with pickle-safe `__getattr__` delegation.

3. **Created `tests/test_ablation_semantics.py`** — 24 test cases across 11 test classes verifying all semantic assumptions.

---

## Confirmed Assumptions

| # | Assumption | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `n_simulations=-1` returns raw net policy | **Confirmed** | `test_no_mcts_returns_legal_probs` passes |
| 2 | Legal masking applied in `query_net_masked()` | **Confirmed** | `test_mcts_respects_mask_with_*` pass |
| 3 | Dirichlet noise disabled with `add_noise=False` | **Confirmed** | `test_deterministic_without_noise` passes |
| 4 | Agent and trainer share same net object | **Confirmed** | Code trace + `gated_trainer.py:53-54` comment |
| 5 | Evaluator has no stored net reference | **Confirmed** | Constructor takes only `n_games`, `n_procs` |
| 6 | Rollouts off by default (`rollout_n=0`) | **Confirmed** | `mcts.py:54` default |
| 7 | `net.train()` returns 5-tuple | **Confirmed** | `test_frozen_train_returns_5tuple` passes |
| 8 | Wrappers delegate via `__getattr__` correctly | **Confirmed** | `test_wrapper_getattr_delegation` passes |
| 9 | Wrappers survive push/pop multiprocessing | **Confirmed** | `test_wrapper_push_pop` passes |
| 10 | Wrappers survive pickle round-trip | **Confirmed** | `test_wrapper_pickle_roundtrip` passes (after fix) |

## Broken Assumptions (Discrepancies)

| # | Assumption | Status | Fix |
|---|-----------|--------|-----|
| 1 | `n_simulations=0` returns useful policy | **BROKEN** | Returns all zeros. Use `n_sims=1` minimum. |
| 2 | `__getattr__` works with pickle naively | **BROKEN** | Infinite recursion during unpickling. Fixed with `_safe_getattr()`. |
| 3 | `--no-mcts` gives raw network output | **PARTIALLY BROKEN** | Temperature scaling still applied. Must force `temperature=1.0`. |

## Lines/Functions to Modify in Later Stages

| File | Function/Line | Modification |
|------|--------------|-------------|
| `scripts/run_doors_direct.py` | `parse_args()` ~line 1219 | Add `--ablation`, `--no-mcts`, `--override` CLI args |
| `scripts/run_doors_direct.py` | `_run_single_seed()` ~line 1090 | After `cfg.build()`, inject wrapper: `agent.net = trainer.net = Wrapper(net)` |
| `scripts/run_doors_direct.py` | `_run_single_seed()` | When `--no-mcts`: set `n_simulations=-1` AND `temperature=1.0` |
| `scripts/run_doors_direct.py` | `setup_experiment_dir()` ~line 150 | Append ablation mode to directory name |

---

## Test Results

```
24 passed in 1.96s

tests/test_ablation_semantics.py::TestWrapperDelegation            5/5  PASSED
tests/test_ablation_semantics.py::TestFrozenTrainSignature         3/3  PASSED
tests/test_ablation_semantics.py::TestFrozenPredictUnchanged       1/1  PASSED
tests/test_ablation_semantics.py::TestUniformPolicyWrapper         2/2  PASSED
tests/test_ablation_semantics.py::TestZeroValueWrapper             2/2  PASSED
tests/test_ablation_semantics.py::TestMaskingUnderWrapper          2/2  PASSED
tests/test_ablation_semantics.py::TestNoMCTSReturnsLegalProbs      2/2  PASSED
tests/test_ablation_semantics.py::TestZeroSimsReturnsAllZeros      1/1  PASSED
tests/test_ablation_semantics.py::TestEvalNoiseDisabled            1/1  PASSED
tests/test_ablation_semantics.py::TestWrapperPickle                3/3  PASSED
tests/test_ablation_semantics.py::TestWrapperPushPopMultiprocessing 2/2 PASSED
```
