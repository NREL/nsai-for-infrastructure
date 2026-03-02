#!/usr/bin/env python3
"""
Verify order-dependence claims under implemented return definitions.

This script compares two 0->1 action orders from the same start state and
checks:
1) Undiscounted episode sum (gamma=1 objective): equal for all potentials.
2) Discounted return with gamma<1 (agent value-target style): may differ for
   leading_ones / binval, but not for onemax in this setup.

Default setup:
  n_sites = 6
  x0      = [1, 0, 1, 0, 0, 0]
  seq_A   = [1, 3, 4, 5]
  seq_B   = [5, 4, 3, 1]
  gamma   = 0.9
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np


N_SITES = 6
X0 = np.array([1, 0, 1, 0, 0, 0], dtype=np.float32)
SEQ_A = [1, 3, 4, 5]  # front-to-back
SEQ_B = [5, 4, 3, 1]  # back-to-front
GAMMA_TEST = 0.9


@dataclass
class SequenceResult:
    rewards: list[float]
    undiscounted_sum: float
    discounted_sum: float
    final_state: np.ndarray
    done: bool


def discounted_return(rewards: list[float], gamma: float) -> float:
    """Compute G = sum_t gamma^t r_t."""
    total = 0.0
    w = 1.0
    for r in rewards:
        total += w * float(r)
        w *= gamma
    return total


def _simulate_sequence_repo_env(
    potential_name: str,
    sequence: list[int],
) -> SequenceResult:
    """Run one sequence using repository BitStringGym + ShapedBitStringGym."""
    # Imported here so the script can still run in restricted environments
    # where torch-backed imports fail; a fallback path is provided in main().
    from alphazeropp.instances.bitstring.game import BitStringGym
    from alphazeropp.instances.bitstring.shaped_env import ShapedBitStringGym
    from alphazeropp.instances.bitstring.potentials import POTENTIAL_REGISTRY

    potential_fn = POTENTIAL_REGISTRY[potential_name]

    base_env = BitStringGym(
        n_sites=N_SITES,
        bit_flip=True,
        sparse_reward=False,
        n_ones=2,
    )
    env = ShapedBitStringGym(
        env=base_env,
        potential_fn=potential_fn,
        reward_mode="dense_potential",
        frozen_states=[X0],
    )

    obs, _ = env.reset()
    if not np.array_equal(obs, X0):
        raise AssertionError(
            f"Expected reset state {X0.tolist()}, got {obs.tolist()}"
        )

    rewards: list[float] = []
    done = False

    for action in sequence:
        if obs[action] != 0.0:
            raise AssertionError(
                f"Sequence is not strictly 0->1 at action {action}, state={obs}"
            )
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        done = bool(terminated or truncated)
        if done:
            break

    return SequenceResult(
        rewards=rewards,
        undiscounted_sum=float(sum(rewards)),
        discounted_sum=discounted_return(rewards, GAMMA_TEST),
        final_state=obs.copy(),
        done=done,
    )


def _onemax(x: np.ndarray) -> int:
    return int(np.sum(x))


def _leading_ones(x: np.ndarray) -> int:
    n = len(x)
    for i in range(n):
        if x[i] != 1:
            return i
    return n


def _binval(x: np.ndarray) -> int:
    result = 0
    for bit in x:
        result = 2 * result + int(bit)
    return result


def _simulate_sequence_fallback(
    potential_fn: Callable[[np.ndarray], int],
    sequence: list[int],
) -> SequenceResult:
    """
    Fallback simulator mirroring shaped reward:
      r_t = (Phi(s_{t+1}) - Phi(s_t)) / N.
    """
    obs = X0.copy()
    prev_phi = potential_fn(obs)
    rewards: list[float] = []

    for action in sequence:
        if obs[action] != 0.0:
            raise AssertionError(
                f"Sequence is not strictly 0->1 at action {action}, state={obs}"
            )
        obs[action] = 1.0 - obs[action]  # bit_flip=True
        curr_phi = potential_fn(obs)
        reward = (curr_phi - prev_phi) / N_SITES
        rewards.append(float(reward))
        prev_phi = curr_phi

    done = bool(np.all(obs == 1.0))
    return SequenceResult(
        rewards=rewards,
        undiscounted_sum=float(sum(rewards)),
        discounted_sum=discounted_return(rewards, GAMMA_TEST),
        final_state=obs.copy(),
        done=done,
    )


def _print_result(label: str, seq: list[int], res: SequenceResult) -> None:
    rewards_str = ", ".join(f"{r:+.6f}" for r in res.rewards)
    print(f"  {label}={seq}")
    print(f"    rewards: [{rewards_str}]")
    print(f"    undiscounted_sum: {res.undiscounted_sum:+.12f}")
    print(
        f"    discounted_sum(gamma={GAMMA_TEST}): {res.discounted_sum:+.12f}"
    )
    print(f"    done: {res.done}, final_state: {res.final_state.tolist()}")


def _assert_done_and_solved(name: str, res_a: SequenceResult, res_b: SequenceResult) -> None:
    for lbl, res in (("A", res_a), ("B", res_b)):
        if not res.done:
            raise AssertionError(f"{name} seq_{lbl} did not terminate")
        if not np.all(res.final_state == 1.0):
            raise AssertionError(
                f"{name} seq_{lbl} did not end at all-ones: {res.final_state}"
            )


def run_repo_mode() -> None:
    potential_names = ["onemax", "leading_ones", "binval"]
    tol_eq = 1e-12
    tol_neq = 1e-9

    print("Mode: repo_env")
    print(f"x0={X0.tolist()}, n_sites={N_SITES}, gamma_test={GAMMA_TEST}")
    print()

    for name in potential_names:
        res_a = _simulate_sequence_repo_env(name, SEQ_A)
        res_b = _simulate_sequence_repo_env(name, SEQ_B)

        print(f"Variant: {name}")
        _print_result("seq_A", SEQ_A, res_a)
        _print_result("seq_B", SEQ_B, res_b)

        _assert_done_and_solved(name, res_a, res_b)

        # Undiscounted telescoping check: equal for same (s0, sT).
        if abs(res_a.undiscounted_sum - res_b.undiscounted_sum) > tol_eq:
            raise AssertionError(
                f"{name}: expected equal undiscounted sums, got "
                f"{res_a.undiscounted_sum} vs {res_b.undiscounted_sum}"
            )

        # Discounted exception check.
        disc_diff = abs(res_a.discounted_sum - res_b.discounted_sum)
        if name == "onemax":
            if disc_diff > tol_eq:
                raise AssertionError(
                    f"{name}: expected equal discounted sums, got "
                    f"{res_a.discounted_sum} vs {res_b.discounted_sum}"
                )
        else:
            if disc_diff <= tol_neq:
                raise AssertionError(
                    f"{name}: expected different discounted sums, got "
                    f"{res_a.discounted_sum} vs {res_b.discounted_sum}"
                )

        print("  assertions: PASS")
        print("-" * 72)

    print("checks passed")


def run_fallback_mode(import_error: Exception) -> None:
    print(
        "Mode: fallback_simulation "
        "(repository env import failed in this runtime)."
    )
    print(f"Import error: {type(import_error).__name__}: {import_error}")
    print(f"x0={X0.tolist()}, n_sites={N_SITES}, gamma_test={GAMMA_TEST}")
    print()

    potentials = {
        "onemax": _onemax,
        "leading_ones": _leading_ones,
        "binval": _binval,
    }
    tol_eq = 1e-12
    tol_neq = 1e-9

    for name, potential_fn in potentials.items():
        res_a = _simulate_sequence_fallback(potential_fn, SEQ_A)
        res_b = _simulate_sequence_fallback(potential_fn, SEQ_B)

        print(f"Variant: {name}")
        _print_result("seq_A", SEQ_A, res_a)
        _print_result("seq_B", SEQ_B, res_b)

        _assert_done_and_solved(name, res_a, res_b)

        if abs(res_a.undiscounted_sum - res_b.undiscounted_sum) > tol_eq:
            raise AssertionError(
                f"{name}: expected equal undiscounted sums, got "
                f"{res_a.undiscounted_sum} vs {res_b.undiscounted_sum}"
            )

        disc_diff = abs(res_a.discounted_sum - res_b.discounted_sum)
        if name == "onemax":
            if disc_diff > tol_eq:
                raise AssertionError(
                    f"{name}: expected equal discounted sums, got "
                    f"{res_a.discounted_sum} vs {res_b.discounted_sum}"
                )
        else:
            if disc_diff <= tol_neq:
                raise AssertionError(
                    f"{name}: expected different discounted sums, got "
                    f"{res_a.discounted_sum} vs {res_b.discounted_sum}"
                )

        print("  assertions: PASS")
        print("-" * 72)

    print("checks passed")


def main() -> None:
    if os.environ.get("CHECK_ORDER_FORCE_FALLBACK", "0") == "1":
        run_fallback_mode(RuntimeError("forced fallback (CHECK_ORDER_FORCE_FALLBACK=1)"))
        return
    try:
        run_repo_mode()
    except Exception as exc:
        run_fallback_mode(exc)


if __name__ == "__main__":
    main()
