#!/usr/bin/env python3
"""Audit the Doors environment for multiple D values.

Prints a table of reachable states, action count, optimal steps/return,
and verifies the oracle solves each instance optimally.

Usage:
    python scripts/audit_doors_env.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig,
    compute_doors_derived_params,
)
from alphazeropp.instances.doors.oracle import (
    optimal_return,
    optimal_steps,
    reachable_state_count,
    run_oracle_episode,
)


def audit(D_values=(2, 3, 5, 10, 20, 50), locs_per_room=2):
    header = (
        f"{'D':>3}  {'M':>4}  {'K':>3}  {'Actions':>7}  "
        f"{'Reachable':>9}  {'Opt Steps':>9}  {'Opt Return':>10}  "
        f"{'Oracle OK':>9}"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for D in D_values:
        params = compute_doors_derived_params(D, locs_per_room)
        M = params["M"]
        K = params["K"]
        action_count = M + K + 1
        n_reachable = reachable_state_count(D, locs_per_room)
        opt_steps = optimal_steps(D)
        opt_ret = optimal_return(D)

        # Run oracle
        cfg = DoorsGameConfig(
            num_rooms=D,
            locs_per_room=locs_per_room,
            horizon=params["horizon"],
        )
        env = cfg.make_env(cfg.obs_size())
        result = run_oracle_episode(env)

        oracle_ok = (
            result["solved"]
            and result["steps"] == opt_steps
            and abs(result["reward"] - opt_ret) < 1e-6
        )
        if not oracle_ok:
            all_ok = False

        status = "PASS" if oracle_ok else "FAIL"
        print(
            f"{D:>3}  {M:>4}  {K:>3}  {action_count:>7}  "
            f"{n_reachable:>9}  {opt_steps:>9}  {opt_ret:>10.4f}  "
            f"{status:>9}"
        )

        if not oracle_ok:
            print(f"     -> oracle: solved={result['solved']}, "
                  f"steps={result['steps']}, reward={result['reward']:.6f}")

    print()
    if all_ok:
        print("All audits passed.")
    else:
        print("SOME AUDITS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    audit()
