"""Solve criterion for the benchmark harness.

A run is considered "solved" when eval_success_rate and eval_return_mean
exceed thresholds for a sustained number of consecutive checkpoints.
"""

from __future__ import annotations

from alphazeropp.benchmark.result_schema import CheckpointResult
from alphazeropp.instances.doors.oracle import optimal_return


def check_sustained_solve(
    checkpoints: list[CheckpointResult],
    D: int,
    success_threshold: float = 0.95,
    return_ratio_threshold: float = 0.95,
    sustained_count: int = 3,
) -> tuple[int, float] | None:
    """Check if the solve criterion is met.

    Parameters
    ----------
    checkpoints : list[CheckpointResult]
        Evaluation checkpoints in chronological order.
    D : int
        Number of rooms (for oracle return computation).
    success_threshold : float
        Minimum eval_success_rate.
    return_ratio_threshold : float
        Minimum eval_return_mean / oracle_return(D).
    sustained_count : int
        Number of consecutive checkpoints that must meet thresholds.

    Returns
    -------
    (solve_env_steps, solve_wall_clock_sec) or None
        The env_steps and wall_clock_sec of the earliest checkpoint where
        the sustained criterion is first met. None if not yet solved.
    """
    if len(checkpoints) < sustained_count:
        return None

    oracle_ret = optimal_return(D)
    return_threshold = return_ratio_threshold * oracle_ret

    # Track consecutive passing checkpoints
    consecutive = 0
    for cp in checkpoints:
        passes = (
            cp.eval_success_rate >= success_threshold
            and cp.eval_return_mean >= return_threshold
        )
        if passes:
            consecutive += 1
            if consecutive >= sustained_count:
                return (cp.env_steps, cp.wall_clock_sec)
        else:
            consecutive = 0

    return None
