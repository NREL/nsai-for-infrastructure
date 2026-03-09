"""Sweep runner: run multiple algorithms across D values and seeds.

Usage:
    python -m alphazeropp.benchmark.sweep --D 2 3 5 --seeds 1 2 3 \
        --algos oracle random tabular_q --output-dir experiments/stage3

    python -m alphazeropp.benchmark.sweep --D 2 3 --seeds 42 \
        --algos oracle random tabular_q --mask-mode precondition \
        --output-dir experiments/stage3 --plot
"""

from __future__ import annotations

import argparse
import logging
import sys

from alphazeropp.benchmark.run import main as run_main


logger = logging.getLogger(__name__)

# Default episode budgets for tabular_q by D
_TABULAR_Q_BUDGETS = {
    2: 500,
    3: 2000,
    5: 5000,
}
_TABULAR_Q_DEFAULT_BUDGET = 10000


def _tabular_q_budget(D: int) -> int:
    return _TABULAR_Q_BUDGETS.get(D, _TABULAR_Q_DEFAULT_BUDGET)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run benchmark sweep across algorithms, D values, and seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--D", type=int, nargs="+", required=True,
                        help="Room counts to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds to sweep")
    parser.add_argument("--algos", nargs="+",
                        default=["oracle", "random", "tabular_q"],
                        help="Algorithms to run")
    parser.add_argument("--mask-mode", default="none",
                        choices=["none", "precondition"],
                        help="Action masking mode")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Evaluation episodes per checkpoint")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Eval interval for learning algos (default: budget/5)")
    parser.add_argument("--output-dir", default="experiments/benchmark",
                        help="Output directory")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots after sweep completes")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total_runs = len(args.algos) * len(args.D) * len(args.seeds)
    run_idx = 0

    for D in args.D:
        for seed in args.seeds:
            for algo in args.algos:
                run_idx += 1
                logger.info("[%d/%d] algo=%s D=%d seed=%d",
                            run_idx, total_runs, algo, D, seed)

                run_args = [
                    "--algo", algo,
                    "--D", str(D),
                    "--seed", str(seed),
                    "--mask-mode", args.mask_mode,
                    "--eval-episodes", str(args.eval_episodes),
                    "--output-dir", args.output_dir,
                ]

                # Set training budget for learning algorithms
                if algo in ("tabular_q",):
                    budget = _tabular_q_budget(D)
                    interval = args.eval_interval or max(1, budget // 5)
                    run_args += [
                        "--total-iterations", str(budget),
                        "--eval-interval", str(interval),
                    ]
                elif algo in ("alphazero",):
                    run_args += ["--total-iterations", "50"]
                    if args.eval_interval:
                        run_args += ["--eval-interval", str(args.eval_interval)]

                try:
                    run_main(run_args)
                except Exception:
                    logger.exception("FAILED: algo=%s D=%d seed=%d", algo, D, seed)

    logger.info("Sweep complete: %d runs", total_runs)

    if args.plot:
        from alphazeropp.benchmark.plot_cli import main as plot_main
        plot_main([args.output_dir])


if __name__ == "__main__":
    main()
