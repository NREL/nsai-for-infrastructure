"""Benchmark CLI entrypoint.

Usage:
    python -m alphazeropp.benchmark.run --algo oracle --D 3 --seed 42
    python -m alphazeropp.benchmark.run --algo random --D 5 --seed 1 --eval-episodes 50
    python -m alphazeropp.benchmark.run --algo alphazero --D 2 --seed 42 \
        --total-iterations 10 --eval-episodes 20 --output-dir experiments/benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from alphazeropp.benchmark.env_factory import make_env_factory
from alphazeropp.benchmark.result_schema import ResultWriter
from alphazeropp.benchmark.solve_criterion import check_sustained_solve


logger = logging.getLogger(__name__)


def _build_adapter(args):
    """Instantiate the appropriate BenchmarkAlgorithm from CLI args."""
    algo = args.algo

    if algo == "oracle":
        from alphazeropp.benchmark.adapters.oracle import OracleAdapter
        return OracleAdapter(D=args.D, locs_per_room=args.locs_per_room,
                             mask_mode=args.mask_mode)

    elif algo == "random":
        from alphazeropp.benchmark.adapters.random_agent import RandomAdapter
        return RandomAdapter(D=args.D, locs_per_room=args.locs_per_room,
                             mask_mode=args.mask_mode)

    elif algo == "alphazero":
        from alphazeropp.benchmark.adapters.alphazero import AlphaZeroAdapter
        return AlphaZeroAdapter(
            D=args.D,
            locs_per_room=args.locs_per_room,
            mask_mode=args.mask_mode,
            n_simulations=args.n_simulations,
            n_games_per_train=args.n_games_per_train,
            n_procs=args.n_procs,
            accept_threshold=args.accept_threshold,
        )

    elif algo == "heuristic":
        from alphazeropp.benchmark.adapters.oracle import OracleAdapter
        return OracleAdapter(D=args.D, locs_per_room=args.locs_per_room,
                             mask_mode=args.mask_mode)

    elif algo == "tabular_q":
        from alphazeropp.benchmark.adapters.tabular_q import TabularQAdapter
        return TabularQAdapter(
            D=args.D,
            locs_per_room=args.locs_per_room,
            mask_mode=args.mask_mode,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )

    elif algo in ("dqn", "maskable_ppo"):
        from alphazeropp.benchmark.adapters.sb3 import get_sb3_adapter
        return get_sb3_adapter(
            algo=algo, D=args.D, locs_per_room=args.locs_per_room,
            mask_mode=args.mask_mode,
        )

    else:
        print(f"Unknown algorithm: {algo}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Doors benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--algo", required=True,
                        choices=["oracle", "random", "heuristic", "tabular_q",
                                 "alphazero", "dqn", "maskable_ppo"],
                        help="Algorithm to benchmark")
    parser.add_argument("--D", type=int, required=True,
                        help="Number of rooms")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")

    # Environment
    parser.add_argument("--locs-per-room", type=int, default=2,
                        help="Locations per room")
    parser.add_argument("--mask-mode", default="none",
                        choices=["none", "precondition"],
                        help="Action masking mode")

    # Training budget
    parser.add_argument("--total-iterations", type=int, default=50,
                        help="Total training budget: iterations (AlphaZero), "
                             "episodes (tabular_q), timesteps (SB3)")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Evaluate every N training units")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per checkpoint")

    # AlphaZero-specific
    parser.add_argument("--n-simulations", type=int, default=120,
                        help="MCTS simulations per move (AlphaZero only)")
    parser.add_argument("--n-games-per-train", type=int, default=50,
                        help="Self-play games per iteration (AlphaZero only)")
    parser.add_argument("--n-procs", type=int, default=-1,
                        help="Parallel workers (-1=sequential)")
    parser.add_argument("--accept-threshold", type=float, default=0.0,
                        help="Gate acceptance threshold (AlphaZero only)")

    # Tabular Q-specific
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Q-learning rate (tabular_q only)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (tabular_q only)")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial exploration rate (tabular_q only)")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="Final exploration rate (tabular_q only)")

    # Output
    parser.add_argument("--output-dir", default="experiments/benchmark",
                        help="Output directory for results")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build adapter
    adapter = _build_adapter(args)
    run_name = f"{adapter.name()}_D{args.D}_seed{args.seed}"

    logger.info("Benchmark: algo=%s D=%d seed=%d mask_mode=%s",
                adapter.name(), args.D, args.seed, args.mask_mode)

    # Create env factories
    train_factory = make_env_factory(
        D=args.D, locs_per_room=args.locs_per_room,
        mask_mode=args.mask_mode, seed=args.seed,
    )
    eval_factory = make_env_factory(
        D=args.D, locs_per_room=args.locs_per_room,
        mask_mode=args.mask_mode,
    )

    # Result writer
    writer = ResultWriter(output_dir=args.output_dir, run_name=run_name)

    # Run
    all_checkpoints = []
    solve_result = None

    for checkpoint in adapter.train_and_yield_checkpoints(
        env_factory=train_factory,
        eval_env_factory=eval_factory,
        total_steps=args.total_iterations,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    ):
        # Check solve criterion
        all_checkpoints.append(checkpoint)
        if solve_result is None:
            solve_result = check_sustained_solve(all_checkpoints, D=args.D)
            if solve_result is not None:
                checkpoint.solved_flag = True
                checkpoint.solve_env_steps = solve_result[0]
                checkpoint.solve_wall_clock_sec = solve_result[1]
                logger.info("SOLVED at env_steps=%d wall_clock=%.1fs",
                            solve_result[0], solve_result[1])

        writer.append(checkpoint)

        logger.info(
            "[Checkpoint %d] env_steps=%d return=%.4f success=%.2f "
            "invalid=%.4f wall=%.1fs",
            checkpoint.eval_checkpoint_idx,
            checkpoint.env_steps,
            checkpoint.eval_return_mean,
            checkpoint.eval_success_rate,
            checkpoint.invalid_action_rate,
            checkpoint.wall_clock_sec,
        )

    # Summary
    logger.info("Done. Results saved to %s and %s",
                writer.csv_path, writer.jsonl_path)
    if solve_result:
        logger.info("Solved: env_steps=%d wall_clock=%.1fs",
                     solve_result[0], solve_result[1])
    else:
        logger.info("Not solved within training budget.")


if __name__ == "__main__":
    main()
