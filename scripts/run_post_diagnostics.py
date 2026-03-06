#!/usr/bin/env python3
"""
Post-experiment diagnostic report generator.

Usage:
    # Single experiment:
    python scripts/run_post_diagnostics.py experiments/doors_derivation/<exp_dir>

    # Multiple experiments (generates per-experiment + comparison):
    python scripts/run_post_diagnostics.py experiments/doors_derivation/<dir1> experiments/doors_derivation/<dir2>

    # All experiments in a parent directory:
    python scripts/run_post_diagnostics.py experiments/doors_derivation/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.utils.post_diagnostics import (
    run_post_diagnostics,
    run_comparison_diagnostics,
)


def find_experiment_dirs(path: Path) -> list[Path]:
    """Find experiment directories (contain program_log.jsonl)."""
    if (path / "program_log.jsonl").exists():
        return [path]
    # Search subdirectories
    dirs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "program_log.jsonl").exists():
            dirs.append(child)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic reports for AlphaZero experiments")
    parser.add_argument("paths", nargs="+", type=str,
                        help="Experiment directories or parent directory")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    args = parser.parse_args()

    # Collect experiment directories
    exp_dirs = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        exp_dirs.extend(find_experiment_dirs(path))

    if not exp_dirs:
        print("No experiment directories found.")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiment(s):")
    for d in exp_dirs:
        print(f"  {d.name}")
    print()

    # Generate per-experiment reports
    for d in exp_dirs:
        run_post_diagnostics(d, show=args.show)

    # Generate comparison report if multiple experiments
    if len(exp_dirs) >= 2:
        print()
        run_comparison_diagnostics(exp_dirs, show=args.show)


if __name__ == "__main__":
    main()
