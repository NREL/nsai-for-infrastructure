"""CLI for generating benchmark plots from CSV result files.

Usage:
    python -m alphazeropp.benchmark.plot_cli experiments/stage3
    python -m alphazeropp.benchmark.plot_cli experiments/stage3 --table-only
    python -m alphazeropp.benchmark.plot_cli experiments/stage3 --x-axis env_steps
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from alphazeropp.benchmark.plotting import (
    plot_comparison_panel,
    plot_learning_curves,
    print_summary_table,
)


def _group_csvs_by_d(csv_paths: list[Path]) -> dict[int, list[Path]]:
    """Group CSV files by D value parsed from filename pattern *_D{d}_*."""
    groups: dict[int, list[Path]] = defaultdict(list)
    for p in csv_paths:
        m = re.search(r"_D(\d+)_", p.name)
        if m:
            groups[int(m.group(1))].append(p)
    return dict(groups)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from CSV results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing CSV result files")
    parser.add_argument("--table-only", action="store_true",
                        help="Only print summary table, skip plots")
    parser.add_argument("--D", type=int, nargs="*", default=None,
                        help="Filter to specific D values (default: all)")
    parser.add_argument("--x-axis", default="train_episodes",
                        help="X-axis field for learning curves")

    args = parser.parse_args(argv)

    csv_paths = sorted(args.results_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_paths)} CSV files in {args.results_dir}\n")

    # Summary table (always printed, covers all D values)
    table = print_summary_table(csv_paths)

    summary_path = args.results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(table)
    print(f"[Summary] Saved to {summary_path}")

    if args.table_only:
        return

    # Generate per-D plots
    d_groups = _group_csvs_by_d(csv_paths)
    if args.D is not None:
        d_groups = {d: v for d, v in d_groups.items() if d in args.D}

    for D in sorted(d_groups.keys()):
        d_csvs = d_groups[D]
        plot_learning_curves(
            d_csvs,
            args.results_dir / f"learning_curves_D{D}.png",
            x_axis=args.x_axis,
            title=f"D={D}: eval_return_mean vs {args.x_axis}",
        )
        plot_comparison_panel(
            d_csvs,
            args.results_dir / f"comparison_panel_D{D}.png",
            x_axis=args.x_axis,
            title=f"D={D}: Benchmark Comparison",
        )


if __name__ == "__main__":
    main()
