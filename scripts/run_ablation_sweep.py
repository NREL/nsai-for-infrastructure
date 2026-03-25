#!/usr/bin/env python3
"""Run ablation sweep: D x mode x seed grid.

Supports resumption: re-running the same command skips already-completed runs.

Usage:
    python scripts/run_ablation_sweep.py
    python scripts/run_ablation_sweep.py --D 15 20 --seeds 42 --n-iterations 10
    python scripts/run_ablation_sweep.py --dry-run
    python scripts/run_ablation_sweep.py --no-resume   # force re-run all

TIP: To survive closing your terminal/lid, run inside tmux or screen:
    tmux new -s sweep
    python scripts/run_ablation_sweep.py
    # Ctrl-B D to detach, 'tmux attach -t sweep' to reattach

Or use nohup:
    nohup python scripts/run_ablation_sweep.py > sweep.log 2>&1 &
    tail -f sweep.log
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

ABLATION_MODES = [
    "full", "policy-only", "value-only-1step",
    "uniform-prior", "zero-value", "unguided-search", "random-net",
]

EXP_BASE = Path("experiments/doors_direct")


def _find_completed_run(D, mode, seed, n_iterations):
    """Check if a completed run already exists for this (D, mode, seed).

    Returns the experiment dir Path if complete, None otherwise.
    """
    if not EXP_BASE.exists():
        return None

    # Match dirs like *_D{D}_*_abl-{mode} or *_D{D}_* (for "full" mode)
    for d in EXP_BASE.iterdir():
        if not d.is_dir():
            continue
        name = d.name

        # Check D value matches
        m_d = re.search(r"_D(\d+)_", name)
        if not m_d or int(m_d.group(1)) != D:
            continue

        # Check mode matches
        m_abl = re.search(r"_abl-([a-z0-9-]+)", name)
        dir_mode = m_abl.group(1) if m_abl else "full"
        if dir_mode != mode:
            continue

        # Check eval_summary.csv exists and has enough rows
        csv_path = d / "eval_summary.csv"
        if not csv_path.exists():
            continue

        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Check seed matches
            if rows and int(float(rows[0].get("seed", -1))) != seed:
                continue
            if len(rows) >= n_iterations:
                return d
        except Exception:
            continue

    return None


def _fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _progress_bar(done, total, width=20):
    """Return a text progress bar like [████████░░░░░░░░░░░░]."""
    filled = int(width * done / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * done / total if total > 0 else 0
    return f"[{bar}] {pct:.0f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation sweep across D values, modes, and seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--D", type=int, nargs="+", default=[15, 20, 25],
                        help="Room counts to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43],
                        help="Random seeds")
    parser.add_argument("--n-iterations", type=int, default=15,
                        help="Training iterations per run")
    parser.add_argument("--locs-per-room", type=int, default=3,
                        help="Locations per room")
    parser.add_argument("--modes", nargs="+", default=ABLATION_MODES,
                        choices=ABLATION_MODES,
                        help="Ablation modes to run")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force re-run all (ignore completed runs)")
    parser.add_argument("--n-procs", type=int, default=None,
                        help="Number of parallel workers per run (overrides default 8)")
    parser.add_argument("--n-simulations", type=int, default=None,
                        help="MCTS simulations per move (overrides default 120)")
    parser.add_argument("--n-games-per-train", type=int, default=None,
                        help="Self-play games per training iteration (overrides default 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    runs = [
        (D, mode, seed)
        for D in args.D
        for mode in args.modes
        for seed in args.seeds
    ]
    total = len(runs)

    # Check which runs are already complete
    skipped = []
    remaining = []
    if not args.no_resume and not args.dry_run:
        for D, mode, seed in runs:
            existing = _find_completed_run(D, mode, seed, args.n_iterations)
            if existing:
                skipped.append((D, mode, seed, existing))
            else:
                remaining.append((D, mode, seed))
    else:
        remaining = runs

    # Banner
    print(f"Ablation sweep: {len(args.D)} D x {len(args.modes)} modes "
          f"x {len(args.seeds)} seeds = {total} total runs")
    print(f"Config: n_iterations={args.n_iterations}, "
          f"locs_per_room={args.locs_per_room}")
    if skipped:
        print(f"Resuming: {len(skipped)} already complete, "
              f"{len(remaining)} remaining")
    print()
    print("TIP: Run inside tmux/screen or with nohup to survive "
          "closing terminal/lid.\n")

    # Print skipped runs
    for D, mode, seed, d in skipped:
        print(f"[SKIP] D={D} mode={mode} seed={seed}  ({d.name})")
    if skipped:
        print()

    failed = []
    run_times = []
    t_start = time.time()

    for idx, (D, mode, seed) in enumerate(remaining, 1):
        cmd = [
            sys.executable, "scripts/run_doors_direct.py",
            "--D", str(D),
            "--locs-per-room", str(args.locs_per_room),
            "--n-iterations", str(args.n_iterations),
            "--seeds", str(seed),
            "--non-interactive",
            "--quiet",
            "--eval-ablation", mode,
        ]
        if args.n_procs is not None:
            cmd.extend(["--n-procs", str(args.n_procs)])
        if args.n_simulations is not None:
            cmd.extend(["--n-simulations", str(args.n_simulations)])
        if args.n_games_per_train is not None:
            cmd.extend(["--n-games-per-train", str(args.n_games_per_train)])

        # Progress bar + ETA
        eta_str = ""
        if run_times:
            avg = sum(run_times) / len(run_times)
            eta = avg * (len(remaining) - idx + 1)
            eta_str = f"  ETA: {_fmt_time(eta)}"

        bar = _progress_bar(idx - 1, len(remaining))
        print(f"\n{'='*60}")
        print(f"{bar}  [{idx}/{len(remaining)}] D={D} mode={mode} seed={seed}{eta_str}")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"  CMD: {' '.join(cmd)}")
            continue

        t0 = time.time()
        # Stream stdout live so per-iteration progress is visible;
        # capture stderr for error reporting on failure.
        proc = subprocess.Popen(cmd, stdout=None, stderr=subprocess.PIPE, text=True)
        _, stderr_output = proc.communicate()
        elapsed = time.time() - t0
        run_times.append(elapsed)

        if proc.returncode != 0:
            print(f"  FAILED ({_fmt_time(elapsed)})")
            if stderr_output:
                print(f"  stderr: {stderr_output[-500:]}")
            failed.append((D, mode, seed))
        else:
            print(f"  OK ({_fmt_time(elapsed)})")

    total_elapsed = time.time() - t_start
    n_succeeded = len(remaining) - len(failed) + len(skipped)
    print(f"\nSweep complete: {n_succeeded}/{total} succeeded "
          f"({len(skipped)} skipped, {len(remaining) - len(failed)} ran) "
          f"in {_fmt_time(total_elapsed)}")
    if failed:
        print("Failed runs:")
        for D, mode, seed in failed:
            print(f"  D={D} mode={mode} seed={seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
