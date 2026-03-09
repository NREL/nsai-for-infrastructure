"""Result schema and I/O for the benchmark harness.

Defines CheckpointResult and provides CSV + JSONL serialization.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any


@dataclass
class CheckpointResult:
    """Evaluation snapshot at one training checkpoint."""

    # Identity
    algorithm: str
    seed: int
    D: int
    locs_per_room: int
    mask_mode: str

    # Training progress
    env_steps: int
    train_episodes: int
    learner_updates: int
    eval_checkpoint_idx: int

    # Eval aggregates
    eval_return_mean: float
    eval_return_std: float
    eval_success_rate: float
    eval_episode_length_mean: float
    max_room_reached_mean: float
    keys_picked_mean: float
    invalid_action_rate: float

    # Timing
    wall_clock_sec: float

    # Solve tracking
    solved_flag: bool = False
    solve_env_steps: int | None = None
    solve_wall_clock_sec: float | None = None

    # Optional extensible fields
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for CSV/JSON serialization."""
        d = asdict(self)
        # Flatten extra into top-level keys with 'extra_' prefix
        extra = d.pop("extra", {})
        for k, v in extra.items():
            d[f"extra_{k}"] = v
        return d


# Column order for CSV output (excluding dynamic extra_ columns)
_CSV_COLUMNS = [f.name for f in fields(CheckpointResult) if f.name != "extra"]


class ResultWriter:
    """Write CheckpointResults to CSV and JSONL files."""

    def __init__(self, output_dir: str | Path, run_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{run_name}.csv"
        self.jsonl_path = self.output_dir / f"{run_name}.jsonl"
        self._csv_header_written = False
        self._all_results: list[CheckpointResult] = []

    def append(self, result: CheckpointResult) -> None:
        """Append a checkpoint result to both CSV and JSONL."""
        self._all_results.append(result)
        self._append_jsonl(result)
        self._append_csv(result)

    def _append_jsonl(self, result: CheckpointResult) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")

    def _append_csv(self, result: CheckpointResult) -> None:
        d = result.to_dict()
        # Determine columns: fixed + any extra_ columns
        extra_cols = sorted(k for k in d if k.startswith("extra_"))
        columns = _CSV_COLUMNS + extra_cols

        if not self._csv_header_written:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerow({k: d.get(k) for k in columns})
            self._csv_header_written = True
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow({k: d.get(k) for k in columns})

    @property
    def results(self) -> list[CheckpointResult]:
        """All results appended so far."""
        return list(self._all_results)
