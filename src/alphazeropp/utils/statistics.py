import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional


class StatisticsManager:
	"""Collects stats dictionaries and persists them to disk."""

	def __init__(self, run_start_time: Optional[int] = None):
		self.run_start_time = run_start_time or int(time.time())
		self._records: list[dict[str, Any]] = []
		self._records_written = 0

	def record(self, data: dict[str, Any]) -> None:
		"""Add a single stats record."""
		if not isinstance(data, dict):
			raise TypeError("data must be a dict")
		self._records.append({"timestamp": int(time.time()), **data})

	def record_many(self, records: Iterable[dict[str, Any]]) -> None:
		"""Add multiple stats records."""
		for record in records:
			self.record(record)

	def clear(self) -> None:
		"""Remove all recorded stats from memory."""
		self._records.clear()
		self._records_written = 0

	def to_list(self) -> list[dict[str, Any]]:
		"""Return a copy of recorded stats."""
		return list(self._records)

	def save_jsonl(self, file_path: str | Path, append: bool = True) -> None:
		"""Write stats to a JSONL file, one record per line."""
		file_path = Path(file_path)
		file_path.parent.mkdir(parents=True, exist_ok=True)

		start_index = self._records_written if append else 0
		records_to_write = self._records[start_index:]
		if not records_to_write:
			return

		mode = "a" if append else "w"
		with file_path.open(mode, encoding="utf-8") as handle:
			for record in records_to_write:
				handle.write(json.dumps(record, sort_keys=True) + "\n")

		if append:
			self._records_written = len(self._records)
