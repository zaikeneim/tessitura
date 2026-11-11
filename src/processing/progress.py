"""Observed processing stages shared by CLI workers and the desktop UI.

Unknown work stays indeterminate. Fractions describe completed work, never
elapsed-time guesses. A stage is completed only by the operation that owns it.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class StageEvent:
    stage_id: str
    label: str
    status: str = "pending"
    fraction: float | None = None
    detail: str = ""
    elapsed_s: float = 0.0
    overall_fraction: float = 0.0
    warning: str | None = None


class StageReporter:
    def __init__(self, stages: Iterable[tuple[str, str]], progress=None, on_stage=None):
        self._rows: dict[str, dict] = {}
        self._started: dict[str, float] = {}
        self._progress: Callable | None = progress
        self._on_stage: Callable | None = on_stage
        self._overall = 0.0
        for stage_id, label in stages:
            if stage_id in self._rows:
                raise ValueError(f"Duplicate processing stage: {stage_id}")
            self._rows[stage_id] = dict(stage_id=stage_id, label=label, status="pending",
                                       fraction=None, detail="", elapsed_s=0.0, warning=None)
        for stage_id in self._rows:
            self._emit(stage_id, legacy=False)

    def _row(self, stage_id):
        if stage_id not in self._rows:
            raise KeyError(f"Processing stage was not planned: {stage_id}")
        return self._rows[stage_id]

    def _emit(self, stage_id, *, legacy=True):
        row = self._row(stage_id)
        if row["status"] == "running":
            row["elapsed_s"] = max(0.0, time.perf_counter() - self._started[stage_id])
        total = len(self._rows)
        units = sum(1.0 if item["status"] in ("completed", "skipped") else
                    (item["fraction"] or 0.0) if item["status"] == "running" else 0.0
                    for item in self._rows.values())
        all_done = total > 0 and all(item["status"] in ("completed", "skipped") for item in self._rows.values())
        fraction = units / total if total else 0.0
        self._overall = max(self._overall, min(fraction, 1.0 if all_done else .99))
        event = StageEvent(**row, overall_fraction=self._overall)
        if self._on_stage:
            self._on_stage(event)
        if legacy and self._progress:
            message = row["label"] + (f": {row['detail']}" if row["detail"] else "")
            self._progress(message, self._overall)
        return event

    def start(self, stage_id, detail=""):
        row = self._row(stage_id)
        if row["status"] in ("completed", "skipped", "failed"):
            raise RuntimeError(f"Cannot restart terminal processing stage: {stage_id}")
        if row["status"] == "pending":
            self._started[stage_id] = time.perf_counter()
        row.update(status="running", detail=detail or row["detail"])
        return self._emit(stage_id)

    def update(self, stage_id, fraction=None, detail=""):
        row = self._row(stage_id)
        if row["status"] == "pending":
            self.start(stage_id)
        if row["status"] != "running":
            return None  # Late callbacks cannot revive a finished operation.
        if fraction is not None:
            fraction = float(fraction)
            if not math.isfinite(fraction):
                raise ValueError("Stage progress must be finite")
            row["fraction"] = max(row["fraction"] or 0.0, min(1.0, max(0.0, fraction)))
        if detail:
            row["detail"] = detail
        return self._emit(stage_id)

    def complete(self, stage_id, detail="", warning=None):
        row = self._row(stage_id)
        if row["status"] in ("skipped", "failed"):
            raise RuntimeError(f"Cannot complete unexecuted processing stage: {stage_id}")
        if row["status"] == "running":
            row["elapsed_s"] = max(0.0, time.perf_counter() - self._started[stage_id])
        row.update(status="completed", fraction=1.0, detail=detail or row["detail"], warning=warning)
        return self._emit(stage_id)

    def skip(self, stage_id, reason):
        if not reason:
            raise ValueError("A skipped stage requires its actual reason")
        row = self._row(stage_id)
        if row["status"] in ("completed", "failed"):
            raise RuntimeError(f"Cannot skip terminal processing stage: {stage_id}")
        row.update(status="skipped", fraction=None, detail=str(reason))
        return self._emit(stage_id)

    def fail(self, stage_id, error):
        row = self._row(stage_id)
        if row["status"] == "running":
            row["elapsed_s"] = max(0.0, time.perf_counter() - self._started[stage_id])
        row.update(status="failed", detail=str(error))
        return self._emit(stage_id)

    def fail_running(self, error):
        for stage_id, row in self._rows.items():
            if row["status"] == "running":
                self.fail(stage_id, error)

    @contextmanager
    def phase(self, stage_id, detail=""):
        self.start(stage_id, detail)
        try:
            yield self
        except Exception as exc:
            self.fail(stage_id, exc)
            raise
        else:
            self.complete(stage_id)

    def snapshot(self):
        return [asdict(StageEvent(**row, overall_fraction=self._overall)) for row in self._rows.values()]

    def assert_complete(self):
        unfinished = [key for key, row in self._rows.items() if row["status"] not in ("completed", "skipped")]
        if unfinished:
            raise RuntimeError("Processing stages unfinished: " + ", ".join(unfinished))
