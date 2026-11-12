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

    @contextmanager
    def guard(self):
        """Keep failures visible without changing the operation's exception."""
        try:
            yield self
        except BaseException as exc:
            self.fail_running(exc)
            raise

    def asr_progress(self, stage_id):
        def report(done, total):
            self.update(stage_id, done / total if total > 0 else None,
                        f"{done:.1f} / {total:.1f} s; poi verifica e recupero ASR")
        return report

    def asr_status(self, stage_id):
        def report(message):
            # Decoder seconds cannot measure the remaining retry/recovery work.
            row = self._row(stage_id)
            if row["status"] != "running":
                return
            row["fraction"] = None
            if message.startswith("Avviso: "):
                previous = row["warning"]
                row["warning"] = f"{previous}; {message}" if previous else message
            self.update(stage_id, detail=message)
        return report

    def transcribed(self, stage_id):
        return self.complete(stage_id, warning=self._row(stage_id)["warning"])

    def note(self, message, _legacy_fraction=None):
        running = [key for key, row in self._rows.items() if row["status"] == "running"]
        if running:
            self.update(running[-1], detail=message)

    def diarization_progress(self, stage_id):
        def report(step, fraction):
            # A pyannote substep is not a measured fraction of the whole model.
            detail = str(step) + (f" ({fraction:.0%})" if fraction is not None else "")
            self.update(stage_id, detail=detail)
        return report

    def aligned(self, stage_id, report):
        value = report.to_dict() if report is not None else {}
        warning = value.get("fallback_reason") or value.get("error") or "; ".join(value.get("notes", []))
        self.complete(stage_id, detail=str(value.get("used_backend", value.get("backend", ""))),
                      warning=str(warning) if warning else None)


SOURCE_STAGES = [
    ("source_denoise", "Riduzione rumore sorgenti"),
    ("echo", "Verifica eco"), ("asr_model", "Caricamento modello ASR"),
    ("mic_asr", "Trascrizione microfono e recupero ASR"),
    ("system_asr", "Trascrizione sistema e recupero ASR"),
    ("mic_alignment", "Allineamento parole microfono"),
    ("mic_diarization", "Diarizzazione microfono"),
    ("system_alignment", "Allineamento parole sistema"),
    ("system_diarization", "Diarizzazione sistema"),
    ("source_merge", "Ricomposizione sorgenti e sovrapposizioni"),
]


def recording_stages(options, *, source=None):
    """Plan actual branch order before expensive processing starts."""
    if source is None:
        source = (options.strategy in ("source_aware", "source_single_speaker")
                  or options.recording_mode in ("headset", "speakers"))
    rows = [("validate", "Validazione profilo e sorgenti"), ("prepare", "Preparazione e sincronizzazione audio")]
    if source:
        rows += SOURCE_STAGES
    else:
        isolated = options.strategy == "speaker_isolated"
        if isolated:
            rows += [("separation", "Separazione voci")]
        rows += [("denoise", "Riduzione rumore")]
        diar = [("diarization", "Diarizzazione: chi parla quando"), ("profiles", "Profili vocali")]
        if options.diarize and options.profile_first:
            rows += diar
        if not isolated:
            rows += [("separation", "Separazione voci")]
        elif options.diarize and options.profile_first:
            rows += [("profile_separation", "Estrazione voci dai turni rilevati")]
        rows += [("asr", "Trascrizione e recupero ASR"),
                 ("alignment", "Allineamento parole")]
        if not (options.diarize and options.profile_first):
            rows += diar
        rows += [("merge", "Attribuzione parole e voci"), ("review", "Revisione deterministica"),
                 ("reconstruct", "Ricostruzione eventi e sovrapposizioni")]
    return rows + [("finalize", "Sessione e risultati intermedi")]
