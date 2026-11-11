"""Contratti e modelli dati per ipotesi ASR, utterance con supporto sovrapposizioni, revisioni e risultati degli stadi.

Questi modelli implementano le specifiche della sezione 6.1 del piano esecutivo
(docs/plans/2026-09-17-gitnexus-plan-recupero-voci-ambientali.md).
Consentono la coesistenza di piu' utterance simultanee nello stesso intervallo
temporale (`overlap_group_id`), la tracciabilita' dell'origine ASR (`hypothesis_ids`),
e la gestione protetta di revisioni e checkpoint degli stadi.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .session import Word


def _validate_finite(value: float, name: str) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} deve essere un numero reale finito, ricevuto: {val}")
    return val


def _validate_non_negative(value: float, name: str) -> float:
    val = _validate_finite(value, name)
    if val < 0.0:
        raise ValueError(f"{name} non puo' essere negativo, ricevuto: {val}")
    return val


@dataclass
class ASRHypothesis:
    """Ipotesi generata da un backend ASR per una determinata AudioView."""

    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    view_id: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    backend: str = "whisper"
    model_revision: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    words: list[Word] = field(default_factory=list)
    raw_scores: dict[str, Any] = field(default_factory=dict)
    retry_reason: str | None = None

    def __post_init__(self) -> None:
        self.start_s = _validate_non_negative(self.start_s, "start_s")
        self.end_s = _validate_non_negative(self.end_s, "end_s")
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) non puo' essere minore di start_s ({self.start_s})"
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["words"] = [asdict(w) if isinstance(w, Word) else dict(w) for w in self.words]
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ASRHypothesis:
        raw_words = data.get("words", [])
        words = [
            Word(**w) if isinstance(w, dict) else w for w in raw_words
        ]
        return cls(
            hypothesis_id=data.get("hypothesis_id", uuid.uuid4().hex[:12]),
            view_id=data.get("view_id", ""),
            start_s=float(data.get("start_s", 0.0)),
            end_s=float(data.get("end_s", 0.0)),
            backend=data.get("backend", "whisper"),
            model_revision=data.get("model_revision", ""),
            parameters=dict(data.get("parameters", {})),
            text=data.get("text", ""),
            words=words,
            raw_scores=dict(data.get("raw_scores", {})),
            retry_reason=data.get("retry_reason"),
        )


@dataclass
class DiarizationTurnRecord:
    """Rappresentazione serializzabile di un turno di diarizzazione (ordinario o esclusivo)."""

    start_s: float
    end_s: float
    speaker: str | None = None
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    is_exclusive: bool = False
    overlaps_with: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.start_s = _validate_non_negative(self.start_s, "start_s")
        self.end_s = _validate_non_negative(self.end_s, "end_s")
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) non puo' essere minore di start_s ({self.start_s})"
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "start": round(self.start_s, 3),
            "end": round(self.end_s, 3),
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "speaker": self.speaker,
            "is_exclusive": self.is_exclusive,
            "overlaps_with": list(self.overlaps_with),
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiarizationTurnRecord:
        start_val = data.get("start_s", data.get("start", 0.0))
        end_val = data.get("end_s", data.get("end", 0.0))
        return cls(
            start_s=float(start_val),
            end_s=float(end_val),
            speaker=data.get("speaker"),
            turn_id=str(data.get("turn_id") or uuid.uuid4().hex[:10]),
            is_exclusive=bool(data.get("is_exclusive", False)),
            overlaps_with=list(data.get("overlaps_with", [])),
            confidence=float(data.get("confidence", 1.0)),
        )


UNASSIGNED_SPEAKER_LABELS: frozenset[str] = frozenset({"", "UNKNOWN", "UNASSIGNED", "NONE"})


@dataclass
class Utterance:
    """Unita' fondamentale di parlato attribuito (SpeechEvent).

    Supporta speaker simultanei: piu' Utterance possono avere lo stesso intervallo
    temporale [start_s, end_s] o sovrapporsi parzialmente con diverso stream_id,
    collegate tramite `overlap_group_id` e `overlapping_event_ids`.
    Preserva separatamente le predizioni originali del modello (`original_prediction`)
    e le correzioni dell'utente (`user_corrections`).
    """

    start_s: float
    end_s: float
    utterance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stream_id: int = 0
    overlap_group_id: str | None = None
    speaker: str | None = "UNKNOWN"
    candidate_speakers: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    words: list[Word] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)
    source_audio_ref: str | None = None
    asr_provenance: dict[str, Any] = field(default_factory=dict)
    diarization_provenance: dict[str, Any] = field(default_factory=dict)
    edit_status: str = "model_raw"
    overlapping_event_ids: list[str] = field(default_factory=list)
    is_overlap: bool = False
    original_prediction: dict[str, Any] | None = None
    user_corrections: list[dict[str, Any]] = field(default_factory=list)
    processing_state: str = "predicted"
    uncertain: bool = False
    parent_event_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start_s = _validate_non_negative(self.start_s, "start_s")
        self.end_s = _validate_non_negative(self.end_s, "end_s")
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) non puo' essere minore di start_s ({self.start_s})"
            )
        if self.stream_id < 0:
            raise ValueError(f"stream_id non puo' essere negativo: {self.stream_id}")
        if self.overlapping_event_ids:
            self.is_overlap = True
        if self.original_prediction is None:
            self.original_prediction = {
                "start_s": round(self.start_s, 3),
                "end_s": round(self.end_s, 3),
                "speaker": self.speaker,
                "text": self.text,
                "stream_id": self.stream_id,
                "overlap_group_id": self.overlap_group_id,
            }

    @property
    def event_id(self) -> str:
        """Alias per utterance_id nel modello SpeechEvent."""
        return self.utterance_id

    @property
    def speaker_id(self) -> str | None:
        """Restituisce l'ID dello speaker o None se il segmento non e' assegnato."""
        if self.speaker is None:
            return None
        cleaned = str(self.speaker).strip()
        if cleaned.upper() in UNASSIGNED_SPEAKER_LABELS:
            return None
        return cleaned

    @speaker_id.setter
    def speaker_id(self, value: str | None) -> None:
        if value is None or str(value).strip().upper() in UNASSIGNED_SPEAKER_LABELS:
            self.speaker = None
        else:
            self.speaker = str(value).strip()

    @property
    def is_unassigned(self) -> bool:
        """Indica se l'evento vocale appartiene alla corsia 'Non assegnato'."""
        return self.speaker_id is None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def record_user_correction(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any,
        author: str = "user",
        reason: str = "",
    ) -> None:
        """Registra una correzione manuale preservando la predizione originale del modello."""
        if old_value == new_value:
            return
        self.user_corrections.append(
            {
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "author": author,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if self.edit_status not in ("user_created",):
            self.edit_status = "user_edited"

    def restore_original(self) -> bool:
        """Ripristina i valori predetti originariamente dal modello."""
        if not self.original_prediction:
            return False
        orig = self.original_prediction
        self.start_s = float(orig.get("start_s", self.start_s))
        self.end_s = float(orig.get("end_s", self.end_s))
        self.speaker = orig.get("speaker", self.speaker)
        self.text = str(orig.get("text", self.text))
        self.stream_id = int(orig.get("stream_id", self.stream_id))
        self.overlap_group_id = orig.get("overlap_group_id", self.overlap_group_id)
        self.edit_status = "model_raw"
        self.user_corrections.append(
            {
                "field": "restore_original",
                "old_value": "user_edited",
                "new_value": "model_raw",
                "author": "user",
                "reason": "restore_original_prediction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return True

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["words"] = [asdict(w) if isinstance(w, Word) else dict(w) for w in self.words]
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Utterance:
        raw_words = data.get("words", [])
        words = [
            Word(**w) if isinstance(w, dict) else w for w in raw_words
        ]
        utt_id = data.get("utterance_id") or data.get("event_id") or uuid.uuid4().hex[:12]
        speaker_val = data.get("speaker", "UNKNOWN") if "speaker" in data else data.get("speaker_id", "UNKNOWN")
        orig_pred = data.get("original_prediction")
        if isinstance(orig_pred, dict):
            orig_pred = dict(orig_pred)
        else:
            orig_pred = None
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            utterance_id=str(utt_id),
            stream_id=int(data.get("stream_id", 0)),
            overlap_group_id=data.get("overlap_group_id"),
            speaker=speaker_val,
            candidate_speakers=list(data.get("candidate_speakers", [])),
            text=data.get("text", ""),
            words=words,
            hypothesis_ids=list(data.get("hypothesis_ids", [])),
            source_audio_ref=data.get("source_audio_ref"),
            asr_provenance=dict(data.get("asr_provenance", {})),
            diarization_provenance=dict(data.get("diarization_provenance", {})),
            edit_status=str(data.get("edit_status", "model_raw")),
            overlapping_event_ids=list(data.get("overlapping_event_ids", [])),
            is_overlap=bool(data.get("is_overlap", bool(data.get("overlapping_event_ids")))),
            original_prediction=orig_pred,
            user_corrections=list(data.get("user_corrections", [])),
            processing_state=str(data.get("processing_state", "missing_transcription" if data.get("edit_status") == "untranscribed_overlap" else "predicted")),
            uncertain=bool(data.get("uncertain", False)),
            parent_event_ids=list(data.get("parent_event_ids", [])),
        )


SpeechEvent = Utterance


def recompute_overlap_relationships(
    utterances: list[Utterance],
    min_overlap_s: float = 0.05,
    preserve_unpaired_groups: bool = False,
) -> list[Utterance]:
    """Ricalcola le relazioni di sovrapposizione (`overlapping_event_ids`, `is_overlap`, `overlap_group_id`, `stream_id`).

    Due eventi sono considerati sovrapposti quando il loro intervallo temporale si
    interseca per almeno `min_overlap_s` secondi (o se uno di essi ha durata nulla
    contenuta nell'altro). Non appiattisce mai eventi simultanei in un unico segmento.
    """
    if not utterances:
        return utterances

    ordered = sorted(
        utterances,
        key=lambda u: (u.start_s, u.end_s, u.speaker or "", u.utterance_id),
    )
    n = len(ordered)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}

    for i in range(n):
        a = ordered[i]
        for j in range(i + 1, n):
            b = ordered[j]
            if b.start_s >= a.end_s and b.start_s > a.start_s:
                break
            # La sovrapposizione descrive voci *diverse* simultanee: due eventi
            # della stessa voce che si accavallano sono un artefatto di timing,
            # non parlato contemporaneo. Il parlato non assegnato e' sempre
            # considerato potenzialmente distinto.
            a_spk, b_spk = a.speaker_id, b.speaker_id
            if a_spk is not None and a_spk == b_spk:
                continue
            ov = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
            if ov >= min_overlap_s or (a.duration_s == 0.0 and a.start_s >= b.start_s and a.end_s <= b.end_s):
                adj[i].append(j)
                adj[j].append(i)

    visited: set[int] = set()
    group_counter = 0

    for i in range(n):
        u = ordered[i]
        neighbors = adj[i]
        u.overlapping_event_ids = [ordered[j].utterance_id for j in neighbors]
        u.is_overlap = bool(neighbors)
        if not neighbors:
            if not preserve_unpaired_groups:
                u.overlap_group_id = None
            u.stream_id = 0
            visited.add(i)
            continue

        if i in visited:
            continue

        # Traverse connected component of overlapping events
        comp: list[int] = []
        queue = [i]
        visited.add(i)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        comp.sort(key=lambda idx: (ordered[idx].start_s, ordered[idx].end_s, ordered[idx].utterance_id))
        existing_gid = next(
            (ordered[idx].overlap_group_id for idx in comp if ordered[idx].overlap_group_id),
            None,
        )
        group_counter += 1
        gid = existing_gid or f"ov_{group_counter}_{ordered[comp[0]].utterance_id[:6]}"

        # Assign distinct stream_id lanes greedily inside component
        lane_ends: list[float] = []
        for idx in comp:
            ev = ordered[idx]
            ev.overlap_group_id = gid
            assigned_lane = None
            for lane_idx, l_end in enumerate(lane_ends):
                if ev.start_s >= l_end - min_overlap_s:
                    assigned_lane = lane_idx
                    lane_ends[lane_idx] = max(l_end, ev.end_s)
                    break
            if assigned_lane is None:
                assigned_lane = len(lane_ends)
                lane_ends.append(ev.end_s)
            ev.stream_id = assigned_lane

    return utterances


@dataclass
class TranscriptRevision:
    """Tracciamento di una revisione proposta o applicata alla trascrizione."""

    revision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_revision_id: str | None = None
    operations: list[dict[str, Any]] = field(default_factory=list)
    author: str = "human"
    evidence_ids: list[str] = field(default_factory=list)
    status: str = "proposed"  # "proposed", "accepted", "rejected"
    source_fingerprint: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        valid_statuses = {"proposed", "accepted", "rejected"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"status non valido: {self.status}, ammessi: {valid_statuses}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptRevision:
        return cls(
            revision_id=data.get("revision_id", uuid.uuid4().hex[:12]),
            parent_revision_id=data.get("parent_revision_id"),
            operations=list(data.get("operations", [])),
            author=data.get("author", "human"),
            evidence_ids=list(data.get("evidence_ids", [])),
            status=data.get("status", "proposed"),
            source_fingerprint=data.get("source_fingerprint", ""),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class StageResult:
    """Risultato dell'elaborazione di uno stadio di pipeline."""

    stage_name: str
    stage_version: str = "1.0.0"
    input_hash: str = ""
    actual_model: str = ""
    output_artifact_ids: list[str] = field(default_factory=list)
    status: str = "success"  # "success", "warning", "error"
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    resource_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_statuses = {"success", "warning", "error"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"status non valido: {self.status}, ammessi: {valid_statuses}"
            )
        self.duration_s = _validate_non_negative(self.duration_s, "duration_s")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageResult:
        return cls(
            stage_name=data["stage_name"],
            stage_version=data.get("stage_version", "1.0.0"),
            input_hash=data.get("input_hash", ""),
            actual_model=data.get("actual_model", ""),
            output_artifact_ids=list(data.get("output_artifact_ids", [])),
            status=data.get("status", "success"),
            duration_s=float(data.get("duration_s", 0.0)),
            warnings=list(data.get("warnings", [])),
            resource_metrics=dict(data.get("resource_metrics", {})),
        )
