"""Contratti e modelli dati per asset audio, viste multicanale, regioni vocali e profili speaker.

Questi modelli implementano le specifiche della sezione 6.1 del piano esecutivo
(docs/plans/2026-09-17-gitnexus-plan-recupero-voci-ambientali.md).
Sono dataclass serializzabili in JSON puro (senza dipendenze da librerie pesanti
come PyTorch, Qt o soundfile), con validazione rigorosa contro NaN, durate
negative e riferimenti inconsistenti.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _validate_finite(value: float, name: str) -> float:
    """Valida che un float sia finito (non NaN, non infinito)."""
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} deve essere un numero reale finito, ricevuto: {val}")
    return val


def _validate_non_negative(value: float, name: str) -> float:
    """Valida che un valore float sia non negativo."""
    val = _validate_finite(value, name)
    if val < 0.0:
        raise ValueError(f"{name} non puo' essere negativo, ricevuto: {val}")
    return val


@dataclass
class AudioAsset:
    """Rappresentazione immutabile di un file audio sorgente.

    Il percorso e' relativo alla cartella del progetto/dati per garantire
    la portabilita'. L'asset originale non viene mai sovrascritto.
    """

    asset_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    relative_path: str = ""
    sha256: str = ""
    sample_rate: int = 16000
    num_channels: int = 1
    num_samples: int = 0
    duration_s: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate deve essere positivo: {self.sample_rate}")
        if self.num_channels <= 0:
            raise ValueError(f"num_channels deve essere positivo: {self.num_channels}")
        if self.num_samples < 0:
            raise ValueError(f"num_samples non puo' essere negativo: {self.num_samples}")
        if self.duration_s <= 0.0 and self.num_samples > 0:
            self.duration_s = self.num_samples / self.sample_rate
        else:
            self.duration_s = _validate_non_negative(self.duration_s, "duration_s")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioAsset:
        return cls(
            asset_id=data.get("asset_id", uuid.uuid4().hex[:12]),
            relative_path=data.get("relative_path", ""),
            sha256=data.get("sha256", ""),
            sample_rate=int(data.get("sample_rate", 16000)),
            num_channels=int(data.get("num_channels", 1)),
            num_samples=int(data.get("num_samples", 0)),
            duration_s=float(data.get("duration_s", 0.0)),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass
class AudioView:
    """Una vista specifica (canali selezionati, porzione temporale, filtri) su un AudioAsset.

    Consente di tracciare la provenienza esatta dei segnali inviati ai modelli
    (raw, per-canale, denoisato, o separato), compensando eventuali ritardi.
    """

    view_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    asset_id: str = ""
    channel_indices: list[int] = field(default_factory=lambda: [0])
    sample_rate: int = 16000
    time_offset_s: float = 0.0
    duration_s: float = 0.0
    transformations: list[dict[str, Any]] = field(default_factory=list)
    delay_compensation_s: float = 0.0
    parent_view_id: str | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate deve essere positivo: {self.sample_rate}")
        self.time_offset_s = _validate_non_negative(self.time_offset_s, "time_offset_s")
        self.duration_s = _validate_non_negative(self.duration_s, "duration_s")
        self.delay_compensation_s = _validate_finite(
            self.delay_compensation_s, "delay_compensation_s"
        )
        if not self.channel_indices:
            raise ValueError("channel_indices deve contenere almeno un indice canale")
        for ch in self.channel_indices:
            if ch < 0:
                raise ValueError(f"Indice canale negativo non ammesso: {ch}")

    @property
    def end_s(self) -> float:
        return self.time_offset_s + self.duration_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioView:
        return cls(
            view_id=data.get("view_id", uuid.uuid4().hex[:12]),
            asset_id=data.get("asset_id", ""),
            channel_indices=list(data.get("channel_indices", [0])),
            sample_rate=int(data.get("sample_rate", 16000)),
            time_offset_s=float(data.get("time_offset_s", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            transformations=list(data.get("transformations", [])),
            delay_compensation_s=float(data.get("delay_compensation_s", 0.0)),
            parent_view_id=data.get("parent_view_id"),
        )


@dataclass
class SpeechRegion:
    """Intervallo temporale rilevato con attivita' vocale e diagnostica associata."""

    start_s: float
    end_s: float
    region_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    vad_confidence: float = 1.0
    overlap_probability: float | None = None
    estimated_quality: float | None = None
    suspect_reasons: list[str] = field(default_factory=list)
    channel_index: int | None = None

    def __post_init__(self) -> None:
        self.start_s = _validate_non_negative(self.start_s, "start_s")
        self.end_s = _validate_non_negative(self.end_s, "end_s")
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) non puo' essere minore di start_s ({self.start_s})")
        self.vad_confidence = _validate_finite(self.vad_confidence, "vad_confidence")
        if self.overlap_probability is not None:
            self.overlap_probability = _validate_finite(
                self.overlap_probability, "overlap_probability"
            )
        if self.estimated_quality is not None:
            self.estimated_quality = _validate_finite(
                self.estimated_quality, "estimated_quality"
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechRegion:
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            region_id=data.get("region_id", uuid.uuid4().hex[:12]),
            vad_confidence=float(data.get("vad_confidence", 1.0)),
            overlap_probability=(
                float(data["overlap_probability"])
                if data.get("overlap_probability") is not None
                else None
            ),
            estimated_quality=(
                float(data["estimated_quality"])
                if data.get("estimated_quality") is not None
                else None
            ),
            suspect_reasons=list(data.get("suspect_reasons", [])),
            channel_index=data.get("channel_index"),
        )


@dataclass
class SpeakerProfile:
    """Profilo vocale con embedding normalizzati e riferimenti a campioni affidabili."""

    speaker_id: str
    embedding_version: str = "pyannote-community-1"
    normalized_embeddings: list[list[float]] = field(default_factory=list)
    accepted_samples: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 1.0
    is_verified: bool = False
    display_name: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.quality_score = _validate_finite(self.quality_score, "quality_score")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(f"quality_score deve essere in [0.0, 1.0]: {self.quality_score}")
        for emb in self.normalized_embeddings:
            for val in emb:
                _validate_finite(val, "embedding_value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerProfile:
        return cls(
            speaker_id=str(data["speaker_id"]),
            embedding_version=str(data.get("embedding_version", "pyannote-community-1")),
            normalized_embeddings=[
                [float(x) for x in emb]
                for emb in data.get("normalized_embeddings", [])
            ],
            accepted_samples=list(data.get("accepted_samples", [])),
            quality_score=float(data.get("quality_score", 1.0)),
            is_verified=bool(data.get("is_verified", False)),
            display_name=data.get("display_name"),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
        )
