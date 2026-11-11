"""Adapter e contratto di separazione delle sorgenti vocali simultanee (P06).

Implementa le specifiche della sezione 6.4 del piano esecutivo:
- Input: AudioView e SpeechRegion (regione di overlap sospetto con margini)
- Output: SeparationResult con zero o piu' tracce candidate, qualita', compensazione ritardo
- Controlli anti-guasto vincolanti:
  * Controllo duplicazione: se la correlazione inter-traccia e' troppo alta (> 0.88),
    il separatore ha sdoppiato la stessa voce -> rifiuto e fallback su raw.
  * Controllo collasso energetico: se una traccia e' silenzio (< -60 dBFS) -> rifiuto.
  * Controllo perdita parlato: se la somma delle tracce non ricostruisce il livello
    della miscela (> 12 dB di caduta), il separatore ha scartato parlato -> rifiuto.
  * Protezione clipping: attenuazione controllata senza distorsione.
  * Rispetto di cancel_event per interrompere l'elaborazione.

Il separatore vero e proprio vive in `separation_backends`: qui restano il
contratto, i guardrail e il ripiego deterministico su traccia grezza.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..models.audio_artifacts import AudioView, SpeakerProfile, SpeechRegion
from .audio_utils import PEAK_CEILING, rms_dbfs
from .separation_backends import SeparationBackend, get_separation_backend


@dataclass
class SeparationResult:
    """Esito della separazione di una regione audio in tracce individuali."""

    region_id: str
    tracks: list[np.ndarray] = field(default_factory=list)
    track_qualities: list[float] = field(default_factory=list)
    delay_compensation_s: float = 0.0
    residual: np.ndarray | None = None
    status: str = "success"  # "success", "fallback_raw", "ambiguous", "cancelled"
    decision: str = "accepted"  # "accepted", "rejected_duplicate", "rejected_energy_collapse", "rejected_speech_loss", "rejected_cancelled"
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "num_tracks": len(self.tracks),
            "track_qualities": [round(q, 4) for q in self.track_qualities],
            "delay_compensation_s": round(self.delay_compensation_s, 4),
            "status": self.status,
            "decision": self.decision,
            "metrics": dict(self.metrics),
        }


def cross_correlation(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calcola la correlazione di Pearson tra due tracce a parita' di lunghezza."""
    min_len = min(len(t1), len(t2))
    if min_len < 100:
        return 0.0
    s1, s2 = t1[:min_len], t2[:min_len]
    std1, std2 = np.std(s1), np.std(s2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 0.0
    c = np.corrcoef(s1, s2)
    val = float(c[0, 1])
    return val if math.isfinite(val) else 0.0


def _fallback_raw(
    region: SpeechRegion,
    raw_track: np.ndarray,
    reason: str,
    *,
    decision: str = "accepted",
    metrics: dict[str, Any] | None = None,
) -> SeparationResult:
    """Esito conservativo: traccia grezza intatta, motivo dichiarato."""
    payload = {"reason": reason}
    payload.update(metrics or {})
    return SeparationResult(
        region_id=region.region_id,
        tracks=[raw_track],
        track_qualities=[1.0],
        delay_compensation_s=0.0,
        status="fallback_raw",
        decision=decision,
        metrics=payload,
    )


def _resolve_backend(backend: str | SeparationBackend) -> SeparationBackend | None:
    if isinstance(backend, SeparationBackend):
        return backend
    try:
        return get_separation_backend("onnx" if backend == "auto" else backend)
    except ValueError:
        return None


def separate_region(
    audio_data: np.ndarray,
    view: AudioView,
    region: SpeechRegion,
    profiles: list[SpeakerProfile] | None = None,
    *,
    cancel_event: Any | None = None,
    backend: str | SeparationBackend = "auto",
    num_speakers: int | None = None,
    duplicate_threshold: float = 0.88,
    silence_threshold_db: float = -60.0,
    speech_loss_threshold_db: float = 12.0,
) -> SeparationResult:
    """Esegue la separazione mirata su una regione sospetta di sovrapposizione.

    Se il modello non e' installato o se l'ambiente non supporta l'inferenza GPU,
    esegue il fallback conservativo su raw dichiarando la decisione.
    Se i canali separati mostrano anomalie (duplicazione o collasso energetico),
    rifiuta l'output per prevenire allucinazioni o voci sdoppiate.
    """
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        return SeparationResult(
            region_id=region.region_id,
            status="cancelled",
            decision="rejected_cancelled",
            metrics={"reason": "cancelled_by_user"},
        )

    # Estrazione campioni della regione
    sr = view.sample_rate
    start_sample = max(0, int(round(region.start_s * sr)))
    end_sample = min(len(audio_data), int(round(region.end_s * sr)))
    segment = audio_data[start_sample:end_sample]

    if len(segment) == 0:
        return SeparationResult(
            region_id=region.region_id,
            status="fallback_raw",
            decision="rejected_energy_collapse",
            metrics={"reason": "empty_segment"},
        )

    # Se mono, espandi vista
    raw_track = segment if segment.ndim == 1 else segment.mean(axis=1)

    candidate_tracks: list[np.ndarray] = []
    delay_comp = 0.0

    if backend == "simulated_split_test" and segment.ndim == 2:
        # Per test multicanale: usa i canali nativi come sorgenti separate
        candidate_tracks = [segment[:, 0].copy(), segment[:, 1].copy()]
    else:
        separator = _resolve_backend(backend)
        if separator is None:
            return _fallback_raw(region, raw_track, "separation_backend_unknown")
        if not separator.is_available():
            return _fallback_raw(
                region,
                raw_track,
                separator.unavailable_reason or "separation_model_not_configured",
                metrics={"backend": separator.name},
            )

        expected = num_speakers or (len(profiles) if profiles else 2)
        try:
            produced = separator.separate_region(raw_track, sr, num_speakers=expected)
        except Exception as exc:
            # Un separatore che fallisce non deve fermare la pipeline: il piano
            # impone di conservare la traccia grezza dichiarando il ripiego.
            return _fallback_raw(
                region,
                raw_track,
                "backend_error",
                metrics={"backend": separator.name, "error": type(exc).__name__},
            )

        candidate_tracks = [np.asarray(track, dtype=np.float32) for track in produced]
        if len(candidate_tracks) < 2:
            return _fallback_raw(
                region,
                raw_track,
                "backend_returned_insufficient_tracks",
                metrics={"backend": separator.name, "num_tracks": len(candidate_tracks)},
            )

    # --- Controlli di validazione acustica sulle tracce separate ---
    if len(candidate_tracks) >= 2:
        corr = cross_correlation(candidate_tracks[0], candidate_tracks[1])
        if corr > duplicate_threshold:
            # Sdoppiamento artificiale della medesima voce
            return SeparationResult(
                region_id=region.region_id,
                tracks=[raw_track],
                track_qualities=[1.0],
                delay_compensation_s=0.0,
                status="fallback_raw",
                decision="rejected_duplicate",
                metrics={"cross_correlation": round(corr, 4), "threshold": duplicate_threshold},
            )

    qualities: list[float] = []
    validated_tracks: list[np.ndarray] = []
    for i, tr in enumerate(candidate_tracks):
        level = rms_dbfs(tr)
        if level < silence_threshold_db:
            # Collasso energetico di una traccia
            return SeparationResult(
                region_id=region.region_id,
                tracks=[raw_track],
                track_qualities=[1.0],
                delay_compensation_s=0.0,
                status="fallback_raw",
                decision="rejected_energy_collapse",
                metrics={"track_index": i, "level_dbfs": round(level, 2)},
            )
        # Protezione anti-clipping
        peak = float(np.max(np.abs(tr))) if tr.size else 0.0
        if peak > PEAK_CEILING:
            tr = (tr * (PEAK_CEILING / peak)).astype(np.float32)
        validated_tracks.append(tr)
        qualities.append(min(1.0, max(0.1, (level + 50.0) / 40.0)))

    # Perdita di parlato: la somma delle tracce deve ricostruire la miscela. Un
    # crollo di livello rispetto al grezzo significa parlato scartato, non separato.
    rebuilt = np.zeros(max(len(track) for track in validated_tracks), dtype=np.float32)
    for track in validated_tracks:
        rebuilt[: len(track)] += track
    speech_loss_db = rms_dbfs(raw_track) - rms_dbfs(rebuilt)
    if math.isfinite(speech_loss_db) and speech_loss_db > speech_loss_threshold_db:
        return _fallback_raw(
            region,
            raw_track,
            "speech_loss_detected",
            decision="rejected_speech_loss",
            metrics={
                "speech_loss_db": round(speech_loss_db, 2),
                "threshold_db": speech_loss_threshold_db,
            },
        )

    return SeparationResult(
        region_id=region.region_id,
        tracks=validated_tracks,
        track_qualities=qualities,
        delay_compensation_s=delay_comp,
        status="success",
        decision="accepted",
        metrics={"num_tracks": len(validated_tracks)},
    )
