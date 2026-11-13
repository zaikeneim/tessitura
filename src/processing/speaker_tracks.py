"""Estrazione e trascrizione per parlante da un canale multivoce (piano §6.4).

Percorso `speaker_isolated`: separa acusticamente un canale che contiene piu'
voci, poi trascrive ogni traccia in modo indipendente e ricompone gli interventi.
Quando il separatore manca o i guardrail P06 rifiutano il risultato, il
passthrough non viene spacciato per separazione riuscita: l'esito dichiara il
ripiego e il chiamante prosegue sul percorso standard.

Misurato in §9.05: nessuno dei modelli provati (SepFormer-WhamR16k,
MossFormer2_SS_16K) batte il percorso standard, per questo il percorso resta
dietro `--strategy speaker_isolated` e non diventa mai il default.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from ..models.audio_artifacts import AudioView, SpeechRegion
from ..models.session import Segment, Word
from ..models.transcript_artifacts import Utterance
from . import audio_utils
from .alignment import align_segments
from .separation import separate_region
from .transcribe import Transcriber


@dataclass
class SpeakerTracksResult:
    """Esito della separazione o ritaglio di un canale multivoce nelle voci che contiene."""

    tracks: list[np.ndarray] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    sample_rate: int = audio_utils.TARGET_SR
    status: str = "fallback_raw"  # "separated" | "sliced" | "fallback_raw"
    decision: str = "accepted"
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def separated(self) -> bool:
        """Ci sono tracce per parlante utilizzabili (separate *o* ritagliate).

        Serve solo all'instradamento. Per dire se un modello ha davvero
        separato le voci acusticamente usare ``acoustic``: un ritaglio per turni
        di una miscela a due voci resta una miscela.
        """
        return self.status in ("separated", "sliced") and len(self.tracks) >= 2

    @property
    def acoustic(self) -> bool:
        """Un separatore acustico ha prodotto le tracce e i guardrail le hanno accettate."""
        return self.status == "separated" and len(self.tracks) >= 2

    @property
    def method(self) -> str:
        if self.acoustic:
            return "acoustic_separation"
        if self.status == "sliced" and len(self.tracks) >= 2:
            return "diarization_slicing"
        return "none"

    def describe(self) -> str:
        """Descrizione per le fasi di elaborazione, senza confondere i due metodi."""
        if self.method == "acoustic_separation":
            return f"Separazione acustica: {len(self.tracks)} tracce"
        if self.method == "diarization_slicing":
            return (f"Ritaglio per turni di diarizzazione: {len(self.tracks)} tracce "
                    "(non è separazione acustica; il parlato sovrapposto resta nella miscela)")
        return "Nessuna traccia per parlante"

    def as_metadata(self) -> dict[str, Any]:
        """Traccia nella sessione cosa e' successo davvero alla separazione."""
        payload = {
            **self.metrics,
            "separation_status": self.status,
            "separation_decision": self.decision,
            "track_count": len(self.tracks),
            "method": self.method,
            "acoustic_separation": self.acoustic,
        }
        return payload


def extract_speaker_tracks(
    audio_path: str | Path,
    *,
    turns: list[Any] | None = None,
    num_speakers: int = 2,
    backend: Any = "auto",
    cancel_event: Any | None = None,
) -> SpeakerTracksResult:
    """Separa o ritaglia un canale multivoce nelle tracce dei singoli parlanti.

    Se vengono forniti i turni della profilazione vocale (turns), isola le voci
    in altrettanti file/array separati basandosi sugli intervalli di parlato di
    ciascun locutore con marcatura esplicita 'sliced' (non millantando separazione acustica).
    Altrimenti tenta la separazione acustica cieca per regione ('separated').
    """
    mono = audio_utils.load_mono_16k(audio_path)
    sample_rate = audio_utils.TARGET_SR
    duration = mono.size / sample_rate
    if duration <= 0:
        return SpeakerTracksResult(sample_rate=sample_rate, metrics={"reason": "empty_audio"})

    # 1. Ritaglio di turni da profilazione vocale / diarizzazione
    if turns:
        speaker_ids = sorted(list({t.speaker for t in turns if getattr(t, "speaker", None)}))
        if len(speaker_ids) >= 2:
            tracks: list[np.ndarray] = []
            margin = 0.25  # 250ms di margine per preservare attacco, rilascio e inizio frasi
            for spk in speaker_ids:
                spk_turns = [t for t in turns if getattr(t, "speaker", None) == spk]
                spk_turns.sort(key=lambda t: t.start)

                # Unione di turni contigui con pause brevi (gap <= 0.8s)
                # senza fondere attraverso interventi altrui
                merged: list[tuple[float, float]] = []
                for t in spk_turns:
                    if not merged:
                        merged.append((t.start, t.end))
                    else:
                        prev_s, prev_e = merged[-1]
                        has_intervening_speaker = any(
                            getattr(other, "speaker", None) != spk
                            and max(prev_e, getattr(other, "start", 0.0)) < min(t.start, getattr(other, "end", 0.0))
                            for other in turns
                        )
                        if t.start <= prev_e + 0.8 and not has_intervening_speaker:
                            merged[-1] = (prev_s, max(prev_e, t.end))
                        else:
                            merged.append((t.start, t.end))

                # Il margine serve a non tagliare attacco e rilascio, ma con turni
                # brevi sconfina nel turno accanto: senza questa sottrazione lo
                # stesso parlato finisce in due tracce e viene trascritto due
                # volte, attribuito a entrambi (misurato: 13 quadrigrammi
                # condivisi su 90 s, frasi intere duplicate).
                foreign = np.zeros(mono.size, dtype=bool)
                for other in turns:
                    if getattr(other, "speaker", None) == spk:
                        continue
                    o_start = int(round(max(0.0, other.start) * sample_rate))
                    o_end = int(round(min(duration, other.end) * sample_rate))
                    if o_end > o_start:
                        foreign[o_start:o_end] = True
                for start, end in merged:
                    a = int(round(max(0.0, start) * sample_rate))
                    b = int(round(min(duration, end) * sample_rate))
                    if b > a:
                        foreign[a:b] = False  # il proprio turno vince sul margine altrui

                spk_track = np.zeros_like(mono)
                for start, end in merged:
                    s_start = max(0.0, start - margin)
                    s_end = min(duration, end + margin)
                    i_start = int(round(s_start * sample_rate))
                    i_end = int(round(s_end * sample_rate))
                    if i_end <= i_start:
                        continue
                    chunk = mono[i_start:i_end].copy()
                    ramp_len = min(160, len(chunk) // 2)
                    if ramp_len > 0:
                        ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
                        chunk[:ramp_len] *= ramp
                        chunk[-ramp_len:] *= ramp[::-1]
                    spk_track[i_start:i_end] = chunk
                spk_track[foreign] = 0.0

                tracks.append(spk_track)

            return SpeakerTracksResult(
                tracks=tracks,
                speaker_ids=speaker_ids,
                sample_rate=sample_rate,
                status="sliced",
                decision="accepted",
                metrics={
                    "method": "turn_slicing",
                    "speaker_count": len(speaker_ids),
                },
            )

    view = AudioView(
        view_id=f"view_{uuid.uuid4().hex[:8]}",
        asset_id=f"asset_{uuid.uuid4().hex[:8]}",
        sample_rate=sample_rate,
        duration_s=duration,
    )
    outcome = separate_region(
        mono,
        view,
        SpeechRegion(start_s=0.0, end_s=duration),
        cancel_event=cancel_event,
        backend=backend,
        num_speakers=num_speakers,
    )

    if outcome.status != "success" or len(outcome.tracks) < 2:
        return SpeakerTracksResult(
            sample_rate=sample_rate,
            status="fallback_raw",
            decision=outcome.decision,
            metrics=dict(outcome.metrics),
        )
    return SpeakerTracksResult(
        tracks=[np.asarray(track, dtype=np.float32) for track in outcome.tracks],
        sample_rate=sample_rate,
        status="separated",
        decision=outcome.decision,
        metrics=dict(outcome.metrics),
    )


def transcribe_speaker_tracks(
    result: SpeakerTracksResult,
    *,
    options: Any | None = None,
    progress: Callable[[str, float], None] | None = None,
    work_dir: Path | None = None,
    stages=None,
) -> tuple[list[Utterance], list[Segment], str, dict]:
    """Trascrive ogni traccia separata e ricompone gli interventi per parlante."""
    if not result.separated:
        raise ValueError("Nessuna traccia separata da trascrivere")

    report = progress or (lambda _msg, _frac: None)
    opts = options
    language = getattr(opts, "language", None)
    initial_prompt = getattr(opts, "initial_prompt", None)
    quality_retries = getattr(opts, "quality_retries", 24)

    from .pipeline import resolve_device

    resolved_device, resolved_compute = resolve_device(getattr(opts, "device", "auto"))
    if getattr(opts, "compute_type", None):
        resolved_compute = opts.compute_type

    if stages:
        stages.start("asr")
    report("Caricamento modello ASR...", 0.55)
    transcriber = Transcriber(
        model_size=getattr(opts, "model_size", "large-v3"),
        device=resolved_device,
        compute_type=resolved_compute,
        batch_size=getattr(opts, "batch_size", 1),
    )

    destination = Path(work_dir) if work_dir else Path(".")
    destination.mkdir(parents=True, exist_ok=True)

    segments: list[Segment] = []
    detected_languages: list[str] = []
    per_track_segments: dict[str, int] = {}
    alignment_reports: dict[str, dict] = {}
    alignment_backend = getattr(opts, "alignment_backend", "auto")

    raw_tracks: list[tuple[str, Path, list[Any], str]] = []
    try:
        for index, track in enumerate(result.tracks):
            speaker = (
                result.speaker_ids[index]
                if result.speaker_ids and index < len(result.speaker_ids)
                else f"SPEAKER_{index:02d}"
            )
            track_path = destination / f"voice_{speaker}.wav"
            sf.write(str(track_path), track, result.sample_rate)

            report(
                f"Trascrizione voce {speaker} ({index + 1}/{len(result.tracks)})...",
                0.55 + 0.35 * (index / max(1, len(result.tracks))),
            )
            transcribed, language_code = transcriber.transcribe(
                track_path,
                language=language,
                vad_filter=True,
                initial_prompt=initial_prompt,
                quality_retries=quality_retries,
                on_progress=(lambda done, total, i=index: stages.update(
                    "asr", (i + done / total) / len(result.tracks) if total else None,
                    f"Voce {i + 1}/{len(result.tracks)}: {done:.1f}/{total:.1f} s")) if stages else None,
                on_status=stages.asr_status("asr") if stages else None,
            )
            if language_code:
                detected_languages.append(language_code)
            raw_tracks.append((speaker, track_path, transcribed, language_code))
    finally:
        transcriber.close()

    if stages:
        stages.transcribed("asr")
        stages.start("alignment")
    for track_index, (speaker, track_path, transcribed, language_code) in enumerate(raw_tracks):
        transcribed, align_report = align_segments(
            transcribed,
            track_path,
            language=language_code or language,
            backend=alignment_backend,
            device=resolved_device,
        )
        alignment_reports[speaker] = align_report.to_dict()
        if stages:
            stages.update("alignment", (track_index + 1) / len(raw_tracks), f"Voce {speaker}")
        count = 0
        for piece in transcribed:
            text = piece.text.strip()
            if not text:
                continue
            words = [
                Word(
                    start=round(item[0], 3),
                    end=round(item[1], 3),
                    text=item[2].strip(),
                    probability=float(item[3]) if len(item) > 3 else 1.0,
                )
                for item in (piece.words or [])
            ]
            segments.append(
                Segment(round(piece.start, 3), round(piece.end, 3), text, speaker, words)
            )
            count += 1
        per_track_segments[speaker] = count

    from .merge import build_utterances_from_segments, utterances_to_segments

    if stages:
        stages.complete("alignment")
        stages.start("merge")
    utterances = build_utterances_from_segments(segments)
    if stages:
        stages.complete("merge")
        stages.skip("review", "Attribuzione dalla traccia separata; revisione standard non prevista")
        stages.start("reconstruct")
    projected = utterances_to_segments(utterances)
    if stages:
        stages.complete("reconstruct")
    detected_language = detected_languages[0] if detected_languages else ""

    metadata = {
        "strategy": "speaker_isolated",
        "detected_languages": detected_languages,
        "segments_per_speaker": per_track_segments,
        "total_utterances": len(utterances),
        "alignment": alignment_reports,
        **result.as_metadata(),
    }
    return utterances, projected, detected_language, metadata
