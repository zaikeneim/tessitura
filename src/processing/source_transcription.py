"""Trascrizione indipendente delle sorgenti audio con diarizzazione per sorgente.

Esegue l'ASR separatamente su microfono e sistema, applica l'offset temporale
globale una sola volta, conserva il Bleed Check per RMS+latenza come diagnostica
senza soppressione e ricompone gli interventi in Utterance preservando tutti
gli overlap per ID.
"""

from __future__ import annotations

import difflib
import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from ..models.session import Segment, Word
from ..models.transcript_artifacts import Utterance
from . import audio_utils
from .alignment import align_segments
from .transcribe import Transcriber, TSegment
from .diarize import Diarizer
from .progress import StageReporter, SOURCE_STAGES


def _attribute_source(segments, path, source, options, device, reports, offset=0.0, stages=None):
    """Identify voices within a source, keeping the source in every speaker ID."""
    stage_id = f"{source}_diarization"
    base = "SPEAKER_MIC" if source == "mic" else "SPEAKER_SYS"
    verified = source == "mic" and getattr(options, "mic_single_speaker_verified", False)
    if verified:
        if stages:
            stages.skip(stage_id, "Microfono monovoce verificato")
        reports[source] = {"status": "verified_single_speaker", "speaker": base}
        return [(segment, base) for segment in segments]
    if not getattr(options, "diarize", True) or not segments:
        if stages:
            stages.skip(stage_id, "Diarizzazione disattivata" if segments else "Nessun parlato riconosciuto")
        reports[source] = {"status": "disabled" if segments else "no_speech", "source_label_only": True}
        return [(segment, base) for segment in segments]
    if stages:
        stages.start(stage_id)
    diarizer = None
    warning = None
    detail = ""
    try:
        diarizer = Diarizer(device=device)
        result = diarizer.diarize_detailed(path, **(
            {"on_progress": stages.diarization_progress(stage_id)} if stages else {}))
        reports[source] = {
            "status": "diarized" if result.turns else "unresolved",
            "timebase": "recording_absolute",
            "offset_s": offset,
            "turns": [{"start": turn.start + offset, "end": turn.end + offset, "speaker": f"{base}_{turn.speaker}"}
                      for turn in result.turns],
            "overlaps": [[start + offset, end + offset] for start, end in result.overlaps],
            "model": result.model_repo,
            "revision": result.model_revision,
        }
        warning = "Nessuna voce risolta" if not result.turns else None
        if not result.turns:
            return [(segment, f"{base}_UNKNOWN") for segment in segments]
        from .merge import merge_transcript_and_diarization
        # Preserve ASR hypothesis boundaries, including simultaneous hypotheses.
        attributed = [attributed for segment in segments
                      for attributed in merge_transcript_and_diarization(
                          [segment], result.exclusive or result.turns,
                          preserve_short_turns=True, overlaps=result.overlaps)]
        return [(TSegment(segment.start, segment.end, segment.text,
                          [(word.start, word.end, word.text, word.probability or 0.0)
                           for word in segment.words]), f"{base}_{segment.speaker}")
                for segment in attributed]
    except Exception as exc:
        # A missing diarization model must not turn source identity into a claim
        # that everybody on a microphone or playback endpoint is one person.
        reports[source] = {"status": "unresolved", "warning": str(exc), "source_label_only": True}
        warning = str(exc)
        detail = "Attribuzione non risolta; identità sconosciuta conservata"
        return [(segment, f"{base}_UNKNOWN") for segment in segments]
    finally:
        if diarizer is not None:
            diarizer.close()
        if stages:
            stages.complete(stage_id, detail, warning=warning)


def _mark_source_overlaps(utterances):
    """Color simultaneous streams and join transitive overlap groups."""
    active = []
    for item in utterances:
        active = [other for other in active if other.end_s > item.start_s]
        if active:
            groups = {other.overlap_group_id for other in active if other.overlap_group_id}
            group = sorted(groups)[0] if groups else f"ov_{uuid.uuid4().hex[:8]}"
            if len(groups) > 1:
                for other in utterances:
                    if other.overlap_group_id in groups:
                        other.overlap_group_id = group
            for other in active:
                other.overlap_group_id = group
            item.overlap_group_id = group
            used = {other.stream_id for other in active}
            item.stream_id = next(stream for stream in range(len(used) + 1) if stream not in used)
        active.append(item)


def _clock_diagnostics(metadata, source):
    """Measure clock drift without mistaking short-term read jitter for drift."""
    from .capture_sync import clock_diagnostics
    return clock_diagnostics(metadata, source)


def _compute_segment_rms_dbfs(path: Path, start_s: float, end_s: float) -> float:
    """Calcola il livello RMS in dBFS di una finestra temporale da file audio."""
    if not path.is_file() or end_s <= start_s:
        return -np.inf
    info = sf.info(str(path))
    start_frame = max(0, int(start_s * info.samplerate))
    end_frame = min(info.frames, int(end_s * info.samplerate))
    if end_frame <= start_frame:
        return -np.inf
    data, _ = sf.read(str(path), start=start_frame, stop=end_frame, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return audio_utils.rms_dbfs(mono)


def _compute_cross_correlation_and_lag(
    path1: Path,
    start1_s: float,
    end1_s: float,
    path2: Path,
    start2_s: float,
    end2_s: float,
    max_lag_s: float = 0.080,
) -> tuple[float, float]:
    """Calcola la correlazione incrociata normalizzata massima e il ritardo di fase (lag).

    Restituisce (max_corr, lag_s) dove lag_s rappresenta il ritardo relativo al picco.
    Se uno dei file non e' leggibile o l'audio e' silenzioso, restituisce (0.0, 0.0).
    """
    if not path1.is_file() or not path2.is_file():
        return 0.0, 0.0

    dur = min(end1_s - start1_s, end2_s - start2_s)
    if dur < 0.1:
        return 0.0, 0.0

    try:
        info1 = sf.info(str(path1))
        info2 = sf.info(str(path2))

        sr = min(info1.samplerate, info2.samplerate)
        sr = min(sr, 16000)

        f1_start = max(0, int(start1_s * info1.samplerate))
        f1_stop = min(info1.frames, int((start1_s + dur) * info1.samplerate))
        if f1_stop <= f1_start:
            return 0.0, 0.0
        d1, sr1 = sf.read(str(path1), start=f1_start, stop=f1_stop, dtype="float32", always_2d=True)
        m1 = d1.mean(axis=1)

        f2_start = max(0, int(start2_s * info2.samplerate))
        f2_stop = min(info2.frames, int((start2_s + dur) * info2.samplerate))
        if f2_stop <= f2_start:
            return 0.0, 0.0
        d2, sr2 = sf.read(str(path2), start=f2_start, stop=f2_stop, dtype="float32", always_2d=True)
        m2 = d2.mean(axis=1)

        if sr1 != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr1), int(sr))
            m1 = resample_poly(m1, sr // g, int(sr1) // g)
        if sr2 != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr2), int(sr))
            m2 = resample_poly(m2, sr // g, int(sr2) // g)

        min_len = min(len(m1), len(m2))
        if min_len < int(sr * 0.1):
            return 0.0, 0.0

        x = m1[:min_len].astype(np.float64)
        y = m2[:min_len].astype(np.float64)

        x -= np.mean(x)
        y -= np.mean(y)
        std_x = float(np.std(x))
        std_y = float(np.std(y))
        if std_x < 1e-6 or std_y < 1e-6:
            return 0.0, 0.0

        from scipy.signal import correlate
        corr = correlate(y, x, mode="full") / (min_len * std_x * std_y)
        lags = np.arange(-min_len + 1, min_len) / float(sr)

        lag_mask = np.abs(lags) <= max_lag_s
        if not np.any(lag_mask):
            return 0.0, 0.0

        corr_win = corr[lag_mask]
        lags_win = lags[lag_mask]

        max_idx = int(np.argmax(corr_win))
        max_corr = float(corr_win[max_idx])
        peak_lag = float(lags_win[max_idx])
        return max_corr, peak_lag
    except Exception:
        return 0.0, 0.0


def transcribe_sources(
    mic_path: str | Path | None,
    system_path: str | Path | None,
    *,
    offsets_s: list[float] | None = None,
    options: Any | None = None,
    progress: Callable[[str, float], None] | None = None,
    work_dir: Path | None = None,
    original_paths: dict | None = None,
    synchronization: dict | None = None,
    stages: StageReporter | None = None,
    on_stage=None,
) -> tuple[list[Utterance], list[Segment], str, dict]:
    """Trascrive mic e sistema in modo indipendente e li ricompone in Utterance."""
    own_stages = stages is None
    stages = stages or StageReporter(SOURCE_STAGES, progress, on_stage)
    report = stages.note
    with stages.guard():
        mic_p = Path(mic_path) if mic_path else None
        sys_p = Path(system_path) if system_path else None

        if not mic_p and not sys_p:
            raise ValueError("Serve almeno una sorgente audio valida")
        for source in (mic_p, sys_p):
            if source is not None and not source.is_file():
                raise ValueError(f"Sorgente audio non disponibile: {source}")

        originals = original_paths or {source: path for source, path in (("mic", mic_p), ("system", sys_p)) if path is not None}
        if synchronization is None:
            from .capture_sync import synchronize_capture_sources
            synchronized = synchronize_capture_sources(
                mic_p, sys_p, offsets_s=offsets_s,
                metadata=getattr(options, "recording_metadata", None), work_dir=work_dir,
            )
            mic_p, sys_p, synchronization = synchronized.mic_path, synchronized.system_path, synchronized.report

        opts = options
        model_size = getattr(opts, "model_size", "large-v3")
        device = getattr(opts, "device", "auto")
        compute_type = getattr(opts, "compute_type", None)
        language = getattr(opts, "language", None)
        initial_prompt = getattr(opts, "initial_prompt", None)
        quality_retries = getattr(opts, "quality_retries", 24)
        headphones_confirmed = getattr(opts, "headphones_confirmed", True)

        from .pipeline import resolve_device
        resolved_dev, resolved_ct = resolve_device(device)
        if compute_type:
            resolved_ct = compute_type

        supplied_offsets = iter(offsets_s or [])
        offset_mic = float(next(supplied_offsets, 0.0)) if mic_p else 0.0
        offset_sys = float(next(supplied_offsets, 0.0)) if sys_p else 0.0
        if any(not math.isfinite(value) or value < 0 for value in (offset_mic, offset_sys)):
            raise ValueError("Gli offset audio devono essere finiti e non negativi")

        stages.skip("source_denoise", "Strategia sorgenti: ASR sugli originali; denoise non applicato"
                    if getattr(opts, "denoise", False) else "Riduzione rumore disattivata")
        stages.start("echo")
        asr_mic_p = mic_p
        echo_report = {"status": "not_requested"}
        if mic_p and sys_p and (
            getattr(opts, "echo_cancellation", False)
            or getattr(opts, "recording_mode", None) == "speakers"
            or headphones_confirmed is False
        ):
            from .echo import evaluate_echo
            output_root = work_dir or mic_p.parent
            echo = evaluate_echo(
                mic_p, sys_p, output_root / f"{mic_p.stem}_echo_derivative.wav",
                mic_offset=offset_mic, system_offset=offset_sys,
                apply=getattr(opts, "echo_cancellation", False),
            )
            # Candidate echo cancellation has no reviewed speech benchmark yet.
            # Transcribe the original microphone even when a candidate derivative
            # exists: low-level local speech must not depend on a residual gate.
            echo_report = {**echo.report, "asr_input": str(mic_p), "candidate_only": True}

            stages.complete("echo", warning=echo_report.get("warning"))
        else:
            stages.skip("echo", "Eco non richiesto o seconda sorgente assente")

        raw_hypotheses: list[dict] = []
        detected_languages: list[str] = []
        alignment_reports: dict[str, dict] = {}
        diarization_reports: dict[str, dict] = {}
        alignment_backend = getattr(opts, "alignment_backend", "auto")

        stages.start("asr_model")
        report("Caricamento modello ASR...", 0.10)
        transcriber = Transcriber(
            model_size=model_size,
            device=resolved_dev,
            compute_type=resolved_ct,
            batch_size=getattr(opts, "batch_size", 1),
            on_download=lambda name, pct: stages.update("asr_model", detail=f"Scaricamento {name}: {pct}%"),
        )
        stages.complete("asr_model")
        try:
            # 1. Trascrizione Microfono
            mic_segs: list[TSegment] = []
            mic_lang = ""
            if mic_p and mic_p.is_file():
                stages.start("mic_asr")
                report("Trascrizione traccia microfono...", 0.25)
                mic_segs, mic_lang = transcriber.transcribe(
                    asr_mic_p,
                    language=language,
                    vad_filter=True,
                    initial_prompt=initial_prompt,
                    quality_retries=quality_retries,
                    on_progress=stages.asr_progress("mic_asr"),
                    on_status=stages.asr_status("mic_asr"),
                )
                if mic_lang:
                    detected_languages.append(mic_lang)

            # 2. Trascrizione Sistema
            if mic_p and sys_p:
                stages.transcribed("mic_asr")
            sys_segs: list[TSegment] = []
            sys_lang = ""
            if sys_p and sys_p.is_file():
                stages.start("system_asr")
                report("Trascrizione traccia audio sistema...", 0.60)
                sys_segs, sys_lang = transcriber.transcribe(
                    sys_p,
                    language=language,
                    vad_filter=True,
                    initial_prompt=initial_prompt,
                    quality_retries=quality_retries,
                    on_progress=stages.asr_progress("system_asr"),
                    on_status=stages.asr_status("system_asr"),
                )
                if sys_lang:
                    detected_languages.append(sys_lang)
        finally:
            transcriber.close()
        stages.transcribed("system_asr" if sys_p else "mic_asr")

        if mic_p and mic_p.is_file() and mic_segs:
            stages.start("mic_alignment")
            mic_segs, mic_align = align_segments(
                mic_segs,
                asr_mic_p,
                language=mic_lang or language,
                backend=alignment_backend,
                device=resolved_dev,
            )
            alignment_reports["mic"] = mic_align.to_dict()
            stages.aligned("mic_alignment", mic_align)
            for tseg, speaker in _attribute_source(
                mic_segs, mic_p, "mic", opts, resolved_dev, diarization_reports, offset_mic, stages=stages
            ):
                if not tseg.text.strip():
                    continue
                words = [
                    Word(
                        start=round(w[0] + offset_mic, 3),
                        end=round(w[1] + offset_mic, 3),
                        text=w[2].strip(),
                        probability=float(w[3]) if len(w) > 3 else 1.0,
                    )
                    for w in (tseg.words or [])
                ]
                raw_hypotheses.append({
                    "source": "mic",
                    "start_s": round(tseg.start + offset_mic, 3),
                    "end_s": round(tseg.end + offset_mic, 3),
                    "speaker": speaker,
                    "text": tseg.text.strip(),
                    "words": words,
                    "hypothesis_id": f"hyp_mic_{uuid.uuid4().hex[:8]}",
                })

        if sys_p and sys_p.is_file() and sys_segs:
            stages.start("system_alignment")
            sys_segs, sys_align = align_segments(
                sys_segs,
                sys_p,
                language=sys_lang or language,
                backend=alignment_backend,
                device=resolved_dev,
            )
            alignment_reports["system"] = sys_align.to_dict()
            stages.aligned("system_alignment", sys_align)
            for tseg, speaker in _attribute_source(
                sys_segs, sys_p, "system", opts, resolved_dev, diarization_reports, offset_sys, stages=stages
            ):
                if not tseg.text.strip():
                    continue
                words = [
                    Word(
                        start=round(w[0] + offset_sys, 3),
                        end=round(w[1] + offset_sys, 3),
                        text=w[2].strip(),
                        probability=float(w[3]) if len(w) > 3 else 1.0,
                    )
                    for w in (tseg.words or [])
                ]
                raw_hypotheses.append({
                    "source": "system",
                    "start_s": round(tseg.start + offset_sys, 3),
                    "end_s": round(tseg.end + offset_sys, 3),
                    "speaker": speaker,
                    "text": tseg.text.strip(),
                    "words": words,
                    "hypothesis_id": f"hyp_sys_{uuid.uuid4().hex[:8]}",
                })

        for source, path, pieces in (("mic", mic_p, mic_segs), ("system", sys_p, sys_segs)):
            if path is None:
                stages.skip(f"{source}_asr", "Sorgente assente")
            if not pieces:
                reason = "Sorgente assente" if path is None else "Nessun parlato riconosciuto"
                stages.skip(f"{source}_alignment", reason)
                stages.skip(f"{source}_diarization", reason)

        stages.start("source_merge")
        detected_lang = detected_languages[0] if detected_languages else "it"
        report("Ricomposizione e verifica sovrapposizioni...", 0.90)

        # 3. Bleed Check (se cuffie non confermate)
        # Correlation, relative level and matching text create review evidence
        # only. They cannot distinguish echo from real double-talk reliably.
        suspected_ids: set[str] = set()
        if headphones_confirmed is False and mic_p and sys_p:
            report("Verifica rientro acustico (Bleed Check)...", 0.92)
            mic_items = [h for h in raw_hypotheses if h["source"] == "mic"]
            sys_items = [h for h in raw_hypotheses if h["source"] == "system"]
            for m in mic_items:
                for s in sys_items:
                    # Sovrapposizione temporale
                    overlap_dur = min(m["end_s"], s["end_s"]) - max(m["start_s"], s["start_s"])
                    if overlap_dur > 0.3:
                        # Similarità testuale
                        ratio = difflib.SequenceMatcher(None, m["text"].casefold(), s["text"].casefold()).ratio()
                        if ratio > 0.65:
                            # Misurazione energetica RMS nelle finestre locali corrispondenti
                            m_start_local = max(0.0, m["start_s"] - offset_mic)
                            m_end_local = max(m_start_local, m["end_s"] - offset_mic)
                            s_start_local = max(0.0, s["start_s"] - offset_sys)
                            s_end_local = max(s_start_local, s["end_s"] - offset_sys)

                            rms_m = _compute_segment_rms_dbfs(mic_p, m_start_local, m_end_local)
                            rms_s = _compute_segment_rms_dbfs(sys_p, s_start_local, s_end_local)
                            delta_rms = abs(rms_m - rms_s)

                            # Se la differenza e' > 12 dB, verifichiamo anche la correlazione incrociata di fase
                            # per escludere falsi positivi quando entrambi dicono la stessa parola
                            if delta_rms > 12.0:
                                corr_coeff, lag_s = _compute_cross_correlation_and_lag(
                                    mic_p, m_start_local, m_end_local,
                                    sys_p, s_start_local, s_end_local,
                                    max_lag_s=0.080,
                                )
                                # Bleed confermato se il segnale e' correlato (r > 0.50) entro finestra fisica <= 80 ms
                                if corr_coeff > 0.50:
                                    if rms_m - rms_s > 12.0:
                                        # Possibile duplicazione attenuata nel sistema: solo revisione.
                                        suspected_ids.add(s["hypothesis_id"])
                                        s["suspected_echo"] = True
                                        s["dominant_id"] = m["hypothesis_id"]
                                        s["bleed_correlation"] = corr_coeff
                                        s["bleed_lag_s"] = lag_s
                                    elif rms_s - rms_m > 12.0:
                                        # Possibile rientro nel microfono: solo revisione.
                                        suspected_ids.add(m["hypothesis_id"])
                                        m["suspected_echo"] = True
                                        m["dominant_id"] = s["hypothesis_id"]
                                        m["bleed_correlation"] = corr_coeff
                                        m["bleed_lag_s"] = lag_s

        # 4. Ricomposizione ordinata per start_s
        sorted_items = sorted(raw_hypotheses, key=lambda x: (x["start_s"], x["end_s"]))
        utterances: list[Utterance] = []

        for item in sorted_items:
            overlap_grp = None
            stream = 0

            # Tutte le sovrapposizioni sono verificate insieme dopo la ricomposizione.

            candidate_meta: list[dict] = [{
                "source": item["source"],
                "audio_path": str(originals[item["source"]]),
                "processing_audio_path": str(mic_p if item["source"] == "mic" else sys_p),
                "offset_s": offset_mic if item["source"] == "mic" else offset_sys,
                "reason": "capture_source_provenance",
            }]
            if item.get("suspected_echo"):
                meta_entry = {
                    "suspected_echo": True,
                    "suppressed_bleed": False,
                    "dominant_id": item.get("dominant_id"),
                    "reason": "possible_acoustic_echo_review",
                }
                if "bleed_correlation" in item:
                    meta_entry["correlation"] = round(float(item["bleed_correlation"]), 3)
                if "bleed_lag_s" in item:
                    meta_entry["lag_ms"] = round(float(item["bleed_lag_s"]) * 1000.0, 1)
                candidate_meta.append(meta_entry)

            utt = Utterance(
                start_s=item["start_s"],
                end_s=item["end_s"],
                utterance_id=str(uuid.uuid4().hex[:12]),
                stream_id=stream,
                overlap_group_id=overlap_grp,
                speaker=item["speaker"],
                candidate_speakers=candidate_meta,
                text=item["text"],
                words=item["words"],
                hypothesis_ids=[item["hypothesis_id"]],
            )
            utterances.append(utt)

        _mark_source_overlaps(utterances)
        # 5. Proiezione su Segment senza eliminare gli interventi simultanei o sospetti.
        from .merge import utterances_to_segments
        segments = utterances_to_segments(utterances)

        metadata = {
            "strategy": getattr(opts, "strategy", "source_aware"),
            "headphones_confirmed": headphones_confirmed,
            "offset_mic_s": offset_mic,
            "offset_sys_s": offset_sys,
            "detected_languages": detected_languages,
            "suppressed_bleed_count": 0,
            "suspected_echo_count": len(suspected_ids),
            "total_utterances": len(utterances),
            "active_segments": len(segments),
            "alignment": alignment_reports,
            "diarization": diarization_reports,
            "echo": echo_report,
            "synchronization": synchronization,
            "preprocessing": {"denoise_applied": False, "echo_applied_to_primary_asr": False,
                              "clock_synchronization": "non_destructive_derivative_if_reliable"},
            "sources": {
                source: {"original": str(originals[source]), "processing_audio": str(path), "offset_s": offset}
                for source, path, offset in (("mic", mic_p, offset_mic), ("system", sys_p, offset_sys))
                if path is not None
            },
        }

        stages.complete("source_merge")
        if own_stages:
            stages.assert_complete()
        return utterances, segments, detected_lang, metadata
