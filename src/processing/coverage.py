"""Analisi della copertura del parlato e pianificazione del recupero omissioni.

Implementa i requisiti del pacchetto P04 (sezione 6.5 del piano esecutivo):
- Rilevamento attivita' vocale indipendente dall'ASR (su raw o per canale)
- Confronto copertura tra parlato reale e segmenti emessi per individuare:
  * frasi omesse (ASR completamente vuoto)
  * parole tagliate ai bordi
  * interiezioni brevi (< 3 parole)
  * voci deboli/distanti
- Pianificazione dei retry distribuita sull'intera durata del file (non solo all'inizio)
- Riconciliazione dei bordi con espansione controllata e protezione anti-allucinazione
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models.audio_artifacts import SpeechRegion
from .audio_utils import rms_dbfs


def detect_speech_activity(
    data: np.ndarray,
    sr: int = 16000,
    *,
    frame_ms: int = 20,
    min_speech_ms: int = 140,
    max_silence_ms: int = 260,
    energy_threshold_db: float = -45.0,
) -> list[SpeechRegion]:
    """Rileva regioni di parlato dall'energia acustica indipendente dall'ASR."""
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data

    if len(mono) == 0:
        return []

    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return []

    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    # Calcola livello in dBFS per ciascun frame
    frame_db = np.full(n_frames, -100.0, dtype=np.float32)
    valid_mask = energies > 1e-7
    frame_db[valid_mask] = 20.0 * np.log10(energies[valid_mask])

    # Soglia per il parlato: usa energy_threshold_db come soglia base.
    # Se il fondo di rumore e' misurabile con dinamica sufficiente, adatta la soglia
    # senza mai superare -35 dBFS (per non oscurare il parlato).
    if len(frame_db) > 10:
        p10 = float(np.percentile(frame_db, 10))
        p90 = float(np.percentile(frame_db, 90))
        if p90 - p10 > 12.0 and p10 > -75.0:
            threshold = min(-35.0, max(energy_threshold_db, p10 + 6.0))
        else:
            threshold = energy_threshold_db
    else:
        threshold = energy_threshold_db

    active_frames = frame_db >= threshold

    # Bridging dei silenzi brevi (< max_silence_ms)
    max_silence_frames = max(1, int(max_silence_ms / frame_ms))
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))

    # Chiusura morfologica 1D per colmare i brevi buchi di silenzio
    bridged = active_frames.copy()
    silence_count = 0
    silence_start = -1
    for i in range(len(bridged)):
        if not bridged[i]:
            if silence_count == 0:
                silence_start = i
            silence_count += 1
        else:
            if 0 < silence_count <= max_silence_frames and silence_start > 0:
                bridged[silence_start:i] = True
            silence_count = 0
            silence_start = -1

    # Raggruppamento in regioni continue
    regions: list[SpeechRegion] = []
    in_region = False
    reg_start_frame = 0

    for i, active in enumerate(bridged):
        if active and not in_region:
            in_region = True
            reg_start_frame = i
        elif not active and in_region:
            in_region = False
            duration_frames = i - reg_start_frame
            if duration_frames >= min_speech_frames:
                start_s = reg_start_frame * frame_ms / 1000.0
                end_s = i * frame_ms / 1000.0
                reg_energies = frame_db[reg_start_frame:i]
                avg_db = float(np.mean(reg_energies))
                conf = min(1.0, max(0.1, (avg_db - threshold + 20.0) / 30.0))
                regions.append(
                    SpeechRegion(
                        start_s=round(start_s, 3),
                        end_s=round(end_s, 3),
                        vad_confidence=round(conf, 3),
                        estimated_quality=round(avg_db, 1),
                    )
                )

    if in_region and (len(bridged) - reg_start_frame) >= min_speech_frames:
        start_s = reg_start_frame * frame_ms / 1000.0
        end_s = len(bridged) * frame_ms / 1000.0
        regions.append(
            SpeechRegion(
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                vad_confidence=0.8,
            )
        )

    return regions


def find_uncovered_regions(
    speech_regions: list[SpeechRegion],
    asr_segments: list[Any],
    *,
    min_omission_s: float = 0.25,
) -> list[SpeechRegion]:
    """Confronta il parlato acustico con i segmenti ASR per individuare vuoti e bordi tagliati."""
    suspects: list[SpeechRegion] = []

    for s_reg in speech_regions:
        overlapping_segs = [
            seg for seg in asr_segments
            if min(s_reg.end_s, float(seg.end)) > max(s_reg.start_s, float(seg.start))
        ]

        if not overlapping_segs:
            # Parlato rilevato ma NESSUN segmento ASR emesso
            if s_reg.duration_s >= min_omission_s:
                s_reg.suspect_reasons = ["empty_asr"]
                suspects.append(s_reg)
            continue

        # Calcolo overlap effettivo
        covered_duration = sum(
            max(0.0, min(s_reg.end_s, float(seg.end)) - max(s_reg.start_s, float(seg.start)))
            for seg in overlapping_segs
        )

        reasons: list[str] = []
        # Mancanza di copertura parziale significativa
        if (s_reg.duration_s - covered_duration) >= 0.4:
            reasons.append("partial_coverage")

        # Bordo tagliato all'inizio o alla fine
        earliest_start = min(float(seg.start) for seg in overlapping_segs)
        latest_end = max(float(seg.end) for seg in overlapping_segs)

        if s_reg.start_s < earliest_start - 0.35:
            reasons.append("boundary_cutoff_start")
        if s_reg.end_s > latest_end + 0.35:
            reasons.append("boundary_cutoff_end")

        # Interiezione breve (< 1.5s) con alto no_speech
        if s_reg.duration_s < 1.5:
            for seg in overlapping_segs:
                if float(getattr(seg, "no_speech_prob", 0.0)) > 0.55:
                    reasons.append("short_interjection_uncertain")
                    break

        if reasons:
            s_reg.suspect_reasons = reasons
            suspects.append(s_reg)

    return suspects


def plan_distributed_recovery(
    suspects: list[SpeechRegion],
    total_duration_s: float,
    *,
    max_retries: int = 15,
    pad_s: float = 1.0,
    slot_duration_s: float = 45.0,
) -> list[tuple[float, float, SpeechRegion]]:
    """Distribuisce i retry uniformemente sull'intero audio per evitare di esaurirli nei primi minuti.

    Raggruppa la traccia in slot temporali e seleziona i sospetti piu' critici in ciascuno.
    """
    if not suspects or total_duration_s <= 0:
        return []

    # Assegna priorita' ai sospetti
    def priority_score(reg: SpeechRegion) -> float:
        score = 0.0
        if "empty_asr" in reg.suspect_reasons:
            score += 10.0
        if any("boundary" in r for r in reg.suspect_reasons):
            score += 5.0
        if any("short" in r for r in reg.suspect_reasons):
            score += 3.0
        score += min(3.0, reg.duration_s)
        return score

    sorted_suspects = sorted(suspects, key=priority_score, reverse=True)

    n_slots = max(1, int(math.ceil(total_duration_s / slot_duration_s)))
    slot_buckets: dict[int, list[SpeechRegion]] = {i: [] for i in range(n_slots)}

    for reg in sorted_suspects:
        slot_idx = min(n_slots - 1, max(0, int(reg.start_s // slot_duration_s)))
        slot_buckets[slot_idx].append(reg)

    planned: list[tuple[float, float, SpeechRegion]] = []
    # Raccogli round-robin dai bucket fino a max_retries
    retry_count = 0
    round_idx = 0
    while retry_count < max_retries and any(slot_buckets.values()):
        added_in_round = False
        for i in range(n_slots):
            if slot_buckets[i]:
                reg = slot_buckets[i].pop(0)
                start_padded = max(0.0, reg.start_s - pad_s)
                end_padded = min(total_duration_s, reg.end_s + pad_s)
                planned.append((start_padded, end_padded, reg))
                retry_count += 1
                added_in_round = True
                if retry_count >= max_retries:
                    break
        if not added_in_round:
            break
        round_idx += 1

    # Ordina cronologicamente le finestre pianificate
    planned.sort(key=lambda item: item[0])
    return planned


def is_hallucination_or_noise(
    text: str,
    audio_slice: np.ndarray,
    *,
    compression_ratio: float = 1.0,
    no_speech_prob: float = 0.0,
    silence_threshold_db: float = -55.0,
) -> bool:
    """Verifica antiallucinazione: rifiuta testo decodificato su silenzio o con ripetizioni patologiche."""
    t = text.strip().casefold()
    if not t:
        return True

    # Se l'audio e' silenzio acustico assoluto, non puo' esserci parlato
    level = rms_dbfs(audio_slice)
    if level < silence_threshold_db:
        return True

    # Segnale di ripetizione o collasso Whisper
    if compression_ratio > 2.6:
        return True

    if no_speech_prob > 0.8:
        return True

    # Pattern allucinati tipici di Whisper su rumore ("Sottotitoli a cura di...", "Grazie per la visione")
    hallucination_phrases = {
        "sottotitoli", "trascrizione a cura", "grazie per la visione", "iscriviti al canale",
        "amara.org", "subtitles by", "thank you for watching",
    }
    if any(h in t for h in hallucination_phrases):
        return True

    return False
