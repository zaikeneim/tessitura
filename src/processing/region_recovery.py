"""Local, bounded recovery proposals. Existing events are never overwritten."""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from ..models.audio_artifacts import AudioView, SpeechRegion
from ..models.session import Word
from ..models.transcript_artifacts import Utterance
from .audio_utils import load_mono_16k
from .checkpoints import file_digest


def recover_region(source, start_s, end_s, output_dir, *, existing_events=(), separate=False,
                   potential_speakers=None, model_size="large-v3", language=None, device="auto",
                   context_s=2., transcriber_factory=None, separator=None):
    """Extract context in native channels, then transcribe each proposed source.

    Candidate speaker IDs remain null: separator channel numbers are not speaker
    identities. Duplicate candidates are visible with their evidence for review.
    """
    if not all(math.isfinite(v) for v in (start_s, end_s, context_s)) or not 0 <= start_s < end_s or end_s - start_s > 120 or not 0 <= context_s <= 10:
        raise ValueError("Seleziona una regione di durata massima 120 s.")
    source = Path(source)
    output = Path(output_dir) / f"region-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    info = sf.info(str(source))
    if start_s >= info.duration:
        raise ValueError("La regione è oltre la fine dell'audio.")
    left, right = max(0, start_s - context_s), min(info.duration, end_s + context_s)
    with sf.SoundFile(str(source)) as reader:
        reader.seek(round(left * info.samplerate))
        native = reader.read(round((right - left) * info.samplerate), dtype="float32", always_2d=True)
    original = output / "original-context.wav"
    sf.write(str(original), native, info.samplerate, subtype="FLOAT")
    audio = load_mono_16k(original)
    tracks, separation, delay = [audio], {"status": "not_requested"}, 0.
    if separate:
        from .separation import separate_region
        separate_fn = separator or separate_region
        result = separate_fn(audio, AudioView(sample_rate=16000, duration_s=len(audio) / 16000),
                             SpeechRegion(start_s=0, end_s=len(audio) / 16000), num_speakers=potential_speakers)
        separation = result.to_dict()
        tracks = result.tracks or [audio]
        delay = result.delay_compensation_s
        if potential_speakers and len(tracks) < potential_speakers:
            separation["incomplete_source_count"] = {"potential": potential_speakers, "returned": len(tracks)}
    from .pipeline import resolve_device
    from .transcribe import Transcriber
    effective_device, compute_type = resolve_device(device)
    factory = transcriber_factory or Transcriber
    transcriber = factory(model_size=model_size, device=effective_device, compute_type=compute_type)
    candidates = []
    started = time.monotonic()
    source_hash = file_digest(source)
    normalized = lambda text: " ".join(text.casefold().split()).strip(" .?!,…")
    try:
        for index, track in enumerate(tracks):
            path = output / f"candidate-source-{index}.wav"
            sf.write(str(path), track, 16000)
            segments, detected = transcriber.transcribe(path, language=language, quality_retries=0)
            for segment in segments:
                absolute_start, absolute_end = left + segment.start - delay, left + segment.end - delay
                if absolute_end <= start_s or absolute_start >= end_s:
                    continue
                # Keep boundary-crossing text and its full context timestamps.
                words = [Word(max(0, left + a - delay), max(0, left + b - delay), text, prob)
                         for a, b, text, prob in segment.words]
                duplicate_ids = [u.event_id for u in existing_events if u.end_s > absolute_start and u.start_s < absolute_end
                                 and normalized(u.text) == normalized(segment.text)]
                duplicate_words = []
                for word in words:
                    for existing in existing_events:
                        matches = [w for w in existing.words if normalized(w.text) == normalized(word.text)
                                   and min(w.end, word.end) >= max(w.start, word.start) - .15]
                        if matches:
                            duplicate_words.append({"text": word.text, "start_s": word.start, "end_s": word.end,
                                                    "event_id": existing.event_id})
                event = Utterance(start_s=max(0, absolute_start), end_s=min(info.duration, absolute_end),
                    speaker=None, text=segment.text.strip(), words=words, source_audio_ref=str(source),
                    processing_state="candidate", uncertain=True,
                    asr_provenance={"model": model_size, "language": detected, "source_sha256": source_hash,
                        "candidate_audio": str(path), "candidate_audio_offset_s": left - delay,
                        "original_context_audio": str(original), "selected_interval": [start_s, end_s],
                        "context_interval": [left, right], "source_index": index, "separation": separation},
                    diarization_provenance={"identity": "unresolved", "source_index_is_not_speaker_id": True})
                candidates.append({"candidate_id": uuid.uuid4().hex, "status": "proposed", "event": event.to_dict(),
                    "duplicate_of_event_ids": duplicate_ids,
                    "potential_duplicate_words": duplicate_words,
                    "review_note": "Possibile duplicato: confrontare prima di accettare" if duplicate_ids else "Candidato non verificato"})
    finally:
        transcriber.close()
    result = {"candidates": candidates, "source_sha256": source_hash, "selection": [start_s, end_s],
              "separation": separation, "elapsed_s": time.monotonic() - started,
              "original_audio": str(source), "output_dir": str(output)}
    from .global_workflow import write_json
    write_json(output / "proposals.json", result)
    return result
