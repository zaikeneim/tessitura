"""Resumable full-recording audit from content-verified model checkpoints."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from ..models.session import AudioSources, Session, Word
from ..models.storage import save_session
from .checkpoints import file_digest
from .global_diarization import audit_events, clean_identity_windows, identity_review, reconstruct_events, annotate_identity_uncertainty
from .transcribe import TSegment
from .waveform import WaveformPeaks


def write_json(path, value):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_clean_embeddings(source, turns, output, *, device="cuda", progress=print):
    """Use the installed diarizer's embedding model; do not rerun clustering.

    Every >=3 second sample excludes all detected overlap. Checkpoints include
    source bytes, exact windows, and model bytes. Samples never trigger a merge.
    """
    from fractions import Fraction
    from scipy.signal import resample_poly
    from .diarize import _model_path
    import torch
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    # Prevent implicit downloads. Missing local models remain an explicit failure.
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model_path = Path(_model_path("pyannote/speaker-diarization-community-1", None)) / "embedding" / "pytorch_model.bin"
        model_hash = file_digest(model_path)
        source_hash = file_digest(source)
        windows = clean_identity_windows(turns)
        signature = {"algorithm": "clean-identity-v1", "source_sha256": source_hash, "model_sha256": model_hash,
                     "windows": windows, "device": device}
        target = Path(output)
        completed = []
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved.get("signature") == signature:
                completed = saved.get("samples", [])
        if len(completed) == len(windows):
            return completed, {"resumed": True, "samples": len(completed)}
        embedding = PretrainedSpeakerEmbedding(str(model_path), device=torch.device(device))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        info = sf.info(str(source))
        ratio = Fraction(embedding.sample_rate, info.samplerate)
        with sf.SoundFile(str(source)) as reader:
            for index, window in enumerate(windows[len(completed):], len(completed)):
                reader.seek(round(window["start_s"] * info.samplerate))
                audio = reader.read(round((window["end_s"] - window["start_s"]) * info.samplerate), dtype="float32", always_2d=True)
                mono = resample_poly(audio.mean(axis=1), ratio.numerator, ratio.denominator).astype(np.float32)
                vector = np.asarray(embedding(torch.from_numpy(mono)[None, None, :]))[0]
                if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    completed.append({**window, "embedding": [], "failure": "invalid_embedding"})
                else:
                    completed.append({**window, "embedding": (vector / np.linalg.norm(vector)).tolist(),
                                      "overlap": False, "source_sha256": source_hash, "model_sha256": model_hash})
                if (index + 1) % 10 == 0:
                    write_json(target, {"signature": signature, "samples": completed})
                    progress(f"Clean identity samples {index + 1}/{len(windows)}")
        metrics = {"elapsed_s": time.monotonic() - started,
                   "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
                   "samples": len(completed), "device": device}
        write_json(target, {"signature": signature, "samples": completed, "metrics": metrics})
        del embedding
        if device == "cuda":
            torch.cuda.empty_cache()
        return completed, metrics
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline


def run_global_audit(source, checkpoint_dir, output_dir, *, embeddings=False, device="cuda", progress=print):
    source, checkpoint_dir, output = Path(source), Path(checkpoint_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "session.json").exists():
        from ..models.storage import load_session
        previous = load_session(output / "session.json")
        if previous.edit_history or any(u.user_corrections for u in previous.utterances):
            raise ValueError("Output contains user corrections. Choose another output directory; corrections were preserved.")
    started = time.monotonic()
    digest = file_digest(source)
    baseline_report_path = checkpoint_dir.parent.parent / "report.json"
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline.get("source_sha256") != digest or not baseline.get("processed_full_file"):
        raise ValueError("Checkpoint source hash or full-recording provenance does not match")
    payloads = {}
    for stage in ("asr", "alignment", "diarization"):
        envelope = json.loads((checkpoint_dir / f"{stage}.json").read_text(encoding="utf-8"))
        if envelope.get("key") != checkpoint_dir.name or envelope.get("phase") != stage:
            raise ValueError("Checkpoint envelope mismatch")
        payloads[stage] = envelope["payload"]
    raw = [TSegment(**s) for s in payloads["asr"]["segments"]]
    aligned = [TSegment(**s) for s in payloads["alignment"]["segments"]]
    diar = payloads["diarization"]["result"]
    progress("Reconstructing all aligned words against ordinary speaker activity")
    events = reconstruct_events(aligned, diar["turns"], diar.get("exclusive", []),
                                 source_audio_ref=str(source), asr_model=baseline.get("configuration", {}).get("model_size"))
    raw_words = [Word(*w) for s in raw for w in s.words]
    aligned_words = [Word(*w) for s in aligned for w in s.words]
    report = audit_events(events, diar["turns"], raw_words=raw_words, aligned_words=aligned_words)
    if report.get("reconstruction_word_loss") or report.get("reconstruction_word_additions"):
        raise RuntimeError("Reconstruction did not preserve all words exactly")
    progress("Scanning every native audio frame and building waveform cache")
    peaks = WaveformPeaks(source, output / "waveform-cache")
    activity = peaks.activity_candidates()
    write_json(output / "energy-activity.json", {"kind": "energy_candidates_not_verified_speech", "intervals": activity})
    report["source"] = {"path": str(source), "sha256": digest, "duration_s": peaks.duration_s,
                        "sample_rate": sf.info(str(source)).samplerate, "channels": sf.info(str(source)).channels}
    report["checkpoint_provenance"] = {"directory": str(checkpoint_dir), "stages": list(payloads),
        "source_hash_verified": True, "asr_runtime_s": baseline.get("phase_runtime_s", {}).get("asr_s"),
        "diarization_runtime_s": baseline.get("phase_runtime_s", {}).get("diarization_s"),
        "diarization_scope": "entire_recording_one_global_clustering", "application_chunk_id_resets": False}
    session = Session(name=f"{source.stem} — timeline globale", audio=AudioSources(mic=str(source)),
        audio_profile="stereo_ambient", utterances=events, media_duration_s=peaks.duration_s,
        model=baseline.get("configuration", {}).get("model_size"), language=payloads["asr"].get("language"),
        diarization=diar, provenance={"source_sha256": digest, "checkpoint_directory": str(checkpoint_dir)})
    from copy import deepcopy
    session.raw_utterances = deepcopy(events)
    session.sync_segments()
    save_session(session, output / "session.json")
    if embeddings:
        progress("Extracting long clean identity evidence; no identity merge is automatic")
        samples, metrics = extract_clean_embeddings(source, diar["turns"], output / "identity-embeddings.json", device=device, progress=progress)
        review = identity_review(samples)
        annotate_identity_uncertainty(session.utterances, review)
        report["identity_processing"] = metrics
        report["identity_review"] = review
        session.diarization["global_identity_review"] = review
        write_json(output / "identity-review.json", review)
    else:
        report["identity_review"] = {"status": "not_run", "reason": "clean sample embeddings not requested"}
    # Exact user clue is evidence of a defect, not a fully timed gold benchmark.
    number_words = [{"start_s": w.start, "end_s": w.end, "text": w.text.strip(), "event_id": u.event_id,
                     "predicted_speaker": u.speaker, "uncertain": u.uncertain}
                    for u in events for w in u.words if u.start_s < 180 and w.text.strip().strip(".,?!…") == "27"]
    report["user_verified_regressions"] = [{"claim": "27 belongs to a different voice from the neighboring utterance",
        "scope": "user_reported_speaker_distinction", "timing_candidates": number_words,
        "speaker_identity": None, "quantitative_reference": False}]
    for word in number_words:
        session.speaker_evidence.append({"event_id": word["event_id"], "start_s": word["start_s"], "end_s": word["end_s"],
            "relation": "different_voice_from_neighbor", "verified_by": "user", "scope": "word",
            "speaker_id": None, "propagation_allowed": False, "text": "27"})
    regions = []
    for span in report["overlap_regions"]:
        regions.append({**span, "kind": "overlap_incomplete_transcription_candidate"})
    for event in events:
        if event.processing_state == "missing_transcription" and not event.is_overlap:
            regions.append({"start_s": event.start_s, "end_s": event.end_s, "event_id": event.event_id,
                            "kind": "activity_without_speaker_specific_text", "requires_review": True})
    regions += report.get("identity_review", {}).get("suspected_regions", [])
    report["suspected_error_timeline"] = sorted(regions, key=lambda r: (r["start_s"], r["end_s"]))
    report["elapsed_s"] = time.monotonic() - started
    report["source_unchanged"] = file_digest(source) == digest
    report["counts"] = dict(Counter(r["kind"] for r in regions))
    session.provenance["global_audit"] = {"report_path": str(output / "report.json"), "source_sha256": digest,
        "word_loss": report.get("reconstruction_word_loss"), "quality_metrics": None}
    save_session(session, output / "session.json")
    write_json(output / "report.json", report)
    # Stratify review candidates through the full file; no machine text becomes gold.
    selected = []
    for quarter in range(4):
        lo, hi = quarter * peaks.duration_s / 4, (quarter + 1) * peaks.duration_s / 4
        for rows, label in ((report["overlap_regions"], "overlap"), (clean_identity_windows(diar["turns"]), "clean_identity")):
            matching = [r for r in rows if lo <= r["start_s"] < hi]
            for r in matching[::max(1, len(matching) // 3)][:3]:
                selected.append({"start_s": max(0, r["start_s"] - 2), "end_s": min(peaks.duration_s, r["end_s"] + 2),
                                 "stratum": label, "events": [], "status": "unverified"})
    selected += [{"start_s": 94, "end_s": 108, "stratum": "screenshot_regression", "events": [], "status": "unverified"},
                 {"start_s": 119, "end_s": 127, "stratum": "user_27_distinct_voice", "events": [], "status": "partial_user_claim"}]
    template = output / "reference-review-template.json"
    if not template.exists():
        write_json(template, {"source_sha256": digest, "verification_level": "unverified_candidate", "reviewed_by": "", "annotation_complete": False, "regions": selected})
    progress(f"Global audit complete: {len(events)} events; {len(regions)} review regions; no accuracy claims")
    return report
