"""Local, bounded recovery proposals. Existing events are never overwritten."""
from __future__ import annotations

import math
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from ..models.audio_artifacts import AudioView, SpeechRegion
from ..models.session import Word
from ..models.transcript_artifacts import Utterance
from .audio_utils import load_mono_16k
from .checkpoints import file_digest


def recover_region(source, start_s, end_s, output_dir, *, existing_events=(), separate=False,
                   potential_speakers=None, model_size="large-v3", language=None, device="auto",
                   context_s=2., transcriber_factory=None, separator=None, enhance=False,
                   compare_baseline=False, source_sha256=None, candidate_speakers=(),
                   initial_prompt=None, on_operation=None, cancel_event=None, deadline=None):
    """Extract context in native channels, then transcribe each proposed source.

    Candidate speaker IDs remain null: separator channel numbers are not speaker
    identities. Duplicate candidates are visible with their evidence for review.
    """
    started = time.monotonic()
    if not all(math.isfinite(v) for v in (start_s, end_s, context_s)) or not 0 <= start_s < end_s or end_s - start_s > 120 or not 0 <= context_s <= 10:
        raise ValueError("Seleziona una regione di durata massima 120 s.")
    source = Path(source)
    output = Path(output_dir) / f"region-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    info = sf.info(str(source))
    if start_s >= info.duration:
        raise ValueError("La regione è oltre la fine dell'audio.")
    left = round(max(0, start_s - context_s) * info.samplerate) / info.samplerate
    right = min(info.duration, end_s + context_s)
    with sf.SoundFile(str(source)) as reader:
        reader.seek(round(left * info.samplerate))
        native = reader.read(round((right - left) * info.samplerate), dtype="float32", always_2d=True)
    original = output / "original-context.wav"
    sf.write(str(original), native, info.samplerate, subtype="FLOAT")
    audio = load_mono_16k(original)
    from .enhancement import analyze_region
    analysis = analyze_region(audio).to_dict()
    source_hash = source_sha256 or file_digest(source)
    views, separations, errors = [], [], []
    operation = on_operation or (lambda *_args: None)

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Recupero annullato; prima passata conservata.")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Limite di tempo del recupero raggiunto; baseline conservata.")

    def add_view(signal, strategy, index=0, delay=0., config=None, separation=None):
        path = output / f"{strategy}-source-{index}.wav"
        sf.write(str(path), signal, 16000, subtype="FLOAT")
        views.append({"path": str(path), "strategy": strategy, "source_index": index,
                      "offset_s": left - delay, "enhancement": config or {},
                      "sha256": file_digest(path),
                      "analysis": analyze_region(signal).to_dict(),
                      "separation": separation or {"status": "not_requested"}})

    def split(signal, strategy, config=None):
        check_cancel()
        operation("separation", strategy)
        from .separation import separate_region
        separate_fn = separator or separate_region
        result = separate_fn(signal, AudioView(sample_rate=16000, duration_s=len(signal) / 16000),
                             SpeechRegion(start_s=0, end_s=len(signal) / 16000), num_speakers=potential_speakers)
        separation = result.to_dict()
        separations.append(separation)
        if potential_speakers and len(result.tracks) < potential_speakers:
            separation["incomplete_source_count"] = {"potential": potential_speakers, "returned": len(result.tracks)}
        if result.status == "success":
            for index, track in enumerate(result.tracks):
                add_view(track, strategy, index, result.delay_compensation_s, config, separation)
        return result

    if compare_baseline or not separate:
        add_view(audio, "original")
    if compare_baseline:
        from .audio_utils import normalize
        add_view(normalize(audio), "normalized")
    enhanced, enhancement = None, {}
    if enhance:
        check_cancel()
        operation("enhance", "Gain statico limitato; nessun filtro monovoce")
        from .enhancement import bounded_gain
        enhanced, enhancement = bounded_gain(audio)
        if enhancement["gain_db"] > 0:
            add_view(enhanced, "bounded_gain", config=enhancement)
    if separate:
        result = split(audio, "acoustic_separation")
        if not compare_baseline and result.status != "success":
            add_view(audio, "original")
        if enhanced is not None and enhancement["gain_db"] > 0:
            split(enhanced, "enhanced_separation", enhancement)

    separation = separations[0] if separations else {"status": "not_requested"}
    from .pipeline import resolve_device
    from .transcribe import Transcriber, _ensure_model_path
    effective_device, compute_type = resolve_device(device)
    factory = transcriber_factory or Transcriber
    # Resolve an installed snapshot first: recovery never downloads weights.
    model_ref = _ensure_model_path(model_size, None, local_only=True) if transcriber_factory is None else model_size
    revision = Path(model_ref).name if Path(model_ref).is_dir() else "unresolved"
    check_cancel()
    operation("asr", "Caricamento ASR locale e confronto indipendente delle viste")
    try:
        transcriber = factory(model_size=model_ref, device=effective_device, compute_type=compute_type)
    except RuntimeError as exc:
        if effective_device != "cuda" or "out of memory" not in str(exc).lower():
            raise
        import gc
        gc.collect()
        effective_device, compute_type = "cpu", "int8"
        transcriber = factory(model_size=model_ref, device=effective_device, compute_type=compute_type)
    candidates, predictions = [], []
    normalized = _normalized_text
    try:
        for view in views:
            check_cancel()
            operation("asr", view["strategy"])
            path, index = Path(view["path"]), view["source_index"]
            offset = view["offset_s"]
            kwargs = {"language": language, "quality_retries": 0}
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            try:
                segments, detected = transcriber.transcribe(path, **kwargs)
            except Exception as exc:
                errors.append({"strategy": view["strategy"], "source_index": index, "error": str(exc)})
                continue
            for segment in segments:
                absolute_start, absolute_end = offset + segment.start, offset + segment.end
                if absolute_end <= start_s or absolute_start >= end_s:
                    continue
                # Keep boundary-crossing text and its full context timestamps.
                words = [Word(max(0, offset + a), min(info.duration, offset + b), text, prob)
                         for a, b, text, prob in segment.words]
                if not (0 <= absolute_start < absolute_end <= info.duration + .05) or any(
                        not math.isfinite(w.start + w.end) or w.end < w.start for w in words):
                    errors.append({"strategy": view["strategy"], "error": "invalid_timestamps"})
                    continue
                duplicate_ids = [u.event_id for u in existing_events if u.end_s > absolute_start and u.start_s < absolute_end
                                 and normalized(u.text) == normalized(segment.text)]
                duplicate_words = []
                for word in words:
                    for existing in existing_events:
                        matches = [w for w in existing.words if normalized(w.text) == normalized(word.text)
                                   and min(w.end, word.end) > max(w.start, word.start)
                                   and abs((w.start + w.end - word.start - word.end) / 2) <= .15]
                        if matches:
                            duplicate_words.append({"text": word.text, "start_s": word.start, "end_s": word.end,
                                                    "event_id": existing.event_id})
                validation = "requires_human_review"
                if (segment.avg_logprob < -1.0 or segment.no_speech_prob > .6
                        or segment.compression_ratio > 2.6):
                    validation = "rejected_asr_quality"
                predictions.append({**view, "text": segment.text, "start_s": absolute_start, "end_s": absolute_end,
                                    "words": [vars(w) for w in words], "validation": validation,
                                    "avg_logprob": segment.avg_logprob, "no_speech_prob": segment.no_speech_prob})
                if compare_baseline:
                    # Context belongs to the prediction, not to a new timeline event.
                    words = [w for w in words if w.end > start_s and w.start < end_s]
                    if validation.startswith("rejected") or duplicate_ids:
                        continue
                    seen = {(d["start_s"], d["end_s"], d["text"]) for d in duplicate_words}
                    novel = [w for w in words if (w.start, w.end, w.text) not in seen]
                    if words and not novel:
                        continue
                    if novel:
                        words = novel
                        absolute_start, absolute_end = words[0].start, words[-1].end
                    elif not start_s <= (absolute_start + absolute_end) / 2 <= end_s:
                        continue
                text = " ".join(w.text.strip() for w in words).strip() if compare_baseline and words else segment.text.strip()
                if not text:
                    continue
                event = Utterance(start_s=max(0, absolute_start), end_s=min(info.duration, absolute_end),
                    speaker=None, text=text, words=words, source_audio_ref=str(source), stream_id=index,
                    processing_state="candidate", uncertain=True,
                    asr_provenance={"model": model_size, "language": detected, "source_sha256": source_hash,
                        "model_revision": revision, "strategy": view["strategy"], "enhancement": view["enhancement"],
                        "candidate_audio": str(path), "candidate_audio_offset_s": offset,
                        "original_context_audio": str(original), "selected_interval": [start_s, end_s],
                        "context_interval": [left, right], "source_index": index, "separation": view["separation"],
                        "sample_rate": 16000, "native_sample_rate": info.samplerate, "native_channels": info.channels,
                        "comparison_views": views, "validation": validation,
                        "device": effective_device, "compute_type": compute_type,
                        "processing_time_s": time.monotonic() - started},
                    candidate_speakers=[{"speaker_id": speaker, "evidence": "activity_only_not_identity"}
                                        for speaker in candidate_speakers],
                    diarization_provenance={"identity": "unresolved", "source_index_is_not_speaker_id": True})
                candidates.append({"candidate_id": uuid.uuid4().hex, "status": "proposed", "event": event.to_dict(),
                    "duplicate_of_event_ids": duplicate_ids,
                    "potential_duplicate_words": duplicate_words,
                    "review_note": "Possibile duplicato: confrontare prima di accettare" if duplicate_ids else "Candidato non verificato"})
    finally:
        transcriber.close()
    result = {"candidates": candidates, "source_sha256": source_hash, "selection": [start_s, end_s],
              "separation": separation, "elapsed_s": time.monotonic() - started,
              "separations": separations, "predictions": predictions, "errors": errors,
              "analysis": analysis, "enhancement": enhancement, "views": views,
              "context_audio": str(original), "context_sha256": file_digest(original),
              "original_audio": str(source), "output_dir": str(output)}
    from .global_workflow import write_json
    write_json(output / "proposals.json", result)
    return result


def _normalized_text(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def plan_recovery(source, events, turns=(), *, max_regions=8, max_audio_s=120.,
                  max_region_s=18., controls=2, cancel_event=None, stratified=False):
    """Bounded, distributed hypotheses, including clean controls; never an accuracy verdict."""
    from .coverage import detect_speech_activity, find_uncovered_regions
    from .enhancement import analyze_region, classify_region, frame_levels_db, speech_level_db
    from scipy.signal import resample_poly
    from math import gcd

    if max_regions < 0 or controls < 0 or not math.isfinite(max_audio_s) or max_audio_s < 0 or not 0 < max_region_s <= 120:
        raise ValueError("Limiti della coda di recupero non validi")
    info = sf.info(str(source))
    hints = []
    ordinary = [{"start": float(t["start"]), "end": float(t["end"]), "speaker": t.get("speaker")}
                for t in turns if t.get("speaker") and t["end"] > t["start"]]

    def add(a, b, labels, speakers=(), priority=0, analysis=None, control=False):
        a, b = max(0., a), min(info.duration, b)
        while b - a >= .14:
            right = min(b, a + max_region_s)
            hints.append({"start_s": a, "end_s": right, "labels": sorted(set(labels)),
                          "speakers": sorted(set(speakers)), "priority": priority,
                          "analysis": analysis or {}, "control": control})
            a = right

    changes = []
    for t in ordinary:
        changes.extend([(t["start"], 1, t["speaker"]), (t["end"], -1, t["speaker"])])
    active, previous = Counter(), 0.
    for point, delta, speaker in sorted(changes):
        speakers = sorted(s for s, count in active.items() if count > 0)
        if len(speakers) >= 2 and point > previous:
            recognized = {e.speaker_id for e in events if e.text.strip() and e.end_s > previous and e.start_s < point}
            labels = ["overlap_3plus" if len(speakers) > 2 else "overlap_2"]
            if len(recognized.intersection(speakers)) < len(speakers):
                labels.append("possible_recognition_omission")
            add(previous, point, labels, speakers, 100 + 15 * (len(labels) > 1) + 20 * (len(speakers) > 2))
        active[speaker] += delta
        previous = point
    for event in events:
        speakers = [event.speaker_id] if event.speaker_id else []
        if not event.text.strip():
            add(event.start_s, event.end_s, ["recognition_omission"], speakers, 90)
        elif event.uncertain or len(event.candidate_speakers) > 1:
            add(event.start_s, event.end_s, ["speaker_identity_uncertainty"], speakers, 45)
        if any(w.probability is not None and w.probability < .55 for w in event.words):
            add(event.start_s, event.end_s, ["low_asr_confidence"], speakers, 60)
    ordered = sorted(ordinary, key=lambda t: (t["start"], t["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if left["speaker"] != right["speaker"] and -.2 <= right["start"] - left["end"] <= .35:
            if min(left["end"] - left["start"], right["end"] - right["start"]) <= 1.5:
                add(max(left["start"], left["end"] - 1.), min(right["end"], right["start"] + 1.5),
                    ["rapid_interruption", "possible_boundary_omission"], [left["speaker"], right["speaker"]], 70)
    textual = [e for e in events if e.text.strip()]
    words = [w for e in textual for w in e.words] or [SimpleNamespace(start=e.start_s, end=e.end_s) for e in textual]
    levels, analyzed = [], []
    with sf.SoundFile(str(source)) as reader:
        for block in reader.blocks(blocksize=info.samplerate * 8, dtype="float32", always_2d=True):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Recupero annullato")
            end = reader.tell() / info.samplerate
            start = end - len(block) / info.samplerate
            factor = gcd(info.samplerate, 16000)
            mono = resample_poly(block.mean(axis=1), 16000 // factor, info.samplerate // factor).astype(np.float32)
            analysis = analyze_region(mono)
            levels.extend(frame_levels_db(mono).tolist())
            speech = detect_speech_activity(mono)
            for region in speech:
                region.start_s += start
                region.end_s += start
            local_words = [w for w in words if w.end > start and w.start < end]
            for region in find_uncovered_regions(speech, local_words):
                add(region.start_s, region.end_s, ["possible_recognition_omission"], priority=65, analysis=analysis.to_dict())
            analyzed.append((start, end, analysis, bool(speech)))
    typical, _ = speech_level_db(np.asarray(levels))
    for start, end, analysis, speech in analyzed:
        labels = classify_region(analysis, typical)
        if labels and speech:
            add(start, end, labels, priority=55 if "weak_speech" in labels else 30, analysis=analysis.to_dict())
    detected = len(hints)
    merged = []
    for hint in sorted(hints, key=lambda r: (r["start_s"], r["end_s"])):
        if merged and hint["speakers"] == merged[-1]["speakers"] and hint["start_s"] <= merged[-1]["end_s"] + .3 and max(hint["end_s"], merged[-1]["end_s"]) - merged[-1]["start_s"] <= max_region_s:
            previous = merged[-1]
            previous["end_s"] = max(previous["end_s"], hint["end_s"])
            previous["labels"] = sorted(set(previous["labels"] + hint["labels"]))
            previous["priority"] = max(previous["priority"], hint["priority"])
        else:
            merged.append(hint)
    ranked = sorted(merged, key=lambda r: (-r["priority"], r["start_s"]))
    # Reserve clean controls, spread through the recording, within the same budget.
    adverse = {"overlap_2", "overlap_3plus", "rapid_interruption", "weak_speech", "low_asr_confidence", "speaker_identity_uncertainty"}
    clean = [e for e in textual if not e.uncertain and e.end_s - e.start_s >= 3
             and not any(adverse.intersection(r["labels"]) and r["start_s"] < e.end_s and r["end_s"] > e.start_s for r in merged)]
    selected_controls = clean[::max(1, len(clean) // max(1, controls))][:controls] if controls else []
    queue, used = [], 0.
    for event in selected_controls:
        length = min(6., event.end_s - event.start_s)
        if len(queue) < max_regions and used + length + 4 <= max_audio_s:
            queue.append({"start_s": event.start_s, "end_s": event.start_s + length,
                          "labels": ["clean_control"], "speakers": [event.speaker_id] if event.speaker_id else [],
                          "priority": 0, "analysis": {}, "control": True})
            used += length + 4
    # One high-priority region per quarter before filling remaining slots.
    distributed, slots = [], set()
    if stratified:
        for label in ("overlap_3plus", "overlap_2", "weak_speech", "reverberation", "noise",
                      "rapid_interruption", "low_asr_confidence", "speaker_identity_uncertainty",
                      "recognition_omission", "possible_recognition_omission"):
            match = next((r for r in ranked if label in r["labels"] and r not in distributed), None)
            if match is not None:
                distributed.append(match)
    quarters = []
    for region in ranked:
        slot = min(3, int(region["start_s"] / max(1., info.duration / 4)))
        if slot not in slots:
            quarters.append(region)
            slots.add(slot)
    distributed = quarters + [r for r in distributed if r not in quarters]
    distributed += [r for r in ranked if r not in distributed]
    for region in distributed:
        cost = region["end_s"] - region["start_s"] + 4
        if len(queue) >= max_regions:
            break
        if used + cost > max_audio_s or any(r["start_s"] < region["end_s"] and r["end_s"] > region["start_s"] for r in queue):
            continue
        queue.append(region)
        used += cost
    return {"regions": sorted(queue, key=lambda r: (-r["priority"], r["start_s"])),
            "detected": detected, "queued": len(queue), "audio_budget_s": used,
            "controls": sum(r["control"] for r in queue), "duration_s": info.duration,
            "detected_by_label": dict(Counter(label for r in hints for label in r["labels"])),
            "scan_windows": [{"start_s": a, "end_s": b, "analysis": x.to_dict()} for a, b, x, _ in analyzed]}


def recover_session_regions(source, session, output_dir, *, mode="adaptive", device="auto",
                            model_size=None, initial_prompt=None, stages=None, max_regions=8,
                            max_audio_s=120., max_elapsed_s=180., cancel_event=None,
                            transcriber_factory=None, separator=None, stratified=False):
    """Automatic second pass. Only proposals and diagnostics mutate the session.

    Each region closes its ASR model before the next separation, keeping heavy
    engines sequential. Atomic checkpoints bind audio, baseline, settings and
    installed model fingerprints; interrupted work can reuse completed regions.
    The runtime limit is checked between regions, not inside native inference.
    """
    from .global_workflow import write_json
    from .progress import RECOVERY_STAGES
    from .separation_backends import OnnxSeparationBackend, separation_dir

    if mode not in ("adaptive", "experimental"):
        raise ValueError("Modalità di recupero non valida")
    if max_regions < 0 or not math.isfinite(max_audio_s) or not math.isfinite(max_elapsed_s) or max_audio_s < 0 or max_elapsed_s < 0:
        raise ValueError("I limiti di recupero non possono essere negativi")
    started = time.monotonic()
    source = Path(source)
    digest = file_digest(source)
    model_size = model_size or session.model or "large-v3"
    baseline = [e.to_dict() for e in session.utterances]
    turns = list(session.diarization.get("turns", []))
    for metadata in session.diarization.get("sources", {}).values():
        turns.extend(metadata.get("turns", []))
    manifest = separation_dir() / "manifest.json"
    signature = {"algorithm": "selective-recovery-v1", "source_sha256": digest,
                 "baseline_sha256": hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
                 "diarization_sha256": hashlib.sha256(json.dumps(turns, sort_keys=True).encode()).hexdigest(),
                 "model": model_size, "language": session.language, "mode": mode, "device": device,
                 "initial_prompt": initial_prompt, "max_regions": max_regions, "max_audio_s": max_audio_s,
                 "stratified": stratified,
                 "separator_manifest_sha256": file_digest(manifest) if manifest.is_file() else None}
    from importlib.metadata import version
    signature["runtime_versions"] = {name: version(name) for name in ("faster-whisper", "ctranslate2", "numpy", "soundfile")}
    signature["implementation_sha256"] = {name: file_digest(Path(__file__).with_name(name)) for name in
                                         ("region_recovery.py", "enhancement.py", "separation.py", "separation_backends.py", "transcribe.py", "audio_utils.py")}
    if transcriber_factory is None:
        from .transcribe import _ensure_model_path
        local_model = Path(_ensure_model_path(model_size, None, local_only=True))
        signature["asr_revision"] = local_model.name
        signature["asr_config_sha256"] = file_digest(local_model / "config.json")
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(output_dir) / f"advanced-recovery-{key}"
    output.mkdir(parents=True, exist_ok=True)
    counts = dict(detected=0, queued=0, processed=0, recovered=0, unresolved=0,
                  candidates=0, controls=0, controls_processed=0, rejected_predictions=0)
    report = {"signature": signature, "counts": counts, "regions": [], "output_dir": str(output)}
    running, operations = set(), Counter()

    def start(stage, detail=""):
        if stages and stage not in running:
            stages.start(stage, detail)
            running.add(stage)

    def operation(kind, detail):
        if kind == "separation" and not available:
            return
        stage = {"enhance": "recovery_enhance", "separation": "recovery_separation", "asr": "recovery_asr"}[kind]
        start(stage, detail)
        operations[kind] += 1
        if stages:
            stages.update(stage, detail=detail, fraction=None)

    start("recovery_detect")
    start("recovery_weak", "Indicatori acustici; non misure di accuratezza")
    plan = plan_recovery(source, session.utterances, turns, max_regions=max_regions, max_audio_s=max_audio_s,
                         cancel_event=cancel_event, stratified=stratified)
    counts.update({k: plan[k] for k in ("detected", "queued", "controls")})
    report["plan"] = plan
    write_json(output / "plan.json", {"signature": signature, **plan})
    if stages:
        stages.complete("recovery_detect", detail=f"{counts['detected']} ipotesi; {counts['queued']} regioni in coda", extra=counts)
        stages.complete("recovery_weak", detail=f"{len(plan['scan_windows'])} finestre analizzate; indicatori non verificati")
    backend = OnnxSeparationBackend()
    available = separator is not None or backend.is_available()
    report["separation_availability"] = {"available": available, "reason": backend.unavailable_reason}
    for index, region in enumerate(plan["regions"]):
        if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() - started >= max_elapsed_s:
            report["stopped_reason"] = "cancelled" if cancel_event is not None and cancel_event.is_set() else "runtime_budget"
            break
        checkpoint = output / f"region-{index:03d}.json"
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else None
        except (OSError, ValueError):
            saved = None
        if (saved and saved.get("signature") == signature
                and saved.get("selection") == [region["start_s"], region["end_s"]]
                and Path(saved.get("context_audio", "")).is_file()
                and file_digest(saved["context_audio"]) == saved.get("context_sha256")
                and all(Path(v["path"]).is_file() and file_digest(v["path"]) == v.get("sha256") for v in saved.get("views", []))):
            result = saved
            result["resumed"] = True
        else:
            try:
                overlap = any(label in region["labels"] for label in ("overlap_2", "overlap_3plus"))
                result = recover_region(source, region["start_s"], region["end_s"], output,
                    existing_events=session.utterances, separate=overlap,
                    potential_speakers=max(2, len(region["speakers"])) if overlap else None,
                    model_size=model_size, language=session.language, device=device,
                    enhance="weak_speech" in region["labels"] or overlap,
                    compare_baseline=True, source_sha256=digest, candidate_speakers=region["speakers"],
                    initial_prompt=initial_prompt, on_operation=operation, cancel_event=cancel_event,
                    transcriber_factory=transcriber_factory, separator=separator,
                    deadline=started + max_elapsed_s)
                result["signature"] = signature
                # Recognized duplicates on independent tracks indicate possible leakage.
                for candidate in result["candidates"]:
                    event = candidate["event"]
                    provenance = event["asr_provenance"]
                    siblings = [c for c in result["candidates"] if c is not candidate
                                and c["event"]["asr_provenance"]["strategy"] == provenance["strategy"]
                                and c["event"]["asr_provenance"]["source_index"] != provenance["source_index"]
                                and _normalized_text(c["event"]["text"]) == _normalized_text(event["text"])
                                and c["event"]["start_s"] < event["end_s"] and c["event"]["end_s"] > event["start_s"]]
                    if siblings and len(_normalized_text(event["text"]).split()) >= 3:
                        candidate["status"] = "rejected_validation"
                        candidate["validation"] = "possible_intertrack_leakage"
                        candidate["duplicate_of_candidate_ids"] = [c["candidate_id"] for c in siblings]
                    candidate["control"] = region["control"]
                    candidate["problem_labels"] = region["labels"]
                write_json(checkpoint, result)
            except InterruptedError:
                report["stopped_reason"] = "cancelled"
                break
            except Exception as exc:
                report["regions"].append({"selection": [region["start_s"], region["end_s"]], "error": str(exc)})
                counts["processed"] += 1
                counts["unresolved"] += not region["control"]
                continue
        known = {c["candidate_id"] for c in session.recovery_candidates}
        session.recovery_candidates.extend(c for c in result["candidates"] if c["candidate_id"] not in known)
        counts["processed"] += 1
        counts["controls_processed"] += region["control"]
        counts["rejected_predictions"] += sum(p["validation"].startswith("rejected") for p in result["predictions"])
        proposed = [c for c in result["candidates"] if c["status"] == "proposed"]
        counts["candidates"] += len(proposed)
        if proposed and not region["control"]:
            counts["recovered"] += 1
        if not region["control"] and (not proposed or any(
                s.get("status") != "success" or s.get("incomplete_source_count") for s in result["separations"])):
            counts["unresolved"] += 1
        report["regions"].append({"selection": result["selection"], "elapsed_s": result["elapsed_s"],
                                  "separations": result["separations"], "errors": result["errors"],
                                  "analysis": result["analysis"], "enhancement": result["enhancement"],
                                  "control": region["control"], "candidates": len(proposed),
                                  "resumed": result.get("resumed", False), "checkpoint": str(checkpoint)})
    counts["unresolved"] += sum(not r["control"] for r in plan["regions"][counts["processed"]:])
    if stages:
        stages.skip("recovery_identity", "Non disponibile: identità acustica delle tracce non verificata; confermare manualmente i suggerimenti di attività")
    start("recovery_validate")
    if stages:
        if not counts["candidates"] and counts["rejected_predictions"]:
            stages.skip("recovery_validate", "Respinto dalla validazione: nessuna proposta supera i controlli ASR", extra=counts)
        else:
            stages.complete("recovery_validate", detail=f"{counts['candidates']} proposte da ascoltare; {counts['unresolved']} regioni irrisolte", extra=counts)
    for stage, _ in RECOVERY_STAGES:
        if stage in ("recovery_detect", "recovery_weak", "recovery_identity", "recovery_validate"):
            continue
        if not stages:
            break
        if stage in running:
            separations = [s for r in report["regions"] for s in r.get("separations", [])]
            if stage == "recovery_separation" and separations and not any(s.get("status") == "success" for s in separations):
                stages.skip(stage, "Respinto dalla validazione: nessuna separazione utilizzabile", extra=counts)
            elif stage == "recovery_asr" and report["regions"] and all("error" in r for r in report["regions"]):
                stages.fail(stage, "Trascrizione candidati fallita; prima passata conservata")
            else:
                stages.complete(stage, detail=f"{counts['processed']} regioni elaborate; risultati non verificati",
                                warning="Alcune regioni non risolte" if counts["unresolved"] else None, extra=counts)
        else:
            reason = "Non necessario: nessuna operazione richiesta"
            if stage == "recovery_tse":
                reason = ("Non disponibile: nessun estrattore target locale convalidato" if mode == "experimental"
                          else "Disattivato: estrazione target solo sperimentale")
            elif stage == "recovery_separation" and not available:
                reason = f"Non disponibile: {backend.unavailable_reason}"
            elif any(r.get("resumed") for r in report["regions"]):
                reason = "Non necessario: risultati ripresi da checkpoint verificati"
            stages.skip(stage, reason)
    report["elapsed_s"] = time.monotonic() - started
    if file_digest(source) != digest:
        raise RuntimeError("Audio sorgente modificato durante il recupero")
    write_json(output / "report.json", report)
    session.provenance["advanced_recovery"] = report
    return report
