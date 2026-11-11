"""Recording-wide reconstruction and conservative identity review.

No guessed second transcript and no local chunk label becomes a global identity.
All words survive reconstruction. Ambiguous mixed words remain unassigned.
"""
from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np

from ..models.session import Word
from ..models.transcript_artifacts import Utterance, recompute_overlap_relationships
from .merge import flatten_words, materialize_overlap_utterances


def activity_spans(turns):
    boundaries = defaultdict(list)
    for t in turns:
        start, end, speaker = (t["start"], t["end"], t["speaker"]) if isinstance(t, dict) else (t.start, t.end, t.speaker)
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid diarization interval")
        boundaries[start].append((speaker, 1))
        boundaries[end].append((speaker, -1))
    active, result, previous = Counter(), [], None
    for time in sorted(boundaries):
        speakers = sorted(s for s, count in active.items() if count > 0)
        if previous is not None and time > previous and speakers:
            result.append({"start_s": previous, "end_s": time, "speakers": speakers,
                           "potential_speaker_count": len(speakers), "overlap": len(speakers) > 1})
        for speaker, delta in boundaries[time]:
            active[speaker] += delta
        previous = time
    return result


def reconstruct_events(tsegments, ordinary_turns, exclusive_turns=(), *, source_audio_ref=None, asr_model=None):
    """Keep every ASR word; annotate uncertain attribution without forced smoothing."""
    spans = activity_spans(ordinary_turns)
    units = flatten_words(tsegments)
    events = []
    current = None
    current_signature = None
    cursor = 0
    for index, unit in enumerate(units):
        while cursor < len(spans) and spans[cursor]["end_s"] <= unit.start:
            cursor += 1
        candidates, overlapping = defaultdict(float), False
        for span in spans[cursor:]:
            if span["start_s"] >= max(unit.end, unit.start + .001):
                break
            duration = min(span["end_s"], max(unit.end, unit.start + .001)) - max(span["start_s"], unit.start)
            if duration > 0:
                for speaker in span["speakers"]:
                    candidates[speaker] += duration
                overlapping |= span["overlap"]
        speaker = next(iter(candidates)) if len(candidates) == 1 and not overlapping else None
        uncertain = speaker is None
        state = "attribution_uncertain" if uncertain else "predicted"
        signature = tuple(sorted(candidates))
        if (current is None or current.speaker_id != speaker or current.uncertain != uncertain
                or signature != current_signature or unit.start - current.end_s > 1.5 or unit.start < current.end_s - .05):
            current = Utterance(start_s=unit.start, end_s=max(unit.end, unit.start),
                utterance_id=uuid.uuid5(uuid.NAMESPACE_URL, f"word-event:{source_audio_ref}:{index}:{unit.start}:{unit.text}").hex,
                speaker=speaker, text="", source_audio_ref=source_audio_ref, uncertain=uncertain,
                processing_state=state, asr_provenance={"model": asr_model, "origin": "aligned_asr", "word_indices": []},
                diarization_provenance={"source": "ordinary_activity", "exclusive_available": bool(exclusive_turns),
                    "assignment": "unresolved_mixture" if uncertain else "single_active_speaker"})
            events.append(current)
            current_signature = signature
        current.words.append(Word(unit.start, unit.end, unit.text, unit.prob))
        current.text = " ".join(w.text.strip() for w in current.words)
        current.end_s = max(current.end_s, unit.end)
        current.asr_provenance["word_indices"].append(index)
        current.candidate_speakers = [{"speaker": s, "activity_seconds": round(v, 5)} for s, v in candidates.items()]
        current.original_prediction = {"start_s": current.start_s, "end_s": current.end_s,
                                       "speaker": current.speaker, "text": current.text,
                                       "words": [asdict(w) for w in current.words]}
    materialize_overlap_utterances(events, ordinary_turns=ordinary_turns,
                                  exclusive_turns=exclusive_turns, source_audio_ref=source_audio_ref)
    recompute_overlap_relationships(events)
    return events


def clean_identity_windows(turns, *, min_s=3., max_s=8.):
    """Only contiguous single-speaker activity contributes identity evidence."""
    pieces = []
    for span in activity_spans(turns):
        if span["overlap"]:
            continue
        speaker = span["speakers"][0]
        if pieces and pieces[-1]["speaker"] == speaker and abs(pieces[-1]["end_s"] - span["start_s"]) < 1e-6:
            pieces[-1]["end_s"] = span["end_s"]
        else:
            pieces.append({"start_s": span["start_s"], "end_s": span["end_s"], "speaker": speaker})
    windows = []
    for piece in pieces:
        start = piece["start_s"] + .15
        while start + min_s <= piece["end_s"] - .15:
            end = min(start + max_s, piece["end_s"] - .15)
            windows.append({"start_s": start, "end_s": end, "speaker": piece["speaker"]})
            start = end
    return windows


def identity_review(embedding_windows, *, similarity_threshold=.72, margin=.08):
    """Propose acoustic matches, never relabel or merge identities automatically.

    Thresholds are conservative review heuristics, not calibrated accuracy.
    Multiple clean samples are required. Original IDs remain stable.
    """
    groups = defaultdict(list)
    for window in embedding_windows:
        vector = np.asarray(window.get("embedding", []), dtype=float)
        if window["end_s"] - window["start_s"] < 3 or window.get("overlap") or vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            continue
        norm = np.linalg.norm(vector)
        if norm > 0:
            groups[window["speaker"]].append((window, vector / norm))
    centroids = {}
    for speaker, samples in groups.items():
        if len(samples) < 3:
            continue
        center = np.median(np.stack([v for _, v in samples]), axis=0)
        norm = np.linalg.norm(center)
        if norm > 0:
            centroids[speaker] = center / norm
    proposals = []
    for speaker, samples in groups.items():
        for window, vector in samples:
            scores = sorted(((spk, float(vector @ center)) for spk, center in centroids.items()), key=lambda pair: (-pair[1], pair[0]))
            best = scores[0] if scores else (None, 0)
            ambiguous = not scores or best[1] < similarity_threshold or (len(scores) > 1 and best[1] - scores[1][1] < margin)
            if ambiguous or best[0] != speaker:
                proposals.append({**{k: v for k, v in window.items() if k != "embedding"},
                    "kind": "identity_uncertain" if ambiguous else "identity_conflict_candidate",
                    "proposed_speaker": None if ambiguous else best[0], "scores": scores,
                    "requires_review": True, "applied": False})
    pairs = []
    names = sorted(centroids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            similarity = float(centroids[a] @ centroids[b])
            if similarity >= similarity_threshold:
                pairs.append({"speakers": [a, b], "similarity": similarity,
                              "kind": "possible_fragmentation", "requires_review": True, "applied": False})
    return {"scope": "recording", "policy": "review_only_no_automatic_merges", "thresholds_calibrated": False,
            "clean_samples_per_speaker": {s: len(v) for s, v in groups.items()},
            "centroids": {s: c.tolist() for s, c in centroids.items()}, "suspected_regions": proposals,
            "possible_fragmentation": pairs}


def audit_events(events, ordinary_turns, *, raw_words=None, aligned_words=None):
    spans = activity_spans(ordinary_turns)
    overlap_regions = []
    for span in spans:
        if not span["overlap"]:
            continue
        visible = [u for u in events if u.end_s > span["start_s"] and u.start_s < span["end_s"]]
        missing = [u for u in visible if u.processing_state == "missing_transcription"]
        overlap_regions.append({**span, "event_ids": [u.event_id for u in visible],
            "existing_asr_text": [u.text for u in visible if u.text],
            "potentially_missing_speakers": sorted({u.speaker_id for u in missing if u.speaker_id}),
            "separation_recommended": bool(missing), "verification": "model_candidate"})
    words = [w for u in events for w in u.words]
    counter = lambda items: Counter((round(w.start, 3), round(w.end, 3), w.text.strip()) for w in items)
    result = {"event_count": len(events), "word_count": len(words), "overlap_regions": overlap_regions,
              "missing_transcription_events": [u.event_id for u in events if u.processing_state == "missing_transcription"],
              "uncertain_identity_events": [u.event_id for u in events if u.uncertain],
              "quality_metrics": None, "reference_status": "manual_annotations_required"}
    if aligned_words is not None:
        result["reconstruction_word_loss"] = sum((counter(aligned_words) - counter(words)).values())
        result["reconstruction_word_additions"] = sum((counter(words) - counter(aligned_words)).values())
    if raw_words is not None and aligned_words is not None:
        # Alignment legitimately moves timestamps; lexical changes are audited separately.
        result["alignment_lexical_loss"] = dict(Counter(w.text.strip() for w in raw_words) - Counter(w.text.strip() for w in aligned_words))
    return result


def annotate_identity_uncertainty(events, review):
    """Show acoustic uncertainty in the same events users actually review."""
    for event in events:
        if event.user_corrections or event.edit_status.startswith("user_"):
            continue
        suspects = [r for r in review.get("suspected_regions", [])
                    if r["end_s"] > event.start_s and r["start_s"] < event.end_s]
        if suspects:
            event.uncertain = True
            event.diarization_provenance["global_identity_review"] = suspects
            if event.processing_state == "predicted":
                event.processing_state = "attribution_uncertain"


def review_confirmed_identities(session, embedding_windows):
    """Use confirmations as recording-wide constraints with clean acoustic support.

    Short interruptions establish distinct identities but cannot alone train a
    global prototype. At least three clean samples and nine seconds are needed.
    All propagation remains an explicit review proposal with a stale-edit guard.
    """
    confirmed = [r for r in session.speaker_evidence if r.get("confirmed") and r.get("speaker_id")]
    by_id = {u.event_id: u for u in session.utterances}
    anchors = defaultdict(dict)
    insufficient = []
    for evidence in confirmed:
        event = by_id.get(evidence.get("event_id"))
        if not event:
            continue
        for index, window in enumerate(embedding_windows):
            overlap = min(event.end_s, window["end_s"]) - max(event.start_s, window["start_s"])
            duration = window["end_s"] - window["start_s"]
            if duration >= 3 and overlap >= duration * .95 and window.get("embedding") and not window.get("overlap"):
                anchors[evidence["speaker_id"]][index] = window
        if not anchors[evidence["speaker_id"]]:
            insufficient.append({"start_s": event.start_s, "end_s": event.end_s, "event_id": event.event_id,
                "speaker": evidence["speaker_id"], "kind": "confirmed_short_anchor_needs_clean_evidence",
                "requires_review": True, "applied": False})
    prototypes = {}
    for speaker, windows in anchors.items():
        values = list(windows.values())
        if len(values) < 3 or sum(w["end_s"] - w["start_s"] for w in values) < 9:
            continue
        vectors = np.array([w["embedding"] for w in values], dtype=float)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        center = np.median(vectors, axis=0)
        prototypes[speaker] = center / max(np.linalg.norm(center), 1e-12)
    proposals = []
    for event in session.utterances:
        if event.user_corrections or event.edit_status.startswith("user_") or event.is_overlap or not event.text:
            continue
        clean = [w for w in embedding_windows if w.get("embedding") and not w.get("overlap")
                 and min(w["end_s"], event.end_s) - max(w["start_s"], event.start_s) >= 3]
        if not clean or not prototypes:
            continue
        coverage = sum(max(0, min(w["end_s"], event.end_s) - max(w["start_s"], event.start_s)) for w in clean)
        if coverage < event.duration_s * .75:
            continue
        vector = np.mean([w["embedding"] for w in clean], axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        scores = sorted(((s, float(vector @ c)) for s, c in prototypes.items()), key=lambda pair: -pair[1])
        if scores[0][1] < .78 or (len(scores) > 1 and scores[0][1] - scores[1][1] < .12):
            continue
        target = scores[0][0]
        if target == event.speaker_id:
            continue
        proposals.append({"candidate_id": uuid.uuid5(uuid.NAMESPACE_URL, f"identity:{event.event_id}:{target}").hex,
            "kind": "speaker_assignment", "status": "proposed", "proposed_event_id": event.event_id,
            "proposed_speaker": target, "event": event.to_dict(), "event_snapshot": event.to_dict(),
            "evidence": {"scope": "recording", "scores": scores, "clean_anchor_count": len(anchors[target]),
                         "matched_windows": [{k: w[k] for k in ("start_s", "end_s", "speaker")} for w in clean],
                         "thresholds_calibrated": False}})
    return {"policy": "confirmed_identity_with_clean_evidence_review_only", "candidates": proposals,
            "suspected_regions": insufficient, "supported_identities": sorted(prototypes),
            "distinct_identity_constraints": [r for r in session.speaker_evidence if r.get("relation") == "distinct_identity_group"]}
