"""Metrics restricted to complete, human-reviewed annotation regions.

DER integrates simultaneous speaker activity without an overlap exemption.
Speaker IDs are reconciled once over all reviewed regions using an optimal
assignment; JER uses its own optimal Jaccard assignment. These are zero-collar
metrics, not the older transcript-span coverage heuristic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation import edit_distance, normalize


@dataclass(frozen=True)
class _Event:
    start: float
    end: float
    speaker: str
    text: str
    words: tuple[tuple[float, float, str], ...]
    event_id: str


def _value(item: Any, primary: str, fallback: str, default=None):
    if isinstance(item, dict):
        return item.get(primary, item.get(fallback, default))
    return getattr(item, primary, getattr(item, fallback, default))


def _time(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Annotation and hypothesis timestamps must be finite nonnegative numbers.")
    return float(value)


def _event(item: Any, index: int, *, reference: bool) -> _Event:
    start = _time(_value(item, "start_s", "start"))
    end = _time(_value(item, "end_s", "end"))
    if end < start or (reference and end == start):
        raise ValueError("Reference events require positive duration; event end must not precede start.")
    speaker = _value(item, "speaker_id", "speaker")
    unresolved = speaker is None or str(speaker).strip().upper() in ("", "UNKNOWN", "UNASSIGNED", "NONE")
    if reference and unresolved:
        raise ValueError("A verified reference must identify every annotated speaker.")
    # Unassigned hypothesis events contribute activity and confusion; they are
    # not eligible for identity matching or collapsed into one phantom voice.
    label = f"\x00unassigned:{index}" if unresolved else str(speaker)
    event_id = str(_value(item, "event_id", "utterance_id", index))
    text = _value(item, "text", "text", "")
    if not isinstance(text, str):
        raise ValueError("Event transcription must be text.")
    if reference and not text.strip():
        raise ValueError("A complete speech reference requires verified text for every event.")
    words = []
    if _value(item, "words_valid", "words_valid", True):
        for word in _value(item, "words", "words", []) or []:
            a = _time(_value(word, "start_s", "start"))
            b = _time(_value(word, "end_s", "end"))
            token = _value(word, "text", "word", "")
            if b < a or not isinstance(token, str):
                raise ValueError("Word timing/text is invalid.")
            if reference and (a < start - 1e-6 or b > end + 1e-6):
                raise ValueError("Verified word timings must lie inside their reference event.")
            words.append((a, b, token))
    if _value(item, "edit_status", "edit_status") == "untranscribed_overlap":
        text, words = "", []
    return _Event(start, end, label, text, tuple(words), event_id)


def _verified_regions(manifest: dict, source_sha256: str) -> tuple[list[tuple[float, float]], list[_Event]]:
    digest = str(manifest.get("source_sha256", "")).lower()
    if (manifest.get("verification_level") != "human_verified"
            or not isinstance(manifest.get("reviewed_by"), str) or not manifest["reviewed_by"].strip()
            or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            or digest != str(source_sha256).lower()):
        raise ValueError("Metrics require a human_verified reference, reviewed_by, and matching source_sha256.")
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("A verified reference must contain reviewed regions.")
    complete = manifest.get("annotation_complete") is True or all(
        isinstance(region, dict) and region.get("annotation_complete") is True for region in regions)
    if manifest.get("annotation_complete") is False or not complete:
        raise ValueError("Metrics require explicit annotation_complete:true for every scored region.")
    windows, events = [], []
    for region in regions:
        if not isinstance(region, dict) or region.get("annotation_complete") is False:
            raise ValueError("Incomplete annotation cannot be scored.")
        if region.get("verification_level", "human_verified") != "human_verified":
            raise ValueError("Unverified regions cannot be scored under a verified manifest.")
        a, b = _time(region.get("start_s")), _time(region.get("end_s"))
        if b <= a or not isinstance(region.get("events"), list):
            raise ValueError("Reviewed regions require positive duration and an events list, including silence.")
        windows.append((a, b))
        for item in region["events"]:
            event = _event(item, len(events), reference=True)
            if event.start < a or event.end > b:
                raise ValueError("Reference events must be wholly inside their reviewed region.")
            events.append(event)
    windows.sort()
    if any(right[0] < left[1] for left, right in zip(windows, windows[1:])):
        raise ValueError("Reviewed regions must not overlap; duplicate annotation would bias metrics.")
    return windows, events


def _activity(windows, reference, hypothesis):
    changes = defaultdict(list)
    for start, end in windows:
        changes[start].append((0, "scope", 1))
        changes[end].append((0, "scope", -1))
    for kind, events in ((1, reference), (2, hypothesis)):
        for event in events:
            for start, end in windows:
                a, b = max(start, event.start), min(end, event.end)
                if b > a:
                    changes[a].append((kind, event.speaker, 1))
                    changes[b].append((kind, event.speaker, -1))
    active = [Counter(), Counter(), Counter()]
    times = sorted(changes)
    intervals = []
    for start, end in zip(times, times[1:]):
        for kind, speaker, delta in changes[start]:
            active[kind][speaker] += delta
        if active[0]["scope"] > 0:
            intervals.append((start, end,
                              {speaker for speaker, count in active[1].items() if count > 0},
                              {speaker for speaker, count in active[2].items() if count > 0}))
    return intervals


def _optimal_mapping(matrix, ref_labels, hyp_labels) -> dict[str, str]:
    if not ref_labels or not hyp_labels:
        return {}
    rows, cols = linear_sum_assignment(-matrix)
    return {hyp_labels[col]: ref_labels[row] for row, col in zip(rows, cols) if matrix[row, col] > 0}


def _tokens(events, windows):
    by_speaker = defaultdict(list)
    unresolved_boundaries = []
    for event in sorted(events, key=lambda e: (e.start, e.end, e.event_id)):
        intersects = any(min(event.end, b) > max(event.start, a) for a, b in windows)
        if not intersects:
            continue
        if event.words:
            for a, b, text in sorted(event.words):
                if any(left <= (a + b) / 2 < right for left, right in windows):
                    by_speaker[event.speaker].extend(normalize(text))
        elif any(a <= event.start and event.end <= b for a, b in windows):
            by_speaker[event.speaker].extend(normalize(event.text))
        elif normalize(event.text):
            unresolved_boundaries.append(event.event_id)
    return dict(by_speaker), unresolved_boundaries


def _word_errors(reference, hypothesis, windows, mapping):
    ref_tokens, ref_boundaries = _tokens(reference, windows)
    hyp_tokens, hyp_boundaries = _tokens(hypothesis, windows)
    if ref_boundaries or hyp_boundaries:
        return {"status": "unavailable", "wer": None,
                "reason": "Events cross scoring boundaries without word timestamps; their words cannot be filtered accurately.",
                "reference_boundary_events": ref_boundaries, "hypothesis_boundary_events": hyp_boundaries}
    combined = defaultdict(list)
    for speaker, tokens in hyp_tokens.items():
        # A private unmatched key keeps insertions separate from all verified
        # reference identities, including unresolved hypothesis speakers.
        combined[mapping.get(speaker, f"\x00unmatched:{speaker}")].extend(tokens)
    errors = insertions = deletions = substitutions = 0
    for speaker in set(ref_tokens) | set(combined):
        distance, ins, delete, sub = edit_distance(ref_tokens.get(speaker, []), combined.get(speaker, []))
        errors += distance
        insertions += ins
        deletions += delete
        substitutions += sub
    denominator = sum(map(len, ref_tokens.values()))
    return {"status": "measured" if denominator else "undefined_no_reference_words",
            "wer": errors / denominator if denominator else None,
            "reference_words": denominator, "hypothesis_words": sum(map(len, hyp_tokens.values())),
            "errors": errors, "insertions": insertions, "deletions": deletions,
            "substitutions": substitutions}


def evaluate_verified_reference(
    reference: dict,
    hypothesis: list[Any],
    *,
    source_sha256: str,
    runtime_metadata: dict | None = None,
) -> dict:
    """Score only complete, reviewed regions tied to the exact recording.

    ``reference`` requires verification_level="human_verified", reviewed_by,
    source_sha256, annotation_complete=true, and regions containing absolute
    start_s/end_s and all speakers' events. Hypotheses may be Utterance objects
    or event dictionaries. Runtime and VRAM metadata never affect accuracy.
    """
    if not isinstance(reference, dict):
        raise ValueError("The verified reference must be a manifest object.")
    windows, ref_events = _verified_regions(reference, source_sha256)
    hyp_events = [_event(item, index, reference=False) for index, item in enumerate(hypothesis)]
    intervals = _activity(windows, ref_events, hyp_events)
    ref_duration, hyp_duration, intersection = defaultdict(float), defaultdict(float), defaultdict(float)
    for start, end, ref_active, hyp_active in intervals:
        duration = end - start
        for speaker in ref_active:
            ref_duration[speaker] += duration
        for speaker in hyp_active:
            hyp_duration[speaker] += duration
        for ref_speaker in ref_active:
            for hyp_speaker in hyp_active:
                intersection[(ref_speaker, hyp_speaker)] += duration
    ref_labels = sorted(ref_duration)
    hyp_labels = sorted(speaker for speaker in hyp_duration if not speaker.startswith("\x00unassigned:"))
    matrix = np.array([[intersection[(r, h)] for h in hyp_labels] for r in ref_labels], dtype=float)
    mapping = _optimal_mapping(matrix, ref_labels, hyp_labels)
    miss = false_alarm = confusion = 0.0
    overlap_windows = []
    for start, end, ref_active, hyp_active in intervals:
        duration = end - start
        correct = sum(mapping.get(speaker) in ref_active for speaker in hyp_active if speaker in mapping)
        miss += max(0, len(ref_active) - len(hyp_active)) * duration
        false_alarm += max(0, len(hyp_active) - len(ref_active)) * duration
        confusion += (min(len(ref_active), len(hyp_active)) - correct) * duration
        if len(ref_active) >= 2:
            if overlap_windows and abs(overlap_windows[-1][1] - start) < 1e-9:
                overlap_windows[-1] = (overlap_windows[-1][0], end)
            else:
                overlap_windows.append((start, end))
    total = sum(ref_duration.values())
    jaccard = np.array([[intersection[(r, h)] / max(1e-12, ref_duration[r] + hyp_duration[h] - intersection[(r, h)])
                         for h in hyp_labels] for r in ref_labels], dtype=float)
    jer_mapping = _optimal_mapping(jaccard, ref_labels, hyp_labels)
    jerk = sum(jaccard[ref_labels.index(r), hyp_labels.index(h)] for h, r in jer_mapping.items())

    associations = defaultdict(list)
    ambiguous = []
    for hyp_label in hyp_labels:
        scores = sorted(((intersection[(r, hyp_label)], r) for r in ref_labels), reverse=True)
        if not scores or scores[0][0] <= 0:
            continue
        winners = [r for weight, r in scores if abs(weight - scores[0][0]) < 1e-9]
        chosen = winners[0] if len(winners) == 1 else mapping.get(hyp_label)
        if chosen in winners:
            associations[chosen].append(hyp_label)
        else:
            ambiguous.append({"hypothesis_speaker": hyp_label, "reference_candidates": winners})
    fragments = {speaker: max(0, len(associations[speaker]) - 1) for speaker in ref_labels}
    overlap_wer = (_word_errors(ref_events, hyp_events, overlap_windows, mapping) if overlap_windows
                   else {"status": "not_applicable_no_reference_overlap", "wer": None})
    overlap_wer["definition"] = "Speaker-attributed WER within reference intervals containing at least two active speakers, using the fixed global DER mapping."
    return {
        "verification_level": "human_verified", "reviewed_by": reference["reviewed_by"],
        "source_sha256": str(source_sha256).lower(), "scope": "reviewed_regions_only",
        "reviewed_regions": [{"start_s": a, "end_s": b} for a, b in windows],
        "reviewed_seconds": sum(b - a for a, b in windows),
        "der": {"der": (miss + false_alarm + confusion) / total if total else None,
                "miss_s": miss, "false_alarm_s": false_alarm, "confusion_s": confusion,
                "reference_speaker_seconds": total, "hypothesis_speaker_seconds": sum(hyp_duration.values()),
                "mapping": mapping, "collar_s": 0.0, "overlap_included": True},
        "jer": {"jer": max(0.0, 1 - jerk / len(ref_labels)) if ref_labels else None, "mapping": jer_mapping,
                "reference_speakers": len(ref_labels), "hypothesis_speakers": len(hyp_duration)},
        "speaker_attributed_wer": _word_errors(ref_events, hyp_events, windows, mapping),
        "overlap_wer": overlap_wer,
        "identity_fragmentation": {"extra_labels": sum(fragments.values()),
                                   "fragmented_reference_speakers": sum(value > 0 for value in fragments.values()),
                                   "extra_labels_by_reference": fragments,
                                   "hypothesis_labels_by_reference": dict(associations),
                                   "ambiguous_associations": ambiguous},
        "performance": dict(runtime_metadata or {}),
        "limitations": [
            "No inference about unreviewed recording regions or incomplete annotations is made.",
            "DER/JER use exact boundaries with zero collar; they include overlap and reviewed silence.",
            "Global identity mapping maximizes reviewed speaker-time overlap; word scores use this same mapping.",
            "Without word timestamps, boundary-crossing events make the affected WER unavailable; words are never proportionally guessed.",
            "JER averages reference speakers; extra hypothesis speakers remain visible in DER false alarms and speaker counts.",
            "Fragmentation counts dominant temporal associations; unresolved coactivity ties are reported separately rather than asserted as errors.",
        ],
    }
