"""Phase 1 & 2 diagnostics for overlapping speech and multi-speaker timeline fidelity.

This module inspects the 9 pipeline stages between the original decoded audio
and the UI rendering layer, pinpointing where a simultaneous speaker or their
words disappear or become flattened into a single speaker stream.

Model activity and signal statistics are evidence to inspect, not manually
verified speech or word ownership. The report counts only demonstrated
representation/serialization losses; ASR and attribution candidates remain
explicitly unverified without reference annotations.

The six diagnostic failure categories are:
1. ``missing_speech_recognition`` — Speech is present in the audio and/or
   non-exclusive diarization for a speaker, but ASR produced no words for that
   speaker stream (e.g., due to stereo downmix attenuation or single-stream
   Whisper decoding during simultaneous speech).
2. ``missing_diarization`` — Speech is present in the audio/ASR, but diarization
   produced no speaker turn covering the interval.
3. ``incorrect_speaker_assignment`` — Words and diarization turns exist, but
   exclusive diarization truncation, word-to-turn overlap scoring, or post-hoc
   smoothing (``_smooth`` / ``_absorb_alternations``) assigned the words to the
   wrong speaker.
4. ``transcript_segment_merging`` — Adjacent or overlapping speech was collapsed
   into a single speaker segment (``build_segments``) or tagged with an orphan
   ``overlap_group_id`` without materializing a simultaneous partner utterance.
5. ``data_serialization_loss`` — Persisted session data discarded raw
   non-exclusive and exclusive diarization turn intervals, keeping only integer
   summary counts or speaker-less overlap ranges.
6. ``ui_rendering_loss`` — The UI displayed an orphan ``"In contemporanea"``
   badge on a single speaker stream or could not display/edit simultaneous
   overlapping speaker events and unassigned speech lanes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FAILURE_CATEGORIES: tuple[str, ...] = (
    "missing_speech_recognition",
    "missing_diarization",
    "incorrect_speaker_assignment",
    "transcript_segment_merging",
    "data_serialization_loss",
    "ui_rendering_loss",
)


def format_mmss(seconds: float) -> str:
    """Format seconds using the application's MM:SS playback timestamp convention."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def format_mmss_precise(seconds: float) -> str:
    """Format seconds as MM:SS.d with tenth-of-a-second precision."""
    s = max(0.0, float(seconds))
    minutes = int(s // 60)
    rem = s - minutes * 60
    return f"{minutes:02d}:{rem:04.1f}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


@dataclass
class StageDiagnosticFinding:
    """A concrete defect or loss identified at one of the 9 pipeline stages."""

    finding_id: str
    category: str
    stage: str
    start_s: float
    end_s: float
    timestamp_mmss: str
    primary_speaker: str | None = None
    missing_or_affected_speaker: str | None = None
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OverlapRegionTrace:
    """9-stage trace for a specific overlapping speech interval or speaker turn."""

    start_s: float
    end_s: float
    timestamp_mmss: str
    non_exclusive_speakers: list[str] = field(default_factory=list)
    exclusive_speakers: list[str] = field(default_factory=list)
    asr_words: list[dict[str, Any]] = field(default_factory=list)
    assigned_speakers: list[str] = field(default_factory=list)
    utterance_ids: list[str] = field(default_factory=list)
    utterance_speakers: list[str] = field(default_factory=list)
    has_orphan_overlap_badge: bool = False
    stereo_correlation: float | None = None
    failure_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OverlapDiagnosticReport:
    """Structured diagnostic report across the 9 pipeline stages."""

    audio_path: str | None = None
    window_start_s: float = 0.0
    window_end_s: float = 0.0
    timestamp_format: str = "MM:SS"
    stage_summary: dict[str, Any] = field(default_factory=dict)
    overlap_traces: list[OverlapRegionTrace] = field(default_factory=list)
    findings: list[StageDiagnosticFinding] = field(default_factory=list)
    intermediate_outputs: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def category_counts(self) -> dict[str, int]:
        counts = {cat: 0 for cat in FAILURE_CATEGORIES}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "window_start_s": round(self.window_start_s, 3),
            "window_end_s": round(self.window_end_s, 3),
            "timestamp_format": self.timestamp_format,
            "category_counts": self.category_counts,
            "stage_summary": self.stage_summary,
            "overlap_traces": [t.to_dict() for t in self.overlap_traces],
            "findings": [f.to_dict() for f in self.findings],
            "intermediate_outputs": self.intermediate_outputs,
            "limitations": self.limitations,
            "provenance": self.provenance,
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Overlap & Speaker Attribution Diagnostic Report",
            "",
            f"- **Audio source**: `{self.audio_path or 'N/A'}`",
            f"- **Inspected window**: `{format_mmss_precise(self.window_start_s)}` – `{format_mmss_precise(self.window_end_s)}` (`{self.window_start_s:.2f}s` – `{self.window_end_s:.2f}s`)",
            f"- **UI Timestamp Convention**: `{self.timestamp_format}` (`01:39` = `99.0s`, `01:41` = `101.0s`)",
            "",
            "## Failure Category Summary",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
        for cat, count in self.category_counts.items():
            lines.append(f"| `{cat}` | {count} |")

        lines.extend(["", "## Evidence and limitations", ""])
        lines.extend(f"- {item}" for item in self.limitations)
        for stage, summary in self.stage_summary.items():
            lines.append(f"- `{stage}`: {'available' if summary.get('available') else 'unavailable'}; "
                         f"{summary.get('origin', 'observed input')}")
        lines.append("- Full supplied intervals, words, and event snapshots are retained in the JSON intermediate_outputs field.")

        lines.extend(["", "## Overlap Region Traces", ""])
        if not self.overlap_traces:
            lines.append("_No overlap regions detected in the inspected window._")
        else:
            for idx, trace in enumerate(self.overlap_traces, start=1):
                lines.extend(
                    [
                        f"### {idx}. `{trace.timestamp_mmss}` (`{trace.start_s:.3f}s` – `{trace.end_s:.3f}s`)",
                        f"- **Non-exclusive diarization speakers**: `{', '.join(trace.non_exclusive_speakers) or 'none'}`",
                        f"- **Exclusive diarization speakers**: `{', '.join(trace.exclusive_speakers) or 'none'}`",
                        f"- **Assigned word speakers**: `{', '.join(trace.assigned_speakers) or 'none'}`",
                        f"- **Materialized utterance speakers**: `{', '.join(trace.utterance_speakers) or 'none'}`",
                        f"- **Orphan `'In contemporanea'` badge**: `{trace.has_orphan_overlap_badge}`",
                        f"- **Failure categories**: `{', '.join(trace.failure_categories) or 'none'}`",
                        "",
                    ]
                )

        lines.extend(["## Detailed Stage Findings", ""])
        if not self.findings:
            lines.append("_No stage defects detected._")
        else:
            for finding in self.findings:
                lines.extend(
                    [
                        f"- **`[{finding.stage}]` `{finding.category}` at `{finding.timestamp_mmss}` (`{finding.start_s:.2f}s–{finding.end_s:.2f}s`)**",
                        f"  - Primary speaker: `{finding.primary_speaker or 'N/A'}` | Missing/affected speaker: `{finding.missing_or_affected_speaker or 'N/A'}`",
                        f"  - {finding.summary}",
                    ]
                )
        return "\n".join(lines) + "\n"


def analyze_stereo_window(
    audio_path: Path | str | None,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Inspect decoded audio channels in ``[start_s, end_s]`` for stereo separation and downmix loss."""
    if not audio_path:
        return {"available": False}
    path = Path(audio_path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(path))
        sr = int(info.samplerate)
        ch = int(info.channels)
        f0 = max(0, int(start_s * sr))
        f1 = min(int(info.frames), int(end_s * sr))
        if f1 <= f0:
            return {"available": False, "samplerate": sr, "channels": ch}
        data, _ = sf.read(str(path), start=f0, stop=f1, dtype="float32", always_2d=True)
        if ch < 2:
            rms_mono = float(np.sqrt(np.mean(data[:, 0] ** 2)))
            return {
                "available": True,
                "samplerate": sr,
                "channels": ch,
                "rms_mono": round(rms_mono, 5),
                "stereo_correlation": 1.0,
                "downmix_attenuation_risk": False,
            }
        left = data[:, 0]
        right = data[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        rms_l = float(np.sqrt(np.mean(left**2)))
        rms_r = float(np.sqrt(np.mean(right**2)))
        rms_mid = float(np.sqrt(np.mean(mid**2)))
        rms_side = float(np.sqrt(np.mean(side**2)))
        corr = float(np.corrcoef(left, right)[0, 1]) if (rms_l > 1e-6 and rms_r > 1e-6) else 1.0
        return {
            "available": True,
            "samplerate": sr,
            "channels": ch,
            "rms_left": round(rms_l, 5),
            "rms_right": round(rms_r, 5),
            "rms_mid": round(rms_mid, 5),
            "rms_side": round(rms_side, 5),
            "stereo_correlation": round(corr, 4),
            "downmix_attenuation_risk": bool(corr < 0.35 and rms_side > 0.5 * rms_mid),
        }
    except Exception as exc:
        log.debug("Stereo window inspection failed for %s: %s", audio_path, exc)
        return {"available": False, "error": str(exc)}


def _normalize_turn(turn: Any) -> dict[str, Any]:
    if isinstance(turn, dict):
        return {
            "start": float(turn.get("start", turn.get("start_s", 0.0))),
            "end": float(turn.get("end", turn.get("end_s", 0.0))),
            "speaker": str(turn.get("speaker") or "UNKNOWN"),
        }
    return {
        "start": float(getattr(turn, "start", getattr(turn, "start_s", 0.0))),
        "end": float(getattr(turn, "end", getattr(turn, "end_s", 0.0))),
        "speaker": str(getattr(turn, "speaker", "UNKNOWN") or "UNKNOWN"),
    }


def _normalize_word(word: Any) -> dict[str, Any]:
    if isinstance(word, dict):
        return {
            "word": str(word.get("word", word.get("text", ""))).strip(),
            "start": float(word.get("start", word.get("start_s", 0.0))),
            "end": float(word.get("end", word.get("end_s", 0.0))),
            "speaker": word.get("speaker"),
            "probability": float(word.get("probability", word.get("confidence", 0.0)) or 0.0),
        }
    return {
        "word": str(getattr(word, "word", getattr(word, "text", ""))).strip(),
        "start": float(getattr(word, "start", getattr(word, "start_s", 0.0))),
        "end": float(getattr(word, "end", getattr(word, "end_s", 0.0))),
        "speaker": getattr(word, "speaker", None),
        "probability": float(getattr(word, "probability", getattr(word, "confidence", 0.0)) or 0.0),
    }


def diagnose_pipeline_stages(
    *,
    audio_path: Path | str | None = None,
    window_start_s: float = 0.0,
    window_end_s: float | None = None,
    vad_regions: list[Any] | None = None,
    ordinary_turns: list[Any] | None = None,
    exclusive_turns: list[Any] | None = None,
    overlaps: list[tuple[float, float]] | list[list[float]] | None = None,
    raw_asr_words: list[Any] | None = None,
    aligned_words: list[Any] | None = None,
    assigned_words: list[Any] | None = None,
    utterances: list[Any] | None = None,
    persisted_diarization: dict[str, Any] | None = None,
    persisted_utterances: list[Any] | None = None,
    rendered_event_ids: list[str] | None = None,
) -> OverlapDiagnosticReport:
    """Compare supplied evidence without treating model predictions as human truth.

    None means unavailable; an empty list is an observed empty output. Speaker
    activity without corresponding words is a review candidate, not proof of
    missing recognition or incorrect word attribution. A rendered-event list
    must come from the actual view; the canonical event list cannot substitute.
    """
    ord_turns = [_normalize_turn(t) for t in (ordinary_turns or [])]
    exc_turns = [_normalize_turn(t) for t in (exclusive_turns or [])]
    raw_words = [_normalize_word(w) for w in (raw_asr_words or [])]
    aln_words = [_normalize_word(w) for w in (aligned_words or [])]
    asg_words = [_normalize_word(w) for w in (assigned_words or [])]
    utts = list(utterances or [])

    if window_end_s is None:
        max_candidates = [window_start_s + 1.0]
        max_candidates.extend(t["end"] for t in ord_turns)
        max_candidates.extend(w["end"] for w in asg_words)
        max_candidates.extend(float(getattr(u, "end_s", 0.0)) for u in utts)
        window_end_s = max(max_candidates)

    w0 = float(window_start_s)
    w1 = float(window_end_s)

    ord_in_win = [t for t in ord_turns if _overlap(t["start"], t["end"], w0, w1) > 0.0]
    exc_in_win = [t for t in exc_turns if _overlap(t["start"], t["end"], w0, w1) > 0.0]
    raw_in_win = [w for w in raw_words if _overlap(w["start"], w["end"], w0, w1) > 0.0]
    aln_in_win = [w for w in aln_words if _overlap(w["start"], w["end"], w0, w1) > 0.0]
    asg_in_win = [w for w in asg_words if _overlap(w["start"], w["end"], w0, w1) > 0.0]
    utts_in_win = [
        u for u in utts if _overlap(float(getattr(u, "start_s", 0.0)), float(getattr(u, "end_s", 0.0)), w0, w1) > 0.0
    ]

    raw_overlaps: list[tuple[float, float]] = []
    if overlaps:
        for pair in overlaps:
            if len(pair) >= 2:
                ov0, ov1 = float(pair[0]), float(pair[1])
                if _overlap(ov0, ov1, w0, w1) > 0.0:
                    raw_overlaps.append((ov0, ov1))
    else:
        # Derive overlaps directly from non-exclusive turns
        for i, a in enumerate(ord_in_win):
            for b in ord_in_win[i + 1 :]:
                if a["speaker"] == b["speaker"]:
                    continue
                ov0 = max(a["start"], b["start"])
                ov1 = min(a["end"], b["end"])
                if ov1 - ov0 >= 0.05:
                    raw_overlaps.append((ov0, ov1))

    stereo_info = analyze_stereo_window(audio_path, w0, w1)

    findings: list[StageDiagnosticFinding] = []
    traces: list[OverlapRegionTrace] = []
    exclusive_reductions: list[dict[str, Any]] = []
    recognition_candidates: list[dict[str, Any]] = []
    assignment_candidates: list[dict[str, Any]] = []
    limitations = [
        "No human-verified word/speaker reference was supplied. Missing spoken content, WER, and incorrect attribution are not measured.",
        "Category counts count demonstrated findings only. Zero is not proof of correctness when evidence or verified annotation is unavailable.",
        "Non-exclusive and exclusive outputs are model predictions; an exclusive reduction is expected in overlap and is not itself a diarization defect.",
        "An overlap badge without a partner event proves a representation gap, not that the UI discarded an existing event.",
    ]
    finding_seq = 0

    def _next_id(prefix: str) -> str:
        nonlocal finding_seq
        finding_seq += 1
        return f"{prefix}_{finding_seq:03d}"

    # Stage 1: Stereo downmix attenuation check
    if stereo_info.get("downmix_attenuation_risk"):
        stereo_info["risk_note"] = (
            "Low channel correlation/high side energy warrants listening to the original channels. "
            "These signal statistics do not establish missing speech recognition or independent speaker channels."
        )

    # Stage 3 & 4 & 5: Non-exclusive speaker turns that have no ASR words or get crushed by exclusive diarization
    for turn in ord_in_win:
        t0, t1, spk = turn["start"], turn["end"], turn["speaker"]
        dur = t1 - t0
        if dur < 0.25:
            continue
        exc_cov = sum(_overlap(t0, t1, e["start"], e["end"]) for e in exc_in_win if e["speaker"] == spk)
        concurrent_other = [
            o for o in ord_in_win if o["speaker"] != spk and _overlap(t0, t1, o["start"], o["end"]) >= 0.10
        ]
        words_in_turn = [w for w in asg_in_win if _overlap(t0, t1, w["start"], w["end"]) > 0.0]
        raw_words_in_turn = [w for w in raw_in_win if _overlap(t0, t1, w["start"], w["end"]) > 0.0]
        spk_words_in_turn = [w for w in words_in_turn if w.get("speaker") == spk]

        if exc_turns and exc_cov < dur * 0.35 and concurrent_other:
            exclusive_reductions.append({
                "turn": turn,
                "non_exclusive_duration_s": round(dur, 3),
                "exclusive_coverage_s": round(exc_cov, 3),
                "concurrent_speakers": [o["speaker"] for o in concurrent_other],
                "interpretation": "Expected exclusive reduction; preserve ordinary activity separately.",
            })

        if raw_asr_words is not None and not raw_words_in_turn:
            recognition_candidates.append({
                "turn": turn, "reason": "No raw ASR word intersects model activity; requires listening and verified annotation.",
            })
        if assigned_words is not None and words_in_turn and not spk_words_in_turn and concurrent_other:
            assignment_candidates.append({
                "turn": turn, "assigned_words": words_in_turn,
                "reason": "No word assigned to this active model speaker; word ownership remains unverified.",
            })

    # Build traces for each overlap interval and check Stage 7 (utterance merging) & Stage 9 (UI orphan badge)
    overlap_group_counts: dict[str, int] = {}
    for u in utts:
        gid = getattr(u, "overlap_group_id", None)
        if gid:
            overlap_group_counts[str(gid)] = overlap_group_counts.get(str(gid), 0) + 1

    for ov0, ov1 in raw_overlaps:
        non_exc_spks = sorted(
            {t["speaker"] for t in ord_in_win if _overlap(ov0, ov1, t["start"], t["end"]) > 0.0}
        )
        exc_spks = sorted(
            {t["speaker"] for t in exc_in_win if _overlap(ov0, ov1, t["start"], t["end"]) > 0.0}
        )
        ov_words = [w for w in asg_in_win if _overlap(ov0, ov1, w["start"], w["end"]) > 0.0]
        asg_spks = sorted({str(w["speaker"]) for w in ov_words if w.get("speaker")})
        ov_utts = [
            u
            for u in utts_in_win
            if _overlap(ov0, ov1, float(getattr(u, "start_s", 0.0)), float(getattr(u, "end_s", 0.0))) > 0.0
        ]
        utt_ids = [str(getattr(u, "utterance_id", "")) for u in ov_utts]
        utt_spks = sorted(
            {
                str(getattr(u, "speaker", None) or "UNASSIGNED")
                for u in ov_utts
            }
        )
        has_orphan_badge = any(
            bool(getattr(u, "overlap_group_id", None))
            and overlap_group_counts.get(str(getattr(u, "overlap_group_id", "")), 0) < 2
            and not getattr(u, "overlapping_event_ids", [])
            for u in ov_utts
        )

        trace_cats: list[str] = []
        if utterances is not None and len(non_exc_spks) >= 2 and set(non_exc_spks) - set(utt_spks):
            trace_cats.append("transcript_segment_merging")
            findings.append(
                StageDiagnosticFinding(
                    finding_id=_next_id("stg7m"),
                    category="transcript_segment_merging",
                    stage="7_speaker_assignment",
                    start_s=ov0,
                    end_s=ov1,
                    timestamp_mmss=format_mmss(ov0),
                    primary_speaker=utt_spks[0] if utt_spks else None,
                    missing_or_affected_speaker=next(
                        (s for s in non_exc_spks if s not in utt_spks), None
                    ),
                    summary=(
                        f"Overlap interval {ov0:.2f}s–{ov1:.2f}s ({format_mmss(ov0)}) has simultaneous "
                        f"speakers {non_exc_spks} in non-exclusive diarization, but utterances only materialize "
                        f"{utt_spks or ['none']}."
                    ),
                    details={
                        "non_exclusive_speakers": non_exc_spks,
                        "utterance_ids": utt_ids,
                        "utterance_speakers": utt_spks,
                    },
                )
            )
        if rendered_event_ids is not None and any(uid not in rendered_event_ids for uid in utt_ids):
            trace_cats.append("ui_rendering_loss")
            findings.append(
                StageDiagnosticFinding(
                    finding_id=_next_id("stg9"),
                    category="ui_rendering_loss",
                    stage="9_ui_rendering",
                    start_s=ov0,
                    end_s=ov1,
                    timestamp_mmss=format_mmss(ov0),
                    primary_speaker=utt_spks[0] if utt_spks else None,
                    missing_or_affected_speaker=next(
                        (s for s in non_exc_spks if s not in utt_spks), None
                    ),
                    summary=(
                        f"Canonical event(s) {[uid for uid in utt_ids if uid not in rendered_event_ids]} "
                        f"at {format_mmss(ov0)} are absent from the supplied rendered-event inventory."
                    ),
                    details={"utterance_ids": utt_ids},
                )
            )

        traces.append(
            OverlapRegionTrace(
                start_s=round(ov0, 3),
                end_s=round(ov1, 3),
                timestamp_mmss=format_mmss(ov0),
                non_exclusive_speakers=non_exc_spks,
                exclusive_speakers=exc_spks,
                asr_words=ov_words,
                assigned_speakers=asg_spks,
                utterance_ids=utt_ids,
                utterance_speakers=utt_spks,
                has_orphan_overlap_badge=has_orphan_badge,
                stereo_correlation=stereo_info.get("stereo_correlation"),
                failure_categories=trace_cats,
            )
        )

    # Also check utterances in window that have an orphan overlap_group_id even if raw_overlaps was empty
    for u in utts_in_win:
        gid = getattr(u, "overlap_group_id", None)
        u_start = float(getattr(u, "start_s", 0.0))
        u_end = float(getattr(u, "end_s", 0.0))
        if (rendered_event_ids is not None and str(getattr(u, "utterance_id", "")) not in rendered_event_ids
                and gid and overlap_group_counts.get(str(gid), 0) < 2 and not getattr(u, "overlapping_event_ids", [])):
            already_reported = any(
                f.stage == "9_ui_rendering" and _overlap(f.start_s, f.end_s, u_start, u_end) > 0.0
                for f in findings
            )
            if not already_reported:
                findings.append(
                    StageDiagnosticFinding(
                        finding_id=_next_id("stg9u"),
                        category="ui_rendering_loss",
                        stage="9_ui_rendering",
                        start_s=u_start,
                        end_s=u_end,
                        timestamp_mmss=format_mmss(u_start),
                        primary_speaker=str(getattr(u, "speaker", None) or "UNASSIGNED"),
                        summary=(
                            f"Canonical event {getattr(u, 'utterance_id', '')} ({format_mmss(u_start)}) "
                            "is absent from the supplied rendered-event inventory."
                        ),
                        details={"utterance_id": getattr(u, "utterance_id", ""), "overlap_group_id": gid},
                    )
                )

    # Stage 8: Check persisted diarization dictionary for serialization loss
    if persisted_diarization is not None:
        has_turns_list = isinstance(persisted_diarization.get("turns"), list) and bool(
            persisted_diarization.get("turns")
        )
        ord_count = int(persisted_diarization.get("ordinary_turns", 0) or 0)
        exc_count = int(persisted_diarization.get("exclusive_turns", 0) or 0)
        missing_fields = []
        if ord_count > 0 and not has_turns_list:
            missing_fields.append("turns")
        if exc_count > 0 and not persisted_diarization.get("exclusive"):
            missing_fields.append("exclusive")
        if missing_fields:
            findings.append(
                StageDiagnosticFinding(
                    finding_id=_next_id("stg8"),
                    category="data_serialization_loss",
                    stage="8_persisted_data",
                    start_s=w0,
                    end_s=w1,
                    timestamp_mmss=format_mmss(w0),
                    summary=(
                        f"Persisted Session.diarization recorded ordinary_turns={ord_count} "
                        f"and exclusive_turns={exc_count}, "
                        f"but lacks the interval/speaker lists: {', '.join(missing_fields)}."
                    ),
                    details={"persisted_keys": sorted(persisted_diarization.keys()), "missing_fields": missing_fields},
                )
            )

    def event_snapshot(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            return dict(event)
        if hasattr(event, "to_dict"):
            return event.to_dict()
        return asdict(event)

    persisted_in_win = [u for u in (persisted_utterances or [])
                        if _overlap(float(getattr(u, "start_s", 0.0)),
                                    float(getattr(u, "end_s", 0.0)), w0, w1) > 0]
    vad_in_win = []
    for region in vad_regions or []:
        if isinstance(region, (tuple, list)):
            a, b = region[:2]
        else:
            a = getattr(region, "start_s", getattr(region, "start", 0.0))
            b = getattr(region, "end_s", getattr(region, "end", 0.0))
        if _overlap(float(a), float(b), w0, w1) > 0:
            vad_in_win.append({"start_s": float(a), "end_s": float(b)})

    stage_summary = {
        "1_decoded_audio": stereo_info,
        "2_vad": {
            "available": vad_regions is not None,
            "regions_in_window": len(vad_in_win) if vad_regions is not None else None,
        },
        "3_non_exclusive_diarization": {
            "available": ordinary_turns is not None,
            "turns_in_window": len(ord_in_win) if ordinary_turns is not None else None,
            "speakers_in_window": sorted({t["speaker"] for t in ord_in_win}),
            "overlaps_in_window": len(raw_overlaps),
        },
        "4_exclusive_diarization": {
            "available": exclusive_turns is not None,
            "turns_in_window": len(exc_in_win) if exclusive_turns is not None else None,
            "speakers_in_window": sorted({t["speaker"] for t in exc_in_win}),
            "activity_reductions": exclusive_reductions,
        },
        "5_raw_asr": {
            "available": raw_asr_words is not None,
            "words_in_window": len(raw_in_win) if raw_asr_words is not None else None,
            "review_candidates": recognition_candidates,
        },
        "6_word_alignment": {
            "available": aligned_words is not None,
            "words_in_window": len(aln_in_win) if aligned_words is not None else None,
        },
        "7_speaker_assignment": {
            "available": assigned_words is not None,
            "assigned_words_in_window": len(asg_in_win) if assigned_words is not None else None,
            "assigned_speakers": sorted({str(w["speaker"]) for w in asg_in_win if w.get("speaker")}),
            "review_candidates": assignment_candidates,
        },
        "8_persisted_data": {
            "available": persisted_diarization is not None or persisted_utterances is not None,
            "has_serialized_turns": bool(
                persisted_diarization and isinstance(persisted_diarization.get("turns"), list)
            ),
            "utterances_in_window": len(persisted_in_win) if persisted_utterances is not None else None,
        },
        "9_ui_rendering": {
            "available": rendered_event_ids is not None,
            "origin": "actual rendered-event inventory" if rendered_event_ids is not None else "canonical event expectation only; view not observed",
            "utterance_speakers_in_window": sorted(
                {str(getattr(u, "speaker", None) or "UNASSIGNED") for u in utts_in_win}
            ),
            "orphan_overlap_badges": sum(1 for t in traces if t.has_orphan_overlap_badge),
        },
    }
    for summary in stage_summary.values():
        if not summary.get("available"):
            summary.setdefault("origin", "not supplied or not observed")
    intermediate_outputs = {
        "1_decoded_audio": {"source": str(audio_path) if audio_path else None, **stereo_info},
        "2_vad": vad_in_win if vad_regions is not None else None,
        "3_non_exclusive_diarization": ord_in_win if ordinary_turns is not None else None,
        "4_exclusive_diarization": exc_in_win if exclusive_turns is not None else None,
        "5_raw_asr": raw_in_win if raw_asr_words is not None else None,
        "6_word_alignment": aln_in_win if aligned_words is not None else None,
        "7_speaker_assignment": {
            "words": asg_in_win if assigned_words is not None else None,
            "canonical_events": [event_snapshot(u) for u in utts_in_win] if utterances is not None else None,
        },
        "8_persisted_data": {
            "diarization": persisted_diarization,
            "events": [event_snapshot(u) for u in persisted_in_win] if persisted_utterances is not None else None,
        },
        "9_ui_rendering": {
            "expected_event_ids": [str(getattr(u, "utterance_id", "")) for u in utts_in_win],
            "observed_rendered_event_ids": rendered_event_ids,
        },
    }

    report = OverlapDiagnosticReport(
        audio_path=str(audio_path) if audio_path else None,
        window_start_s=w0,
        window_end_s=w1,
        timestamp_format="MM:SS",
        stage_summary=stage_summary,
        overlap_traces=traces,
        findings=findings,
        intermediate_outputs=intermediate_outputs,
        limitations=limitations,
    )
    log.info(
        "Overlap diagnostics [%s - %s]: %d overlap traces, %d findings (%s)",
        format_mmss(w0),
        format_mmss(w1),
        len(traces),
        len(findings),
        report.category_counts,
    )
    log.info("Supplied stage evidence: %s", {stage: summary.get("available", False)
                                            for stage, summary in stage_summary.items()})
    return report


def diagnose_session_overlaps(
    session: Any,
    window_start_s: float = 0.0,
    window_end_s: float | None = None,
) -> OverlapDiagnosticReport:
    """Run overlap diagnostics directly against a loaded ``Session`` object."""
    diar = getattr(session, "diarization", {}) or {}
    audio_path = None
    if getattr(session, "audio", None) is not None:
        resolved = session.audio.resolved_primary()
        if resolved is not None:
            audio_path = str(resolved)
    if not audio_path and getattr(session, "audio_path", ""):
        audio_path = str(session.audio_path)

    ordinary_turns = diar.get("turns")
    exclusive_turns = diar.get("exclusive")
    overlaps = diar.get("overlaps") or []
    utterances = list(getattr(session, "utterances", []) or [])

    report = diagnose_pipeline_stages(
        audio_path=audio_path,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        ordinary_turns=ordinary_turns,
        exclusive_turns=exclusive_turns,
        overlaps=overlaps,
        utterances=utterances,
        persisted_diarization=diar,
        persisted_utterances=utterances,
    )
    report.provenance = {"mode": "persisted_session", "session_id": getattr(session, "id", None)}
    report.stage_summary["1_decoded_audio"]["origin"] = "resolved session playback source; may be processed audio"
    report.stage_summary["8_persisted_data"]["origin"] = "loaded session JSON"
    report.limitations.append("Persisted events cannot substitute for raw ASR, alignment, VAD, or pre-merge assignment outputs; absent stages remain unavailable.")
    return report

