"""Build a reproducible local Phase 0 reference corpus from a timestamped transcript.

The input transcript is treated as data.  Each line must have the form
``[Speaker HH:MM:SS] text``.  The source provides the manually assigned speaker
and turn start; the end of a turn is the next annotated start (or the selected
window end).  This conservative policy is recorded in the manifest so that
later DER/tcpWER reports cannot be mistaken for frame-perfect annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LINE = re.compile(r"^\[(?P<speaker>[^ ]+) (?P<time>\d\d:\d\d:\d\d)\]\s*(?P<text>.*)$")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def seconds(value: str) -> float:
    hours, minutes, secs = (int(part) for part in value.split(":"))
    if minutes >= 60 or secs >= 60:
        raise ValueError(f"Timestamp non valido: {value}")
    return float(hours * 3600 + minutes * 60 + secs)


def parse_transcript(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE.fullmatch(raw)
        if match is None:
            raise ValueError(f"Riga {line_number} non conforme: {raw!r}")
        start = seconds(match["time"])
        text = " ".join(match["text"].split())
        if not text:
            raise ValueError(f"Riga {line_number} senza testo")
        rows.append({"line": line_number, "start": start,
                     "speaker": match["speaker"], "text": text})
    if not rows:
        raise ValueError("Trascrizione vuota")
    if any(right["start"] < left["start"] for left, right in zip(rows, rows[1:])):
        raise ValueError("Timestamp non monotoni")
    return rows


def build_manifest(audio: Path, transcript: Path, *, duration: float,
                   window_start: float, window_end: float,
                   audio_profile: str,
                   verification_level: str = "imported_draft") -> tuple[dict, str, str, str]:
    if window_start < 0 or window_end <= window_start or window_end > duration:
        raise ValueError("Finestra Phase 0 non valida")
    valid_levels = {
        "imported_draft": "imported_draft_v1",
        "human_verified_text": "verified_text_draft_timing_v1",
        "human_verified_timing_and_speakers": "gold_reference_v1",
    }
    if verification_level not in valid_levels:
        raise ValueError(
            f"verification_level non valido: {verification_level}. "
            f"Valori ammessi: {list(valid_levels.keys())}"
        )
    status = valid_levels[verification_level]
    rows = parse_transcript(transcript)
    selected = [(index, row) for index, row in enumerate(rows)
                if window_start <= row["start"] < window_end]
    if not selected:
        raise ValueError("La finestra non contiene turni")

    turns = []
    grouped = defaultdict(list)
    for source_index, row in selected:
        next_index = source_index + 1
        # Equal starts are retained as simultaneous/overlapping annotations;
        # end each of them at the next strictly later source timestamp.
        while next_index < len(rows) and rows[next_index]["start"] <= row["start"]:
            next_index += 1
        next_start = rows[next_index]["start"] if next_index < len(rows) else duration
        end = min(next_start, window_end)
        if end <= row["start"]:
            raise ValueError(f"Intervallo nullo alla riga {row['line']}")
        turn = {"start": row["start"], "end": end, "speaker": row["speaker"],
                "text": row["text"], "source_line": row["line"]}
        turns.append(turn)
        grouped[row["speaker"]].append(row["text"])

    counts = Counter(turn["speaker"] for turn in turns)
    durations = defaultdict(float)
    for turn in turns:
        durations[turn["speaker"]] += turn["end"] - turn["start"]
    manifest = {
        "schema_version": 1,
        "corpus": "tessitura-phase0",
        "status": status,
        "annotation": {
            "verification_level": verification_level,
            "annotator": (
                "user-provided timestamped transcript (unverified)"
                if verification_level == "imported_draft"
                else "human verified"
            ),
            "speaker_labels": "preserved from source transcript",
            "turn_start": "source timestamp",
            "turn_end": "next source timestamp, clipped to the selected window",
            "overlap_policy": "same-start turns retain overlapping spans; no frame-level overlap annotation",
            "review_note": (
                "Draft reference for preliminary alignment; cannot be certified as gold without human verification."
                if verification_level == "imported_draft"
                else "Use this reference for baseline metrics; revise end times when frame-level DER is required."
            ),
        },
        "source": {
            "audio_path": str(audio.resolve()),
            "audio_sha256": digest(audio),
            "audio_duration_s": duration,
            "transcript_path": str(transcript.resolve()),
            "transcript_sha256": digest(transcript),
        },
        "audio_profile": audio_profile,
        "audio_profile_basis": (
            "single stereo file; no separate microphone/system sources were supplied; "
            "classified conservatively as ambient stereo"
        ),
        "window": {"start_s": window_start, "end_s": window_end,
                   "duration_s": window_end - window_start},
        "speakers": [
            {"id": speaker, "turns": counts[speaker],
             "duration_s": round(durations[speaker], 3),
             "reference_words": len(" ".join(grouped[speaker]).split())}
            for speaker in sorted(grouped)
        ],
        "turns": turns,
        "reference": {speaker: " ".join(grouped[speaker])
                      for speaker in sorted(grouped)},
        "hypothesis": None,
    }
    readable = "\n".join(
        f"[{turn['speaker']} {int(turn['start']) // 3600:02d}:"
        f"{int(turn['start']) % 3600 // 60:02d}:"
        f"{int(turn['start']) % 60:02d} -> {turn['end']:.3f}] {turn['text']}"
        for turn in turns
    ) + "\n"
    session_id = audio.stem + "_phase0"
    stm = "\n".join(
        f"{session_id} 1 {turn['speaker']} {turn['start']:.3f} "
        f"{turn['end']:.3f} {turn['text']}"
        for turn in turns
    ) + "\n"
    rttm = "\n".join(
        f"SPEAKER {session_id} 1 {turn['start']:.3f} "
        f"{turn['end'] - turn['start']:.3f} <NA> <NA> {turn['speaker']} <NA> <NA>"
        for turn in turns
    ) + "\n"
    return manifest, readable, stm, rttm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readable-out", type=Path)
    parser.add_argument("--stm-out", type=Path)
    parser.add_argument("--rttm-out", type=Path)
    parser.add_argument("--duration", type=float, required=True,
                        help="Durata audio misurata con ffprobe, in secondi")
    parser.add_argument("--audio-profile", choices=("stereo_ambient", "mic_system"),
                        required=True,
                        help="Profilo dichiarato prima della valutazione")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--verification-level",
                        choices=("imported_draft", "human_verified_text", "human_verified_timing_and_speakers"),
                        default="imported_draft",
                        help="Livello di verifica dell'annotazione (default: imported_draft)")
    args = parser.parse_args()
    manifest, readable, stm, rttm = build_manifest(
        args.audio, args.transcript, duration=args.duration,
        window_start=args.start, window_end=args.start + args.minutes * 60,
        audio_profile=args.audio_profile,
        verification_level=args.verification_level,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if args.readable_out:
        args.readable_out.parent.mkdir(parents=True, exist_ok=True)
        args.readable_out.write_text(readable, encoding="utf-8")
    for path, content in ((args.stm_out, stm), (args.rttm_out, rttm)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(f"Creato {args.out}: {len(manifest['turns'])} turni, "
          f"{len(manifest['speakers'])} speaker, {manifest['window']['duration_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
