"""Reproducible, streaming diagnostics for imported recordings.

The source and its channels remain untouched. Energy, cached ASR, and cached
speaker overlaps are diagnostic hints, never reference annotations. No quality
metric is calculated against machine-generated text.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import paths
from .checkpoints import file_digest


@dataclass(frozen=True)
class DiagnosticOptions:
    window_s: float = 1.0
    segment_s: float = 30.0
    context_s: float = 3.0
    silence_dbfs: float = -65.0
    seed: int = 1729
    overlap_regions: int = 8
    degraded_regions: int = 4
    control_regions: int = 4

    def validate(self) -> None:
        if not all(math.isfinite(x) for x in (
            self.window_s, self.segment_s, self.context_s, self.silence_dbfs
        )):
            raise ValueError("Diagnostic options must be finite")
        if not 0.1 <= self.window_s <= 10 or not 20 <= self.segment_s <= 60:
            raise ValueError("Use 0.1–10 second windows and 20–60 second segments")
        if not 0 <= self.context_s <= 30:
            raise ValueError("Context must be between 0 and 30 seconds")
        if any(n < 0 or n > 64 for n in (
            self.overlap_regions, self.degraded_regions, self.control_regions
        )):
            raise ValueError("Each region target must be between 0 and 64")


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _db(values: np.ndarray) -> list[float]:
    return [float(20 * np.log10(max(float(value), 1e-10))) for value in values]


def _probe(source: Path) -> dict:
    info = sf.info(source)
    result = {"duration_s": info.duration, "sample_rate": info.samplerate,
              "channels": info.channels, "frames": info.frames,
              "codec": info.subtype, "container": info.format,
              "approximate_file_bitrate": source.stat().st_size * 8 / max(info.duration, 1e-9)}
    bundled = Path(sys.prefix) / "Library" / "bin" / "ffprobe.exe"
    executable = str(bundled) if bundled.is_file() else shutil.which("ffprobe")
    if executable:
        try:
            process = subprocess.run(
                [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
                capture_output=True, text=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if process.returncode == 0:
                result["ffprobe"] = json.loads(process.stdout)
            else:
                result["probe_error"] = process.stderr[-4000:]
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result["probe_error"] = str(exc)
    return result


def _cached_segments(session_path: Path | None) -> list[dict]:
    if session_path is None:
        return []
    session = json.loads(session_path.read_text(encoding="utf-8"))
    rows = session.get("utterances") or session.get("segments", [])
    result = []
    for item in rows:
        start = float(item.get("start_s", item.get("start", 0)))
        end = float(item.get("end_s", item.get("end", start)))
        if not (math.isfinite(start) and math.isfinite(end) and end > start >= 0):
            continue
        result.append({"start_s": start, "end_s": end,
                       "speaker": str(item.get("speaker", "unknown")),
                       "text": str(item.get("text", ""))})
    return sorted(result, key=lambda item: (item["start_s"], item["end_s"]))


def find_cached_session(source: str | Path) -> Path | None:
    """Find a cached transcript by source filename, without assuming topology."""
    source = Path(source)
    matches = []
    for candidate in paths.sessions_dir().glob("*.json"):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if any(source.stem in str(p) for p in value.get("audio", {}).values() if p):
                matches.append(candidate)
        except (OSError, ValueError, TypeError):
            continue
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _coverage(rows: list[dict], start: float, end: float) -> float:
    cursor, duration = start, 0.0
    for row in rows:
        left, right = max(start, row["start_s"]), min(end, row["end_s"])
        if right > max(left, cursor):
            duration += right - max(left, cursor)
            cursor = right
    return duration


def _cached_overlap(rows: list[dict], start: float, end: float) -> bool:
    for index, row in enumerate(rows):
        for other in rows[index + 1:]:
            if other["start_s"] >= min(end, row["end_s"]):
                break
            if (other["speaker"] != row["speaker"]
                    and min(row["end_s"], other["end_s"], end)
                    > max(row["start_s"], other["start_s"], start)):
                return True
    return False


def _scan(source: Path, decoded: Path, segments: list[dict], options: DiagnosticOptions) -> dict:
    """Read every frame using bounded memory and retain native channel samples."""
    windows, decode_errors = [], []
    with sf.SoundFile(source) as reader:
        rate, channels = reader.samplerate, reader.channels
        blocksize = max(1, round(rate * options.window_s))
        power = np.zeros(channels, dtype=np.float64)
        sums = np.zeros(channels, dtype=np.float64)
        peak = np.zeros(channels)
        clipped = np.zeros(channels, dtype=np.int64)
        products = 0.0
        count = 0
        # RF64 supports recordings longer than WAV's 4 GiB size limit.
        with sf.SoundFile(decoded, "w", samplerate=rate, channels=channels,
                          format="RF64", subtype="FLOAT") as writer:
            try:
                while True:
                    block = reader.read(blocksize, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    if not np.isfinite(block).all():
                        raise ValueError("Decoder produced nonfinite samples")
                    writer.write(block)
                    squares = np.sum(np.square(block, dtype=np.float64), axis=0)
                    block_peak = np.max(np.abs(block), axis=0)
                    block_clip = np.sum(np.abs(block) >= .999, axis=0)
                    start, end = count / rate, (count + len(block)) / rate
                    current = [s for s in segments if s["end_s"] > start and s["start_s"] < end]
                    rms = _db(np.sqrt(squares / len(block)))
                    conditions = []
                    if max(rms) < options.silence_dbfs:
                        conditions.append("silence_candidate")
                    elif max(rms) < -40:
                        conditions.append("weak_signal_candidate")
                    if np.any(block_clip):
                        conditions.append("clipping")
                    if _cached_overlap(current, start, end):
                        conditions.append("cached_speaker_overlap_candidate")
                    coverage = _coverage(current, start, end)
                    if max(rms) > -50 and coverage < (end - start) * .1:
                        conditions.append("signal_without_cached_text")
                    if windows and abs(max(rms) - max(windows[-1]["rms_dbfs"])) > 30:
                        conditions.append("level_discontinuity_candidate")
                    if len({s["speaker"] for s in current}) > 1:
                        conditions.append("cached_speaker_change")
                    windows.append({"start_s": start, "end_s": end, "rms_dbfs": rms,
                                    "peak": block_peak.tolist(),
                                    "clipping_ratio": (block_clip / len(block)).tolist(),
                                    "conditions": conditions,
                                    "cached_transcript_coverage_s": coverage})
                    power += squares
                    sums += np.sum(block, axis=0, dtype=np.float64)
                    peak = np.maximum(peak, block_peak)
                    clipped += block_clip
                    if channels == 2:
                        products += float(np.dot(block[:, 0].astype(np.float64), block[:, 1]))
                    count += len(block)
            except (RuntimeError, ValueError, OSError) as exc:
                decode_errors.append({"at_s": count / rate, "error": str(exc)})
        if count != reader.frames:
            decode_errors.append({"error": "Decoded frame count differs from header",
                                  "expected_frames": reader.frames, "decoded_frames": count})
    if not count:
        raise ValueError("No audio frames decoded")
    correlation = None
    if channels == 2:
        centered = power - sums * sums / count
        denominator = math.sqrt(max(0, float(centered[0] * centered[1])))
        if denominator > 1e-12:
            correlation = max(-1., min(1., (products - sums[0] * sums[1] / count) / denominator))
    return {"decoded_frames": count, "decoded_duration_s": count / rate,
            "sample_rate": rate, "channels": channels,
            "rms_dbfs_per_channel": _db(np.sqrt(power / count)),
            "peak_per_channel": peak.tolist(), "clipping_ratio_per_channel": (clipped / count).tolist(),
            "channel_correlation": correlation,
            "channel_policy": "Native channels preserved; topology is not inferred from stereo",
            "decode_errors": decode_errors, "timeline": windows}


def select_benchmark_regions(timeline: list[dict], segments: list[dict], duration: float,
                             options: DiagnosticOptions) -> list[dict]:
    """Select time-distributed hints plus seeded random controls, without quotas."""
    selected = []

    def add(window: dict, reason: str) -> bool:
        center = (window["start_s"] + window["end_s"]) / 2
        start = max(0, min(center - options.segment_s / 2, duration - options.segment_s))
        end = min(duration, start + options.segment_s)
        if any(min(end, r["end_s"]) - max(start, r["start_s"]) > (end - start) * .5
               for r in selected):
            return False
        rows = [s for s in segments if s["end_s"] > start and s["start_s"] < end]
        conditions = sorted({c for w in timeline if w["end_s"] > start and w["start_s"] < end
                             for c in w["conditions"]})
        selected.append({"region_id": f"region-{len(selected) + 1:02d}",
                         "start_s": start, "end_s": end,
                         "context_start_s": max(0, start - options.context_s),
                         "context_end_s": min(duration, end + options.context_s),
                         "selection_reason": reason, "conditions": conditions,
                         "existing_transcript": rows,
                         "preprocessing": {"raw": True, "channels": "preserve", "filters": []},
                         "reference": {"verification_level": "unverified_candidate", "turns": rows}})
        return True

    groups = [
        ("cached_speaker_overlap_candidate", options.overlap_regions,
         [w for w in timeline if "cached_speaker_overlap_candidate" in w["conditions"]]),
        ("degraded_or_omitted_signal_candidate", options.degraded_regions,
         [w for w in timeline if any(c in w["conditions"] for c in
             ("clipping", "weak_signal_candidate", "signal_without_cached_text"))]),
    ]
    for reason, target, candidates in groups:
        # Start from equal temporal strata, then fill with remaining candidates.
        if not candidates or not target:
            continue
        priority = [candidates[min(len(candidates) - 1, int((i + .5) * len(candidates) / target))]
                    for i in range(target)] + candidates
        accepted = 0
        for window in priority:
            if add(window, reason):
                accepted += 1
                if accepted >= target:
                    break
    rng = random.Random(options.seed)
    clean = [w for w in timeline if not any(c in w["conditions"] for c in
              ("clipping", "silence_candidate", "weak_signal_candidate", "cached_speaker_overlap_candidate"))]
    rng.shuffle(clean)
    random_windows = list(timeline)
    rng.shuffle(random_windows)
    targets = [("clean_control_candidate", max(0, options.control_regions - 2), clean),
               ("seeded_random_control", min(2, options.control_regions), random_windows)]
    for reason, target, candidates in targets:
        accepted = 0
        for window in candidates:
            if not target:
                break
            if add(window, reason):
                accepted += 1
                if accepted >= target:
                    break
    return sorted(selected, key=lambda r: r["start_s"])


def diagnose_recording(source: str | Path, *, output_dir: str | Path | None = None,
                       session_path: str | Path | None = None,
                       options: DiagnosticOptions | None = None) -> Path:
    """Scan an entire recording and export lossless samples and a review manifest."""
    start_time = time.perf_counter()
    source = Path(source).resolve(strict=True)
    options = options or DiagnosticOptions()
    options.validate()
    source_hash = file_digest(source)
    session = Path(session_path).resolve(strict=True) if session_path else find_cached_session(source)
    baseline_hash = file_digest(session) if session else None
    config = {"version": 1, "options": asdict(options), "source_sha256": source_hash,
              "cached_session_sha256": baseline_hash}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root = Path(output_dir).resolve() if output_dir else paths.cache_dir() / "diagnostics" / key[:20]
    if root == source.parent or source == root or source in root.parents:
        raise ValueError("Diagnostic output must be separate from source audio")
    install_root = paths.app_root().resolve()
    if paths.is_installed_mode() and (root == install_root or install_root in root.parents):
        raise ValueError("Diagnostic artifacts must be outside the installation directory")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    decoded = root / "decoded-original.wav"
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if (cached.get("configuration") == config and decoded.is_file()
                and (root / "timeline.json").is_file() and (root / "reference-review.json").is_file()
                and all((root / r["audio_file"]).is_file() for r in cached.get("regions", []))):
            try:
                if sf.info(decoded).frames == cached["scan"]["decoded_frames"]:
                    return report_path
            except (OSError, RuntimeError, KeyError):
                pass
    segments = _cached_segments(session)
    pending_decode = root / "decoded-original.tmp.wav"
    scan = _scan(source, pending_decode, segments, options)
    os.replace(pending_decode, decoded)
    regions = select_benchmark_regions(scan["timeline"], segments, scan["decoded_duration_s"], options)
    with sf.SoundFile(decoded) as reader:
        for region in regions:
            left = round(region["context_start_s"] * reader.samplerate)
            right = round(region["context_end_s"] * reader.samplerate)
            reader.seek(left)
            samples = reader.read(right - left, dtype="float32", always_2d=True)
            region["audio_file"] = region["region_id"] + ".wav"
            region["sample_origin_s"] = left / reader.samplerate
            sf.write(root / region["audio_file"], samples, reader.samplerate, subtype="FLOAT")
    timeline = scan.pop("timeline")
    _write_json(root / "timeline.json", timeline)
    report = {"schema_version": 1, "source": str(source), "source_sha256": source_hash,
              "configuration": config, "metadata": _probe(source), "scan": scan,
              "cached_session": str(session) if session else None,
              "cached_session_association": ("explicit_session_unverified_source_hash" if session_path
                                             else "filename_match_unverified_source_hash") if session else None,
              "decoded_audio": str(decoded), "timeline": "timeline.json", "regions": regions,
              "runtime_s": time.perf_counter() - start_time,
              "quality_metrics": None,
              "limitations": [
                  "Energy marks signal activity, not verified speech or speaker count.",
                  "Cached ASR, speaker changes, and overlaps are model predictions, not ground truth.",
                  "Missing cached text and level discontinuities require listening; noise and natural pauses can trigger them.",
                  "An imported MP3 cannot establish capture topology or recover speech removed before encoding.",
                  "No WER or DER is valid until a human verifies the reference text and speaker timing.",
              ]}
    _write_json(root / "reference-review.json", {
        "source_sha256": source_hash, "verification_level": "unverified_candidate",
        "instructions": "Correct turns against original playback. Keep overlapping turns and absolute timestamps. "
                        "Mark human_verified_text or human_verified_timing_and_speakers only after review.",
        "regions": [{"region_id": r["region_id"], "start_s": r["start_s"], "end_s": r["end_s"],
                     **r["reference"]} for r in regions],
    })
    _write_json(report_path, report)
    return report_path
