"""Non-destructive correction of measured, stable capture-clock drift.

Native tracks remain immutable. Reliable clock slopes produce source-local,
multichannel derivatives at the same nominal sample rate. Explicit source-start
offsets remain separate. Poor clocks never trigger a guessed time warp.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import firwin, resample_poly

from .. import paths

_OUTPUT_BLOCK = 32768


@dataclass
class SynchronizedCapture:
    mic_path: Path | None
    system_path: Path | None
    report: dict


def clock_diagnostics(metadata: dict | None, source: str) -> dict:
    """Fit sample-clock rate while separating read jitter and fixed read latency."""
    report = {"status": "insufficient_timing_anchors", "review_required": False,
              "correction_applied": False, "uncertainty_seconds": None,
              "start_latency_measured": False, "reliable_scale": False}
    try:
        track = (metadata or {}).get("tracks", {}).get(source, {})
        anchors = track.get("timing", [])
        report["timing_source"] = track.get("timing_source", "unspecified")
        fallbacks = track.get("timestamp_fallbacks", 0)
        if isinstance(fallbacks, bool) or not isinstance(fallbacks, int) or fallbacks < 0:
            raise ValueError("Invalid native timestamp fallback count")
        report["timestamp_fallbacks"] = fallbacks
        native_clock = report["timing_source"] in ("wasapi_adc_clock", "blocking_read_clock_estimate") and fallbacks == 0
        # Clock trust is independent of recording length. A short or silent
        # capture with estimated timestamps still needs explicit review.
        if not native_clock:
            report.update(status="unverified_capture_clock", review_required=True)
        pairs = np.asarray([(float(item["audio_seconds"]), float(item["clock_seconds"]))
                            for item in anchors], dtype=np.float64)
    except (AttributeError, KeyError, TypeError, ValueError):
        report.update(status="invalid_timing_anchors", review_required=True)
        return report
    if not len(pairs):
        return report
    if not np.all(np.isfinite(pairs)) or np.any(np.diff(pairs[:, 0]) <= 0) or np.any(np.diff(pairs[:, 1]) <= 0):
        report.update(status="invalid_timing_anchors", review_required=True)
        return report
    span = float(pairs[-1, 0] - pairs[0, 0])
    report.update(anchor_count=len(pairs), measured_span_seconds=span,
                  last_anchor_audio_seconds=float(pairs[-1, 0]))
    if len(pairs) < 6 or span < 30:
        return report
    # Bound robust pairwise fitting even when multi-hour captures have many anchors.
    bounded = pairs[np.linspace(0, len(pairs) - 1, min(64, len(pairs))).astype(int)]
    slopes = [(right[1] - left[1]) / (right[0] - left[0])
              for index, left in enumerate(bounded) for right in bounded[index + 1:]
              if right[0] - left[0] >= span / 4]
    slope = float(np.median(slopes))
    residual = bounded[:, 1] - slope * bounded[:, 0]
    jitter = float(np.quantile(np.abs(residual - np.median(residual)), 0.95))
    drift = (slope - 1.0) * span
    reliable = native_clock and jitter <= 0.005 and 0.99 <= slope <= 1.01
    review = abs(drift) > max(0.05, jitter * 3) or jitter > 0.05 or not native_clock or not 0.99 <= slope <= 1.01
    report.update(
        status="synchronization_review_required" if review else "no_significant_drift_measured",
        review_required=review, reliable_scale=reliable, estimated_scale=slope,
        estimated_drift_ppm=(slope - 1.0) * 1_000_000,
        estimated_drift_seconds=drift, uncertainty_seconds=jitter * 3,
        # Read completion contains device/buffer latency, so this intercept is
        # diagnostic only. Start offsets continue to use the shared capture clock.
        fitted_read_latency_seconds=float(np.median(residual)),
    )
    return report


def resample_capture_clock(source: str | Path, destination: str | Path, scale: float) -> tuple[Path, dict]:
    """Stream a continuous polyphase resample without changing native channels.

    Each bounded window begins on an exact rational phase boundary and includes
    FIR context. Cropping restores global output positions, preventing per-block
    rounding drift or seams. The original file is never a valid destination.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Clock derivative must not overwrite an original track")
    if not np.isfinite(scale) or not 0.99 <= scale <= 1.01:
        raise ValueError("Capture-clock scale must be finite and within 1%")
    info = sf.info(str(source))
    ratio = Fraction(float(scale)).limit_denominator(100000)
    up, down = ratio.numerator, ratio.denominator
    effective = up / down
    approximation_seconds = abs(effective - scale) * info.duration
    if approximation_seconds > 0.005:
        raise ValueError("Rational clock approximation exceeds 5 ms over the recording")
    _stream_resample(source, destination, up, down, sample_rate=info.samplerate)
    return destination, {
        "backend": "scipy_polyphase_with_phase_aligned_streaming_context",
        "requested_scale": float(scale), "applied_scale": effective,
        "ratio": [up, down], "approximation_error_seconds": approximation_seconds,
        "original_frames": info.frames, "output_frames": round(info.frames * effective),
        "sample_rate": info.samplerate, "channels": info.channels,
    }


def _stream_resample(source, destination, up, down, *, sample_rate, mono=False, offset_s=0.0):
    """Reuse phase-aligned FIR windows for clock and playback derivatives."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Audio derivative must not overwrite an original track")
    info = sf.info(str(source))
    maximum = max(up, down)
    # Match scipy.signal.resample_poly's default Kaiser low-pass. Build once,
    # then reuse for all windows instead of fitting independent block filters.
    if up != down:
        kernel = firwin(20 * maximum + 1, 1.0 / maximum, window=("kaiser", 5.0)).astype(np.float32)
    else:
        kernel = None
    context = max(32, (10 * maximum + up - 1) // up + 4)
    output_frames = round(info.frames * up / down)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".wav", delete=False) as pending:
        temporary = Path(pending.name)
    try:
        with sf.SoundFile(str(source)) as reader, sf.SoundFile(
            str(temporary), "w", samplerate=sample_rate, channels=1 if mono else info.channels, subtype="FLOAT"
        ) as writer:
            leading = max(0, round(offset_s * sample_rate))
            while leading:
                count = min(leading, _OUTPUT_BLOCK)
                writer.write(np.zeros((count, 1 if mono else info.channels), dtype=np.float32))
                leading -= count
            for first in range(0, output_frames, _OUTPUT_BLOCK):
                last = min(output_frames, first + _OUTPUT_BLOCK)
                # A multiple of down has an integer global output position.
                left = ((first * down // up - context) // down) * down
                right = (last * down + up - 1) // up + context
                valid_left, valid_right = max(0, left), min(info.frames, right)
                reader.seek(valid_left)
                data = reader.read(max(0, valid_right - valid_left), dtype="float32", always_2d=True)
                if left < 0 or right > info.frames:
                    data = np.pad(data, ((max(0, -left), max(0, right - info.frames)), (0, 0)))
                converted = resample_poly(data, up, down, axis=0, window=kernel) if kernel is not None else data
                offset = first - left * up // down
                block = converted[offset:offset + last - first]
                if len(block) != last - first:
                    raise RuntimeError("Incomplete capture-clock derivative")
                writer.write(block.mean(axis=1, keepdims=True) if mono else block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def create_solo_playback(source, destination, offset_s):
    """Global-zero, mono 16 kHz playback view; native originals stay separate."""
    info = sf.info(str(source))
    ratio = Fraction(16000, info.samplerate)
    return _stream_resample(source, destination, ratio.numerator, ratio.denominator,
                            sample_rate=16000, mono=True, offset_s=offset_s)


def synchronize_capture_sources(mic_path, system_path, *, offsets_s=None,
                                metadata: dict | None = None, work_dir=None,
                                make_playback=True) -> SynchronizedCapture:
    """Prepare one common sample-clock timeline while preserving native originals."""
    originals = {source: Path(path) for source, path in (("mic", mic_path), ("system", system_path)) if path is not None}
    processing = dict(originals)
    protected = {original.resolve() for original in originals.values()}
    offsets = iter(offsets_s or [])
    source_reports = {}
    output_root = Path(work_dir) if work_dir is not None else None
    for source, original in originals.items():
        offset = float(next(offsets, 0.0))
        diagnostic = clock_diagnostics(metadata, source)
        entry = {"original": str(original), "processing_audio": str(original), "offset_s": offset,
                 "status": "native_timeline", "clock_diagnostics": diagnostic,
                 "scale": 1.0, "correction_applied": False}
        info = sf.info(str(original))
        span = diagnostic.get("measured_span_seconds", 0.0)
        if diagnostic.get("last_anchor_audio_seconds", 0.0) > info.duration + 0.25:
            diagnostic.update(status="timing_anchors_exceed_original", reliable_scale=False, review_required=True)
        projected_uncertainty = diagnostic.get("uncertainty_seconds")
        if projected_uncertainty is not None:
            projected_uncertainty *= max(1.0, info.duration / max(span, 1.0))
        entry["projected_uncertainty_seconds"] = projected_uncertainty
        scale = diagnostic.get("estimated_scale", 1.0)
        full_drift = abs(scale - 1.0) * info.duration
        if diagnostic.get("reliable_scale") and projected_uncertainty is not None and projected_uncertainty <= 0.02 and full_drift > max(0.02, projected_uncertainty * 3):
            if output_root is None:
                root = paths.work_dir()
                root.mkdir(parents=True, exist_ok=True)
                output_root = Path(tempfile.mkdtemp(prefix="capture-sync-", dir=root))
            destination = output_root / f"{source}_clock_synchronized.wav"
            try:
                if destination.resolve() in protected:
                    raise ValueError("Synchronization derivative collides with an original source")
                derivative, details = resample_capture_clock(original, destination, scale)
                processing[source] = derivative
                entry.update(status="synchronized_derivative", processing_audio=str(derivative),
                             scale=details["applied_scale"], correction_applied=True, resampling=details)
                diagnostic.update(status="clock_drift_corrected", review_required=False, correction_applied=True)
            except (OSError, RuntimeError, ValueError) as exc:
                entry.update(status="synchronization_review_required", warning=str(exc))
                diagnostic.update(review_required=True, correction_applied=False)
        elif diagnostic["review_required"]:
            entry["status"] = "synchronization_review_required"
        if make_playback and (metadata or offset != 0.0 or entry["correction_applied"]):
            if output_root is None:
                root = paths.work_dir()
                root.mkdir(parents=True, exist_ok=True)
                output_root = Path(tempfile.mkdtemp(prefix="capture-sync-", dir=root))
            try:
                destination = output_root / f"{source}_synchronized_playback.wav"
                if destination.resolve() in protected:
                    raise ValueError("Playback derivative collides with an original source")
                playback = create_solo_playback(processing[source], destination, offset)
                entry["playback_audio"] = str(playback)
            except (OSError, RuntimeError, ValueError) as exc:
                entry["playback_warning"] = str(exc)
                diagnostic["review_required"] = True
        source_reports[source] = entry
    report = {
        "timebase": "common_capture_clock_with_explicit_start_offsets",
        "sources": source_reports,
        "clock_diagnostics": {source: entry["clock_diagnostics"] for source, entry in source_reports.items()},
        "review_required": any(entry["clock_diagnostics"]["review_required"] for entry in source_reports.values()),
        "uncertain": any(not entry["clock_diagnostics"].get("reliable_scale") for entry in source_reports.values()),
    }
    return SynchronizedCapture(processing.get("mic"), processing.get("system"), report)
