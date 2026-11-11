"""Conservative, reversible evaluation of a delayed playback reference.

This is a bounded offline linear echo experiment, not a general room AEC.
Only stable, nearly reference-only calibration windows permit subtraction.
Coefficients stay frozen. Unexpected residual energy bypasses cancellation,
including double-talk, room changes, clipping and nonlinear echo paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly

_SR = 16000


@dataclass
class EchoResult:
    audio_path: Path
    report: dict


def _window(path: Path, start_s: float, seconds: float) -> np.ndarray:
    """Read a bounded mono reference window; pad unavailable timeline with zero."""
    count = max(0, round(seconds * _SR))
    result = np.zeros(count, dtype=np.float32)
    info = sf.info(str(path))
    stop_s = start_s + seconds
    if stop_s <= 0 or start_s >= info.duration:
        return result
    begin = max(0, round(start_s * info.samplerate))
    end = min(info.frames, round(stop_s * info.samplerate))
    if end <= begin:
        return result
    data, rate = sf.read(str(path), start=begin, stop=end, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != _SR:
        divisor = gcd(rate, _SR)
        mono = resample_poly(mono, _SR // divisor, rate // divisor)
    offset = max(0, round(-start_s * _SR))
    n = min(len(mono), count - offset)
    if n > 0:
        result[offset:offset + n] = mono[:n]
    return result


def evaluate_echo(
    mic_path: str | Path,
    system_path: str | Path,
    output_path: str | Path,
    *,
    mic_offset: float = 0.0,
    system_offset: float = 0.0,
    apply: bool = False,
) -> EchoResult:
    """Measure delay and evaluate a derivative with frozen, double-talk-safe gating.

    Different device rates and start offsets are resolved on the common timeline.
    Originals are never modified. Uncertain windows remain exactly the raw signal
    at the working sample rate. No transcript hypotheses are discarded here.
    """
    mic, system, destination = Path(mic_path), Path(system_path), Path(output_path)
    if destination.resolve() in (mic.resolve(), system.resolve()):
        raise ValueError("Echo derivative must not overwrite an original track")
    duration = sf.info(str(mic)).duration
    difference = float(mic_offset) - float(system_offset)
    calibration = []
    # Spread bounded calibration across the recording, including long sessions.
    for start in np.linspace(0.0, max(0.0, duration - 1.0), min(32, max(1, int(duration)))):
        x = _window(system, float(start) + difference, 1.0)
        y = _window(mic, float(start), 1.0)
        energy = float(np.dot(x, x))
        y_energy = float(np.dot(y, y))
        if min(energy, y_energy) < 1e-7:
            continue
        correlations = correlate(y, x, method="fft")
        lags = correlation_lags(len(y), len(x))
        valid = (lags >= 0) & (lags <= round(0.25 * _SR))
        eligible = np.flatnonzero(valid)
        peak = eligible[int(np.argmax(np.abs(correlations[valid])))]
        lag = int(lags[peak])
        aligned = _window(system, float(start) + difference - lag / _SR, 1.0)
        power = float(np.dot(aligned, aligned))
        if power < 1e-7:
            continue
        gain = float(np.dot(y, aligned) / power)
        residual = y - gain * aligned
        residual_power = float(np.mean(np.square(residual, dtype=np.float64)))
        coherence = 1.0 - residual_power / max(y_energy / len(y), 1e-12)
        # Even moderately mixed speech is insufficient evidence to fit an echo path.
        if coherence >= 0.995 and 0.001 <= abs(gain) <= 4.0:
            calibration.append((lag, gain, residual_power, coherence))

    report = {
        "backend": "conservative_delayed_reference",
        "requested": bool(apply),
        "original_mic": str(mic),
        "original_system": str(system),
        "calibration_windows": len(calibration),
        "status": "insufficient_reference_only_calibration",
        "double_talk_protection": True,
        "cancelled_seconds": 0.0,
        "bypassed_seconds": 0.0,
        "limitations": "Linear stable echo only; uncertain/reverberant windows bypass cancellation.",
    }
    if len(calibration) < 2:
        return EchoResult(mic, report)
    lag = int(np.median([item[0] for item in calibration]))
    gain = float(np.median([item[1] for item in calibration]))
    stable = [item for item in calibration
              if abs(item[0] - lag) <= 2 and abs(item[1] - gain) <= max(0.01, abs(gain) * 0.05)]
    if len(stable) < 2 or len(stable) < len(calibration) * 0.8:
        report["status"] = "unstable_echo_path"
        return EchoResult(mic, report)
    noise_power = float(np.median([item[2] for item in stable]))
    report.update(delay_s=lag / _SR, gain=gain, status="measured_only")
    if not apply:
        return EchoResult(mic, report)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = round(duration * _SR)
    # Streaming output keeps memory bounded for long capture sessions.
    with sf.SoundFile(str(destination), "w", samplerate=_SR, channels=1, subtype="FLOAT") as output:
        for begin in range(0, frames, _SR):
            count = min(_SR, frames - begin)
            start_s = begin / _SR
            raw = _window(mic, start_s, count / _SR)
            reference = _window(system, start_s + difference - lag / _SR, count / _SR)
            predicted = gain * reference
            residual = raw - predicted
            cleaned = raw.copy()
            # A 20 ms gate never updates the learned path during local speech.
            for offset in range(0, count, _SR // 50):
                stop = min(count, offset + _SR // 50)
                predicted_power = float(np.mean(np.square(predicted[offset:stop], dtype=np.float64)))
                residual_power = float(np.mean(np.square(residual[offset:stop], dtype=np.float64)))
                safe = predicted_power > 1e-10 and residual_power <= max(noise_power * 16, predicted_power * 0.01, 1e-10)
                seconds = (stop - offset) / _SR
                if safe:
                    cleaned[offset:stop] = residual[offset:stop]
                    report["cancelled_seconds"] += seconds
                else:
                    report["bypassed_seconds"] += seconds
            output.write(cleaned)
    report.update(status="derivative_created", derivative=str(destination))
    return EchoResult(destination, report)
