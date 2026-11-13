"""Region analysis and conservative enhancement views for the recovery pass.

Indicators select candidates; they are never proof that recognition improves.
The only enhancement implemented here is a *static*, bounded gain followed by
a peak limiter:

- no noise gate (weak words are never removed);
- no time-varying automatic gain, so a loud speaker interrupting a quiet one
  does not make the quiet one pump down and up;
- the limiter acts only on samples above the ceiling, so the quiet voice keeps
  its boost while the loud voice's peaks are held.

Neural enhancement (e.g. DeepFilterNet) and dereverberation (e.g. WPE) are
reported as unavailable unless their packages are installed: nothing is
downloaded. In overlapping speech a single-speaker enhancer may suppress the
weaker voice as interference, so those views are never applied to overlaps
without separation.
"""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

import numpy as np

SR = 16000
FRAME = SR // 50          # 20 ms
MAX_GAIN_DB = 12.0        # bounded boost for weak speech
TARGET_SPEECH_DBFS = -26.0
CEILING = 0.89            # about -1 dBFS, below the pipeline's peak ceiling
# Indicative, uncalibrated thresholds (documented in the recovery report).
WEAK_RELATIVE_DB = 10.0   # this much below the recording's typical speech
WEAK_ABSOLUTE_DBFS = -40.0
NOISY_SNR_DB = 15.0
REVERB_TAIL_DB = -12.0


@dataclass(frozen=True)
class RegionAnalysis:
    rms_dbfs: float
    peak: float
    clip_fraction: float
    noise_floor_dbfs: float
    speech_level_dbfs: float
    snr_estimate_db: float
    active_fraction: float
    spectral_centroid_hz: float
    high_band_ratio: float
    tail_ratio_db: float

    def to_dict(self) -> dict:
        return {key: (round(value, 4) if np.isfinite(value) else None)
                for key, value in asdict(self).items()}


def _db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-9)))


def frame_levels_db(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """RMS of 20 ms frames in dBFS."""
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, round(sr / 50))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    frames = audio[: count * frame].reshape(count, frame).astype(np.float64)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-9))


def speech_level_db(levels: np.ndarray) -> tuple[float, float]:
    """(speech level, noise floor) from frame levels: no VAD model needed."""
    if levels.size == 0:
        return float("-inf"), float("-inf")
    noise = float(np.percentile(levels, 10))
    active = levels[levels > noise + 6.0]
    speech = float(np.median(active)) if active.size else float(np.median(levels))
    return speech, noise


def analyze_region(audio: np.ndarray, sr: int = SR) -> RegionAnalysis:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        nan = float("nan")
        return RegionAnalysis(nan, 0.0, 0.0, nan, nan, nan, 0.0, nan, nan, nan)
    levels = frame_levels_db(audio, sr)
    speech, noise = speech_level_db(levels)
    active = levels > noise + 6.0 if levels.size else np.zeros(0, dtype=bool)
    peak = float(np.max(np.abs(audio)))
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    total = float(spectrum.sum()) or 1e-12
    centroid = float((freqs * spectrum).sum() / total)
    high = float(spectrum[freqs >= 4000].sum() / total)
    # Reverberation proxy: energy 60-200 ms after speech offsets, relative to
    # the speech before them. Indicative only (no RT60 estimate).
    tails = []
    for index in np.flatnonzero(active[:-1] & ~active[1:]) if active.size > 1 else []:
        before = levels[max(0, index - 4): index + 1]
        after = levels[index + 3: index + 10]
        if before.size and after.size:
            tails.append(float(np.mean(after) - np.mean(before)))
    return RegionAnalysis(
        rms_dbfs=_db(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))),
        peak=peak,
        clip_fraction=float(np.mean(np.abs(audio) >= 0.999)),
        noise_floor_dbfs=noise,
        speech_level_dbfs=speech,
        snr_estimate_db=speech - noise,
        active_fraction=float(active.mean()) if active.size else 0.0,
        spectral_centroid_hz=centroid,
        high_band_ratio=high,
        tail_ratio_db=float(np.median(tails)) if tails else float("nan"),
    )


def classify_region(analysis: RegionAnalysis, recording_speech_dbfs: float) -> list[str]:
    """Problem labels used to pick candidates and views, never as quality proof."""
    labels = []
    if np.isfinite(analysis.speech_level_dbfs) and (
            analysis.speech_level_dbfs < WEAK_ABSOLUTE_DBFS
            or (np.isfinite(recording_speech_dbfs)
                and analysis.speech_level_dbfs < recording_speech_dbfs - WEAK_RELATIVE_DB)):
        labels.append("weak_speech")
    if np.isfinite(analysis.snr_estimate_db) and analysis.snr_estimate_db < NOISY_SNR_DB:
        labels.append("noise")
    if np.isfinite(analysis.tail_ratio_db) and analysis.tail_ratio_db > REVERB_TAIL_DB:
        labels.append("reverberation")
    if analysis.clip_fraction > 0.001:
        labels.append("clipping")
    return labels


def bounded_gain(audio: np.ndarray, *, max_gain_db: float = MAX_GAIN_DB,
                 target_dbfs: float = TARGET_SPEECH_DBFS) -> tuple[np.ndarray, dict]:
    """Static gain towards the quieter speech of the region, then a peak limiter.

    The level is taken from the 25th percentile of active frames, so a quiet
    voice is not ignored because a louder one dominates the median.
    """
    audio = np.asarray(audio, dtype=np.float32)
    levels = frame_levels_db(audio)
    _speech, noise = speech_level_db(levels)
    active = levels[levels > noise + 6.0] if levels.size else levels
    quiet = float(np.percentile(active, 25)) if active.size else float("-inf")
    gain_db = 0.0 if not np.isfinite(quiet) else float(np.clip(target_dbfs - quiet, 0.0, max_gain_db))
    boosted = audio.astype(np.float64) * (10 ** (gain_db / 20))
    limited = _soft_limit(boosted, CEILING)
    config = {"method": "static_bounded_gain+peak_limiter", "gain_db": round(gain_db, 2),
              "max_gain_db": max_gain_db, "target_dbfs": target_dbfs, "ceiling": CEILING,
              "limited_fraction": round(float(np.mean(np.abs(boosted) > CEILING)), 5),
              "noise_gate": False, "time_varying_gain": False}
    return limited.astype(np.float32), config


def _soft_limit(signal: np.ndarray, ceiling: float) -> np.ndarray:
    """Identity below the knee, smooth saturation above: only peaks change."""
    knee = ceiling * 0.8
    out = signal.copy()
    over = np.abs(signal) > knee
    if over.any():
        excess = (np.abs(signal[over]) - knee) / (ceiling - knee)
        out[over] = np.sign(signal[over]) * (knee + (ceiling - knee) * np.tanh(excess))
    return out


def neural_enhancer_status() -> tuple[bool, str]:
    if importlib.util.find_spec("df") is not None:
        return True, "DeepFilterNet installato (non usato automaticamente: richiede A/B su parlato verificato)"
    return False, "DeepFilterNet non installato; nessun download automatico"


def dereverberation_status() -> tuple[bool, str]:
    if importlib.util.find_spec("nara_wpe") is not None:
        return True, "WPE installato (non usato automaticamente: richiede A/B su parlato verificato)"
    return False, "WPE (nara_wpe) non installato; nessun download automatico"
