"""Bounded-memory, channel-preserving waveform pyramid for long recordings."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


class WaveformPeaks:
    def __init__(self, path, cache_dir=None):
        self.path = Path(path)
        info = sf.info(str(self.path))
        self.duration_s = info.duration
        self.step_s = .05
        stat = self.path.stat()
        key = hashlib.sha256(f"peaks-v1:{self.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        cache = Path(cache_dir) / f"{key}.npz" if cache_dir else None
        if cache is not None and cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as value:
                    self.low, self.high, self.rms = value["low"], value["high"], value["rms"]
                return
            except (OSError, ValueError, KeyError):
                pass
        lows, highs, energies = [], [], []
        width = max(1, round(info.samplerate * self.step_s))
        self.step_s = width / info.samplerate
        with sf.SoundFile(str(self.path)) as reader:
            while True:
                block = reader.read(width * 200, dtype="float32", always_2d=True)
                if not len(block):
                    break
                for left in range(0, len(block), width):
                    piece = block[left:left + width]
                    lows.append(piece.min(axis=0))
                    highs.append(piece.max(axis=0))
                    energies.append(np.sqrt(np.mean(piece.astype(np.float64) ** 2, axis=0)))
        self.low, self.high, self.rms = (np.asarray(x, dtype=np.float32) for x in (lows, highs, energies))
        if cache is not None:
            import os
            import tempfile
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".npz", dir=cache.parent, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, low=self.low, high=self.high, rms=self.rms)
                os.replace(temporary, cache)
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, start_s, end_s, bins):
        bins = max(1, int(bins))
        if end_s <= start_s or not len(self.low):
            return []
        # Min/max across native channels cannot cancel opposite-phase voices.
        edges = np.linspace(start_s / self.step_s, end_s / self.step_s, bins + 1).astype(int)
        result = []
        for left, right in zip(edges, edges[1:]):
            if left >= len(self.low) or right < 0:
                result.append((0., 0.))
                continue
            left = max(0, min(len(self.low) - 1, left))
            right = max(left + 1, min(len(self.low), right))
            result.append((float(self.low[left:right].min()), float(self.high[left:right].max())))
        return result

    def activity_candidates(self, threshold_dbfs=-45.0):
        """Energy candidates, not verified speech or a neural VAD prediction."""
        active = np.max(self.rms, axis=1) >= 10 ** (threshold_dbfs / 20)
        result, opened = [], None
        for index, value in enumerate([*active, False]):
            if value and opened is None:
                opened = index * self.step_s
            elif not value and opened is not None:
                end = min(index * self.step_s, self.duration_s)
                if end - opened >= .14:
                    result.append({"start_s": opened, "end_s": end, "kind": "energy_activity_candidate"})
                opened = None
        return result
