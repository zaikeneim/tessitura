"""Reproducible CPU benchmarks with exact-output assertions; no private audio."""
import json
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()
import numpy as np
import soundfile as sf
from src.processing import audio_utils, merge
from src.processing.diarize import Turn


def measure(fn):
    times = []
    result = None
    for _ in range(3):
        start = perf_counter()
        result = fn()
        times.append(perf_counter()-start)
    return median(times), result


def main():
    words = [merge.WordUnit(i*.5, i*.5+.4, " word.") for i in range(2400)]
    turns = [Turn(i*3, i*3+2.8, str(i % 3)) for i in range(400)]
    def full_scan():
        labels = [None]*len(words)
        for a, b in merge._phrases(words):
            labels[a:b] = merge._phrase_labels(words, a, b, turns)
        return merge._fill_gaps(merge._smooth(words, labels))
    old, expected = measure(full_scan)
    new, actual = measure(lambda: merge.assign_speakers(words, turns))
    assert actual == expected
    report = {"speaker_matching": {"reference_seconds": old, "optimized_seconds": new,
                                    "speedup": old/new, "identical": True}}
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        rng = np.random.default_rng(7)
        sources = []
        for i, rate in enumerate((44100, 48000)):
            path = folder / f"input{i}.wav"
            sf.write(path, rng.normal(0, .08, (rate*60, 2)), rate)
            sources.append(path)
        def before():
            audio_utils.mix_sources(sources, folder/"old.wav", offsets_s=[0, .137])
            return audio_utils.source_dominance(sources, offsets_s=[0, .137])
        def after():
            result = []
            audio_utils.mix_sources(sources, folder/"new.wav", offsets_s=[0, .137], dominance_result=result)
            return result[0]
        old, expected = measure(before)
        new, actual = measure(after)
        np.testing.assert_array_equal(expected[0], actual[0])
        assert (folder/"old.wav").read_bytes() == (folder/"new.wav").read_bytes()
        report["audio_preparation_60s"] = {"reference_seconds": old, "optimized_seconds": new,
                                           "speedup": old/new, "identical": True}
    out = ROOT / "artifacts/performance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
