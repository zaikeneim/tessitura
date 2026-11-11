import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.processing.long_diagnostics import DiagnosticOptions, diagnose_recording, select_benchmark_regions
from src.processing.checkpoints import file_digest


class LongRecordingDiagnosticsTests(unittest.TestCase):
    def test_complete_scan_preserves_opposed_channels_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            rate = 8000
            left = np.sin(np.arange(rate * 22 + 17) * .1).astype(np.float32) * .3
            native = np.column_stack([left, -left])
            sf.write(source, native, rate, subtype="FLOAT")
            before = file_digest(source)
            with patch("src.processing.long_diagnostics.find_cached_session", return_value=None):
                path = diagnose_recording(source, output_dir=root / "diagnostics")
            report = json.loads(path.read_text(encoding="utf-8"))
            actual, sample_rate = sf.read(report["decoded_audio"], dtype="float32", always_2d=True)
            np.testing.assert_array_equal(native, actual)
            self.assertEqual(sample_rate, rate)
            self.assertEqual(before, file_digest(source))
            self.assertEqual(report["scan"]["decoded_frames"], len(native))
            self.assertAlmostEqual(report["scan"]["channel_correlation"], -1.)
            self.assertGreater(report["scan"]["rms_dbfs_per_channel"][0], -20.)
            timeline = json.loads((path.parent / "timeline.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(timeline[-1]["end_s"], len(native) / rate)
            self.assertIsNone(report["quality_metrics"])

    def test_cached_overlaps_remain_candidates_and_reference_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ambient.wav"
            sf.write(source, np.full((8000 * 90, 2), .05, dtype=np.float32), 8000)
            session = root / "session.json"
            session.write_text(json.dumps({"segments": [
                {"start": 20, "end": 30, "speaker": "A", "text": "voice one"},
                {"start": 22, "end": 28, "speaker": "B", "text": "voice two"},
                {"start": 31, "end": 35, "speaker": "B", "text": "next turn"},
            ]}), encoding="utf-8")
            path = diagnose_recording(source, output_dir=root / "diagnostics", session_path=session)
            report = json.loads(path.read_text(encoding="utf-8"))
            overlaps = [r for r in report["regions"] if r["selection_reason"] == "cached_speaker_overlap_candidate"]
            self.assertTrue(overlaps)
            rows = overlaps[0]["existing_transcript"]
            self.assertEqual(rows[0]["start_s"], 20)
            self.assertEqual(rows[1]["start_s"], 22)
            self.assertEqual(overlaps[0]["reference"]["verification_level"], "unverified_candidate")
            timeline = json.loads((path.parent / "timeline.json").read_text(encoding="utf-8"))
            self.assertIn("cached_speaker_overlap_candidate", timeline[22]["conditions"])
            self.assertNotIn("cached_speaker_overlap_candidate", timeline[31]["conditions"])

    def test_selection_spans_entire_recording_and_seeded_controls(self):
        options = DiagnosticOptions()
        timeline = [{"start_s": float(i), "end_s": float(i + 1), "rms_dbfs": [-25.],
                     "conditions": ["cached_speaker_overlap_candidate"] if i % 50 < 5 else []}
                    for i in range(7200)]
        first = select_benchmark_regions(timeline, [], 7200., options)
        second = select_benchmark_regions(timeline, [], 7200., options)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)  # No fabricated degraded regions to fill quotas.
        self.assertTrue(any(r["start_s"] < 1000 for r in first))
        self.assertTrue(any(r["end_s"] > 6000 for r in first))
        self.assertEqual(sum(r["selection_reason"] == "seeded_random_control" for r in first), 2)

    def test_cache_is_invalidated_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            sf.write(source, np.zeros(8000), 8000)
            with patch("src.processing.long_diagnostics.find_cached_session", return_value=None):
                first_path = diagnose_recording(source, output_dir=root / "diagnostics")
                first = json.loads(first_path.read_text(encoding="utf-8"))
                sf.write(source, np.ones(8000) * .1, 8000)
                second_path = diagnose_recording(source, output_dir=root / "diagnostics")
                second = json.loads(second_path.read_text(encoding="utf-8"))
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            self.assertGreater(second["scan"]["rms_dbfs_per_channel"][0], -30)

    def test_invalid_options_fail_before_decode(self):
        for options in [DiagnosticOptions(window_s=0), DiagnosticOptions(segment_s=5),
                        replace(DiagnosticOptions(), context_s=float("nan"))]:
            with self.assertRaises(ValueError):
                options.validate()

    def test_experiments_keep_channel_and_absolute_timing_with_fixed_model(self):
        from src.processing.diagnostic_experiments import run_diagnostic_experiments
        from src.processing.transcribe import TSegment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "region-01.wav"
            sf.write(sample, np.full((16000, 2), .1, dtype=np.float32), 16000)
            model = root / "model"
            model.mkdir()
            (model / "model.bin").write_bytes(b"fixed-model")
            report = root / "report.json"
            report.write_text(json.dumps({"source_sha256": "source", "regions": [
                {"region_id": "region-01", "audio_file": sample.name, "sample_origin_s": 123.5}
            ]}), encoding="utf-8")
            with patch("src.processing.transcribe._ensure_model_path", return_value=str(model)), \
                    patch("src.processing.transcribe.Transcriber") as factory, \
                    patch("src.processing.denoise.denoise_array", side_effect=lambda data, sr, **kwargs: data) as denoise:
                factory.return_value.transcribe.return_value = ([TSegment(
                    start=.1, end=.5, text="speech", words=[(.1, .5, "speech", .9)])], "en")
                result = run_diagnostic_experiments(report, count=1, model="fixed", device="cpu")
                prepared_calls = denoise.call_count
                run_diagnostic_experiments(report, count=1, model="fixed", device="cpu")
                self.assertEqual(denoise.call_count, prepared_calls)
            value = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(len(value["results"]), 8)
            self.assertEqual({r["configuration"]["channel"] for r in value["results"]}, {0, 1})
            self.assertTrue(all(r["segments"][0]["start"] == 123.6 for r in value["results"]))
            self.assertTrue(all(r["segments"][0]["words"][0][0] == 123.6 for r in value["results"]))
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(sum(not c.kwargs["vad_filter"] for c in factory.return_value.transcribe.call_args_list), 2)
            self.assertIsNone(value["quality_metrics"])


if __name__ == "__main__":
    unittest.main()
