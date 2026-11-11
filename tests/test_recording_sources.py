"""Source topology regressions use controlled signals, never private benchmarks."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src.processing.diarize import DiarizationResult, Turn
from src.processing.echo import evaluate_echo
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.source_transcription import _clock_diagnostics, transcribe_sources
from src.processing.transcribe import TSegment


def _segment(start, end, text):
    return TSegment(start, end, text, [(start, end, text, 0.95)])


class SourceTopologyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mic, self.system = self.root / "mic.wav", self.root / "headphones.wav"
        rng = np.random.default_rng(7)
        signal = rng.normal(0, 0.15, 16000 * 6).astype(np.float32)
        sf.write(self.mic, signal, 16000, subtype="FLOAT")
        sf.write(self.system, signal * 0.05, 16000, subtype="FLOAT")
        self.transcriber = self.enterContext(patch("src.processing.source_transcription.Transcriber"))
        self.enterContext(patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")))
        self.enterContext(patch("src.processing.source_transcription.align_segments",
                                side_effect=lambda segments, *args, **kwargs: (segments, MagicMock(to_dict=lambda: {}))))
        self.diarizer = self.enterContext(patch("src.processing.source_transcription.Diarizer"))
        self.diarizer.return_value.diarize_detailed.return_value = DiarizationResult([], [], [])

    def options(self, **changes):
        values = dict(strategy="source_aware", recording_mode="headset", denoise=False,
                      diarize=True, audio_profile="mic_system", alignment_backend="whisper")
        values.update(changes)
        return PipelineOptions(**values)

    def test_verified_local_skips_diarization_but_two_remote_speakers_remain_distinct(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(0, 3, "local")], "it"),
            ([_segment(0, 1, "first"), _segment(2, 3, "second")], "it"),
        ]
        remote = [Turn(0, 1, "A"), Turn(2, 3, "B")]
        self.diarizer.return_value.diarize_detailed.return_value = DiarizationResult(remote, remote, [])
        utterances, segments, _, meta = transcribe_sources(
            self.mic, self.system, options=self.options(mic_single_speaker_verified=True),
        )
        self.assertEqual(self.diarizer.call_count, 1)
        self.assertEqual({item.speaker for item in utterances}, {"SPEAKER_MIC", "SPEAKER_SYS_A", "SPEAKER_SYS_B"})
        self.assertEqual(meta["diarization"]["mic"]["status"], "verified_single_speaker")
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(segment.source_ids for segment in segments))

    def test_ambient_mic_can_contain_multiple_people(self):
        self.transcriber.return_value.transcribe.return_value = (
            [_segment(0, 1, "person one"), _segment(2, 3, "person two")], "it")
        turns = [Turn(0, 1, "A"), Turn(2, 3, "B")]
        self.diarizer.return_value.diarize_detailed.return_value = DiarizationResult(turns, turns, [])
        utterances, _, _, meta = transcribe_sources(
            self.mic, None, options=self.options(recording_mode="ambient"),
        )
        self.assertEqual({item.speaker for item in utterances}, {"SPEAKER_MIC_A", "SPEAKER_MIC_B"})
        self.assertEqual(meta["diarization"]["mic"]["status"], "diarized")

    def test_unverified_headset_diarizes_both_tracks(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(0, 1, "local")], "it"), ([_segment(0, 1, "remote")], "it"),
        ]
        transcribe_sources(self.mic, self.system, options=self.options())
        self.assertEqual(self.diarizer.call_count, 2)

    def test_diarization_failure_preserves_speech_with_unresolved_identity(self):
        self.transcriber.return_value.transcribe.return_value = ([_segment(0, 1, "quiet person")], "it")
        self.diarizer.side_effect = RuntimeError("model unavailable")
        utterances, segments, _, meta = transcribe_sources(self.mic, None, options=self.options())
        self.assertEqual(segments[0].text, "quiet person")
        self.assertEqual(utterances[0].speaker, "SPEAKER_MIC_UNKNOWN")
        self.assertEqual(meta["diarization"]["mic"]["warning"], "model unavailable")

    def test_matching_overlapping_words_are_never_dropped_as_echo(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(0, 2, "yes")], "it"), ([_segment(0, 2, "yes")], "it"),
        ]
        utterances, segments, _, meta = transcribe_sources(
            self.mic, self.system,
            options=self.options(diarize=False, headphones_confirmed=False, recording_mode="speakers"),
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(meta["suppressed_bleed_count"], 0)
        self.assertEqual(len({item.overlap_group_id for item in utterances}), 1)
        self.assertTrue(all(item.overlap_group_id for item in utterances))

    def test_system_only_offset_uses_first_present_source(self):
        self.transcriber.return_value.transcribe.return_value = ([_segment(1, 2, "remote")], "it")
        utterances, _, _, meta = transcribe_sources(
            None, self.system, offsets_s=[7200.125], options=self.options(diarize=False),
        )
        self.assertEqual(utterances[0].start_s, 7201.125)
        self.assertEqual(utterances[0].words[0].start, 7201.125)
        self.assertEqual(meta["sources"]["system"]["offset_s"], 7200.125)

    def test_capture_topology_routes_to_source_aware_and_serializes_provenance(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(1, 2, "local")], "it"), ([_segment(1, 2, "remote")], "it"),
        ]
        session = process_recording(
            self.mic, self.system, options=self.options(strategy="standard", diarize=False),
            offsets_s=[0.0, 0.125], work_dir=self.root,
        )
        self.assertEqual(session.pipeline_strategy, "source_aware")
        self.assertEqual(session.capture_manifest["recording_mode"], "headset")
        self.assertEqual(session.capture_manifest["sources"]["system"]["original"], str(self.system))
        self.assertEqual(session.utterances[1].start_s, 1.125)
        self.assertTrue(all(item.candidate_speakers[0]["reason"] == "capture_source_provenance"
                            for item in session.utterances))
        restored = type(session).from_dict(session.to_dict())
        self.assertEqual(restored.capture_manifest, session.capture_manifest)
        self.assertEqual(len(restored.segments), 2)

    def test_missing_source_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Sorgente audio non disponibile"):
            transcribe_sources(self.root / "disconnected.wav", self.system, options=self.options())
        self.transcriber.assert_not_called()

    def test_echo_derivative_is_never_the_only_microphone_asr_input(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(0, 2, "quiet local speech")], "it"), ([_segment(0, 2, "remote")], "it"),
        ]
        from src.processing.echo import EchoResult
        derivative = self.root / "echo_candidate.wav"
        with patch("src.processing.echo.evaluate_echo",
                   return_value=EchoResult(derivative, {"status": "derivative_created", "derivative": str(derivative)})):
            _, segments, _, meta = transcribe_sources(
                self.mic, self.system,
                options=self.options(diarize=False, echo_cancellation=True, recording_mode="speakers"),
                work_dir=self.root,
            )
        self.assertEqual(self.transcriber.return_value.transcribe.call_args_list[0].args[0], self.mic)
        self.assertTrue(meta["echo"]["candidate_only"])
        self.assertEqual(meta["echo"]["asr_input"], str(self.mic))
        self.assertIn("quiet local speech", [segment.text for segment in segments])

    def test_verified_headset_with_only_microphone_uses_source_identity(self):
        self.transcriber.return_value.transcribe.return_value = ([_segment(0, 1, "local")], "it")
        session = process_recording(
            self.mic, None,
            options=self.options(strategy="standard", audio_profile="stereo_ambient",
                                 mic_single_speaker_verified=True),
            work_dir=self.root,
        )
        self.diarizer.assert_not_called()
        self.assertEqual(session.segments[0].speaker, "SPEAKER_MIC")
        self.assertEqual(session.pipeline_strategy, "source_aware")

    def test_three_simultaneous_hypotheses_keep_distinct_streams(self):
        self.transcriber.return_value.transcribe.side_effect = [
            ([_segment(0, 2, "local one"), _segment(0, 2, "local two")], "it"),
            ([_segment(0, 2, "remote")], "it"),
        ]
        utterances, segments, _, _ = transcribe_sources(
            self.mic, self.system, options=self.options(diarize=False),
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual({item.stream_id for item in utterances}, {0, 1, 2})
        self.assertEqual(len({item.overlap_group_id for item in utterances}), 1)

    def test_recording_metadata_survives_export_without_aliasing_caller(self):
        self.transcriber.return_value.transcribe.return_value = ([_segment(0, 1, "local")], "it")
        native = {"tracks": {"mic": {"sample_rate": 48000, "device": "USB microphone"}}}
        session = process_recording(
            self.mic, None,
            options=self.options(audio_profile="stereo_ambient", diarize=False,
                                 recording_metadata=native), work_dir=self.root,
        )
        native["tracks"]["mic"]["sample_rate"] = 8000
        self.assertEqual(session.capture_manifest["recording"]["tracks"]["mic"]["sample_rate"], 48000)
        self.assertEqual(session.provenance["recording"]["tracks"]["mic"]["sample_rate"], 48000)


class EchoEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mic, self.system, self.output = [self.root / name for name in ("mic.wav", "playback.wav", "derived.wav")]
        rng = np.random.default_rng(19)
        self.remote = rng.normal(0, 0.15, 16000 * 8).astype(np.float32)
        self.delay = 640
        self.echo = np.pad(self.remote * 0.3, (self.delay, 0))[:len(self.remote)]

    def write(self, microphone):
        sf.write(self.system, self.remote, 16000, subtype="FLOAT")
        sf.write(self.mic, microphone, 16000, subtype="FLOAT")

    def test_delayed_echo_is_reduced_without_modifying_originals(self):
        self.write(self.echo)
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.mic, self.system)]
        result = evaluate_echo(self.mic, self.system, self.output, apply=True)
        cleaned, rate = sf.read(result.audio_path)
        self.assertEqual(result.report["status"], "derivative_created")
        self.assertAlmostEqual(result.report["delay_s"], 0.04, places=3)
        self.assertGreater(result.report["cancelled_seconds"], 7.0)
        self.assertLess(np.sqrt(np.mean(cleaned ** 2)), 1e-5)
        self.assertEqual(rate, 16000)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.mic, self.system)])

    def test_double_talk_bypasses_cancellation_and_preserves_local_voice(self):
        local = np.zeros_like(self.echo)
        local[16000 * 2:16000 * 4] = np.random.default_rng(29).normal(0, 0.1, 32000)
        raw = self.echo + local
        self.write(raw)
        result = evaluate_echo(self.mic, self.system, self.output, apply=True)
        cleaned, _ = sf.read(result.audio_path)
        self.assertEqual(result.report["status"], "derivative_created")
        self.assertGreater(result.report["bypassed_seconds"], 1.9)
        np.testing.assert_array_equal(cleaned[32000:64000], raw[32000:64000])
        self.assertLess(np.sqrt(np.mean(cleaned[80000:96000] ** 2)), 1e-5)

    def test_continuous_double_talk_has_no_reference_only_calibration(self):
        local = np.random.default_rng(31).normal(0, 0.2, len(self.echo)).astype(np.float32)
        self.write(self.echo + local)
        result = evaluate_echo(self.mic, self.system, self.output, apply=True)
        self.assertEqual(result.audio_path, self.mic)
        self.assertEqual(result.report["status"], "insufficient_reference_only_calibration")
        self.assertFalse(self.output.exists())

    def test_measurement_only_never_changes_asr_input(self):
        self.write(self.echo)
        result = evaluate_echo(self.mic, self.system, self.output)
        self.assertEqual(result.audio_path, self.mic)
        self.assertEqual(result.report["status"], "measured_only")
        self.assertFalse(self.output.exists())

    def test_start_offsets_are_separate_from_acoustic_delay(self):
        self.write(np.pad(self.remote * 0.3, (1920, 0))[:len(self.remote)])
        result = evaluate_echo(self.mic, self.system, self.output, system_offset=0.08)
        self.assertEqual(result.report["status"], "measured_only")
        self.assertAlmostEqual(result.report["delay_s"], 0.04, places=3)

    def test_original_cannot_be_used_as_derivative_destination(self):
        self.write(self.echo)
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            evaluate_echo(self.mic, self.system, self.mic, apply=True)

    def test_different_native_device_rates_align_reference(self):
        from scipy.signal import resample_poly
        sf.write(self.system, resample_poly(self.remote, 3, 1), 48000, subtype="FLOAT")
        sf.write(self.mic, resample_poly(self.echo, 441, 160), 44100, subtype="FLOAT")
        result = evaluate_echo(self.mic, self.system, self.output, apply=True)
        self.assertEqual(result.report["status"], "derivative_created")
        self.assertAlmostEqual(result.report["delay_s"], 0.04, places=3)
        self.assertGreater(result.report["cancelled_seconds"], 7.0)


class ClockDiagnosticsTests(unittest.TestCase):
    def metadata(self, scale=1.0, jitter=0.0, seconds=7200):
        rng = np.random.default_rng(71)
        return {"tracks": {"mic": {"timing_source": "blocking_read_clock_estimate", "timing": [
            {"audio_seconds": float(moment),
             "clock_seconds": float(moment * scale + 0.05 + rng.uniform(-jitter, jitter))}
            for moment in np.linspace(1, seconds, 100)
        ]}}}

    def test_two_hour_clock_drift_requires_review_without_silent_warp(self):
        report = _clock_diagnostics(self.metadata(scale=1.0001, jitter=0.001), "mic")
        self.assertTrue(report["review_required"])
        self.assertFalse(report["correction_applied"])
        self.assertAlmostEqual(report["estimated_drift_ppm"], 100, delta=1)
        self.assertAlmostEqual(report["estimated_drift_seconds"], 0.72, delta=0.01)
        self.assertLess(report["uncertainty_seconds"], 0.01)

    def test_short_read_jitter_is_not_treated_as_clock_drift(self):
        report = _clock_diagnostics(self.metadata(jitter=0.001), "mic")
        self.assertFalse(report["review_required"])
        self.assertEqual(report["status"], "no_significant_drift_measured")

    def test_short_recording_has_explicit_timing_uncertainty(self):
        report = _clock_diagnostics(self.metadata(seconds=10), "mic")
        self.assertEqual(report["status"], "insufficient_timing_anchors")
        self.assertIsNone(report["uncertainty_seconds"])

    def test_invalid_clock_anchors_cannot_be_an_all_clear(self):
        for metadata in ({"tracks": {"mic": {"timing": [{"audio_seconds": float("nan"), "clock_seconds": 1}]}}},
                         {"tracks": {"mic": []}}, {"tracks": []}):
            report = _clock_diagnostics(metadata, "mic")
            self.assertTrue(report["review_required"])


if __name__ == "__main__":
    unittest.main()
