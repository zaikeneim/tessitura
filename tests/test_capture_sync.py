"""Clock synchronization uses controlled signals, native tracks and bounded IO."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.models.session import Session
from src.processing.capture_sync import (
    clock_diagnostics, create_solo_playback, resample_capture_clock, synchronize_capture_sources,
)
from src.processing.pipeline import PipelineOptions, _portable_source_metadata, process_recording
from src.processing.transcribe import TSegment


def metadata(seconds, scale, *, source="mic", timing_source="blocking_read_clock_estimate", jitter=0.0):
    rng = np.random.default_rng(98)
    return {"tracks": {source: {
        "timing_source": timing_source,
        "timing": [{"audio_seconds": float(time),
                    "clock_seconds": float(time * scale + .04 + rng.uniform(-jitter, jitter))}
                   for time in np.linspace(0.5, seconds - .5, 80)],
    }}}


class CaptureClockResamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_streaming_clock_resample_matches_continuous_polyphase_across_seams(self):
        samples = np.random.default_rng(18).normal(0, .1, (120007, 2)).astype(np.float32)
        original = self.root / "native.wav"
        sf.write(original, samples, 48000, subtype="FLOAT")
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        for scale in (1.0001, .9999):
            with self.subTest(scale=scale):
                destination, report = resample_capture_clock(original, self.root / f"{scale}.wav", scale)
                actual, rate = sf.read(destination, dtype="float32", always_2d=True)
                up, down = report["ratio"]
                expected = resample_poly(samples, up, down, axis=0)[:len(actual)]
                self.assertEqual(rate, 48000)
                self.assertEqual(actual.shape, (round(len(samples) * report["applied_scale"]), 2))
                np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), digest)

    def test_solo_playback_is_mono_16k_with_global_zero_padding(self):
        source = self.root / "native.wav"
        samples = np.random.default_rng(19).normal(0, .1, (48001, 2)).astype(np.float32)
        sf.write(source, samples, 48000, subtype="FLOAT")
        output = create_solo_playback(source, self.root / "playback.wav", .125)
        actual, rate = sf.read(output, dtype="float32", always_2d=True)
        self.assertEqual(rate, 16000)
        self.assertEqual(actual.shape[1], 1)
        np.testing.assert_array_equal(actual[:2000], 0)
        expected = resample_poly(samples, 1, 3, axis=0).mean(axis=1)
        np.testing.assert_allclose(actual[2000:, 0], expected[:len(actual) - 2000], atol=2e-6)

    def test_explicit_offset_without_sidecar_still_gets_global_playback(self):
        original = self.root / "mic.wav"
        sf.write(original, np.full((1600, 2), .25, dtype=np.float32), 16000, subtype="FLOAT")
        result = synchronize_capture_sources(original, None, offsets_s=[.125], work_dir=self.root)
        self.assertEqual(result.mic_path, original)
        playback = result.report["sources"]["mic"]["playback_audio"]
        samples, rate = sf.read(playback, dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 3600)
        np.testing.assert_array_equal(samples[:2000], 0)
        np.testing.assert_array_equal(samples[2000:], .25)

    def test_two_hour_virtual_clock_corrects_drift_and_keeps_native_channels(self):
        rate, seconds, scale = 20, 7200, 1.0001
        signal = np.zeros((rate * seconds, 2), dtype=np.float32)
        signal[rate * 3600, 0], signal[rate * 5400, 1] = .5, .3
        original = self.root / "two-hour-native.wav"
        sf.write(original, signal, rate, subtype="FLOAT")
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        result = synchronize_capture_sources(
            original, None, offsets_s=[.125], metadata=metadata(seconds, scale),
            work_dir=self.root, make_playback=False,
        )
        corrected, actual_rate = sf.read(result.mic_path, dtype="float32", always_2d=True)
        self.assertEqual(actual_rate, rate)
        self.assertEqual(corrected.shape[1], 2)
        self.assertAlmostEqual(len(corrected) / rate, seconds * scale, delta=1 / rate)
        self.assertAlmostEqual(np.argmax(np.abs(corrected[:, 0])) / rate, 3600 * scale, delta=1 / rate)
        self.assertAlmostEqual(np.argmax(np.abs(corrected[:, 1])) / rate, 5400 * scale, delta=1 / rate)
        source = result.report["sources"]["mic"]
        self.assertTrue(source["correction_applied"])
        self.assertEqual(source["offset_s"], .125)
        self.assertFalse(result.report["review_required"])
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), digest)

    def test_serialized_sync_and_echo_paths_are_portable_without_mutating_operational_paths(self):
        original = str(self.root / "recordings" / "mic.wav")
        processed = str(self.root / "work" / "run" / "mic_clock_synchronized.wav")
        playback = str(self.root / "work" / "run" / "mic_synchronized_playback.wav")
        derivative = str(self.root / "work" / "run" / "echo_candidate.wav")
        external = str(self.root.parent / "external-benchmark.mp3")
        operational = {
            "synchronization": {"sources": {"mic": {
                "original": original, "processing_audio": processed, "playback_audio": playback}}},
            "sources": {"mic": {"original": original, "processing_audio": processed}},
            "echo": {"original_mic": original, "original_system": external,
                     "derivative": derivative, "asr_input": processed},
        }
        with patch("src.processing.pipeline.paths.data_dir", return_value=self.root):
            stored = _portable_source_metadata(operational)
        self.assertEqual(stored["synchronization"]["sources"]["mic"]["playback_audio"],
                         "work/run/mic_synchronized_playback.wav")
        self.assertEqual(stored["sources"]["mic"]["original"], "recordings/mic.wav")
        self.assertEqual(stored["echo"]["derivative"], "work/run/echo_candidate.wav")
        self.assertEqual(stored["echo"]["original_system"], external)
        self.assertEqual(operational["sources"]["mic"]["processing_audio"], processed)
        self.assertEqual(operational["echo"]["derivative"], derivative)

    def test_callback_clock_estimate_cannot_authorize_a_time_warp(self):
        original = self.root / "native.wav"
        sf.write(original, np.zeros(16000 * 40, dtype=np.float32), 16000)
        result = synchronize_capture_sources(
            original, None, metadata=metadata(40, 1.005, timing_source="callback_clock_estimate"),
            work_dir=self.root, make_playback=False,
        )
        self.assertEqual(result.mic_path, original)
        self.assertTrue(result.report["review_required"])
        self.assertTrue(result.report["uncertain"])
        self.assertFalse(result.report["sources"]["mic"]["correction_applied"])

    def test_untrusted_short_or_silent_clock_requires_review_before_fit(self):
        for timing_source in ("callback_clock_estimate", "mixed_adc_and_estimated_clock", "unknown"):
            for seconds in (0, 5):
                with self.subTest(timing_source=timing_source, seconds=seconds):
                    capture = metadata(5, 1.0, timing_source=timing_source)
                    if seconds == 0:
                        capture["tracks"]["mic"]["timing"] = []
                    report = clock_diagnostics(capture, "mic")
                    self.assertTrue(report["review_required"])
                    self.assertFalse(report["reliable_scale"])
                    self.assertFalse(report["correction_applied"])

    def test_any_fallback_refuses_correction_even_when_last_clock_says_native(self):
        original = self.root / "native.wav"
        sf.write(original, np.zeros(16000 * 40, dtype=np.float32), 16000)
        for seconds in (5, 40):
            with self.subTest(seconds=seconds):
                capture = metadata(seconds, 1.005, timing_source="wasapi_adc_clock")
                capture["tracks"]["mic"]["timestamp_fallbacks"] = 1
                result = synchronize_capture_sources(original, None, metadata=capture,
                                                     work_dir=self.root, make_playback=False)
                self.assertEqual(result.mic_path, original)
                self.assertTrue(result.report["review_required"])
                self.assertTrue(result.report["uncertain"])
                self.assertFalse(result.report["sources"]["mic"]["correction_applied"])

    def test_unstable_or_wrong_recording_anchors_preserve_original_and_require_review(self):
        original = self.root / "native.wav"
        sf.write(original, np.zeros(16000 * 40, dtype=np.float32), 16000)
        for anchors in (metadata(40, 1.005, jitter=.1), metadata(7200, 1.0001)):
            with self.subTest(anchors=anchors["tracks"]["mic"]["timing_source"]):
                result = synchronize_capture_sources(original, None, metadata=anchors,
                                                     work_dir=self.root, make_playback=False)
                self.assertEqual(result.mic_path, original)
                self.assertTrue(result.report["review_required"])

    def test_source_asr_and_mix_share_clock_derivative_and_overlap_roundtrip(self):
        rate, seconds = 20, 7200
        mic, system = self.root / "mic.wav", self.root / "system.wav"
        signal = np.zeros((rate * seconds, 2), dtype=np.float32)
        signal[rate * 3600, 0] = .5
        sf.write(mic, signal, rate, subtype="FLOAT")
        sf.write(system, signal, rate, subtype="FLOAT")
        capture = metadata(seconds, 1.0001)
        capture["tracks"].update(metadata(seconds, 1.0, source="system", timing_source="wasapi_adc_clock")["tracks"])
        mixed_inputs = []

        def mix(sources, destination, **kwargs):
            mixed_inputs.extend(map(Path, sources))
            sf.write(destination, np.zeros(100, dtype=np.float32), 16000)
            return destination

        def playback(source, destination, offset):
            sf.write(destination, np.zeros(100, dtype=np.float32), 16000)
            return destination

        fake_segments = [
            ([TSegment(3600.36, 3602.36, "quiet local", [(3600.36, 3602.36, "quiet local", .95)])], "it"),
            ([TSegment(3600.4, 3602.4, "remote", [(3600.4, 3602.4, "remote", .95)])], "it"),
        ]
        with patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")), \
             patch("src.processing.capture_sync.create_solo_playback", side_effect=playback), \
             patch("src.processing.pipeline.audio_utils.mix_sources", side_effect=mix), \
             patch("src.processing.source_transcription.Transcriber") as transcriber, \
             patch("src.processing.source_transcription.align_segments",
                   side_effect=lambda segments, *args, **kwargs: (segments, MagicMock(to_dict=lambda: {}))):
            transcriber.return_value.transcribe.side_effect = fake_segments
            session = process_recording(
                mic, system, offsets_s=[.125, 0.0], work_dir=self.root,
                options=PipelineOptions(strategy="source_aware", recording_mode="headset", diarize=False,
                                        audio_profile="mic_system", recording_metadata=capture),
            )
        asr_inputs = [Path(call.args[0]) for call in transcriber.return_value.transcribe.call_args_list]
        self.assertEqual(mixed_inputs, asr_inputs)
        self.assertNotEqual(asr_inputs[0], mic)
        self.assertEqual(Path(session.audio.mic), mic)
        self.assertEqual(Path(session.audio.system), system)
        self.assertEqual(session.utterances[0].start_s, 3600.4)
        self.assertEqual(session.utterances[1].start_s, 3600.485)
        self.assertEqual(len({item.overlap_group_id for item in session.utterances}), 1)
        self.assertTrue(all(item.overlap_group_id for item in session.utterances))
        restored = Session.from_dict(session.to_dict())
        self.assertEqual(restored.capture_manifest, session.capture_manifest)
        self.assertEqual([item.start_s for item in restored.utterances], [3600.4, 3600.485])
        self.assertEqual(restored.capture_manifest["sources"]["mic"]["original"], str(mic))
        self.assertIn("playback_audio", restored.capture_manifest["synchronization"]["sources"]["mic"])


if __name__ == "__main__":
    unittest.main()
