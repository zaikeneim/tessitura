import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

import numpy as np
import soundfile as sf
from src.processing import audio_utils
from src.processing.transcribe import TSegment, _ensure_model_path


class AudioReuseTests(unittest.TestCase):
    def test_reused_channel_hint_is_bit_exact(self):
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as folder:
            sources = []
            for i, rate in enumerate((44100, 48000)):
                path = Path(folder) / f"source{i}.wav"
                sf.write(path, rng.normal(0, 0.05, (rate*2+i*200, 2)), rate)
                sources.append(path)
            offsets = [0, 0.137]
            expected, frame = audio_utils.source_dominance(sources, offsets_s=offsets)
            original = audio_utils.mix_sources(sources, Path(folder)/"original.wav", offsets_s=offsets)
            reused = []
            with patch.object(audio_utils, "load_mono_16k", wraps=audio_utils.load_mono_16k) as load:
                output = audio_utils.mix_sources(sources, Path(folder)/"reused.wav",
                                                 offsets_s=offsets, dominance_result=reused)
            self.assertEqual(load.call_count, 2)
            np.testing.assert_array_equal(expected, reused[0][0])
            self.assertEqual(frame, reused[0][1])
            self.assertEqual(original.read_bytes(), output.read_bytes())

    def test_pipeline_does_not_overwrite_previous_session_audio(self):
        from src.processing.pipeline import process_recording, PipelineOptions
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.wav"
            sf.write(source, np.ones(16000, dtype="float32")*.1, 16000)
            opts = PipelineOptions(denoise=False, diarize=False, device="cpu",
                                   audio_profile="stereo_ambient")
            with patch("src.processing.transcribe.Transcriber") as transcriber:
                transcriber.return_value.transcribe.return_value = ([TSegment(0, 1, "test", [])], "it")
                with patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")):
                    first = process_recording(source, options=opts, work_dir=folder)
                    first_path = Path(first.audio.mixed)
                    before = first_path.read_bytes()
                    sf.write(source, np.zeros(16000, dtype="float32"), 16000)
                    second = process_recording(source, options=opts, work_dir=folder)
            self.assertNotEqual(first.audio.mixed, second.audio.mixed)
            self.assertEqual(first_path.read_bytes(), before)

    def test_audio_profile_is_required_before_model_loading(self):
        from src.processing.pipeline import process_recording, PipelineOptions
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            sf.write(source, np.ones(16000, dtype="float32") * .1, 16000)
            with patch("src.processing.pipeline.resolve_device") as resolve:
                with self.assertRaisesRegex(ValueError, "Profilo audio obbligatorio"):
                    process_recording(
                        source,
                        options=PipelineOptions(denoise=False, diarize=False),
                        work_dir=folder,
                    )
            resolve.assert_not_called()

    def test_stereo_ambient_never_uses_channel_hint(self):
        from src.processing.pipeline import process_recording, PipelineOptions
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "room.wav"
            sf.write(source, np.zeros((16000, 2), dtype="float32"), 16000)
            opts = PipelineOptions(denoise=False, diarize=False, device="cpu",
                                   audio_profile="stereo_ambient")
            with patch("src.processing.transcribe.Transcriber") as transcriber:
                transcriber.return_value.transcribe.return_value = (
                    [TSegment(0, 1, "test", [])], "it"
                )
                with patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")):
                    session = process_recording(source, options=opts, work_dir=folder)
            self.assertEqual(session.audio_profile, "stereo_ambient")
            self.assertEqual(transcriber.return_value.transcribe.call_count, 1)

    def test_mic_system_profile_rejects_mono_single_source(self):
        from src.processing.pipeline import process_recording, PipelineOptions
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "mono.wav"
            sf.write(source, np.zeros(16000, dtype="float32"), 16000)
            with self.assertRaisesRegex(ValueError, "richiede due sorgenti"):
                process_recording(
                    source,
                    options=PipelineOptions(denoise=False, diarize=False,
                                            audio_profile="mic_system"),
                    work_dir=folder,
                )

    def test_invalid_quality_options_fail_before_device_resolution(self):
        from src.processing.pipeline import process_recording, PipelineOptions
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            sf.write(source, np.zeros(16000, dtype="float32"), 16000)
            for kwargs, message in (
                ({"denoise_strength": 1.1}, "denoise_strength"),
                ({"quality_retries": -1}, "quality_retries"),
                ({"num_speakers": 0}, "num_speakers"),
            ):
                with self.subTest(message=message), patch(
                    "src.processing.pipeline.resolve_device"
                ) as resolve:
                    with self.assertRaisesRegex(ValueError, message):
                        process_recording(
                            source,
                            options=PipelineOptions(
                                denoise=False, diarize=False,
                                audio_profile="stereo_ambient", **kwargs
                            ),
                            work_dir=folder,
                        )
                    resolve.assert_not_called()


class ModelAndUpdateTests(unittest.TestCase):
    def test_invalid_timestamps_are_rejected_before_gui_load(self):
        from src.models.session import Session
        for start, end in ((float("nan"), 1), (0, float("inf")), ("0", 1), (2, 1), (-1, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                Session.from_dict({"segments": [{"start": start, "end": end, "text": "x"}]})

    def test_diarization_cached_bundle_needs_no_token_or_network(self):
        from src.processing.diarize import _model_path
        with tempfile.TemporaryDirectory() as folder:
            for name in ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin",
                         "plda/plda.npz", "plda/xvec_transform.npz"):
                file = Path(folder)/name
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(b"cached")
            with patch("huggingface_hub.snapshot_download", return_value=folder) as download:
                self.assertEqual(_model_path("pyannote/test", None), folder)
                self.assertTrue(download.call_args.kwargs["local_files_only"])
                self.assertEqual(download.call_count, 1)

    def test_complete_cached_model_never_contacts_network(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
                (Path(folder)/name).write_text("cached")
            with patch("huggingface_hub.snapshot_download", return_value=folder) as download:
                self.assertEqual(_ensure_model_path("tiny", None), folder)
                self.assertEqual(download.call_count, 1)
                self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_incomplete_cache_is_rejected_by_offline_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"config.json").write_text("{}")
            with patch("huggingface_hub.snapshot_download", return_value=folder) as download:
                with self.assertRaises(RuntimeError):
                    _ensure_model_path("tiny", None, local_only=True)
                self.assertEqual(download.call_count, 1)

    def test_outdated_malformed_json_is_reported(self):
        from src.updates import _pip_outdated
        with patch("src.updates.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "not json", "")):
            self.assertIsNotNone(_pip_outdated()[2])

    def test_package_names_are_canonicalized(self):
        from src.updates import _pip_outdated, Report
        packages = [{"name": "pyannote_audio", "version": "4", "latest_version": "5"}]
        with patch("src.updates.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(packages), "")):
            direct, count, error = _pip_outdated()
        self.assertEqual(len(direct), 1)
        self.assertEqual(count, 0)
        self.assertIsNone(error)
        self.assertFalse(Report(indirect_count=1).is_empty())


class EncoderFailureTests(unittest.TestCase):
    def test_failed_conversion_preserves_source_and_previous_mp3(self):
        from src.audio.encode import to_mp3
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.wav"
            output = source.with_suffix(".mp3")
            sf.write(source, np.ones(16000, dtype="float32")*.1, 16000)
            output.write_bytes(b"previous MP3")
            with patch("src.audio.encode.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                with self.assertRaises(subprocess.CalledProcessError):
                    to_mp3(source)
            self.assertTrue(source.exists())
            self.assertEqual(output.read_bytes(), b"previous MP3")
            self.assertEqual(len(list(Path(folder).iterdir())), 2)

    def test_cleanup_failure_still_returns_valid_audio(self):
        from src.audio.encode import to_mp3
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.wav"
            sf.write(source, np.ones(16000, dtype="float32")*.1, 16000)
            original_unlink = Path.unlink
            def unlink(path, *args, **kwargs):
                if path == source:
                    raise PermissionError("locked WAV")
                return original_unlink(path, *args, **kwargs)
            with patch.object(Path, "unlink", unlink), self.assertLogs("src.audio.encode", level="WARNING"):
                result = to_mp3(source, delete_source=True)
            self.assertTrue(source.exists())
            self.assertGreater(sf.info(result).duration, .9)


class CaptureTests(unittest.TestCase):
    def test_both_sources_receive_stop_before_either_is_joined(self):
        import threading
        from types import SimpleNamespace
        from src.audio.capture import Recorder
        recorder = Recorder()
        sources = []
        for name in ("mic", "system"):
            sources.append(SimpleNamespace(_stop=threading.Event(), stopped_at=None,
                           started_at=1., error=None, dropped_seconds=lambda: 0.))
        def stop():
            self.assertTrue(all(source._stop.is_set() for source in sources))
            self.assertEqual(sources[0].stopped_at, sources[1].stopped_at)
            return Path("saved.wav")
        for source in sources:
            source.stop = stop
        recorder._mic_rec, recorder._sys_rec = sources
        recorder.is_recording = True
        result = recorder.stop()
        self.assertFalse(recorder.is_recording)
        self.assertFalse(result.errors)

    def test_disk_error_drains_queue_and_signals_stop(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.audio.capture import _StreamRecorder
        rec = _StreamRecorder(None, SimpleNamespace(), Path("unused.wav"))
        rec._wave = Mock()
        rec._wave.writeframes.side_effect = OSError("disk full")
        rec._queue.put(b"\0\0")
        rec._queue.put(b"\0\0")
        rec._queue.put(None)
        writer = threading.Thread(target=rec._write_loop)
        writer.start()
        writer.join(timeout=2)
        self.assertFalse(writer.is_alive())
        self.assertTrue(rec._stop.is_set())
        self.assertTrue(rec._queue.empty())
        self.assertEqual(rec._frames_written, 0)


if __name__ == "__main__":
    unittest.main()
