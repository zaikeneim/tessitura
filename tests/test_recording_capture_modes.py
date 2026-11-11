"""Controlled capture fixtures, separate from the imported MP3 benchmark."""

import json
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

import numpy as np
from src.audio.capture import Recorder
from src.audio.devices import DeviceInfo, default_system_loopback, list_input_devices, resolve_device
from src.audio.recording_metadata import load_recording_metadata, recording_sources


def endpoint(index, name, *, loopback=False, rate=48000, host=0):
    return dict(index=index, name=name, maxInputChannels=2, maxOutputChannels=0,
                defaultSampleRate=rate, isLoopbackDevice=loopback, hostApi=host)


class Stream:
    def __init__(self, info, fail=False, silent=False, callback=None):
        self.info, self.fail, self.silent = info, fail, silent
        self.active = False
        self.reads = []
        self.callback = callback
        self.thread = None

    def get_time(self):
        return time.perf_counter()

    def is_active(self):
        return self.active

    def start_stream(self):
        self.active = True
        if self.callback is not None:
            def feed():
                while self.active:
                    data = self.read(2048, True)
                    _, status = self.callback(data, 2048, {"input_buffer_adc_time": time.perf_counter() - 2048 / self.info["defaultSampleRate"]}, 0)
                    if status != 0:
                        self.reads.pop()
                        break
            self.thread = threading.Thread(target=feed, daemon=True)
            self.thread.start()

    def read(self, frames, exception_on_overflow):
        assert exception_on_overflow, "Dropped frames must be reported"
        time.sleep(.002)
        if self.fail and len(self.reads) >= 3:
            raise OSError("disconnected or input overflow")
        # Strong playback and weaker simultaneous voice remain in both channels.
        t = np.arange(frames) / self.info["defaultSampleRate"]
        x = np.column_stack((6000*np.sin(2*np.pi*440*t) + 300*np.sin(2*np.pi*713*t),
                             230*np.sin(2*np.pi*817*t))).astype("<i2")
        data = np.zeros_like(x).tobytes() if self.silent else x.tobytes()
        self.reads.append(data)
        return data

    def stop_stream(self):
        self.active = False
        if self.thread is not None:
            self.thread.join()

    def close(self):
        self.active = False


class Backend:
    def __init__(self, *, fail=False, silent=False):
        self.devices = [endpoint(0, "Room / headset mic", rate=44100),
                        endpoint(1, "Speakers [Loopback]", loopback=True),
                        endpoint(2, "Headphones [Loopback]", loopback=True)]
        self.streams = {}
        self.fail, self.silent = fail, silent
        self.terminated = False

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return next(device for device in self.devices if device["index"] == index)

    def get_loopback_device_info_generator(self):
        return iter(device for device in self.devices if device["isLoopbackDevice"])

    def get_default_input_device_info(self):
        return self.devices[0]

    def open(self, **kwargs):
        assert kwargs["start"] is False
        info = self.get_device_info_by_index(kwargs["input_device_index"])
        assert kwargs["rate"] == info["defaultSampleRate"]
        stream = Stream(info, fail=self.fail and not info["isLoopbackDevice"],
                        silent=self.silent and info["isLoopbackDevice"], callback=kwargs.get("stream_callback"))
        self.streams[info["index"]] = stream
        return stream

    def terminate(self):
        self.terminated = True


class RecordingCaptureTests(unittest.TestCase):
    def capture(self, folder, *, mode="ambient", system=True, fail=False, silent=False):
        backend = Backend(fail=fail, silent=silent)
        recorder = Recorder(folder)
        with patch("src.audio.capture.pyaudio.PyAudio", return_value=backend):
            recorder.start(record_system=system, mic_device=DeviceInfo(0, "Room / headset mic", 2, 44100, False),
                           system_device=DeviceInfo(2, "Headphones [Loopback]", 2, 48000, True) if system else None,
                           recording_mode=mode)
            time.sleep(.035)
            levels = recorder.levels()
            errors = recorder.capture_errors()
            result = recorder.stop()
        self.assertTrue(backend.terminated)
        return result, backend, levels, errors

    def test_ambient_one_or_multiple_overlapping_voices_are_bit_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            result, backend, levels, _ = self.capture(folder, system=False)
            self.assertEqual(result.recording_mode, "ambient")
            self.assertFalse(result.mic_single_speaker_verified)
            self.assertIsNone(result.system_path)
            with wave.open(str(result.mic_path)) as audio:
                self.assertEqual(audio.getnchannels(), 2)
                self.assertEqual(audio.readframes(audio.getnframes()), b"".join(backend.streams[0].reads))
            self.assertGreater(levels["mic"], 0)

    def test_all_modes_allow_separate_headphone_or_speaker_playback(self):
        for mode in ("ambient", "headset", "speakers"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                result, backend, levels, _ = self.capture(folder, mode=mode)
                self.assertTrue(result.mic_path.is_file())
                self.assertTrue(result.system_path.is_file())
                self.assertIn(2, backend.streams)
                self.assertNotIn(1, backend.streams)  # Never substitute default speakers.
                metadata = load_recording_metadata(result.mic_path)
                self.assertEqual(metadata["recording_mode"], mode)
                self.assertEqual(recording_sources(result.mic_path, metadata),
                                 (result.mic_path, result.system_path, result.offsets()))
                self.assertEqual(metadata["tracks"]["mic"]["sample_rate"], 44100)
                self.assertEqual(metadata["tracks"]["system"]["sample_rate"], 48000)
                self.assertGreater(levels["system"], 0)
                for index, path in ((0, result.mic_path), (2, result.system_path)):
                    with wave.open(str(path)) as audio:
                        data = audio.readframes(audio.getnframes())
                        frame_bytes = audio.getnchannels() * audio.getsampwidth()
                    track = metadata["tracks"]["mic" if index == 0 else "system"]
                    # Padding is explicitly tagged silence; every captured sample
                    # outside those gaps must remain byte-identical.
                    for gap in reversed(track["gaps"]):
                        left, right = gap["start_frame"] * frame_bytes, (gap["start_frame"] + gap["frames"]) * frame_bytes
                        self.assertEqual(set(data[left:right]), {0})
                        data = data[:left] + data[right:]
                    self.assertEqual(data, b"".join(backend.streams[index].reads))

    def test_disconnect_or_overflow_reports_error_and_retains_both_tracks(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _, errors = self.capture(folder, fail=True)
            self.assertTrue(errors)
            self.assertIn("overflow", " ".join(result.errors))
            self.assertTrue(result.mic_path.is_file())
            self.assertTrue(result.system_path.is_file())
            self.assertEqual(json.loads(result.metadata_path.read_text(encoding="utf-8"))["state"], "review_required")

    def test_silent_or_wrong_loopback_prompts_endpoint_check(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _, _ = self.capture(folder, mode="headset", silent=True)
            self.assertIn("nessun audio rilevato", " ".join(result.errors))

    def test_device_switch_resolves_new_index_and_native_format(self):
        backend = Backend()
        selected = DeviceInfo(2, "Headphones [Loopback]", 2, 48000, True, 0)
        backend.devices[2] = endpoint(7, "Headphones [Loopback]", loopback=True, rate=44100)
        current = resolve_device(backend, selected)
        self.assertEqual((current.index, current.sample_rate), (7, 44100))
        backend.devices.pop()
        with self.assertRaisesRegex(RuntimeError, "scollegato"):
            resolve_device(backend, selected)

    def test_same_named_inputs_on_different_hosts_are_not_hidden(self):
        backend = Backend()
        backend.devices.append(endpoint(3, "Room / headset mic", host=1))
        self.assertEqual(len(list_input_devices(backend)), 2)
        with self.assertRaisesRegex(RuntimeError, "ambiguo"):
            resolve_device(backend, DeviceInfo(0, "Room / headset mic", 2, 44100, False))

    def test_same_named_same_host_inputs_are_ambiguous(self):
        backend = Backend()
        backend.devices.append(endpoint(3, "Room / headset mic"))
        self.assertEqual(len(list_input_devices(backend)), 2)
        with self.assertRaisesRegex(RuntimeError, "ambiguo"):
            resolve_device(backend, DeviceInfo(3, "Room / headset mic", 2, 44100, False, 0))

    def test_native_loopback_timestamps_preserve_start_mid_and_tail_silence(self):
        from src.audio.capture import _StreamRecorder
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "system.wav"
            rec = _StreamRecorder(None, DeviceInfo(0, "Headphones", 1, 16000, True), path)
            rec.channels, rec.sample_rate, rec.started_at = 1, 16000, 100.0
            rec._wave = wave.open(str(path), "wb")
            rec._wave.setnchannels(1)
            rec._wave.setsampwidth(2)
            rec._wave.setframerate(16000)
            one = np.full(1600, 1234, dtype="<i2").tobytes()
            rec._queue.put((100.5, one))
            rec._queue.put((101.0, one))
            rec._queue.put((101.5, b""))
            rec._queue.put(None)
            rec._write_loop()
            rec._wave.close()
            with wave.open(str(path)) as audio:
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            self.assertEqual(len(samples), 24000)
            np.testing.assert_array_equal(samples[8000:9600], 1234)
            np.testing.assert_array_equal(samples[16000:17600], 1234)
            self.assertEqual(np.count_nonzero(samples), 3200)
            self.assertAlmostEqual(rec.padded_seconds, 1.3)

    def test_start_stream_failure_does_not_wait_for_a_nonexistent_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend()
            original_open = backend.open
            def open_stream(**kwargs):
                stream = original_open(**kwargs)
                stream.start_stream = lambda: (_ for _ in ()).throw(OSError("unplugged at start"))
                return stream
            backend.open = open_stream
            with patch("src.audio.capture.pyaudio.PyAudio", return_value=backend):
                with self.assertRaisesRegex(OSError, "unplugged"):
                    Recorder(folder).start(record_system=False)
            self.assertTrue(backend.terminated)

    def test_unmatched_default_output_never_selects_another_loopback(self):
        backend = Backend()
        backend.get_host_api_info_by_type = lambda _: {"defaultOutputDevice": 9}
        backend.devices.append(endpoint(9, "Other output"))
        self.assertIsNone(default_system_loopback(backend))

    def test_second_source_failure_closes_first_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend()
            with patch("src.audio.capture.pyaudio.PyAudio", return_value=backend):
                recorder = Recorder(folder)
                with self.assertRaisesRegex(RuntimeError, "scollegato"):
                    recorder.start(mic_device=DeviceInfo(0, "Room / headset mic", 2, 44100, False),
                                   system_device=DeviceInfo(99, "Missing headset", 2, 48000, True))
            self.assertFalse(recorder.is_recording)
            self.assertTrue(backend.terminated)
            self.assertEqual(len(list(Path(folder).glob("*_mic.wav"))), 1)


if __name__ == "__main__":
    unittest.main()
