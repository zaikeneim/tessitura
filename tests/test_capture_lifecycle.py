"""Bounded removal cleanup and trustworthy live capture indicators."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.audio.capture import Recorder, _StreamRecorder
from src.audio.devices import DeviceInfo


class CaptureLifecycleTests(unittest.TestCase):
    def test_failed_native_stop_closes_stream_before_waiting_for_reader(self):
        released = threading.Event()
        class RemovedDriver:
            def stop_stream(self):
                raise OSError("removed device")
            def close(self):
                released.set()
        with tempfile.TemporaryDirectory() as folder:
            rec = _StreamRecorder(None, DeviceInfo(0, "mic", 1, 16000, False), Path(folder) / "mic.wav")
            rec._stream = RemovedDriver()
            rec._reader = threading.Thread(target=released.wait, daemon=True)
            rec._reader.start()
            started = time.perf_counter()
            rec.stop()
            self.assertLess(time.perf_counter() - started, 1)
            self.assertFalse(rec._reader.is_alive())
            self.assertIn("removed device", str(rec.error))

    def test_idle_loopback_meter_decays_without_losing_peak_presence(self):
        recorder = Recorder()
        recorder._sys_rec = SimpleNamespace(level=.2, peak=.6, last_packet_at=10.)
        with patch("src.audio.capture.time.perf_counter", return_value=10.1):
            self.assertEqual(recorder.levels()["system"], .2)
        with patch("src.audio.capture.time.perf_counter", return_value=10.3):
            self.assertEqual(recorder.levels()["system"], 0)
        self.assertEqual(recorder._sys_rec.peak, .6)

    def test_one_missing_adc_timestamp_permanently_marks_clock_uncertain(self):
        rec = _StreamRecorder(None, DeviceInfo(0, "loopback", 1, 16000, True), Path("unused.wav"))
        rec._clock_origin = 90
        data = np.full(160, 100, dtype="<i2").tobytes()
        rec._capture_callback(data, 160, {"input_buffer_adc_time": 0}, 0)
        rec._capture_callback(data, 160, {"input_buffer_adc_time": 10}, 0)
        rec._capture_callback(data, 160, {"input_buffer_adc_time": 11}, 0)
        self.assertEqual(rec.timestamp_fallbacks, 1)
        self.assertEqual(rec.timing_source, "mixed_adc_and_estimated_clock")


if __name__ == "__main__":
    unittest.main()
