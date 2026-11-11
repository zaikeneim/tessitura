"""Native callback timing preserves short playback-idle gaps."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.audio.capture import _StreamRecorder
from src.audio.devices import DeviceInfo


class NativeLoopbackGapTests(unittest.TestCase):
    def test_fifty_millisecond_idle_gap_preserves_burst_positions(self):
        for rate in (16000, 48000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "system.wav"
                recorder = _StreamRecorder(None, DeviceInfo(0, "loopback", 1, rate, True), path)
                recorder.channels, recorder.sample_rate, recorder.started_at = 1, rate, 100.0
                recorder._wave = wave.open(str(path), "wb")
                recorder._wave.setnchannels(1)
                recorder._wave.setsampwidth(2)
                recorder._wave.setframerate(rate)
                burst = np.full(round(rate * 0.01), 1234, dtype="<i2").tobytes()
                recorder._queue.put((100.0, burst))
                recorder._queue.put((100.06, burst))
                recorder._queue.put((100.1, b""))
                recorder._queue.put(None)
                recorder._write_loop()
                recorder._wave.close()
                with wave.open(str(path)) as audio:
                    samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                self.assertEqual(len(samples), round(rate * 0.1))
                np.testing.assert_array_equal(samples[:round(rate * .01)], 1234)
                np.testing.assert_array_equal(samples[round(rate * .01):round(rate * .06)], 0)
                np.testing.assert_array_equal(samples[round(rate * .06):round(rate * .07)], 1234)


if __name__ == "__main__":
    unittest.main()
