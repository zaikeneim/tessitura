"""Test unitari per AudioPlayer multitraccia (Passo 6)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from src.gui.audio_player import AudioPlayer
from src.models.session import Session

app = QApplication.instance() or QApplication([])


def _create_dummy_wav(path: Path, duration_s: float = 1.0):
    data = np.zeros(int(16000 * duration_s), dtype=np.float32)
    sf.write(str(path), data, 16000)


class AudioPlayerDualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.mix_wav = self.work / "mixed.wav"
        self.mic_wav = self.work / "mic.wav"
        self.sys_wav = self.work / "sys.wav"
        _create_dummy_wav(self.mix_wav)
        _create_dummy_wav(self.mic_wav)
        _create_dummy_wav(self.sys_wav)

        self.player = AudioPlayer()

    def tearDown(self):
        self.player.clear()
        self.tmp.cleanup()

    def test_single_track_hides_selector(self):
        self.player.load(self.mix_wav)
        self.assertTrue(self.player._track_combo.isHidden())
        self.assertEqual(len(self.player._tracks), 1)

    def test_dual_tracks_shows_selector(self):
        self.player.set_tracks(
            mix_path=self.mix_wav,
            mic_path=self.mic_wav,
            system_path=self.sys_wav,
        )
        self.assertFalse(self.player._track_combo.isHidden())
        self.assertEqual(self.player._track_combo.count(), 3)
        self.assertEqual(self.player._current_track_key, "mix")

    def test_switch_track_preserves_position(self):
        self.player.set_tracks(
            mix_path=self.mix_wav,
            mic_path=self.mic_wav,
            system_path=self.sys_wav,
        )
        # Imposta posizione
        self.player._slider.setRange(0, 1000)
        self.player._slider.setValue(500)

        # Switch a microfono
        mic_idx = self.player._track_combo.findData("mic")
        self.player._track_combo.setCurrentIndex(mic_idx)

        self.assertEqual(self.player._current_track_key, "mic")
        self.assertEqual(self.player._pending_seek_ms, 500)
        self.assertEqual(self.player._slider.value(), 500)


if __name__ == "__main__":
    unittest.main()
