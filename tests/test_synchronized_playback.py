"""Solo playback must use the same global timeline as mixed playback and ASR."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()
from PySide6.QtWidgets import QApplication
from src.gui.audio_player import AudioPlayer
from src.models.session import AudioSources, Session


class SynchronizedPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_player_prefers_globally_aligned_derivatives_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mix, mic, system, mic_view, system_view = [root / name for name in
                ("mixed.wav", "mic.wav", "system.wav", "mic-view.wav", "system-view.wav")]
            for path in (mix, mic, system, mic_view, system_view):
                path.write_bytes(b"fixture")
            session = Session(audio=AudioSources(str(mic), str(system), str(mix)), capture_manifest={
                "synchronization": {"sources": {"mic": {"playback_audio": str(mic_view)},
                                               "system": {"playback_audio": str(system_view)}}}})
            player = AudioPlayer()
            with patch.object(player, "set_tracks") as tracks:
                player.load_session_audio(session)
            self.assertEqual(tracks.call_args.kwargs, dict(mix_path=mix, mic_path=mic_view, system_path=system_view))
            self.assertEqual(session.audio.mic, str(mic))
            self.assertEqual(session.audio.system, str(system))
            mic_view.unlink()
            with patch.object(player, "set_tracks") as tracks:
                player.load_session_audio(session)
            self.assertEqual(tracks.call_args.kwargs["mic_path"], mic)
            player.clear()


if __name__ == "__main__":
    unittest.main()
