"""Desktop regressions for one canonical, non-destructive multitrack transcript.

These tests exercise the real window/controller/storage wiring. Audio inference
and recording hardware are unnecessary; all audio fixtures are short local WAVs.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThreadPool
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.gui.speaker_timeline_view import SpeakerTimelineView
from src.gui.transcript_controller import TranscriptController
from src.hardware import Hardware
from src.models import storage
from src.models.session import AudioSources, Segment, Session
from src.models.transcript_artifacts import Utterance


class TimelineDesktopIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "small"),
        ):
            self.window = MainWindow()
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.window.timeline_view._waveform_timer.stop()
        self.window._dirty = False
        self.window.player.clear()
        self.window.close()
        self.window.deleteLater()
        QThreadPool.globalInstance().waitForDone(3000)
        self.app.processEvents()
        self.folder.cleanup()

    def _display(self, session):
        self.window.session = session
        self.window._showing_raw = False
        self.window._display_transcript()
        self.app.processEvents()
        return self.window.controller

    def _assert_views_show(self, expected):
        """Compare event ID, spoken text and displayed speaker in both presentations."""
        session = self.window.controller.session
        bubbles = self.window.conversation_view._bubble_by_id
        self.assertEqual(set(bubbles), set(expected))
        for event_id, (text, speaker_id) in expected.items():
            bubble = bubbles[event_id]
            self.assertEqual(bubble.text_label.text(), text)
            self.assertEqual(bubble.speaker_label.text(), session.display_name(speaker_id))
        self.window.timeline_view.canvas.set_window(90, 30)
        blocks = {block["utterance"].event_id: block["utterance"]
                  for block in self.window.timeline_view.canvas._blocks}
        self.assertEqual(set(blocks), set(expected))
        for event_id, (text, speaker_id) in expected.items():
            self.assertEqual((blocks[event_id].text, blocks[event_id].speaker_id), (text, speaker_id))

    def test_manual_overlap_split_assignment_survives_atomic_save_and_reload(self):
        session = Session(schema_version=2, segments=[Segment(99, 104, "Existing stream", "A", id="original")])
        controller = self._display(session)
        self.assertIs(controller.session, session)
        event = controller.create_event(99, 103, "Missing voice", "B")
        self.assertTrue(controller.split_event(event.event_id, 101, "Missing", "voice"))
        parts = sorted((u for u in session.utterances if u.event_id != "original"), key=lambda u: u.start_s)
        speaker_id = controller.create_speaker("Corrected speaker")
        self.assertTrue(controller.reassign_speaker(parts[0].event_id, speaker_id))
        expected = {"original": ("Existing stream", "A"), parts[0].event_id: ("Missing", speaker_id),
                    parts[1].event_id: ("voice", "B")}
        self._assert_views_show(expected)
        self.assertTrue(self.window._dirty)
        self.assertEqual({u.start_s for u in session.utterances if u.start_s == 99}, {99})
        self.assertEqual(sum(u.start_s == 99 for u in session.utterances), 2)

        path = self.root / "session.json"
        path.write_text('{"previous": "complete file"}', encoding="utf-8")
        previous = path.read_bytes()
        with patch("src.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(path), "JSON (*.json)")), patch(
            "src.models.storage.os.replace", side_effect=OSError("replace failed"),
        ), patch("src.gui.main_window.QMessageBox.critical"):
            self.window._save_session()
        self.assertEqual(path.read_bytes(), previous, "A failed atomic save must preserve the previous complete file.")
        self.assertTrue(self.window._dirty)
        with patch("src.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(path), "JSON (*.json)")):
            self.window._save_session()
        self.assertFalse(self.window._dirty)
        loaded = storage.load_session(path)
        self.assertEqual([u.to_dict() for u in loaded.utterances], [u.to_dict() for u in session.utterances])
        self.assertEqual(len(loaded.edit_history), len(session.edit_history))
        self._display(loaded)
        self._assert_views_show(expected)
        self.assertTrue(self.window.controller.undo())
        self.assertEqual(next(u for u in loaded.utterances if u.event_id == parts[0].event_id).speaker_id, "B")
        self.assertTrue(self.window.controller.redo())
        self._assert_views_show(expected)

    def test_legacy_raw_preview_shows_original_without_mutating_corrections(self):
        session = Session(schema_version=2,
                          segments=[Segment(99, 104, "Original words", "A", id="legacy")],
                          raw_segments=[Segment(99, 104, "Original words", "A", id="legacy")])
        controller = self._display(session)
        self.assertTrue(controller.edit_text("legacy", "Verified correction"))
        self._assert_views_show({"legacy": ("Verified correction", "A")})
        self.window._toggle_raw_transcript(True)
        self._assert_views_show({"legacy": ("Original words", "A")})
        self.assertEqual(self.window.transcript.item(0, 2).text(), "Original words")
        self.assertTrue(controller.is_read_only)
        self.assertFalse(controller.edit_text("legacy", "Must not change raw"))
        self.assertEqual(session.utterances[0].text, "Verified correction")
        self.window._toggle_raw_transcript(False)
        self.assertIs(controller.session, session)
        self._assert_views_show({"legacy": ("Verified correction", "A")})

    def test_loading_another_session_keeps_comparator_player_and_waveform_consistent(self):
        sessions = []
        for index in (1, 2):
            original, processed = self.root / f"original-{index}.wav", self.root / f"processed-{index}.wav"
            samples = np.sin(np.arange(3200, dtype=np.float32) * .03) * .1
            sf.write(original, np.column_stack((samples, -samples)), 16000)
            sf.write(processed, samples, 16000)
            sessions.append(Session(audio=AudioSources(mic=str(original), mixed=str(processed)),
                                    audio_profile="stereo_ambient"))
        self._display(sessions[0])
        self.assertTrue(self.window._load_audio(sessions[0]))
        combo = self.window.timeline_view.audio_variant
        combo.setCurrentIndex(combo.findData("processed"))
        combo.setCurrentIndex(combo.findData("original"))
        self.assertEqual(Path(self.window.player._player.source().toLocalFile()), Path(sessions[0].audio.mic))
        self.assertEqual(Path(self.window.timeline_view._audio_path), Path(sessions[0].audio.mic))

        self._display(sessions[1])
        self.assertTrue(self.window._load_audio(sessions[1]))
        actual = Path(self.window.player._player.source().toLocalFile())
        self.assertIn(actual, [Path(sessions[1].audio.mic), Path(sessions[1].audio.mixed)])
        expected_variant = "original" if actual == Path(sessions[1].audio.mic) else "processed"
        self.assertEqual(combo.currentData(), expected_variant, "The comparison label must describe the playing source.")
        self.assertEqual(Path(self.window.timeline_view._audio_path), actual, "Waveform and playback need the same source.")

    def test_region_playback_stops_at_end_and_normal_seek_clears_the_boundary(self):
        backend = MagicMock()
        backend.playbackState.return_value = QMediaPlayer.StoppedState
        backend.duration.return_value = 7200000
        with patch.object(self.window.player, "_player", backend):
            self.window.timeline_view._select_region(99, 101)
            self.window.timeline_view._play_region()
            backend.setPosition.assert_called_with(99000)
            backend.play.assert_called_once()
            self.window.player._on_position(100999)
            backend.pause.assert_not_called()
            self.window.player._on_position(101000)
            backend.pause.assert_called_once()
            self.window.player._on_position(101001)
            backend.pause.assert_called_once()
            self.window.player.play_region(103, 105)
            self.window.player.seek_seconds(110)
            backend.pause.reset_mock()
            self.window.player._on_position(111000)
            backend.pause.assert_not_called()

    def test_two_hour_navigation_keeps_only_visible_blocks_in_a_viewport_sized_canvas(self):
        controller = TranscriptController()
        view = SpeakerTimelineView(controller)
        try:
            events = [Utterance(start_s=float(start), end_s=float(start + 2), speaker="A", text=f"Event {start}")
                      for start in range(0, 7200, 10)]
            controller.load_session(Session(utterances=events, media_duration_s=7200))
            view.resize(1200, 700)
            view.show()
            self.app.processEvents()
            self.assertLessEqual(view.canvas.width(), view.scroll.viewport().width())
            self.assertLess(len(view.canvas._blocks), 10)
            view.position.setValue(7140000)
            self.app.processEvents()
            self.assertGreaterEqual(view.canvas.window_start_s, 7140)
            self.assertTrue(any(block["utterance"].start_s == 7190 for block in view.canvas._blocks))
            self.assertLess(len(view.canvas._blocks), 10)
            self.assertLessEqual(view.canvas.width(), view.scroll.viewport().width())
            self.assertLess(view.canvas.height(), 1000)
        finally:
            view._waveform_timer.stop()
            view.close()
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
