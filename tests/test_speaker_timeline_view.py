"""Test unitari per SpeakerTimelineView (Passo 6)."""

from __future__ import annotations

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from src.gui.speaker_timeline_view import SpeakerTimelineView
from src.gui.transcript_controller import TranscriptController
from src.models.session import Session
from src.models.transcript_artifacts import Utterance

app = QApplication.instance() or QApplication([])


class SpeakerTimelineViewTests(unittest.TestCase):
    def setUp(self):
        self.session = Session(
            name="Sessione Timeline",
            language="it",
            audio_profile="mic_system",
            utterances=[
                Utterance(
                    start_s=1.0,
                    end_s=3.0,
                    utterance_id="utt_mic",
                    stream_id=0,
                    speaker="SPEAKER_MIC",
                    text="Parla il microfono",
                ),
                Utterance(
                    start_s=2.0,
                    end_s=5.0,
                    utterance_id="utt_sys",
                    stream_id=1,
                    overlap_group_id="ov_1",
                    speaker="SPEAKER_SYS",
                    text="Parla il sistema in sovrapposizione",
                ),
            ],
            speaker_names={"SPEAKER_MIC": "Docente", "SPEAKER_SYS": "Studente"},
        )
        self.controller = TranscriptController()
        self.view = SpeakerTimelineView(self.controller)

    def test_load_session_initializes_columns_and_blocks(self):
        self.controller.load_session(self.session)
        canvas = self.view.canvas
        self.assertEqual(len(canvas._speakers), 3)
        self.assertIn(None, canvas._speakers)
        self.assertIn("SPEAKER_MIC", canvas._speakers)
        self.assertIn("SPEAKER_SYS", canvas._speakers)
        self.assertEqual(len(canvas._blocks), 2)

    def test_overlapping_blocks_aligned_in_time(self):
        self.controller.load_session(self.session)
        canvas = self.view.canvas
        b_mic = next(b for b in canvas._blocks if b["utterance"].utterance_id == "utt_mic")
        b_sys = next(b for b in canvas._blocks if b["utterance"].utterance_id == "utt_sys")

        # Corsie distinte sull'asse verticale, tempo assoluto sull'asse orizzontale.
        self.assertNotEqual(b_mic["rect"].x(), b_sys["rect"].x())

        # Sovrapposizione verticale (y range interseca)
        mic_top = b_mic["rect"].top()
        mic_bottom = b_mic["rect"].bottom()
        sys_top = b_sys["rect"].top()
        sys_bottom = b_sys["rect"].bottom()

        self.assertNotEqual(mic_top, sys_top)
        self.assertTrue(b_mic["rect"].right() > b_sys["rect"].left() and b_sys["rect"].right() > b_mic["rect"].left())

    def test_selection_populates_inspector(self):
        self.controller.load_session(self.session)
        self.view._on_utterance_selected("utt_mic")

        self.assertIn("Docente", self.view.insp_speaker.text())
        self.assertEqual(self.view.insp_text.toPlainText(), "Parla il microfono")

    def test_inspector_edit_updates_controller_and_session(self):
        self.controller.load_session(self.session)
        self.view._on_utterance_selected("utt_mic")

        self.view.insp_text.setPlainText("Parla il docente corretto")
        self.view._save_inspector_text()

        self.assertEqual(self.session.utterances[0].text, "Parla il docente corretto")

    def test_autoscroll_toggle_on_user_scroll(self):
        self.controller.load_session(self.session)
        self.assertTrue(self.view._auto_scroll)
        self.assertTrue(self.view.follow_btn.isHidden())

        # Utente scorre manualmente
        self.view._on_user_scrolled()
        self.assertFalse(self.view._auto_scroll)
        self.assertFalse(self.view.follow_btn.isHidden())

        # Utente preme "Segui riproduzione"
        self.view._resume_autoscroll()
        self.assertTrue(self.view._auto_scroll)
        self.assertTrue(self.view.follow_btn.isHidden())


if __name__ == "__main__":
    unittest.main()
