"""Test unitari per ConversationView (Passo 5)."""

from __future__ import annotations

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from src.gui.conversation_view import ConversationBubbleWidget, ConversationView
from src.gui.transcript_controller import TranscriptController
from src.models.session import Session
from src.models.transcript_artifacts import Utterance

app = QApplication.instance() or QApplication([])


class ConversationViewTests(unittest.TestCase):
    def setUp(self):
        self.session = Session(
            name="Test Conv Session",
            language="it",
            audio_profile="mic_system",
            utterances=[
                Utterance(
                    start_s=0.0,
                    end_s=2.5,
                    utterance_id="utt_01",
                    stream_id=0,
                    speaker="SPEAKER_MIC",
                    text="Ciao, come stai?",
                ),
                Utterance(
                    start_s=2.0,
                    end_s=4.0,
                    utterance_id="utt_02",
                    stream_id=1,
                    overlap_group_id="ov_1",
                    speaker="SPEAKER_SYS",
                    text="Tutto bene, grazie!",
                ),
            ],
            speaker_names={"SPEAKER_MIC": "Io", "SPEAKER_SYS": "Remoto"},
        )
        self.controller = TranscriptController()
        self.view = ConversationView(self.controller)

    def test_load_session_creates_bubbles_chronologically(self):
        self.controller.load_session(self.session)
        self.assertEqual(len(self.view._bubbles), 2)
        b1, b2 = self.view._bubbles[0], self.view._bubbles[1]

        self.assertEqual(b1.utterance_id, "utt_01")
        self.assertEqual(b1.display_name, "Io")
        self.assertEqual(b1.text_label.text(), "Ciao, come stai?")

        self.assertEqual(b2.utterance_id, "utt_02")
        self.assertEqual(b2.display_name, "Remoto")
        self.assertEqual(b2.utterance.overlap_group_id, "ov_1")

    def test_seek_requested_on_time_button_click(self):
        self.controller.load_session(self.session)
        seek_target = []
        self.controller.seekRequested.connect(lambda s: seek_target.append(s))

        b2 = self.view._bubbles[1]
        b2.time_btn.click()
        self.assertEqual(seek_target, [2.0])

    def test_text_edit_synchronizes_controller_and_session(self):
        self.controller.load_session(self.session)
        b1 = self.view._bubbles[0]

        # Simula modifica testo in-place
        b1._toggle_edit()
        b1.editor.setPlainText("Ciao, come stai oggi?")
        b1._save_edit()

        # Verifica sincronizzazione su view, controller e session
        self.assertEqual(b1.text_label.text(), "Ciao, come stai oggi?")
        self.assertEqual(self.session.utterances[0].text, "Ciao, come stai oggi?")

    def test_playback_position_highlights_multiple_active(self):
        self.controller.load_session(self.session)
        # A t=2.2s entrambi gli interventi sono attivi
        self.controller.update_playback_position(2.2)

        b1, b2 = self.view._bubbles[0], self.view._bubbles[1]
        self.assertTrue(b1._is_active)
        self.assertTrue(b2._is_active)

        # A t=3.0s solo il secondo è attivo
        self.controller.update_playback_position(3.0)
        self.assertFalse(b1._is_active)
        self.assertTrue(b2._is_active)

    def test_speakers_renamed_updates_bubble_labels(self):
        self.controller.load_session(self.session)
        self.controller.rename_speaker("SPEAKER_MIC", "Alessandro")

        b1, b2 = self.view._bubbles[0], self.view._bubbles[1]
        self.assertEqual(b1.speaker_label.text(), "Alessandro")
        self.assertEqual(b2.speaker_label.text(), "Remoto")


if __name__ == "__main__":
    unittest.main()
