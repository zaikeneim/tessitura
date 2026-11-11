"""Test per TranscriptController (Passo 2).

Verifica editing per ID atomico, sincronizzazione Utterance/Segment,
gestione viste a tre vie, undo/redo e identificazione di evidenziazioni simultanee.
"""

from __future__ import annotations

import unittest
from PySide6.QtCore import QCoreApplication

from src.models.session import Segment, Session, Word
from src.models.transcript_artifacts import Utterance
from src.gui.transcript_controller import TranscriptController

app = QCoreApplication.instance() or QCoreApplication([])


class TranscriptControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = TranscriptController()
        self.session = Session(
            name="Sessione Controller Test",
            segments=[
                Segment(start=0.0, end=3.0, text="Primo intervento", speaker="S0", id="u1"),
                Segment(start=1.5, end=4.0, text="Intervento sovrapposto", speaker="S1", id="u2"),
            ],
            utterances=[
                Utterance(start_s=0.0, end_s=3.0, text="Primo intervento", speaker="S0", utterance_id="u1", stream_id=0, overlap_group_id="ov_1"),
                Utterance(start_s=1.5, end_s=4.0, text="Intervento sovrapposto", speaker="S1", utterance_id="u2", stream_id=1, overlap_group_id="ov_1"),
            ],
            speaker_names={"S0": "Alice", "S1": "Bob"},
        )
        self.controller.load_session(self.session)

    def test_view_switching_without_recalculation(self):
        """Il cambio vista aggiorna lo stato della sessione ed emette il segnale."""
        emitted_views = []
        self.controller.activeViewChanged.connect(emitted_views.append)

        self.assertEqual(self.controller.active_view, "timeline")
        self.controller.set_active_view("timeline")
        self.assertEqual(self.controller.active_view, "timeline")
        self.assertEqual(self.session.active_view, "timeline")

        self.controller.set_active_view("table")
        self.assertEqual(self.controller.active_view, "table")
        self.assertEqual(self.session.active_view, "table")

        self.assertEqual(emitted_views, ["table"])

    def test_edit_text_synchronizes_utterance_and_segment(self):
        """La modifica del testo aggiorna Utterance e Segment atomico e supporta undo/redo."""
        edited_signals = []
        self.controller.utteranceEdited.connect(lambda uid, txt: edited_signals.append((uid, txt)))

        res = self.controller.edit_text("u1", "Nuovo testo per u1")
        self.assertTrue(res)
        self.assertEqual(self.session.utterances[0].text, "Nuovo testo per u1")
        self.assertEqual(self.session.segments[0].text, "Nuovo testo per u1")
        self.assertEqual(len(edited_signals), 1)
        self.assertEqual(edited_signals[0], ("u1", "Nuovo testo per u1"))

        # Undo
        self.assertTrue(self.controller.undo())
        self.assertEqual(self.session.utterances[0].text, "Primo intervento")
        self.assertEqual(self.session.segments[0].text, "Primo intervento")

        # Redo
        self.assertTrue(self.controller.redo())
        self.assertEqual(self.session.utterances[0].text, "Nuovo testo per u1")
        self.assertEqual(self.session.segments[0].text, "Nuovo testo per u1")

    def test_reassign_speaker(self):
        """La riassegnazione dello speaker aggiorna sia Utterance che Segment."""
        changed_signals = []
        self.controller.speakerChanged.connect(lambda uid, spk: changed_signals.append((uid, spk)))

        res = self.controller.reassign_speaker("u2", "S0")
        self.assertTrue(res)
        self.assertEqual(self.session.utterances[1].speaker, "S0")
        self.assertEqual(self.session.segments[1].speaker, "S0")
        self.assertEqual(changed_signals, [("u2", "S0")])

    def test_simultaneous_highlights_at_playback_position(self):
        """A t=2.0s sia u1 (0-3s) che u2 (1.5-4s) risultano contemporaneamente attivi."""
        highlights = self.controller.update_playback_position(2.0)
        self.assertIn("u1", highlights)
        self.assertIn("u2", highlights)
        self.assertEqual(len(highlights), 2)

        # A t=0.5s solo u1 è attivo
        highlights_early = self.controller.update_playback_position(0.5)
        self.assertEqual(highlights_early, ["u1"])

        # A t=3.5s solo u2 è attivo
        highlights_late = self.controller.update_playback_position(3.5)
        self.assertEqual(highlights_late, ["u2"])

        # A t=5.0s nessuno è attivo
        highlights_none = self.controller.update_playback_position(5.0)
        self.assertEqual(highlights_none, [])


if __name__ == "__main__":
    unittest.main()
