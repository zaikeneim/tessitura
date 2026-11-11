"""Test d'integrazione GUI per il raggruppamento segmenti e preservazione dell'allineamento.

Esecuzione: python -m tests.test_gui_review
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PySide6.QtWidgets import QApplication

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.models.session import Segment, Session, Word


class GroupingAfterManualEditTests(unittest.TestCase):
    """Il raggruppamento deve restare disponibile dopo le correzioni a mano."""

    @classmethod
    def setUpClass(cls):
        from src.gui.theme import apply_theme
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def _window(self, session):
        from src.gui.main_window import MainWindow

        window = MainWindow()
        self.addCleanup(window.deleteLater)
        window.session = session
        window._display_transcript()
        window.speakers.set_session(session)
        window._set_busy(False)
        return window

    def test_riassegnare_una_voce_riabilita_il_raggruppamento(self):
        """Due segmenti diventano contigui della stessa voce solo dopo la correzione."""
        session = Session(segments=[
            Segment(0, 2, "Prima.", "SPEAKER_00", [Word(0, 2, "Prima.")]),
            Segment(2, 4, "Seconda.", "SPEAKER_01", [Word(2, 4, "Seconda.")]),
        ])
        window = self._window(session)
        self.assertFalse(window.group_segments_action.isEnabled())

        session.edit_segment(1, speaker="SPEAKER_00")
        window._on_speaker_reassigned(1, "SPEAKER_00")

        self.assertTrue(window.group_segments_action.isEnabled())

    def test_il_raggruppamento_si_disabilita_quando_non_serve_piu(self):
        session = Session(segments=[
            Segment(0, 2, "Prima.", "SPEAKER_00", [Word(0, 2, "Prima.")]),
            Segment(2, 4, "Seconda.", "SPEAKER_00", [Word(2, 4, "Seconda.")]),
        ])
        window = self._window(session)
        self.assertTrue(window.group_segments_action.isEnabled())

        session.group_adjacent_speaker_segments()
        window._mark_dirty()

        self.assertFalse(window.group_segments_action.isEnabled())

    def test_una_modifica_al_testo_non_spegne_il_raggruppamento(self):
        session = Session(segments=[
            Segment(0, 2, "Prima.", "SPEAKER_00", [Word(0, 2, "Prima.")]),
            Segment(2, 4, "Seconda.", "SPEAKER_00", [Word(2, 4, "Seconda.")]),
        ])
        window = self._window(session)

        session.edit_segment(0, text="Prima, corretta.")
        window._mark_dirty()

        self.assertTrue(window.group_segments_action.isEnabled())


class MergePreservesAlignmentStateTests(unittest.TestCase):
    def test_unire_un_segmento_corretto_a_mano_non_riabilita_l_allineamento(self):
        session = Session(segments=[
            Segment(0, 1, "uno", "A", [Word(0, 1, "uno")]),
            Segment(1, 2, "due", "A", [Word(1, 2, "due")]),
        ])
        session.edit_segment(1, text="due corretto")

        self.assertEqual(session.group_adjacent_speaker_segments(), 1)
        self.assertFalse(session.segments[0].words_valid)
        self.assertIn("timing_da_riallineare", session.segments[0].warnings)


if __name__ == "__main__":
    unittest.main()
