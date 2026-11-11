"""Test per il dialog a due passi di selezione modalità registrazione (Passo 4)."""

from __future__ import annotations

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QDialog

from src.gui.recording_mode_dialog import RecordingModeDialog, ask_recording_mode

app = QApplication.instance() or QApplication([])


class RecordingModeDialogTests(unittest.TestCase):
    def test_step1_standard_selection(self):
        """Selezionare l'opzione standard chiude il dialog con strategy standard."""
        dlg = RecordingModeDialog()
        # Simula click su standard
        dlg._on_select_standard()
        self.assertEqual(dlg.result(), QDialog.Accepted)
        self.assertEqual(dlg.strategy, "standard")
        self.assertTrue(dlg.headphones_confirmed)

    def test_step1_to_step2_and_headphones_confirmed(self):
        """Selezionare duale passa a Step 2, e confermare le cuffie imposta source_single_speaker con cuffie."""
        dlg = RecordingModeDialog()
        self.assertEqual(dlg.stack.currentIndex(), 0)

        # Click su dual
        dlg._on_select_source_single_speaker()
        self.assertEqual(dlg.stack.currentIndex(), 1)

        # Conferma cuffie
        dlg._on_confirm_headphones()
        self.assertEqual(dlg.result(), QDialog.Accepted)
        self.assertEqual(dlg.strategy, "source_aware")
        self.assertTrue(dlg.headphones_confirmed)

    def test_step2_unconfirmed_headphones(self):
        """Duale con altoparlanti imposta source_single_speaker con headphones_confirmed=False (bleed check)."""
        dlg = RecordingModeDialog()
        dlg._on_select_source_single_speaker()
        dlg._on_unconfirmed_headphones()
        self.assertEqual(dlg.result(), QDialog.Accepted)
        self.assertEqual(dlg.strategy, "source_aware")
        self.assertFalse(dlg.headphones_confirmed)

    def test_step2_escape_returns_to_step1(self):
        """Premere Esc in Step 2 torna a Step 1 senza chiudere il dialog."""
        dlg = RecordingModeDialog()
        dlg._on_select_source_single_speaker()
        self.assertEqual(dlg.stack.currentIndex(), 1)

        # Simula tasto Esc
        event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        dlg.keyPressEvent(event)
        self.assertEqual(dlg.stack.currentIndex(), 0)

    def test_step2_reject_conservative_fallback(self):
        """Annullare in Step 2 adotta fallback conservativo source_single_speaker con bleed check attivo."""
        dlg = RecordingModeDialog()
        dlg._on_select_source_single_speaker()
        dlg.reject()
        self.assertEqual(dlg.result(), QDialog.Accepted)
        self.assertEqual(dlg.strategy, "source_aware")
        self.assertFalse(dlg.headphones_confirmed)


if __name__ == "__main__":
    unittest.main()
