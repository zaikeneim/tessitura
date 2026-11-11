"""Workflow navigation must preserve actions, options and compact-screen access."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from pathlib import Path
from unittest.mock import patch

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton
from src.gui.main_window import MainWindow
from src.gui.theme import apply_theme
from src.hardware import Hardware
from src.models.session import Segment, Session


class WorkflowNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if font.exists():
                QFontDatabase.addApplicationFont(str(font))
        apply_theme(cls.app)

    def setUp(self):
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect",
            return_value=Hardware("cpu", "int8", "small", None, 0),
        ):
            self.window = MainWindow()
        self.window.resize(1000, 700)
        self.window.show()
        QTest.qWait(100)

    def tearDown(self):
        self.window._dirty = False
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_navigation_preserves_values_and_primary_actions_fit(self):
        w = self.window
        w.speakers_spin.setValue(4)
        w.glossary_edit.setText("Tessitura")
        for index in range(w.control_tabs.count()):
            w.control_tabs.setCurrentIndex(index)
            self.app.processEvents()
            self.assertTrue(w.process_file_btn.isVisible())
            self.assertTrue(w.process_btn.isVisible())
            self.assertEqual(w.speakers_spin.value(), 4)
            self.assertEqual(w.glossary_edit.text(), "Tessitura")
            scroll = w.control_tabs.widget(index)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            self.assertTrue(w.centralWidget().rect().contains(
                w.process_btn.mapTo(w.centralWidget(), w.process_btn.rect().bottomRight())
            ))

    def test_disclosure_keyboard_and_speaker_panel(self):
        w = self.window
        w.control_tabs.setCurrentIndex(1)
        self.assertFalse(w.advanced_options.isVisible())
        w.advanced_toggle.setFocus()
        QTest.keyClick(w.advanced_toggle, Qt.Key_Space)
        self.assertTrue(w.advanced_options.isVisible())
        self.assertFalse(w.denoise_check.isChecked())
        self.assertTrue(w.diarize_check.isChecked())
        QTest.keyClick(w.advanced_toggle, Qt.Key_Space)
        self.assertFalse(w.advanced_options.isVisible())
        self.assertFalse(w.denoise_check.isChecked())
        self.assertTrue(w.diarize_check.isChecked())
        w.voices_toggle.click()
        self.assertFalse(w.speaker_sidebar.isVisible())
        w.voices_toggle.click()
        self.assertTrue(w.speaker_sidebar.isVisible())

    def test_toolbar_actions_follow_session_and_busy_state(self):
        w = self.window
        actions = (w.group_segments_action, w.raw_transcript_action)
        buttons = [b for b in w.findChildren(QToolButton) if b.defaultAction() in actions]
        self.assertEqual(len(buttons), 2)
        self.assertTrue(all(not b.isEnabled() for b in buttons))
        w.session = Session(
            segments=[Segment(0, 1, "Prima", "SPEAKER_00"), Segment(1, 2, "Seconda", "SPEAKER_00")],
            raw_segments=[Segment(0, 1, "Prima", "SPEAKER_00"), Segment(1, 2, "Seconda", "SPEAKER_00")],
        )
        w._set_busy(False)
        self.assertTrue(all(b.isEnabled() for b in buttons))
        self.assertEqual({b.text() for b in buttons}, {"Raggruppa voci", "Trascrizione grezza"})
        w._set_busy(True)
        self.assertTrue(all(not b.isEnabled() for b in buttons))


if __name__ == "__main__":
    unittest.main()
