"""Uscite delle finestre con lavori attivi.

Verifica che i dialoghi sui modelli non si chiudano con un download o
verifica in corso (Annulla, Escape, OK).

Esecuzione: python -m tests.test_gui_lifecycle
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QDialog

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.gui.model_settings import ModelSettingsDialog
from src.hardware import Hardware


class _FakeWorker:
    """Sta al posto di un QThread vivo, senza avviarne uno vero."""

    def __init__(self, running=True):
        self._running = running

    def isRunning(self):
        return self._running


def _hardware():
    return Hardware("cuda", "int8_float16", "large-v3", "GPU di prova", 8.0, 8)


class ModelSettingsExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self):
        with patch.object(ModelSettingsDialog, "_refresh", lambda *_a, **_k: None):
            return ModelSettingsDialog(_hardware(), "large-v3")

    def test_annulla_escape_e_ok_non_chiudono_con_un_lavoro_attivo(self):
        """Prima la protezione stava solo in closeEvent: reject() la scavalcava."""
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog._download_worker = _FakeWorker(running=True)
        closed = []

        with patch.object(QDialog, "done", lambda _self, code: closed.append(code)):
            dialog.reject()      # Annulla ed Escape passano da qui
            dialog.accept()      # anche il pulsante OK

        self.assertEqual(closed, [])
        self.assertIn("Attendi", dialog.status_label.text())

    def test_senza_lavori_attivi_la_finestra_si_chiude(self):
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        closed = []

        with patch.object(QDialog, "done", lambda _self, code: closed.append(code)):
            dialog.reject()

        self.assertEqual(closed, [QDialog.Rejected])

    def test_il_worker_resta_di_proprieta_fino_a_finished(self):
        """Il gestore del risultato non deve azzerare il riferimento al thread."""
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog._download_worker = _FakeWorker(running=True)

        dialog._on_download_done(object())

        self.assertIsNotNone(dialog._download_worker)
        self.assertTrue(dialog._busy())


if __name__ == "__main__":
    unittest.main()
