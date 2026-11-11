"""Test dell'integrazione dei Progetti nella GUI di MainWindow.

Esecuzione: python -m tests.test_project_gui
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QFileDialog

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.gui.main_window import MainWindow
from src.models.project import open_or_create
from src.models.session import Segment, Session


class ProjectGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    def _audio(self, *names):
        for name in names:
            (self.folder / name).write_bytes(b"finto-audio")

    def test_combine_audio_apre_file_dialog_senza_errori(self):
        """Verifica la risoluzione dell'UnboundLocalError in _combine_audio."""
        dialog_calls = []
        with patch.object(
            QFileDialog, "getOpenFileNames",
            lambda *args, **kwargs: dialog_calls.append((args, kwargs)) or ([], "")
        ):
            self.window._combine_audio()

        self.assertEqual(len(dialog_calls), 1)
        # Verifica che il filtro supporti tutti i formati
        filter_str = dialog_calls[0][0][3]
        self.assertIn("*.wav", filter_str)
        self.assertIn("*.opus", filter_str)
        self.assertIn("*.aac", filter_str)

    def test_apri_e_chiudi_progetto_aggiorna_interfaccia(self):
        project = open_or_create(self.folder, "Campagna Gioco")
        project.description = "D&D campagna 2026"
        project.save()

        self.window._open_project(self.folder)

        self.assertIsNotNone(self.window.project)
        self.assertEqual(self.window.project.name, "Campagna Gioco")
        self.assertIn("Campagna Gioco", self.window.project_label.text())
        self.assertTrue(self.window.project_details_btn.isEnabled())
        self.assertTrue(self.window.project_details_action.isEnabled())
        self.assertTrue(self.window.close_project_action.isEnabled())

        # Chiudi progetto
        self.window._close_project()
        self.assertIsNone(self.window.project)
        self.assertIn("Nessun progetto", self.window.project_label.text())
        self.assertFalse(self.window.project_details_btn.isEnabled())

    def test_glossario_progetto_alimenta_prompt_whisper(self):
        project = open_or_create(self.folder, "Progetto Lavoro")
        project.set_glossary(["AcmeCorp", "SprintReview"])
        project.save()

        self.window._open_project(self.folder)
        self.window.audio_profile_combo.setCurrentIndex(1)  # Seleziona profilo
        self.window.glossary_edit.setText("TermineEstemporaneo")

        options = self.window._build_options()

        self.assertIn("AcmeCorp", options.initial_prompt)
        self.assertIn("SprintReview", options.initial_prompt)
        self.assertIn("TermineEstemporaneo", options.initial_prompt)

    def test_salvataggio_sessione_aggiorna_memoria_progetto(self):
        project = open_or_create(self.folder, "Progetto Meeting")
        project.save()
        self.window._open_project(self.folder)

        audio_file = self.folder / "meeting_01.wav"
        audio_file.write_bytes(b"audio")
        self.window.last_mic = audio_file

        session = Session(
            name="Meeting 1",
            segments=[Segment(0, 5, "Buongiorno a tutti", "SPEAKER_00")],
        )
        session.rename_speaker("SPEAKER_00", "Alice")

        self.window._on_done(session)

        # La sessione e' memorizzata nel progetto
        self.assertEqual(len(self.window.project.sessions), 1)
        self.assertIn("Alice", self.window.project.speakers)
        self.assertEqual(self.window.project.sessions[0].name, "Meeting 1")
        self.assertIn("Meeting 1", self.window.project.memory_text())

    def test_salvare_promuove_correzioni_confermate_nella_memoria_locale(self):
        project = open_or_create(self.folder, "Progetto adattivo")
        project.save()
        self.window._open_project(self.folder)
        session = Session(
            segments=[Segment(0, 2, "Parla Chisrael", "SPEAKER_00")],
            raw_segments=[Segment(0, 2, "Parla Chisrael", "SPEAKER_00")],
        )
        session.speaker_names["SPEAKER_00"] = "Alice"
        session.diarization = {
            "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0]},
            "embedding_model": {"repo": "pyannote/test", "revision": "r1"},
        }
        self.window.session = session

        self.window._on_speaker_name_confirmed("SPEAKER_00", "Alice")
        session.edit_segment(0, text="Parla Kisrael")
        with patch.object(
            QFileDialog, "getSaveFileName",
            return_value=(str(self.folder / "sessione-corretta.json"), "JSON (*.json)"),
        ):
            self.window._save_session()
        reread = open_or_create(self.folder)

        self.assertIn("Kisrael", reread.glossary_prompt())
        self.assertIn("Alice", reread.speaker_profiles(
            {"repo": "pyannote/test", "revision": "r1"}
        ))

    def test_una_sessione_non_salvata_non_inquina_il_progetto(self):
        project = open_or_create(self.folder, "Progetto adattivo")
        project.save()
        self.window._open_project(self.folder)
        session = Session(
            segments=[Segment(0, 2, "Parla Chisrael", "SPEAKER_00")],
            raw_segments=[Segment(0, 2, "Parla Chisrael", "SPEAKER_00")],
        )
        session.speaker_names["SPEAKER_00"] = "Alice"
        session.diarization["speaker_embeddings"] = {"SPEAKER_00": [1.0, 0.0]}
        self.window.session = session

        self.window._on_speaker_name_confirmed("SPEAKER_00", "Alice")
        session.edit_segment(0, text="Parla Kisrael")
        reread = open_or_create(self.folder)

        self.assertEqual(reread.learning_sessions, {})

    def test_errore_memoria_non_annulla_il_salvataggio_della_sessione(self):
        project = open_or_create(self.folder, "Progetto adattivo")
        project.save()
        self.window._open_project(self.folder)
        self.window.session = Session(
            segments=[Segment(0, 1, "Chisrael")],
            raw_segments=[Segment(0, 1, "Chisrael")],
        )
        self.window.session.edit_segment(0, text="Kisrael")
        destination = self.folder / "sessione-salva.json"

        with patch.object(
            QFileDialog, "getSaveFileName",
            return_value=(str(destination), "JSON (*.json)"),
        ), patch.object(self.window.project, "save", side_effect=OSError("disco pieno")):
            self.window._save_session()

        self.assertTrue(destination.is_file())
        self.assertFalse(self.window._dirty)
        self.assertEqual(self.window.project.learning_sessions, {})
        self.assertIn("memoria progetto non aggiornata", self.window.status_label.text())

    def test_batch_processing_elabora_coda_e_salva_nel_progetto(self):
        self._audio("sessione_01.wav", "sessione_02.wav")
        project = open_or_create(self.folder, "Progetto Batch")
        project.save()
        self.window._open_project(self.folder)
        self.window.audio_profile_combo.setCurrentIndex(1)

        # Verifica stato coda
        self.assertEqual(len(project.pending_audio()), 2)
        self.assertIn("(2)", self.window.batch_process_btn.text())

        # Simula completamento di un file
        s1 = Session(name="sessione_01", segments=[Segment(0, 2, "Primo audio", "SPEAKER_00")])
        self.window._on_batch_file_done(self.folder / "sessione_01.wav", s1)

        self.assertEqual(len(self.window.project.sessions), 1)
        self.assertEqual(len(self.window.project.pending_audio()), 1)
        self.assertIn("(1)", self.window.batch_process_btn.text())


if __name__ == "__main__":
    unittest.main()
