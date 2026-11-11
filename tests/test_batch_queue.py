"""Coda di elaborazione di un progetto: ordine, progresso, errori, interruzione.

Il worker viene eseguito davvero (thread Qt), con la pipeline sostituita da una
finta: quello che deve reggere e' la coda, non la trascrizione. Una cartella di
progetto contiene decine di registrazioni e un singolo file illeggibile non
deve fermare le altre venti.

Esecuzione: python -m tests.test_batch_queue
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.gui import workers
from src.gui.workers import BatchProcessWorker
from src.models.session import Segment, Session
from src.processing.audio_profile import AudioProfile
from src.processing.pipeline import PipelineOptions


def _options():
    return PipelineOptions(audio_profile=AudioProfile.STEREO_AMBIENT, diarize=False)


class BatchQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)
        self.messages: list[tuple[str, float]] = []
        self.completed: list[Path] = []
        self.failures: list[tuple[Path, str]] = []

    def _audio(self, *names) -> list[Path]:
        created = []
        for name in names:
            path = self.folder / name
            path.write_bytes(b"finto")
            created.append(path)
        return created

    def _run(self, worker: BatchProcessWorker, timeout_ms: int = 15000):
        """Esegue il worker e torna quando ha finito, senza bloccare il test."""
        loop = QEventLoop()
        worker.progress.connect(lambda message, frac: self.messages.append((message, frac)))
        worker.file_done.connect(lambda audio, _session: self.completed.append(Path(audio)))
        worker.file_failed.connect(lambda audio, error: self.failures.append((Path(audio), error)))
        worker.finished.connect(loop.quit)
        worker.start()
        QCoreApplication.processEvents()
        loop.exec()
        worker.wait(timeout_ms)

    def test_i_file_vengono_elaborati_in_ordine_uno_alla_volta(self):
        files = self._audio("a.wav", "b.wav", "c.wav")
        seen: list[str] = []

        def fake_pipeline(audio, _system=None, *, options=None, session_name=None,
                          progress=None, **_kwargs):
            seen.append(Path(audio).name)
            if progress:
                progress("Trascrizione", 0.5)
                progress("Completato", 1.0)
            return Session(name=session_name or "", segments=[Segment(0, 1, "ciao")])

        with patch.object(workers, "process_recording", fake_pipeline):
            self._run(BatchProcessWorker(files, _options(), project_name="Campagna"))

        self.assertEqual(seen, ["a.wav", "b.wav", "c.wav"])
        self.assertEqual([item.name for item in self.completed], ["a.wav", "b.wav", "c.wav"])

    def test_il_progresso_e_complessivo_e_non_torna_indietro(self):
        files = self._audio("a.wav", "b.wav")

        def fake_pipeline(audio, _system=None, *, options=None, session_name=None,
                          progress=None, **_kwargs):
            for step in (0.25, 0.5, 1.0):
                progress(f"fase {step}", step)
            return Session(segments=[Segment(0, 1, "ciao")])

        with patch.object(workers, "process_recording", fake_pipeline):
            self._run(BatchProcessWorker(files, _options(), project_name="Campagna"))

        fractions = [frac for _message, frac in self.messages]
        self.assertEqual(fractions, sorted(fractions))      # mai all'indietro
        self.assertLessEqual(max(fractions), 1.0)
        self.assertAlmostEqual(max(fractions), 1.0, places=6)
        # A meta' coda il progresso complessivo e' a meta', non al 100% del file
        self.assertTrue(any(abs(frac - 0.5) < 1e-6 for frac in fractions))

    def test_il_messaggio_dice_progetto_file_corrente_e_fase(self):
        files = self._audio("seduta.wav", "altra.wav")

        def fake_pipeline(audio, _system=None, *, options=None, session_name=None,
                          progress=None, **_kwargs):
            progress("Trascrizione 00:10 / 01:00", 0.4)
            return Session(segments=[Segment(0, 1, "ciao")])

        with patch.object(workers, "process_recording", fake_pipeline):
            self._run(BatchProcessWorker(files, _options(), project_name="Campagna"))

        first = self.messages[0][0]
        self.assertIn("Campagna", first)
        self.assertIn("file 1/2", first)
        self.assertIn("seduta.wav", first)
        self.assertIn("Trascrizione", first)

    def test_un_file_illeggibile_non_ferma_la_coda(self):
        files = self._audio("buono1.wav", "rotto.wav", "buono2.wav")

        def fake_pipeline(audio, _system=None, *, options=None, session_name=None,
                          progress=None, **_kwargs):
            if Path(audio).name == "rotto.wav":
                raise RuntimeError("formato non riconosciuto")
            return Session(segments=[Segment(0, 1, "ciao")])

        with patch.object(workers, "process_recording", fake_pipeline):
            self._run(BatchProcessWorker(files, _options(), project_name="P"))

        self.assertEqual([item.name for item in self.completed], ["buono1.wav", "buono2.wav"])
        self.assertEqual(len(self.failures), 1)
        self.assertIn("formato non riconosciuto", self.failures[0][1])

    def test_interrompere_ferma_la_coda_dopo_il_file_in_corso(self):
        files = self._audio("a.wav", "b.wav", "c.wav")
        worker = BatchProcessWorker(files, _options(), project_name="P")

        def fake_pipeline(audio, _system=None, *, options=None, session_name=None,
                          progress=None, **_kwargs):
            worker.stop()   # richiesta di arresto durante il primo file
            return Session(segments=[Segment(0, 1, "ciao")])

        with patch.object(workers, "process_recording", fake_pipeline):
            self._run(worker)

        # Il file in corso viene completato, i successivi no.
        self.assertEqual([item.name for item in self.completed], ["a.wav"])
        self.assertTrue(worker.stopping)

    def test_una_coda_vuota_finisce_subito_senza_errori(self):
        with patch.object(workers, "process_recording", lambda *a, **k: None):
            self._run(BatchProcessWorker([], _options(), project_name="P"))

        self.assertEqual(self.completed, [])
        self.assertEqual(self.failures, [])


if __name__ == "__main__":
    unittest.main()
