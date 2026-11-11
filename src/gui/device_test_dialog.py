"""Short, independent capture check using the production recorder."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from .. import paths
from ..audio.capture import Recorder
from .workers import StopRecordWorker


class DeviceTestDialog(QDialog):
    def __init__(self, microphone, loopback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prova ingresso e uscita — 5 secondi")
        self.recorder = Recorder(paths.cache_dir() / "device-tests")
        self._worker = None
        self._finished = False
        layout = QVBoxLayout(self)
        self.note = QLabel("Parla e riproduci audio nell'applicazione sull'uscita selezionata.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.meters = {}
        for kind, label in (("mic", "Ingresso microfono"), ("system", "Uscita loopback")):
            layout.addWidget(QLabel(label))
            meter = QProgressBar()
            meter.setRange(0, 100)
            layout.addWidget(meter)
            self.meters[kind] = meter
        self.close_button = QPushButton("Ferma prova")
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.close_button)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._update_levels)
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(self._stop)
        try:
            self.recorder.start(record_mic=microphone is not None, record_system=loopback is not None,
                                mic_device=microphone, system_device=loopback)
        except Exception as exc:
            self.note.setText(f"Prova non avviata: {exc}")
            self._finished = True
            self.close_button.setText("Chiudi")
        else:
            self.timer.start()
            self.deadline.start(5000)

    def _update_levels(self):
        for kind, level in self.recorder.levels().items():
            self.meters[kind].setValue(min(100, int(level * 300)))
        if self.recorder.capture_errors():
            self._stop()

    def _stop(self):
        if self._worker is not None or not self.recorder.is_recording:
            return
        self.timer.stop()
        self.deadline.stop()
        self.close_button.setEnabled(False)
        self._worker = StopRecordWorker(self.recorder, self)
        self._worker.done.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._worker_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _worker_finished(self):
        self._worker = None
        self._finished = True
        self.close_button.setText("Chiudi")
        self.close_button.setEnabled(True)

    def _done(self, result):
        self.note.setText("; ".join(result.errors) if result.errors else
                          "Prova completata. Controlla i livelli di entrambe le sorgenti. "
                          "Le tracce di prova restano nella cache, separate dalle registrazioni importate.")

    def _failed(self, message):
        self.note.setText(f"Errore nella prova: {message}. Le tracce originali sono conservate.")

    def reject(self):
        if not self._finished:
            self._stop()
            return
        super().reject()

    def closeEvent(self, event):
        if not self._finished:
            self._stop()
            event.ignore()
        else:
            super().closeEvent(event)
