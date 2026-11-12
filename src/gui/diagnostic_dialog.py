"""Long-recording diagnostics and bounded comparison inside the desktop UI."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
                               QListWidget, QPushButton, QTextEdit, QVBoxLayout)

from .audio_player import AudioPlayer
from .stage_view import StageView


class DiagnosticWorker(QThread):
    stage = Signal(object)
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, source, *, report=None, model="large-v3", device="cpu", parent=None):
        super().__init__(parent)
        self.source, self.report, self.model, self.device = source, report, model, device

    def run(self):
        try:
            if self.report is None:
                from ..processing.long_diagnostics import diagnose_recording
                result = diagnose_recording(self.source, on_stage=self.stage.emit, progress=self.progress.emit)
            else:
                from ..processing.diagnostic_experiments import run_diagnostic_experiments
                result = run_diagnostic_experiments(self.report, count=2, model=self.model, device=self.device,
                                                    on_stage=self.stage.emit, progress=self.progress.emit)
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DiagnosticDialog(QDialog):
    def __init__(self, source, *, model="large-v3", device="cpu", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostica registrazione importata")
        self.resize(900, 680)
        self.source, self.model, self.device = Path(source), model, device
        self._worker = None
        self.report_path = None
        self.report = None
        self._comparisons = []
        layout = QVBoxLayout(self)
        self.status = QLabel(f"{self.source.name}: analisi completa senza modificare l'originale.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.stage_view = StageView()
        layout.addWidget(self.stage_view)
        self.regions = QListWidget()
        self.regions.currentRowChanged.connect(self._select_region)
        layout.addWidget(self.regions, 1)
        self.variants = QComboBox()
        self.variants.currentIndexChanged.connect(self._load_variant)
        layout.addWidget(self.variants)
        self.player = AudioPlayer()
        layout.addWidget(self.player)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        layout.addWidget(self.transcript, 1)
        actions = QHBoxLayout()
        self.compare = QPushButton("Confronta 2 campioni")
        self.compare.clicked.connect(self._compare)
        self.compare.setEnabled(False)
        actions.addWidget(self.compare)
        self.export = QPushButton("Esporta rapporto e riferimento…")
        self.export.clicked.connect(self._export)
        self.export.setEnabled(False)
        actions.addWidget(self.export)
        self.folder = QPushButton("Apri risultati")
        self.folder.clicked.connect(self._open_results)
        self.folder.setEnabled(False)
        actions.addWidget(self.folder)
        layout.addLayout(actions)
        note = QLabel("ASR e sovrapposizioni sono ipotesi da verificare. Correggi reference-review.json ascoltando "
                      "i campioni; conserva le voci simultanee. WER/DER richiedono un riferimento verificato.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._start()

    def _start(self, report=None):
        self.compare.setEnabled(False)
        self.progress.setValue(0)
        self.stage_view.begin_run()
        self._worker = DiagnosticWorker(self.source, report=report, model=self.model, device=self.device, parent=self)
        self._worker.done.connect(self._done)
        self._worker.stage.connect(self.stage_view.on_stage)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, message, fraction):
        self.status.setText(message)
        self.progress.setValue(int(fraction * 100))

    def _finished(self):
        self._worker = None
        self.compare.setEnabled(self.report_path is not None)

    def _done(self, result):
        result = Path(result)
        value = json.loads(result.read_text(encoding="utf-8"))
        if result.name == "comparison.json":
            self._comparisons = value["results"]
            self._select_region(self.regions.currentRow())
            errors = sum("error" in item for item in self._comparisons)
            self.status.setText(f"Confronto terminato: {errors} varianti fallite. "
                                "Ascolta le varianti; più parole non dimostrano maggiore accuratezza.")
            return
        self.report_path, self.report = result, value
        self.regions.clear()
        for region in value["regions"]:
            self.regions.addItem(f"{region['start_s']:.2f}–{region['end_s']:.2f}s · {region['selection_reason']}")
        self.regions.setCurrentRow(0)
        self.export.setEnabled(True)
        self.folder.setEnabled(True)
        complete = not value['scan']['decode_errors'] and abs(value['scan']['decoded_duration_s'] - value['metadata']['duration_s']) < .1
        coverage = "completi" if complete else "(analisi parziale: verifica gli errori)"
        self.status.setText(f"Analizzati {value['scan']['decoded_duration_s']:.3f}s {coverage}. "
                            f"{value['scan']['channels']} canali conservati, {len(value['regions'])} campioni. "
                            f"Errori di decodifica: {len(value['scan']['decode_errors'])}.")

    def _failed(self, error):
        self.status.setText(f"Diagnostica non completata: {error}. Originale conservato.")

    def _select_region(self, index):
        self.variants.clear()
        if self.report is None or not 0 <= index < len(self.report["regions"]):
            return
        region = self.report["regions"][index]
        self.variants.addItem("Originale decodificato (tutti i canali)",
                              {"audio": str(self.report_path.parent / region["audio_file"]),
                               "text": region.get("existing_transcript", region.get("reference", {}))})
        for comparison in self._comparisons:
            config = comparison["configuration"]
            if config["region_id"] == region["region_id"] and not comparison.get("error"):
                self.variants.addItem(f"Canale {config['channel'] + 1}: {config['variant']}",
                                     {"audio": comparison["audio"], "text": comparison.get("segments", [])})
        self._load_variant()

    def _load_variant(self):
        value = self.variants.currentData()
        if not value:
            return
        self.player.load(value["audio"])
        self.transcript.setPlainText(json.dumps(value["text"], ensure_ascii=False, indent=2))

    def _compare(self):
        if self._worker is None and self.report_path is not None:
            self.status.setText(f"Confronto su 2 campioni, modello fisso {self.model}. Nessun cambio automatico del profilo.")
            self._start(self.report_path)

    def _export(self):
        if self.report_path is None:
            return
        destination = QFileDialog.getExistingDirectory(self, "Cartella per rapporto e riferimento da correggere")
        if not destination:
            return
        try:
            root = self.report_path.parent
            target_root = Path(tempfile.mkdtemp(prefix="tessitura-diagnostics-", dir=destination))
            names = ["report.json", "timeline.json", "reference-review.json"]
            names.extend(region["audio_file"] for region in self.report["regions"])
            for name in names:
                shutil.copy2(root / name, target_root / name)
            self.status.setText(f"Rapporto, riferimento e campioni originali esportati in {target_root}.")
        except OSError as exc:
            self._failed(str(exc))

    def _open_results(self):
        if self.report_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path.parent)))

    def reject(self):
        if self._worker is not None:
            self.status.setText("Attendi il termine dell'analisi prima di chiudere.")
            return
        self.player.clear()
        super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
            self.status.setText("Attendi il termine dell'analisi prima di chiudere.")
        else:
            self.player.clear()
            super().closeEvent(event)
