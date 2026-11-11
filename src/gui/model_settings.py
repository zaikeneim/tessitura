"""Schermata separata per scegliere e verificare tutti i motori locali."""

from __future__ import annotations

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..hardware import Hardware
from ..processing.model_catalog import (
    ModelEntry,
    ModelStatus,
    catalog_status,
    diarization_catalog,
    transcription_catalog,
)


class ModelCatalogWorker(QThread):
    done = Signal(object)
    progress = Signal(str)

    def __init__(self, entries, remote: bool, parent=None):
        super().__init__(parent)
        self._entries = tuple(entries)
        self._remote = remote

    def run(self) -> None:
        try:
            self.done.emit(catalog_status(
                self._entries,
                check_remote=self._remote,
                progress=lambda message: self.progress.emit(message),
            ))
        except Exception as exc:  # noqa: BLE001 - la schermata resta utilizzabile
            self.done.emit(exc)


class ModelDownloadWorker(QThread):
    done = Signal(object)
    progress = Signal(str)

    def __init__(self, entry: ModelEntry, parent=None):
        super().__init__(parent)
        self._entry = entry

    def run(self) -> None:
        try:
            def on_download(name: str, pct: int) -> None:
                self.progress.emit(f"Scaricamento {name}: {pct}%")

            from ..processing.model_store import ensure
            token = None
            if self._entry.category == "diarizzazione":
                from ..processing.diarize import read_hf_token
                token = read_hf_token()
            # The same transactional store is used by all model families:
            # validate the whole snapshot before the GUI reports success.
            ensure(self._entry.repo, self._entry.category,
                   on_download=on_download, token=token)
            self.done.emit(None)
        except Exception as exc:  # noqa: BLE001 - la cache esistente resta intatta
            self.done.emit(exc)


class ModelSettingsDialog(QDialog):
    def __init__(
        self,
        hardware: Hardware,
        current_transcription: str,
        current_analysis: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._hardware = hardware
        self._current_transcription = current_transcription
        self._current_analysis = current_analysis
        self._statuses: dict[str, ModelStatus] = {}
        self._worker: ModelCatalogWorker | None = None
        self._download_worker: ModelDownloadWorker | None = None
        self._entries = {
            "trascrizione": transcription_catalog(),
            "diarizzazione": diarization_catalog(),
        }
        self._build_ui()
        self._refresh(remote=False)

    def _build_ui(self) -> None:
        self.setWindowTitle("Impostazioni — modelli e motori")
        self.resize(980, 680)
        root = QVBoxLayout(self)
        intro = QLabel(
            "Qui scegli i motori locali e controlli cache e revisioni. "
            "La GPU rilevata viene utilizzata per trascrizione e diarizzazione."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)
        hardware_label = QLabel(self._hardware.describe())
        hardware_label.setObjectName("muted")
        root.addWidget(hardware_label)

        self.tabs = QTabWidget()
        transcription_page, self.transcription_combo, self.transcription_table = self._make_tab(
            "Trascrizione Whisper", "Modello Whisper attivo", transcription_catalog(),
            self._current_transcription,
        )
        diarization_page, self.diarization_combo, self.diarization_table = self._make_tab(
            "Diarizzazione pyannote", "Modello diarizzazione attivo", diarization_catalog(),
            diarization_catalog()[0].key,
        )
        self.diarization_combo.setEnabled(False)
        self.tabs.addTab(transcription_page, "Trascrizione")
        self.tabs.addTab(diarization_page, "Diarizzazione")
        self.active_info = QLabel()
        self.active_info.setObjectName("modelSummary")
        self.active_info.setWordWrap(True)
        root.addWidget(self.active_info)
        root.addWidget(self.tabs, 1)
        for combo in (self.transcription_combo, self.diarization_combo):
            combo.currentIndexChanged.connect(self._update_active_info)
        self.tabs.currentChanged.connect(self._update_active_info)

        bottom = QVBoxLayout()
        self.status_label = QLabel("Lettura cache locale…")
        self.status_label.setObjectName("muted")
        bottom.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Verifica aggiornamenti online")
        self.refresh_button.clicked.connect(lambda: self._refresh(remote=True))
        actions.addWidget(self.refresh_button)
        self.download_button = QPushButton("Scarica selezionato")
        self.download_button.clicked.connect(self._download_selected)
        actions.addWidget(self.download_button)
        actions.addStretch()
        bottom.addLayout(actions)
        root.addLayout(bottom)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_active_info()

    def _make_tab(self, title: str, label: str, entries, selected: str):
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox(title)
        form = QFormLayout(group)
        combo = QComboBox()
        for entry in entries:
            combo.addItem(entry.label, entry.key)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))
        form.addRow(label + ":", combo)
        layout.addWidget(group)
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(
            ["Modello", "Repository", "Stato", "Locale", "Remota", "Dimensione / contesto"]
        )
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)
        return page, combo, table

    @staticmethod
    def _size_text(entry: ModelEntry) -> str:
        size = f"~{entry.approx_gb:g} GB" if entry.approx_gb else "n/d"
        context = f" · {entry.context_tokens:,} token" if entry.context_tokens else ""
        return size + context

    def _refresh(self, *, remote: bool) -> None:
        if self._worker is not None or self._download_worker is not None:
            return
        entries = (*self._entries["trascrizione"], *self._entries["diarizzazione"])
        self.refresh_button.setEnabled(False)
        self.status_label.setText(
            "Verifica revisioni remote…" if remote else "Lettura cache locale…"
        )
        self._worker = ModelCatalogWorker(entries, remote=remote, parent=self)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.done.connect(self._on_catalog_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker.start()

    def _on_catalog_done(self, result) -> None:
        self.refresh_button.setEnabled(True)
        if isinstance(result, Exception):
            self.status_label.setText(f"Catalogo locale disponibile; verifica fallita: {result}")
            return
        self._statuses = {item.entry.repo: item for item in result}
        self._fill_table(self.transcription_table, "trascrizione")
        self._fill_table(self.diarization_table, "diarizzazione")
        self._update_active_info()
        self.status_label.setText("Catalogo aggiornato.")

    def _selected_entry(self) -> ModelEntry | None:
        category = ("trascrizione", "diarizzazione")[self.tabs.currentIndex()]
        key = (
            self.transcription_combo.currentData(),
            self.diarization_combo.currentData(),
        )[self.tabs.currentIndex()]
        return next((entry for entry in self._entries[category] if entry.key == key), None)

    def _update_active_info(self, *_args) -> None:
        """Aggiorna la scheda sintetica del motore selezionato nella scheda attiva."""
        entry = self._selected_entry()
        if entry is None:
            self.active_info.setText("Nessun motore selezionato.")
            return
        status = self._statuses.get(entry.repo)
        state = status.state if status is not None else "stato non ancora verificato"
        revision = ""
        if status is not None and status.local_revision:
            revision = f" · revisione locale {status.local_revision[:12]}"
        self.active_info.setText(
            f"Motore selezionato · {entry.label}  ·  {state}{revision}\n"
            f"Repository: {entry.repo}  ·  {self._size_text(entry)}"
        )

    def _download_selected(self) -> None:
        if self._download_worker is not None:
            return
        entry = self._selected_entry()
        if entry is None:
            return
        self.refresh_button.setEnabled(False)
        self.download_button.setEnabled(False)
        self.status_label.setText(f"Preparazione download di {entry.label}…")
        self._download_worker = ModelDownloadWorker(entry, parent=self)
        self._download_worker.progress.connect(self.status_label.setText)
        self._download_worker.done.connect(self._on_download_done)
        self._download_worker.finished.connect(self._download_worker.deleteLater)
        self._download_worker.finished.connect(lambda: setattr(self, "_download_worker", None))
        self._download_worker.start()

    def _on_download_done(self, result) -> None:
        self.refresh_button.setEnabled(True)
        self.download_button.setEnabled(True)
        if isinstance(result, Exception):
            self.status_label.setText(f"Download non riuscito: {result}")
            QMessageBox.warning(self, "Modelli", str(result))
            return
        self.status_label.setText("Modello scaricato e pronto in cache.")
        # Il riferimento NON si azzera qui: il thread ha emesso il risultato ma
        # puo' essere ancora in esecuzione, e perderlo adesso farebbe credere
        # alle uscite della finestra che non ci sia piu' niente da attendere.
        # Ci pensa gia' finished -> setattr(None).
        QTimer.singleShot(0, lambda: self._refresh(remote=False))

    def _fill_table(self, table: QTableWidget, category: str) -> None:
        rows = [item for item in self._statuses.values() if item.entry.category == category]
        table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = (
                item.entry.label,
                item.entry.repo,
                item.state,
                item.local_revision[:12] if item.local_revision else "—",
                item.remote_revision[:12] if item.remote_revision else "—",
                self._size_text(item.entry),
            )
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))

    def _apply(self) -> None:
        self.accept()

    def selected_transcription_model(self) -> str:
        return str(self.transcription_combo.currentData())

    def _busy(self) -> bool:
        """Un worker sui modelli e' ancora vivo."""
        return any(worker is not None and worker.isRunning()
                   for worker in (self._worker, self._download_worker))

    def done(self, result: int) -> None:
        """Unico passaggio per accept(), reject(), Escape e pulsanti.

        La protezione stava solo in closeEvent, che copre la X della finestra:
        Annulla ed Escape chiudevano il dialogo con un download in corso,
        lasciando il thread senza finestra a cui riferire.
        """
        if self._busy():
            self.status_label.setText("Attendi il completamento dell'operazione sui modelli…")
            return
        super().done(result)

    def closeEvent(self, event) -> None:
        if self._busy():
            self.status_label.setText("Attendi il completamento dell'operazione sui modelli…")
            event.ignore()
            return
        event.accept()
