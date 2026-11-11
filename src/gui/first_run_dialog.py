"""Finestra di configurazione iniziale dei percorsi dati, modelli AI e verifica spazio disco."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.processing.first_run import (
    FirstRunStatus,
    check_first_run_status,
    configure_user_directories,
)
from src.hardware import Hardware


class FirstRunDialog(QDialog):
    """Guida di primo avvio per verificare hardware, spazio su disco e cartelle modelli/registrazioni."""

    def __init__(self, hw: Hardware, parent=None) -> None:
        super().__init__(parent)
        self.hw = hw
        self.status: FirstRunStatus = check_first_run_status(hw)
        self.open_model_manager_requested: bool = False

        self.setWindowTitle("Configurazione iniziale di Tessitura")
        self.setMinimumWidth(640)
        self._build_ui()
        self._refresh_status_labels()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel(
            "Benvenuto in <b>Tessitura</b>. Tutte le registrazioni, le configurazioni e i modelli AI "
            "vengono salvati fuori dalla cartella di installazione del programma, così aggiornamenti "
            "e disinstallazioni non eliminano mai il tuo lavoro."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Hardware & Disk Box
        hw_box = QGroupBox("Hardware e stato modelli locali")
        hw_layout = QVBoxLayout(hw_box)
        self.hw_summary_label = QLabel()
        self.hw_summary_label.setWordWrap(True)
        self.disk_summary_label = QLabel()
        self.disk_summary_label.setWordWrap(True)
        self.models_state_label = QLabel()
        self.models_state_label.setWordWrap(True)
        hw_layout.addWidget(self.hw_summary_label)
        hw_layout.addWidget(self.disk_summary_label)
        hw_layout.addWidget(self.models_state_label)
        layout.addWidget(hw_box)

        # Directories Box
        dir_box = QGroupBox("Cartelle dati e modelli AI")
        dir_layout = QVBoxLayout(dir_box)

        rec_lbl = QLabel("Cartella registrazioni e sessioni:")
        dir_layout.addWidget(rec_lbl)
        rec_row = QHBoxLayout()
        self.recordings_edit = QLineEdit(str(self.status.recordings_dir))
        rec_browse = QPushButton("Sfoglia…")
        rec_browse.clicked.connect(self._browse_recordings)
        rec_row.addWidget(self.recordings_edit, 1)
        rec_row.addWidget(rec_browse)
        dir_layout.addLayout(rec_row)

        mod_lbl = QLabel("Cartella cache modelli AI (Whisper / pyannote):")
        dir_layout.addWidget(mod_lbl)
        mod_row = QHBoxLayout()
        self.models_edit = QLineEdit(str(self.status.models_dir))
        mod_browse = QPushButton("Sfoglia…")
        mod_browse.clicked.connect(self._browse_models)
        mod_row.addWidget(self.models_edit, 1)
        mod_row.addWidget(mod_browse)
        dir_layout.addLayout(mod_row)

        hf_lbl = QLabel(
            "Token Hugging Face (opzionale, richiesto solo per scaricare per la prima volta "
            "il modello di diarizzazione pyannote/speaker-diarization-community-1):"
        )
        hf_lbl.setWordWrap(True)
        dir_layout.addWidget(hf_lbl)
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        self.hf_token_edit.setPlaceholderText(
            "Già configurato" if self.status.hf_token_present else "hf_... (lascia vuoto se non necessario)"
        )
        dir_layout.addWidget(self.hf_token_edit)
        layout.addWidget(dir_box)

        # Action buttons
        btn_row = QHBoxLayout()
        self.defer_btn = QPushButton("Usa più tardi (solo registrazione / offline)")
        self.defer_btn.clicked.connect(self._on_defer)
        self.models_btn = QPushButton("Salva e gestisci download modelli…")
        self.models_btn.clicked.connect(self._on_save_and_manage_models)
        self.save_btn = QPushButton("Salva e continua")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._on_save)

        btn_row.addWidget(self.defer_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.models_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

    def _refresh_status_labels(self) -> None:
        gpu_txt = (
            f"GPU CUDA rilevata: {self.status.gpu_name} ({self.status.vram_gb:.1f} GB VRAM)"
            if self.status.device == "cuda" and self.status.gpu_name
            else f"Dispositivo di calcolo: {self.status.device.upper()} (nessuna GPU CUDA attiva)"
        )
        self.hw_summary_label.setText(
            f"{gpu_txt} — Modello ASR consigliato: <b>{self.status.recommended_asr_model}</b>"
        )
        disk_status = (
            "Spazio sufficiente"
            if self.status.has_sufficient_disk
            else "Attenzione: spazio libero ridotto per il download completo dei modelli"
        )
        self.disk_summary_label.setText(
            f"Spazio libero su disco: <b>{self.status.free_disk_gb:.1f} GB</b> "
            f"(download stimato modelli mancanti: ~{self.status.estimated_download_gb:.1f} GB — {disk_status})."
        )
        asr_state = "Presente (pronto offline)" if self.status.asr_model_cached else "Da scaricare"
        diar_state = "Presente (pronto offline)" if self.status.diarization_model_cached else "Da scaricare"
        self.models_state_label.setText(
            f"Stato modelli: ASR ({self.status.recommended_asr_model}): <b>{asr_state}</b> | "
            f"Diarizzazione (pyannote community-1): <b>{diar_state}</b>"
        )

    def _browse_recordings(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella per registrazioni e sessioni", self.recordings_edit.text()
        )
        if chosen:
            self.recordings_edit.setText(chosen)

    def _browse_models(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella per i modelli AI", self.models_edit.text()
        )
        if chosen:
            self.models_edit.setText(chosen)

    def _apply_configuration(self) -> bool:
        rec_dir = self.recordings_edit.text().strip()
        mod_dir = self.models_edit.text().strip()
        token = self.hf_token_edit.text().strip()
        try:
            configure_user_directories(
                models_dir=Path(mod_dir) if mod_dir else None,
                recordings_dir=Path(rec_dir) if rec_dir else None,
                hf_token=token or None,
                mark_completed=True,
            )
            return True
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Errore salvataggio cartelle",
                f"Impossibile configurare le cartelle selezionate:\n{exc}",
            )
            return False

    def _on_defer(self) -> None:
        if self._apply_configuration():
            self.open_model_manager_requested = False
            self.accept()

    def _on_save(self) -> None:
        if self._apply_configuration():
            self.open_model_manager_requested = False
            self.accept()

    def _on_save_and_manage_models(self) -> None:
        if self._apply_configuration():
            self.open_model_manager_requested = True
            self.accept()
