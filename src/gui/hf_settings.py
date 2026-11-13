"""Hugging Face section of the settings and the diarization model assistant.

The widgets never read the stored token: they only describe where it comes
from. A typed token goes straight to ``hf_access`` (vault or verification)
and the field is cleared after saving.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..processing import hf_access
from ..processing.hf_access import AccessIssue, ModelAccessError

LINK_COLOR = "#5ee0ca"  # accento del tema: il blu predefinito non si legge sul fondo scuro


def _link(url: str, text: str | None = None) -> str:
    return f'<a href="{url}" style="color:{LINK_COLOR}">{text or url}</a>'


STATE_TEXT = {
    "not_configured": ("Non configurato", "#8395aa"),
    "unverified": ("Non verificato", "#d6b35a"),
    "checking": ("Verifica in corso…", "#5ea9e0"),
    "valid": ("Valido", "#5ee0a0"),
    "invalid": ("Non valido", "#ff7b7b"),
}


class HFVerifyWorker(QThread):
    done = Signal(object)

    def __init__(self, token: str | None, store=None, parent=None):
        super().__init__(parent)
        self._token = token
        self._store = store

    def run(self) -> None:
        try:
            result = hf_access.verify_configuration(self._token, store=self._store)
        except Exception as exc:  # noqa: BLE001 - never leave the screen waiting
            result = {"state": "unverified", "kind": None, "url": "",
                      "message": hf_access.redact(f"Verifica non riuscita: {type(exc).__name__}", self._token)}
        self._token = None
        self.done.emit(result)


class HuggingFaceSettingsWidget(QWidget):
    """Token field, save/verify/remove and the state indicator."""

    configurationChanged = Signal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._worker: HFVerifyWorker | None = None
        self.state = "not_configured"
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        group = QGroupBox("Hugging Face")
        layout = QVBoxLayout(group)
        intro = QLabel(
            "Il token serve solo per scaricare la prima volta il modello di diarizzazione "
            f"<b>{hf_access.DIARIZATION_REPO}</b>. Viene salvato nel Gestore credenziali del "
            "sistema, mai in un file. La variabile d'ambiente HF_TOKEN, se impostata, ha la precedenza."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QLabel("Hugging Face Access Token"))
        row = QHBoxLayout()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("hf_…")
        self.token_edit.setAccessibleName("Hugging Face Access Token")
        self.token_edit.textChanged.connect(self._update_buttons)
        self.show_button = QPushButton("Mostra")
        self.show_button.setCheckable(True)
        self.show_button.toggled.connect(self._toggle_visibility)
        row.addWidget(self.token_edit, 1)
        row.addWidget(self.show_button)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Salva token")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.save)
        self.verify_button = QPushButton("Verifica connessione")
        self.verify_button.clicked.connect(self.verify)
        self.remove_button = QPushButton("Rimuovi token")
        self.remove_button.clicked.connect(self.remove)
        for button in (self.save_button, self.verify_button, self.remove_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("Stato:"))
        self.state_label = QLabel()
        self.state_label.setObjectName("hfState")
        state_row.addWidget(self.state_label)
        state_row.addStretch()
        layout.addLayout(state_row)
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setObjectName("muted")
        layout.addWidget(self.detail_label)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setOpenExternalLinks(True)
        layout.addWidget(self.message_label)

        legacy_row = QHBoxLayout()
        self.legacy_label = QLabel()
        self.legacy_label.setWordWrap(True)
        self.legacy_button = QPushButton("Sposta nel Gestore credenziali")
        self.legacy_button.clicked.connect(self.import_legacy)
        legacy_row.addWidget(self.legacy_label, 1)
        legacy_row.addWidget(self.legacy_button)
        layout.addLayout(legacy_row)

        self.tokens_link = QLabel(
            _link(hf_access.TOKENS_URL, "Crea o gestisci i token su huggingface.co/settings/tokens")
        )
        self.tokens_link.setOpenExternalLinks(True)
        layout.addWidget(self.tokens_link)
        root.addWidget(group)
        root.addStretch()

    # --- state -------------------------------------------------------------

    def _set_state(self, state: str, message: str = "", url: str = "") -> None:
        self.state = state
        text, color = STATE_TEXT[state]
        self.state_label.setText(f'<span style="color:{color}">●</span> <b>{text}</b>')
        link = f" {_link(url)}" if url else ""
        self.message_label.setText(f"{message}{link}" if message or link else "")

    def refresh(self) -> None:
        info = hf_access.token_info(self._store)
        self.detail_label.setText(f"Origine: {info.describe()}")
        if self.state != "checking":
            self._set_state("unverified" if info.present else "not_configured")
        self.legacy_label.setVisible(bool(info.legacy_file))
        self.legacy_button.setVisible(bool(info.legacy_file) and info.store_available)
        if info.legacy_file:
            self.legacy_label.setText(
                f"Trovato un vecchio token in chiaro: {info.legacy_file}. "
                "Spostalo nel Gestore credenziali: il file verrà eliminato."
            )
        self._info = info
        self._update_buttons()

    def _update_buttons(self, *_args) -> None:
        busy = self._worker is not None
        info = getattr(self, "_info", None)
        typed = bool(self.token_edit.text().strip())
        self.save_button.setEnabled(not busy and typed and bool(info and info.store_available))
        self.verify_button.setEnabled(not busy and (typed or bool(info and info.present)))
        self.remove_button.setEnabled(not busy and bool(info and info.store_has_token))

    def _toggle_visibility(self, visible: bool) -> None:
        self.token_edit.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.show_button.setText("Nascondi" if visible else "Mostra")

    # --- actions -----------------------------------------------------------

    def save(self) -> None:
        typed = self.token_edit.text()
        try:
            info = hf_access.save_token(typed, self._store)
        except (ValueError, hf_access.CredentialStoreError) as exc:
            self._set_state(self.state, hf_access.redact(str(exc), typed))
            return
        finally:
            typed = None
        self.token_edit.clear()
        self.show_button.setChecked(False)
        self.refresh()
        self.configurationChanged.emit()
        note = ("" if info.source is hf_access.TokenSource.STORE
                else f" Attenzione: è attiva {info.detail}, che ha la precedenza.")
        self.message_label.setText(f"Token salvato nel {info.store_name}.{note}")
        self.verify()

    def verify(self) -> None:
        if self._worker is not None:
            return
        typed = self.token_edit.text().strip() or None
        self._set_state("checking", "Verifica del token e dell'accesso al modello…")
        self._worker = HFVerifyWorker(typed, self._store, self)
        self._worker.done.connect(self._on_verified)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._worker_finished)
        self._update_buttons()
        self._worker.start()

    def _worker_finished(self) -> None:
        self._worker = None
        self._update_buttons()

    def _on_verified(self, result: dict) -> None:
        state = result.get("state", "unverified")
        message = result.get("message", "")
        if state == "valid":
            who = f"Account {result['account']}" if result.get("account") else "Token valido"
            if result.get("typed"):
                who += " · token non ancora salvato"
            message = f"{who}. {message}"
        self.state = state
        self._set_state(state, message, result.get("url", ""))
        self._last_result = result

    def remove(self) -> None:
        answer = QMessageBox.question(
            self, "Rimuovi token",
            "Rimuovere il token Hugging Face dal Gestore credenziali?\n\n"
            "I modelli già scaricati restano utilizzabili.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            hf_access.remove_token(self._store)
        except hf_access.CredentialStoreError as exc:
            self.message_label.setText(str(exc))
            return
        self.state = "not_configured"
        self.refresh()
        self.configurationChanged.emit()
        info = self._info
        self.message_label.setText(
            "Token rimosso." + (f" {info.describe()}." if info.present else ""))

    def import_legacy(self) -> None:
        try:
            removed = hf_access.import_legacy_token(self._store)
        except (ValueError, hf_access.CredentialStoreError) as exc:
            self.message_label.setText(hf_access.redact(str(exc)))
            return
        self.refresh()
        self.configurationChanged.emit()
        self.message_label.setText(f"Token spostato nel Gestore credenziali; file eliminato: {removed}")

    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()


class HuggingFaceDialog(QDialog):
    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Impostazioni — Hugging Face")
        self.resize(700, 560)
        layout = QVBoxLayout(self)
        self.settings = HuggingFaceSettingsWidget(store, self)
        layout.addWidget(self.settings)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def done(self, result: int) -> None:
        if self.settings.busy():
            self.settings.message_label.setText("Attendi il termine della verifica…")
            return
        super().done(result)


class DiarizationSetupWorker(QThread):
    step = Signal(str, str, str)
    progress = Signal(str, int)
    done = Signal(object)

    def __init__(self, store=None, repair: bool = False, parent=None):
        super().__init__(parent)
        self._store = store
        self._repair = repair

    def run(self) -> None:
        try:
            path = hf_access.ensure_model(
                store=self._store, repair=self._repair,
                on_step=lambda name, state, detail: self.step.emit(name, state, detail),
                on_download=lambda name, pct: self.progress.emit(name, int(pct)),
            )
            self.done.emit({"ok": True, "path": path})
        except ModelAccessError as exc:
            self.done.emit({"ok": False, **exc.to_dict()})
        except Exception as exc:  # noqa: BLE001 - e.g. model in use by another process
            self.done.emit({"ok": False, "kind": None, "url": "",
                            "message": hf_access.redact(str(exc) or type(exc).__name__)})


STEP_LABELS = {
    "cache": "Cache locale",
    "token": "Credenziali Hugging Face",
    "access": "Accesso al repository",
    "terms": "Condizioni d'uso del modello",
    "download": "Download",
    "integrity": "Integrità della cache",
}
STEP_MARKS = {"pending": "In attesa", "running": "In corso", "done": "OK",
              "failed": "Da risolvere", "skipped": "Non necessario"}


class DiarizationModelDialog(QDialog):
    """Explains why the model is needed, checks access and downloads it.

    Accepted only when the verified model is in the cache, so the caller can
    resume the interrupted operation.
    """

    def __init__(self, reason: dict | None = None, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._worker: DiarizationSetupWorker | None = None
        self.setWindowTitle("Modello di diarizzazione")
        self.resize(680, 480)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Per distinguere le voci serve il modello "
            f"<b>{hf_access.DIARIZATION_REPO}</b>. Viene scaricato una volta sola "
            "(circa 30 MB) e poi usato senza connessione."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setOpenExternalLinks(True)
        layout.addWidget(self.reason_label)

        self.steps = QTreeWidget()
        self.steps.setHeaderLabels(["Passo", "Stato", "Dettaglio"])
        self.steps.setRootIsDecorated(False)
        self._items = {}
        for key in hf_access.STEPS:
            item = QTreeWidgetItem([STEP_LABELS[key], STEP_MARKS["pending"], ""])
            self.steps.addTopLevelItem(item)
            self._items[key] = item
        self.steps.resizeColumnToContents(0)
        self.steps.setColumnWidth(1, 120)
        layout.addWidget(self.steps, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("muted")
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)

        actions = QHBoxLayout()
        self.configure_button = QPushButton("Configura Hugging Face")
        self.configure_button.clicked.connect(self.configure)
        self.model_page_button = QPushButton("Apri pagina del modello")
        self.model_page_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(hf_access.model_url())))
        self.repair_button = QPushButton("Elimina cache danneggiata e riscarica")
        self.repair_button.clicked.connect(self.repair)
        self.start_button = QPushButton("Verifica e scarica")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self.start(repair=False))
        self.cancel_button = QPushButton("Annulla")
        self.cancel_button.clicked.connect(self.reject)
        for widget in (self.configure_button, self.model_page_button, self.repair_button):
            actions.addWidget(widget)
        actions.addStretch()
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.outcome: dict | None = None
        self._show_reason(reason)
        self._evaluate_locally()

    def _show_reason(self, reason: dict | None) -> None:
        kind = (reason or {}).get("kind")
        message = (reason or {}).get("message") or AccessIssue.MODEL_NOT_CACHED.message
        url = (reason or {}).get("url") or ""
        link = f" {_link(url)}" if url else ""
        self.reason_label.setText(f"{message}{link}")
        self.model_page_button.setVisible(kind in (AccessIssue.TERMS_NOT_ACCEPTED.value,
                                                   AccessIssue.ACCESS_DENIED.value))
        self.repair_button.setVisible(kind == AccessIssue.CACHE_CORRUPT.value)
        self.configure_button.setDefault(kind is None or AccessIssue(kind).needs_token_setup)

    def _evaluate_locally(self) -> None:
        """No network: cache state and token presence only."""
        for key in hf_access.STEPS:
            self._set_step(key, "pending", "")
        state = hf_access.local_cache_state()
        if state == "ready":
            self._set_step("cache", "done", "Modello completo in cache")
            return
        self._set_step("cache", "failed", "Cache incompleta: verrà completata" if state == "incomplete"
                       else "Modello non presente in cache")
        info = hf_access.token_info(self._store)
        if info.present:
            self._set_step("token", "pending", info.describe())
        else:
            self._set_step("token", "failed", AccessIssue.TOKEN_MISSING.message)
        self.start_button.setEnabled(info.present)

    def _set_step(self, key: str, state: str, detail: str) -> None:
        item = self._items[key]
        item.setText(1, STEP_MARKS.get(state, state))
        item.setText(2, detail)
        item.setToolTip(2, detail)
        item.setData(1, 32, state)  # Qt.UserRole

    def step_state(self, key: str) -> str:
        return self._items[key].data(1, 32)

    def configure(self) -> None:
        dialog = HuggingFaceDialog(self._store, self)
        dialog.exec()
        self._evaluate_locally()
        if hf_access.token_info(self._store).present:
            self.reason_label.setText("Token configurato: avvia la verifica e il download.")

    def repair(self) -> None:
        answer = QMessageBox.question(
            self, "Ripara cache",
            "Eliminare la cache del modello di diarizzazione e scaricarlo di nuovo?\n\n"
            "Le sessioni salvate non vengono toccate.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.start(repair=True)

    def start(self, *, repair: bool = False) -> None:
        if self._worker is not None:
            return
        for button in (self.start_button, self.configure_button, self.repair_button):
            button.setEnabled(False)
        self.progress.setValue(0)
        self.progress_label.setText("")
        self._worker = DiarizationSetupWorker(self._store, repair, self)
        self._worker.step.connect(self._set_step)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _on_progress(self, name: str, pct: int) -> None:
        self.progress.setValue(pct)
        self.progress_label.setText(f"Scaricamento {name}: {pct}%")

    def _worker_finished(self) -> None:
        self._worker = None
        for button in (self.start_button, self.configure_button, self.repair_button):
            button.setEnabled(True)
        if self.outcome and self.outcome.get("ok"):
            QTimer.singleShot(0, self.accept)

    def _on_done(self, result: dict) -> None:
        self.outcome = result
        if result.get("ok"):
            self.progress.setValue(100)
            self.reason_label.setText("Modello verificato e pronto. L'operazione riprende.")
            return
        self.start_button.setText("Riprova")
        self._show_reason(result)

    def done(self, result: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.progress_label.setText("Attendi il termine del download: i file parziali verranno riutilizzati.")
            return
        super().done(result)


def ensure_diarization_model(parent=None, reason: dict | None = None, store=None) -> bool:
    """True when the verified model is available, asking the user if needed."""
    if reason is None and hf_access.diarization_model_ready():
        return True
    dialog = DiarizationModelDialog(reason, store, parent)
    return dialog.exec() == QDialog.Accepted and hf_access.diarization_model_ready()
