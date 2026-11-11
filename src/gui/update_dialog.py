"""Finestra di gestione aggiornamenti GitHub Releases per Tessitura."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.config import load_settings, save_settings
from src.update_manager import (
    InstallerVerificationError,
    PackageVerificationResult,
    ReleaseInfo,
    UpdateBlockedByActiveOperationError,
    UpdateCheckResult,
    UpdateManager,
)


class GitHubUpdateCheckWorker(QThread):
    """Esegue il controllo delle release GitHub in un thread separato."""

    done = Signal(object)  # UpdateCheckResult
    failed = Signal(str)

    def __init__(
        self,
        manager: UpdateManager,
        *,
        channel: str,
        force: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._channel = channel
        self._force = force

    def run(self) -> None:
        try:
            result = self._manager.check_for_updates(
                channel=self._channel,
                force=self._force,
            )
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class GitHubInstallerDownloadWorker(QThread):
    """Scarica e verifica l'installer della release GitHub in un thread separato."""

    progress = Signal(int, int)  # downloaded_bytes, total_bytes
    done = Signal(object)        # PackageVerificationResult
    failed = Signal(str)

    def __init__(
        self,
        manager: UpdateManager,
        release: ReleaseInfo,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._release = release

    def run(self) -> None:
        try:
            installer_path, expected_sha256 = self._manager.download_release_installer(
                self._release,
                progress_callback=lambda dl, tot: self.progress.emit(dl, tot),
            )
            expected_size = (
                self._release.installer_asset.size
                if self._release.installer_asset is not None
                else 0
            )
            verification = self._manager.verify_installer_package(
                installer_path,
                target_version=self._release.version,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                release_html_url=self._release.html_url,
            )
            self.done.emit(verification)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class UpdateDialog(QDialog):
    """Dialogo 'Impostazioni -> Aggiornamenti…' per controllo, verifica e installazione assistita."""

    def __init__(
        self,
        manager: UpdateManager,
        *,
        initial_result: UpdateCheckResult | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self.current_result: UpdateCheckResult | None = initial_result
        self.verified_package: PackageVerificationResult | None = None
        self._check_worker: GitHubUpdateCheckWorker | None = None
        self._download_worker: GitHubInstallerDownloadWorker | None = None

        self.setWindowTitle("Aggiornamenti di Tessitura")
        self.setMinimumWidth(680)
        self.setMinimumHeight(520)
        self._build_ui()
        self._load_preferences()
        if self.current_result is not None:
            self._apply_check_result(self.current_result)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Stato attuale e canale
        status_box = QGroupBox("Versione e preferenze di aggiornamento")
        status_layout = QVBoxLayout(status_box)

        self.installed_label = QLabel(
            f"Versione installata: <b>Tessitura {self.manager.current_version}</b> "
            f"(Policy di verifica: <code>{self.manager.trust_policy}</code>)"
        )
        status_layout.addWidget(self.installed_label)

        self.last_check_label = QLabel("Ultimo controllo: mai eseguito")
        self.last_check_label.setObjectName("muted")
        status_layout.addWidget(self.last_check_label)

        pref_row = QHBoxLayout()
        pref_row.addWidget(QLabel("Canale di rilascio:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Stabile (consigliato)", "stable")
        self.channel_combo.addItem("Beta (anteprime e pre-release)", "beta")
        self.channel_combo.currentIndexChanged.connect(self._on_preferences_changed)
        pref_row.addWidget(self.channel_combo, 1)

        self.auto_check_box = QCheckBox("Controlla automaticamente all'avvio (ogni 24 ore)")
        self.auto_check_box.toggled.connect(self._on_preferences_changed)
        pref_row.addWidget(self.auto_check_box)
        status_layout.addLayout(pref_row)

        action_top = QHBoxLayout()
        self.check_btn = QPushButton("Verifica aggiornamenti ora")
        self.check_btn.clicked.connect(self._start_check)
        self.open_github_btn = QPushButton("Apri GitHub Releases…")
        self.open_github_btn.clicked.connect(self._open_release_page)
        action_top.addWidget(self.check_btn)
        action_top.addWidget(self.open_github_page_button())
        action_top.addStretch()
        status_layout.addLayout(action_top)
        layout.addWidget(status_box)

        # Dettagli release disponibile e changelog
        release_box = QGroupBox("Stato aggiornamenti e note di rilascio")
        release_layout = QVBoxLayout(release_box)

        self.update_headline = QLabel("Premi 'Verifica aggiornamenti ora' per interrogare GitHub Releases.")
        self.update_headline.setWordWrap(True)
        release_layout.addWidget(self.update_headline)

        self.asset_info_label = QLabel("")
        self.asset_info_label.setObjectName("muted")
        self.asset_info_label.setWordWrap(True)
        release_layout.addWidget(self.asset_info_label)

        self.changelog_view = QPlainTextEdit()
        self.changelog_view.setReadOnly(True)
        self.changelog_view.setPlaceholderText("Le note di rilascio (changelog) appariranno qui.")
        release_layout.addWidget(self.changelog_view, 1)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.hide()
        release_layout.addWidget(self.download_progress)
        layout.addWidget(release_box, 1)

        # Pulsanti di azione inferiori
        bottom_row = QHBoxLayout()
        self.remind_btn = QPushButton("Ricordamelo più tardi")
        self.remind_btn.clicked.connect(self._on_remind_later)
        self.remind_btn.setEnabled(False)

        self.download_btn = QPushButton("Scarica e verifica pacchetto")
        self.download_btn.clicked.connect(self._start_download)
        self.download_btn.setEnabled(False)

        self.install_btn = QPushButton("Installa aggiornamento e chiudi Tessitura")
        self.install_btn.setObjectName("primary")
        self.install_btn.clicked.connect(self._start_assisted_install)
        self.install_btn.setEnabled(False)

        close_btn = QPushButton("Chiudi")
        close_btn.clicked.connect(self.reject)

        bottom_row.addWidget(self.remind_btn)
        bottom_row.addStretch()
        bottom_row.addWidget(self.download_btn)
        bottom_row.addWidget(self.install_btn)
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)

    def open_github_page_button(self) -> QPushButton:
        return self.open_github_btn

    def _load_preferences(self) -> None:
        settings = load_settings(self.manager.config_file)
        idx = self.channel_combo.findData(settings.update_channel)
        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(max(0, idx))
        self.channel_combo.blockSignals(False)

        self.auto_check_box.blockSignals(True)
        self.auto_check_box.setChecked(settings.auto_check_updates)
        self.auto_check_box.blockSignals(False)

        if settings.last_update_check_ts:
            self.last_check_label.setText(f"Ultimo controllo: {settings.last_update_check_ts}")

    def _on_preferences_changed(self) -> None:
        settings = load_settings(self.manager.config_file)
        settings.update_channel = str(self.channel_combo.currentData() or "stable")
        settings.auto_check_updates = self.auto_check_box.isChecked()
        save_settings(settings, self.manager.config_file)

    def _start_check(self) -> None:
        if self._check_worker is not None and self._check_worker.isRunning():
            return
        channel = str(self.channel_combo.currentData() or "stable")
        self.check_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.verified_package = None
        self.update_headline.setText("Controllo aggiornamenti su GitHub Releases in corso…")

        self._check_worker = GitHubUpdateCheckWorker(
            self.manager,
            channel=channel,
            force=True,
            parent=self,
        )
        self._check_worker.done.connect(self._on_check_done)
        self._check_worker.failed.connect(self._on_check_failed)
        self._check_worker.finished.connect(self._check_worker.deleteLater)
        self._check_worker.finished.connect(lambda: setattr(self, "_check_worker", None))
        self._check_worker.start()

    def _on_check_done(self, result: UpdateCheckResult) -> None:
        self.check_btn.setEnabled(True)
        self.current_result = result
        self._apply_check_result(result)

    def _on_check_failed(self, error_msg: str) -> None:
        self.check_btn.setEnabled(True)
        self.update_headline.setText(f"Controllo aggiornamenti non riuscito: {error_msg}")

    def _apply_check_result(self, result: UpdateCheckResult) -> None:
        self.last_check_label.setText(
            f"Ultimo controllo: {result.checked_at}"
            + (" (da cache locale)" if result.from_cache else "")
        )
        self.update_headline.setText(result.message)

        rel = result.latest_release
        if rel is None:
            self.asset_info_label.setText("")
            self.changelog_view.setPlainText("")
            self.download_btn.setEnabled(False)
            self.remind_btn.setEnabled(False)
            return

        header = f"Release: {rel.name} ({rel.tag_name}) — Pubblicata: {rel.published_at or 'N/D'}"
        if rel.installer_asset is not None:
            size_mb = rel.installer_asset.size / (1024 * 1024) if rel.installer_asset.size > 0 else 0.0
            header += f" | Pacchetto: {rel.installer_asset.name} ({size_mb:.1f} MB)"
        self.asset_info_label.setText(header)
        self.changelog_view.setPlainText(rel.body or "Nessuna nota di rilascio allegata.")

        can_download = (
            result.update_available
            and rel.installer_asset is not None
            and rel.checksum_asset is not None
        )
        self.download_btn.setEnabled(can_download)
        self.remind_btn.setEnabled(result.update_available)
        if result.update_available and rel.installer_asset is None:
            self.asset_info_label.setText(
                header + " — Attenzione: asset installer Windows non presente in questa release."
            )

    def _start_download(self) -> None:
        if (
            self.current_result is None
            or self.current_result.latest_release is None
            or (self._download_worker is not None and self._download_worker.isRunning())
        ):
            return
        self.download_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.download_progress.setValue(0)
        self.download_progress.show()
        self.update_headline.setText("Download e verifica SHA-256 / Authenticode del pacchetto in corso…")

        self._download_worker = GitHubInstallerDownloadWorker(
            self.manager,
            self.current_result.latest_release,
            parent=self,
        )
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.done.connect(self._on_download_verified)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.finished.connect(self._download_worker.deleteLater)
        self._download_worker.finished.connect(lambda: setattr(self, "_download_worker", None))
        self._download_worker.start()

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            pct = int(min(100, (downloaded * 100) // total))
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(pct)
        else:
            self.download_progress.setRange(0, 0)

    def _on_download_verified(self, verification: PackageVerificationResult) -> None:
        self.check_btn.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(100)
        self.verified_package = verification

        if verification.verified and verification.can_auto_execute:
            self.update_headline.setText(
                f"Pacchetto {verification.installer_path.name} verificato. {verification.reason}"
            )
            ok, busy_reason = self.manager.can_start_update()
            self.install_btn.setEnabled(ok)
            if not ok:
                self.asset_info_label.setText(busy_reason)
        else:
            self.install_btn.setEnabled(False)
            self.update_headline.setText(
                f"Avvio automatico bloccato dalla policy di sicurezza: {verification.reason}"
            )
            QMessageBox.warning(
                self,
                "Verifica firma pacchetto",
                f"{verification.reason}\n\n"
                "Puoi aprire la pagina ufficiale GitHub Releases per ispezionare il rilascio.",
            )

    def _on_download_failed(self, error_msg: str) -> None:
        self.check_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.download_progress.hide()
        self.update_headline.setText(f"Errore di verifica/download: {error_msg}")
        QMessageBox.critical(self, "Download aggiornamento fallito", error_msg)

    def _start_assisted_install(self) -> None:
        if self.verified_package is None:
            return
        ok, busy_reason = self.manager.can_start_update()
        if not ok:
            QMessageBox.warning(self, "Operazione in corso", busy_reason)
            return

        warning_extra = ""
        if self.verified_package.requires_user_warning:
            warning_extra = f"\n\nNota di sicurezza: {self.verified_package.reason}\n"

        answer = QMessageBox.question(
            self,
            "Conferma installazione aggiornamento",
            f"Vuoi installare Tessitura {self.verified_package.version} adesso?\n\n"
            "• Le impostazioni verranno salvate in backup automatico.\n"
            "• Le tue registrazioni, sessioni e modelli AI non verranno toccati.\n"
            "• Tessitura verrà chiusa e verrà avviato l'installer ufficiale."
            f"{warning_extra}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            channel = str(self.channel_combo.currentData() or "stable")
            self.manager.prepare_and_launch_installer(
                self.verified_package,
                channel=channel,
            )
            self.accept()
        except (UpdateBlockedByActiveOperationError, InstallerVerificationError) as exc:
            QMessageBox.critical(self, "Impossibile avviare l'aggiornamento", str(exc))

    def _on_remind_later(self) -> None:
        settings = load_settings(self.manager.config_file)
        remind_dt = datetime.now(timezone.utc) + timedelta(hours=24)
        settings.remind_later_until_ts = remind_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        save_settings(settings, self.manager.config_file)
        self.reject()

    def _open_release_page(self) -> None:
        url = self.manager.releases_html_url
        if (
            self.current_result is not None
            and self.current_result.latest_release is not None
            and self.current_result.latest_release.html_url
        ):
            url = self.current_result.latest_release.html_url
        QDesktopServices.openUrl(QUrl(url))

