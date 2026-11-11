"""Unit and offscreen GUI tests for FirstRunDialog, UpdateDialog, and first-run model provisioning."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from PySide6.QtWidgets import QApplication
from src.gui.first_run_dialog import FirstRunDialog
from src.gui.update_dialog import UpdateDialog
from src.hardware import Hardware
from src.processing.first_run import check_first_run_status, configure_user_directories
from src.update_manager import (
    ReleaseAssetInfo,
    ReleaseInfo,
    UpdateCheckResult,
    UpdateManager,
)


class FirstRunAndUpdateGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        for widget in list(self.app.topLevelWidgets()):
            widget.close()
            widget.deleteLater()
        self.app.processEvents()
        self._tmp.cleanup()

    def test_first_run_status_and_custom_directory_configuration(self) -> None:
        cfg_file = self.root / "config" / "settings.json"
        custom_models = self.root / "custom_models"
        custom_rec = self.root / "custom_recordings"

        hw = Hardware("cuda", "int8_float16", "large-v3", "NVIDIA RTX 4070 Laptop GPU", 8.0, 16)
        status = check_first_run_status(hw, config_file=cfg_file)
        self.assertEqual(status.recommended_asr_model, "large-v3")
        self.assertGreater(status.free_disk_gb, 0.0)

        saved = configure_user_directories(
            models_dir=custom_models,
            recordings_dir=custom_rec,
            mark_completed=True,
            config_file=cfg_file,
        )
        self.assertTrue(saved.first_run_completed)
        self.assertEqual(Path(saved.models_dir), custom_models.resolve())
        self.assertEqual(Path(saved.recordings_dir), custom_rec.resolve())

    def test_first_run_dialog_renders_hardware_and_directories(self) -> None:
        hw = Hardware("cuda", "int8_float16", "large-v3", "NVIDIA RTX 4070 Laptop GPU", 8.0, 16)
        dlg = FirstRunDialog(hw)
        self.addCleanup(dlg.deleteLater)
        self.assertIn("large-v3", dlg.hw_summary_label.text())
        self.assertIn("GB", dlg.disk_summary_label.text())

    def test_update_dialog_displays_release_and_changelog(self) -> None:
        cfg_file = self.root / "config" / "settings.json"
        mgr = UpdateManager(
            current_version="1.0.0",
            config_file=cfg_file,
            cache_file=self.root / "cache" / "gh.json",
            downloads_dir=self.root / "cache" / "updates",
        )
        rel = ReleaseInfo(
            tag_name="v1.1.0",
            version="1.1.0",
            name="Tessitura 1.1.0",
            body="- Supporto installer Windows\n- Aggiornamenti automatici GitHub Releases",
            html_url="https://github.com/aless/tessitura/releases/tag/v1.1.0",
            published_at="2026-11-11T10:00:00Z",
            prerelease=False,
            draft=False,
            installer_asset=ReleaseAssetInfo(
                name="Tessitura-Setup-1.1.0.exe",
                download_url="https://example.test/Tessitura-Setup-1.1.0.exe",
                size=52428800,
            ),
            checksum_asset=ReleaseAssetInfo(
                name="SHA256SUMS.txt",
                download_url="https://example.test/SHA256SUMS.txt",
                size=96,
            ),
        )
        initial = UpdateCheckResult(
            status="update_available",
            channel="stable",
            current_version="1.0.0",
            checked_at="2026-11-11T10:05:00Z",
            update_available=True,
            latest_release=rel,
            message="Nuova versione disponibile: 1.1.0 (installata: 1.0.0).",
        )
        dlg = UpdateDialog(mgr, initial_result=initial)
        self.addCleanup(dlg.deleteLater)
        self.assertTrue(dlg.download_btn.isEnabled())
        self.assertFalse(dlg.install_btn.isEnabled())
        self.assertIn("Supporto installer Windows", dlg.changelog_view.toPlainText())
        self.assertIn("Tessitura-Setup-1.1.0.exe", dlg.asset_info_label.text())


if __name__ == "__main__":
    unittest.main()

