"""Tests for installed-mode directory separation, configuration persistence, backups, and migrations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import paths
from src.config import AppSettings, load_settings, save_settings
from src.migrations import (
    create_state_backup,
    migrate_session_dict,
    restore_state_backup,
    validate_and_migrate_startup,
)
from src.version import CONFIG_SCHEMA_VERSION, SESSION_SCHEMA_VERSION


class MigrationsAndConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        paths.reset_caches()

    def tearDown(self) -> None:
        paths.reset_caches()
        self._tmp.cleanup()

    def test_installed_mode_separates_mutable_directories_from_app_root(self) -> None:
        install_root = self.root / "Programs" / "Tessitura"
        install_root.mkdir(parents=True)
        (install_root / ".installed").write_text("version=1.0.0\n", encoding="utf-8")

        appdata_roaming = self.root / "AppData" / "Roaming"
        appdata_local = self.root / "AppData" / "Local"

        env_overrides = {
            "APPDATA": str(appdata_roaming),
            "LOCALAPPDATA": str(appdata_local),
            "TESSITURA_INSTALL_DIR": str(install_root),
            "TESSITURA_INSTALLED": "1",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            os.environ.pop("TESSITURA_DATA_DIR", None)
            os.environ.pop("KISP_DATA_DIR", None)
            os.environ.pop("TESSITURA_MODELS_DIR", None)
            os.environ.pop("TESSITURA_CONFIG_DIR", None)
            os.environ.pop("TESSITURA_CACHE_DIR", None)
            os.environ.pop("TESSITURA_LOGS_DIR", None)
            os.environ.pop("TESSITURA_USER_DATA_DIR", None)
            paths.reset_caches()

            with patch.object(paths, "app_root", return_value=install_root):
                self.assertTrue(paths.is_installed_mode())
                cfg = paths.config_dir()
                dat = paths.data_dir()
                mod = paths.models_dir()
                cac = paths.cache_dir()
                lgs = paths.logs_dir()

                # None of the mutable user directories may be inside install_root!
                for d in (cfg, dat, mod, cac, lgs):
                    self.assertNotEqual(d.resolve(), install_root.resolve())
                    self.assertNotIn(install_root.resolve(), d.resolve().parents)

                self.assertEqual(cfg, appdata_roaming / "Tessitura")
                self.assertEqual(mod, appdata_local / "Tessitura" / "models")
                self.assertEqual(cac, appdata_local / "Tessitura" / "cache")
                self.assertEqual(lgs, appdata_local / "Tessitura" / "logs")
                self.assertEqual(dat, appdata_local / "Tessitura" / "data")

    def test_config_migration_v0_to_v1_creates_backup(self) -> None:
        cfg_file = self.root / "config" / "settings.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(
            json.dumps(
                {
                    "config_schema_version": 0,
                    "app_version": "0.9.0",
                    "channel": "beta",
                    "auto_check": False,
                }
            ),
            encoding="utf-8",
        )
        report = validate_and_migrate_startup(current_version="1.0.0", config_file=cfg_file)
        self.assertIn(report.status, {"migrated", "update_verified"})
        self.assertEqual(report.from_schema, 0)
        self.assertEqual(report.to_schema, CONFIG_SCHEMA_VERSION)
        self.assertIsNotNone(report.backup_dir)
        self.assertTrue((Path(report.backup_dir) / "settings.json").is_file())

        loaded = load_settings(cfg_file)
        self.assertEqual(loaded.config_schema_version, CONFIG_SCHEMA_VERSION)
        self.assertEqual(loaded.update_channel, "beta")
        self.assertFalse(loaded.auto_check_updates)
        self.assertEqual(loaded.app_version, "1.0.0")
        self.assertEqual(loaded.previous_version, "0.9.0")

    def test_post_update_verification_and_rollback_detection(self) -> None:
        cfg_file = self.root / "config" / "settings.json"
        settings = AppSettings(
            app_version="1.0.0",
            pending_update={
                "target_version": "1.1.0",
                "previous_version": "1.0.0",
                "installer_path": "C:/temp/Tessitura-Setup-1.1.0.exe",
                "backup_dir": str(self.root / "backups" / "b1"),
            },
        )
        save_settings(settings, cfg_file)

        # Case 1: User cancelled the installer, app restarts still at 1.0.0
        rep_cancel = validate_and_migrate_startup(current_version="1.0.0", config_file=cfg_file)
        self.assertEqual(rep_cancel.status, "update_cancelled_or_rolled_back")
        after_cancel = load_settings(cfg_file)
        self.assertIsNone(after_cancel.pending_update)

        # Case 2: Installer succeeded, app starts at 1.1.0
        after_cancel.pending_update = {
            "target_version": "1.1.0",
            "previous_version": "1.0.0",
        }
        save_settings(after_cancel, cfg_file)
        rep_ok = validate_and_migrate_startup(current_version="1.1.0", config_file=cfg_file)
        self.assertEqual(rep_ok.status, "update_verified")
        after_ok = load_settings(cfg_file)
        self.assertEqual(after_ok.app_version, "1.1.0")
        self.assertEqual(after_ok.previous_version, "1.0.0")
        self.assertIsNone(after_ok.pending_update)

    def test_incompatible_newer_schema_is_not_overwritten(self) -> None:
        cfg_file = self.root / "config" / "settings.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        future_payload = {
            "config_schema_version": 99,
            "app_version": "9.0.0",
            "future_setting": "preserve_me",
        }
        cfg_file.write_text(json.dumps(future_payload), encoding="utf-8")

        report = validate_and_migrate_startup(current_version="1.0.0", config_file=cfg_file)
        self.assertEqual(report.status, "incompatible_newer_schema")
        # File on disk must remain untouched!
        on_disk = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["config_schema_version"], 99)
        self.assertEqual(on_disk["future_setting"], "preserve_me")

    def test_backup_and_restore_roundtrip(self) -> None:
        cfg_root = self.root / "cfg"
        data_root = self.root / "dat"
        cfg_root.mkdir(parents=True)
        data_root.mkdir(parents=True)
        (cfg_root / "settings.json").write_text('{"app_version": "1.0.0"}', encoding="utf-8")
        (data_root / "projects.json").write_text('{"projects": ["a"]}', encoding="utf-8")

        backup = create_state_backup(
            config_root=cfg_root,
            data_root=data_root,
            previous_version="1.0.0",
            target_version="1.1.0",
        )
        (cfg_root / "settings.json").write_text('{"app_version": "corrupted"}', encoding="utf-8")

        restored = restore_state_backup(
            backup.backup_dir,
            config_root=cfg_root,
            data_root=data_root,
        )
        self.assertIn("settings.json", restored)
        self.assertIn("projects.json", restored)
        self.assertEqual(
            json.loads((cfg_root / "settings.json").read_text(encoding="utf-8"))["app_version"],
            "1.0.0",
        )

    def test_migrate_legacy_session_dict_to_v4(self) -> None:
        legacy = {
            "id": "sess-1",
            "schema_version": 2,
            "segments": [{"start": 0.0, "end": 1.2, "text": "Ciao", "speaker": "SPEAKER_00"}],
        }
        migrated = migrate_session_dict(legacy)
        self.assertEqual(migrated["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(migrated["migrated_from_schema_version"], 2)
        self.assertEqual(len(migrated["segments"]), 1)


if __name__ == "__main__":
    unittest.main()

