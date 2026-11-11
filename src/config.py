"""Persistent user configuration for Tessitura stored in %APPDATA%\\Tessitura\\settings.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from src import paths
from src.version import APP_VERSION, CONFIG_SCHEMA_VERSION, load_build_config


@dataclass
class AppSettings:
    """User-level persistent settings outside the installation directory."""

    config_schema_version: int = CONFIG_SCHEMA_VERSION
    app_version: str = APP_VERSION
    previous_version: str | None = None
    recordings_dir: str | None = None
    models_dir: str | None = None
    update_channel: str = "stable"
    auto_check_updates: bool = True
    last_update_check_ts: str | None = None
    skipped_update_version: str | None = None
    remind_later_until_ts: str | None = None
    first_run_completed: bool = False
    pending_update: dict[str, Any] | None = None
    last_migration_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppSettings":
        build_cfg = load_build_config()
        channel = str(raw.get("update_channel") or build_cfg.default_channel).lower()
        if channel not in {"stable", "beta"}:
            channel = build_cfg.default_channel

        recordings_dir = raw.get("recordings_dir")
        if isinstance(recordings_dir, str):
            recordings_dir = recordings_dir.strip() or None
        else:
            recordings_dir = None

        models_dir = raw.get("models_dir")
        if isinstance(models_dir, str):
            models_dir = models_dir.strip() or None
        else:
            models_dir = None

        pending = raw.get("pending_update")
        if not isinstance(pending, dict):
            pending = None

        migration_report = raw.get("last_migration_report")
        if not isinstance(migration_report, dict):
            migration_report = None

        try:
            schema_ver = int(raw.get("config_schema_version", CONFIG_SCHEMA_VERSION))
        except (TypeError, ValueError):
            schema_ver = CONFIG_SCHEMA_VERSION

        return cls(
            config_schema_version=schema_ver,
            app_version=str(raw.get("app_version") or build_cfg.version),
            previous_version=(
                str(raw["previous_version"])
                if raw.get("previous_version") is not None
                else None
            ),
            recordings_dir=recordings_dir,
            models_dir=models_dir,
            update_channel=channel,
            auto_check_updates=bool(raw.get("auto_check_updates", True)),
            last_update_check_ts=(
                str(raw["last_update_check_ts"])
                if raw.get("last_update_check_ts")
                else None
            ),
            skipped_update_version=(
                str(raw["skipped_update_version"])
                if raw.get("skipped_update_version")
                else None
            ),
            remind_later_until_ts=(
                str(raw["remind_later_until_ts"])
                if raw.get("remind_later_until_ts")
                else None
            ),
            first_run_completed=bool(raw.get("first_run_completed", False)),
            pending_update=pending,
            last_migration_report=migration_report,
        )


def settings_path(config_root: Path | None = None) -> Path:
    root = config_root if config_root is not None else paths.config_dir()
    return root / "settings.json"


def load_settings(path: Path | None = None) -> AppSettings:
    target = path if path is not None else settings_path()
    if not target.is_file():
        build_cfg = load_build_config()
        return AppSettings(
            config_schema_version=CONFIG_SCHEMA_VERSION,
            app_version=build_cfg.version,
            update_channel=build_cfg.default_channel,
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            return AppSettings()
        return AppSettings.from_dict(raw)
    except (OSError, json.JSONDecodeError):
        return AppSettings()


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    target = path if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".json.tmp")
    payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, target)
    return target


def update_settings(path: Path | None = None, **changes: Any) -> AppSettings:
    current = load_settings(path)
    data = current.to_dict()
    data.update(changes)
    updated = AppSettings.from_dict(data)
    save_settings(updated, path)
    return updated

