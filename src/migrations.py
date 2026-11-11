"""Configuration and data schema migration, backup, and post-update validation for Tessitura."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from src import paths
from src.config import AppSettings, load_settings, save_settings, settings_path
from src.version import (
    CONFIG_SCHEMA_VERSION,
    PROJECT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    SemVer,
    get_app_version,
)


class IncompatibleSchemaError(RuntimeError):
    """Raised when a configuration or data file requires a newer version of Tessitura."""


@dataclass(frozen=True)
class BackupRecord:
    """Metadata describing a configuration and state backup created before migration/update."""

    backup_dir: Path
    created_at: str
    previous_version: str
    target_version: str | None
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_dir": str(self.backup_dir),
            "created_at": self.created_at,
            "previous_version": self.previous_version,
            "target_version": self.target_version,
            "files": list(self.files),
        }


@dataclass(frozen=True)
class MigrationReport:
    """Result of startup configuration/data validation and post-update check."""

    status: str  # "ok" | "migrated" | "update_verified" | "update_cancelled_or_rolled_back" | "downgrade_detected" | "incompatible_newer_schema"
    current_version: str
    previous_version: str | None = None
    target_version: str | None = None
    from_schema: int = CONFIG_SCHEMA_VERSION
    to_schema: int = CONFIG_SCHEMA_VERSION
    backup_dir: str | None = None
    message: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_state_backup(
    *,
    config_root: Path | None = None,
    data_root: Path | None = None,
    previous_version: str | None = None,
    target_version: str | None = None,
) -> BackupRecord:
    """Create a timestamped backup of configuration and lightweight metadata before an update or migration."""
    cfg_root = config_root if config_root is not None else paths.config_dir()
    d_root = data_root if data_root is not None else paths.data_dir()
    prev_ver = previous_version or get_app_version()
    ts = _utc_compact_ts()
    suffix = f"{prev_ver}_to_{target_version}" if target_version else prev_ver
    safe_suffix = "".join(c if c.isalnum() or c in "._-" else "_" for c in suffix)
    backup_dir = cfg_root / "backups" / f"backup-{safe_suffix}-{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    cfg_file = cfg_root / "settings.json"
    if cfg_file.is_file():
        shutil.copy2(cfg_file, backup_dir / "settings.json")
        copied_files.append("settings.json")

    projects_file = d_root / "projects.json"
    if projects_file.is_file():
        shutil.copy2(projects_file, backup_dir / "projects.json")
        copied_files.append("projects.json")

    manifest = {
        "created_at": _utc_now_iso(),
        "previous_version": prev_ver,
        "target_version": target_version,
        "files": copied_files,
    }
    (backup_dir / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return BackupRecord(
        backup_dir=backup_dir,
        created_at=manifest["created_at"],
        previous_version=prev_ver,
        target_version=target_version,
        files=tuple(copied_files),
    )


def restore_state_backup(
    backup_dir: Path,
    *,
    config_root: Path | None = None,
    data_root: Path | None = None,
) -> list[str]:
    """Restore backed up configuration files from a backup directory."""
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"Backup directory does not exist: {backup_dir}")
    cfg_root = config_root if config_root is not None else paths.config_dir()
    d_root = data_root if data_root is not None else paths.data_dir()
    restored: list[str] = []

    src_settings = backup_dir / "settings.json"
    if src_settings.is_file():
        cfg_root.mkdir(parents=True, exist_ok=True)
        tmp = cfg_root / "settings.json.restore.tmp"
        shutil.copy2(src_settings, tmp)
        os.replace(tmp, cfg_root / "settings.json")
        restored.append("settings.json")

    src_projects = backup_dir / "projects.json"
    if src_projects.is_file():
        d_root.mkdir(parents=True, exist_ok=True)
        tmp = d_root / "projects.json.restore.tmp"
        shutil.copy2(src_projects, tmp)
        os.replace(tmp, d_root / "projects.json")
        restored.append("projects.json")

    return restored


def _migrate_config_dict_v0_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    migrated["config_schema_version"] = 1
    if "update_channel" not in migrated:
        migrated["update_channel"] = str(migrated.get("channel") or "stable").lower()
    if "auto_check_updates" not in migrated:
        migrated["auto_check_updates"] = bool(migrated.get("auto_check", True))
    return migrated


def validate_and_migrate_startup(
    *,
    current_version: str | None = None,
    config_file: Path | None = None,
) -> MigrationReport:
    """Validate config schema, run migrations if needed, and verify post-update state."""
    curr_ver = current_version or get_app_version()
    target_cfg = config_file if config_file is not None else settings_path()
    cfg_root = target_cfg.parent

    if not target_cfg.is_file():
        settings = AppSettings(
            config_schema_version=CONFIG_SCHEMA_VERSION,
            app_version=curr_ver,
        )
        save_settings(settings, target_cfg)
        return MigrationReport(
            status="ok",
            current_version=curr_ver,
            from_schema=CONFIG_SCHEMA_VERSION,
            to_schema=CONFIG_SCHEMA_VERSION,
            message="Configurazione iniziale creata.",
        )

    try:
        raw = json.loads(target_cfg.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, json.JSONDecodeError):
        backup = create_state_backup(
            config_root=cfg_root,
            previous_version=curr_ver,
        )
        settings = AppSettings(
            config_schema_version=CONFIG_SCHEMA_VERSION,
            app_version=curr_ver,
        )
        save_settings(settings, target_cfg)
        return MigrationReport(
            status="migrated",
            current_version=curr_ver,
            backup_dir=str(backup.backup_dir),
            message="File settings.json danneggiato; creato backup e ripristinata configurazione predefinita.",
            warnings=("Il file settings.json precedente non era valido ed è stato salvato in backup.",),
        )

    raw_schema = raw.get("config_schema_version", 0)
    try:
        schema_ver = int(raw_schema)
    except (TypeError, ValueError):
        schema_ver = 0

    if schema_ver > CONFIG_SCHEMA_VERSION:
        # Newer schema than supported by this executable: do NOT overwrite!
        return MigrationReport(
            status="incompatible_newer_schema",
            current_version=curr_ver,
            previous_version=str(raw.get("app_version") or ""),
            from_schema=schema_ver,
            to_schema=CONFIG_SCHEMA_VERSION,
            message=(
                f"La configurazione esistente usa lo schema v{schema_ver}, "
                f"più recente dello schema supportato (v{CONFIG_SCHEMA_VERSION}) da Tessitura {curr_ver}. "
                "Il file di configurazione non è stato sovrascritto."
            ),
            warnings=(
                "Rilevata configurazione creata da una versione più recente di Tessitura.",
            ),
        )

    backup_dir_str: str | None = None
    migrated_schema = False
    if schema_ver < CONFIG_SCHEMA_VERSION:
        backup = create_state_backup(
            config_root=cfg_root,
            previous_version=str(raw.get("app_version") or curr_ver),
            target_version=curr_ver,
        )
        backup_dir_str = str(backup.backup_dir)
        if schema_ver < 1:
            raw = _migrate_config_dict_v0_to_v1(raw)
        migrated_schema = True

    settings = AppSettings.from_dict(raw)
    warnings: list[str] = []
    status = "migrated" if migrated_schema else "ok"
    message = "Configurazione verificata."

    # Check if a pending update was recorded before launching the installer
    pending = settings.pending_update
    prev_recorded = settings.app_version
    target_ver: str | None = None

    if isinstance(pending, dict):
        target_ver = str(pending.get("target_version") or "").strip() or None
        pending_prev = str(pending.get("previous_version") or prev_recorded or "").strip() or None
        pending_backup = str(pending.get("backup_dir") or "").strip() or None
        if pending_backup and not backup_dir_str:
            backup_dir_str = pending_backup

        try:
            sem_curr = SemVer.parse(curr_ver)
            sem_target = SemVer.parse(target_ver) if target_ver else None
            sem_prev = SemVer.parse(pending_prev) if pending_prev else None
        except ValueError:
            sem_curr = None
            sem_target = None
            sem_prev = None

        if sem_curr and sem_target and sem_curr == sem_target:
            status = "update_verified"
            settings.previous_version = pending_prev
            settings.app_version = curr_ver
            settings.pending_update = None
            message = f"Aggiornamento completato con successo da {pending_prev} a {curr_ver}."
        elif sem_curr and sem_prev and sem_curr == sem_prev:
            status = "update_cancelled_or_rolled_back"
            settings.pending_update = None
            message = (
                f"L'aggiornamento verso {target_ver} non è stato completato; "
                f"Tessitura è rimasta alla versione {curr_ver}."
            )
            warnings.append(message)
        else:
            status = "update_verified"
            settings.previous_version = pending_prev
            settings.app_version = curr_ver
            settings.pending_update = None
            message = f"Versione attiva {curr_ver} verificata dopo operazione di aggiornamento."
    else:
        # Check normal version transition (upgrade or downgrade)
        if prev_recorded and prev_recorded != curr_ver:
            try:
                sem_curr = SemVer.parse(curr_ver)
                sem_prev = SemVer.parse(prev_recorded)
                if sem_curr < sem_prev:
                    status = "downgrade_detected"
                    warnings.append(
                        f"Rilevato avvio di una versione precedente ({curr_ver}) rispetto a {prev_recorded}."
                    )
                    message = warnings[-1]
                elif sem_curr > sem_prev:
                    status = "update_verified"
                    message = f"Aggiornamento rilevato da {prev_recorded} a {curr_ver}."
            except ValueError:
                pass
            settings.previous_version = prev_recorded
            settings.app_version = curr_ver

    report = MigrationReport(
        status=status,
        current_version=curr_ver,
        previous_version=settings.previous_version,
        target_version=target_ver,
        from_schema=schema_ver,
        to_schema=CONFIG_SCHEMA_VERSION,
        backup_dir=backup_dir_str,
        message=message,
        warnings=tuple(warnings),
    )
    settings.last_migration_report = report.to_dict()
    save_settings(settings, target_cfg)
    return report


def migrate_session_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand legacy sessions to canonical events without deleting old fields."""
    result = dict(payload)
    schema = int(result.get("schema_version") or 1)
    if schema < SESSION_SCHEMA_VERSION:
        from src.models.session import Session
        session = Session.from_dict(payload)
        session.migrate_to_v6()
        result.update(session.to_dict())
        result.setdefault("segments", [])
        result.setdefault("words", [])
        result.setdefault("speaker_names", {})
        result.setdefault("overlaps", [])
        result.setdefault("uncertain_spans", [])
        result.setdefault("pipeline_reports", {})
        result["migrated_from_schema_version"] = schema
        result["schema_version"] = SESSION_SCHEMA_VERSION
    return result

