"""Crash-safe, content-addressed pipeline checkpoints.

The checkpoint contains only serializable intermediate data. The key includes
the audio bytes, effective options, model revisions and algorithm version, so
changing a parameter cannot silently reuse a stale ASR/diarization result.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

ALGORITHM_VERSION = "pipeline-v3"
PHASES = ("prepare", "asr", "alignment", "diarization", "merge", "review")


def _canonical(value):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Opzioni non finite")
        return round(value, 12)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def file_digest(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(audio_paths, options, model_revisions=None, *, algorithm=ALGORITHM_VERSION) -> str:
    payload = {
        "algorithm": algorithm,
        "audio": [{"path": Path(path).name, "sha256": file_digest(path)} for path in audio_paths],
        "options": _canonical(options),
        "models": _canonical(model_revisions or {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class CheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, phase: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or phase not in PHASES:
            raise ValueError("Checkpoint non valido")
        return self.root / key / f"{phase}.json"

    def save(
        self,
        key: str,
        phase: str,
        payload: dict,
        *,
        depends_on: list[str] | None = None,
        input_hash: str | None = None,
    ) -> Path:
        if not isinstance(payload, dict):
            raise TypeError("Il checkpoint deve essere un oggetto")
        target = self._path(key, phase)
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "key": key,
            "phase": phase,
            "version": ALGORITHM_VERSION,
            "depends_on": depends_on or [],
            "input_hash": input_hash or "",
            "payload": payload,
        }
        fd, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
        return target

    def load(self, key: str, phase: str, *, check_dependencies: bool = True) -> dict | None:
        target = self._path(key, phase)
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
            if (value.get("key") != key or value.get("phase") != phase or
                    value.get("version") != ALGORITHM_VERSION or
                    not isinstance(value.get("payload"), dict)):
                return None
            if check_dependencies:
                dependencies = value.get("depends_on", [])
                for dep_phase in dependencies:
                    if not self.load(key, dep_phase, check_dependencies=False):
                        # Dipendenza mancante o invalidata
                        return None
            return value["payload"]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def invalidate(self, key: str, from_phase: str) -> int:
        """Remove this and all later phases after an option/model change."""
        if from_phase not in PHASES:
            raise ValueError("Fase non valida")
        start = PHASES.index(from_phase)
        removed = 0
        for phase in PHASES[start:]:
            target = self._path(key, phase)
            try:
                target.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        return removed

    def clear(self, key: str) -> None:
        import shutil
        folder = (self.root / key).resolve()
        if folder.parent != self.root or len(key) != 64:
            raise ValueError("Chiave checkpoint non sicura")
        shutil.rmtree(folder, ignore_errors=True)
