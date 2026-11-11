"""Cataloghi e stato cache dei motori di trascrizione e diarizzazione."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable

from .model_store import cached_path, local_revision


@dataclass(frozen=True)
class ModelEntry:
    category: str
    key: str
    label: str
    repo: str
    approx_gb: float | None = None
    context_tokens: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelStatus:
    entry: ModelEntry
    cached: bool
    local_revision: str | None = None
    remote_revision: str | None = None
    error: str | None = None

    @property
    def update_available(self) -> bool:
        return bool(
            self.cached and self.local_revision and self.remote_revision
            and self.local_revision != self.remote_revision
        )

    @property
    def state(self) -> str:
        if self.error:
            return f"non verificato: {self.error}"
        if not self.cached:
            return "non scaricato"
        if self.update_available:
            return "aggiornamento disponibile"
        if not self.remote_revision:
            return "in cache · revisione remota non verificata"
        return "in cache e aggiornato"


def _whisper_repos() -> dict[str, str]:
    fallback = {
        "large-v3": "Systran/faster-whisper-large-v3",
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
        "tiny": "Systran/faster-whisper-tiny",
    }
    try:
        from faster_whisper.utils import _MODELS

        return {key: _MODELS.get(key, repo) for key, repo in fallback.items()}
    except Exception:
        return fallback


def transcription_catalog() -> tuple[ModelEntry, ...]:
    repos = _whisper_repos()
    sizes = {
        "large-v3": 3.1, "large-v3-turbo": 1.6, "medium": 1.5,
        "small": 0.5, "base": 0.15, "tiny": 0.08,
    }
    return tuple(
        ModelEntry(
            category="trascrizione",
            key=key,
            label=key,
            repo=repos[key],
            approx_gb=sizes.get(key),
            notes="faster-whisper / Whisper",
        )
        for key in ("large-v3", "large-v3-turbo", "medium", "small", "base", "tiny")
    )


def diarization_catalog() -> tuple[ModelEntry, ...]:
    from .diarize import Diarizer

    return (
        ModelEntry(
            category="diarizzazione",
            key=Diarizer.MODEL,
            label="pyannote speaker-diarization-community-1",
            repo=Diarizer.MODEL,
            approx_gb=None,
            notes="richiede token HuggingFace e accettazione delle condizioni del modello",
        ),
    )


def _cached(repo: str) -> bool:
    try:
        category = "diarizzazione" if repo.startswith("pyannote/") else "trascrizione"
        return cached_path(repo, category) is not None
    except Exception:
        return False


def _status(entry: ModelEntry, remote: bool) -> ModelStatus:
    cached = _cached(entry.repo)
    local = local_revision(entry.repo)
    if not remote:
        return ModelStatus(entry, cached, local_revision=local)
    try:
        from huggingface_hub import HfApi

        remote_revision = HfApi().model_info(entry.repo, timeout=15).sha or None
        return ModelStatus(entry, cached, local, remote_revision)
    except Exception as exc:  # noqa: BLE001 - lo stato locale resta valido
        return ModelStatus(entry, cached, local, error=f"rete: {type(exc).__name__}")


def catalog_status(
    entries: Iterable[ModelEntry],
    *,
    check_remote: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[ModelStatus]:
    """Stato cache/revisione per un catalogo, con richieste remote parallele."""
    entries = tuple(entries)
    say = progress or (lambda _message: None)
    if not check_remote:
        return [_status(entry, False) for entry in entries]
    result: dict[str, ModelStatus] = {}
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(entries)))) as executor:
        futures = {executor.submit(_status, entry, True): entry for entry in entries}
        for index, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            result[item.entry.repo] = item
            say(f"Verifica motori: {index}/{len(entries)}")
    return [result[entry.repo] for entry in entries]
