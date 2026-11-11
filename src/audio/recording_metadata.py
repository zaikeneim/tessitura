"""Restore capture provenance and native tracks from a sibling sidecar."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_recording_metadata(source: str | Path) -> dict | None:
    source = Path(source).resolve()
    stem = source.stem
    for suffix in ("_mic", "_system"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    sidecar = source.with_name(stem + ".recording.json")
    if not sidecar.is_file():
        return None
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("recording_mode") not in ("ambient", "headset", "speakers"):
        raise ValueError("Metadati di acquisizione non riconosciuti")
    if value.get("state") not in ("complete", "review_required"):
        raise ValueError("Acquisizione incompleta: sincronizzazione non verificata")
    for key in ("mic_single_speaker_verified", "echo_cancellation"):
        if type(value.get(key, False)) is not bool:
            raise ValueError(f"Opzione non valida nei metadati: {key}")
    tracks = value.get("tracks", {})
    if not isinstance(tracks, dict) or not tracks or set(tracks) - {"mic", "system"}:
        raise ValueError("Metadati senza sorgenti valide")
    for track in tracks.values():
        if not isinstance(track, dict):
            raise ValueError("Sorgente non valida nei metadati")
        name = track.get("path")
        if not isinstance(name, str) or Path(name).name != name or Path(name).is_absolute():
            raise ValueError("Percorso sorgente non valido nei metadati")
        offset = track.get("offset_seconds")
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0:
            raise ValueError("Offset sorgente non valido nei metadati")
        if not (source.parent / name).is_file():
            raise FileNotFoundError(f"Traccia originale mancante: {name}")
    exports = {stem + ".mp3"} | {Path(track["path"]).with_suffix(".mp3").name for track in tracks.values()}
    if source.name not in [track["path"] for track in tracks.values()] and source.name not in exports:
        raise ValueError("Il file selezionato non appartiene alla registrazione")
    return value


def recording_sources(source: str | Path, metadata: dict) -> tuple[Path | None, Path | None, list[float]]:
    directory = Path(source).resolve().parent
    tracks = metadata["tracks"]
    sources = [directory / tracks[kind]["path"] if kind in tracks else None for kind in ("mic", "system")]
    offsets = [float(tracks[kind].get("offset_seconds", 0.0)) for kind in ("mic", "system") if kind in tracks]
    return sources[0], sources[1], offsets
