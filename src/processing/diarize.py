"""Diarizzazione (chi parla quando) con pyannote.audio.

Richiede un token HuggingFace e l'accettazione delle condizioni del modello
`pyannote/speaker-diarization-community-1` (bundle all-in-one di pyannote 4.x).
Il token si legge dalla variabile d'ambiente HF_TOKEN o da un file `.hf_token`
cercato nella cartella dell'app, nella cartella dati e nella home dell'utente:
cosi' l'app funziona anche lanciata da un'altra cartella o su un altro PC.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from .adaptation import normalize_embedding


def read_hf_token() -> str | None:
    """Token HF da env (HF_TOKEN/HUGGINGFACE_TOKEN) o da un file .hf_token."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    for p in paths.hf_token_candidates():
        try:
            if p.is_file():
                token = p.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except OSError:
            continue
    return None


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    """Tutto quello che community-1 sa dire, non solo la diarizzazione ordinaria.

    - `turns`: output ordinario, che ammette due voci nello stesso istante ed e'
      l'unico che dice dove il parlato e' davvero sovrapposto.
    - `exclusive`: una voce sola per istante. E' quello che la scheda del modello
      indica per riconciliare i turni con i tempi imprecisi dell'ASR: assegnare
      una parola quando due turni la coprono entrambi e' una monetina.
    - `overlaps`: gli intervalli con almeno due voci contemporanee. Misurati sul
      materiale reale del corpus valgono il 10% del parlato: non e' un caso
      limite da ignorare, e va mostrato invece che deciso a caso.
    """

    turns: list[Turn]
    exclusive: list[Turn]
    overlaps: list[tuple[float, float]]
    # Centroidi restituiti da pyannote, allineati alle etichette della sua
    # Annotation. Servono per riconoscere in modo prudente le stesse voci nelle
    # sedute successive; il default mantiene compatibili i backend più vecchi.
    speaker_embeddings: dict[str, list[float]] = field(default_factory=dict)
    model_repo: str = ""
    model_revision: str = ""

    def overlap_seconds(self) -> float:
        return sum(end - start for start, end in self.overlaps)


def _to_turns(annotation) -> list[Turn]:
    if annotation is None:
        return []
    turns = [
        Turn(start=segment.start, end=segment.end, speaker=speaker)
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: turn.start)
    return turns


def _overlapping_spans(turns: list[Turn]) -> list[tuple[float, float]]:
    """Intervalli coperti da almeno due voci *diverse*.

    Si contano le voci attive, non gli intervalli: due turni della stessa voce
    che si accavallano (annotazioni duplicate o frammentate) non sono parlato
    simultaneo. Le fini vengono elaborate prima degli inizi allo stesso istante,
    quindi due turni adiacenti non producono sovrapposizione.
    """
    events: list[tuple[float, int, str]] = []
    for turn in turns:
        events.append((turn.start, 1, turn.speaker))
        events.append((turn.end, -1, turn.speaker))
    events.sort(key=lambda item: (item[0], item[1]))
    spans: list[tuple[float, float]] = []
    active: dict[str, int] = {}
    opened: float | None = None
    for time, delta, speaker in events:
        was_overlapping = sum(1 for count in active.values() if count > 0) >= 2
        active[speaker] = active.get(speaker, 0) + delta
        overlapping = sum(1 for count in active.values() if count > 0) >= 2
        if overlapping and not was_overlapping:
            opened = time
        elif not overlapping and was_overlapping and opened is not None:
            if time > opened:
                spans.append((opened, time))
            opened = None
    return spans


def _speaker_embedding_map(annotation, raw_embeddings) -> dict[str, list[float]]:
    """Converte i centroidi pyannote in un mapping JSON-safe etichetta→vettore."""
    if annotation is None or raw_embeddings is None:
        return {}
    try:
        labels = [str(label) for label in annotation.labels()]
        rows = raw_embeddings.tolist() if hasattr(raw_embeddings, "tolist") else list(raw_embeddings)
    except (AttributeError, TypeError, ValueError):
        return {}
    if len(labels) != len(rows):
        return {}
    result: dict[str, list[float]] = {}
    for label, row in zip(labels, rows):
        vector = normalize_embedding(row)
        if vector is not None:
            result[label] = [round(item, 8) for item in vector]
    return result


def _model_path(repo: str, token: str | None) -> str:
    """Use the complete local bundle without checking/replacing its revision."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    files = ["config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin",
             "plda/plda.npz", "plda/xvec_transform.npz"]
    try:
        cached = Path(snapshot_download(repo, local_files_only=True))
        if all((cached / file).is_file() and (cached / file).stat().st_size > 0 for file in files):
            return str(cached)
    except LocalEntryNotFoundError:
        pass
    if not token:
        raise RuntimeError("Modello diarizzazione non presente completo in cache. Serve un token HuggingFace per il primo download.")
    return snapshot_download(repo, token=token, allow_patterns=files)


class Diarizer:
    """Wrapper su pyannote. Carica la pipeline una volta e la riusa."""

    # pyannote.audio 4.x usa il modello "community-1" (bundle all-in-one).
    MODEL = "pyannote/speaker-diarization-community-1"

    def __init__(self, token: str | None = None, device: str = "cuda"):
        import torch
        from pyannote.audio import Pipeline

        from ..bootstrap import install_system_certs

        install_system_certs()  # il primo load scarica i modelli da HuggingFace
        token = token or read_hf_token()
        model_path = Path(_model_path(self.MODEL, token))
        self.model_revision = (
            model_path.name if model_path.parent.name == "snapshots" else ""
        )
        self._pipeline = Pipeline.from_pretrained(str(model_path), token=token)
        if self._pipeline is None:
            raise RuntimeError(
                "Impossibile caricare la pipeline pyannote: token non valido o "
                "condizioni dei modelli non accettate su huggingface.co."
            )
        self._pipeline.to(torch.device(device))

    # Nomi leggibili dei passi interni di pyannote, per la barra di avanzamento.
    STEP_NAMES = {
        "segmentation": "analisi dei turni",
        "embeddings": "impronte vocali",
        "speaker_counting": "conteggio voci",
        "discrete_diarization": "assegnazione voci",
        "clustering": "raggruppamento voci",
    }

    def diarize(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> list[Turn]:
        """Solo i turni ordinari: firma storica, usata da chi non vuole il resto."""
        return self.diarize_detailed(audio_path, **kwargs).turns

    def diarize_detailed(
        self,
        audio_path: str | Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[str, float | None], None] | None = None,
    ) -> DiarizationResult:
        """Turni ordinari, turni esclusivi e zone di parlato sovrapposto.

        Se conosci il numero di partecipanti passa num_speakers per risultati
        migliori; altrimenti puoi vincolare con min_speakers/max_speakers.

        on_progress(passo, frazione) viene chiamato durante l'elaborazione:
        pyannote espone un hook per ogni fase interna, senza il quale la UI
        resterebbe ferma per minuti e sembrerebbe bloccata.
        """
        kwargs: dict = {}
        if on_progress is not None:
            def _hook(step_name, step_artifact=None, file=None, total=None, completed=None):
                label = self.STEP_NAMES.get(step_name, step_name)
                frac = (completed / total) if (total and completed is not None) else None
                on_progress(label, frac)

            kwargs["hook"] = _hook
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        output = self._pipeline(str(audio_path), **kwargs)
        # pyannote 4.x restituisce DiarizeOutput con .speaker_diarization (Annotation);
        # versioni precedenti restituiscono direttamente un'Annotation.
        annotation = getattr(output, "speaker_diarization", output)
        turns = _to_turns(annotation)
        exclusive = _to_turns(getattr(output, "exclusive_speaker_diarization", None))
        return DiarizationResult(
            turns=turns,
            # Senza output esclusivo (modelli piu' vecchi) si prosegue con
            # l'ordinario: peggio nelle sovrapposizioni, ma non si rompe nulla.
            exclusive=exclusive or turns,
            overlaps=_overlapping_spans(turns),
            speaker_embeddings=_speaker_embedding_map(
                annotation, getattr(output, "speaker_embeddings", None)
            ),
            model_repo=self.MODEL,
            model_revision=self.model_revision,
        )

    def close(self) -> None:
        """Rilascia la pipeline pyannote e svuota la cache VRAM."""
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    unload = close
