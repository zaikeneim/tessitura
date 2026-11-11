"""Progetti: piu' registrazioni che appartengono alla stessa storia.

Una campagna di gioco, le riunioni di un cliente, un corso: sono decine di
sedute con le stesse persone, gli stessi nomi propri e lo stesso gergo. Finora
ogni registrazione ripartiva da zero — stessi nomi sbagliati ogni volta, stessa
fatica per rinominare le voci, nessuna memoria di cosa era stato deciso la volta
prima.

Un progetto e' una cartella con dentro gli audio e un file
``tessitura-progetto.json``. Tenere lo stato dentro la cartella, e non in un
registro centrale, significa che spostare o copiare la cartella porta con se'
tutto: glossario, nomi delle voci e memoria delle sedute precedenti.

La memoria e' breve, leggibile e cancellabile: l'utente puo' capire cosa verra'
passato al modello locale. I pesi dei modelli restano invariati, mentre termini
corretti, preferenze esplicite e profili vocali confermati persistono nel
progetto e migliorano le elaborazioni successive.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_FILE = "tessitura-progetto.json"

# Estensioni che la pipeline sa leggere (ffmpeg le apre tutte).
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma")

# Quante sedute passate entrano nel contesto passato al modello. Oltre questa
# soglia il contesto crescerebbe senza portare informazione: le sedute vecchie
# restano nel file, ma non vengono piu' rilette a ogni domanda.
MEMORY_SESSIONS = 8
LEARNING_SESSION_LIMIT = 100
LEARNED_TERM_LIMIT = 64
ANALYSIS_PREFERENCE_LIMIT = 24
ANALYSIS_PREFERENCE_CHARS = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonnegative_int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _embedding_model(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {"repo": "", "revision": ""}
    return {
        "repo": str(value.get("repo", "") or "")[:240],
        "revision": str(value.get("revision", "") or "")[:160],
    }


def _sanitize_learning_sessions(value) -> dict[str, dict]:
    """Carica memoria adattiva senza accettare strutture o vettori illimitati."""
    if not isinstance(value, dict):
        return {}
    from ..processing.adaptation import normalize_embedding

    result: dict[str, dict] = {}
    for raw_id, raw in list(value.items())[-LEARNING_SESSION_LIMIT:]:
        if not isinstance(raw, dict):
            continue
        session_id = str(raw_id).strip()
        if not session_id:
            continue
        corrections: list[dict[str, str]] = []
        raw_corrections = raw.get("transcription_corrections", [])
        if not isinstance(raw_corrections, list):
            raw_corrections = []
        for item in raw_corrections[:LEARNED_TERM_LIMIT]:
            if not isinstance(item, dict):
                continue
            before = " ".join(str(item.get("before", "")).split())[:100]
            after = " ".join(str(item.get("after", "")).split())[:100]
            if after:
                corrections.append({"before": before, "after": after})
        embeddings: dict[str, list[float]] = {}
        raw_embeddings = raw.get("speaker_embeddings", {})
        if isinstance(raw_embeddings, dict):
            for raw_name, raw_vector in list(raw_embeddings.items())[:32]:
                name = " ".join(str(raw_name).split())[:120]
                vector = normalize_embedding(raw_vector)
                if name and vector is not None:
                    embeddings[name] = [round(item, 8) for item in vector]
        if corrections or embeddings:
            result[session_id] = {
                "transcription_corrections": corrections,
                "speaker_embeddings": embeddings,
                "embedding_model": _embedding_model(raw.get("embedding_model")),
                "updated_at": str(raw.get("updated_at", "") or ""),
            }
    return result


@dataclass
class ProjectSession:
    """Una seduta gia' elaborata dentro il progetto."""

    audio: str                 # percorso relativo alla cartella del progetto
    session_file: str | None = None   # JSON della sessione, relativo alla cartella
    name: str = ""
    date: str = field(default_factory=_now)
    duration_s: float = 0.0
    speakers: list[str] = field(default_factory=list)
    summary: str = ""          # poche righe: e' la memoria della seduta


@dataclass
class Project:
    """Un progetto e il suo stato sul disco."""

    name: str
    folder: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    created_at: str = field(default_factory=_now)
    # Nomi propri e termini ricorrenti: finiscono nel prompt di Whisper, che su
    # nomi inventati o gergo sbaglia sistematicamente senza un suggerimento.
    glossary: list[str] = field(default_factory=list)
    # Voci ricorrenti: "Marco", "Kelial". Servono a proporre gli stessi nomi
    # seduta dopo seduta invece di ripartire da SPEAKER_00.
    speakers: list[str] = field(default_factory=list)
    sessions: list[ProjectSession] = field(default_factory=list)
    # Note dell'utente sul progetto, passate al modello insieme ai riassunti.
    notes: str = ""
    # Memoria adattiva locale. Ogni contributo e' sostituibile usando l'id della
    # sessione: dopo Annulla non resta in circolo una correzione revocata.
    learning_enabled: bool = True
    learning_sessions: dict[str, dict] = field(default_factory=dict)
    analysis_preferences: list[str] = field(default_factory=list)
    learning_revision: int = 0

    # --- percorsi ------------------------------------------------------------

    @property
    def path(self) -> Path:
        return Path(self.folder)

    @property
    def file(self) -> Path:
        return self.path / PROJECT_FILE

    def sessions_dir(self) -> Path:
        directory = self.path / "sessioni"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def relative(self, path: str | Path) -> str:
        """Percorso da salvare: relativo alla cartella, cosi' resta portabile."""
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.path.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()

    def resolve(self, stored: str | None) -> Path | None:
        if not stored:
            return None
        candidate = Path(stored.replace("\\", "/"))
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        full = self.path / candidate
        return full if full.exists() else None

    # --- audio del progetto --------------------------------------------------

    def audio_files(self) -> list[Path]:
        """Tutti gli audio nella cartella, ordinati per nome.

        L'ordine per nome e' prevedibile e di solito cronologico (i registratori
        datano i file); l'ordine di sistema non lo sarebbe.
        """
        if not self.path.is_dir():
            return []
        found = [
            item for item in self.path.rglob("*")
            if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
        ]
        return sorted(found, key=lambda item: item.name.lower())

    def processed_audio(self) -> set[str]:
        return {session.audio for session in self.sessions}

    def pending_audio(self) -> list[Path]:
        """Audio della cartella non ancora elaborati in questo progetto."""
        done = self.processed_audio()
        return [item for item in self.audio_files() if self.relative(item) not in done]

    # --- memoria per il modello locale ---------------------------------------

    def glossary_prompt(self) -> str:
        """Suggerimento per Whisper: nomi propri e termini del progetto."""
        terms = [
            term for term in (*self.glossary, *self.speakers, *self.learned_terms())
            if term.strip()
        ]
        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            cleaned = term.strip()
            key = cleaned.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(cleaned)
        return ", ".join(unique)

    def memory_text(self, limit: int = MEMORY_SESSIONS) -> str:
        """Contesto testuale da passare al modello locale.

        Volutamente breve: descrizione, note, voci ricorrenti, glossario e i
        riassunti delle ultime sedute. Un contesto che cresce senza limite
        mangerebbe il budget di token che serve alla domanda vera.
        """
        blocks: list[str] = [f"Progetto: {self.name}"]
        if self.description.strip():
            blocks.append(f"Descrizione: {self.description.strip()}")
        if self.notes.strip():
            blocks.append(f"Note: {self.notes.strip()}")
        if self.speakers:
            blocks.append("Voci ricorrenti: " + ", ".join(self.speakers))
        if self.glossary:
            blocks.append("Termini e nomi propri: " + ", ".join(self.glossary))
        learned = self.learned_terms()
        if learned:
            blocks.append("Correzioni di trascrizione confermate: " + ", ".join(learned))
        if self.learning_enabled and self.analysis_preferences:
            lines = [f"- {item}" for item in self.analysis_preferences]
            blocks.append("Preferenze di analisi confermate:\n" + "\n".join(lines))
        recent = [item for item in self.sessions if item.summary.strip()][-limit:]
        if recent:
            lines = [f"- {item.name or item.audio}: {item.summary.strip()}" for item in recent]
            blocks.append("Sedute precedenti:\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    # --- aggiornamento -------------------------------------------------------

    def remember_session(self, audio: str | Path, session, *, session_file=None,
                         summary: str = "") -> ProjectSession:
        """Registra (o aggiorna) una seduta elaborata e ne assorbe le voci.

        I nomi assegnati dall'utente diventano voci ricorrenti del progetto:
        e' il dato che alla seduta successiva evita di ripartire da SPEAKER_00.
        """
        relative = self.relative(audio)
        names = [name for name in getattr(session, "speaker_names", {}).values() if name.strip()]
        entry = ProjectSession(
            audio=relative,
            session_file=self.relative(session_file) if session_file else None,
            name=getattr(session, "name", "") or Path(relative).stem,
            duration_s=float(getattr(session, "duration", 0.0) or 0.0),
            speakers=sorted(set(names)),
            summary=summary.strip(),
        )
        existing = next((item for item in self.sessions if item.audio == relative), None)
        if existing is not None:
            entry.date = existing.date
            entry.summary = entry.summary or existing.summary
            self.sessions[self.sessions.index(existing)] = entry
        else:
            self.sessions.append(entry)
        self.add_speakers(names)
        self.learn_from_session(session)
        return entry

    def add_speakers(self, names) -> None:
        for name in names:
            cleaned = " ".join(str(name).split())
            if cleaned and cleaned not in self.speakers:
                self.speakers.append(cleaned)

    def set_glossary(self, terms: str | list[str]) -> None:
        if isinstance(terms, str):
            terms = re.split(r"[,;\n]", terms)
        self.glossary = [
            " ".join(term.split()) for term in terms if str(term).strip()
        ]

    def learned_terms(self) -> list[str]:
        """Termini corretti più recenti, deduplicati e limitati."""
        if not self.learning_enabled:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for contribution in reversed(list(self.learning_sessions.values())):
            for correction in reversed(contribution.get("transcription_corrections", [])):
                term = " ".join(str(correction.get("after", "")).split())
                key = term.casefold()
                if term and key not in seen:
                    seen.add(key)
                    result.append(term)
                    if len(result) >= LEARNED_TERM_LIMIT:
                        return result
        return result

    def speaker_profiles(self, embedding_model=None) -> dict[str, list[float]]:
        """Centroidi medi per le identità vocali confermate dall'utente."""
        if not self.learning_enabled:
            return {}
        from ..processing.adaptation import mean_embedding

        expected_model = _embedding_model(embedding_model)
        samples: dict[str, list[list[float]]] = {}
        for contribution in self.learning_sessions.values():
            if _embedding_model(contribution.get("embedding_model")) != expected_model:
                continue
            for name, vector in contribution.get("speaker_embeddings", {}).items():
                samples.setdefault(str(name), []).append(vector)
        return {
            name: profile
            for name, values in samples.items()
            if (profile := mean_embedding(values)) is not None
        }

    def suggested_name_matches(self, session) -> dict[str, dict]:
        """Nomi proposti con la relativa evidenza di similarità vocale."""
        from ..processing.adaptation import match_speaker_profiles

        diarization = getattr(session, "diarization", {}) or {}
        if not isinstance(diarization, dict):
            return {}
        embeddings = diarization.get("speaker_embeddings", {})
        if not isinstance(embeddings, dict):
            return {}
        profiles = self.speaker_profiles(diarization.get("embedding_model"))
        if profiles:
            return match_speaker_profiles(embeddings, profiles)
        has_profiles = any(
            contribution.get("speaker_embeddings")
            for contribution in self.learning_sessions.values()
        )
        if has_profiles:
            return {}

        # Compatibilità per progetti creati prima dei profili vocali. Questo
        # fallback viene usato solo finché non esiste alcun profilo confermato.
        ids = session.speaker_ids()
        if not ids or len(ids) != len(self.speakers):
            return {}
        return {
            speaker_id: {"name": name, "similarity": None, "source": "ordine"}
            for speaker_id, name in zip(ids, self.speakers)
        }

    def suggested_names(self, session) -> dict[str, str]:
        """Nomi del progetto proposti per le voci di una nuova seduta.

        I progetti nuovi usano profili vocali compatibili col modello corrente;
        quelli senza profili conservano il precedente fallback per ordine e
        soltanto quando il numero di partecipanti coincide.
        """
        return {
            speaker_id: str(match["name"])
            for speaker_id, match in self.suggested_name_matches(session).items()
        }

    def learn_from_session(self, session) -> dict[str, int | bool]:
        """Sostituisce il contributo adattivo della sessione corrente.

        Il testo entra in memoria soltanto dopo una modifica manuale. Un profilo
        vocale entra soltanto per etichette che la GUI ha marcato come confermate.
        """
        report: dict[str, int | bool] = {
            "changed": False, "corrections": 0, "speaker_profiles": 0,
        }
        if not self.learning_enabled:
            return report

        from ..processing.adaptation import normalize_embedding, transcription_corrections

        raw_segments = sorted(
            list(getattr(session, "raw_segments", None) or []),
            key=lambda item: (item.start, item.end),
        )
        corrections: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        raw_cursor = 0
        segments = sorted(
            list(getattr(session, "segments", []) or []),
            key=lambda item: (item.start, item.end),
        )
        review = getattr(session, "transcript_review", None) or {}
        refinement = review.get("llm_text_refinement", {})
        automatic_text = refinement.get("automatic_text", {}) if isinstance(refinement, dict) else {}
        automatic_spans = refinement.get("automatic_spans", {}) if isinstance(refinement, dict) else {}
        automatic_text = {k: v for k, v in automatic_text.items() if isinstance(v, str)} if isinstance(automatic_text, dict) else {}
        automatic_spans = automatic_spans if isinstance(automatic_spans, dict) else {}
        for segment in segments:
            if getattr(segment, "words_valid", True):
                continue
            while raw_cursor < len(raw_segments) and raw_segments[raw_cursor].end <= segment.start:
                raw_cursor += 1
            overlapping = []
            index = raw_cursor
            while index < len(raw_segments) and raw_segments[index].start < segment.end:
                raw = raw_segments[index]
                if raw.end > segment.start:
                    overlapping.append(raw)
                index += 1
            before = " ".join(raw.text.strip() for raw in overlapping if raw.text.strip())
            # Machine-generated text is not human feedback. For a later manual
            # correction, learn only the delta from the automatic baseline.
            if segment.id in automatic_text:
                if automatic_spans.get(segment.id) != [segment.start, segment.end]:
                    continue
                before = automatic_text[segment.id]
            elif "testo_revisionato_llm" in getattr(segment, "warnings", []):
                # Grouped/split segments may no longer have the original ID.
                continue
            for correction in transcription_corrections(before, segment.text):
                key = (correction["before"].casefold(), correction["after"].casefold())
                if key not in seen:
                    seen.add(key)
                    corrections.append(correction)

        provenance = getattr(session, "provenance", {}) or {}
        if not isinstance(provenance, dict):
            provenance = {}
        raw_confirmed = provenance.get("confirmed_speakers", [])
        confirmed = {
            str(value) for value in raw_confirmed
            if isinstance(raw_confirmed, list) and str(value).strip()
        }
        diarization = getattr(session, "diarization", {}) or {}
        if not isinstance(diarization, dict):
            diarization = {}
        raw_embeddings = diarization.get("speaker_embeddings", {})
        if not isinstance(raw_embeddings, dict):
            raw_embeddings = {}
        embedding_model = _embedding_model(
            diarization.get("embedding_model")
        )
        speaker_embeddings: dict[str, list[float]] = {}
        for speaker_id in sorted(confirmed):
            name = " ".join(str(getattr(session, "speaker_names", {}).get(speaker_id, "")).split())
            vector = normalize_embedding(raw_embeddings.get(speaker_id))
            if name and vector is not None:
                speaker_embeddings[name] = [round(item, 8) for item in vector]
        if not speaker_embeddings:
            embedding_model = _embedding_model(None)

        semantic = {
            "transcription_corrections": corrections[:LEARNED_TERM_LIMIT],
            "speaker_embeddings": speaker_embeddings,
            "embedding_model": embedding_model,
        }
        session_id = str(getattr(session, "id", "") or "").strip()
        if not session_id:
            return report
        old = self.learning_sessions.get(session_id)
        old_semantic = {
            "transcription_corrections": (old or {}).get("transcription_corrections", []),
            "speaker_embeddings": (old or {}).get("speaker_embeddings", {}),
            "embedding_model": _embedding_model((old or {}).get("embedding_model")),
        }
        if semantic == old_semantic:
            report["corrections"] = len(corrections)
            report["speaker_profiles"] = len(speaker_embeddings)
            return report

        if corrections or speaker_embeddings:
            self.learning_sessions[session_id] = {**semantic, "updated_at": _now()}
            while len(self.learning_sessions) > LEARNING_SESSION_LIMIT:
                oldest = min(
                    self.learning_sessions,
                    key=lambda key: str(self.learning_sessions[key].get("updated_at", "")),
                )
                self.learning_sessions.pop(oldest, None)
        else:
            self.learning_sessions.pop(session_id, None)
        self.learning_revision += 1
        report.update(
            changed=True,
            corrections=len(corrections),
            speaker_profiles=len(speaker_embeddings),
        )
        return report

    def remember_analysis_feedback(self, text: str) -> bool:
        """Memorizza solo un'istruzione che l'utente ha scelto esplicitamente."""
        if not self.learning_enabled:
            return False
        cleaned = " ".join(str(text or "").split())[:ANALYSIS_PREFERENCE_CHARS]
        if not cleaned:
            return False
        key = cleaned.casefold()
        previous = [item for item in self.analysis_preferences if item.casefold() != key]
        updated = [*previous, cleaned][-ANALYSIS_PREFERENCE_LIMIT:]
        if updated == self.analysis_preferences:
            return False
        self.analysis_preferences = updated
        self.learning_revision += 1
        return True

    def clear_learning(self) -> None:
        if self.learning_sessions or self.analysis_preferences:
            self.learning_sessions.clear()
            self.analysis_preferences.clear()
            self.learning_revision += 1

    def learning_summary(self) -> dict[str, int]:
        speaker_names = {
            str(name)
            for item in self.learning_sessions.values()
            for name in item.get("speaker_embeddings", {})
        }
        return {
            "sessions": len(self.learning_sessions),
            "corrections": sum(
                len(item.get("transcription_corrections", []))
                for item in self.learning_sessions.values()
            ),
            "speakers": len(speaker_names),
            "preferences": len(self.analysis_preferences),
            "revision": self.learning_revision,
        }

    # --- persistenza ---------------------------------------------------------

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("folder", None)  # la cartella e' dove sta il file, non un dato
        return data

    @classmethod
    def from_dict(cls, data: dict, folder: str | Path) -> "Project":
        if not isinstance(data, dict):
            raise ValueError("Progetto non valido")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Il progetto deve avere un nome")
        sessions = []
        for item in data.get("sessions", []) or []:
            if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
                raise ValueError("Seduta del progetto non valida")
            sessions.append(ProjectSession(
                audio=item["audio"],
                session_file=item.get("session_file"),
                name=item.get("name", ""),
                date=item.get("date") or _now(),
                duration_s=float(item.get("duration_s", 0.0) or 0.0),
                speakers=[str(value) for value in item.get("speakers", []) or []],
                summary=str(item.get("summary", "") or ""),
            ))
        learning_sessions = _sanitize_learning_sessions(data.get("learning_sessions", {}))
        preferences: list[str] = []
        raw_preferences = data.get("analysis_preferences", [])
        if not isinstance(raw_preferences, list):
            raw_preferences = []
        for value in raw_preferences[-ANALYSIS_PREFERENCE_LIMIT:]:
            cleaned = " ".join(str(value).split())[:ANALYSIS_PREFERENCE_CHARS]
            if cleaned and cleaned.casefold() not in {item.casefold() for item in preferences}:
                preferences.append(cleaned)
        return cls(
            name=name.strip(),
            folder=str(folder),
            id=data.get("id") or uuid.uuid4().hex[:12],
            description=str(data.get("description", "") or ""),
            created_at=data.get("created_at") or _now(),
            glossary=[str(value) for value in data.get("glossary", []) or []],
            speakers=[str(value) for value in data.get("speakers", []) or []],
            sessions=sessions,
            notes=str(data.get("notes", "") or ""),
            learning_enabled=(
                data.get("learning_enabled")
                if isinstance(data.get("learning_enabled"), bool) else True
            ),
            learning_sessions=learning_sessions,
            analysis_preferences=preferences[-ANALYSIS_PREFERENCE_LIMIT:],
            learning_revision=_nonnegative_int(data.get("learning_revision", 0)),
        )

    def save(self) -> Path:
        """Scrittura atomica: un salvataggio interrotto non perde il progetto."""
        self.path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=self.path, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.file


def load_project(folder: str | Path) -> Project:
    """Apre il progetto di una cartella."""
    directory = Path(folder)
    target = directory / PROJECT_FILE
    if not target.is_file():
        raise FileNotFoundError(f"Nessun progetto in {directory}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return Project.from_dict(data, directory)


def open_or_create(folder: str | Path, name: str | None = None) -> Project:
    """Apre il progetto della cartella, creandolo se non c'e' ancora."""
    directory = Path(folder)
    if (directory / PROJECT_FILE).is_file():
        return load_project(directory)
    project = Project(name=(name or directory.name).strip() or directory.name,
                      folder=str(directory))
    project.save()
    return project


def is_project(folder: str | Path) -> bool:
    return (Path(folder) / PROJECT_FILE).is_file()
