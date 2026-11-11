"""Operazioni sui progetti usate dalla finestra principale.

Stanno qui e non dentro MainWindow perche' sono logica verificabile senza aprire
una finestra: cosa salvare, dove, con quale nome e cosa ricordare del progetto.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..models import storage
from ..models.project import Project

# Lunghezza massima del riassunto tenuto in memoria per una seduta. La memoria
# del progetto viene riletta a ogni richiesta: sedute intere renderebbero il
# contesto piu' lungo della domanda.
SUMMARY_CHARS = 600


def session_filename(audio: Path, session) -> str:
    """Nome stabile del JSON: parte dal nome dell'audio, non da un id casuale."""
    stem = Path(audio).stem or getattr(session, "id", "sessione")
    return f"{stem}.json"


def store_session(project: Project, audio, session, summary: str = "") -> Path:
    """Salva la sessione dentro il progetto e ne aggiorna la memoria.

    Il JSON finisce in ``<progetto>/sessioni``: la cartella resta autosufficiente
    e spostarla porta con se' audio, trascrizioni e memoria.
    """
    destination = project.sessions_dir() / session_filename(audio, session)
    storage.save_session(session, destination)
    project.remember_session(audio, session, session_file=destination, summary=summary)
    project.save()
    return destination


def summary_for_memory(session) -> str:
    """Poche righe da ricordare della seduta, dal resoconto se c'e'.

    Senza analisi LLM si ripiega sulle prime battute: meglio un promemoria
    grezzo che una seduta senza traccia nella memoria del progetto.
    """
    analysis = getattr(session, "analysis", None)
    if isinstance(analysis, dict):
        text = str(analysis.get("summary", "") or "").strip()
        if text:
            return text[:SUMMARY_CHARS]
        points = [str(item) for item in (analysis.get("key_points") or []) if str(item).strip()]
        if points:
            return "; ".join(points)[:SUMMARY_CHARS]
    opening = " ".join(segment.text.strip() for segment in getattr(session, "segments", [])[:5])
    return opening.strip()[:SUMMARY_CHARS]


def apply_project_names(project: Project, session) -> dict[str, str]:
    """Propone i nomi del progetto alle voci della nuova seduta.

    Se esistono profili vocali usa solo corrispondenze sopra la soglia
    conservativa. Per i vecchi progetti resta il fallback per ordine, soltanto
    quando il numero di voci coincide. Ritorna i nomi effettivamente applicati.
    """
    matches = project.suggested_name_matches(session)
    proposed = {speaker_id: str(match["name"]) for speaker_id, match in matches.items()}
    for speaker_id, name in proposed.items():
        session.rename_speaker(speaker_id, name)
    if matches:
        provenance = dict(getattr(session, "provenance", {}) or {})
        provenance["speaker_name_suggestions"] = matches
        session.provenance = provenance
    return proposed


def confirm_speakers(session, speaker_ids) -> None:
    """Marca identità vocali confermate dall'utente, mai quelle solo proposte."""
    valid = {
        str(speaker_id) for speaker_id in speaker_ids
        if str(getattr(session, "speaker_names", {}).get(str(speaker_id), "")).strip()
    }
    provenance = dict(getattr(session, "provenance", {}) or {})
    previous = provenance.get("confirmed_speakers", [])
    confirmed = set(previous) if isinstance(previous, list) else set()
    confirmed.update(valid)
    provenance["confirmed_speakers"] = sorted(confirmed)
    session.provenance = provenance


def unconfirm_speaker(session, speaker_id: str) -> None:
    provenance = dict(getattr(session, "provenance", {}) or {})
    previous = provenance.get("confirmed_speakers", [])
    confirmed = set(previous) if isinstance(previous, list) else set()
    confirmed.discard(str(speaker_id))
    provenance["confirmed_speakers"] = sorted(confirmed)
    session.provenance = provenance


def sync_project_learning(project: Project | None, session) -> dict[str, int | bool]:
    """Aggiorna la memoria e rende il salvataggio sessione un punto di retry."""
    if project is None:
        return {"changed": False, "corrections": 0, "speaker_profiles": 0}
    previous_sessions = deepcopy(project.learning_sessions)
    previous_revision = project.learning_revision
    report = project.learn_from_session(session)
    # Il salvataggio intenzionale della sessione è anche un punto di retry: se
    # una precedente scrittura del progetto era fallita dopo la mutazione in
    # memoria, il confronto semantico restituisce changed=False ma il file deve
    # comunque essere riscritto.
    if project.learning_enabled:
        try:
            project.save()
        except Exception:
            project.learning_sessions = previous_sessions
            project.learning_revision = previous_revision
            raise
    return report


def tag_session(project: Project, session) -> None:
    """Annota nella sessione a quale progetto appartiene."""
    provenance = dict(getattr(session, "provenance", {}) or {})
    provenance["project"] = {"id": project.id, "name": project.name,
                             "folder": str(project.path)}
    session.provenance = provenance


def combined_glossary(project: Project | None, typed: str) -> str | None:
    """Glossario passato a Whisper: quello del progetto piu' quello digitato.

    I nomi propri di una campagna o di un cliente sono esattamente le parole che
    Whisper sbaglia in modo sistematico; ripeterli a mano a ogni seduta era il
    lavoro che il progetto deve togliere.
    """
    parts = []
    if project is not None:
        from_project = project.glossary_prompt()
        if from_project:
            parts.append(from_project)
    if typed.strip():
        parts.append(typed.strip())
    if not parts:
        return None
    seen: list[str] = []
    for term in ", ".join(parts).split(","):
        cleaned = " ".join(term.split())
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return ", ".join(seen)
