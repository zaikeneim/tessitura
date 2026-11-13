"""Mediatore unificato per le viste di trascrizione (Conversazione, Timeline, Tabellare).

Gestisce lo stato della sessione, le modifiche atomiche per utterance_id,
lo switch tra viste senza riesecuzione ASR, l'undo/redo e l'evidenziazione
simultanea multi-speaker durante il playback.
"""

from __future__ import annotations

from typing import Any
from PySide6.QtCore import QObject, Signal

from ..models.session import Session
from ..models.transcript_artifacts import Utterance


class TranscriptController(QObject):
    """Controller centrale tra Session e le viste di visualizzazione/editing."""

    sessionChanged = Signal(object)           # Session
    utteranceEdited = Signal(str, str)        # utterance_id, nuovo testo
    speakerChanged = Signal(str, str)         # utterance_id, nuovo speaker_id
    speakerNamesChanged = Signal(dict)        # mappa id -> nome leggibile
    activeViewChanged = Signal(str)           # "conversation" | "timeline" | "table"
    selectionChanged = Signal(str)            # utterance_id selezionato
    activeHighlightsChanged = Signal(list)    # list[str] di utterance_id attivi al playback
    seekRequested = Signal(float)             # secondi per AudioPlayer
    readOnlyChanged = Signal(bool)
    changed = Signal()                       # qualsiasi modifica persistente

    SUPPORTED_VIEWS = ("conversation", "timeline", "table")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session: Session | None = None
        self._active_view: str = "timeline"
        self._selected_id: str | None = None
        self._active_highlight_ids: set[str] = set()
        self._read_only: bool = False

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def active_view(self) -> str:
        return self._active_view

    @property
    def selected_id(self) -> str | None:
        return self._selected_id

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def set_read_only(self, read_only: bool) -> None:
        if self._read_only != read_only:
            self._read_only = bool(read_only)
            self.readOnlyChanged.emit(self._read_only)

    def load_session(self, session: Session | None) -> None:
        """Carica il modello canonico e sincronizza tutte le viste senza inferenza."""
        self._session = session
        self._selected_id = None
        self._active_highlight_ids.clear()
        if session is not None:
            session.migrate_to_v6()
            view = session.active_view if session.active_view in self.SUPPORTED_VIEWS else "timeline"
            self._active_view = view
        else:
            self._active_view = "timeline"

        self.sessionChanged.emit(self._session)
        self.activeViewChanged.emit(self._active_view)
        if self._session:
            self.speakerNamesChanged.emit(self._session.speaker_names)

    def set_active_view(self, view_name: str) -> None:
        """Commuta la vista attiva salvandola nella sessione senza ricalcoli."""
        if view_name not in self.SUPPORTED_VIEWS:
            raise ValueError(f"Vista non supportata: {view_name}. Supportate: {self.SUPPORTED_VIEWS}")
        if self._active_view != view_name:
            self._active_view = view_name
            if self._session is not None and not self._read_only:
                self._session.active_view = view_name
                self.changed.emit()
            self.activeViewChanged.emit(view_name)

    def _notify_changed(self) -> None:
        """Aggiorna le proiezioni dopo una modifica del modello canonico."""
        if self._session is None:
            return
        if not any(u.utterance_id == self._selected_id for u in self._session.utterances):
            self._selected_id = None
        self._active_highlight_ids.clear()
        self.sessionChanged.emit(self._session)
        self.speakerNamesChanged.emit(self._session.speaker_names)
        if self._selected_id is not None:
            self.selectionChanged.emit(self._selected_id)
        self.changed.emit()

    def edit_event(self, utterance_id: str, **changes: Any) -> bool:
        """Corregge un evento per ID, preservando predizioni e cronologia."""
        if self._read_only or self._session is None:
            return False
        from ..models.timeline import TimelineEditor

        # Ricerca dell'Utterance
        target = next((u for u in self._session.utterances if u.utterance_id == utterance_id), None)
        if target is None:
            return False
        old_text, old_speaker = target.text, target.speaker
        # invalidamento allineamento per-parola su testo modificato
        # Sincronizzazione atomica sul Segment corrispondente per ID esatto
        result = TimelineEditor(self._session).edit_event(utterance_id, **changes)
        if result:
            self._notify_changed()
            if target.text != old_text:
                self.utteranceEdited.emit(utterance_id, target.text)
            if target.speaker != old_speaker:
                self.speakerChanged.emit(utterance_id, target.speaker or "")
        return bool(result)

    def edit_text(self, utterance_id: str, new_text: str) -> bool:
        """Modifica atomicamente il testo di un intervento per ID."""
        return self.edit_event(utterance_id, text=new_text)

    def reassign_speaker(self, utterance_id: str, new_speaker_id: str | None) -> bool:
        """Riassegna la voce attribuita a un intervento per ID."""
        # Sincronizzazione atomica sul Segment corrispondente per ID esatto
        return self.edit_event(utterance_id, speaker_id=new_speaker_id)

    def create_event(self, start_s: float, end_s: float, text: str = "",
                     speaker_id: str | None = None) -> Utterance | None:
        """Aggiunge parlato mancante senza modificare gli altri eventi."""
        if self._read_only or self._session is None:
            return None
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).create_event(start_s, end_s, text=text, speaker_id=speaker_id)
        self._selected_id = result.utterance_id
        self._notify_changed()
        return result

    def split_event(self, utterance_id: str, seconds: float,
                    left_text: str | None = None, right_text: str | None = None) -> bool:
        """Divide un evento conservando l'origine delle due parti."""
        if self._read_only or self._session is None:
            return False
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).split_event(
            utterance_id, seconds, left_text=left_text, right_text=right_text)
        if result:
            self._notify_changed()
        return bool(result)

    def merge_events(self, left_id: str, right_id: str) -> bool:
        """Unisce soltanto eventi adiacenti compatibili."""
        if self._read_only or self._session is None:
            return False
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).merge_events(left_id, right_id)
        if result:
            self._notify_changed()
        return bool(result)

    def create_speaker(self, name: str) -> str | None:
        """Crea un'identita' utilizzabile nelle corsie e nelle altre viste."""
        if self._read_only or self._session is None:
            return None
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).create_speaker(name)
        self._notify_changed()
        return result

    def confirm_identity(self, utterance_id: str, speaker_id: str) -> bool:
        """Conferma esplicitamente l'identita' di una voce proposta."""
        if self._read_only or self._session is None:
            return False
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).confirm_identity(utterance_id, speaker_id)
        if result:
            self._notify_changed()
            self.speakerChanged.emit(utterance_id, speaker_id)
        return bool(result)

    def accept_candidate(self, candidate_id: str, speaker_id: str | None = None) -> Utterance | None:
        """Accetta una proposta esplicita conservando gli eventi gia' verificati."""
        if self._read_only or self._session is None:
            return None
        from ..models.timeline import TimelineEditor

        result = TimelineEditor(self._session).accept_candidate(candidate_id, speaker_id=speaker_id)
        self._selected_id = result.utterance_id
        self._notify_changed()
        return result

    def rename_speaker(self, speaker_id: str, display_name: str) -> bool:
        """Rinomina uno speaker nella mappa globale delle voci."""
        if self._read_only or self._session is None:
            return False
        if self._session.speaker_names.get(speaker_id, "") == display_name.strip():
            return False
        self._session.rename_speaker(speaker_id, display_name)
        self._notify_changed()
        return True

    def reject_candidate(self, candidate_id: str) -> bool:
        """Reject a proposal without changing any canonical event."""
        if self._read_only or self._session is None:
            return False
        from ..models.timeline import TimelineEditor
        result = TimelineEditor(self._session).reject_candidate(candidate_id)
        if result:
            self._notify_changed()
        return result

    def undo(self) -> bool:
        """Annulla l'ultima modifica atomica."""
        if self._read_only or self._session is None:
            return False
        res = self._session.undo()
        if res:
            self._notify_changed()
        return res

    def redo(self) -> bool:
        """Ripristina l'ultima modifica annullata."""
        if self._read_only or self._session is None:
            return False
        res = self._session.redo()
        if res:
            self._notify_changed()
        return res

    def select_utterance(self, utterance_id: str | None) -> None:
        """Imposta l'intervento correntemente selezionato dall'utente."""
        if self._selected_id != utterance_id:
            self._selected_id = utterance_id
            if utterance_id is not None:
                self.selectionChanged.emit(utterance_id)

    def request_seek(self, seconds: float) -> None:
        """Inoltra una richiesta di seek al player audio."""
        self.seekRequested.emit(max(0.0, float(seconds)))

    def update_playback_position(self, seconds: float) -> list[str]:
        """Evidenzia tutti gli interventi contemporaneamente attivi alla posizione di playback.

        Risolve il limite di evidenziazione del solo primo elemento presente nella tabella legacy.
        """
        if self._session is None:
            return []

        active: set[str] = set()
        for u in self._session.utterances:
            if u.start_s <= seconds <= u.end_s:
                active.add(u.utterance_id)

        if active != self._active_highlight_ids:
            self._active_highlight_ids = active
            active_list = list(active)
            self.activeHighlightsChanged.emit(active_list)
            return active_list

        return list(self._active_highlight_ids)
