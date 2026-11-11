"""Vista Conversazione a fumetti/card con supporto overlap e sincronizzazione audio (Passo 5).

Presenta gli interventi della sessione (`Session.utterances`) in un layout
conversazionale cronologico continuo con:
- Evidenziazione degli overlap vocali (`[In contemporanea]`).
- Modifica in-place del testo per Utterance ID tramite `TranscriptController`.
- Seeking temporale al clic sui timestamp.
- Evidenziazione multi-intervento sincronizzata durante la riproduzione.
- Autoscroll controllato con rilevamento dello scroll manuale.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..models.session import Session
from ..models.transcript_artifacts import Utterance
from .transcript_view import SPEAKER_COLORS, _fmt

if TYPE_CHECKING:
    from .transcript_controller import TranscriptController


def _speaker_color(speaker: str, display_name: str) -> str:
    speaker = speaker or "UNASSIGNED"
    if "MIC" in speaker.upper():
        return "#5ee0ca"  # Teal / Emerald
    if "SYS" in speaker.upper():
        return "#4F8DFD"  # Blue
    idx = abs(hash(speaker)) % len(SPEAKER_COLORS)
    return SPEAKER_COLORS[idx]


class ConversationBubbleWidget(QFrame):
    """Singolo fumetto / card per un Utterance."""

    seekRequested = Signal(float)
    textEdited = Signal(str, str)  # utterance_id, new_text

    def __init__(
        self,
        utterance: Utterance,
        display_name: str,
        color_hex: str,
        parent=None,
    ):
        super().__init__(parent)
        self.utterance = utterance
        self.utterance_id = utterance.utterance_id
        self.display_name = display_name
        self.color_hex = color_hex
        self._is_active = False
        self._is_editing = False

        self.setObjectName("conversationBubble")
        self._update_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header: Timestamp, Speaker, Badges, Modifica
        header = QHBoxLayout()
        header.setSpacing(8)

        # Timestamp button
        self.time_btn = QPushButton(_fmt(utterance.start_s))
        self.time_btn.setCursor(Qt.PointingHandCursor)
        self.time_btn.setStyleSheet("""
            QPushButton {
                background: #233247;
                color: #b7c6d9;
                font-family: Consolas, monospace;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 7px;
                border: 1px solid #34465d;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #2c4058;
                color: #5ee0ca;
                border-color: #5ee0ca;
            }
        """)
        self.time_btn.clicked.connect(lambda: self.seekRequested.emit(self.utterance.start_s))
        header.addWidget(self.time_btn)

        # Speaker label
        self.speaker_label = QLabel(display_name)
        self.speaker_label.setStyleSheet(
            f"font-weight: 700; font-size: 13px; color: {color_hex};"
        )
        header.addWidget(self.speaker_label)

        # Badges
        if utterance.is_overlap:
            overlap_badge = QLabel("In contemporanea")
            overlap_badge.setStyleSheet("""
                background: #382c16;
                color: #f0883e;
                font-size: 10px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 4px;
                border: 1px solid #6b4d1b;
            """)
            overlap_badge.setToolTip("Questo intervento si sovrappone temporalmente a un'altra voce")
            header.addWidget(overlap_badge)
        if utterance.uncertain or utterance.processing_state == "missing_transcription":
            header.addWidget(QLabel("Trascrizione mancante" if utterance.processing_state == "missing_transcription" else "Attribuzione incerta"))

        if getattr(utterance, "suppressed_bleed", False):
            bleed_badge = QLabel("Rientro filtrato")
            bleed_badge.setStyleSheet("""
                background: #2d181e;
                color: #ff7b72;
                font-size: 10px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 4px;
                border: 1px solid #6e2a37;
            """)
            header.addWidget(bleed_badge)

        header.addStretch()

        # Edit button
        self.edit_btn = QPushButton("Modifica")
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #708197;
                font-size: 11px;
                border: none;
                padding: 2px 6px;
            }
            QPushButton:hover {
                color: #edf3fa;
                background: #213044;
                border-radius: 3px;
            }
        """)
        self.edit_btn.clicked.connect(self._toggle_edit)
        header.addWidget(self.edit_btn)
        layout.addLayout(header)

        # Body: Text Label or Text Editor
        self.text_label = QLabel(utterance.text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setStyleSheet("font-size: 13px; color: #edf3fa; line-height: 1.4;")
        layout.addWidget(self.text_label)

        # Editor (nascosto inizialmente)
        self.editor_frame = QWidget()
        editor_layout = QVBoxLayout(self.editor_frame)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(6)

        self.editor = QPlainTextEdit(utterance.text)
        self.editor.setMaximumHeight(80)
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background: #141e2c;
                color: #edf3fa;
                border: 1px solid #5ee0ca;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
        """)
        editor_layout.addWidget(self.editor)

        editor_btns = QHBoxLayout()
        editor_btns.addStretch()
        self.save_btn = QPushButton("Salva")
        self.save_btn.setStyleSheet("background: #5ee0ca; color: #102c29; font-weight: 700;")
        self.save_btn.clicked.connect(self._save_edit)
        editor_btns.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Annulla")
        self.cancel_btn.clicked.connect(self._cancel_edit)
        editor_btns.addWidget(self.cancel_btn)
        editor_layout.addLayout(editor_btns)

        self.editor_frame.hide()
        layout.addWidget(self.editor_frame)

    def _update_style(self) -> None:
        bg = "#182332"
        border = "#2a394d"
        if self._is_active:
            bg = "#1f3044"
            border = "#5ee0ca"

        self.setStyleSheet(f"""
            QFrame#conversationBubble {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 10px;
                margin-bottom: 8px;
            }}
        """)

    def set_active(self, active: bool) -> None:
        if self._is_active != active:
            self._is_active = active
            self._update_style()

    def update_text(self, new_text: str) -> None:
        self.utterance.text = new_text
        self.text_label.setText(new_text)
        self.editor.setPlainText(new_text)

    def update_speaker(self, display_name: str, color_hex: str) -> None:
        self.display_name = display_name
        self.color_hex = color_hex
        self.speaker_label.setText(display_name)
        self.speaker_label.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {color_hex};")

    def _toggle_edit(self) -> None:
        if self._is_editing:
            self._cancel_edit()
        else:
            self._is_editing = True
            self.text_label.hide()
            self.edit_btn.setText("Chiudi")
            self.editor.setPlainText(self.utterance.text)
            self.editor_frame.show()
            self.editor.setFocus()

    def set_read_only(self, read_only: bool) -> None:
        self.edit_btn.setEnabled(not read_only)
        self.edit_btn.setVisible(not read_only)
        if read_only and self._is_editing:
            self._cancel_edit()

    def _save_edit(self) -> None:
        new_text = self.editor.toPlainText().strip()
        if new_text and new_text != self.utterance.text:
            self.textEdited.emit(self.utterance_id, new_text)
        self._cancel_edit()

    def _cancel_edit(self) -> None:
        self._is_editing = False
        self.editor_frame.hide()
        self.edit_btn.setText("Modifica")
        self.text_label.show()


class ConversationView(QScrollArea):
    """Vista scrollabile della conversazione continua ordinata cronologicamente."""

    def __init__(self, controller: TranscriptController | None = None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setObjectName("conversationScrollArea")
        self.setStyleSheet("""
            QScrollArea#conversationScrollArea {
                background: #101722;
                border: none;
            }
        """)

        self.container = QWidget()
        self.container.setStyleSheet("background: #101722;")
        self.card_layout = QVBoxLayout(self.container)
        self.card_layout.setContentsMargins(18, 14, 18, 14)
        self.card_layout.setSpacing(6)
        self.card_layout.addStretch()
        self.setWidget(self.container)

        self._bubbles: list[ConversationBubbleWidget] = []
        self._bubble_by_id: dict[str, ConversationBubbleWidget] = {}
        self._auto_scroll_enabled = True

        # Rileva scroll manuale per non forzare l'autoscroll se l'utente sta leggendo
        self.verticalScrollBar().sliderPressed.connect(self._on_user_scroll_start)
        self.viewport().installEventFilter(self)

        if self.controller is not None:
            self._connect_controller(self.controller)

    def eventFilter(self, watched, event) -> bool:
        from PySide6.QtCore import QEvent
        if watched == self.viewport() and event.type() == QEvent.Wheel:
            self._on_user_scroll_start()
        return super().eventFilter(watched, event)

    def set_controller(self, controller: TranscriptController) -> None:
        self.controller = controller
        self._connect_controller(controller)

    def _connect_controller(self, controller: TranscriptController) -> None:
        controller.sessionChanged.connect(self.load_session)
        controller.utteranceEdited.connect(self._on_utterance_edited)
        controller.activeHighlightsChanged.connect(self._on_active_highlights_changed)
        controller.speakerNamesChanged.connect(self._on_speaker_names_changed)
        controller.readOnlyChanged.connect(self.set_read_only)

    def set_read_only(self, read_only: bool) -> None:
        for bubble in self._bubbles:
            bubble.set_read_only(read_only)

    def load_session(self, session: Session | None) -> None:
        """Costruisce le card dei messaggi in ordine temporale."""
        # Text/timing edits keep widget identity when event IDs are unchanged.
        # Structural edits rebuild; observers never hold a stale replacement.
        ordered = sorted(session.utterances, key=lambda u: (u.start_s, u.end_s)) if session else []
        if ordered and [u.event_id for u in ordered] == [b.utterance_id for b in self._bubbles]:
            for event, bubble in zip(ordered, self._bubbles):
                bubble.utterance = event
                bubble.update_text(event.text)
                bubble.update_speaker(session.display_name(event.speaker_id), _speaker_color(event.speaker, session.display_name(event.speaker_id)))
                bubble.time_btn.setText(_fmt(event.start_s))
                bubble.set_read_only(self.controller.is_read_only if self.controller else False)
            return
        # Pulisce i widget esistenti
        while self.card_layout.count() > 1:
            item = self.card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._bubbles.clear()
        self._bubble_by_id.clear()

        if session is None:
            return

        utterances = session.utterances or []
        # Ordina per start_s in modo rigoroso
        sorted_utts = sorted(utterances, key=lambda u: (u.start_s, u.end_s))
        is_ro = self.controller.is_read_only if self.controller is not None else False

        for utt in sorted_utts:
            name = session.display_name(utt.speaker_id)
            color = _speaker_color(utt.speaker, name)
            bubble = ConversationBubbleWidget(utt, name, color, self.container)
            bubble.set_read_only(is_ro)
            bubble.seekRequested.connect(self._on_seek_requested)
            bubble.textEdited.connect(self._on_text_edited)

            self.card_layout.insertWidget(len(self._bubbles), bubble)
            self._bubbles.append(bubble)
            self._bubble_by_id[utt.utterance_id] = bubble

    def _on_seek_requested(self, seconds: float) -> None:
        self._auto_scroll_enabled = True
        if self.controller is not None:
            self.controller.request_seek(seconds)

    def _on_text_edited(self, utterance_id: str, new_text: str) -> None:
        if self.controller is not None:
            self.controller.edit_text(utterance_id, new_text)

    def _on_utterance_edited(self, utterance_id: str, new_text: str) -> None:
        bubble = self._bubble_by_id.get(utterance_id)
        if bubble is not None:
            bubble.update_text(new_text)

    def _on_speaker_names_changed(self, names: dict) -> None:
        for bubble in self._bubbles:
            name = names.get(bubble.utterance.speaker, bubble.utterance.speaker or "Non assegnato")
            color = _speaker_color(bubble.utterance.speaker, name)
            bubble.update_speaker(name, color)

    def _on_active_highlights_changed(self, active_ids: list) -> None:
        """Evidenzia le card attive e gestisce lo scorrimento automatico."""
        active_set = set(active_ids)
        first_active_bubble: ConversationBubbleWidget | None = None

        for bubble in self._bubbles:
            is_active = bubble.utterance_id in active_set
            bubble.set_active(is_active)
            if is_active and first_active_bubble is None:
                first_active_bubble = bubble

        if first_active_bubble is not None and self._auto_scroll_enabled:
            self.ensureWidgetVisible(first_active_bubble, 50, 50)

    def _on_user_scroll_start(self) -> None:
        # Se l'utente scorre manualmente, mette in pausa l'autoscroll forzato
        self._auto_scroll_enabled = False

    def resume_autoscroll(self) -> None:
        self._auto_scroll_enabled = True
