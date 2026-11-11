"""Pannello per assegnare un nome a ciascuna voce rilevata (SPEAKER_xx -> nome)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ..models.session import Session
from .transcript_view import speaker_color


class SpeakerPanel(QScrollArea):
    """Elenca le voci con un campo per rinominarle; emette namesChanged a ogni modifica."""

    namesChanged = Signal()
    speakerConfirmed = Signal(str, str)
    exportSpeakerRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._form = QFormLayout(self._container)
        self._form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self._form.setVerticalSpacing(12)
        self._form.addRow(QLabel("Le voci rilevate appariranno qui."))
        self.setWidget(self._container)
        self._session: Session | None = None
        self._edits: dict[str, QLineEdit] = {}

    def set_session(self, session: Session) -> None:
        self._session = session
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._edits = {}
        for i, sid in enumerate(session.speaker_ids()):
            color = speaker_color(i)
            label = QLabel(f"● {sid}")
            label.setStyleSheet(f"color: {color.name()}; font-weight: bold;")
            
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            edit = QLineEdit(session.speaker_names.get(sid, ""))
            edit.setPlaceholderText("Nome…")
            edit.textChanged.connect(lambda text, s=sid: self._on_change(s, text))
            edit.editingFinished.connect(lambda s=sid, widget=edit: self._on_confirm(s, widget.text()))
            self._edits[sid] = edit
            row_layout.addWidget(edit, 1)

            export_btn = QPushButton("TXT")
            export_btn.setToolTip(f"Esporta la sola trascrizione di {session.display_name(sid)}")
            export_btn.setFixedWidth(36)
            export_btn.setStyleSheet("""
                QPushButton {
                    background: #192738;
                    color: #5ee0ca;
                    font-size: 10px;
                    font-weight: 700;
                    border: 1px solid #2d4561;
                    border-radius: 4px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background: #23374e;
                    border-color: #5ee0ca;
                }
            """)
            export_btn.clicked.connect(lambda _=False, s=sid: self.exportSpeakerRequested.emit(s))
            row_layout.addWidget(export_btn)

            self._form.addRow(label, row_layout)

    def _on_change(self, speaker_id: str, text: str) -> None:
        if self._session is not None:
            self._session.rename_speaker(speaker_id, text)
            self.namesChanged.emit()

    def _on_confirm(self, speaker_id: str, text: str) -> None:
        """La perdita del focus è la conferma esplicita usata per il profilo."""
        if self._session is not None:
            self.speakerConfirmed.emit(speaker_id, " ".join(text.split()))
