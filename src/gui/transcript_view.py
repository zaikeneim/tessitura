"""Vista tabellare della trascrizione: tempo, voce (colorata), testo editabile.

Cliccando una riga si emette seekRequested per portare il player a quel punto;
durante la riproduzione highlight_at() evidenzia la riga corrente.
"""

from __future__ import annotations
from bisect import bisect_left, bisect_right

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
)

from ..models.session import Session

# Ruolo in cui la cella "Voce" conserva l'id della voce, distinto dal nome
# mostrato: rinominare una voce non deve cambiare a chi e' attribuita la frase.
SPEAKER_ID_ROLE = Qt.UserRole + 1

# Palette per distinguere le voci a colpo d'occhio.
SPEAKER_COLORS = [
    "#4F8DFD", "#E5534B", "#3FB950", "#D29922", "#A371F7",
    "#EC6CB9", "#00B0D8", "#F0883E", "#57AB5A", "#986EE2",
]


def speaker_color(index: int) -> QColor:
    return QColor(SPEAKER_COLORS[index % len(SPEAKER_COLORS)])


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


class SpeakerDelegate(QStyledItemDelegate):
    """Menu a tendina con le voci della sessione, per correggere l'attribuzione.

    La diarizzazione su registrazioni ambientali sbaglia: serve poter riassegnare
    una frase senza rifare l'elaborazione.
    """

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        for speaker_id, name in self.parent().speaker_options():
            combo.addItem(name, speaker_id)
        return combo

    def setEditorData(self, editor: QComboBox, index) -> None:
        current = index.data(SPEAKER_ID_ROLE)
        pos = editor.findData(current)
        if pos >= 0:
            editor.setCurrentIndex(pos)

    def setModelData(self, editor: QComboBox, model, index) -> None:
        model.setData(index, editor.currentData(), SPEAKER_ID_ROLE)
        model.setData(index, editor.currentText(), Qt.DisplayRole)


class TranscriptView(QTableWidget):
    seekRequested = Signal(float)     # secondi
    textEdited = Signal(int, str)     # indice segmento, nuovo testo
    speakerChanged = Signal(int, str)  # indice segmento, nuovo id voce

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Tempo", "Voce", "Testo"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )
        self._editable_triggers = self.editTriggers()
        self.setWordWrap(True)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().setMinimumSectionSize(46)
        self.setColumnWidth(0, 85)
        self.setColumnWidth(1, 140)

        self._session: Session | None = None
        self._speaker_index: dict[str, int] = {}
        self._speaker_rows: dict[str, list[int]] = {}
        self._display_names: dict[str, str] = {}
        self._loading = False
        self._current_row = -1
        self._prefix_ends: list[float] = []
        self._starts: list[float] = []
        self._ordered = True
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self.resizeRowsToContents)
        self.horizontalHeader().sectionResized.connect(lambda *_: self._resize_timer.start())

        self.setItemDelegateForColumn(1, SpeakerDelegate(self))

        self.cellClicked.connect(self._on_cell_clicked)
        self.itemChanged.connect(self._on_item_changed)

    def speaker_options(self) -> list[tuple[str, str]]:
        """(id, nome) delle voci disponibili, nell'ordine in cui compaiono."""
        if self._session is None:
            return []
        return [(sid, self._session.display_name(sid)) for sid in self._speaker_index]

    def set_read_only(self, read_only: bool) -> None:
        """Blocca modifiche quando viene mostrata la copia grezza."""
        self.setEditTriggers(
            QAbstractItemView.NoEditTriggers if read_only else self._editable_triggers
        )

    def set_session(self, session: Session) -> None:
        self._loading = True
        self.setUpdatesEnabled(False)
        self.clearSelection()
        self._current_row = -1
        self._session = session
        self._starts = [seg.start for seg in session.segments]
        self._ordered = all(a <= b for a, b in zip(self._starts, self._starts[1:]))
        self._prefix_ends = []
        latest = float("-inf")
        for seg in session.segments:
            latest = max(latest, seg.end)
            self._prefix_ends.append(latest)
        self._speaker_index = {sid: i for i, sid in enumerate(session.speaker_ids())}
        self._speaker_rows = {sid: [] for sid in self._speaker_index}
        self._display_names = {sid: session.display_name(sid) for sid in self._speaker_index}
        self.setRowCount(len(session.segments))
        for row, seg in enumerate(session.segments):
            self._speaker_rows[seg.speaker].append(row)
            time_item = QTableWidgetItem(_fmt(seg.start))
            time_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.setItem(row, 0, time_item)

            spk_item = QTableWidgetItem(session.display_name(seg.speaker))
            spk_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            spk_item.setData(SPEAKER_ID_ROLE, seg.speaker)
            spk_item.setForeground(QBrush(speaker_color(self._speaker_index.get(seg.speaker, 0))))
            spk_item.setToolTip("Doppio clic per correggere la voce attribuita")
            self.setItem(row, 1, spk_item)

            text_item = QTableWidgetItem(seg.text)
            if seg.warnings:
                tips = []
                for w in seg.warnings:
                    if w.startswith("overlap_group:"):
                        parts = w.split(":")
                        stream_info = parts[2] if len(parts) > 2 else "simultanea"
                        tips.append(f"🎙️ Voce simultanea ({stream_info})")
                if "voce_sovrapposta" in seg.warnings:
                    cand_names = [
                        c.get("display_name", c.get("speaker"))
                        for c in seg.candidates if "speaker" in c
                    ]
                    cand_str = f" (voci concorrenti: {', '.join(cand_names)})" if cand_names else ""
                    tips.append(f"⚠️ Voce sovrapposta{cand_str}")
                if "frase_troncata" in seg.warnings:
                    tips.append("⚠️ Frase interrotta / troncata")
                if tips:
                    text_item.setToolTip("\n".join(tips))
            self.setItem(row, 2, text_item)

        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)
        self.resizeRowsToContents()
        self._loading = False
        self.setUpdatesEnabled(True)

    def refresh_speakers(self) -> None:
        """Aggiorna i nomi delle voci mostrati dopo una rinomina."""
        if self._session is None:
            return
        self._loading = True
        for sid, rows in self._speaker_rows.items():
            name = self._session.display_name(sid)
            if name == self._display_names[sid]:
                continue
            self._display_names[sid] = name
            for row in rows:
                self.item(row, 1).setText(name)
        self._loading = False

    def highlight_at(self, seconds: float) -> None:
        if self._session is None:
            return
        # Skip all rows that ended before this timestamp, preserving the first
        # match even for overlapping or imported out-of-order segments.
        stop = bisect_right(self._starts, seconds) if self._ordered else len(self._starts)
        for i in range(bisect_left(self._prefix_ends, seconds), stop):
            seg = self._session.segments[i]
            if seg.start <= seconds <= seg.end:
                if i != self._current_row:
                    self._current_row = i
                    self.selectRow(i)
                    self.scrollToItem(self.item(i, 0), QAbstractItemView.PositionAtCenter)
                return
        if self._current_row != -1:
            self._current_row = -1
            self.clearSelection()

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if _col == 0 and self._session and 0 <= row < len(self._session.segments):
            self.seekRequested.emit(self._session.segments[row].start)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self._session is None:
            return
        row = item.row()
        if not (0 <= row < len(self._session.segments)):
            return
        if item.column() == 2:
            # Centralizza l'edit per mantenere raw/timing, revision e undo.
            self._session.edit_segment(row, text=item.text())
            self.textEdited.emit(row, item.text())
            self.resizeRowToContents(row)
        elif item.column() == 1:
            self._reassign_speaker(row, item)

    def _reassign_speaker(self, row: int, item: QTableWidgetItem) -> None:
        """Sposta il segmento su un'altra voce dopo la scelta nel menu a tendina."""
        new_id = item.data(SPEAKER_ID_ROLE)
        segment = self._session.segments[row]
        if not new_id or new_id == segment.speaker:
            return

        self._speaker_rows.get(segment.speaker, []).remove(row)
        self._speaker_rows.setdefault(new_id, []).append(row)
        self._session.edit_segment(row, speaker=new_id)

        self._loading = True
        item.setForeground(QBrush(speaker_color(self._speaker_index.get(new_id, 0))))
        self._loading = False
        self.speakerChanged.emit(row, new_id)
