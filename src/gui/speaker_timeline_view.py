"""Horizontal multitrack review of canonical speech events.

Time is absolute across all lanes. Lanes describe speakers, not independent
recording channels. Only the visible time window is laid out and drawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from ..models.session import Session
from ..models.transcript_artifacts import Utterance
from .transcript_view import SPEAKER_COLORS, _fmt

if TYPE_CHECKING:
    from .transcript_controller import TranscriptController

LABEL_WIDTH = 140
RULER_HEIGHT = 32
WAVEFORM_HEIGHT = 68
LANE_HEIGHT = 48


class _WaveformSignals(QObject):
    done = Signal(object)


class _WaveformTask(QRunnable):
    """Read peaks outside the GUI thread without retaining a full decoded file."""

    def __init__(self, key, source=None):
        super().__init__()
        self.key = key
        self.source = source
        self.signals = _WaveformSignals()

    def run(self):
        path, start_s, end_s, bins = self.key
        try:
            from ..processing.waveform import WaveformPeaks
            source = self.source if self.source is not None else WaveformPeaks(path)
            self.signals.done.emit({"key": self.key, "peaks": source.read(start_s, end_s, bins),
                                    "duration_s": source.duration_s, "source": source, "error": ""})
        except Exception as exc:
            self.signals.done.emit({"key": self.key, "peaks": [], "duration_s": 0.0, "source": None, "error": str(exc)})


class TimelineCanvas(QWidget):
    """Viewport-sized canvas with absolute time, waveform, speaker lanes and selection."""

    seekRequested = Signal(float)
    utteranceSelected = Signal(str)
    regionSelected = Signal(float, float)
    viewportChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumWidth(480)
        self._session: Session | None = None
        self._speakers: list[str | None] = [None]
        self._utterances: list[Utterance] = []
        self._blocks: list[dict] = []
        self._lanes: list[dict] = []
        self._active_ids: set[str] = set()
        self._selected_id: str | None = None
        self._current_time_s = 0.0
        self._total_duration_s = 60.0
        self.window_start_s = 0.0
        self.window_duration_s = 60.0
        self.selected_region: tuple[float, float] | None = None
        self._drag_start_s: float | None = None
        self._peaks: list[tuple[float, float]] = []
        self.setStyleSheet("background: #101722;")

    @property
    def pixels_per_second(self) -> float:
        return max(1, self.width() - LABEL_WIDTH - 8) / self.window_duration_s

    def time_x(self, seconds: float) -> float:
        return LABEL_WIDTH + (seconds - self.window_start_s) * self.pixels_per_second

    def x_time(self, x: float) -> float:
        return min(self._total_duration_s, max(0.0, self.window_start_s + (x - LABEL_WIDTH) / self.pixels_per_second))

    def set_session_data(self, session: Session | None) -> None:
        previous_session = self._session
        self._session = session
        self._utterances = session.utterances if session is not None else []
        speaker_ids = session.speaker_ids() if session is not None else []
        self._speakers = [speaker_id for speaker_id in speaker_ids
                          if speaker_id and speaker_id.upper() not in ("UNKNOWN", "UNASSIGNED", "NONE")]
        self._speakers.append(None)
        if session is not None:
            self._total_duration_s = max(10.0, session.duration,
                                         self._total_duration_s if previous_session is session else 0.0)
            if previous_session is not session:
                self.window_start_s = 0.0
                self._active_ids.clear()
                self._peaks = []
        else:
            self._total_duration_s = 60.0
            self._selected_id = None
            self.selected_region = None
            self._peaks = []
        self._recompute_layout()
        self.update()

    def set_duration(self, seconds: float) -> None:
        self._total_duration_s = max(self._total_duration_s, seconds)

    def set_window(self, start_s: float, duration_s: float) -> None:
        self.window_duration_s = max(5.0, min(7200.0, float(duration_s)))
        self.window_start_s = max(0.0, min(float(start_s), max(0.0, self._total_duration_s - self.window_duration_s)))
        self._peaks = []
        self._recompute_layout()
        self.update()
        self.viewportChanged.emit()

    def _recompute_layout(self) -> None:
        self._blocks.clear()
        self._lanes.clear()
        top = RULER_HEIGHT + WAVEFORM_HEIGHT + 8
        end_s = self.window_start_s + self.window_duration_s
        for speaker_index, speaker_id in enumerate(self._speakers):
            lane_events = sorted((u for u in self._utterances
                                  if u.speaker_id == speaker_id and u.end_s >= self.window_start_s and u.start_s <= end_s),
                                 key=lambda u: (u.start_s, u.end_s, u.event_id))
            sublane_ends: list[float] = []
            for utterance in lane_events:
                x = self.time_x(utterance.start_s)
                right = max(x + 3.0, self.time_x(utterance.end_s))
                sublane = next((i for i, previous in enumerate(sublane_ends) if x >= previous + 1), len(sublane_ends))
                if sublane == len(sublane_ends):
                    sublane_ends.append(right)
                else:
                    sublane_ends[sublane] = right
                rect = QRectF(x, top + sublane * LANE_HEIGHT + 5, right - x, LANE_HEIGHT - 10)
                self._blocks.append({"rect": rect, "utterance": utterance,
                                     "color": QColor(SPEAKER_COLORS[speaker_index % len(SPEAKER_COLORS)])})
            height = max(1, len(sublane_ends)) * LANE_HEIGHT
            self._lanes.append({"speaker": speaker_id, "top": top, "height": height})
            top += height
        self.setMinimumHeight(max(260, top + 10))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._recompute_layout()
        self.viewportChanged.emit()

    def set_active_highlights(self, active_ids: list[str]) -> None:
        self._active_ids = set(active_ids)
        self.update()

    def set_current_time(self, seconds: float) -> None:
        self._current_time_s = seconds
        self.update()

    def set_selected_id(self, utterance_id: str | None) -> None:
        self._selected_id = utterance_id
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or event.position().x() < LABEL_WIDTH:
            return
        for block in self._blocks:
            if block["rect"].contains(event.position()):
                utterance = block["utterance"]
                self.utteranceSelected.emit(utterance.event_id)
                self.seekRequested.emit(utterance.start_s)
                return
        self._drag_start_s = self.x_time(event.position().x())
        self.selected_region = (self._drag_start_s, self._drag_start_s)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_s is not None:
            current = self.x_time(event.position().x())
            self.selected_region = tuple(sorted((self._drag_start_s, current)))
            self.update()
            return
        for block in self._blocks:
            if block["rect"].contains(event.position()):
                utterance = block["utterance"]
                self.setToolTip(f"{utterance.start_s:.3f}–{utterance.end_s:.3f} s\n{utterance.text or 'Trascrizione mancante'}")
                return
        self.setToolTip("Trascina per selezionare una regione. I canali audio non corrispondono alle corsie delle voci.")

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._drag_start_s is None:
            return
        start_s, end_s = sorted((self._drag_start_s, self.x_time(event.position().x())))
        self._drag_start_s = None
        if end_s - start_s >= 0.05:
            self.selected_region = (start_s, end_s)
            self.regionSelected.emit(start_s, end_s)
        else:
            self.seekRequested.emit(start_s)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QColor("#101722"))
        painter.setFont(QFont("Segoe UI", 9))
        for lane in self._lanes:
            top, height = lane["top"], lane["height"]
            painter.fillRect(QRectF(0, top, LABEL_WIDTH, height), QColor("#172333"))
            painter.setPen(QColor("#ced8e5"))
            name = self._session.display_name(lane["speaker"]) if self._session else "Non assegnato"
            painter.drawText(QRectF(8, top + 5, LABEL_WIDTH - 16, height - 10), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, name)
            painter.setPen(QColor("#263548"))
            painter.drawLine(QPointF(0, top + height), QPointF(self.width(), top + height))
        painter.save()
        painter.setClipRect(QRectF(LABEL_WIDTH, 0, self.width() - LABEL_WIDTH, self.height()))
        if self._peaks:
            center = RULER_HEIGHT + WAVEFORM_HEIGHT / 2
            width = max(1, self.width() - LABEL_WIDTH - 8)
            painter.setPen(QPen(QColor("#648dad"), 1))
            for index, (low, high) in enumerate(self._peaks):
                x = LABEL_WIDTH + index * width / len(self._peaks)
                painter.drawLine(QPointF(x, center - high * 28), QPointF(x, center - low * 28))
        else:
            painter.setPen(QColor("#8395aa"))
            painter.drawText(LABEL_WIDTH + 10, RULER_HEIGHT + 35, "Forma d'onda: caricamento o audio non disponibile")
        raw_step = self.window_duration_s / 8
        scale = 10 ** math.floor(math.log10(max(raw_step, 0.01)))
        step = next((value * scale for value in (1, 2, 5, 10) if value * scale >= raw_step), 10 * scale)
        tick = math.ceil(self.window_start_s / step) * step
        painter.setFont(QFont("Consolas", 8))
        while tick <= self.window_start_s + self.window_duration_s:
            x = self.time_x(tick)
            painter.setPen(QPen(QColor("#263548"), 1, Qt.DotLine))
            painter.drawLine(QPointF(x, RULER_HEIGHT), QPointF(x, self.height()))
            painter.setPen(QColor("#b5c6d9"))
            label = _fmt(tick) if step >= 1 else f"{_fmt(tick)}.{int(tick % 1 * 1000):03d}"
            painter.drawText(QPointF(x + 3, 21), label)
            tick += step
        for block in self._blocks:
            utterance, rect, color = block["utterance"], block["rect"], block["color"]
            missing = utterance.processing_state == "missing_transcription"
            rejected = utterance.processing_state == "rejected_activity"
            border = QColor("#8793a2") if rejected else QColor("#ffb76b") if missing or utterance.uncertain else color
            fill = QColor(color)
            fill.setAlpha(90 if utterance.event_id in self._active_ids else 38)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#ffffff") if utterance.event_id == self._selected_id else border,
                                2 if utterance.event_id == self._selected_id else 1,
                                Qt.DashLine if missing or utterance.uncertain else Qt.SolidLine))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(QColor("#eff4fa"))
            prefix = "[Falsa attività] " if rejected else "[↔] " if utterance.is_overlap else ""
            text = prefix + (utterance.text or "Trascrizione mancante")
            if rect.width() > 20:
                painter.drawText(rect.adjusted(5, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter,
                                 painter.fontMetrics().elidedText(text, Qt.ElideRight, max(1, int(rect.width() - 10))))
        if self.selected_region:
            start_s, end_s = self.selected_region
            painter.fillRect(QRectF(self.time_x(start_s), RULER_HEIGHT, max(1, (end_s - start_s) * self.pixels_per_second),
                                    self.height() - RULER_HEIGHT), QColor(94, 224, 202, 24))
        x = self.time_x(self._current_time_s)
        painter.setPen(QPen(QColor("#5ee0ca"), 2))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.restore()
        painter.setPen(QColor("#7f96ad"))
        painter.drawText(8, 21, "Tempo MM:SS")
        painter.drawText(8, RULER_HEIGHT + 34, "Audio condiviso")


class SpeakerTimelineView(QWidget):
    """Timeline, waveform and undoable event correction through one controller."""

    regionPlaybackRequested = Signal(float, float)
    reprocessRequested = Signal(float, float, bool)
    audioVariantRequested = Signal(str)
    identityReviewRequested = Signal()

    def __init__(self, controller: TranscriptController | None = None, parent=None):
        super().__init__(parent)
        self.controller = None
        self._session: Session | None = None
        self._selected_utterance: Utterance | None = None
        self._auto_scroll = True
        self._audio_path: str | None = None
        self._waveform_task = None
        self._waveform_source = None
        self._waveform_key = None
        self._pending_waveform_key = None
        self._region = (0.0, 1.0)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.bar = QHBoxLayout()
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(5, 7200)
        self.zoom.setValue(60)
        self.zoom.setSuffix(" s visibili")
        self.zoom.setAccessibleName("Ampiezza della finestra temporale")
        self.zoom.valueChanged.connect(self._zoom_changed)
        self.bar.addWidget(QLabel("Zoom"))
        self.bar.addWidget(self.zoom)
        self.audio_variant = QComboBox()
        self.audio_variant.addItem("Audio originale", "original")
        self.audio_variant.addItem("Audio elaborato", "processed")
        self.audio_variant.currentIndexChanged.connect(lambda: self.audioVariantRequested.emit(self.audio_variant.currentData()))
        self.bar.addWidget(self.audio_variant)
        self.identity_review_btn = QPushButton("Rivedi identità nell’intero audio")
        self.identity_review_btn.clicked.connect(self._request_identity_review)
        self.bar.addWidget(self.identity_review_btn)
        self.follow_btn = QPushButton("Segui riproduzione")
        self.follow_btn.clicked.connect(self._resume_autoscroll)
        self.follow_btn.hide()
        self.bar.addStretch()
        self.bar.addWidget(self.follow_btn)
        main_layout.addLayout(self.bar)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter, 1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvas = TimelineCanvas()
        self.scroll.setWidget(self.canvas)
        self.splitter.addWidget(self.scroll)
        inspector_scroll = QScrollArea()
        self.inspector_scroll = inspector_scroll
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inspector_scroll.setMinimumWidth(300)
        inspector_scroll.setMaximumWidth(390)
        self.inspector = QFrame()
        inspector_scroll.setWidget(self.inspector)
        self.splitter.addWidget(inspector_scroll)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self._edit_widgets = [self.identity_review_btn]
        self._build_inspector()
        self.position = QSlider(Qt.Horizontal)
        self.position.setAccessibleName("Scorrimento temporale assoluto")
        self.position.valueChanged.connect(self._offset_changed)
        self.position.sliderPressed.connect(self._on_user_scrolled)
        main_layout.addWidget(self.position)
        self.canvas.seekRequested.connect(self._on_seek_requested)
        self.canvas.utteranceSelected.connect(self._on_utterance_selected)
        self.canvas.regionSelected.connect(self._region_dragged)
        self._waveform_timer = QTimer(self)
        self._waveform_timer.setSingleShot(True)
        self._waveform_timer.setInterval(80)
        self._waveform_timer.timeout.connect(self._request_waveform)
        self.canvas.viewportChanged.connect(self._waveform_timer.start)
        if controller is not None:
            self.set_controller(controller)

    def _spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0, 864000)
        spin.setSuffix(" s")
        return spin

    def _button(self, layout, label, callback, *, edits=True):
        button = QPushButton(label)
        button.setMinimumWidth(0)
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        button.clicked.connect(callback)
        layout.addWidget(button)
        if edits:
            self._edit_widgets.append(button)
        return button

    def _build_inspector(self) -> None:
        layout = QVBoxLayout(self.inspector)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(QLabel("EVENTO SELEZIONATO"))
        self.insp_time = QLabel("Tempo: --:--")
        self.insp_speaker = QLabel("Voce: --")
        layout.addWidget(self.insp_time)
        layout.addWidget(self.insp_speaker)
        self.event_status = QLabel("")
        self.event_status.setWordWrap(True)
        layout.addWidget(self.event_status)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.speaker_combo = QComboBox()
        self.speaker_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.speaker_combo.setMinimumContentsLength(12)
        self.speaker_combo.setMinimumWidth(0)
        self.speaker_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.speaker_combo.currentTextChanged.connect(self.speaker_combo.setToolTip)
        self.start_spin, self.end_spin = self._spin(), self._spin()
        form.addRow("Voce", self.speaker_combo)
        form.addRow("Inizio assoluto", self.start_spin)
        form.addRow("Fine assoluta", self.end_spin)
        layout.addLayout(form)
        self._edit_widgets.extend([self.speaker_combo, self.start_spin, self.end_spin])
        row = QHBoxLayout()
        row.setSpacing(6)
        self._button(row, "Nuova voce", self._create_speaker)
        self._button(row, "Rinomina", self._rename_speaker)
        layout.addLayout(row)
        confirm_identity = self._button(layout, "Conferma identità", self._confirm_identity)
        confirm_identity.setToolTip("Conferma l'identità della voce soltanto per l'evento selezionato.")
        self.insp_text = QPlainTextEdit()
        self.insp_text.setPlaceholderText("Inserisci manualmente il parlato mancante")
        self.insp_text.setMinimumHeight(95)
        layout.addWidget(self.insp_text)
        self.uncertain_check = QCheckBox("Evento incerto")
        self.uncertain_check.setToolTip("Marca l'evento come incerto e da verificare.")
        layout.addWidget(self.uncertain_check)
        self._edit_widgets.extend([self.insp_text, self.uncertain_check])
        self.save_text_btn = self._button(layout, "Salva modifiche", self._save_inspector_text)
        row = QHBoxLayout()
        row.setSpacing(6)
        self._button(row, "Dividi", self._split_event)
        merge = self._button(row, "Unisci", self._merge_event)
        merge.setToolTip("Unisci al successivo evento compatibile della stessa voce e sorgente.")
        layout.addLayout(row)
        self._button(layout, "Mostra predizione originale", self._show_prediction, edits=False)
        self.reject_activity_btn = self._button(layout, "Marca come falsa attività", self._mark_false_activity)
        self.reject_activity_btn.setToolTip("Conserva l'evento e la cronologia; indica che l'attività rilevata non è parlato da trascrivere.")
        layout.addWidget(QLabel("IDENTITÀ DA VERIFICARE NELL'INTERO AUDIO"))
        self.identity_suspects = QListWidget()
        self.identity_suspects.setMaximumHeight(110)
        self.identity_suspects.currentRowChanged.connect(self._identity_suspect_selected)
        layout.addWidget(self.identity_suspects)
        self.identity_detail = QPlainTextEdit()
        self.identity_detail.setReadOnly(True)
        self.identity_detail.setMaximumHeight(130)
        layout.addWidget(self.identity_detail)
        layout.addWidget(QLabel("REGIONE AUDIO"))
        region_form = QFormLayout()
        region_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        region_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.region_start, self.region_end = self._spin(), self._spin()
        self.region_end.setValue(1)
        region_form.addRow("Inizio", self.region_start)
        region_form.addRow("Fine", self.region_end)
        self.region_start.valueChanged.connect(self._region_spins_changed)
        self.region_end.valueChanged.connect(self._region_spins_changed)
        layout.addLayout(region_form)
        self._button(layout, "Riproduci regione", self._play_region, edits=False)
        self._button(layout, "Crea intervento mancante", self._create_event)
        self._button(layout, "Ritrascrivi solo la regione", lambda: self._reprocess(False))
        self._button(layout, "Separa e trascrivi la regione", lambda: self._reprocess(True))
        lane_note = QLabel("Le corsie rappresentano voci; non separano i canali audio.")
        lane_note.setWordWrap(True)
        layout.addWidget(lane_note)
        layout.addWidget(QLabel("CANDIDATI DA VERIFICARE"))
        self.candidates = QListWidget()
        self.candidates.setMaximumHeight(100)
        self.candidates.currentRowChanged.connect(self._candidate_selected)
        layout.addWidget(self.candidates)
        self.candidate_detail = QPlainTextEdit()
        self.candidate_detail.setReadOnly(True)
        self.candidate_detail.setMaximumHeight(130)
        layout.addWidget(self.candidate_detail)
        self.accept_candidate_btn = self._button(layout, "Accetta come evento distinto", self._accept_candidate)
        self._button(layout, "Rifiuta candidato", self._reject_candidate)
        self.candidate_views = QComboBox()
        self.candidate_views.setAccessibleName("Confronto audio del candidato")
        layout.addWidget(self.candidate_views)
        from .audio_player import AudioPlayer
        self.candidate_player = AudioPlayer(self.inspector)
        layout.addWidget(self.candidate_player)
        self._button(layout, "Ascolta vista selezionata", self._play_candidate, edits=False)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addStretch()
        for label in self.inspector.findChildren(QLabel):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        for editor in self.inspector.findChildren(QPlainTextEdit):
            editor.setMinimumWidth(0)
            editor.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        for items in (self.identity_suspects, self.candidates):
            items.setMinimumWidth(0)
            items.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            items.setWordWrap(True)
            items.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def set_controller(self, controller: TranscriptController) -> None:
        if self.controller is controller:
            return
        self.controller = controller
        self._connect_controller(controller)

    def _connect_controller(self, controller: TranscriptController) -> None:
        controller.sessionChanged.connect(self.load_session)
        controller.activeHighlightsChanged.connect(self._on_active_highlights_changed)
        controller.speakerNamesChanged.connect(self._on_speakers_changed)
        controller.selectionChanged.connect(self._on_utterance_selected)
        controller.readOnlyChanged.connect(self.set_read_only)
        self.set_read_only(controller.is_read_only)

    def set_read_only(self, read_only: bool) -> None:
        for widget in self._edit_widgets:
            widget.setEnabled(not read_only)

    def load_session(self, session: Session | None) -> None:
        previous_session = self._session
        self._session = session
        selected_id = self.controller.selected_id if self.controller else None
        self._selected_utterance = None
        self._clear_inspector()
        self.canvas.set_session_data(session)
        if previous_session is not session:
            self._select_region(0.0, 1.0)
            self._auto_scroll = True
            self.follow_btn.hide()
        self._refresh_speakers()
        self.identity_suspects.clear()
        review = session.diarization.get("global_identity_review", {}) if session else {}
        for region in review.get("suspected_regions", []):
            start_s, end_s = float(region.get("start_s", 0)), float(region.get("end_s", 0))
            reason = region.get("reason") or region.get("kind", "identità da verificare")
            item = QListWidgetItem(f"{start_s:.3f}–{end_s:.3f}s · {reason}")
            item.setData(Qt.UserRole, region)
            self.identity_suspects.addItem(item)
        self.candidates.clear()
        for candidate in session.recovery_candidates if session else []:
            if candidate.get("status") != "proposed":
                continue
            event = candidate.get("event") or candidate.get("event_snapshot", {})
            proposal = self._proposed_speaker(candidate)
            assignment = candidate.get("kind") == "speaker_assignment"
            suffix = f"Attribuzione proposta: {session.display_name(proposal)} · {event.get('text', '')}" if assignment else event.get("text") or "Trascrizione mancante"
            item = QListWidgetItem(f"{float(event.get('start_s', 0)):.3f}s · {suffix}")
            item.setData(Qt.UserRole, candidate)
            self.candidates.addItem(item)
        self._update_scroll_range()
        if selected_id:
            self._on_utterance_selected(selected_id)
        if previous_session is not session or not self._audio_path:
            path = session.audio.resolved_primary() if session else None
            self.set_audio_path(path)
        if self.controller:
            self.set_read_only(self.controller.is_read_only)

    set_session = load_session

    def set_audio_path(self, path: str | Path | None) -> None:
        path = str(path) if path else None
        if self._audio_path != path:
            self._audio_path = path
            self._waveform_key = None
            self._waveform_source = None
            self.canvas._peaks = []
        self._waveform_timer.start()

    def _request_waveform(self) -> None:
        if not self._audio_path:
            return
        key = (self._audio_path, round(self.canvas.window_start_s, 3),
               round(self.canvas.window_start_s + self.canvas.window_duration_s, 3),
               max(64, self.canvas.width() - LABEL_WIDTH - 8))
        if key == self._waveform_key:
            return
        if self._waveform_task is not None:
            self._pending_waveform_key = key
            return
        self._waveform_key = key
        task = _WaveformTask(key, self._waveform_source)
        self._waveform_task = task
        task.signals.done.connect(self._waveform_done)
        QThreadPool.globalInstance().start(task)

    def _waveform_done(self, result) -> None:
        self._waveform_task = None
        if result["key"][0] == self._audio_path:
            self._waveform_source = result["source"]
        current_window = (round(self.canvas.window_start_s, 3),
                          round(self.canvas.window_start_s + self.canvas.window_duration_s, 3))
        if (result["key"] == self._waveform_key and result["key"][0] == self._audio_path
                and result["key"][1:3] == current_window):
            self.canvas._peaks = result["peaks"]
            self.canvas.set_duration(result["duration_s"])
            self._update_scroll_range()
            self.canvas.update()
            if result["error"]:
                self.message.setText(f"Forma d'onda non disponibile: {result['error']}")
        if self._pending_waveform_key:
            self._pending_waveform_key = None
            self._waveform_timer.start()

    def _update_scroll_range(self) -> None:
        self.position.blockSignals(True)
        self.position.setRange(0, max(0, int((self.canvas._total_duration_s - self.canvas.window_duration_s) * 1000)))
        self.position.setValue(int(self.canvas.window_start_s * 1000))
        self.position.blockSignals(False)

    def _offset_changed(self, milliseconds: int) -> None:
        self.canvas.set_window(milliseconds / 1000, self.zoom.value())

    def _zoom_changed(self, seconds: float) -> None:
        self.canvas.set_window(self.canvas.window_start_s, seconds)
        self._update_scroll_range()

    def _on_seek_requested(self, seconds: float) -> None:
        if self.controller:
            self.controller.request_seek(seconds)

    def _on_utterance_selected(self, utterance_id: str) -> None:
        if self.controller and self.controller.selected_id != utterance_id:
            self.controller.select_utterance(utterance_id)
            return
        if not self._session:
            return
        utterance = next((u for u in self._session.utterances if u.event_id == utterance_id), None)
        if utterance is None:
            return
        self._selected_utterance = utterance
        if not self.canvas.window_start_s <= utterance.start_s < self.canvas.window_start_s + self.canvas.window_duration_s:
            self.canvas.set_window(max(0, utterance.start_s - self.canvas.window_duration_s * .15), self.zoom.value())
            self._update_scroll_range()
        self.canvas.set_selected_id(utterance_id)
        self.insp_time.setText(f"Tempo: {_fmt(utterance.start_s)} – {_fmt(utterance.end_s)}")
        self.insp_speaker.setText(f"Voce: {self._session.display_name(utterance.speaker_id)}")
        self.start_spin.setValue(utterance.start_s)
        self.end_spin.setValue(utterance.end_s)
        self.speaker_combo.setCurrentIndex(max(0, self.speaker_combo.findData(utterance.speaker_id)))
        self.insp_text.setPlainText(utterance.text)
        self.uncertain_check.setChecked(utterance.uncertain)
        statuses = {
            "missing_transcription": "Trascrizione incompleta: inserisci il parlato mancante o richiedi candidati.",
            "rejected_activity": "Falsa attività: evento e cronologia conservati. Il marcatore non elimina il testo.",
        }
        status = statuses.get(utterance.processing_state, utterance.processing_state)
        self.event_status.setText(status)
        self._select_region(utterance.start_s, utterance.end_s)

    def _refresh_speakers(self) -> None:
        current = self.speaker_combo.currentData()
        self.speaker_combo.clear()
        self.speaker_combo.addItem("Non assegnato", None)
        if self._session:
            for speaker_id in self._session.speaker_ids():
                if speaker_id and speaker_id.upper() not in ("UNKNOWN", "UNASSIGNED", "NONE"):
                    self.speaker_combo.addItem(self._session.display_name(speaker_id), speaker_id)
        self.speaker_combo.setCurrentIndex(max(0, self.speaker_combo.findData(current)))

    def _run_edit(self, callback):
        if not self.controller or self.controller.is_read_only:
            return None
        try:
            result = callback()
            self.message.setText("")
            return result
        except (ValueError, StopIteration, KeyError) as exc:
            self.message.setText(str(exc) or "Evento o candidato non disponibile.")
            return None

    def _save_inspector_text(self) -> None:
        if not self._selected_utterance:
            return
        event_id = self._selected_utterance.event_id
        changes = dict(text=self.insp_text.toPlainText(), speaker_id=self.speaker_combo.currentData(),
                       start_s=self.start_spin.value(), end_s=self.end_spin.value(), uncertain=self.uncertain_check.isChecked())
        self._run_edit(lambda: self.controller.edit_event(event_id, **changes))

    def _create_speaker(self) -> None:
        name, accepted = QInputDialog.getText(self, "Nuova voce", "Nome della voce:")
        if accepted:
            event_id = self._selected_utterance.event_id if self._selected_utterance else None
            speaker_id = self._run_edit(lambda: self.controller.create_speaker(name))
            if speaker_id and event_id:
                self._run_edit(lambda: self.controller.reassign_speaker(event_id, speaker_id))

    def _rename_speaker(self) -> None:
        speaker_id = self.speaker_combo.currentData()
        if speaker_id is None or not self._session:
            return
        name, accepted = QInputDialog.getText(self, "Rinomina voce", "Nome:", text=self._session.display_name(speaker_id))
        if accepted:
            self._run_edit(lambda: self.controller.rename_speaker(speaker_id, name))

    def _confirm_identity(self) -> None:
        speaker_id = self.speaker_combo.currentData()
        if self._selected_utterance and speaker_id:
            event_id = self._selected_utterance.event_id
            self._run_edit(lambda: self.controller.confirm_identity(event_id, speaker_id))

    def _request_identity_review(self) -> None:
        if self.controller and not self.controller.is_read_only:
            self.identityReviewRequested.emit()

    def _mark_false_activity(self) -> None:
        if self._selected_utterance:
            event_id = self._selected_utterance.event_id
            self._run_edit(lambda: self.controller.edit_event(
                event_id, processing_state="rejected_activity", uncertain=False))

    def _identity_suspect_selected(self, _row: int) -> None:
        item = self.identity_suspects.currentItem()
        region = item.data(Qt.UserRole) if item else None
        if not region or not self._session:
            self.identity_detail.clear()
            return
        start_s, end_s = float(region.get("start_s", 0)), float(region.get("end_s", 0))
        event_id = region.get("proposed_event_id") or region.get("event_id")
        matches = [event for event in self._session.utterances
                   if event.start_s < end_s and event.end_s > start_s]
        selected = next((event for event in matches if event.event_id == event_id), None)
        if selected is None and matches:
            speaker = region.get("speaker")
            selected = max(matches, key=lambda event: (
                event.speaker_id == speaker, min(event.end_s, end_s) - max(event.start_s, start_s)))
        if selected:
            self._on_utterance_selected(selected.event_id)
        self._select_region(start_s, end_s)
        review = self._session.diarization.get("global_identity_review", {})
        evidence = {"reason": region.get("reason") or region.get("kind"),
                    "source_audio_ref": selected.source_audio_ref if selected else self._session.audio.primary(),
                    "region": region,
                    "review_model": {key: value for key, value in review.items()
                                     if key not in ("centroids", "suspected_regions", "possible_fragmentation")}}
        self.identity_detail.setPlainText(json.dumps(evidence, ensure_ascii=False, indent=2))
        self._on_seek_requested(start_s)

    def _split_event(self) -> None:
        event = self._selected_utterance
        if event is None:
            return
        initial = self.canvas._current_time_s
        if not event.start_s < initial < event.end_s:
            initial = (event.start_s + event.end_s) / 2
        seconds, accepted = QInputDialog.getDouble(self, "Dividi evento", "Tempo assoluto di divisione (s):", initial,
                                                  event.start_s + .001, event.end_s - .001, 3)
        if not accepted:
            return
        left_text = right_text = None
        left_words = [w for w in event.words if (w.start + w.end) / 2 < seconds]
        right_words = [w for w in event.words if (w.start + w.end) / 2 >= seconds]
        if not left_words or not right_words:
            left_text, accepted = QInputDialog.getMultiLineText(self, "Prima parte", "Testo prima della divisione:", event.text)
            if not accepted:
                return
            right_text, accepted = QInputDialog.getMultiLineText(self, "Seconda parte", "Testo dopo la divisione:")
            if not accepted:
                return
        self._run_edit(lambda: self.controller.split_event(event.event_id, seconds, left_text, right_text))

    def _merge_event(self) -> None:
        event = self._selected_utterance
        if event is None or self._session is None:
            return
        lane = sorted((u for u in self._session.utterances if u.speaker_id == event.speaker_id),
                      key=lambda u: (u.start_s, u.end_s, u.event_id))
        index = lane.index(event)
        if index + 1 >= len(lane):
            self.message.setText("Nessun evento successivo nella stessa corsia.")
            return
        self._run_edit(lambda: self.controller.merge_events(event.event_id, lane[index + 1].event_id))

    def _show_prediction(self) -> None:
        if self._selected_utterance:
            event = self._selected_utterance
            text = json.dumps({"original_prediction": event.original_prediction,
                               "asr_provenance": event.asr_provenance,
                               "diarization_provenance": event.diarization_provenance,
                               "user_corrections": event.user_corrections}, ensure_ascii=False, indent=2)
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Predizione e correzioni")
            dialog.setText("La predizione originale resta separata dalle correzioni.")
            dialog.setDetailedText(text)
            dialog.exec()

    def _select_region(self, start_s: float, end_s: float) -> None:
        self._region = (start_s, end_s)
        self.region_start.blockSignals(True)
        self.region_end.blockSignals(True)
        self.region_start.setValue(start_s)
        self.region_end.setValue(end_s)
        self.region_start.blockSignals(False)
        self.region_end.blockSignals(False)
        self.canvas.selected_region = self._region
        self.canvas.update()

    def _region_dragged(self, start_s: float, end_s: float) -> None:
        self._on_user_scrolled()
        self._select_region(start_s, end_s)

    def _region_spins_changed(self) -> None:
        self._region = (self.region_start.value(), self.region_end.value())
        self.canvas.selected_region = self._region
        self.canvas.update()

    def _valid_region(self) -> bool:
        if self._region[0] < self._region[1]:
            return True
        self.message.setText("La fine della regione deve seguire l'inizio.")
        return False

    def _play_region(self) -> None:
        if self._valid_region():
            self.regionPlaybackRequested.emit(*self._region)

    def _create_event(self) -> None:
        if not self._valid_region():
            return
        text, accepted = QInputDialog.getMultiLineText(self, "Intervento mancante", "Trascrizione manuale (può restare vuota):")
        if accepted:
            start_s, end_s = self._region
            speaker_id = self.speaker_combo.currentData()
            self._run_edit(lambda: self.controller.create_event(start_s, end_s, text, speaker_id))

    def _reprocess(self, separate: bool) -> None:
        if self.controller and not self.controller.is_read_only and self._valid_region():
            self.reprocessRequested.emit(*self._region, separate)

    def _candidate_selected(self, _row: int) -> None:
        item = self.candidates.currentItem()
        candidate = item.data(Qt.UserRole) if item else None
        self.candidate_detail.setPlainText(json.dumps(candidate, ensure_ascii=False, indent=2) if candidate else "")
        assignment = bool(candidate and candidate.get("kind") == "speaker_assignment")
        self.accept_candidate_btn.setText("Accetta attribuzione proposta" if assignment else "Accetta come evento distinto")
        self.candidate_views.clear()
        self.candidate_player.clear()
        if candidate:
            event = candidate.get("event") or candidate.get("event_snapshot", {})
            if not assignment:
                self.speaker_combo.setCurrentIndex(max(0, self.speaker_combo.findData(None)))
            provenance = event.get("asr_provenance", {})
            context = provenance.get("context_interval", [0., 0.])
            original = provenance.get("original_context_audio")
            if original:
                self.candidate_views.addItem("Originale: canali nativi", {"path": original, "offset_s": context[0]})
            names = {"original": "Originale mono", "normalized": "Normalizzazione Tessitura",
                     "bounded_gain": "Gain conservativo", "acoustic_separation": "Sorgente separata",
                     "enhanced_separation": "Gain e separazione acustica"}
            for view in provenance.get("comparison_views", []):
                label = names.get(view["strategy"], view["strategy"])
                self.candidate_views.addItem(f"{label} · traccia {view['source_index'] + 1}", view)
            if not provenance.get("comparison_views") and provenance.get("candidate_audio"):
                self.candidate_views.addItem("Audio candidato", {"path": provenance["candidate_audio"],
                    "offset_s": provenance.get("candidate_audio_offset_s", 0.)})
            if assignment:
                event_id = candidate.get("proposed_event_id")
                if event_id:
                    self._on_utterance_selected(event_id)
                proposed_speaker = self._proposed_speaker(candidate)
                index = self.speaker_combo.findData(proposed_speaker)
                if index < 0 and proposed_speaker and self._session:
                    self.speaker_combo.addItem(self._session.display_name(proposed_speaker), proposed_speaker)
                    index = self.speaker_combo.count() - 1
                self.speaker_combo.setCurrentIndex(max(0, index))
            self._select_region(float(event.get("start_s", 0)), float(event.get("end_s", 0)))

    @staticmethod
    def _proposed_speaker(candidate):
        return (candidate.get("proposed_speaker") or candidate.get("proposed_speaker_id")
                or candidate.get("speaker_id") or (candidate.get("event") or {}).get("speaker"))

    def _accept_candidate(self) -> None:
        item = self.candidates.currentItem()
        if item:
            candidate_id = item.data(Qt.UserRole)["candidate_id"]
            speaker_id = self.speaker_combo.currentData()
            self._run_edit(lambda: self.controller.accept_candidate(candidate_id, speaker_id))

    def _reject_candidate(self) -> None:
        item = self.candidates.currentItem()
        if item:
            self._run_edit(lambda: self.controller.reject_candidate(item.data(Qt.UserRole)["candidate_id"]))

    def _play_candidate(self) -> None:
        view = self.candidate_views.currentData()
        if not view:
            return
        from ..paths import resolve_media
        path = resolve_media(view["path"])
        if path is None or not path.is_file():
            self.message.setText("Audio candidato non più disponibile: ritrascrivi la regione.")
            return
        self.candidate_player.load(path)
        offset = view.get("offset_s", 0.)
        self.candidate_player.play_region(max(0., self._region[0] - offset), max(0., self._region[1] - offset))

    def _on_speakers_changed(self, _names: dict) -> None:
        self._refresh_speakers()
        if self._selected_utterance and self._session:
            self.insp_speaker.setText(f"Voce: {self._session.display_name(self._selected_utterance.speaker_id)}")
        self.canvas.update()

    def _on_active_highlights_changed(self, active_ids: list) -> None:
        self.canvas.set_active_highlights(active_ids)

    def update_playback_position(self, seconds: float) -> None:
        self.canvas.set_current_time(seconds)
        if self._auto_scroll and not self.canvas.window_start_s <= seconds <= self.canvas.window_start_s + self.canvas.window_duration_s * .85:
            self.canvas.set_window(max(0.0, seconds - self.canvas.window_duration_s * .35), self.zoom.value())
            self._update_scroll_range()

    def _on_user_scrolled(self) -> None:
        self._auto_scroll = False
        self.follow_btn.show()

    def _resume_autoscroll(self) -> None:
        self._auto_scroll = True
        self.follow_btn.hide()
        self.canvas.set_window(max(0, self.canvas._current_time_s - self.canvas.window_duration_s * .35), self.zoom.value())
        self._update_scroll_range()

    def _clear_inspector(self) -> None:
        self.insp_time.setText("Tempo: --:--")
        self.insp_speaker.setText("Voce: --")
        self.insp_text.clear()
        self.event_status.clear()
        self.uncertain_check.setChecked(False)
        self.canvas.set_selected_id(None)
