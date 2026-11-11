"""Player audio con slider, tempo, sincronizzazione e selettore multitraccia (Passo 6).

Consente l'ascolto del Mix complessivo oppure l'isolamento (solo microfono o
solo audio di sistema) per verificare sovrapposizioni e passaggi critici.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget
from .theme import icon

if TYPE_CHECKING:
    from ..models.session import Session


def _fmt(ms: int) -> str:
    s = int(ms // 1000)
    return f"{s // 60:02d}:{s % 60:02d}"


class AudioPlayer(QWidget):
    """Riproduttore audio con supporto a traccia singola o tracce isolate (mic/sistema)."""

    positionChanged = Signal(float)
    sourceChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._tracks: dict[str, Path] = {}
        self._current_track_key: str = "mix"

        self._play_btn = QPushButton()
        self._play_btn.setIcon(icon("play", "#5ee0ca"))
        self._play_btn.setToolTip("Riproduci / pausa")
        self._play_btn.setAccessibleName("Riproduci / pausa")
        self._play_btn.setFixedWidth(44)
        self._play_btn.clicked.connect(self.toggle)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._player.setPosition)
        self._slider.setAccessibleName("Posizione audio")

        self._time = QLabel("00:00 / 00:00")

        # Selettore traccia per audio duale
        self._track_combo = QComboBox()
        self._track_combo.setToolTip("Scegli se ascoltare il mix o isolare un singolo canale")
        self._track_combo.addItem("Mix completo", "mix")
        self._track_combo.addItem("Solo Microfono", "mic")
        self._track_combo.addItem("Solo Sistema", "system")
        self._track_combo.currentIndexChanged.connect(self._on_track_changed)
        self._track_combo.hide()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._time)
        layout.addWidget(self._track_combo)

        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self.clear()

    def clear(self) -> None:
        self._region_end_ms = None
        self._player.stop()
        self._player.setSource(QUrl())
        self._slider.setRange(0, 0)
        self._time.setText("00:00 / 00:00")
        self._tracks.clear()
        self._current_track_key = "mix"
        self._track_combo.blockSignals(True)
        self._track_combo.clear()
        self._track_combo.addItem("Mix completo", "mix")
        self._track_combo.addItem("Solo Microfono", "mic")
        self._track_combo.addItem("Solo Sistema", "system")
        self._track_combo.setCurrentIndex(0)
        self._track_combo.blockSignals(False)
        self._track_combo.hide()
        self.setEnabled(False)

    def _on_duration(self, ms: int) -> None:
        self._slider.setRange(0, ms)
        if getattr(self, "_pending_seek_ms", 0) > 0 and ms >= self._pending_seek_ms:
            target = self._pending_seek_ms
            self._pending_seek_ms = 0
            self._player.setPosition(target)
        self._on_position(self._player.position())

    def load(self, path: str | Path) -> None:
        """Caricamento standard di un singolo file audio."""
        self.clear()
        p = Path(path).resolve()
        self._tracks = {"mix": p}
        self._current_track_key = "mix"
        self._player.setSource(QUrl.fromLocalFile(str(p)))
        self.sourceChanged.emit(str(p))
        self._track_combo.hide()
        self.setEnabled(True)

    def set_tracks(
        self,
        mix_path: str | Path | None,
        mic_path: str | Path | None = None,
        system_path: str | Path | None = None,
    ) -> None:
        """Configura le tracce disponibili consentendo lo switch dinamico durante il playback."""
        self.clear()
        tracks: dict[str, Path] = {}
        if mix_path:
            mp = Path(mix_path).resolve()
            if mp.is_file():
                tracks["mix"] = mp
        if mic_path:
            m_p = Path(mic_path).resolve()
            if m_p.is_file():
                tracks["mic"] = m_p
        if system_path:
            s_p = Path(system_path).resolve()
            if s_p.is_file():
                tracks["system"] = s_p

        self._tracks = tracks
        if not tracks:
            return

        # Popola la combo box
        self._track_combo.blockSignals(True)
        self._track_combo.clear()
        if "mix" in tracks:
            self._track_combo.addItem("Mix completo", "mix")
        if "mic" in tracks:
            self._track_combo.addItem("Solo Microfono", "mic")
        if "system" in tracks:
            self._track_combo.addItem("Solo Sistema", "system")
        self._track_combo.setCurrentIndex(0)
        self._track_combo.blockSignals(False)

        # Mostra la tendina solo se c'è più di una traccia disponibile
        self._track_combo.setVisible(len(tracks) > 1)

        first_key = self._track_combo.currentData() or next(iter(tracks.keys()))
        self._current_track_key = first_key
        self._player.setSource(QUrl.fromLocalFile(str(tracks[first_key])))
        self.sourceChanged.emit(str(tracks[first_key]))
        self.setEnabled(True)

    def load_session_audio(self, session: Session) -> None:
        """Carica le sorgenti audio dalla sessione (mix, mic e system se presenti)."""
        from ..paths import resolve_media
        audio = session.audio
        mix_p = resolve_media(audio.mixed) if audio.mixed else None
        mic_p = resolve_media(audio.mic) if audio.mic else None
        sys_p = resolve_media(audio.system) if audio.system else None
        synchronized = (session.capture_manifest or {}).get("synchronization", {}).get("sources", {})
        # Solo playback shares the mix's global zero, including start offsets
        # and measured clock corrections. Native originals remain in Session.audio.
        prepared_mic = synchronized.get("mic", {}).get("playback_audio")
        prepared_system = synchronized.get("system", {}).get("playback_audio")
        if prepared_mic:
            candidate = resolve_media(prepared_mic)
            if candidate is not None:
                mic_p = candidate
        if prepared_system:
            candidate = resolve_media(prepared_system)
            if candidate is not None:
                sys_p = candidate

        if mic_p and sys_p:
            self.set_tracks(mix_path=mix_p, mic_path=mic_p, system_path=sys_p)
        elif mix_p:
            self.load(mix_p)
        elif mic_p:
            self.load(mic_p)
        # Ambient original and processed mono share recording zero; speaker
        # lanes describe annotations and never imply independent audio sources.
        if session.audio_profile == "stereo_ambient" and mic_p and mix_p:
            self._tracks["original"] = mic_p
            self._tracks["processed"] = mix_p
            for label, key in (("Originale", "original"), ("Elaborato", "processed")):
                if self._track_combo.findData(key) < 0:
                    self._track_combo.addItem(label, key)
            self._track_combo.show()

    def _on_track_changed(self, index: int) -> None:
        track_key = self._track_combo.currentData()
        if not track_key or track_key not in self._tracks or track_key == self._current_track_key:
            return

        pos_ms = max(self._player.position(), self._slider.value())
        was_playing = self._player.playbackState() == QMediaPlayer.PlayingState

        self._current_track_key = track_key
        path = self._tracks[track_key]
        self._pending_seek_ms = pos_ms
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self.sourceChanged.emit(str(path))
        self._player.setPosition(pos_ms)
        self._slider.setValue(pos_ms)

        if was_playing:
            self._player.play()

    def toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def seek_seconds(self, seconds: float) -> None:
        self._region_end_ms = None
        self._player.setPosition(int(seconds * 1000))
        if self._player.playbackState() != QMediaPlayer.PlayingState:
            self._player.play()

    def play_region(self, start_s: float, end_s: float) -> None:
        self.seek_seconds(start_s)
        self._region_end_ms = int(end_s * 1000)

    def select_variant(self, variant: str) -> None:
        index = self._track_combo.findData(variant)
        if index >= 0:
            self._track_combo.setCurrentIndex(index)

    def _on_position(self, ms: int) -> None:
        if self._region_end_ms is not None and ms >= self._region_end_ms:
            self._player.pause()
            self._region_end_ms = None
        if not self._slider.isSliderDown():
            self._slider.blockSignals(True)
            self._slider.setValue(ms)
            self._slider.blockSignals(False)
        self._time.setText(f"{_fmt(ms)} / {_fmt(self._player.duration())}")
        self.positionChanged.emit(ms / 1000.0)

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self._play_btn.setIcon(icon("pause" if playing else "play", "#5ee0ca"))
