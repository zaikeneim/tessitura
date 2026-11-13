"""Worker in thread separati, per non bloccare la GUI durante l'elaborazione."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..audio.encode import merge_to_stereo_mp3
from ..processing.hf_access import ModelAccessError, redact
from ..processing.pipeline import PipelineOptions, process_recording


class StopRecordWorker(QThread):
    """Drain recording queues outside the GUI thread, without losing frames."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, recorder, parent=None):
        super().__init__(parent)
        self._recorder = recorder

    def run(self) -> None:
        try:
            self.done.emit(self._recorder.stop())
        except Exception as exc:
            self.failed.emit(str(exc))


class EncodeWorker(QThread):
    """Unisce le tracce appena registrate in un solo MP3, fuori dal thread UI."""

    # percorso unito, se e' stereo mic/sistema, eventuale avviso
    done = Signal(object, bool, str)

    def __init__(self, mic_path, system_path, offsets_s=None, parent=None):
        super().__init__(parent)
        self._mic = mic_path
        self._system = system_path
        self._offsets = offsets_s or []

    def run(self) -> None:
        dual = self._mic is not None and self._system is not None
        try:
            merged = merge_to_stereo_mp3(
                self._mic, self._system, offsets_s=self._offsets or None,
                delete_sources=False,
            )
            self.done.emit(merged, dual, "")
        except Exception as exc:  # noqa: BLE001 - si tengono i WAV e si prosegue
            self.done.emit(None, False, f"conversione MP3 fallita ({exc})")


class UpdateCheckWorker(QThread):
    """Controlla librerie e modelli senza bloccare la UI (serve la rete)."""

    progress = Signal(str)
    done = Signal(object)   # updates.Report
    failed = Signal(str)

    def run(self) -> None:
        try:
            from ..updates import check

            self.done.emit(check(progress=lambda m: self.progress.emit(m)))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ProcessWorker(QThread):
    """Esegue la pipeline (denoise/trascrizione/diarizzazione) fuori dal thread UI."""

    progress = Signal(str, float)  # messaggio, frazione 0..1
    stage = Signal(object)
    done = Signal(object)          # Session
    failed = Signal(str)
    # Modello di diarizzazione non disponibile: dizionario di
    # ModelAccessError.to_dict(), senza traceback ne' segreti.
    access_failed = Signal(object)

    def __init__(self, mic_path, system_path, options: PipelineOptions, parent=None,
                 offsets_s: list[float] | None = None, dual_channel: bool = False):
        super().__init__(parent)
        self._mic = mic_path
        self._system = system_path
        self._options = options
        self._offsets = offsets_s or []
        self._dual = dual_channel

    def run(self) -> None:
        try:
            session = process_recording(
                self._mic,
                self._system,
                options=self._options,
                offsets_s=self._offsets or None,
                dual_channel=self._dual,
                progress=lambda msg, frac: self.progress.emit(msg, frac),
                on_stage=self.stage.emit,
            )
            self.done.emit(session)
        except ModelAccessError as exc:
            self.access_failed.emit(exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - lo mostriamo nella UI
            self.failed.emit(redact(f"{exc}\n\n{traceback.format_exc()}"))


class CombineAudioWorker(QThread):
    """Unisce piu' registrazioni in un solo MP3 fuori dal thread della GUI.

    ffmpeg su file lunghi impiega decine di secondi: farlo nel thread della UI
    congelerebbe la finestra proprio mentre l'utente aspetta un risultato.
    """

    progress = Signal(str, float)
    done = Signal(object)          # percorso dell'MP3 prodotto
    failed = Signal(str)

    def __init__(self, sources, destination=None, mode: str = "concat", parent=None):
        super().__init__(parent)
        self._sources = list(sources)
        self._destination = destination
        self._mode = mode

    def run(self) -> None:
        try:
            from ..audio.encode import combine_to_mp3

            result = combine_to_mp3(
                self._sources,
                self._destination,
                mode=self._mode,
                on_progress=lambda message, frac: self.progress.emit(message, frac),
            )
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001 - l'errore va mostrato, non ingoiato
            self.failed.emit(str(exc))


class BatchProcessWorker(QThread):
    """Elabora una coda di registrazioni, una alla volta.

    Una alla volta e non in parallelo: trascrizione e diarizzazione occupano
    gia' tutta la GPU, e due elaborazioni insieme finirebbero per litigare sulla
    memoria invece di andare piu' veloci.

    Il progresso riportato e' quello complessivo della coda: la frazione della
    fase in corso viene riscalata dentro la quota del file corrente, cosi' la
    barra non torna indietro a ogni file nuovo.
    """

    progress = Signal(str, float)   # messaggio, frazione complessiva 0..1
    stage = Signal(str, object)     # file corrente, StageEvent
    file_done = Signal(object, object)   # percorso audio, Session
    file_failed = Signal(object, str)    # percorso audio, errore
    done = Signal(object)                # elenco dei percorsi elaborati
    failed = Signal(str)
    access_failed = Signal(object)       # ModelAccessError.to_dict(): coda fermata

    def __init__(self, audio_paths, options: PipelineOptions, parent=None,
                 project_name: str = ""):
        super().__init__(parent)
        self._paths = [Path(item) for item in audio_paths]
        self._options = options
        self._project = project_name
        self._stop = False

    def stop(self) -> None:
        """Ferma la coda dopo il file in corso: un'elaborazione non si spezza."""
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop

    def run(self) -> None:
        completed: list[Path] = []
        total = len(self._paths)
        try:
            for index, audio in enumerate(self._paths):
                if self._stop:
                    break
                prefix = f"{self._project} · " if self._project else ""
                label = f"{prefix}file {index + 1}/{total}: {audio.name}"

                def report(message: str, frac: float, _label=label, _index=index) -> None:
                    overall = (_index + max(0.0, min(1.0, frac))) / total
                    self.progress.emit(f"{_label} — {message}", overall)

                try:
                    session = process_recording(
                        audio, None, options=self._options,
                        session_name=audio.stem, progress=report,
                        on_stage=lambda event, label=label: self.stage.emit(label, event),
                    )
                except ModelAccessError as exc:
                    # Lo stesso modello manca per tutti i file: proseguire
                    # produrrebbe solo altri errori identici. La coda si ferma
                    # e la finestra propone la configurazione, poi la ripresa.
                    self.file_failed.emit(audio, exc.user_message)
                    self.access_failed.emit(exc.to_dict())
                    break
                except Exception as exc:  # noqa: BLE001
                    # Un file illeggibile non deve fermare la coda: si segnala e
                    # si prosegue, altrimenti una cartella da venti registrazioni
                    # si blocca sulla prima corrotta.
                    self.file_failed.emit(audio, redact(f"{exc}"))
                    continue
                completed.append(audio)
                self.file_done.emit(audio, session)
            self.done.emit(completed)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(redact(f"{exc}\n\n{traceback.format_exc()}"))
