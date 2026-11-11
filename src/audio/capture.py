"""Registrazione audio su file WAV: microfono e/o output di sistema (loopback).

Ogni sorgente viene registrata su un file WAV separato in un thread dedicato.
Tenere mic e sistema separati e' voluto: aiuta sia il denoise sia la
diarizzazione (la propria voce e quella degli altri arrivano da canali diversi).
Il missaggio eventuale si fa dopo, in fase di elaborazione.

Due accorgimenti che risolvono errori di allineamento visti sulle registrazioni:

- I due stream non partono nello stesso istante (aprire il loopback WASAPI costa
  decine/centinaia di ms). L'istante di partenza di ciascuno viene registrato e
  restituito come *offset*, cosi' il missaggio puo' riallinearli. Senza, la
  diarizzazione lavorava su tracce sfasate e attribuiva le parole alla voce
  sbagliata.

- La lettura dallo stream e la scrittura su disco sono su thread diversi con una
  coda in mezzo: se il disco rallenta, il buffer della scheda audio non va in
  overflow e non si perdono pezzi di audio (che comprimevano la timeline e
  facevano sballare tutti i timestamp successivi).
"""

from __future__ import annotations

import queue
import json
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import paths
from .devices import (
    DeviceInfo,
    default_microphone,
    default_system_loopback,
    pyaudio,
    require_backend,
    resolve_device,
)

_CHUNK = 2048
_FORMAT = pyaudio.paInt16 if pyaudio is not None else 8
_SAMPLE_WIDTH = 2  # byte per campione a 16 bit
_QUEUE_MAX = 512   # ~1 minuto di buffer prima di bloccare


class _StreamRecorder:
    """Registra un singolo dispositivo su un file WAV finche' non viene fermato."""

    def __init__(self, p: pyaudio.PyAudio, device: DeviceInfo, path: Path):
        self._p = p
        self._device = device
        self._path = path
        self._wave: wave.Wave_write | None = None
        self._stream: pyaudio.Stream | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._queue: queue.Queue[bytes | tuple[float, bytes] | None] = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop = threading.Event()
        self._frames_written = 0
        self.error: Exception | None = None
        self.started_at: float | None = None  # time.perf_counter() alla partenza
        self.stopped_at: float | None = None
        self.channels = 1
        self.sample_rate = 16000
        self.level = 0.0
        self.peak = 0.0
        self.timing: list[dict] = []
        self._frames_read = 0
        self._clock_origin: float | None = None
        self._last_block_end: float | None = None
        self.gaps: list[dict] = []
        self.padded_seconds = 0.0
        self.timing_source = "blocking_read_clock_estimate"
        self.timestamp_fallbacks = 0
        self._native_timestamps = 0
        self.last_packet_at: float | None = None
        self._abandon_reader = False
        self._close_wave_when_done = False

    # --- proprieta' diagnostiche -------------------------------------------

    @property
    def recorded_seconds(self) -> float:
        return self._frames_written / float(self.sample_rate or 1)

    @property
    def wall_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else time.perf_counter()
        return max(0.0, end - self.started_at)

    def dropped_seconds(self) -> float:
        """Secondi di audio persi (differenza tra tempo reale e campioni scritti)."""
        return max(0.0, self.wall_seconds - self.recorded_seconds)

    # --- ciclo di vita ------------------------------------------------------

    def start(self) -> None:
        self.channels = max(1, self._device.channels)
        self.sample_rate = self._device.sample_rate
        self._path.parent.mkdir(parents=True, exist_ok=True)

        wf = wave.open(str(self._path), "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(self.sample_rate)
        self._wave = wf

        callback_options = {"stream_callback": self._capture_callback} if self._device.is_loopback else {}
        self._stream = self._p.open(
            format=_FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            # Host-sized loopback packets avoid a partial application buffer
            # spanning two playback bursts with an unreported idle gap inside.
            frames_per_buffer=pyaudio.paFramesPerBufferUnspecified if self._device.is_loopback else _CHUNK,
            input=True,
            input_device_index=self._device.index,
            start=False,
            **callback_options,
        )
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self.started_at = time.perf_counter()
        if self._device.is_loopback:
            stream_time = self._stream.get_time()
            self._clock_origin = self.started_at - stream_time if stream_time > 0 else None
        self._stream.start_stream()
        self._writer.start()
        if not self._device.is_loopback:
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

    def _capture_callback(self, data, frame_count, time_info, status):
        """Keep WASAPI's ADC timestamps: idle playback produces no buffers."""
        if self._stop.is_set():
            return None, pyaudio.paComplete
        if status:
            self.error = RuntimeError(f"WASAPI segnala perdita o discontinuità audio ({status})")
            self._stop.set()
            return None, pyaudio.paAbort
        self.last_packet_at = time.perf_counter()
        adc_time = float(time_info.get("input_buffer_adc_time", 0.0))
        current_time = float(time_info.get("current_time", 0.0))
        if self._clock_origin is None and np.isfinite(current_time) and current_time > 0:
            self._clock_origin = self.last_packet_at - current_time
        if np.isfinite(adc_time) and adc_time > 0 and self._clock_origin is not None:
            first_sample = adc_time + self._clock_origin
            self._native_timestamps += 1
            self.timing_source = "wasapi_adc_clock" if not self.timestamp_fallbacks else "mixed_adc_and_estimated_clock"
        else:
            first_sample = time.perf_counter() - frame_count / self.sample_rate
            self.timestamp_fallbacks += 1
            self.timing_source = "callback_clock_estimate" if not self._native_timestamps else "mixed_adc_and_estimated_clock"
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.level = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        self.peak = max(self.peak, float(np.max(np.abs(samples))) if samples.size else 0.0)
        try:
            self._queue.put_nowait((first_sample, data))
        except queue.Full:
            self.error = RuntimeError("Buffer audio pieno: registrazione interrotta")
            self._stop.set()
            return None, pyaudio.paAbort
        return None, pyaudio.paContinue

    def _pad_gap(self, seconds: float, reason: str) -> None:
        frames = max(0, round(seconds * self.sample_rate))
        if not frames:
            return
        self.gaps.append({"start_frame": self._frames_written, "frames": frames, "reason": reason})
        self.padded_seconds += frames / self.sample_rate
        frame_bytes = _SAMPLE_WIDTH * self.channels
        while frames:
            count = min(frames, self.sample_rate)
            self._wave.writeframes(b"\0" * (count * frame_bytes))
            self._frames_written += count
            frames -= count

    def _read_loop(self) -> None:
        try:
            assert self._stream is not None
            while not self._stop.is_set():
                # A lost buffer must never silently shorten the shared timeline.
                # On overflow/disconnection retain the partial originals and stop.
                data = self._stream.read(_CHUNK, exception_on_overflow=True)
                if self._abandon_reader:
                    break
                self.last_packet_at = time.perf_counter()
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                self.level = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                self.peak = max(self.peak, float(np.max(np.abs(samples))) if samples.size else 0.0)
                self._frames_read += len(data) // (_SAMPLE_WIDTH * self.channels)
                if not self.timing or self._frames_read / self.sample_rate - self.timing[-1]["audio_seconds"] >= 5:
                    self.timing.append({"audio_seconds": self._frames_read / self.sample_rate,
                                        "clock_seconds": time.perf_counter() - self.started_at})
                try:
                    self._queue.put(data, timeout=5)
                except queue.Full:  # disco irrimediabilmente indietro
                    self.error = RuntimeError("Buffer audio pieno: registrazione interrotta")
                    self._stop.set()
        except Exception as exc:  # noqa: BLE001 - riportato a chi chiama stop()
            if not self._stop.is_set():
                self.error = exc
                self.stopped_at = time.perf_counter()
                self._stop.set()
        finally:
            self._queue.put(None)  # sentinella di fine per il writer

    def _write_loop(self) -> None:
        frame_bytes = _SAMPLE_WIDTH * max(1, self.channels)
        while True:
            item = self._queue.get()
            if item is None:
                break
            timestamp = None
            if isinstance(item, tuple):
                timestamp, item = item
            if self._wave is not None:
                try:
                    if timestamp is not None:
                        if self._last_block_end is None:
                            self.started_at = min(self.started_at, timestamp)
                            self._pad_gap(timestamp - self.started_at, "playback_idle_at_start")
                        else:
                            gap = timestamp - self._last_block_end
                            # Ignore normal driver timestamp jitter. Only a gap
                            # between consecutive buffers inserts silence.
                            if gap > .005 or not item:
                                self._pad_gap(gap, "playback_idle")
                    self._wave.writeframes(item)
                except Exception as exc:  # noqa: BLE001
                    self.error = exc
                    # Keep draining so the reader can always enqueue its final
                    # sentinel, even after a disk failure.
                    self._stop.set()
                    continue
            self._frames_written += len(item) // frame_bytes
            if timestamp is not None:
                self._last_block_end = timestamp + len(item) / frame_bytes / self.sample_rate
                if not self.timing or self.recorded_seconds - self.timing[-1]["audio_seconds"] >= 5:
                    self.timing.append({"audio_seconds": self.recorded_seconds,
                                        "clock_seconds": self._last_block_end - self.started_at})
        if self._close_wave_when_done:
            self._finalize_wave()

    def _finalize_wave(self) -> None:
        if self._wave is not None:
            try:
                self._wave.close()
            except Exception as exc:
                self.error = self.error or exc
            self._wave = None

    def stop(self) -> Path:
        self._stop.set()
        self._close_wave_when_done = True
        if self.stopped_at is None:
            self.stopped_at = time.perf_counter()
        if self._stream is not None:
            try:
                self._stream.stop_stream()  # release a blocked read before joining
            except Exception as exc:
                self.error = self.error or exc
                try:
                    self._stream.close()  # release read even when stop_stream fails
                except Exception as close_exc:
                    self.error = self.error or close_exc
                self._stream = None
        if self._reader is not None:
            self._reader.join(timeout=5)
            if self._reader.is_alive():
                self._abandon_reader = True
                self.error = RuntimeError(f"{self.error or ''} Lettura audio non terminata; traccia parziale conservata".strip())
        if (self._reader is None or self._abandon_reader) and self._writer is not None and self._writer.is_alive():
            try:
                if self._device.is_loopback and self.error is None:
                    # WASAPI also omits the tail when playback stops before capture.
                    self._queue.put((self.stopped_at, b""), timeout=1)
                self._queue.put(None, timeout=1)
            except queue.Full:
                self.error = RuntimeError("Scrittura audio bloccata; traccia parziale conservata")
        if self._writer is not None and self._writer.ident is not None:
            self._writer.join(timeout=5)
            if self._writer.is_alive():
                self.error = RuntimeError("Scrittura audio non terminata; traccia parziale conservata")
                return self._path
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as exc:
                self.error = self.error or exc
            self._stream = None
        self._finalize_wave()
        return self._path


@dataclass
class RecordingResult:
    mic_path: Path | None = None
    system_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    # Sfasamento di partenza di ogni traccia rispetto alla prima, in secondi.
    mic_offset: float = 0.0
    system_offset: float = 0.0
    recording_mode: str = "ambient"
    mic_single_speaker_verified: bool = False
    echo_cancellation: bool = False
    metadata_path: Path | None = None

    def offsets(self) -> list[float]:
        """Offset nell'ordine [mic, sistema], solo per le tracce presenti."""
        out: list[float] = []
        if self.mic_path is not None:
            out.append(self.mic_offset)
        if self.system_path is not None:
            out.append(self.system_offset)
        return out


class Recorder:
    """Avvia/ferma la registrazione di microfono e/o output di sistema.

    Esempio:
        rec = Recorder()
        rec.start(record_mic=True, record_system=True)
        ...                      # parla / fai partire la riunione
        result = rec.stop()      # -> RecordingResult con i percorsi dei WAV
    """

    def __init__(self, out_dir: str | Path | None = None):
        self._out_dir = Path(out_dir) if out_dir is not None else None
        self._p: pyaudio.PyAudio | None = None
        self._mic_rec: _StreamRecorder | None = None
        self._sys_rec: _StreamRecorder | None = None
        self._basename = ""
        self.is_recording = False
        self._recording_mode = "ambient"
        self._mic_single_speaker_verified = False
        self._echo_cancellation = False
        self._metadata: dict = {}

    def levels(self) -> dict[str, float]:
        """Independent RMS levels for preview and recording meters."""
        now = time.perf_counter()
        return {kind: float(getattr(rec, "level", 0.0))
                if getattr(rec, "last_packet_at", None) is not None and now - rec.last_packet_at < .25 else 0.0
                for kind, rec in (("mic", self._mic_rec), ("system", self._sys_rec))}

    def capture_errors(self) -> list[str]:
        for rec in (self._mic_rec, self._sys_rec):
            stream = getattr(rec, "_stream", None)
            if rec is not None and rec.error is None and not rec._stop.is_set() and stream is not None:
                try:
                    if not stream.is_active():
                        rec.error = RuntimeError("Dispositivo scollegato o stream audio interrotto")
                except (OSError, RuntimeError) as exc:
                    rec.error = exc
        return [f"{kind}: {rec.error}" for kind, rec in
                (("mic", self._mic_rec), ("system", self._sys_rec))
                if rec is not None and rec.error is not None]

    @property
    def out_dir(self) -> Path:
        return self._out_dir if self._out_dir is not None else paths.recordings_dir()

    def start(
        self,
        *,
        record_mic: bool = True,
        record_system: bool = True,
        mic_device: DeviceInfo | None = None,
        system_device: DeviceInfo | None = None,
        recording_mode: str = "ambient",
        mic_single_speaker_verified: bool = False,
        echo_cancellation: bool = False,
    ) -> None:
        if self.is_recording:
            raise RuntimeError("Registrazione gia' in corso")
        if not (record_mic or record_system):
            raise ValueError("Seleziona almeno una sorgente (mic o sistema)")
        if recording_mode not in ("ambient", "headset", "speakers"):
            raise ValueError(f"Modalità di registrazione sconosciuta: {recording_mode}")
        require_backend()

        self._p = pyaudio.PyAudio()
        self._basename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._recording_mode = recording_mode
        self._mic_single_speaker_verified = bool(mic_single_speaker_verified)
        self._echo_cancellation = bool(echo_cancellation)
        self._metadata = {"version": 1, "recording_mode": recording_mode,
                          "mic_single_speaker_verified": self._mic_single_speaker_verified,
                          "echo_cancellation": self._echo_cancellation,
                          "started_at": datetime.now().astimezone().isoformat(),
                          "state": "recording", "tracks": {}}

        try:
            if record_mic:
                dev = resolve_device(self._p, mic_device) if mic_device else default_microphone(self._p)
                if dev is None:
                    raise RuntimeError("Nessun microfono disponibile")
                if dev.is_loopback:
                    raise ValueError("Seleziona un microfono come dispositivo di ingresso")
                path = self.out_dir / f"{self._basename}_mic.wav"
                self._mic_rec = _StreamRecorder(self._p, dev, path)
                self._mic_rec.start()
                self._metadata["tracks"]["mic"] = {"device": asdict(dev), "path": path.name}

            if record_system:
                dev = resolve_device(self._p, system_device) if system_device else default_system_loopback(self._p)
                if dev is None:
                    raise RuntimeError("Nessun dispositivo di loopback di sistema disponibile")
                if not dev.is_loopback:
                    raise ValueError("Seleziona un endpoint loopback come dispositivo di uscita")
                path = self.out_dir / f"{self._basename}_system.wav"
                self._sys_rec = _StreamRecorder(self._p, dev, path)
                self._sys_rec.start()
                self._metadata["tracks"]["system"] = {"device": asdict(dev), "path": path.name}
            self._save_metadata()
        except Exception:
            # Non lasciare mezza registrazione attiva se la seconda sorgente fallisce.
            for rec in (self._mic_rec, self._sys_rec):
                if rec is not None:
                    try:
                        rec.stop()
                    except Exception:  # noqa: BLE001
                        pass
            self._mic_rec = self._sys_rec = None
            self._p.terminate()
            self._p = None
            raise

        self.is_recording = True

    def _save_metadata(self) -> Path:
        destination = self.out_dir / f"{self._basename}.recording.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def stop(self) -> RecordingResult:
        if not self.is_recording:
            raise RuntimeError("Nessuna registrazione in corso")
        self.capture_errors()
        result = RecordingResult(recording_mode=self._recording_mode,
                                 mic_single_speaker_verified=self._mic_single_speaker_verified,
                                 echo_cancellation=self._echo_cancellation)

        # Stop both sources at the same instant, before waiting for either
        # writer. Slow disk flushing must not lengthen the second recording.
        stopped_at = time.perf_counter()
        for rec in (self._mic_rec, self._sys_rec):
            if rec is not None:
                if rec.stopped_at is None:
                    rec.stopped_at = stopped_at
                rec._stop.set()

        finalized = []
        for rec, kind in ((self._mic_rec, "mic"), (self._sys_rec, "system")):
            if rec is None:
                continue
            try:
                path = rec.stop()
            except Exception as exc:
                rec.error = rec.error or exc
                path = getattr(rec, "_path", None)
            finalized.append((rec, kind, path))
        starts = [rec.started_at for rec, _, _ in finalized if rec.started_at is not None]
        origin = min(starts) if starts else 0.0
        for rec, kind, path in finalized:
            offset = (rec.started_at - origin) if rec.started_at is not None else 0.0
            if kind == "mic":
                result.mic_path, result.mic_offset = path, offset
            else:
                result.system_path, result.system_offset = path, offset
            if rec.error:
                result.errors.append(f"{kind}: {rec.error}")
            lost = rec.dropped_seconds()
            if lost > 0.3:
                result.errors.append(
                    f"{kind}: persi ~{lost:.1f}s di audio (il PC non teneva il passo)"
                )
            if self._metadata:
                track = self._metadata["tracks"][kind]
                track.update(offset_seconds=offset, frames=rec._frames_written,
                             sample_rate=rec.sample_rate, channels=rec.channels,
                             wall_seconds=rec.wall_seconds, dropped_seconds=lost,
                             peak=rec.peak, timing=rec.timing, timing_source=rec.timing_source,
                             timestamp_fallbacks=rec.timestamp_fallbacks,
                             padded_seconds=rec.padded_seconds, gaps=rec.gaps)
                if kind == "system" and rec.peak < 0.001:
                    result.errors.append("system: nessun audio rilevato sul loopback; verifica l'uscita dell'applicazione")

        self._mic_rec = self._sys_rec = None
        if self._p is not None:
            try:
                self._p.terminate()
            except Exception as exc:
                result.errors.append(f"Chiusura backend audio: {exc}")
            self._p = None

        self.is_recording = False
        if self._metadata:
            self._metadata.update(state="complete" if not result.errors else "review_required",
                                  errors=result.errors, stopped_at=datetime.now().astimezone().isoformat())
            try:
                result.metadata_path = self._save_metadata()
            except OSError as exc:
                result.errors.append(f"Metadati non salvati; i WAV originali restano disponibili: {exc}")
        return result


if __name__ == "__main__":
    print("Registro 5 secondi (mic + sistema)... parla e/o fai uscire audio dal PC.")
    rec = Recorder()
    rec.start(record_mic=True, record_system=True)
    time.sleep(5)
    res = rec.stop()
    print("Mic    :", res.mic_path, f"(offset {res.mic_offset * 1000:.0f} ms)")
    print("Sistema:", res.system_path, f"(offset {res.system_offset * 1000:.0f} ms)")
    if res.errors:
        print("Errori:", res.errors)
