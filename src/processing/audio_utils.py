"""Utility audio: caricamento mono 16 kHz, normalizzazione e missaggio sorgenti.

Whisper e pyannote lavorano su audio mono a 16 kHz. Qui portiamo qualsiasi
registrazione (mic e/o loopback, con sample rate e canali diversi) in quel
formato, la portiamo a un livello sonoro sensato (registrazioni troppo basse
facevano scartare il parlato dal VAD di Whisper) e, se ci sono piu' sorgenti,
le sommiamo su un unico file allineato.
"""

from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..models.audio_artifacts import AudioAsset, AudioView

TARGET_SR = 16000
TARGET_RMS_DBFS = -20.0   # livello "da parlato" atteso dai modelli
PEAK_CEILING = 0.95       # margine anti-clipping


def audio_info(path: str | Path):
    """Restituisce le informazioni del file senza decodificarlo tutto."""
    return sf.info(str(path))


def channel_count(path: str | Path) -> int:
    """Numero di canali del file, utile per validare il profilo scelto."""
    try:
        return int(audio_info(path).channels)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Impossibile leggere i canali audio di {path}: {exc}") from exc


def load_mono_16k(path: str | Path) -> np.ndarray:
    """Carica un file audio come array float32 mono a 16 kHz."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float32, copy=False)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, int(sr) // g).astype(np.float32, copy=False)
    return mono


def rms_dbfs(x: np.ndarray) -> float:
    """Livello RMS in dBFS (-inf se silenzio)."""
    if x.size == 0:
        return -np.inf
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(r) if r > 0 else -np.inf


def normalize(x: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Porta il segnale al livello RMS voluto, senza superare PEAK_CEILING.

    Usa l'RMS delle sole parti "attive" (sopra il 20esimo percentile di energia):
    con molti silenzi l'RMS globale sottostima il parlato e la normalizzazione
    spingerebbe troppo.
    """
    if x.size == 0:
        return x
    frame = TARGET_SR // 50  # 20 ms
    n = (len(x) // frame) * frame
    if n >= frame:
        frames = x[:n].reshape(-1, frame)
        energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        active = frames[energy >= max(np.percentile(energy, 60), 1e-6)]
        ref = active.ravel() if active.size else x
    else:
        ref = x
    level = rms_dbfs(ref)
    if not np.isfinite(level):
        return x
    gain = 10.0 ** ((target_dbfs - level) / 20.0)
    peak = float(np.max(np.abs(x)))
    if peak * gain > PEAK_CEILING:
        gain = PEAK_CEILING / peak if peak > 0 else 1.0
    return (x * gain).astype(np.float32, copy=False)


def split_dual_channel(path: str | Path, out_dir: str | Path) -> list[Path]:
    """Separa un file stereo mic/sistema nei due file mono 16 kHz corrispondenti.

    La registrazione viene archiviata in un unico MP3 stereo (mic a sinistra,
    sistema a destra). La pipeline pero' ragiona per sorgente, quindi qui il file
    torna a essere due tracce, gia' allineate perche' l'offset e' stato applicato
    al momento dell'unione.
    """
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    out: list[Path] = []
    for i, name in enumerate(("mic", "system")):
        ch = data[:, min(i, data.shape[1] - 1)]
        if sr != TARGET_SR:
            g = gcd(int(sr), TARGET_SR)
            ch = resample_poly(ch, TARGET_SR // g, int(sr) // g).astype(np.float32)
        dst = out_dir / f"{stem}_{name}16k.wav"
        sf.write(str(dst), ch, TARGET_SR, subtype="PCM_16")
        out.append(dst)
    return out


def mix_sources(
    paths: list[str | Path],
    out_path: str | Path,
    *,
    offsets_s: list[float] | None = None,
    do_normalize: bool = True,
    dominance_result: list | None = None,
) -> Path:
    """Somma piu' sorgenti in un unico WAV mono 16 kHz.

    offsets_s (opzionale) sposta ogni sorgente nel tempo: serve quando mic e
    loopback non sono partiti nello stesso istante. Ogni sorgente viene portata
    allo stesso livello prima della somma, cosi' una traccia piu' bassa (tipico
    del microfono rispetto all'audio di sistema) non viene coperta dall'altra.
    """
    srcs = [p for p in paths if p is not None]
    if not srcs:
        raise ValueError("Nessuna sorgente audio da missare")
    offs = list(offsets_s or [0.0] * len(srcs))
    if len(offs) < len(srcs):
        offs += [0.0] * (len(srcs) - len(offs))

    signals: list[np.ndarray] = []
    starts: list[int] = []
    for p, off in zip(srcs, offs):
        s = load_mono_16k(p)
        if do_normalize and len(srcs) > 1:
            s = normalize(s)  # pareggia i livelli tra le sorgenti
        signals.append(s)
        starts.append(max(0, int(round(off * TARGET_SR))))

    if dominance_result is not None:
        # The channel hint uses exactly these normalized samples. Reuse them
        # before releasing the source arrays instead of decoding both files twice.
        normalized = signals if do_normalize and len(srcs) > 1 else [normalize(s) for s in signals]
        dominance_result.append(_dominance_from_signals(normalized, starts, 20))

    length = max(st + len(s) for st, s in zip(starts, signals))
    mixed = np.zeros(length, dtype=np.float32)
    for st, s in zip(starts, signals):
        mixed[st : st + len(s)] += s

    if do_normalize:
        mixed = normalize(mixed)
    else:
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > PEAK_CEILING:
            mixed = (mixed * (PEAK_CEILING / peak)).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")
    return out_path


def source_dominance(
    paths: list[str | Path],
    *,
    offsets_s: list[float] | None = None,
    frame_ms: int = 20,
) -> tuple[np.ndarray, float]:
    """Per ogni frame, indice della sorgente piu' forte (-1 se tutte in silenzio).

    Serve a distinguere "chi sta al microfono" (io) da "chi arriva dall'audio di
    sistema" (gli altri in call): e' un'informazione di canale, molto piu'
    affidabile del solo timbro per quel caso specifico.
    """
    srcs = [p for p in paths if p is not None]
    offs = list(offsets_s or [0.0] * len(srcs))
    if len(offs) < len(srcs):
        offs += [0.0] * (len(srcs) - len(offs))
    signals = []
    starts = []
    for p, off in zip(srcs, offs):
        signals.append(normalize(load_mono_16k(p)))
        starts.append(max(0, int(round(off * TARGET_SR))))
    return _dominance_from_signals(signals, starts, frame_ms)


def _dominance_from_signals(signals, starts, frame_ms):
    frame = max(1, TARGET_SR * frame_ms // 1000)
    envelopes: list[np.ndarray] = []
    for s, pad in zip(signals, starts):
        if pad:
            s = np.concatenate([np.zeros(pad, dtype=np.float32), s])
        n = (len(s) // frame) * frame
        env = (
            np.sqrt(np.mean(np.square(s[:n].reshape(-1, frame), dtype=np.float64), axis=1))
            if n >= frame
            else np.zeros(0)
        )
        envelopes.append(env.astype(np.float32))

    if not envelopes:
        return np.zeros(0, dtype=np.int8), frame_ms / 1000.0
    n_frames = max(len(e) for e in envelopes)
    grid = np.zeros((len(envelopes), n_frames), dtype=np.float32)
    for i, e in enumerate(envelopes):
        grid[i, : len(e)] = e

    floor = 10.0 ** (-45.0 / 20.0)  # sotto -45 dBFS: silenzio
    winner = np.argmax(grid, axis=0).astype(np.int8)
    winner[np.max(grid, axis=0) < floor] = -1
    return winner, frame_ms / 1000.0


def load_multichannel(
    path: str | Path, target_sr: int = TARGET_SR
) -> tuple[np.ndarray, int]:
    """Carica un file audio preservando tutti i canali originari.

    Restituisce una tupla (dati, sample_rate) dove dati ha forma (n_samples, n_channels)
    in float32. Se target_sr differisce dal sample rate originario, effettua il
    ricampionamento polifasico su ciascun canale in modo indipendente senza perdita
    di correlazione di fase.
    """
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != target_sr:
        g = gcd(int(sr), target_sr)
        up = target_sr // g
        down = int(sr) // g
        resampled_channels = [
            resample_poly(data[:, ch], up, down).astype(np.float32, copy=False)
            for ch in range(data.shape[1])
        ]
        data = np.column_stack(resampled_channels).astype(np.float32, copy=False)
        sr = target_sr
    return data, sr


def create_audio_asset(
    path: str | Path, base_dir: str | Path | None = None
) -> AudioAsset:
    """Crea un descrittore AudioAsset calcolando hash e leggendo i metadati del file.

    Se viene fornito base_dir, memorizza il percorso come relativo per garantire
    la portabilita'. L'asset e' immutabile e non modifica il file su disco.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File audio non trovato: {p}")

    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    info = sf.info(str(p))
    rel_path = str(p.relative_to(base_dir)) if base_dir else str(p)

    return AudioAsset(
        relative_path=rel_path,
        sha256=file_hash,
        sample_rate=int(info.samplerate),
        num_channels=int(info.channels),
        num_samples=int(info.frames),
        duration_s=float(info.duration),
        provenance={
            "source_filename": p.name,
            "format": info.format,
            "subtype": info.subtype,
        },
    )


def create_audio_view(
    asset: AudioAsset,
    *,
    channel_indices: list[int] | None = None,
    time_offset_s: float = 0.0,
    duration_s: float | None = None,
    transformations: list[dict[str, Any]] | None = None,
    delay_compensation_s: float = 0.0,
    parent_view_id: str | None = None,
) -> AudioView:
    """Costruisce una vista coerente con l'asset specificato."""
    if channel_indices is None:
        channel_indices = list(range(asset.num_channels))
    for ch in channel_indices:
        if ch >= asset.num_channels:
            raise ValueError(
                f"Canale {ch} fuori intervallo per asset con {asset.num_channels} canali"
            )

    dur = duration_s if duration_s is not None else max(0.0, asset.duration_s - time_offset_s)

    return AudioView(
        asset_id=asset.asset_id,
        channel_indices=channel_indices,
        sample_rate=asset.sample_rate,
        time_offset_s=time_offset_s,
        duration_s=dur,
        transformations=transformations or [],
        delay_compensation_s=delay_compensation_s,
        parent_view_id=parent_view_id,
    )


def extract_view_audio(
    audio_data: np.ndarray,
    view: AudioView,
    source_sr: int = TARGET_SR,
) -> np.ndarray:
    """Estrae i campioni corrispondenti a un'AudioView dai dati audio decodificati."""
    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]

    start_sample = max(0, int(round(view.time_offset_s * source_sr)))
    if view.duration_s > 0:
        end_sample = min(audio_data.shape[0], start_sample + int(round(view.duration_s * source_sr)))
    else:
        end_sample = audio_data.shape[0]

    selected = audio_data[start_sample:end_sample, view.channel_indices]
    if selected.shape[1] == 1:
        return selected[:, 0].astype(np.float32, copy=False)
    return selected.astype(np.float32, copy=False)
