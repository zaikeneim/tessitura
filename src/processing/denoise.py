"""Riduzione del rumore di fondo.

`noisereduce` (spectral gating): leggero, senza GPU, buono per rumori stazionari
(ventole, brusio, hum).

ATTENZIONE ai valori: con prop_decrease=0.9 e stima non stazionaria il gating
mangiava anche il parlato piu' basso (misurato: -36%..-66% di energia sulle parti
di parlato piu' morbido), e quelle parti sparivano dalla trascrizione. Il default
ora e' conservativo e, dopo il filtro, il livello viene riportato a quello di
partenza: togliere rumore non deve abbassare la voce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import normalize, rms_dbfs

# Se il denoise toglie piu' di questo, sta togliendo voce e non rumore.
_MAX_LOSS_DB = 6.0


def denoise_array(
    data: np.ndarray,
    sr: int,
    *,
    stationary: bool = True,
    prop_decrease: float = 0.5,
) -> np.ndarray:
    """Riduce il rumore mantenendo il livello del parlato.

    stationary=True e' piu' prudente sul parlato: stima un profilo di rumore
    unico invece di inseguire il segnale frame per frame (che sul parlato lento
    o lontano finisce per considerarlo rumore).
    """
    import noisereduce as nr

    before = rms_dbfs(data if data.ndim == 1 else data.mean(axis=1))

    if data.ndim > 1:
        channels = [
            nr.reduce_noise(
                y=data[:, ch], sr=sr, stationary=stationary, prop_decrease=prop_decrease
            )
            for ch in range(data.shape[1])
        ]
        reduced = np.stack(channels, axis=1).astype(np.float32)
    else:
        reduced = nr.reduce_noise(
            y=data, sr=sr, stationary=stationary, prop_decrease=prop_decrease
        ).astype(np.float32)

    # Recupera il livello perso dal gating (con tetto: se ha tolto troppo,
    # e' meglio un po' di rumore che parlato scomparso).
    after = rms_dbfs(reduced if reduced.ndim == 1 else reduced.mean(axis=1))
    if np.isfinite(before) and np.isfinite(after):
        loss = before - after
        if loss > 0:
            gain = 10.0 ** (min(loss, _MAX_LOSS_DB + 6.0) / 20.0)
            reduced = (reduced * gain).astype(np.float32)
    peak = float(np.max(np.abs(reduced))) if reduced.size else 0.0
    if peak > 0.95:
        reduced = (reduced * (0.95 / peak)).astype(np.float32)
    return reduced


def denoise_file(
    in_path: str | Path,
    out_path: str | Path,
    *,
    stationary: bool = True,
    prop_decrease: float = 0.5,
) -> Path:
    """Applica la riduzione del rumore a un file audio e salva il risultato.

    Se il risultato perde troppa energia rispetto all'originale (segno che il
    gating sta mangiando la voce), scrive l'audio originale e basta.
    """
    data, sr = sf.read(str(in_path), dtype="float32")
    reduced = denoise_array(data, sr, stationary=stationary, prop_decrease=prop_decrease)

    ref = data if data.ndim == 1 else data.mean(axis=1)
    out = reduced if reduced.ndim == 1 else reduced.mean(axis=1)
    if np.isfinite(rms_dbfs(ref)) and (rms_dbfs(ref) - rms_dbfs(out)) > _MAX_LOSS_DB:
        reduced = data  # denoise troppo distruttivo: meglio l'originale

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), normalize(reduced) if reduced.ndim == 1 else reduced, sr,
             subtype="PCM_16")
    return out_path
