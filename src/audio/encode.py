"""Conversione delle registrazioni in MP3 a fine sessione.

Un'ora di riunione occupa ~1.2 GB in WAV (due tracce a 44.1 kHz stereo) e ~80 MB
in MP3 a 192 kbps. La compressione e' con perdita: per conservare tutta
l'informazione originale, tenere i WAV (comportamento predefinito della GUI).

Si registra su WAV e si converte dopo: infilare un encoder nel ciclo di cattura
rischia buffer overflow, e un frame perso comprime la timeline mandando fuori
sincrono tutti i timestamp successivi.
"""

from __future__ import annotations

import shutil
import os
import tempfile
import logging
import subprocess
import sys
from pathlib import Path

BITRATE = "192k"


def ffmpeg_path() -> str:
    """ffmpeg dell'ambiente conda, trovato anche se l'env non e' attivo."""
    bundled = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg non trovato nell'ambiente conda")


def _duration(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def merge_to_stereo_mp3(
    mic_wav: str | Path | None,
    system_wav: str | Path | None,
    *,
    offsets_s: list[float] | None = None,
    delete_sources: bool = False,
) -> Path:
    """Unisce le due tracce in un solo MP3 stereo: mic a sinistra, sistema a destra.

    Un file solo da gestire, ma i due canali restano separati: e' l'informazione
    che permette poi di distinguere chi parla al microfono da chi arriva dalla
    chiamata. Gli sfasamenti di partenza dei due stream si compensano qui, quindi
    a valle le tracce sono gia' allineate.
    """
    srcs = [Path(p) for p in (mic_wav, system_wav) if p is not None]
    if not srcs:
        raise ValueError("Serve almeno una sorgente audio")
    if len(srcs) == 1:
        return to_mp3(srcs[0], delete_source=delete_sources)

    offs = list(offsets_s or [0.0, 0.0])
    offs += [0.0] * (2 - len(offs))
    offs = [max(0.0, off) for off in offs[:2]]
    dst = srcs[0].with_name(srcs[0].name.split("_mic")[0] + ".mp3")
    longest = max(_duration(p) + off for p, off in zip(srcs, offs))

    # mono + riallineamento per canale, poi join in stereo: lo fa ffmpeg in
    # streaming, senza tenere in memoria un'ora di audio.
    chains = []
    for i, off in enumerate(offs[:2]):
        chains.append(
            f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,"
            f"adelay={int(off * 1000)}:all=1,apad,atrim=duration={longest:.6f}[c{i}]"
        )
    graph = ";".join(chains) + ";[c0][c1]join=inputs=2:channel_layout=stereo[out]"

    _encode_atomic([
        "-i", str(srcs[0]), "-i", str(srcs[1]),
        "-filter_complex", graph, "-map", "[out]",
    ], dst, longest)
    if delete_sources:
        _remove_sources(srcs)

    return dst


# Modi di combinazione offerti all'utente.
CONCAT = "concat"   # uno dopo l'altro: registrazione spezzata in piu' file
MIX = "mix"         # sovrapposti: piu' microfoni della stessa scena


def combine_to_mp3(
    sources: list[str | Path],
    destination: str | Path | None = None,
    *,
    mode: str = CONCAT,
    delete_sources: bool = False,
    on_progress=None,
) -> Path:
    """Unisce due o piu' registrazioni in un solo MP3.

    Due situazioni diverse, che non vanno confuse:

    - ``CONCAT`` mette i file uno dopo l'altro, nell'ordine ricevuto. E' il caso
      della sessione spezzata in piu' file dal registratore: la durata finale e'
      la somma e i tempi del secondo file scalano di quanto dura il primo.
    - ``MIX`` li sovrappone: due microfoni che riprendono la stessa scena. La
      durata e' quella del file piu' lungo. La somma di segnali sfasati puo'
      cancellarsi in parte: qui non si tenta di allinearli, e chi combina due
      riprese della stessa stanza deve controllare il risultato prima di
      trascriverlo.

    Il livello non viene ridotto automaticamente (``normalize=0``); un limiter
    finale evita la saturazione senza abbassare tutto il parlato.
    """
    paths = [Path(item) for item in sources]
    if len(paths) < 2:
        raise ValueError("Servono almeno due file audio da combinare.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("File audio non trovati: " + ", ".join(missing))
    if mode not in (CONCAT, MIX):
        raise ValueError(f"Modo di combinazione sconosciuto: {mode}")

    report = on_progress or (lambda _message, _frac: None)
    report("Lettura delle durate…", 0.05)
    durations = [_duration(path) for path in paths]
    expected = sum(durations) if mode == CONCAT else max(durations)

    if destination is None:
        stem = paths[0].stem
        suffix = "unione" if mode == CONCAT else "sovrapposizione"
        destination = paths[0].with_name(f"{stem}_{suffix}.mp3")
    destination = Path(destination)
    if any(path.resolve() == destination.resolve() for path in paths):
        raise ValueError("Il file di destinazione coincide con una sorgente.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Uniformare frequenza e numero di canali prima di unire: file registrati da
    # dispositivi diversi hanno spesso parametri diversi, e concat pretende
    # flussi compatibili.
    prepared = [
        f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[p{index}]"
        for index in range(len(paths))
    ]
    labels = "".join(f"[p{index}]" for index in range(len(paths)))
    if mode == CONCAT:
        joined = f"{labels}concat=n={len(paths)}:v=0:a=1[out]"
    else:
        joined = (f"{labels}amix=inputs={len(paths)}:duration=longest:normalize=0,"
                  f"alimiter=limit=0.95[out]")
    graph = ";".join([*prepared, joined])

    inputs: list[str] = []
    for path in paths:
        inputs += ["-i", str(path)]
    report(f"Unione di {len(paths)} file…", 0.2)
    # Tolleranza proporzionale: ogni giunzione puo' spostare la durata di pochi
    # millisecondi, e con molti file gli scarti si sommano.
    _encode_atomic([*inputs, "-filter_complex", graph, "-map", "[out]"],
                   destination, expected, tolerance=0.15 + 0.05 * len(paths))
    report("Combinazione completata", 1.0)
    if delete_sources:
        _remove_sources(paths)
    return destination


def to_mp3(wav_path: str | Path, *, delete_source: bool = False) -> Path:
    """Converte un WAV in MP3 e (se richiesto) cancella l'originale.

    Il WAV viene cancellato solo dopo aver verificato che l'MP3 sia leggibile e
    di durata compatibile: un ffmpeg interrotto a meta' lascerebbe altrimenti un
    file troncato al posto della registrazione.
    """
    src = Path(wav_path)
    dst = src.with_suffix(".mp3")
    if src.resolve() == dst.resolve():
        raise ValueError("Il file di destinazione coincide con la sorgente")
    _encode_atomic(["-i", str(src)], dst, _duration(src))
    if delete_source:
        _remove_sources([src])
    return dst


def _encode_atomic(inputs: list[str], destination: Path, duration: float,
                   tolerance: float = 0.15) -> None:
    """Publish only a verified output; retain any previous MP3 on failure."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".mp3", delete=False) as file:
        temporary = Path(file.name)
    try:
        subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-xerror", "-y",
             *inputs, "-codec:a", "libmp3lame", "-b:a", BITRATE, str(temporary)],
            check=True, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if abs(_duration(temporary) - duration) > tolerance:
            raise RuntimeError("MP3 incompleto: le sorgenti sono state conservate")
        with temporary.open("rb+") as file:
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_sources(sources: list[Path]) -> None:
    for source in sources:
        try:
            source.unlink()
        except OSError:
            # The output is already valid. A cleanup error must not make the UI
            # fall back to a source that may already have been deleted.
            logging.getLogger(__name__).warning("WAV conservato: %s", source, exc_info=True)
