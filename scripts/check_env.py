"""Verifica rapida dell'ambiente: GPU, faster-whisper, dispositivi audio, token HF.

    Uso (dalla root del progetto, con l'env tessitura attivo):
    python -m scripts.check_env
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permette di eseguire lo script anche senza installare il pacchetto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import (  # noqa: E402
    add_conda_dll_dirs,
    install_system_certs,
    silence_noisy_warnings,
    use_bundled_models,
)


def check_torch() -> bool:
    import torch

    ok = torch.cuda.is_available()
    print(f"[torch]   versione: {torch.__version__}")
    print(f"[torch]   CUDA disponibile: {ok}")
    if ok:
        print(f"[torch]   GPU: {torch.cuda.get_device_name(0)}")
    return True  # CPU-only installations are valid too.


def check_faster_whisper() -> bool:
    """Carica un modello piccolo su GPU e trascrive un segnale di test (no crash = OK)."""
    import numpy as np
    import soundfile as sf

    from src.processing.transcribe import Transcriber

    from src import paths

    tmp = paths.work_dir() / "_smoke.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sig = (0.05 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(tmp), sig, sr)

    try:
        from src.hardware import detect
        hw = detect()
        tr = Transcriber(model_size="tiny", device=hw.device, compute_type=hw.compute_type)
        segs, lang = tr.transcribe(tmp, language=None)
        print(f"[whisper] modello 'tiny' caricato su {hw.device} e trascrizione eseguita OK "
              f"(lingua rilevata: {lang}, segmenti: {len(segs)})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[whisper] ERRORE: {exc}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def check_diarization_stack() -> bool:
    """Le due DLL che rompono pyannote: pyexpat (ordinali libexpat) e torchcodec (ffmpeg)."""
    ok = True
    try:
        import pyexpat  # noqa: F401

        print("[pyexpat] OK")
    except ImportError as exc:
        print(f"[pyexpat] ERRORE: {exc}")
        print("          libexpat non combacia col build di python: "
              "conda install -n tessitura --override-channels -c conda-forge libexpat")
        ok = False
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder  # noqa: F401

        print(f"[torchcodec] OK ({torchcodec.__version__})")
    except Exception as exc:  # noqa: BLE001
        print(f"[torchcodec] ERRORE: {exc}")
        print("          versione di ffmpeg non supportata da torchcodec (serve <9): "
              'conda install -n tessitura --override-channels -c conda-forge "ffmpeg<9"')
        ok = False
    return ok


def check_audio_devices() -> bool:
    try:
        from src.audio.devices import describe_all

        print("[audio]   dispositivi rilevati:")
        print("\n".join("  " + line for line in describe_all().splitlines()))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[audio]   ERRORE: {exc}")
        return False


def check_hf_token() -> bool:
    from src.processing.diarize import read_hf_token

    token = read_hf_token()
    if token:
        print(f"[hf]      token trovato (…{token[-4:]}) — diarizzazione disponibile")
        return True
    print("[hf]      token HuggingFace NON trovato — la diarizzazione non funzionera'.")
    print("          Crea il file .hf_token nella root o imposta HF_TOKEN.")
    return False


def main() -> int:
    add_conda_dll_dirs()
    use_bundled_models()  # stessa cache modelli che usa l'app
    install_system_certs()
    silence_noisy_warnings()
    print("=== Verifica ambiente Tessitura ===")
    from src import paths

    print(f"[app]     cartella: {paths.app_root()}")
    print(f"[app]     modelli : {paths.models_dir()}")
    results = {
        "torch/CUDA": check_torch(),
        "faster-whisper": check_faster_whisper(),
        "stack diarizzazione": check_diarization_stack(),
        "audio devices": check_audio_devices(),
        "HF token": check_hf_token(),
    }
    print("\n=== Riepilogo ===")
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'NO '} {name}")
    # Il token HF non e' bloccante per l'ambiente di base.
    core_ok = (
        results["torch/CUDA"]
        and results["faster-whisper"]
        and results["stack diarizzazione"]
        and results["audio devices"]
    )
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
