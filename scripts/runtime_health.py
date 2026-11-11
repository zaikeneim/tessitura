"""Offline activation gate; never installs packages or downloads a model."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import add_conda_dll_dirs, use_bundled_models

ROOTS = ("torch", "torchaudio", "torchcodec", "ctranslate2", "faster-whisper",
         "pyannote.audio", "noisereduce", "numpy", "scipy", "onnxruntime", "tokenizers")


def protected_versions() -> dict[str, str]:
    """Pin the installed inference dependency closure, including CUDA suffixes.

    Upgrading an inference engine needs corpus-level accuracy validation; a
    startup smoke test alone cannot establish equal transcription accuracy.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    todo = list(ROOTS)
    versions = {}
    while todo:
        name = canonicalize_name(todo.pop())
        if name in versions:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name in {canonicalize_name(root) for root in ROOTS}:
                raise
            continue
        versions[name] = dist.version
        # Conservatively include installed optional dependencies too.
        for requirement in dist.requires or []:
            todo.append(Requirement(requirement).name)
    return dict(sorted(versions.items()))


def snapshot() -> dict:
    import torch
    return {"versions": protected_versions(), "cuda": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda}


def verify(baseline: dict) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    current = snapshot()
    if current != baseline:
        raise RuntimeError("Backend o dipendenze di inferenza modificati: attivazione rifiutata.")
    import pyexpat  # noqa: F401
    import numpy as np
    import soundfile as sf
    from pyannote.audio import Pipeline  # noqa: F401
    from torchcodec.decoders import AudioDecoder
    from src.processing.transcribe import Transcriber, _ensure_model_path

    # Offline-only: a missing cached model is an explicit failed gate.
    model = _ensure_model_path("tiny", None, local_only=True)
    device = "cuda" if current["cuda"] else "cpu"
    transcriber = Transcriber(model_size=str(model), device=device,
                              compute_type="int8_float16" if current["cuda"] else "int8")
    with tempfile.TemporaryDirectory() as folder:
        wav = Path(folder) / "smoke.wav"
        sf.write(wav, np.zeros(16000, dtype="float32"), 16000)
        AudioDecoder(str(wav)).get_all_samples()
        transcriber.transcribe(wav, language="en", vad_filter=False)
        del transcriber
        from src.processing.diarize import Diarizer
        diarizer = Diarizer(device=device)
        diarizer.diarize(wav)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from src.gui.audio_player import AudioPlayer
    from src.gui.theme import apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    player = AudioPlayer()
    player.close()
    print("Gate offline superato: backend, Whisper, decoder, pyannote e Qt.")


def main() -> None:
    add_conda_dll_dirs()
    use_bundled_models()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--constraints", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        args.snapshot.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")
    elif args.constraints:
        args.constraints.write_text("".join(f"{name}=={version}\n" for name, version in protected_versions().items()), encoding="utf-8")
    else:
        verify(json.loads(args.verify.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
