"""Backend ASR pluggabili e contratti di trascrizione.

Definisce l'interfaccia unificata ASRBackend, l'implementazione Whisper (faster-whisper)
e il registro dei backend disponibili.
"""

from __future__ import annotations

import abc
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from ..models.audio_artifacts import AudioView
from ..models.session import Word
from ..models.transcript_artifacts import ASRHypothesis


class ASRBackend(abc.ABC):
    """Interfaccia astratta per un backend di trascrizione automatica."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.options = kwargs

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Nome identificativo del backend (es. 'whisper', 'qwen3', 'parakeet')."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Verifica se il backend e i relativi pesi sono disponibili localmente."""
        pass

    @abc.abstractmethod
    def transcribe(
        self,
        audio: str | Path | np.ndarray | AudioView,
        language: str | None = None,
        on_progress: Callable[[float, float], None] | None = None,
        **kwargs: Any,
    ) -> list[ASRHypothesis]:
        """Esegue la trascrizione restituendo una lista di ipotesi conformi al contratto."""
        pass


class WhisperBackend(ASRBackend):
    """Backend ASR predefinito basato su faster-whisper."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.device = device
        self.compute_type = compute_type
        self._transcriber = None

    def close(self) -> None:
        if self._transcriber is not None:
            if hasattr(self._transcriber, "close"):
                self._transcriber.close()
            self._transcriber = None

    @property
    def name(self) -> str:
        return "whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_transcriber(self):
        if self._transcriber is None:
            from .transcribe import Transcriber
            self._transcriber = Transcriber(
                model_size=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._transcriber

    def transcribe(
        self,
        audio: str | Path | np.ndarray | AudioView,
        language: str | None = None,
        on_progress: Callable[[float, float], None] | None = None,
        **kwargs: Any,
    ) -> list[ASRHypothesis]:
        view_id = ""
        temp_wav: str | None = None

        if isinstance(audio, AudioView):
            view_id = audio.view_id
            samples = audio.samples
            sr = audio.sample_rate
            # Salva temporaneamente per faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
            sf.write(temp_wav, samples, sr)
            audio_path = temp_wav
        elif isinstance(audio, np.ndarray):
            sr = kwargs.get("sr", 16000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
            sf.write(temp_wav, audio, sr)
            audio_path = temp_wav
        else:
            audio_path = str(audio)

        try:
            transcriber = self._get_transcriber()
            tsegments, detected_lang = transcriber.transcribe(
                audio_path,
                language=language,
                on_progress=on_progress,
                quality_retries=kwargs.get("quality_retries", 24),
            )

            hypotheses: list[ASRHypothesis] = []
            for tseg in tsegments:
                words = [
                    Word(start=w[0], end=w[1], text=w[2], probability=w[3])
                    for w in tseg.words
                ]
                hyp = ASRHypothesis(
                    hypothesis_id=uuid.uuid4().hex[:12],
                    view_id=view_id,
                    start_s=tseg.start,
                    end_s=tseg.end,
                    backend="whisper",
                    model_revision=self.model_name,
                    parameters={"language": detected_lang, "device": self.device},
                    text=tseg.text.strip(),
                    words=words,
                    raw_scores={
                        "avg_logprob": tseg.avg_logprob,
                        "compression_ratio": tseg.compression_ratio,
                        "no_speech_prob": tseg.no_speech_prob,
                    },
                )
                hypotheses.append(hyp)
            return hypotheses
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except OSError:
                    pass


# --- Registro dei backend ---

_REGISTRY: dict[str, type[ASRBackend]] = {
    "whisper": WhisperBackend,
}


def register_asr_backend(name: str, backend_cls: type[ASRBackend]) -> None:
    """Registra un nuovo backend ASR personalizzato."""
    _REGISTRY[name.lower()] = backend_cls


def get_asr_backend(name: str = "whisper", **kwargs: Any) -> ASRBackend:
    """Restituisce un'istanza del backend ASR richiesto."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Backend ASR sconosciuto: '{name}'. Backends registrati: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_available_asr_backends() -> dict[str, dict[str, Any]]:
    """Elenca tutti i backend registrati indicandone la disponibilita' effettiva."""
    res = {}
    for name, cls in _REGISTRY.items():
        instance = cls(model_name="default")
        available = instance.is_available()
        res[name] = {
            "name": name,
            "class": cls.__name__,
            "available": available,
            "offline_safe": True,
        }
    return res
