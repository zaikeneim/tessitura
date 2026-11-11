"""Rilevamento dell'hardware e scelta automatica di modello e precisione.

L'app deve girare su PC diversi senza riconfigurarla a mano: qui si guarda che
GPU c'e' (e con quanta VRAM) e si sceglie di conseguenza la taglia del modello
Whisper e il tipo di calcolo. Su una macchina senza GPU NVIDIA `large-v3`
richiederebbe ore per una riunione: meglio proporre da subito un modello che
finisce in un tempo sensato, dicendolo chiaramente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Taglie disponibili, dalla piu' accurata alla piu' veloce.
# large-v3-turbo ha il decoder distillato: stessa qualita' su parlato
# pulito, circa 4x piu' veloce e meno VRAM. Utile sui portatili.
MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]

# Velocita' indicativa in "volte il tempo reale" (quanto audio si trascrive in un
# secondo). Servono solo a dare una stima onesta all'utente, non sono garanzie.
_SPEED: dict[tuple[str, str], float] = {
    ("cuda", "large-v3"): 8.0,
    ("cuda", "large-v3-turbo"): 30.0,
    ("cuda", "medium"): 14.0,
    ("cuda", "small"): 25.0,
    ("cuda", "base"): 40.0,
    ("cuda", "tiny"): 60.0,
    ("cpu", "large-v3"): 0.25,
    ("cpu", "large-v3-turbo"): 0.9,
    ("cpu", "medium"): 0.7,
    ("cpu", "small"): 1.8,
    ("cpu", "base"): 4.0,
    ("cpu", "tiny"): 8.0,
}


@dataclass
class Hardware:
    device: str                 # "cuda" | "cpu"
    compute_type: str           # "float16" | "int8_float16" | "int8"
    model: str                  # taglia Whisper consigliata
    gpu_name: str | None = None
    vram_gb: float | None = None
    cpu_threads: int = 0

    def describe(self) -> str:
        """Riga da mostrare nella UI: cosa e' stato trovato e cosa verra' usato."""
        if self.device == "cuda":
            vram = f", {self.vram_gb:.0f} GB VRAM" if self.vram_gb else ""
            return f"GPU: {self.gpu_name}{vram} — modello consigliato: {self.model}"
        return (
            f"Nessuna GPU NVIDIA: uso la CPU ({self.cpu_threads} thread) — "
            f"modello consigliato: {self.model}"
        )

    def estimate_seconds(self, audio_seconds: float, model: str | None = None) -> float:
        """Stima grossolana del tempo di elaborazione, diarizzazione inclusa."""
        speed = _SPEED.get((self.device, model or self.model), 1.0)
        transcription = audio_seconds / speed
        diarization = audio_seconds / (30.0 if self.device == "cuda" else 3.0)
        return transcription + diarization + 10.0  # + caricamento modelli


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} secondi"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} minuti"
    return f"{minutes / 60.0:.1f} ore"


def estimate_text(hw: Hardware, audio_seconds: float, model: str | None = None) -> str:
    """Frase pronta per la UI con la stima dei tempi."""
    return (
        f"Stima: ~{_format_duration(hw.estimate_seconds(audio_seconds, model))} "
        f"per {_format_duration(audio_seconds)} di audio ({model or hw.model}, {hw.device})."
    )


_cache: Hardware | None = None


def detect(force: str | None = None, refresh: bool = False) -> Hardware:
    """Rileva l'hardware e propone modello e precisione.

    force: "cuda" o "cpu" per imporre un dispositivo (utile per confronti).
    """
    global _cache
    if _cache is not None and not refresh and force is None:
        return _cache

    cpu_threads = os.cpu_count() or 4
    gpu_name: str | None = None
    vram_gb: float | None = None
    has_cuda = False

    if force != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                has_cuda = True
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                vram_gb = props.total_memory / (1024 ** 3)
        except Exception:
            has_cuda = False

    if force == "cuda":
        has_cuda = True

    if has_cuda:
        # large-v3 in int8_float16 dimezza l'uso di VRAM dei pesi mantenendo
        # il calcolo FP16, lasciando ampio margine su schede da 8 GB VRAM.
        if vram_gb is None or vram_gb >= 6.0:
            model, compute = "large-v3", "int8_float16"
        elif vram_gb >= 4.0:
            model, compute = "medium", "int8_float16"
        else:
            model, compute = "small", "int8_float16"
        hw = Hardware("cuda", compute, model, gpu_name, vram_gb, cpu_threads)
    else:
        # Su CPU large-v3 e' fuori scala (ore per una riunione): si parte da small,
        # o base se la macchina ha pochi core.
        model = "small" if cpu_threads >= 8 else "base"
        hw = Hardware("cpu", "int8", model, None, None, cpu_threads)

    if force is None:
        _cache = hw
    return hw
