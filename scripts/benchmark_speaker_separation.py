"""Benchmark C: separazione acustica contro traccia grezza su canale multivoce (§9.1).

Domanda a cui rispondere: su un solo canale che contiene piu' voci, separare
prima di trascrivere recupera parole rispetto al percorso standard?

Banco di prova: le due tracce reali mic e sistema sono monovocali e note, quindi
la loro somma e' un canale multivoce con verita' a terra disponibile. Le
trascrizioni delle tracce pulite fanno da riferimento; i due rami in gara vedono
solo la somma.

  RAW = percorso standard oggi in produzione: ASR sulla miscela + diarizzazione
  SEP = percorso speaker_isolated: separazione ONNX, poi una ASR per traccia

Metrica: cpWER, che assegna le ipotesi ai riferimenti nella combinazione
migliore, piu' tempi di elaborazione e parole recuperate.

Esecuzione:
    python -m scripts.benchmark_speaker_separation --seconds 180
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing.audio_profile import AudioProfile  # noqa: E402
from src.processing.audio_utils import PEAK_CEILING, load_mono_16k  # noqa: E402
from src.processing.evaluation import cpwer, wer  # noqa: E402
from src.processing.pipeline import PipelineOptions, process_recording  # noqa: E402
from src.processing.separation_backends import (  # noqa: E402
    MODEL_SAMPLE_RATE,
    OnnxSeparationBackend,
    separation_dir,
)
from src.processing.transcribe import Transcriber  # noqa: E402

DEFAULT_MIC = "data/recordings/esempio_mic.wav"
DEFAULT_SYSTEM = "data/recordings/esempio_system.wav"


def _write(path: Path, signal: np.ndarray) -> Path:
    sf.write(str(path), signal, MODEL_SAMPLE_RATE)
    return path


def _transcribe_text(transcriber: Transcriber, path: Path) -> str:
    segments, _lang = transcriber.transcribe(path, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _reset_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0


def _words(text: str) -> int:
    return len(text.split())


def run(
    mic_source: Path,
    system_source: Path,
    seconds: float,
    model_size: str,
    device: str,
    chunk_seconds: float,
    separator_name: str,
) -> dict:
    print("=" * 60)
    print(f"  BENCHMARK SEPARAZIONE — finestra {seconds:.0f}s, modello {model_size}, {device}")
    print("=" * 60)

    mic_signal = load_mono_16k(mic_source)[: int(seconds * MODEL_SAMPLE_RATE)]
    system_signal = load_mono_16k(system_source)[: int(seconds * MODEL_SAMPLE_RATE)]
    width = min(mic_signal.size, system_signal.size)
    mic_signal, system_signal = mic_signal[:width], system_signal[:width]

    mixture = (mic_signal + system_signal).astype(np.float32)
    peak = float(np.max(np.abs(mixture))) or 1.0
    if peak > PEAK_CEILING:
        mixture *= PEAK_CEILING / peak
    actual = width / MODEL_SAMPLE_RATE
    print(f"-> Canale multivoce costruito: {actual:.1f}s, picco {peak:.2f}")

    with tempfile.TemporaryDirectory(prefix="bench_sep_") as tmp:
        root = Path(tmp)
        mic_wav = _write(root / "mic.wav", mic_signal)
        system_wav = _write(root / "system.wav", system_signal)
        mixture_wav = _write(root / "mixture.wav", mixture)

        # --- Riferimento: le sorgenti pulite, una voce per traccia ---
        print("\n-> [RIFERIMENTO] ASR sulle tracce pulite...")
        _reset_vram()
        transcriber = Transcriber(model_size=model_size, device=device, compute_type="float16")
        reference = {
            "mic": _transcribe_text(transcriber, mic_wav),
            "system": _transcribe_text(transcriber, system_wav),
        }
        print(f"   parole di riferimento: mic={_words(reference['mic'])}, "
              f"system={_words(reference['system'])}")

        # --- RAW: percorso standard sulla miscela ---
        print("\n-> [RAW] Percorso standard: miscela + diarizzazione...")
        _reset_vram()
        started = time.perf_counter()
        session = process_recording(
            mic_path=mixture_wav,
            options=PipelineOptions(
                model_size=model_size,
                device=device,
                diarize=True,
                # Canale unico con piu' voci: e' il caso "ambientale" del piano.
                audio_profile=AudioProfile.STEREO_AMBIENT.value,
                num_speakers=2,
                use_channel_hint=False,
                profile_first=False,
                analysis_mode="off",
            ),
            work_dir=root / "work_raw",
        )
        raw_time = time.perf_counter() - started
        raw_vram = _peak_vram_mb()

        grouped: dict[str, list[str]] = {}
        for segment in session.segments:
            grouped.setdefault(segment.speaker, []).append(segment.text.strip())
        raw_hypothesis = {key: " ".join(value) for key, value in grouped.items()}
        print(f"   {raw_time:.1f}s | VRAM {raw_vram:.0f} MB | voci: {len(raw_hypothesis)} | "
              f"parole: {sum(_words(text) for text in raw_hypothesis.values())}")

        # --- SEP: separazione ONNX, poi una ASR per traccia ---
        print("\n-> [SEP] Separazione ONNX + ASR per traccia...")
        _reset_vram()
        backend = OnnxSeparationBackend(
            model_path=separation_dir() / separator_name,
            chunk_seconds=chunk_seconds,
            overlap_seconds=0.5,
        )
        if not backend.is_available():
            raise SystemExit(f"Separatore non disponibile: {backend.unavailable_reason}")

        started = time.perf_counter()
        tracks = backend.separate_region(mixture, MODEL_SAMPLE_RATE, num_speakers=2)
        separation_time = time.perf_counter() - started
        print(f"   separazione: {separation_time:.1f}s per {actual:.0f}s "
              f"(RTF {separation_time / actual:.2f}x, {backend.resolve_providers()[0]})")

        asr_started = time.perf_counter()
        separated_transcriber = Transcriber(
            model_size=model_size, device=device, compute_type="float16"
        )
        sep_hypothesis = {
            f"track_{index}": _transcribe_text(
                separated_transcriber, _write(root / f"track_{index}.wav", track)
            )
            for index, track in enumerate(tracks)
        }
        sep_asr_time = time.perf_counter() - asr_started
        sep_time = separation_time + sep_asr_time
        sep_vram = _peak_vram_mb()
        print(f"   {sep_time:.1f}s totali | VRAM {sep_vram:.0f} MB | tracce: {len(tracks)} | "
              f"parole: {sum(_words(text) for text in sep_hypothesis.values())}")

        raw_score = cpwer(reference, raw_hypothesis)
        sep_score = cpwer(reference, sep_hypothesis)

        return {
            "window_seconds": actual,
            "model_size": model_size,
            "device": device,
            "chunk_seconds": chunk_seconds,
            "separator": separator_name,
            "reference_words": {key: _words(text) for key, text in reference.items()},
            "raw": {
                "wall_time_seconds": round(raw_time, 2),
                "peak_vram_mb": round(raw_vram, 1),
                "speakers": len(raw_hypothesis),
                "words": sum(_words(text) for text in raw_hypothesis.values()),
                "cpwer": raw_score,
            },
            "separated": {
                "wall_time_seconds": round(sep_time, 2),
                "separation_seconds": round(separation_time, 2),
                "asr_seconds": round(sep_asr_time, 2),
                "real_time_factor": round(separation_time / actual, 3),
                "peak_vram_mb": round(sep_vram, 1),
                "tracks": len(tracks),
                "words": sum(_words(text) for text in sep_hypothesis.values()),
                "cpwer": sep_score,
                # mapping di cpwer: riferimento -> traccia ipotizzata
                "per_track_wer": {
                    ref_key: wer(reference[ref_key], sep_hypothesis[track_key])["wer"]
                    for ref_key, track_key in (sep_score.get("mapping") or {}).items()
                    if track_key in sep_hypothesis and ref_key in reference
                },
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic", default=DEFAULT_MIC)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--separator", default="sepformer_whamr16k.onnx",
                        help="Nome del file .onnx dentro la cartella dei separatori")
    parser.add_argument("--out", default="artifacts/benchmark_speaker_separation.json")
    args = parser.parse_args()

    result = run(
        Path(args.mic),
        Path(args.system),
        args.seconds,
        args.model_size,
        args.device,
        args.chunk_seconds,
        args.separator,
    )

    raw_cpwer = result["raw"]["cpwer"].get("cpwer")
    sep_cpwer = result["separated"]["cpwer"].get("cpwer")
    print("\n" + "=" * 60)
    print("  ESITO")
    print("=" * 60)
    print(f"  cpWER RAW (standard):  {raw_cpwer}")
    print(f"  cpWER SEP (separato):  {sep_cpwer}")
    if isinstance(raw_cpwer, float) and isinstance(sep_cpwer, float):
        delta = raw_cpwer - sep_cpwer
        verdict = "SEP migliore" if delta > 0 else "RAW migliore o pari"
        print(f"  Differenza: {delta:+.4f} -> {verdict}")
        result["verdict"] = {"delta_cpwer": round(delta, 4), "separation_better": delta > 0}

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Risultati salvati in: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
