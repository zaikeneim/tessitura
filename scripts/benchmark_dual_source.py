"""Benchmark preliminare A vs B (Standard Mix+Diarizzazione vs ASR Duale Indipendente).

Confronta su hardware reale (CUDA / RTX 4070):
  A: Mix delle sorgenti -> Whisper -> Diarizzazione Pyannote -> Channel hint
  B: Due passate indipendenti Transcriber.transcribe() (mic e sistema), senza diarizzazione

Finestre: 5m (300s), 15m (900s), Full (~3324s).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

import numpy as np
import soundfile as sf
import torch

from src.processing.audio_profile import AudioProfile
from src.processing import audio_utils
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.transcribe import Transcriber


def slice_wav(input_path: Path, output_path: Path, max_seconds: float) -> float:
    """Estrae i primi max_seconds dal file input_path salvandoli in output_path."""
    info = sf.info(str(input_path))
    duration = min(float(info.duration), max_seconds)
    frames_to_read = int(duration * info.samplerate)
    data, sr = sf.read(str(input_path), frames=frames_to_read, dtype="float32", always_2d=True)
    sf.write(str(output_path), data, sr)
    return duration


def run_benchmark_for_window(
    mic_source: Path,
    system_source: Path,
    duration_s: float,
    model_size: str = "medium",
    device: str = "cuda",
) -> dict:
    print(f"\n=======================================================")
    print(f"  AVVIO BENCHMARK PER FINESTRA: {duration_s:.0f} secondi ({duration_s/60:.1f} min)")
    print(f"  Modello: {model_size}, Device: {device}")
    print(f"=======================================================")

    with tempfile.TemporaryDirectory(prefix="bench_slice_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        mic_slice = tmp_path / "mic_slice.wav"
        sys_slice = tmp_path / "sys_slice.wav"

        print(f"-> Preparazione slice audio ({duration_s}s)...")
        dur_mic = slice_wav(mic_source, mic_slice, duration_s)
        dur_sys = slice_wav(system_source, sys_slice, duration_s)
        actual_dur = min(dur_mic, dur_sys)
        print(f"   Durata effettiva: mic={dur_mic:.1f}s, sys={dur_sys:.1f}s (usata: {actual_dur:.1f}s)")

        # -------------------------------------------------------------
        # METODO A: Standard (Mix -> Whisper -> Diarize pyannote)
        # -------------------------------------------------------------
        print("\n-> [METODO A] Avvio percorso standard (Mix + Pyannote)...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0_a = time.perf_counter()
        opts_a = PipelineOptions(
            model_size=model_size,
            device=device,
            diarize=True,
            audio_profile=AudioProfile.MIC_SYSTEM.value,
            use_channel_hint=True,
            profile_first=False,
            analysis_mode="off",
        )
        session_a = process_recording(
            mic_path=mic_slice,
            system_path=sys_slice,
            options=opts_a,
            work_dir=tmp_path / "work_a",
            offsets_s=[0.0, 0.0],
        )
        time_a = time.perf_counter() - t0_a
        vram_a = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        words_a = sum(len(s.text.split()) for s in session_a.segments)
        segs_a = len(session_a.segments)

        print(f"   [METODO A] Completato in {time_a:.2f}s | VRAM Picco: {vram_a:.1f} MB | Segmenti: {segs_a} | Parole: {words_a}")

        # -------------------------------------------------------------
        # METODO B: Duale Indipendente (ASR mic + ASR sys, no diarize)
        # -------------------------------------------------------------
        print("\n-> [METODO B] Avvio percorso duale indipendente (2x ASR, no diarize)...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0_b = time.perf_counter()

        # Riutilizzo singolo Transcriber sequenziale per risparmiare VRAM
        transcriber = Transcriber(model_size=model_size, device=device, compute_type="int8_float16")
        try:
            # Trascrizione MIC
            t0_b_mic = time.perf_counter()
            segments_mic, lang_mic = transcriber.transcribe(mic_slice, vad_filter=True)
            time_b_mic = time.perf_counter() - t0_b_mic

            # Trascrizione SYSTEM
            t0_b_sys = time.perf_counter()
            segments_sys, lang_sys = transcriber.transcribe(sys_slice, vad_filter=True)
            time_b_sys = time.perf_counter() - t0_b_sys
        finally:
            transcriber.close()

        time_b = time.perf_counter() - t0_b
        vram_b = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        words_b_mic = sum(len(s.text.split()) for s in segments_mic)
        words_b_sys = sum(len(s.text.split()) for s in segments_sys)
        words_b = words_b_mic + words_b_sys
        segs_b = len(segments_mic) + len(segments_sys)

        print(f"   [METODO B] Completato in {time_b:.2f}s (mic: {time_b_mic:.2f}s, sys: {time_b_sys:.2f}s) | VRAM Picco: {vram_b:.1f} MB | Segmenti: {segs_b} | Parole: {words_b} (mic: {words_b_mic}, sys: {words_b_sys})")

        speedup_b_vs_a = time_a / time_b if time_b > 0 else 0.0
        ratio_time = time_b / time_a if time_a > 0 else 0.0

        print(f"\n-> RIEPILOGO FINESTRA {duration_s:.0f}s:")
        print(f"   Tempo A (Standard):     {time_a:.2f}s")
        print(f"   Tempo B (Duale):        {time_b:.2f}s  (Rapporto T_B / T_A: {ratio_time:.2f}x, Speedup: {speedup_b_vs_a:.2f}x)")
        print(f"   VRAM A:                 {vram_a:.1f} MB")
        print(f"   VRAM B:                 {vram_b:.1f} MB")
        print(f"   Parole A vs B:          {words_a} vs {words_b} (diff: {words_b - words_a})")

        return {
            "requested_seconds": duration_s,
            "actual_duration_seconds": actual_dur,
            "method_a_standard": {
                "wall_time_seconds": round(time_a, 3),
                "peak_vram_mb": round(vram_a, 1),
                "segment_count": segs_a,
                "word_count": words_a,
            },
            "method_b_dual": {
                "wall_time_seconds": round(time_b, 3),
                "wall_time_mic_seconds": round(time_b_mic, 3),
                "wall_time_sys_seconds": round(time_b_sys, 3),
                "peak_vram_mb": round(vram_b, 1),
                "segment_count": segs_b,
                "word_count_total": words_b,
                "word_count_mic": words_b_mic,
                "word_count_sys": words_b_sys,
            },
            "comparison": {
                "ratio_tb_over_ta": round(ratio_time, 3),
                "speedup_b_vs_a": round(speedup_b_vs_a, 3),
                "is_b_faster": time_b < time_a,
                "delta_words": words_b - words_a,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark preliminare A vs B su sorgenti duali")
    parser.add_argument("--mic", type=str, default="data/recordings/esempio_mic.wav")
    parser.add_argument("--system", type=str, default="data/recordings/esempio_system.wav")
    parser.add_argument("--windows", type=str, default="300,900", help="Finestre in secondi separate da virgola (es. 300,900,3300)")
    parser.add_argument("--model", type=str, default="medium", help="Modello whisper (small, medium, large-v3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mic_p = ROOT / args.mic
    sys_p = ROOT / args.system

    if not mic_p.is_file() or not sys_p.is_file():
        print(f"ERRORE: file non trovati: {mic_p} / {sys_p}", file=sys.stderr)
        sys.exit(1)

    windows = [float(w.strip()) for w in args.windows.split(",") if w.strip()]
    results = []

    for w in windows:
        res = run_benchmark_for_window(mic_p, sys_p, w, model_size=args.model, device=args.device)
        results.append(res)

    output_json = ROOT / "artifacts/benchmark_dual_source.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_size": args.model,
        "mic_file": str(mic_p.relative_to(ROOT)),
        "sys_file": str(sys_p.relative_to(ROOT)),
        "runs": results,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Risultati salvati in: {output_json}")


if __name__ == "__main__":
    main()
