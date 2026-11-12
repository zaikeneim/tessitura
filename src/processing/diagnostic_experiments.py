"""Bounded fixed-model comparisons; outputs are listening candidates, not scores."""
from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import TARGET_SR, normalize
from .checkpoints import file_digest
from .long_diagnostics import _write_json
from .progress import StageReporter


def run_diagnostic_experiments(report_path: str | Path, *, count: int = 2,
                               model: str = "large-v3", device: str = "cuda",
                               language: str | None = "it", separator: str | None = None,
                               progress=None, on_stage=None) -> Path:
    """Compare unchanged ASR on raw, current, conservative and no-VAD samples.

    Every native channel is tested independently. Separation outputs retain
    overlapping absolute timestamps and require human review before acceptance.
    Cached results include source, model, and effective experiment settings.
    """
    stages = StageReporter([("prepare", "Verifica campioni e modelli locali"),
                            ("compare", "Confronto varianti per canale"),
                            ("save", "Salvataggio confronto")], progress, on_stage)
    with stages.guard():
        stages.start("prepare")
        if not 1 <= count <= 16:
            raise ValueError("Experiments must be bounded to 1–16 regions")
        from .denoise import denoise_array, denoise_file
        from .transcribe import Transcriber, _ensure_model_path
        from scipy.signal import resample_poly

        report_path = Path(report_path).resolve(strict=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        root = report_path.parent / "experiments"
        root.mkdir(exist_ok=True)
        model_path = _ensure_model_path(model, None, local_only=True)
        model_hash = file_digest(Path(model_path) / "model.bin")
        configuration = {"version": 2, "source_sha256": report["source_sha256"],
                         "model": str(model_path), "model_sha256": model_hash,
                         "device": device, "compute_type": "int8_float16" if device == "cuda" else "int8",
                         "language": language, "quality_retries": 0, "separator": separator,
                         "variants": {"raw": {}, "current": {"denoise_strength": .5, "pipeline": "normalize_pcm16_then_denoise_file"},
                                      "conservative": {"denoise_strength": .25}, "raw_no_vad": {"vad": False}}}
        separator_backend = None
        if separator:
            from .separation_backends import OnnxSeparationBackend
            separator_backend = OnnxSeparationBackend(model_name=separator)
            if not separator_backend.is_available():
                raise RuntimeError(f"Local verified separator unavailable: {separator_backend.unavailable_reason}")
            configuration["separator_sha256"] = file_digest(separator_backend.model_path)
        transcriber = None
        results = []
        # Spread the bounded experiment set across the selected timeline.
        regions = report["regions"]
        indices = sorted({min(len(regions) - 1, int(i * len(regions) / count))
                          for i in range(min(count, len(regions)))})
        total_channels = sum(sf.info(report_path.parent / regions[i]["audio_file"]).channels for i in indices)
        completed_channels = 0
        stages.complete("prepare")
        if total_channels:
            stages.start("compare")
        else:
            stages.skip("compare", "Nessun campione disponibile")
        try:
            for index in indices:
                region = regions[index]
                samples, rate = sf.read(report_path.parent / region["audio_file"], dtype="float32", always_2d=True)
                for channel in range(samples.shape[1]):
                    common = {**configuration, "region_id": region["region_id"],
                              "sample_sha256": file_digest(report_path.parent / region["audio_file"]),
                              "sample_origin_s": region["sample_origin_s"], "channel": channel}
                    complete_path = root / f"{region['region_id']}-channel-{channel}-complete.json"
                    if complete_path.is_file():
                        cached_channel = json.loads(complete_path.read_text(encoding="utf-8"))
                        if cached_channel.get("configuration") == common and all(
                                Path(r["audio"]).is_file() for r in cached_channel.get("results", [])):
                            results.extend(cached_channel["results"])
                            completed_channels += 1
                            stages.update("compare", completed_channels / total_channels, "Canale dalla cache verificata")
                            continue
                    channel_result_start = len(results)
                    signal = samples[:, channel]
                    if rate != TARGET_SR:
                        divisor = math.gcd(rate, TARGET_SR)
                        signal = resample_poly(signal, TARGET_SR // divisor, rate // divisor).astype(np.float32)
                    prepared = root / f"{region['region_id']}-channel-{channel}-normalized.wav"
                    sf.write(prepared, normalize(signal), TARGET_SR, subtype="PCM_16")
                    current_path = root / f"{region['region_id']}-channel-{channel}-pipeline.wav"
                    denoise_file(prepared, current_path, prop_decrease=.5)
                    current_signal, _ = sf.read(current_path, dtype="float32")
                    variants = {"raw": signal, "current": current_signal,
                                "conservative": denoise_array(signal, TARGET_SR, prop_decrease=.25),
                                "raw_no_vad": signal}
                    if separator_backend:
                        separated = separator_backend.separate_region(signal, TARGET_SR)
                        variants.update({f"separated_source_{i}": track for i, track in enumerate(separated)})
                    for variant_index, (variant, treated) in enumerate(variants.items()):
                        label = f"{region['region_id']} · canale {channel + 1} · {variant}"
                        stages.update("compare", (completed_channels + variant_index / len(variants)) / total_channels, label)
                        target = root / f"{region['region_id']}-channel-{channel}-{variant}.json"
                        config = {**common, "variant": variant}
                        if target.is_file():
                            cached = json.loads(target.read_text(encoding="utf-8"))
                            if cached.get("configuration") == config and "error" not in cached and Path(cached["audio"]).is_file():
                                results.append(cached)
                                continue
                        audio_path = target.with_suffix(".wav")
                        sf.write(audio_path, treated, TARGET_SR, subtype="FLOAT")
                        started = time.perf_counter()
                        if transcriber is None:
                            transcriber = Transcriber(model_size=str(model_path), device=device,
                                                      compute_type=configuration["compute_type"])
                        value = {"configuration": config, "audio": str(audio_path),
                                 "requires_human_review": True, "quality_metrics": None}
                        try:
                            segments, detected_language = transcriber.transcribe(
                                audio_path, language=language, vad_filter=variant != "raw_no_vad", quality_retries=0,
                                on_progress=lambda done, total: stages.update("compare",
                                    (completed_channels + (variant_index + (done / total if total else 0)) / len(variants)) / total_channels,
                                    f"{label}: {done:.1f}/{total:.1f} s"),
                                on_status=stages.asr_status("compare"))
                            origin = region["sample_origin_s"]
                            rows = []
                            for segment in segments:
                                row = asdict(segment)
                                row["start"] += origin
                                row["end"] += origin
                                row["words"] = [(a + origin, b + origin, text, probability)
                                                for a, b, text, probability in segment.words]
                                rows.append(row)
                            value.update({"segments": rows, "language": detected_language,
                                          "candidate_words": sum(len(s.text.split()) for s in segments),
                                          "runtime_s": time.perf_counter() - started})
                        except Exception as exc:
                            value.update({"error": str(exc), "runtime_s": time.perf_counter() - started})
                        _write_json(target, value)
                        results.append(value)
                    completed_channels += 1
                    stages.update("compare", completed_channels / total_channels, label)
                    channel_results = results[channel_result_start:]
                    if channel_results and all("error" not in r for r in channel_results):
                        _write_json(complete_path, {"configuration": common, "results": channel_results})
        finally:
            if transcriber is not None:
                transcriber.close()
        errors = sum("error" in item for item in results)
        if total_channels:
            stages.complete("compare", warning=f"{errors} varianti fallite; consulta il rapporto" if errors else None)
        stages.start("save")
        result_path = root / "comparison.json"
        selection = {"region_ids": [regions[i]["region_id"] for i in indices],
                     "diagnostic_configuration": report.get("configuration")}
        comparison = {"configuration": configuration, "selection": selection, "results": results,
                                 "quality_metrics": None,
                                 "interpretation": "Word counts are diagnostic candidates. More words can mean hallucinations. "
                                                   "Do not choose a winning profile without reviewed reference speech. "
                                                   "Separated sources may duplicate words or invent speech; retain overlapping timestamps."}
        config_hash = hashlib.sha256(json.dumps([configuration, selection], sort_keys=True).encode()).hexdigest()[:16]
        _write_json(root / f"comparison-{config_hash}.json", comparison)
        _write_json(result_path, comparison)
        stages.complete("save")
        stages.assert_complete()
        return result_path
