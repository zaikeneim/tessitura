"""Benchmark local recovery on a source-bound baseline; never infer accuracy from word counts."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--separator", default="mossformer2_ss_16k",
                        choices=("mossformer2_ss_16k", "sepformer_whamr16k"))
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--max-audio", type=float, default=120.)
    parser.add_argument("--max-runtime", type=float, default=600.)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from src.bootstrap import add_conda_dll_dirs, use_bundled_models
    add_conda_dll_dirs()
    use_bundled_models()
    from src.models.storage import load_session, save_session
    from src.processing.checkpoints import file_digest
    from src.processing.global_workflow import write_json
    from src.processing.progress import RECOVERY_STAGES, StageReporter
    from src.processing.region_recovery import recover_session_regions
    from src.processing.resources import acquire_inference
    from src.processing.separation import separate_region
    from src.processing.separation_backends import OnnxSeparationBackend
    import onnxruntime as ort
    import psutil

    digest = file_digest(args.audio)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if baseline_report.get("source_sha256") != digest:
        raise ValueError("Il report baseline non corrisponde all'hash dell'audio.")
    session = load_session(args.session)
    bound_hash = session.provenance.get("source_sha256")
    if bound_hash is not None and bound_hash != digest:
        raise ValueError("La sessione baseline appartiene a un altro audio.")
    if bound_hash is None and Path(baseline_report.get("session", "")).resolve() != args.session.resolve():
        raise ValueError("Il report non identifica questa sessione baseline.")
    baseline = copy.deepcopy([e.to_dict() for e in session.utterances])
    backend = OnnxSeparationBackend(model_name=args.separator)
    if not backend.is_available():
        raise RuntimeError(f"Separatore non disponibile: {backend.unavailable_reason}")
    # A fresh native session per region keeps separation and ASR sequential.
    def separator(audio, view, region, **kwargs):
        return separate_region(audio, view, region, backend=OnnxSeparationBackend(model_name=args.separator), **kwargs)
    peaks = {"rss_bytes": 0, "device_vram_mib": 0, "samples": 0}
    stop = threading.Event()
    process = psutil.Process()
    def sample():
        while not stop.is_set():
            peaks["rss_bytes"] = max(peaks["rss_bytes"], process.memory_info().rss)
            try:
                run = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if run.returncode == 0:
                    peaks["device_vram_mib"] = max(peaks["device_vram_mib"], int(run.stdout.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            peaks["samples"] += 1
            stop.wait(.5)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.monotonic()
    stages = StageReporter(RECOVERY_STAGES, progress=lambda message, _: print(message, flush=True))
    try:
        with acquire_inference():
            report = recover_session_regions(args.audio, session, args.output, stages=stages,
                model_size=args.model, device=args.device, separator=separator, stratified=True,
                max_regions=args.max_regions, max_audio_s=args.max_audio, max_elapsed_s=args.max_runtime)
    finally:
        stop.set()
        monitor.join(timeout=4)
    if baseline != [e.to_dict() for e in session.utterances] or file_digest(args.audio) != digest:
        raise RuntimeError("Baseline o audio modificati: benchmark non valido.")
    args.output.mkdir(parents=True, exist_ok=True)
    session.provenance["processing_stages"] = session.provenance.get("processing_stages", []) + stages.snapshot()
    save_session(session, args.output / "review-session.json")
    metrics = {"status": "unavailable", "reason": "Nessun riferimento completo umanamente verificato"}
    if args.reference:
        from src.processing.verified_evaluation import evaluate_verified_reference
        try:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            metrics = {"status": "baseline_only", "baseline": evaluate_verified_reference(reference, session.utterances, source_sha256=digest)}
        except ValueError as exc:
            metrics = {"status": "unavailable", "reason": str(exc)}
    template = {"source_sha256": digest, "verification_level": "unverified_candidate", "annotation_complete": False,
                "reviewed_by": "", "regions": [{"start_s": r["start_s"], "end_s": r["end_s"],
                    "problem_labels": r["labels"], "annotation_complete": False, "events": []} for r in report["plan"]["regions"]]}
    write_json(args.output / "reference-review-template.json", template)
    result = {"source_sha256": digest, "baseline_unchanged": True, "baseline_events": len(baseline),
              "baseline_words": sum(len(e["words"]) for e in baseline), "elapsed_s": time.monotonic() - started,
              "peaks_sampled": peaks, "memory_scope": "Process RSS; total device VRAM, sampled every ~0.5 s",
              "onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(),
              "selected_providers": backend.resolve_providers(), "separation_model": args.separator,
              "accuracy": metrics, "recovery": report}
    write_json(args.output / "benchmark.json", result)
    print(json.dumps({"counts": report["counts"], "elapsed_s": result["elapsed_s"],
                      "peaks_sampled": peaks, "accuracy": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
