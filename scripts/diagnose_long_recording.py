"""Scan a complete imported recording and export representative lossless samples.

Example: python -m scripts.diagnose_long_recording recording.mp3 --experiments 2
Private artifacts default to Tessitura's user cache, outside the installation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--session", type=Path, help="Use cached session ASR/diarization hints")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=30.)
    parser.add_argument("--context-seconds", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--experiments", type=int, default=0, help="Bounded regions; fixed ASR, 4 variants per native channel")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--language", default="it")
    parser.add_argument("--separator", help="Optional verified local ONNX separator model name")
    args = parser.parse_args()
    from src.bootstrap import add_conda_dll_dirs, use_bundled_models
    add_conda_dll_dirs()
    use_bundled_models()
    from src.processing.long_diagnostics import DiagnosticOptions, diagnose_recording
    report = diagnose_recording(args.source, output_dir=args.out, session_path=args.session,
                               options=DiagnosticOptions(segment_s=args.segment_seconds,
                                                         context_s=args.context_seconds, seed=args.seed))
    value = json.loads(report.read_text(encoding="utf-8"))
    result = {"report": str(report), "duration_s": value["scan"]["decoded_duration_s"],
              "channels": value["scan"]["channels"], "regions": len(value["regions"]),
              "decode_errors": value["scan"]["decode_errors"]}
    if args.experiments:
        from src.processing.diagnostic_experiments import run_diagnostic_experiments
        result["experiments"] = str(run_diagnostic_experiments(
            report, count=args.experiments, model=args.model, device=args.device,
            language=args.language or None, separator=args.separator))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
