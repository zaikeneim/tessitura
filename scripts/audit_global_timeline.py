"""Run the full recording audit without replacing or re-running cached models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.bootstrap import add_conda_dll_dirs, use_bundled_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "global-timeline")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    add_conda_dll_dirs()
    use_bundled_models()
    from src.processing.global_workflow import run_global_audit
    run_global_audit(args.audio, args.checkpoint_dir, args.output, embeddings=args.embeddings, device=args.device,
                     progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
