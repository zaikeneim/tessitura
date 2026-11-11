"""Elabora un file audio da riga di comando: trascrizione + diarizzazione -> sessione.

Uso (dalla root, env tessitura attivo):
    python -m scripts.run_pipeline "audio.mp3" --profile stereo_ambient \
        --model large-v3 --speakers 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import (  # noqa: E402
    add_conda_dll_dirs,
    install_system_certs,
    silence_noisy_warnings,
)


def _progress(msg: str, frac: float) -> None:
    print(f"[{int(frac * 100):3d}%] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trascrivi e diarizza un file audio")
    parser.add_argument("audio")
    parser.add_argument(
        "--profile",
        required=True,
        choices=["mic_system", "stereo_ambient"],
        help="profilo obbligatorio: mic_system oppure stereo_ambient",
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--prompt", default=None,
                        help="nomi/termini da fornire come glossario a Whisper")
    parser.add_argument("--speakers", type=int, default=0, help="numero voci (0 = auto)")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--speakers", type=int, default=0, help="numero voci (0 = auto)")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--denoise-strength", type=float, default=0.5,
                        help="0..1, quanto rumore togliere (>0.6 inizia a mangiare voce)")
    parser.add_argument("--quality-retries", type=int, default=24,
                        help="retry mirati per segmenti poco affidabili (0 disattiva)")
    parser.add_argument("--review", choices=["safe", "off"], default="safe",
                        help="controllo attribuzione prima dell'output (default: safe)")
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("--system-audio", default=None, help="traccia audio di sistema separata")
    parser.add_argument("--strategy",
                        choices=["standard", "source_single_speaker", "speaker_isolated"],
                        default="standard",
                        help="strategia di pipeline. speaker_isolated separa acusticamente un "
                             "canale multivoce ed e' sperimentale: senza un modello di "
                             "separazione verificato ripiega sul percorso standard")
    parser.add_argument("--headphones-confirmed", action="store_true", default=True,
                        help="conferma l'uso delle cuffie per disattivare bleed check")
    parser.add_argument("--no-headphones", dest="headphones_confirmed", action="store_false",
                        help="segnala l'uso di altoparlanti (attiva bleed check)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None, help="percorso JSON di output")
    parser.add_argument("--preview", type=int, default=20, help="quante righe mostrare")
    args = parser.parse_args()

    try:  # accenti corretti in console su Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    add_conda_dll_dirs()
    install_system_certs()
    silence_noisy_warnings()

    from src import paths
    from src.models import storage
    from src.processing.audio_profile import AudioProfile
    from src.processing.pipeline import PipelineOptions, process_recording, resolve_device

    device, compute = resolve_device("auto")
    print(f"Device: {device} ({compute})", flush=True)

    options = PipelineOptions(
        denoise=not args.no_denoise,
        diarize=not args.no_diarize,
        language=args.lang,
        model_size=args.model,
        device=device,
        compute_type=compute,
        num_speakers=args.speakers or None,
        denoise_strength=args.denoise_strength,
        quality_retries=max(0, args.quality_retries),
        initial_prompt=args.prompt,
        audio_profile=AudioProfile.parse(args.profile),
        analysis_mode=args.review,
        strategy=args.strategy,
        headphones_confirmed=args.headphones_confirmed,
    )

    name = args.name or Path(args.audio).stem
    sys_audio = Path(args.system_audio) if args.system_audio else None
    session = process_recording(Path(args.audio), sys_audio, options=options,
                                session_name=name, progress=_progress)

    out = Path(args.out) if args.out else paths.sessions_dir() / f"{session.id}.json"
    storage.save_session(session, out)

    print("\n=== RISULTATO ===")
    print(f"Lingua: {session.language} | segmenti: {len(session.segments)} | "
          f"voci: {len(session.speaker_ids())} | durata: {session.duration:.0f}s")
    print(f"Sessione salvata in: {out}\n")
    for seg in session.segments[: args.preview]:
        m, s = divmod(int(seg.start), 60)
        print(f"[{m:02d}:{s:02d}] {seg.speaker}: {seg.text}")
    if len(session.segments) > args.preview:
        print(f"... (+{len(session.segments) - args.preview} segmenti)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
