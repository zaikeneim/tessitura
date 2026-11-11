"""Esporta un separatore SpeechBrain in ONNX e ne registra l'impronta nel manifest.

La pipeline carica i pesi di separazione solo se compaiono nel manifest con lo
SHA-256 corrispondente (regola offline del piano §6.4). Questo script e' il
percorso dichiarato per produrre quei file: scarica il checkpoint ufficiale,
traccia il grafo encoder -> masknet -> decoder, scrive l'ONNX accanto al
manifest aggiornato e verifica l'esito ricaricandolo dal backend di produzione.

Esecuzione:
    python -m scripts.export_separation_onnx
    python -m scripts.export_separation_onnx --model speechbrain/sepformer-whamr16k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import install_system_certs  # noqa: E402
from src.processing.separation_backends import (  # noqa: E402
    MANIFEST_NAME,
    MODEL_SAMPLE_RATE,
    OnnxSeparationBackend,
    file_sha256,
    separation_dir,
)

DEFAULT_MODEL = "speechbrain/sepformer-whamr16k"


class _SeparatorGraph(torch.nn.Module):
    """Percorso di inferenza del separatore, senza la logica di I/O di SpeechBrain."""

    def __init__(self, encoder, masknet, decoder, num_speakers: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.masknet = masknet
        self.decoder = decoder
        self.num_speakers = num_speakers

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(mix)
        masks = self.masknet(encoded)
        stacked = torch.stack([encoded] * self.num_speakers) * masks
        sources = torch.cat(
            [self.decoder(stacked[i]).unsqueeze(-1) for i in range(self.num_speakers)],
            dim=-1,
        )
        # L'encoder convoluzionale altera la lunghezza: si riallinea al mix.
        length = mix.size(1)
        if sources.size(1) < length:
            sources = torch.nn.functional.pad(sources, (0, 0, 0, length - sources.size(1)))
        return sources[:, :length, :].permute(0, 2, 1)


def load_separator(model_id: str, cache_dir: Path):
    """Scarica il checkpoint con l'hub e lo apre da disco.

    Il fetching interno di SpeechBrain 1.1 non regge huggingface_hub 1.x (chiude
    il client httpx prima della richiesta), quindi lo scaricamento passa da qui e
    a SpeechBrain arriva una cartella locale gia' pronta.
    """
    from huggingface_hub import snapshot_download
    from speechbrain.inference.separation import SepformerSeparation

    local_dir = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
        )
    )
    print(f"   checkpoint locale: {local_dir}")
    return SepformerSeparation.from_hparams(
        source=str(local_dir),
        savedir=str(cache_dir),
        run_opts={"device": "cpu"},
    )


def export(model_id: str, destination: Path, trace_seconds: float, opset: int) -> Path:
    cache_dir = destination.parent / "_speechbrain" / model_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> Caricamento {model_id} (cache: {cache_dir})")
    separator = load_separator(model_id, cache_dir)
    num_speakers = int(getattr(separator.hparams, "num_spks", 2))
    graph = _SeparatorGraph(
        separator.mods.encoder, separator.mods.masknet, separator.mods.decoder, num_speakers
    ).eval()

    sample = torch.randn(1, int(trace_seconds * MODEL_SAMPLE_RATE))
    with torch.no_grad():
        reference = graph(sample)
    print(f"   parlanti: {num_speakers} | uscita tracciata: {tuple(reference.shape)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Export ONNX (opset {opset}) verso {destination.name}")
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (sample,),
            str(destination),
            input_names=["mix"],
            output_names=["sources"],
            dynamic_axes={"mix": {1: "samples"}, "sources": {2: "samples"}},
            opset_version=opset,
            dynamo=False,
        )
    return destination


def update_manifest(
    model_path: Path,
    model_id: str,
    num_speakers: int,
    max_input_seconds: float | None = None,
) -> dict:
    manifest_path = model_path.parent / MANIFEST_NAME
    payload = {"models": {}}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing.get("models"), dict):
                payload = existing
        except (OSError, ValueError):
            pass

    entry = {
        "sha256": file_sha256(model_path),
        "source": model_id,
        "sample_rate": MODEL_SAMPLE_RATE,
        "num_speakers": num_speakers,
        "exported_by": "scripts/export_separation_onnx.py",
    }
    if max_input_seconds:
        # Il grafo non regge ingressi piu' lunghi di cosi': il backend ci si attiene.
        entry["max_input_seconds"] = float(max_input_seconds)
    payload["models"][model_path.name] = entry
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload["models"][model_path.name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Checkpoint SpeechBrain")
    parser.add_argument("--out", default=None, help="Percorso del file .onnx")
    parser.add_argument("--trace-seconds", type=float, default=5.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--register",
        default=None,
        help="Registra nel manifest un .onnx gia' prodotto altrove, senza riesportarlo. "
        "Serve per i modelli il cui pacchetto e' incompatibile con questo ambiente "
        "e va esportato in un venv isolato.",
    )
    parser.add_argument(
        "--max-input-seconds",
        type=float,
        default=None,
        help="Lunghezza massima d'ingresso che il grafo regge davvero. Serve agli export "
        "che congelano la lunghezza tracciata negli embedding posizionali: il backend "
        "non supera mai questo valore.",
    )
    args = parser.parse_args()

    install_system_certs()

    if args.register:
        destination = Path(args.register)
        if not destination.exists():
            print(f"[FAIL] File assente: {destination}")
            return 1
    else:
        destination = (
            Path(args.out)
            if args.out
            else separation_dir() / f"{args.model.split('/')[-1].replace('-', '_')}.onnx"
        )
        export(args.model, destination, args.trace_seconds, args.opset)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"   file: {destination} ({size_mb:.1f} MB)")

    backend = OnnxSeparationBackend(model_path=destination)
    entry = update_manifest(destination, args.model, 2, args.max_input_seconds)
    print(f"-> Manifest aggiornato: sha256={entry['sha256'][:16]}...")

    if not backend.is_available():
        print(f"[FAIL] Backend non disponibile dopo l'export: {backend.unavailable_reason}")
        return 1

    probe = (0.1 * np.random.default_rng(0).standard_normal(MODEL_SAMPLE_RATE * 3)).astype(
        np.float32
    )
    tracks = backend.separate_region(probe, MODEL_SAMPLE_RATE)
    if len(tracks) < 2:
        print(f"[FAIL] L'inferenza ha prodotto {len(tracks)} tracce")
        return 1

    print(f"[OK] {len(tracks)} tracce, {len(tracks[0])} campioni, provider "
          f"{backend.resolve_providers()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
