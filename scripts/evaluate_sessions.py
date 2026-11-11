"""Evaluate a local session JSON against a Phase 0 gold manifest.

The manifest may be inspected without a hypothesis (useful immediately after
annotation).  With ``--hypothesis`` or a manifest ``hypothesis`` path it also
reports WER, cpWER and the transparent time-weighted diarization fallback.
No audio or transcript is uploaded; MeetEval/pyannote remain optional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep the documented ``python scripts/evaluate_sessions.py`` invocation
# working as well as ``python -m scripts.evaluate_sessions``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import storage
from src.processing.evaluation import Span, cpwer, diarization_confusion, wer


def _hypothesis_path(manifest: Path, value: str | None, override: Path | None) -> Path | None:
    candidate = override or (Path(value) if value else None)
    if candidate is None:
        return None
    return candidate if candidate.is_absolute() else (manifest.parent / candidate).resolve()


def _reference_spans(data: dict) -> list[Span]:
    return [Span(float(item["start"]), float(item["end"]), str(item["speaker"]))
            for item in data.get("turns", [])
            if isinstance(item, dict)]


def _session_text_and_spans(session) -> tuple[dict[str, str], list[Span]]:
    text: dict[str, list[str]] = {}
    spans: list[Span] = []
    for segment in session.segments:
        text.setdefault(segment.speaker, []).append(segment.text)
        spans.append(Span(float(segment.start), float(segment.end), segment.speaker))
    return {speaker: " ".join(parts) for speaker, parts in text.items()}, spans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--hypothesis", type=Path,
                        help="Sessione JSON da confrontare; sovrascrive il campo del manifest")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    verification_level = data.get("annotation", {}).get(
        "verification_level",
        "imported_draft" if data.get("status") != "gold_reference_v1" else "legacy_gold_unverified"
    )
    report = {
        "corpus": data.get("corpus"),
        "status": data.get("status"),
        "verification_level": verification_level,
        "promotion_gate": (
            "BLOCKED_PENDING_HUMAN_VERIFICATION"
            if verification_level in ("imported_draft", "legacy_gold_unverified")
            else "HUMAN_VERIFIED"
        ),
        "window": data.get("window"),
        "speakers": data.get("speakers", []),
        "turns": len(data.get("turns", [])),
        "source": data.get("source", {}),
    }
    hypothesis = _hypothesis_path(args.manifest, data.get("hypothesis"), args.hypothesis)
    if hypothesis is None:
        report["evaluation"] = "pending_hypothesis"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    session = storage.load_session(hypothesis)
    actual, actual_spans = _session_text_and_spans(session)
    reference = data["reference"]
    reference_text = " ".join(item["text"] for item in data.get("turns", []))
    hypothesis_text = " ".join(segment.text for segment in session.segments)
    report.update({
        "hypothesis": str(hypothesis),
        "overall": wer(reference_text, hypothesis_text),
        "cpwer": cpwer(reference, actual),
    })
    reference_spans = _reference_spans(data)
    if reference_spans:
        overlap_allowed = "overlap" in str(data.get("annotation", {}).get("overlap_policy", ""))
        # Il riferimento usa i nomi delle persone, il sistema le etichette
        # SPEAKER_xx: senza applicare l'accoppiamento gia' calcolato da cpWER
        # ogni sovrapposizione risulterebbe una confusione, e il numero direbbe
        # soltanto che le due nomenclature sono diverse.
        inverse = {hypothesis_label: name
                   for name, hypothesis_label in report["cpwer"].get("mapping", {}).items()}
        mapped_spans = [Span(span.start, span.end, inverse.get(span.speaker, span.speaker))
                        for span in actual_spans]
        report["diarization_fallback"] = diarization_confusion(
            reference_spans, mapped_spans, allow_overlap=overlap_allowed
        )
        report["diarization_fallback"]["speaker_mapping"] = report["cpwer"].get("mapping", {})
        report["diarization_fallback"]["speaker_count_error"] = (
            len({span.speaker for span in actual_spans}) - len({span.speaker for span in reference_spans})
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
