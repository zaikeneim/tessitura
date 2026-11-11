"""Misura la frammentazione della trascrizione su sessioni salvate.

Metrica primaria: **ping-pong**. Segmenti consecutivi che alternano parlante su
timestamp praticamente contigui, cioe' una frase unica spezzata dal merge invece
che un dialogo reale. Misurato al 9% sulle sessioni di `data/sessions` prima
della correzione dei confini di segmento.

Metriche di contorno: frammenti <=3 parole e segmenti senza chiusura, utili per
vedere se una correzione sposta il difetto invece di risolverlo.

Uso:
    python -m scripts.measure_fragmentation                # tutte le sessioni
    python -m scripts.measure_fragmentation --json         # output per la CI
    python -m scripts.measure_fragmentation FILE [FILE...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TERMINAL = (".", "?", "!", "…")

# Una catena piu' corta di cosi' e' un normale scambio di battute.
MIN_CHAIN = 4
# Sopra questo stacco le due voci non si stanno contendendo la stessa frase.
MAX_GAP_S = 0.35


def chains(segments: list[dict]) -> list[list[dict]]:
    """Catene di segmenti che alternano parlante su tempi contigui."""
    ordered = sorted(
        (s for s in segments if s.get("text", "").strip()),
        key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)),
    )
    found: list[list[dict]] = []
    current: list[dict] = ordered[:1]
    for previous, segment in zip(ordered, ordered[1:]):
        contiguous = segment.get("start", 0.0) - previous.get("end", 0.0) <= MAX_GAP_S
        if segment.get("speaker") != previous.get("speaker") and contiguous:
            current.append(segment)
            continue
        if len(current) >= MIN_CHAIN:
            found.append(current)
        current = [segment]
    if len(current) >= MIN_CHAIN:
        found.append(current)
    return found


def measure(path: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = [s for s in payload.get("segments", []) if s.get("text", "").strip()]
    if not segments:
        return None
    found = chains(segments)
    trapped = sum(len(chain) for chain in found)
    return {
        "session": path.name,
        "segments": len(segments),
        "chains": len(found),
        "segments_in_chains": trapped,
        "pingpong_pct": round(100.0 * trapped / len(segments), 1),
        "longest_chain": max((len(c) for c in found), default=0),
        "fragments_pct": round(
            100.0 * sum(1 for s in segments if len(s["text"].split()) <= 3) / len(segments), 1
        ),
        "unterminated_pct": round(
            100.0
            * sum(1 for s in segments if not s["text"].strip().endswith(TERMINAL))
            / len(segments),
            1,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", help="file di sessione; vuoto = data/sessions/*.json")
    parser.add_argument("--json", action="store_true", help="output JSON per la CI")
    args = parser.parse_args()

    paths = [Path(p) for p in args.sessions] or sorted(Path("data/sessions").glob("*.json"))
    rows = [row for row in (measure(p) for p in paths if p.is_file()) if row]
    if not rows:
        print("Nessuna sessione con segmenti trovata.")
        return 1

    total_segments = sum(r["segments"] for r in rows)
    total_trapped = sum(r["segments_in_chains"] for r in rows)
    overall = round(100.0 * total_trapped / total_segments, 1)

    if args.json:
        print(json.dumps({"sessions": rows, "pingpong_pct": overall}, indent=2))
        return 0

    print(f"{'sessione':<26}{'segm.':>7}{'catene':>8}{'in catena':>11}{'ping-pong':>11}{'framm.':>9}")
    for row in rows:
        print(
            f"{row['session']:<26}{row['segments']:>7}{row['chains']:>8}"
            f"{row['segments_in_chains']:>11}{row['pingpong_pct']:>10}%{row['fragments_pct']:>8}%"
        )
    print(f"\nPing-pong complessivo: {total_trapped}/{total_segments} = {overall}%")
    print(f"Catena piu' lunga: {max(r['longest_chain'] for r in rows)} segmenti")
    return 0


if __name__ == "__main__":
    sys.exit(main())
