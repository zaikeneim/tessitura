"""Riconciliatore conservativo dei candidati di trascrizione (Piano 2026-09-21 §6.D).

Responsabilità:
- Confronto e fusione di candidati della stessa sorgente e dello stesso parlante su regioni compatibili
- De-duplicazione delle parole introdotte esclusivamente dai margini di contesto
- Preservazione rigorosa del parlato realmente simultaneo (speaker diversi o stream diversi)
- Preservazione delle ripetizioni reali ("no, no", "sì, sì")
- Nessuna cancellazione distruttiva basata solo su timestamp
"""

from __future__ import annotations

import difflib
from copy import deepcopy
from typing import Any

from ..models.session import Segment, Word
from ..models.transcript_artifacts import Utterance


def reconcile_margin_overlap(
    left_words: list[Word],
    right_words: list[Word],
    overlap_s: float,
) -> tuple[list[Word], bool]:
    """Riconcilia parole duplicate ai confini di due finestre adiacenti dello stesso parlante.

    Se le parole di chiusura del blocco sinistro coincidono temporalmente e lessicalmente
    con le parole di apertura del blocco destro a causa del margine di contesto,
    elimina il duplicato conservando la parola con probabilità maggiore.
    """
    if not left_words or not right_words or overlap_s <= 0.05:
        return left_words + right_words, False

    # Trova parole nel margine di sovrapposizione
    left_overlap = [w for w in left_words if w.end >= right_words[0].start - 0.15]
    right_overlap = [w for w in right_words if w.start <= left_words[-1].end + 0.15]

    if not left_overlap or not right_overlap:
        return left_words + right_words, False

    left_tokens = [w.text.strip().casefold() for w in left_overlap]
    right_tokens = [w.text.strip().casefold() for w in right_overlap]

    # Cerca la sovrapposizione massima di token identici
    matcher = difflib.SequenceMatcher(None, left_tokens, right_tokens)
    match = matcher.find_longest_match(0, len(left_tokens), 0, len(right_tokens))

    if match.size >= 1:
        # Verifica che il match sia temporalmente coerente (duplicato da margine)
        l_idx = len(left_words) - len(left_overlap) + match.a
        r_idx = match.b + match.size
        # Unisci: mantieni la porzione sinistra fino a l_idx + match.size,
        # poi la porzione destra da r_idx in poi
        reconciled = left_words[: l_idx + match.size] + right_words[r_idx:]
        return reconciled, True

    return left_words + right_words, False


def reconcile_segments(segments: list[Segment]) -> list[Segment]:
    """Riconcilia una lista di segmenti eliminando artefatti da margine senza toccare overlap."""
    if not segments or len(segments) < 2:
        return segments

    # La fusione riscrive testo e confini del segmento che assorbe: senza copia
    # li riscriverebbe negli oggetti del chiamante, che spesso tiene la versione
    # precedente per confronto o per annullare.
    segments = deepcopy(segments)

    # Raggruppa per speaker preservando l'ordine
    by_speaker: dict[str, list[Segment]] = {}
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        by_speaker.setdefault(seg.speaker, []).append(seg)

    reconciled_all: list[Segment] = []

    for spk, spk_segs in by_speaker.items():
        if len(spk_segs) <= 1:
            reconciled_all.extend(spk_segs)
            continue

        merged: list[Segment] = [spk_segs[0]]
        for seg in spk_segs[1:]:
            prev = merged[-1]
            ov = max(0.0, min(prev.end, seg.end) - max(prev.start, seg.start))

            # Se i segmenti si sovrappongono o sono contigui con gap minimo e parole duplicate
            if ov > 0.05 and prev.words and seg.words:
                combined_words, deduped = reconcile_margin_overlap(prev.words, seg.words, ov)
                if deduped:
                    # Aggiorna il segmento precedente con il testo e le parole fuse
                    prev.end = max(prev.end, seg.end)
                    prev.words = combined_words
                    prev.text = " ".join(w.text for w in combined_words).strip()
                    continue

            merged.append(seg)

        reconciled_all.extend(merged)

    # Riordina cronologicamente per start
    reconciled_all.sort(key=lambda s: (s.start, s.end))
    return reconciled_all


def reconcile_utterances(utterances: list[Utterance]) -> list[Utterance]:
    """Riconcilia la lista di Utterance preservando stream_id e overlap_group_id."""
    if not utterances or len(utterances) < 2:
        return utterances

    # Stesso motivo di reconcile_segments: qui si riscrivono anche hypothesis_ids,
    # una lista annidata, che senza copia resterebbe condivisa col chiamante.
    utterances = deepcopy(utterances)

    by_speaker: dict[str, list[Utterance]] = {}
    for u in sorted(utterances, key=lambda item: (item.start_s, item.end_s)):
        by_speaker.setdefault(u.speaker, []).append(u)

    reconciled_all: list[Utterance] = []

    for spk, spk_utts in by_speaker.items():
        if len(spk_utts) <= 1:
            reconciled_all.extend(spk_utts)
            continue

        merged: list[Utterance] = [spk_utts[0]]
        for u in spk_utts[1:]:
            prev = merged[-1]
            ov = max(0.0, min(prev.end_s, u.end_s) - max(prev.start_s, u.start_s))

            if ov > 0.05 and prev.words and u.words:
                combined_words, deduped = reconcile_margin_overlap(prev.words, u.words, ov)
                if deduped:
                    prev.end_s = max(prev.end_s, u.end_s)
                    prev.words = combined_words
                    prev.text = " ".join(w.text for w in combined_words).strip()
                    if u.hypothesis_ids:
                        for hid in u.hypothesis_ids:
                            if hid not in prev.hypothesis_ids:
                                prev.hypothesis_ids.append(hid)
                    continue

            merged.append(u)

        reconciled_all.extend(merged)

    reconciled_all.sort(key=lambda item: (item.start_s, item.end_s))
    return reconciled_all
