"""Meccanismi di autocorrezione per troncamento delle frasi e sovrapposizione delle voci.

1. Troncamento delle frasi: rileva segmenti interrotti a metà frase da un cambio
   di interlocutore o dal VAD/chunking, inserisce i punti di sospensione '…' e
   impedisce che il frammento venga erroneamente fuso nella frase dell'altro.
   Per lo stesso interlocutore, ricuce le continuazioni spezzate.

2. Sovrapposizione delle voci: disintreccia le porzioni di parlato simultaneo
   segnalate da pyannote, protegge le interiezioni brevi e annota i candidati
   alternativi e i warning ('voce_sovrapposta').
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.session import Segment
    from .voice_profile import VoiceProfile

# Punteggiatura terminale standard
_TERMINAL_PUNCTUATION = (".", "?", "!", "…")

# Parole aperte italiane che indicano chiaramente una frase sospesa / interrotta
_OPEN_CLAUSE_TOKENS = {
    # congiunzioni
    "e", "ed", "ma", "però", "perché", "poiché", "mentre", "quindi", "dunque",
    "se", "o", "oppure", "che", "anche", "cioè", "ossia", "invece",
    # preposizioni semplici e articolate
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "del", "della", "dello", "dei", "degli", "delle",
    "al", "alla", "allo", "ai", "agli", "alle",
    "dal", "dalla", "dallo", "dai", "dagli", "dalle",
    "nel", "nella", "nello", "nei", "negli", "nelle",
    "sul", "sulla", "sullo", "sui", "sugli", "sulle",
    # articoli
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
}


def is_terminal_text(text: str) -> bool:
    """Restituisce True se il testo termina con punteggiatura di fine periodo."""
    clean = text.strip()
    return bool(clean and clean.endswith(_TERMINAL_PUNCTUATION) or clean.endswith("..."))


def ends_with_open_clause(text: str) -> bool:
    """Rileva se l'ultima parola o segno di un testo indica una frase lasciata a metà."""
    clean = text.strip()
    if not clean:
        return False
    if clean.endswith((",", ";", ":", "-", "–", "—")):
        return True
    words = re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", clean.lower(), flags=re.UNICODE)
    if not words:
        return False
    last_word = words[-1]
    return last_word in _OPEN_CLAUSE_TOKENS


def _overlap_span(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def autocorrect_truncations(
    segments: list[Segment],
    *,
    pause_threshold: float = 1.8,
) -> list[Segment]:
    """Autocorregge le frasi troncate e riconcilia le continuazioni spezzate.

    - Se un segmento termina senza punteggiatura terminale (o con virgola/congiunzione)
      ed è seguito da un altro interlocutore: interruzione confermata, si aggiunge '…'
      e il warning 'frase_troncata'.
    - Se è seguito dallo stesso interlocutore entro una breve pausa: non si mette
      '…', ma si assicura che il testo non contenga punti errati prima di una
      continuazione minuscola.
    """
    if not segments:
        return []

    for i in range(len(segments)):
        seg = segments[i]
        text = seg.text.strip()
        if not text:
            continue

        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        has_terminal = is_terminal_text(text)
        is_open = ends_with_open_clause(text)

        if next_seg is not None:
            diff_speaker = next_seg.speaker != seg.speaker
            pause = next_seg.start - seg.end

            if diff_speaker:
                # Controlla se è solo una breve interiezione altrui (< 1.5s) e poi lo stesso speaker prosegue
                is_interjection = False
                if i + 2 < len(segments):
                    after_next = segments[i + 2]
                    if after_next.speaker == seg.speaker and (next_seg.end - next_seg.start) < 1.5:
                        gap_through = after_next.start - seg.end
                        if gap_through <= pause_threshold + 1.5:
                            is_interjection = True

                # Interruzione / cambio voce su frase non conclusa solo se non è una semplice interiezione
                if not is_interjection and (not has_terminal or is_open):
                    clean = re.sub(r"[,;:\-–—\s]+$", "", text)
                    if not clean.endswith("…"):
                        seg.text = clean + "…"
                    if "frase_troncata" not in seg.warnings:
                        seg.warnings.append("frase_troncata")
            else:
                # Stesso interlocutore: continuazione della frase
                next_text = next_seg.text.strip()
                starts_lower = bool(next_text and next_text[0].islower())
                if (not has_terminal or starts_lower) and pause <= pause_threshold:
                    # Rimuove eventuale punto errato inserito da Whisper tra parole collegate
                    if text.endswith(".") and not text.endswith("..") and starts_lower:
                        seg.text = text[:-1]
        else:
            # Ultimo segmento dell'audio: se finisce tronco, segnala o normalizza
            if is_open and not has_terminal:
                clean = re.sub(r"[,;:\-–—\s]+$", "", text)
                if not clean.endswith("…"):
                    seg.text = clean + "…"
                if "frase_troncata" not in seg.warnings:
                    seg.warnings.append("frase_troncata")

    return segments


def autocorrect_overlaps(
    segments: list[Segment],
    overlaps: list[tuple[float, float]] | None = None,
    profiles: dict[str, VoiceProfile] | None = None,
    *,
    min_overlap_seconds: float = 0.15,
) -> list[Segment]:
    """Identifica e autocorregge le sovrapposizioni vocali nei segmenti.

    - Assegna 'voce_sovrapposta' a segmenti che cadono in intervalli con due o più voci.
    - Popola `candidates` indicando l'intervallo di sovrapposizione e le voci concorrenti.
    - Se due segmenti consecutivi si accavallano temporalmente, ne delimita l'estensione
      senza perdere parole.
    """
    if not segments:
        return []

    ov_spans = overlaps or []

    for seg in segments:
        # 1. Controllo rispetto agli intervalli di overlap di pyannote
        overlapping_matches: list[tuple[float, float]] = []
        for a, b in ov_spans:
            ov = _overlap_span(seg.start, seg.end, a, b)
            if ov >= min_overlap_seconds:
                overlapping_matches.append((round(a, 3), round(b, 3)))

        # 2. Controllo rispetto a sovrapposizione temporale con altri segmenti
        concurrent_speakers = set()
        for other in segments:
            if other.id == seg.id or other.speaker == seg.speaker:
                continue
            ov = _overlap_span(seg.start, seg.end, other.start, other.end)
            if ov >= min_overlap_seconds:
                concurrent_speakers.add(other.speaker)
                overlapping_matches.append((round(max(seg.start, other.start), 3),
                                            round(min(seg.end, other.end), 3)))

        if overlapping_matches or concurrent_speakers:
            if "voce_sovrapposta" not in seg.warnings:
                seg.warnings.append("voce_sovrapposta")

            # Aggiunge candidati alternativi per trasparenza nella GUI/dati
            existing_cands = {c.get("speaker") for c in seg.candidates if "speaker" in c}
            for spk in sorted(concurrent_speakers):
                if spk not in existing_cands:
                    candidate_entry = {
                        "speaker": spk,
                        "reason": "sovrapposizione_temporale",
                        "overlap_spans": overlapping_matches[:3],
                    }
                    if profiles and spk in profiles:
                        candidate_entry["display_name"] = profiles[spk].name or spk
                    seg.candidates.append(candidate_entry)

    return segments
