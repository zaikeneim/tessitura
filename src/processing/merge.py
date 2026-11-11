"""Assegnazione delle voci alle parole e raggruppamento in segmenti.

Il metodo ingenuo (per ogni parola, il turno pyannote con piu' sovrapposizione)
produce errori visibili: bastava un turno sbagliato di mezzo secondo per spezzare
una frase a meta' e attribuire due pezzi a due persone diverse
(caso reale: "Si. Per" a SPEAKER_01 e "rendere una linea" a SPEAKER_00).

Qui l'unita' di decisione non e' la parola ma la *frase*:
  1. le parole si raggruppano in frasi, tagliando dove un cambio di voce e'
     plausibile: una pausa percepibile o la fine di un periodo;
  2. ogni frase riceve la voce con piu' sovrapposizione totale, sommata per voce
     (la stessa voce ha spesso piu' turni contigui). Una frase si divide solo se
     dentro ci sono due tratti *lunghi* di voci diverse: li' il cambio e' reale;
  3. livellamento dei residui e raggruppamento in segmenti leggibili.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from copy import deepcopy
from typing import Any

from ..models.session import Segment, Word
from ..models.transcript_artifacts import (
    ASRHypothesis,
    Utterance,
    recompute_overlap_relationships,
)
from .diarize import Turn

# Il voto per frase gia' evita i tagli a meta' discorso, quindi il livellamento
# interviene solo sui casi minimi: una parola sola e piu' corta di mezzo secondo,
# dove la diarizzazione e' meno affidabile del contesto attorno.
MIN_RUN_SECONDS = 0.3
MIN_RUN_WORDS = 2
# Massima distanza per attaccare una parola a un turno che non la copre.
MAX_SNAP_SECONDS = 2.0
# Silenzio oltre il quale si apre comunque un nuovo segmento della stessa voce.
SEGMENT_GAP_SECONDS = 1.2

DEFAULT_SPEAKER = "SPEAKER_00"

# Testo dell'evento simultaneo rilevato dalla diarizzazione ma non isolato
# dall'ASR a flusso singolo: va trascritto a mano dalla timeline.
UNTRANSCRIBED_OVERLAP_TEXT = "[parlato simultaneo da trascrivere]"
# Sotto questa durata una sovrapposizione senza parole non diventa un evento.
MIN_UNTRANSCRIBED_OVERLAP_S = 0.25


@dataclass
class WordUnit:
    start: float
    end: float
    text: str
    prob: float | None = None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _vote(start: float, end: float, turns: list[Turn]) -> str | None:
    """Voce con piu' sovrapposizione totale sulla parola, o la piu' vicina."""
    if end <= start:  # parole a durata zero: finestra minima attorno al punto
        start, end = start - 0.05, start + 0.05

    scores: dict[str, float] = {}
    for tr in turns:
        ov = _overlap(start, end, tr.start, tr.end)
        if ov > 0.0:
            scores[tr.speaker] = scores.get(tr.speaker, 0.0) + ov
    if scores:
        return max(scores, key=lambda k: scores[k])

    best: str | None = None
    best_dist = MAX_SNAP_SECONDS
    for tr in turns:
        dist = tr.start - end if tr.start > end else start - tr.end
        if 0.0 <= dist <= best_dist:
            best_dist = dist
            best = tr.speaker
    return best


def _runs(labels: list[str | None]) -> list[tuple[int, int]]:
    """Intervalli [inizio, fine) di etichette uguali consecutive."""
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        out.append((i, j))
        i = j
    return out


def _smooth(
    words: list[WordUnit],
    labels: list[str | None],
    *,
    preserve_short_turns: bool = False,
    overlaps: list[tuple[float, float]] | None = None,
) -> list[str | None]:
    """Riassorbe le sequenze troppo corte per essere un vero cambio di voce."""
    if preserve_short_turns:
        # Le etichette sono già state riempite parola per parola. In una
        # sessione multi-voce anche un turno in apertura o chiusura può essere
        # una risposta reale, quindi non lo spostiamo verso il vicino.
        return labels
    for _ in range(3):  # poche passate bastano a stabilizzare
        runs = _runs(labels)
        if len(runs) < 2:
            break
        changed = False
        for k in range(len(runs)):
            i, j = runs[k]
            dur = words[j - 1].end - words[i].start
            if (j - i) >= MIN_RUN_WORDS or dur >= MIN_RUN_SECONDS:
                continue
            # Se la sequenza breve cade in un'area di sovrapposizione accertata,
            # non appiattirla sul vicino: è un intervento simultaneo o interiezione reale.
            if overlaps:
                w_start = words[i].start
                w_end = words[j - 1].end
                if any(max(w_start, o_start) < min(w_end, o_end) for o_start, o_end in overlaps):
                    continue
            prev_lbl = labels[runs[k - 1][0]] if k > 0 else None
            next_lbl = labels[runs[k + 1][0]] if k < len(runs) - 1 else None
            if prev_lbl is None and next_lbl is None:
                continue
            if prev_lbl is None:               # sequenza corta a inizio audio
                winner = next_lbl
            elif next_lbl is None:             # sequenza corta a fine audio
                winner = prev_lbl
            elif prev_lbl == next_lbl:
                winner = prev_lbl              # incastonata nella stessa voce
            else:
                if preserve_short_turns:
                    # In un dialogo con più voci, una risposta di una sola
                    # parola tra due speaker diversi è spesso un'interruzione
                    # reale ("No", "Cosa?", "Sì"), non rumore di diarizzazione.
                    continue
                prev_len = runs[k - 1][1] - runs[k - 1][0]
                next_len = runs[k + 1][1] - runs[k + 1][0]
                winner = prev_lbl if prev_len >= next_len else next_lbl
            if winner is not None and winner != labels[i]:
                for idx in range(i, j):
                    labels[idx] = winner
                changed = True
        if not changed:
            break
    return labels


# --- raggruppamento in frasi ------------------------------------------------

# Una pausa cosi' lunga separa due gruppi di parole; sotto, si sta parlando di
# seguito e un cambio di voce in mezzo e' quasi sempre un errore.
PHRASE_BREAK_PAUSE = 0.35
# Oltre questa durata una frase viene comunque spezzata: un monologo lungo puo'
# contenere davvero un cambio di voce.
MAX_PHRASE_SECONDS = 12.0
# Dentro una frase, un tratto di voce diversa lungo almeno cosi' e' un cambio vero
# (parlato accavallato senza pausa) e non un errore di allineamento.
STRONG_TURN_SECONDS = 0.65
MIN_STRONG_TURN_SECONDS = 0.35
_SENTENCE_END = (".", "?", "!", "…")


def _phrases(
    units: list[WordUnit],
    turns: list[Turn] | None = None,
) -> list[tuple[int, int]]:
    """Spezza la sequenza di parole in gruppi coesi [inizio, fine).

    Si taglia su:
    - una pausa percepibile (>= 0.35s);
    - la fine di una frase (. ? ! …);
    - superamento della durata massima;
    - un cambio di interlocutore verificato dai turni quando la parola successiva
      inizia un nuovo enunciato (maiuscola o pausa >= 0.15s) o quella precedente
      era troncata/aperta.
    """
    if not units:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(units)):
        pause = units[i].start - units[i - 1].end
        prev_text = units[i - 1].text.strip()
        curr_text = units[i].text.strip()
        too_long = units[i - 1].end - units[start].start >= MAX_PHRASE_SECONDS

        is_break = pause >= PHRASE_BREAK_PAUSE or prev_text.endswith(_SENTENCE_END) or too_long

        # Controllo transizione turno tra interlocutori diversi
        if not is_break and turns and len(turns) > 0 and pause >= 0.10:
            spk_prev = _vote(units[i - 1].start, units[i - 1].end, turns)
            spk_curr = _vote(units[i].start, units[i].end, turns)
            if spk_prev and spk_curr and spk_prev != spk_curr:
                starts_capital = bool(curr_text and curr_text[0].isupper())
                from .autocorrect import ends_with_open_clause
                if starts_capital or pause >= 0.20 or ends_with_open_clause(prev_text):
                    is_break = True

        if is_break:
            out.append((start, i))
            start = i
    out.append((start, len(units)))
    return out


def _vote_phrase(units: list[WordUnit], i0: int, i1: int, turns: list[Turn]) -> str | None:
    """Voce del gruppo di parole: somma della sovrapposizione su tutte le parole."""
    scores: dict[str, float] = {}
    for u in units[i0:i1]:
        start, end = (u.start, u.end) if u.end > u.start else (u.start - 0.05, u.start + 0.05)
        for tr in turns:
            ov = _overlap(start, end, tr.start, tr.end)
            if ov > 0.0:
                scores[tr.speaker] = scores.get(tr.speaker, 0.0) + ov
    if scores:
        return max(scores, key=lambda k: scores[k])
    return _vote(units[i0].start, units[i1 - 1].end, turns)


def _phrase_labels(
    units: list[WordUnit], i0: int, i1: int, turns: list[Turn],
    *, preserve_short_turns: bool = False,
    voice_profiles: dict | None = None,
    overlaps: list[tuple[float, float]] | None = None,
) -> list[str | None]:
    """Etichette per le parole di una frase.

    Di norma la frase intera prende una voce sola. Se dentro la frase la
    diarizzazione vede un turno consistente di un'altra voce sostenuto dall'audio,
    dal profilo acustico o da una regione di sovrapposizione accertata, la frase
    viene divisa invece di essere appiattita.
    """
    sub = units[i0:i1]
    raw = _fill_gaps([_vote(u.start, u.end, turns) for u in sub])
    if preserve_short_turns:
        # Con un numero di speaker noto, i turni pyannote sono informazione
        # utile anche quando il cambio avviene senza pausa percettibile.
        return list(raw)
    runs = _runs(raw)  # type: ignore[arg-type]

    strong = set()
    for k, (a, b) in enumerate(runs):
        dur = sub[b - 1].end - sub[a].start
        num_words = b - a
        spk = raw[a]
        # Se dura almeno STRONG_TURN_SECONDS, è sicuramente un cambio reale
        if dur >= STRONG_TURN_SECONDS:
            strong.add(k)
        elif dur >= MIN_STRONG_TURN_SECONDS and num_words >= 2:
            # Turno breve ma con più parole e supporto acustico
            strong.add(k)
        elif voice_profiles and spk in voice_profiles and dur >= MIN_STRONG_TURN_SECONDS:
            # Voce appartenente a un profilo acustico accertato
            strong.add(k)
        elif overlaps and any(max(sub[a].start, o_start) < min(sub[b - 1].end, o_end) for o_start, o_end in overlaps):
            # Tratto ricadente in sovrapposizione accertata
            strong.add(k)

    if len(strong) < 2:
        return [_vote_phrase(units, i0, i1, turns)] * len(sub)

    out: list[str | None] = [None] * len(sub)
    for k, (a, b) in enumerate(runs):
        if k in strong:
            for i in range(a, b):
                out[i] = raw[a]
    return list(_fill_gaps(out))  # i tratti brevi seguono il tratto lungo vicino


def _fill_gaps(labels: list[str | None]) -> list[str]:
    """Parole senza voce assegnata: prendono quella del vicino piu' prossimo."""
    known = [i for i, l in enumerate(labels) if l is not None]
    if not known:
        return [DEFAULT_SPEAKER] * len(labels)
    out: list[str] = []
    cursor = 0
    for i, l in enumerate(labels):
        if l is not None:
            out.append(l)
            continue
        while cursor + 1 < len(known) and abs(known[cursor + 1] - i) < abs(known[cursor] - i):
            cursor += 1
        nearest = known[cursor]
        out.append(labels[nearest] or DEFAULT_SPEAKER)
    return out


def flatten_words(tsegments) -> list[WordUnit]:
    """Da segmenti Whisper a lista piatta di parole (fallback: il segmento intero)."""
    units: list[WordUnit] = []
    for tseg in tsegments:
        if tseg.words:
            for start, end, text, prob in tseg.words:
                units.append(WordUnit(start=start, end=end, text=text, prob=prob))
        else:
            units.append(WordUnit(start=tseg.start, end=tseg.end, text=tseg.text))
    units.sort(key=lambda w: (w.start, w.end))
    return units


def assign_speakers(
    units: list[WordUnit],
    turns: list[Turn],
    *,
    preserve_short_turns: bool = False,
    voice_profiles: dict | None = None,
    overlaps: list[tuple[float, float]] | None = None,
) -> list[str]:
    """Etichetta ogni parola, preservando i turni brevi quando richiesto."""
    if not units:
        return []
    if not turns:
        return [DEFAULT_SPEAKER] * len(units)

    labels: list[str | None] = [None] * len(units)
    # Only nearby turns can overlap a phrase or be snapped to it. The prefix
    # maximum keeps long overlapping turns in the window as well.
    original_order = {id(turn): i for i, turn in enumerate(turns)}
    sorted_turns = sorted(turns, key=lambda turn: turn.start)
    starts = [turn.start for turn in sorted_turns]
    ends: list[float] = []
    latest = float("-inf")
    for turn in sorted_turns:
        latest = max(latest, turn.end)
        ends.append(latest)
    for i0, i1 in _phrases(units, sorted_turns):
        start = min(unit.start for unit in units[i0:i1]) - MAX_SNAP_SECONDS - 0.05
        end = max(unit.end for unit in units[i0:i1]) + MAX_SNAP_SECONDS + 0.05
        nearby = sorted_turns[bisect_left(ends, start):bisect_right(starts, end)]
        nearby.sort(key=lambda turn: original_order[id(turn)])
        labels[i0:i1] = _phrase_labels(
            units, i0, i1, nearby,
            preserve_short_turns=preserve_short_turns,
            voice_profiles=voice_profiles,
            overlaps=overlaps,
        )

    labels = _smooth(units, labels, preserve_short_turns=preserve_short_turns, overlaps=overlaps)
    return _absorb_alternations(
        units,
        _fill_gaps(labels),
        overlaps=overlaps,
        preserve_short_turns=preserve_short_turns,
    )


# Una catena di alternanze piu' corta di cosi' e' un normale scambio di battute.
MIN_ALTERNATION_RUNS = 4
# Oltre questo stacco le due voci non si stanno contendendo la stessa frase.
ALTERNATION_MAX_GAP = 0.35
# Una sequenza piu' lunga di cosi' e' un turno, non un frammento di frase.
ALTERNATION_MAX_WORDS = 3


def _within_overlap(
    words: list[WordUnit],
    run: tuple[int, int],
    overlaps: list[tuple[float, float]],
) -> bool:
    """La sequenza cade in una sovrapposizione attestata dalla diarizzazione."""
    start, end = words[run[0]].start, words[run[1] - 1].end
    return any(max(start, lo) < min(end, hi) for lo, hi in overlaps)


def _continues_phrase(words: list[WordUnit], boundary: int) -> bool:
    """La parola prima di `boundary` lascia la frase aperta e non c'e' stacco.

    Un cambio di voce qui non e' un turno: e' un taglio in mezzo a una frase.
    """
    if boundary <= 0 or boundary >= len(words):
        return False
    if words[boundary].start - words[boundary - 1].end > ALTERNATION_MAX_GAP:
        return False
    return not words[boundary - 1].text.strip().endswith(_SENTENCE_END)


def _absorb_alternations(
    words: list[WordUnit],
    labels: list[str],
    *,
    overlaps: list[tuple[float, float]] | None = None,
    preserve_short_turns: bool = False,
) -> list[str]:
    """Ricuce le frasi spezzate dal rumore di attribuzione.

    Whisper emette una sequenza continua di parole e la diarizzazione ci alterna
    sopra l'etichetta; `build_segments` apre un segmento a ogni cambio e frantuma
    la frase. Misurato sulle sessioni reali: 8.6% dei segmenti finisce in catene
    lunghe fino a 15, con articoli strappati a meta' sintagma ("abbiamo fatto
    tutta" / "la" / "camminata e la" / "tirata").

    Due pattern, entrambi riconosciuti dal fatto che il taglio cade fra due parole
    consecutive senza pausa e con la frase ancora aperta:

    - **isola**: una manciata di parole incastonata fra due tratti della stessa
      voce, riassorbita in quella voce;
    - **catena**: alternanza prolungata fra due voci, collassata su quella che
      occupa piu' tempo.

    Un turno vero comincia dopo una pausa o dopo una frase chiusa, e resta intatto.

    Limite: l'attribuzione alternativa non viene conservata. Recuperarla richiede
    il percorso target-speaker (piano 2026-09-21, livello 2), non un'euristica.
    """

    def is_fragment(run: tuple[int, int]) -> bool:
        return (run[1] - run[0]) <= ALTERNATION_MAX_WORDS

    def glued(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[1] == right[0] and _continues_phrase(words, right[0])

    def paint(span: list[tuple[int, int]], winner: str) -> bool:
        touched = False
        for i, j in span:
            for idx in range(i, j):
                if labels[idx] != winner:
                    labels[idx] = winner
                    touched = True
        return touched

    for _ in range(4):  # riassorbire puo' scoprire nuovi pattern
        runs = _runs(labels)
        if len(runs) < 3:
            return labels
        changed = False

        # Catene prima delle isole: un'alternanza prolungata fra due voci sulla
        # stessa frase va trattata nel suo complesso per collassare sull'oratore dominante,
        # prima che la passata sulle isole ne consumi i singoli tratti interni.
        runs = _runs(labels)
        k = 0
        while k < len(runs):
            if not is_fragment(runs[k]):
                k += 1
                continue
            end = k
            while (
                end + 1 < len(runs)
                and is_fragment(runs[end + 1])
                and glued(runs[end], runs[end + 1])
            ):
                end += 1
            chain = runs[k : end + 1]
            if len(chain) >= MIN_ALTERNATION_RUNS and len({labels[i] for i, _ in chain}) == 2:
                occupancy: dict[str, float] = {}
                for i, j in chain:
                    occupancy[labels[i]] = occupancy.get(labels[i], 0.0) + (
                        words[j - 1].end - words[i].start
                    )
                changed |= paint(chain, max(occupancy, key=lambda spk: occupancy[spk]))
            k = end + 1

        # Isole: poche parole incastonate fra due tratti della stessa voce.
        # Dove pyannote attesta sovrapposizione, una singola interiezione puo'
        # essere un intervento reale: la si lascia stare. Un'alternanza lunga no,
        # e viene comunque ricucita dalla passata sulle catene sopra.
        if not preserve_short_turns:
            runs = _runs(labels)
            for k in range(1, len(runs) - 1):
                before, island, after = runs[k - 1], runs[k], runs[k + 1]
                if not is_fragment(island) or labels[before[0]] != labels[after[0]]:
                    continue
                if overlaps and _within_overlap(words, island, overlaps):
                    continue
                if glued(before, island) and glued(island, after):
                    changed |= paint([island], labels[before[0]])

        if not changed:
            break
    return labels


def build_segments(
    units: list[WordUnit],
    labels: list[str],
    *,
    voice_profiles: dict | None = None,
    overlaps: list[tuple[float, float]] | None = None,
    autocorrect_enabled: bool = True,
) -> list[Segment]:
    """Raggruppa parole consecutive della stessa voce in segmenti leggibili."""
    # `zip` troncherebbe sulla lista piu' corta, facendo sparire le ultime parole
    # dalla trascrizione senza dirlo. Un'etichetta mancante si ricostruisce, una
    # parola persa no: si completa la lista e si lascia traccia dell'anomalia.
    if len(labels) != len(units):
        logging.getLogger(__name__).warning(
            "Etichette (%d) e parole (%d) non combaciano: completate con la voce vicina",
            len(labels), len(units),
        )
        labels = list(labels[: len(units)])
        filler = labels[-1] if labels else DEFAULT_SPEAKER
        labels.extend([filler] * (len(units) - len(labels)))

    segments: list[Segment] = []
    current: Segment | None = None
    text_parts: list[str] = []
    for unit, spk in zip(units, labels):
        word = Word(start=unit.start, end=unit.end, text=unit.text.strip(),
                    probability=unit.prob)
        same_voice = current is not None and spk == current.speaker
        close_enough = current is not None and (unit.start - current.end) < SEGMENT_GAP_SECONDS
        if same_voice and close_enough:
            current.end = max(current.end, unit.end)
            text_parts.append(unit.text if unit.text.startswith(" ") else " " + unit.text)
            current.words.append(word)
        else:
            if current is not None:
                current.text = "".join(text_parts).strip()
                segments.append(current)
            current = Segment(start=unit.start, end=unit.end, text=unit.text.strip(),
                              speaker=spk, words=[word])
            text_parts = [unit.text.strip()]
    if current is not None:
        current.text = "".join(text_parts).strip()
        segments.append(current)

    cleaned = [s for s in segments if s.text]

    # Applicazione dell'autocorrezione per troncamenti e sovrapposizioni
    if autocorrect_enabled and cleaned:
        from .autocorrect import autocorrect_truncations, autocorrect_overlaps

        cleaned = autocorrect_truncations(cleaned)
        if overlaps:
            cleaned = autocorrect_overlaps(cleaned, overlaps=overlaps, profiles=voice_profiles)

    return cleaned


def merge_transcript_and_diarization(
    tsegments,
    turns: list[Turn],
    *,
    preserve_short_turns: bool = False,
    voice_profiles: dict | None = None,
    overlaps: list[tuple[float, float]] | None = None,
) -> list[Segment]:
    """Trascrizione + turni di parola -> segmenti con voce assegnata."""
    units = flatten_words(tsegments)
    labels = assign_speakers(
        units, turns,
        preserve_short_turns=preserve_short_turns,
        voice_profiles=voice_profiles,
        overlaps=overlaps,
    )
    return build_segments(
        units, labels,
        voice_profiles=voice_profiles,
        overlaps=overlaps,
    )


def apply_channel_hint(
    units: list[WordUnit],
    labels: list[str],
    dominance,
    frame_s: float,
    channel_names: dict[int, str],
) -> list[str]:
    """Corregge le etichette usando la sorgente audio (mic vs sistema).

    Con due tracce separate sappiamo per costruzione *da quale canale* arriva una
    parola: se una parola viene chiaramente dal microfono e' di chi usa il PC, non
    di un partecipante remoto. E' un indizio molto piu' forte del timbro, e qui
    rinomina le voci mantenendo comunque la separazione fatta da pyannote dentro
    ciascun canale.
    """
    if dominance is None or len(dominance) == 0 or not channel_names:
        return labels

    # Per ogni voce trovata da pyannote, quale canale la genera in prevalenza.
    per_speaker: dict[str, dict[int, float]] = {}
    for unit, spk in zip(units, labels):
        i0 = int(unit.start / frame_s)
        i1 = max(i0 + 1, int(unit.end / frame_s))
        counts = per_speaker.setdefault(spk, {})
        for ch in dominance[i0:i1]:
            ch = int(ch)
            if ch >= 0:
                counts[ch] = counts.get(ch, 0.0) + 1.0

    renamed: dict[str, str] = {}
    used: dict[str, int] = {}
    for spk, counts in per_speaker.items():
        if not counts:
            continue
        ch = max(counts, key=lambda k: counts[k])
        total = sum(counts.values())
        if counts[ch] / total < 0.6:  # voce ambigua: la lascio com'e'
            continue
        base = channel_names.get(ch)
        if not base:
            continue
        n = used.get(base, 0)
        used[base] = n + 1
        renamed[spk] = base if n == 0 else f"{base}_{n + 1}"

    return [renamed.get(spk, spk) for spk in labels]


def build_utterances(
    hypotheses: list[ASRHypothesis],
    turns: list[Turn] | None = None,
    *,
    overlaps: list[tuple[float, float]] | None = None,
    voice_profiles: dict | None = None,
) -> list[Utterance]:
    """Costruisce utterance preservando interventi simultanei e assegnando speaker.

    Non appiattisce piu' flussi sovrapposti in una sequenza monovocale: se due
    ipotesi ASR coprono lo stesso intervallo (es. canali separati o parlatori simultanei),
    vengono generate due Utterance collegate tramite lo stesso `overlap_group_id` e con
    `stream_id` distinti.
    """
    if not hypotheses:
        return []

    sorted_hyps = sorted(hypotheses, key=lambda h: (h.start_s, h.end_s))
    utterances: list[Utterance] = []

    # Rilevamento sovrapposizioni tra ipotesi
    for i, hyp in enumerate(sorted_hyps):
        # Determina speaker per questa ipotesi
        assigned_speaker = hyp.parameters.get("speaker")
        if not assigned_speaker and turns:
            units = [
                WordUnit(start=w.start, end=w.end, text=w.text, prob=w.probability)
                for w in hyp.words
            ]
            if units:
                labels = assign_speakers(
                    units, turns,
                    preserve_short_turns=True,
                    voice_profiles=voice_profiles,
                    overlaps=overlaps,
                )
                from collections import Counter
                counts = Counter(labels)
                if counts:
                    assigned_speaker = counts.most_common(1)[0][0]
            else:
                assigned_speaker = _vote(hyp.start_s, hyp.end_s, turns)
        if not assigned_speaker:
            assigned_speaker = DEFAULT_SPEAKER

        # Controllo overlap con altre ipotesi già create
        overlap_grp = None
        stream = 0
        for other_utt in utterances:
            ov = max(0.0, min(hyp.end_s, other_utt.end_s) - max(hyp.start_s, other_utt.start_s))
            if ov > 0.1:
                if other_utt.overlap_group_id is None:
                    import uuid
                    other_utt.overlap_group_id = f"ov_{uuid.uuid4().hex[:8]}"
                overlap_grp = other_utt.overlap_group_id
                stream = other_utt.stream_id + 1
                break

        # Se non c'è overlap diretto con altre ipotesi, verifica lista overlaps acustici
        if overlap_grp is None and overlaps:
            for o_start, o_end in overlaps:
                if max(hyp.start_s, o_start) < min(hyp.end_s, o_end):
                    import uuid
                    overlap_grp = f"ov_{uuid.uuid4().hex[:8]}"
                    break

        utt = Utterance(
            start_s=hyp.start_s,
            end_s=hyp.end_s,
            stream_id=stream,
            overlap_group_id=overlap_grp,
            speaker=assigned_speaker,
            text=hyp.text,
            words=hyp.words,
            hypothesis_ids=[hyp.hypothesis_id] if hyp.hypothesis_id else [],
            source_audio_ref=hyp.view_id or None,
            asr_provenance={
                "backend": hyp.backend,
                "model_revision": hyp.model_revision,
                "view_id": hyp.view_id,
            },
        )
        utterances.append(utt)

    recompute_overlap_relationships(utterances, preserve_unpaired_groups=True)
    return utterances


def utterances_to_segments(utterances: list[Utterance]) -> list[Segment]:
    """Proiezione compatibile da Utterance a Segment.

    Preserva tutte le utterance (incluse quelle simultanee nello stesso intervallo),
    ordinandole per start_s e mantenendo riferimenti e parole.
    """
    segments: list[Segment] = []
    for utt in sorted(utterances, key=lambda u: (u.start_s, u.end_s, u.stream_id)):
        spk_str = utt.speaker if utt.speaker is not None else "UNASSIGNED"
        seg = Segment(
            id=utt.utterance_id,
            start=utt.start_s,
            end=utt.end_s,
            text=utt.text or (UNTRANSCRIBED_OVERLAP_TEXT if utt.processing_state == "missing_transcription" else ""),
            speaker=spk_str,
            words=deepcopy(utt.words),
            source_ids=list(utt.hypothesis_ids),
            candidates=deepcopy(utt.candidate_speakers),
        )
        if utt.is_overlap or utt.overlap_group_id:
            if "voce_sovrapposta" not in seg.warnings:
                seg.warnings.append("voce_sovrapposta")
            if utt.overlap_group_id:
                seg.warnings.append(f"overlap_group:{utt.overlap_group_id}:stream_{utt.stream_id}")
        if utt.processing_state == "missing_transcription":
            seg.warnings.append("trascrizione_mancante")
        if utt.uncertain:
            seg.warnings.append("attribuzione_incerta")
        if utt.user_corrections and not utt.words:
            seg.words_valid = False
            seg.warnings.append("timing_da_riallineare")
        segments.append(seg)
    return segments


def materialize_overlap_utterances(
    utterances: list[Utterance],
    *,
    overlaps: list[tuple[float, float]] | None = None,
    ordinary_turns: list[Any] | None = None,
    exclusive_turns: list[Any] | None = None,
    source_audio_ref: str | None = None,
) -> list[Utterance]:
    """Preserve ordinary activity as explicit, untranscribed events.

    Exclusive turns aid word alignment only. Activity does not identify words:
    never move punctuation, extend transcripts, or guess a secondary utterance.
    Corrected events are kept verbatim. Repeated reconstruction is idempotent.
    """
    import math
    import uuid

    turns = []
    for turn in ordinary_turns or []:
        if isinstance(turn, dict):
            start = float(turn.get("start", turn.get("start_s", 0)))
            end = float(turn.get("end", turn.get("end_s", 0)))
            speaker = turn.get("speaker")
        else:
            start, end, speaker = float(turn.start), float(turn.end), turn.speaker
        if math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            turns.append((start, end, speaker))

    # Ordinary intervals, not exclusive alignment turns, define all activity.
    for start, end, speaker in turns:
        covered = sorted((max(start, u.start_s), min(end, u.end_s))
                         for u in utterances if u.speaker_id == speaker
                         and u.end_s > start and u.start_s < end)
        cursor = start
        gaps = []
        for left, right in covered:
            if left > cursor + 0.02:
                gaps.append((cursor, left))
            cursor = max(cursor, right)
        if end > cursor + 0.02:
            gaps.append((cursor, end))
        for left, right in gaps:
            identity = f"activity:{source_audio_ref}:{speaker}:{left:.6f}:{right:.6f}"
            event_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
            if any(u.event_id == event_id for u in utterances):
                continue
            simultaneous = [other[2] for other in turns if other[2] != speaker
                            and min(right, other[1]) > max(left, other[0])]
            utterances.append(Utterance(
                start_s=left, end_s=right, utterance_id=event_id, speaker=speaker,
                text="", source_audio_ref=source_audio_ref,
                processing_state="missing_transcription", uncertain=True,
                edit_status="untranscribed_overlap" if simultaneous else "untranscribed_activity",
                asr_provenance={"status": "no_speaker_specific_text", "not_verified_missing_speech": True},
                diarization_provenance={"source": "non_exclusive_diarization",
                    "turn_interval": [start, end], "potential_speakers": sorted(set(simultaneous + [speaker])),
                    "identity_status": "predicted", "exclusive_is_alignment_aid": bool(exclusive_turns)},
            ))
    utterances.sort(key=lambda u: (u.start_s, u.end_s, u.event_id))
    recompute_overlap_relationships(utterances)
    return utterances


def build_utterances_from_segments(
    segments: list[Segment],
    *,
    overlaps: list[tuple[float, float]] | None = None,
    ordinary_turns: list[Any] | None = None,
    exclusive_turns: list[Any] | None = None,
    source_audio_ref: str | None = None,
    asr_model: str | None = None,
    audio_path: Any | None = None,
) -> list[Utterance]:
    """Costruisce Utterance a partire da Segment, rilevando sovrapposizioni e assegnando stream_id."""
    ref = source_audio_ref or (str(audio_path) if audio_path else None)
    utterances: list[Utterance] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        overlap_grp = None
        stream = 0
        for other_utt in utterances:
            ov = max(0.0, min(seg.end, other_utt.end_s) - max(seg.start, other_utt.start_s))
            if ov > 0.1:
                if other_utt.overlap_group_id is None:
                    import uuid
                    other_utt.overlap_group_id = f"ov_{uuid.uuid4().hex[:8]}"
                overlap_grp = other_utt.overlap_group_id
                stream = other_utt.stream_id + 1
                break
        if overlap_grp is None and overlaps:
            for o_start, o_end in overlaps:
                if max(seg.start, o_start) < min(seg.end, o_end):
                    import uuid
                    overlap_grp = f"ov_{uuid.uuid4().hex[:8]}"
                    break

        utt = Utterance(
            start_s=seg.start,
            end_s=seg.end,
            utterance_id=seg.id,
            stream_id=stream,
            overlap_group_id=overlap_grp,
            speaker=seg.speaker,
            candidate_speakers=deepcopy(seg.candidates),
            text=seg.text,
            words=deepcopy(seg.words),
            hypothesis_ids=list(seg.source_ids),
            source_audio_ref=ref,
            asr_provenance={"model": asr_model} if asr_model else {},
            diarization_provenance={
                "has_exclusive_aid": bool(exclusive_turns),
                "has_ordinary_turns": bool(ordinary_turns),
            },
        )
        utterances.append(utt)

    if ordinary_turns or any(seg.candidates for seg in segments):
        return materialize_overlap_utterances(
            utterances,
            overlaps=overlaps,
            ordinary_turns=ordinary_turns,
            exclusive_turns=exclusive_turns,
            source_audio_ref=ref,
        )

    recompute_overlap_relationships(utterances, preserve_unpaired_groups=True)
    return utterances


