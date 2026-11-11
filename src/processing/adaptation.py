"""Apprendimento locale da correzioni esplicite e verificabili.

I pesi dei modelli non vengono modificati durante l'uso.  Questo modulo
costruisce invece una memoria di progetto piccola, ispezionabile e reversibile:
termini corretti per Whisper e profili vocali confermati per la diarizzazione.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import math
import re
from collections.abc import Iterable, Mapping, Sequence

_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or ""))


def transcription_corrections(before: str, after: str, *, limit: int = 12) -> list[dict[str, str]]:
    """Estrae sostituzioni brevi da una correzione manuale.

    Frasi lunghe e cambi di sola punteggiatura vengono scartati: inserirli nel
    prompt di Whisper ridurrebbe lo spazio utile e potrebbe introdurre bias.
    """
    old = _words(before)
    new = _words(after)
    if not new or [word.casefold() for word in old] == [word.casefold() for word in new]:
        return []

    matcher = SequenceMatcher(
        None,
        [word.casefold() for word in old],
        [word.casefold() for word in new],
        autojunk=False,
    )
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        replacement = new[j1:j2]
        # Il prompt iniziale funziona bene per nomi/termini e male come memoria
        # di intere frasi. Manteniamo solo correzioni locali.
        if len(replacement) > 5 or not any(len(word) >= 3 for word in replacement):
            continue
        after_text = " ".join(replacement).strip()
        before_text = " ".join(old[i1:i2]).strip()
        if not after_text or not before_text or len(after_text) > 100:
            continue
        # Un cambio di contenuto ("casa" → "ufficio") non è un errore ASR e
        # non deve influenzare Whisper. Conserviamo solo coppie abbastanza
        # simili da rappresentare verosimilmente una correzione ortografica.
        lexical_similarity = SequenceMatcher(
            None, before_text.casefold(), after_text.casefold(), autojunk=False
        ).ratio()
        if lexical_similarity < 0.45:
            continue
        key = (before_text.casefold(), after_text.casefold())
        if key in seen:
            continue
        seen.add(key)
        found.append({"before": before_text, "after": after_text})
        if len(found) >= max(0, limit):
            break
    return found


def normalize_embedding(value, *, max_dimension: int = 4096) -> list[float] | None:
    """Valida e normalizza un embedding, senza dipendere da NumPy."""
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not vector or len(vector) > max_dimension or not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return [item / norm for item in vector]


def mean_embedding(values: Iterable[Sequence[float]]) -> list[float] | None:
    """Media normalizzata di profili compatibili; ignora dati corrotti."""
    vectors = [vector for value in values if (vector := normalize_embedding(value)) is not None]
    if not vectors:
        return None
    counts: dict[int, int] = {}
    for vector in vectors:
        counts[len(vector)] = counts.get(len(vector), 0) + 1
    dimension = max(counts, key=lambda size: (counts[size], -size))
    vectors = [vector for vector in vectors if len(vector) == dimension]
    if not vectors:
        return None
    mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]
    return normalize_embedding(mean)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def match_speaker_profiles(
    current: Mapping[str, Sequence[float]],
    profiles: Mapping[str, Sequence[float]],
    *,
    threshold: float = 0.72,
    margin: float = 0.08,
) -> dict[str, dict[str, float | str]]:
    """Abbina voci a profili noti con assegnazione globale uno-a-uno.

    La soglia e l'opzione di lasciare una voce senza nome impediscono che un
    profilo debole venga presentato come identità certa. Per gruppi normali si
    usa una ricerca esatta; oltre 12 profili si passa a un greedy deterministico.
    """
    active = {
        str(speaker): vector
        for speaker, raw in current.items()
        if (vector := normalize_embedding(raw)) is not None
    }
    known = {
        str(name): vector
        for name, raw in profiles.items()
        if str(name).strip() and (vector := normalize_embedding(raw)) is not None
    }
    if not active or not known:
        return {}

    speakers = sorted(active)
    names = sorted(known, key=str.casefold)
    scores = [
        [_cosine(active[speaker], known[name]) for name in names]
        for speaker in speakers
    ]

    def eligible(row: int, column: int) -> bool:
        score = scores[row][column]
        if score < threshold:
            return False
        row_alternatives = [value for index, value in enumerate(scores[row]) if index != column]
        if row_alternatives and score - max(row_alternatives) < margin:
            return False
        column_alternatives = [
            scores[index][column] for index in range(len(speakers)) if index != row
        ]
        return not column_alternatives or score - max(column_alternatives) >= margin

    pairs: list[tuple[int, int]] = []
    if len(speakers) <= 12 and len(names) <= 12:
        memo: dict[tuple[int, int], tuple[float, tuple[tuple[int, int], ...]]] = {}

        def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
            key = (row, used)
            if key in memo:
                return memo[key]
            if row >= len(speakers):
                return 0.0, ()
            best_score, best_pairs = solve(row + 1, used)
            for column, score in enumerate(scores[row]):
                if used & (1 << column) or not eligible(row, column):
                    continue
                tail_score, tail_pairs = solve(row + 1, used | (1 << column))
                candidate = (score + tail_score, ((row, column), *tail_pairs))
                if candidate[0] > best_score + 1e-12 or (
                    abs(candidate[0] - best_score) <= 1e-12 and candidate[1] < best_pairs
                ):
                    best_score, best_pairs = candidate
            memo[key] = best_score, best_pairs
            return memo[key]

        pairs = list(solve(0, 0)[1])
    else:
        candidates = sorted(
            (
                (score, row, column)
                for row, row_scores in enumerate(scores)
                for column, score in enumerate(row_scores)
                if eligible(row, column)
            ),
            key=lambda item: (-item[0], speakers[item[1]], names[item[2]].casefold()),
        )
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        for _score, row, column in candidates:
            if row not in used_rows and column not in used_columns:
                pairs.append((row, column))
                used_rows.add(row)
                used_columns.add(column)

    return {
        speakers[row]: {"name": names[column], "similarity": round(scores[row][column], 6)}
        for row, column in pairs
    }
