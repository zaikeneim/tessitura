"""Offline quality metrics for the local benchmark harness.

The fallback metrics are intentionally small and transparent. If pyannote or
MeetEval are installed, the benchmark script can export STM/RTTM and delegate
official DER/tcpWER calculation; these functions keep CI useful without those
heavy optional packages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def normalize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ÖØ-öø-ÿ']+", text.casefold())


def edit_distance(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    # Keep only two rows and four integer arrays. The old full matrix used
    # O(N*M) tuples and became needlessly large for a 60-minute corpus.
    width = len(hypothesis) + 1
    previous_distance = list(range(width))
    previous_insertions = list(range(width))
    previous_deletions = [0] * width
    previous_substitutions = [0] * width
    for i, ref in enumerate(reference, 1):
        current_distance = [i] + [0] * (width - 1)
        current_insertions = [0] * width
        current_deletions = [i] + [0] * (width - 1)
        current_substitutions = [0] * width
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current_distance[j] = previous_distance[j - 1]
                current_insertions[j] = previous_insertions[j - 1]
                current_deletions[j] = previous_deletions[j - 1]
                current_substitutions[j] = previous_substitutions[j - 1]
                continue
            deletion = (previous_distance[j] + 1, previous_insertions[j],
                        previous_deletions[j] + 1, previous_substitutions[j])
            insertion = (current_distance[j - 1] + 1, current_insertions[j - 1] + 1,
                         current_deletions[j - 1], current_substitutions[j - 1])
            substitution = (previous_distance[j - 1] + 1, previous_insertions[j - 1],
                            previous_deletions[j - 1], previous_substitutions[j - 1] + 1)
            distance, insertions, deletions, substitutions = min(
                (deletion, insertion, substitution), key=lambda item: item[0]
            )
            current_distance[j] = distance
            current_insertions[j] = insertions
            current_deletions[j] = deletions
            current_substitutions[j] = substitutions
        previous_distance = current_distance
        previous_insertions = current_insertions
        previous_deletions = current_deletions
        previous_substitutions = current_substitutions
    return (previous_distance[-1], previous_insertions[-1],
            previous_deletions[-1], previous_substitutions[-1])


def _distance_only(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance for cpWER without allocating operation counters."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                previous[j - 1] if ref == hyp else min(
                    previous[j] + 1, current[j - 1] + 1, previous[j - 1] + 1
                )
            )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if ref == hyp:
        return {"wer": 0.0, "insertions": 0, "deletions": 0,
                "substitutions": 0, "reference_words": len(ref),
                "hypothesis_words": len(hyp)}
    distance, insertions, deletions, substitutions = edit_distance(ref, hyp)
    return {"wer": distance / max(1, len(ref)), "insertions": insertions,
            "deletions": deletions, "substitutions": substitutions,
            "reference_words": len(ref), "hypothesis_words": len(hyp)}


def cpwer(reference: dict[str, str], hypothesis: dict[str, str]) -> dict:
    """Minimum concatenated WER over speaker permutations (small-N fallback)."""
    import itertools
    ref_speakers = list(reference)
    hyp_speakers = list(hypothesis)
    if len(ref_speakers) != len(hyp_speakers):
        return {"cpwer": None, "reason": "speaker_count_mismatch"}
    ref_tokens = {speaker: normalize(reference[speaker]) for speaker in ref_speakers}
    hyp_tokens = {speaker: normalize(hypothesis[speaker]) for speaker in hyp_speakers}
    pair_scores = {}
    for ref_speaker in ref_speakers:
        for hyp_speaker in hyp_speakers:
            tokens = ref_tokens[ref_speaker]
            candidate = hyp_tokens[hyp_speaker]
            distance = 0 if tokens == candidate else _distance_only(tokens, candidate)
            pair_scores[(ref_speaker, hyp_speaker)] = distance / max(1, len(tokens))
    best = None
    best_mapping = None
    for permutation in itertools.permutations(hyp_speakers):
        total = sum(pair_scores[(r, h)] for r, h in zip(ref_speakers, permutation))
        if best is None or total < best:
            best, best_mapping = total, dict(zip(ref_speakers, permutation))
    denominator = max(1, sum(len(normalize(value)) for value in reference.values()))
    return {"cpwer": best / len(ref_speakers), "mapping": best_mapping,
            "reference_words": denominator}


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    speaker: str


def diarization_confusion(reference: list[Span], hypothesis: list[Span], *,
                          allow_overlap: bool = False) -> dict:
    """Time-weighted confusion/coverage; official DER remains optional.

    When the reference explicitly contains overlapping speakers, cross-speaker
    overlap is not counted as confusion by this fallback. Official DER tools
    should be used for a definitive overlap-aware score.
    """
    total = sum(max(0, item.end - item.start) for item in reference)
    missing = 0.0
    confusion = 0.0
    missed_turns = 0
    for item in reference:
        annotated_overlap = allow_overlap and any(
            other is not item and other.speaker != item.speaker
            and min(item.end, other.end) > max(item.start, other.start)
            for other in reference
        )
        covered = 0.0
        for candidate in hypothesis:
            overlap = max(0, min(item.end, candidate.end) - max(item.start, candidate.start))
            if overlap:
                covered += overlap
                if candidate.speaker != item.speaker and not annotated_overlap:
                    confusion += overlap
        missing += max(0, item.end - item.start - covered)
        if covered <= 0:
            missed_turns += 1
    return {
        # Quota del tempo dei turni di riferimento non coperta da alcun
        # segmento. ATTENZIONE: quando i confini del riferimento sono grossolani
        # (fine del turno = inizio del successivo) il turno contiene anche le
        # sue pause, quindi questo valore misura in gran parte silenzio interno
        # e non parlato perso. Inseguirlo porterebbe ad allargare i segmenti,
        # cioe' il contrario di quello che serve.
        "missing": missing / max(total, 1e-9),
        "confusion": confusion / max(total, 1e-9),
        "reference_speech": total,
        # Turni senza nemmeno un segmento sovrapposto: questa e' perdita vera e
        # non dipende dalla larghezza degli intervalli.
        "missed_turns": missed_turns,
        "missed_turn_rate": missed_turns / max(len(reference), 1),
        "hypothesis_speech": sum(max(0, item.end - item.start) for item in hypothesis),
    }
