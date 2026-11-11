"""Controllo deterministico della trascrizione prima della visualizzazione.

La diarizzazione ambientale non offre una verità assoluta: un cambio di voce
molto breve può essere reale oppure un errore di allineamento. Questo modulo
non inventa voci e non riscrive il testo. Confronta due attribuzioni già
calcolate dal merge e sceglie quella che aderisce meglio ai turni pyannote,
penalizzando i cambi di voce non sostenuti dall'audio.

La decisione è conservativa. Se i due candidati sono quasi equivalenti si
mantiene il risultato iniziale e il segmento viene marcato come ambiguo per
una revisione manuale o LLM successiva.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.session import Segment
from . import merge
from .audio_profile import AudioProfile
from .diarize import Turn


@dataclass(frozen=True)
class ReviewResult:
    labels: list[str]
    segments: list[Segment]
    metadata: dict


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _agreement(units: list[merge.WordUnit], turns: list[Turn], labels: list[str]) -> float:
    """Quota della durata delle parole coperta dal turno della stessa voce."""
    total = 0.0
    matched = 0.0
    for unit, label in zip(units, labels):
        duration = max(0.05, unit.end - unit.start)
        total += duration
        matched += max(
            (_overlap(unit.start, unit.end, turn.start, turn.end)
             for turn in turns if turn.speaker == label),
            default=0.0,
        )
    return matched / total if total else 0.0


def _supported(unit: merge.WordUnit, label: str, turns: list[Turn]) -> bool:
    """La parola ricade in un turno attribuito proprio a quella voce."""
    return any(
        turn.speaker == label and _overlap(unit.start, unit.end, turn.start, turn.end) > 0.0
        for turn in turns
    )


# Peso della penalità rispetto all'accordo. Entrambi i termini stanno in 0..1,
# quindi la penalità non può più superare l'evidenza acustica.
_UNSUPPORTED_WEIGHT = 0.5
# Differenza di punteggio sotto la quale i due candidati si considerano equivalenti.
_MARGIN = 0.08
# Soglia più severa quando il numero di voci non è stato dichiarato dall'utente:
# lì i turni brevi di pyannote possono essere sovrasegmentazione, e l'accordo non
# sa distinguere un turno vero da uno inventato. Valore di partenza prudente, da
# calibrare sul corpus annotato della fase 0: non è una soglia misurata.
_MARGIN_UNVERIFIED_COUNT = 0.20
# Sotto questa pausa un cambio di voce va sostenuto dai turni per essere credibile.
_PAUSE_FOR_FREE_CHANGE = 0.30


def _unsupported_change_rate(
    units: list[merge.WordUnit], turns: list[Turn], labels: list[str]
) -> float:
    """Quota dei cambi di voce che i turni non sostengono.

    Prima si sommava una penalità fissa per ogni cambio ravvicinato, senza
    normalizzare e senza guardare i turni: su un dialogo serrato la somma
    superava l'accordo massimo possibile e vinceva sempre l'attribuzione con
    meno cambi, cioè una voce sola. Qui si penalizza solo il cambio che l'audio
    non sostiene, e si misura una quota, non un totale che cresce col parlato.
    """
    changes = 0
    unsupported = 0
    for previous, current, left, right in zip(labels, labels[1:], units, units[1:]):
        if previous == current:
            continue
        changes += 1
        if max(0.0, right.start - left.end) >= _PAUSE_FOR_FREE_CHANGE:
            continue  # pausa percepibile: il cambio è plausibile di per sé
        if not (_supported(left, previous, turns) and _supported(right, current, turns)):
            unsupported += 1
    return unsupported / changes if changes else 0.0


def _score(units: list[merge.WordUnit], turns: list[Turn], labels: list[str]) -> float:
    return _agreement(units, turns, labels) - _UNSUPPORTED_WEIGHT * _unsupported_change_rate(
        units, turns, labels
    )


def review_labels(
    units: list[merge.WordUnit],
    turns: list[Turn],
    baseline: list[str],
    *,
    profile: AudioProfile | str | None = None,
    preserve_short_turns: bool = False,
    speaker_count_known: bool = True,
) -> ReviewResult:
    """Valuta due strategie di attribuzione senza modificare il testo.

    Il candidato conservativo è sempre quello ricevuto dal pipeline. Il
    candidato alternativo usa i turni parola-per-parola; è utile soprattutto
    su un file ambientale con più persone e interiezioni brevi.
    """
    profile = AudioProfile.parse(profile)
    baseline = list(baseline)
    # Con due sorgenti il channel hint è un'informazione osservata (non una
    # semplice euristica): non va sovrascritto da una seconda attribuzione
    # basata soltanto sui turni acustici.
    if profile is AudioProfile.MIC_SYSTEM:
        return ReviewResult(
            labels=baseline,
            segments=merge.build_segments(units, baseline),
            metadata={
                "mode": "safe",
                "profile": profile.value,
                "selected": "channel_hint",
                "candidate_scores": {},
                "ambiguous_words": 0,
                "needs_review": False,
                "changed_words": 0,
            },
        )
    if not units or not turns or len(units) != len(baseline):
        return ReviewResult(
            labels=baseline,
            segments=merge.build_segments(units, baseline),
            metadata={
                "mode": "safe",
                "profile": profile.value if profile else None,
                "selected": "baseline",
                # Senza turni non esiste un secondo candidato: dichiararne uno
                # con punteggio zero faceva sopravvivere il vecchio nome e
                # lasciava credere che un confronto fosse avvenuto.
                "candidate_scores": {"baseline": 0.0},
                "ambiguous": True if units and not turns else False,
                "changed_words": 0,
            },
        )

    # Il candidato alternativo ribalta il trattamento dei turni brevi: chiamarlo
    # sempre "word_level" nascondeva che, con un baseline che li preserva,
    # l'alternativa è proprio quella che li assorbe.
    alternate_preserves = not preserve_short_turns
    alternate_name = "preserve_short_turns" if alternate_preserves else "absorb_short_turns"
    alternate = merge.assign_speakers(
        units, turns, preserve_short_turns=alternate_preserves
    )
    candidates = {"baseline": baseline, alternate_name: alternate}
    scores = {name: _score(units, turns, labels) for name, labels in candidates.items()}
    margin = scores[alternate_name] - scores["baseline"]
    # Una piccola differenza è rumore: mantenere il baseline evita oscillazioni
    # tra due esecuzioni e rende la revisione riproducibile.
    # Con il conteggio delle voci non dichiarato si alza l'asticella invece di
    # astenersi del tutto: astenersi sempre riporterebbe il difetto R1 proprio
    # sul percorso predefinito della GUI, dove i turni brevi veri sparivano.
    threshold = _MARGIN if speaker_count_known else _MARGIN_UNVERIFIED_COUNT
    use_alternate = margin > threshold
    selected_name = alternate_name if use_alternate else "baseline"
    selected = list(candidates[selected_name])
    changed = sum(a != b for a, b in zip(baseline, selected))
    # Astensione: i due candidati divergono ma nessuno dei due è chiaramente
    # sostenuto dall'audio. Si tiene il baseline e si segnala la revisione,
    # invece di far decidere una differenza che sta dentro il rumore.
    disagreement = sum(a != b for a, b in zip(baseline, alternate))
    # Segnalato quando il baseline vince senza essere chiaramente migliore: la
    # soglia è quella effettivamente usata, altrimenti il caso incerto per
    # conteggio non dichiarato verrebbe deciso in silenzio.
    ambiguous = disagreement if (not use_alternate and margin > -threshold) else 0
    return ReviewResult(
        labels=selected,
        segments=merge.build_segments(units, selected),
        metadata={
            "mode": "safe",
            "profile": profile.value if profile else None,
            "selected": selected_name,
            "candidate_scores": {name: round(value, 6) for name, value in scores.items()},
            # Evidenza dietro il punteggio, per poter rivedere la decisione.
            "candidate_agreement": {
                name: round(_agreement(units, turns, labels), 6)
                for name, labels in candidates.items()
            },
            "candidate_unsupported_changes": {
                name: round(_unsupported_change_rate(units, turns, labels), 6)
                for name, labels in candidates.items()
            },
            "margin": round(margin, 6),
            "changed_words": changed,
            "ambiguous_words": ambiguous,
            "needs_review": bool(ambiguous),
        },
    )
