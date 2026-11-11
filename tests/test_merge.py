"""Test dell'assegnazione delle voci (il punto dove nascevano gli errori peggiori).

Si eseguono senza GPU e senza modelli: `python -m pytest tests/` oppure
`python tests/test_merge.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.diarize import Turn  # noqa: E402
from src.processing.merge import (  # noqa: E402
    WordUnit,
    assign_speakers,
    build_segments,
)


def _speakers(words, turns):
    return assign_speakers(words, turns)


def test_frase_non_spezzata_a_meta():
    """Regressione: "Si. Per" / "rendere una linea" finivano a due voci diverse.

    Il turno di SPEAKER_01 sfora di 0.03s sulla parola "Per": senza il voto per
    frase, quella parola veniva strappata alla frase a cui appartiene.
    """
    words = [
        WordUnit(69.00, 69.50, " Ho"),
        WordUnit(69.50, 70.20, " visto"),
        WordUnit(70.20, 70.90, " tutto."),
        WordUnit(71.12, 71.52, " Sì."),
        WordUnit(72.02, 72.42, " Per"),
        WordUnit(72.42, 73.20, " rendere"),
        WordUnit(73.20, 73.80, " una"),
        WordUnit(73.80, 74.30, " linea."),
    ]
    turns = [
        Turn(60.00, 71.00, "SPEAKER_00"),
        Turn(71.10, 72.45, "SPEAKER_01"),
        Turn(72.45, 90.00, "SPEAKER_00"),
    ]
    labels = _speakers(words, turns)
    assert labels[4:] == ["SPEAKER_00"] * 4, "la frase deve restare di una voce sola"
    assert labels[3] == "SPEAKER_01", "l'intercalare di un'altra voce va conservato"

    segments = build_segments(words, labels)
    assert [s.text for s in segments] == [
        "Ho visto tutto.",
        "Sì.",
        "Per rendere una linea.",
    ]


def test_cambio_di_voce_lungo_viene_rispettato():
    """Due tratti lunghi di voci diverse restano divisi anche senza pausa."""
    words = [WordUnit(i, i + 0.9, f" p{i}") for i in range(8)]
    turns = [Turn(0, 4, "A"), Turn(4, 8, "B")]
    assert _speakers(words, turns) == ["A"] * 4 + ["B"] * 4


def test_dialogo_multi_voce_preserva_interiezione_breve():
    """Con il numero di voci noto, una risposta breve tra due turni resta visibile."""
    words = [
        WordUnit(0.00, 0.20, " Sì"),
        WordUnit(0.20, 0.40, " No"),
        WordUnit(0.40, 0.60, " davvero."),
    ]
    turns = [Turn(0.00, 0.20, "A"), Turn(0.20, 0.40, "B"), Turn(0.40, 1.0, "A")]
    assert _speakers(words, turns) == ["A", "A", "A"]
    from src.processing.merge import assign_speakers
    assert assign_speakers(words, turns, preserve_short_turns=True) == ["A", "B", "A"]


def test_turno_spurio_breve_ignorato():
    """Un turno di 0.2s dentro una frase e' rumore della diarizzazione."""
    words = [
        WordUnit(0.00, 0.60, " questa"),
        WordUnit(0.60, 1.20, " e"),
        WordUnit(1.25, 1.45, " una"),
        WordUnit(1.50, 2.10, " frase"),
        WordUnit(2.10, 2.80, " sola."),
    ]
    turns = [Turn(0, 1.2, "A"), Turn(1.2, 1.45, "B"), Turn(1.45, 3.0, "A")]
    assert set(_speakers(words, turns)) == {"A"}


def test_dialogo_alternato():
    words = [
        WordUnit(0.0, 0.8, " Ciao"), WordUnit(0.8, 1.4, " come"), WordUnit(1.4, 2.0, " stai?"),
        WordUnit(2.6, 3.2, " Bene"), WordUnit(3.2, 3.9, " grazie"), WordUnit(3.9, 4.5, " tu?"),
        WordUnit(5.1, 5.8, " Anche"), WordUnit(5.8, 6.4, " io."),
    ]
    turns = [Turn(0, 2.2, "A"), Turn(2.4, 4.7, "B"), Turn(4.9, 6.6, "A")]
    assert _speakers(words, turns) == ["A"] * 3 + ["B"] * 3 + ["A"] * 2


def test_parola_a_durata_zero():
    """Whisper a volte emette start == end: non deve far sparire l'assegnazione."""
    words = [WordUnit(5.0, 5.0, " ciao"), WordUnit(5.2, 5.9, " mondo")]
    turns = [Turn(4.5, 6.0, "A")]
    assert _speakers(words, turns) == ["A", "A"]


def test_senza_turni_di_diarizzazione():
    words = [WordUnit(0, 1, " a"), WordUnit(1, 2, " b")]
    assert _speakers(words, []) == ["SPEAKER_00"] * 2


def test_input_vuoto():
    assert _speakers([], [Turn(0, 1, "A")]) == []
    assert build_segments([], []) == []


def test_segmenti_separati_da_una_pausa_lunga():
    """Stessa voce ma con un silenzio in mezzo: due segmenti, non uno solo."""
    words = [
        WordUnit(0.0, 0.5, " uno"),
        WordUnit(0.5, 1.0, " due."),
        WordUnit(30.0, 30.5, " tre"),
    ]
    turns = [Turn(0, 40, "A")]
    segments = build_segments(words, _speakers(words, turns))
    assert len(segments) == 2
    assert segments[0].text == "uno due."
    assert segments[1].text == "tre"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FALLITO {name}: {exc}")
    print(f"\n{'tutti i test passano' if not failures else f'{failures} test falliti'}")
    raise SystemExit(1 if failures else 0)
