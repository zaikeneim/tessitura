import unittest

from src.models.session import Segment, Session, Word
from src.processing.diarize import Turn
from src.processing.merge import WordUnit, assign_speakers
from src.processing.transcript_review import review_labels


class TranscriptReviewTests(unittest.TestCase):
    def test_ambient_review_keeps_short_interjection_when_supported_by_turn(self):
        units = [
            WordUnit(0.00, 0.60, " una"),
            WordUnit(0.60, 1.20, " frase."),
            WordUnit(1.20, 1.38, " sì"),
            WordUnit(1.38, 2.00, " continua."),
        ]
        turns = [
            Turn(0.00, 1.20, "A"),
            Turn(1.20, 1.38, "B"),
            Turn(1.38, 2.00, "A"),
        ]
        baseline = assign_speakers(units, turns, preserve_short_turns=False)
        result = review_labels(
            units, turns, baseline, profile="stereo_ambient", preserve_short_turns=False
        )
        # Il turno B copre esattamente "sì": assorbirlo cancellerebbe un vero
        # cambio di voce, che è il difetto R1. Prima di quella correzione qui
        # si otteneva ["A", "A", "A", "A"].
        self.assertEqual(result.labels, ["A", "A", "B", "A"])
        self.assertEqual(result.metadata["selected"], "preserve_short_turns")
        self.assertGreater(
            result.metadata["candidate_agreement"]["preserve_short_turns"],
            result.metadata["candidate_agreement"]["baseline"],
        )

    def test_alternating_speakers_survive_review(self):
        """Controesempio R1: turni A/B perfetti non devono collassare su una voce.

        Con la penalità cumulativa non normalizzata il baseline corretto
        totalizzava -0,52 contro 0,50 e la revisione sceglieva una voce sola.
        """
        units = [WordUnit(i * 0.5, (i + 1) * 0.5, f" w{i}") for i in range(20)]
        turns = [
            Turn(i * 0.5, (i + 1) * 0.5, "A" if i % 2 == 0 else "B") for i in range(20)
        ]
        baseline = ["A" if i % 2 == 0 else "B" for i in range(20)]
        result = review_labels(
            units, turns, baseline, profile="stereo_ambient", preserve_short_turns=True
        )
        self.assertEqual(result.labels, baseline)
        self.assertEqual(result.metadata["selected"], "baseline")
        self.assertEqual(result.metadata["changed_words"], 0)
        self.assertEqual(sorted(set(result.labels)), ["A", "B"])

    def test_unverified_speaker_count_needs_a_wider_margin(self):
        """Senza numero di voci dichiarato non si reintroducono turni brevi al margine.

        Stesso caso del test precedente (margine 0,09): con il conteggio noto
        l'interiezione viene conservata, con il conteggio automatico si resta
        sul baseline e il segmento viene segnalato invece che deciso.
        """
        units = [
            WordUnit(0.00, 0.60, " una"),
            WordUnit(0.60, 1.20, " frase."),
            WordUnit(1.20, 1.38, " sì"),
            WordUnit(1.38, 2.00, " continua."),
        ]
        turns = [Turn(0.00, 1.20, "A"), Turn(1.20, 1.38, "B"), Turn(1.38, 2.00, "A")]
        baseline = assign_speakers(units, turns, preserve_short_turns=False)

        unverified = review_labels(
            units, turns, baseline, profile="stereo_ambient",
            preserve_short_turns=False, speaker_count_known=False,
        )
        self.assertEqual(unverified.labels, baseline)
        self.assertEqual(unverified.metadata["selected"], "baseline")
        self.assertTrue(unverified.metadata["needs_review"])

    def test_degenerate_input_declares_no_second_candidate(self):
        """Senza turni non c'è confronto: non va dichiarato un candidato fittizio."""
        units = [WordUnit(0.0, 1.0, " ciao")]
        result = review_labels(units, [], ["A"], profile="stereo_ambient")
        self.assertEqual(result.metadata["candidate_scores"], {"baseline": 0.0})
        self.assertEqual(result.labels, ["A"])

    def test_candidate_is_named_after_what_it_does(self):
        """Il candidato alternativo cambia significato col baseline: va nominato."""
        units = [WordUnit(0.0, 0.5, " a"), WordUnit(0.5, 0.7, " b"), WordUnit(0.7, 1.2, " c")]
        turns = [Turn(0.0, 0.5, "A"), Turn(0.5, 0.7, "B"), Turn(0.7, 1.2, "A")]
        preserving = review_labels(
            units, turns, ["A", "A", "A"], profile="stereo_ambient",
            preserve_short_turns=True,
        )
        absorbing = review_labels(
            units, turns, ["A", "B", "A"], profile="stereo_ambient",
            preserve_short_turns=False,
        )
        self.assertIn("absorb_short_turns", preserving.metadata["candidate_scores"])
        self.assertIn("preserve_short_turns", absorbing.metadata["candidate_scores"])

    def test_mic_system_review_never_overwrites_channel_assignment(self):
        units = [WordUnit(0, 1, " ciao"), WordUnit(1, 2, " remoto")]
        turns = [Turn(0, 1, "A"), Turn(1, 2, "B")]
        result = review_labels(
            units, turns, ["IO", "REMOTO"], profile="mic_system"
        )
        self.assertEqual(result.labels, ["IO", "REMOTO"])
        self.assertEqual(result.metadata["selected"], "channel_hint")

    def test_session_roundtrip_keeps_raw_and_review_metadata(self):
        segment = Segment(
            0, 1, "ciao", speaker="A", words=[Word(0, 1, "ciao", 0.9)]
        )
        session = Session(
            segments=[segment], raw_segments=[segment],
            transcript_review={"mode": "safe", "needs_review": True},
        )
        restored = Session.from_dict(session.to_dict())
        self.assertEqual(restored.raw_segments[0].text, "ciao")
        self.assertTrue(restored.transcript_review["needs_review"])



class DiarizationOutputsTests(unittest.TestCase):
    """R12: ordinaria, esclusiva e sovrapposizioni devono restare distinte."""

    def test_overlapping_spans_finds_only_real_overlap(self):
        from src.processing.diarize import Turn as T, _overlapping_spans

        spans = _overlapping_spans([T(0, 5, "A"), T(3, 8, "B"), T(20, 22, "A")])
        self.assertEqual(spans, [(3, 5)])

    def test_adjacent_turns_are_not_overlap(self):
        from src.processing.diarize import Turn as T, _overlapping_spans

        self.assertEqual(_overlapping_spans([T(0, 5, "A"), T(5, 9, "B")]), [])

    def test_three_voices_at_once_count_once(self):
        from src.processing.diarize import Turn as T, _overlapping_spans

        spans = _overlapping_spans([T(0, 9, "A"), T(1, 8, "B"), T(2, 7, "C")])
        self.assertEqual(spans, [(1, 8)])

    def test_result_falls_back_to_ordinary_without_exclusive_output(self):
        from src.processing.diarize import DiarizationResult, Turn as T

        turns = [T(0, 5, "A")]
        result = DiarizationResult(turns=turns, exclusive=turns, overlaps=[])
        self.assertEqual(result.exclusive, turns)
        self.assertEqual(result.overlap_seconds(), 0.0)


if __name__ == "__main__":
    unittest.main()
