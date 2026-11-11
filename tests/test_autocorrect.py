"""Test per l'autocorrezione di troncamenti e sovrapposizioni."""

import unittest

from src.models.session import Segment, Word
from src.processing.autocorrect import (
    autocorrect_truncations,
    autocorrect_overlaps,
    ends_with_open_clause,
    is_terminal_text,
)
from src.processing.merge import WordUnit, build_segments


class AutocorrectTests(unittest.TestCase):
    def test_ends_with_open_clause(self):
        self.assertTrue(ends_with_open_clause("io penso che"))
        self.assertTrue(ends_with_open_clause("dobbiamo andare a,"))
        self.assertTrue(ends_with_open_clause("perché"))
        self.assertTrue(ends_with_open_clause("volevo dire-"))
        self.assertFalse(ends_with_open_clause("ho finito il lavoro."))
        self.assertFalse(ends_with_open_clause("tutto chiaro!"))

    def test_is_terminal_text(self):
        self.assertTrue(is_terminal_text("Fine periodo."))
        self.assertTrue(is_terminal_text("Cosa dici?"))
        self.assertTrue(is_terminal_text("Attento!"))
        self.assertTrue(is_terminal_text("Sospeso…"))
        self.assertFalse(is_terminal_text("Frase non finita"))
        self.assertFalse(is_terminal_text("Frase con virgola,"))

    def test_autocorrect_truncation_when_interrupted(self):
        segments = [
            Segment(0.0, 2.5, "Volevo precisare che", speaker="SPEAKER_00"),
            Segment(2.6, 5.0, "Non sono d'accordo con questa linea.", speaker="SPEAKER_01"),
        ]
        corrected = autocorrect_truncations(segments)
        self.assertEqual(corrected[0].text, "Volevo precisare che…")
        self.assertIn("frase_troncata", corrected[0].warnings)
        self.assertEqual(corrected[1].text, "Non sono d'accordo con questa linea.")
        self.assertNotIn("frase_troncata", corrected[1].warnings)

    def test_autocorrect_truncation_trailing_comma(self):
        segments = [
            Segment(0.0, 2.0, "Secondo me,", speaker="SPEAKER_00"),
            Segment(2.1, 4.0, "Esatto.", speaker="SPEAKER_01"),
        ]
        corrected = autocorrect_truncations(segments)
        self.assertEqual(corrected[0].text, "Secondo me…")
        self.assertIn("frase_troncata", corrected[0].warnings)

    def test_same_speaker_continuation_preserves_flow(self):
        segments = [
            Segment(0.0, 2.0, "Abbiamo deciso di", speaker="SPEAKER_00"),
            Segment(2.2, 4.0, "proseguire con il piano.", speaker="SPEAKER_00"),
        ]
        corrected = autocorrect_truncations(segments)
        # Tra segmenti della stessa voce non si inserisce l'ellissi di interruzione
        self.assertEqual(corrected[0].text, "Abbiamo deciso di")
        self.assertNotIn("frase_troncata", corrected[0].warnings)

    def test_autocorrect_overlaps_detection(self):
        segments = [
            Segment(10.0, 15.0, "Stiamo esaminando il bilancio trimestrale.", speaker="SPEAKER_00"),
            Segment(12.0, 13.0, "Sì.", speaker="SPEAKER_01"),
        ]
        overlaps = [(12.0, 13.0)]
        corrected = autocorrect_overlaps(segments, overlaps=overlaps)
        self.assertIn("voce_sovrapposta", corrected[0].warnings)
        self.assertIn("voce_sovrapposta", corrected[1].warnings)

        # Il segmento di SPEAKER_00 deve avere SPEAKER_01 nei candidati
        spk_01_cands = [c["speaker"] for c in corrected[0].candidates]
        self.assertIn("SPEAKER_01", spk_01_cands)

    def test_build_segments_integrates_autocorrection(self):
        units = [
            WordUnit(0.0, 0.5, "Volevo"),
            WordUnit(0.5, 1.0, "dire"),
            WordUnit(1.0, 1.4, "che"),
            WordUnit(1.5, 2.0, "No!"),
        ]
        labels = ["A", "A", "A", "B"]
        segments = build_segments(units, labels, autocorrect_enabled=True)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "A")
        self.assertEqual(segments[0].text, "Volevo dire che…")
        self.assertIn("frase_troncata", segments[0].warnings)
        self.assertEqual(segments[1].speaker, "B")
        self.assertEqual(segments[1].text, "No!")


if __name__ == "__main__":
    unittest.main()
