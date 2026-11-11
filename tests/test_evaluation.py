import unittest
from src.processing.evaluation import Span, cpwer, diarization_confusion, wer


class EvaluationTests(unittest.TestCase):
    def test_wer_reports_edit_components(self):
        result = wer("uno due tre", "uno quattro")
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["substitutions"], 1)
        self.assertAlmostEqual(result["wer"], 2 / 3)

    def test_cpwer_finds_speaker_permutation(self):
        result = cpwer({"ref-a": "ciao mondo", "ref-b": "buongiorno"},
                       {"hyp-0": "buongiorno", "hyp-1": "ciao mondo"})
        self.assertEqual(result["cpwer"], 0)
        self.assertEqual(result["mapping"], {"ref-a": "hyp-1", "ref-b": "hyp-0"})

    def test_diarization_fallback_exposes_missing_and_confusion(self):
        result = diarization_confusion([Span(0, 2, "A")], [Span(0, 1, "B")])
        self.assertEqual(result["missing"], 0.5)
        self.assertEqual(result["confusion"], 0.5)

    def test_turni_larghi_non_si_scambiano_per_parlato_perso(self):
        """`missing` con confini grossolani misura silenzio, non parole perse.

        I turni del riferimento finiscono dove inizia il successivo, quindi
        contengono le proprie pause; i segmenti riconosciuti sono stretti sul
        parlato. Senza il conteggio dei turni davvero scoperti, quel divario
        sembrerebbe un terzo di parlato mancante.
        """
        reference = [Span(0, 10, "A"), Span(10, 20, "B")]
        hypothesis = [Span(1, 4, "A"), Span(12, 15, "B")]

        result = diarization_confusion(reference, hypothesis)

        self.assertAlmostEqual(result["missing"], 0.7)     # quasi tutto silenzio
        self.assertEqual(result["missed_turns"], 0)        # nessun turno perso
        self.assertEqual(result["missed_turn_rate"], 0.0)
        self.assertEqual(result["hypothesis_speech"], 6)

    def test_un_turno_senza_riscontro_e_perdita_vera(self):
        reference = [Span(0, 10, "A"), Span(10, 20, "B")]
        hypothesis = [Span(1, 9, "A")]

        result = diarization_confusion(reference, hypothesis)

        self.assertEqual(result["missed_turns"], 1)
        self.assertEqual(result["missed_turn_rate"], 0.5)

    def test_explicit_reference_overlap_is_not_counted_as_confusion(self):
        reference = [Span(0, 2, "A"), Span(1, 3, "B")]
        result = diarization_confusion(reference, reference, allow_overlap=True)
        self.assertEqual(result["missing"], 0.0)
        self.assertEqual(result["confusion"], 0.0)


if __name__ == "__main__":
    unittest.main()
