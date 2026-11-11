"""Memoria adattiva locale: estrazione correzioni e profili vocali."""

import unittest

from src.processing.adaptation import (
    match_speaker_profiles,
    mean_embedding,
    normalize_embedding,
    transcription_corrections,
)
from src.processing.diarize import _speaker_embedding_map


class TranscriptionCorrectionTests(unittest.TestCase):
    def test_una_correzione_lessicale_diventa_un_termine(self):
        result = transcription_corrections(
            "Domani incontriamo Chisrael nel porto.",
            "Domani incontriamo Kisrael nel porto.",
        )

        self.assertEqual(result, [{"before": "Chisrael", "after": "Kisrael"}])

    def test_sola_punteggiatura_non_diventa_memoria(self):
        self.assertEqual(transcription_corrections("Ciao mondo", "Ciao, mondo!"), [])

    def test_frasi_lunghe_non_inquinano_il_prompt(self):
        result = transcription_corrections("", "uno due tre quattro cinque sei sette")
        self.assertEqual(result, [])

    def test_un_cambio_di_contenuto_non_diventa_suggerimento_whisper(self):
        result = transcription_corrections(
            "Andiamo tutti a casa", "Andiamo tutti in ufficio"
        )
        self.assertEqual(result, [])


class SpeakerProfileTests(unittest.TestCase):
    def test_normalizza_e_media_scartando_campioni_invalidi(self):
        self.assertEqual(normalize_embedding([3, 4]), [0.6, 0.8])
        profile = mean_embedding(([1, 0], [0.8, 0.2], [float("nan"), 1]))
        self.assertIsNotNone(profile)
        self.assertGreater(profile[0], 0.98)

    def test_abbinamento_globale_uno_a_uno(self):
        matches = match_speaker_profiles(
            {"SPEAKER_00": [0.05, 1], "SPEAKER_01": [1, 0.05]},
            {"Anna": [0, 1], "Marco": [1, 0]},
        )

        self.assertEqual(matches["SPEAKER_00"]["name"], "Anna")
        self.assertEqual(matches["SPEAKER_01"]["name"], "Marco")

    def test_sotto_soglia_la_voce_resta_senza_nome(self):
        matches = match_speaker_profiles(
            {"SPEAKER_00": [1, 0]}, {"Anna": [0, 1]}, threshold=0.72
        )
        self.assertEqual(matches, {})

    def test_due_voci_quasi_identiche_restano_da_confermare(self):
        matches = match_speaker_profiles(
            {"SPEAKER_00": [1, 0], "SPEAKER_01": [0.999, 0.02]},
            {"Marco": [1, 0]},
        )
        self.assertEqual(matches, {})

    def test_centroidi_pyannote_restano_allineati_alle_etichette(self):
        class Annotation:
            @staticmethod
            def labels():
                return ["SPEAKER_01", "SPEAKER_00"]

        result = _speaker_embedding_map(Annotation(), [[0, 2], [3, 0]])

        self.assertEqual(result["SPEAKER_01"], [0.0, 1.0])
        self.assertEqual(result["SPEAKER_00"], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
