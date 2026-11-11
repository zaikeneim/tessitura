"""Test di integrazione overlap e pipeline simultanea (P07).

Verifica che due voci simultanee generino due eventi distinti preservati,
che l'ordine delle ipotesi non introduca permutazioni spurie, e che
le regioni di overlap proteggano le interiezioni brevi dal livellamento.
"""

from __future__ import annotations

import unittest

from src.models.session import Word
from src.models.transcript_artifacts import ASRHypothesis, Utterance
from src.processing.diarize import Turn
from src.processing.merge import (
    WordUnit,
    assign_speakers,
    build_utterances,
    utterances_to_segments,
)


class OverlapPipelineTests(unittest.TestCase):
    def test_due_voci_simultanee_preservate(self):
        """Due ipotesi ASR nello stesso intervallo temporale generano due Utterance collegate."""
        hyp_a = ASRHypothesis(
            hypothesis_id="hyp_001",
            view_id="view_raw",
            start_s=10.0,
            end_s=14.0,
            text="Stavo dicendo che il bilancio trimestrale e' positivo.",
            words=[
                Word(10.0, 10.8, "Stavo"),
                Word(10.8, 11.5, "dicendo"),
                Word(11.5, 12.2, "che"),
                Word(12.2, 13.0, "il"),
                Word(13.0, 14.0, "bilancio"),
            ],
            parameters={"speaker": "SPEAKER_00"},
        )
        hyp_b = ASRHypothesis(
            hypothesis_id="hyp_002",
            view_id="view_separated_ch1",
            start_s=11.2,
            end_s=12.5,
            text="Esattamente d'accordo.",
            words=[
                Word(11.2, 11.8, "Esattamente"),
                Word(11.8, 12.5, "d'accordo"),
            ],
            parameters={"speaker": "SPEAKER_01"},
        )

        utterances = build_utterances([hyp_a, hyp_b])

        self.assertEqual(len(utterances), 2, "Entrambe le utterance simultanee devono essere preservate")
        utt0 = next(u for u in utterances if u.speaker == "SPEAKER_00")
        utt1 = next(u for u in utterances if u.speaker == "SPEAKER_01")

        self.assertEqual(utt0.text, "Stavo dicendo che il bilancio trimestrale e' positivo.")
        self.assertEqual(utt1.text, "Esattamente d'accordo.")
        self.assertIsNotNone(utt0.overlap_group_id)
        self.assertEqual(utt0.overlap_group_id, utt1.overlap_group_id, "Devono condividere lo stesso overlap_group_id")
        self.assertNotEqual(utt0.stream_id, utt1.stream_id, "I due stream simultanei devono avere stream_id distinti")

        # Conversione compatibile in segmenti
        segments = utterances_to_segments(utterances)
        self.assertEqual(len(segments), 2)
        self.assertEqual({s.speaker for s in segments}, {"SPEAKER_00", "SPEAKER_01"})

    def test_overlap_protegge_interiezione_in_assign_speakers(self):
        """Una parola breve all'interno di una zona di overlap accertata non viene appiattita."""
        words = [
            WordUnit(0.00, 0.80, " Questa"),
            WordUnit(0.80, 1.40, " e'"),
            WordUnit(1.45, 1.65, " No"),      # Breve interiezione (0.2s)
            WordUnit(1.70, 2.30, " una"),
            WordUnit(2.30, 3.00, " frase."),
        ]
        turns = [
            Turn(0.0, 1.4, "SPEAKER_00"),
            Turn(1.4, 1.7, "SPEAKER_01"),
            Turn(1.7, 3.0, "SPEAKER_00"),
        ]

        # Senza overlap specificato, una parola di 0.2s tra la stessa voce viene appiattita
        labels_no_ov = assign_speakers(words, turns, overlaps=None)
        self.assertEqual(labels_no_ov[2], "SPEAKER_00")

        # Con la regione di overlap attestata, la voce SPEAKER_01 viene preservata
        labels_with_ov = assign_speakers(words, turns, overlaps=[(1.40, 1.70)])
        self.assertEqual(labels_with_ov[2], "SPEAKER_01", "L'interiezione in zona di overlap deve essere preservata")

    def test_permutazione_ipotesi_non_cambia_risultato(self):
        """L'ordine di inserimento delle ipotesi non altera il raggruppamento e l'attribuzione."""
        hyp_a = ASRHypothesis(
            hypothesis_id="h1",
            start_s=2.0,
            end_s=5.0,
            text="Intervento principale",
            parameters={"speaker": "SPEAKER_00"},
        )
        hyp_b = ASRHypothesis(
            hypothesis_id="h2",
            start_s=3.0,
            end_s=4.0,
            text="Sovrapposizione",
            parameters={"speaker": "SPEAKER_01"},
        )

        utts_order_1 = build_utterances([hyp_a, hyp_b])
        utts_order_2 = build_utterances([hyp_b, hyp_a])

        self.assertEqual(len(utts_order_1), 2)
        self.assertEqual(len(utts_order_2), 2)
        # Entrambi gli ordini producono lo stesso insieme di testi e speaker
        set_1 = {(u.start_s, u.speaker, u.text) for u in utts_order_1}
        set_2 = {(u.start_s, u.speaker, u.text) for u in utts_order_2}
        self.assertEqual(set_1, set_2)


if __name__ == "__main__":
    unittest.main()
