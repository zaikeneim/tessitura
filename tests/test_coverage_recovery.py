"""Test per l'analisi della copertura del parlato e il recupero delle omissioni (P04).

Scenari obbligatori dal piano esecutivo:
- VAD indica parlato, ASR iniziale vuoto -> suspect 'empty_asr' e retry pianificato
- Parola a cavallo del bordo -> espansione consentita, parola presente una sola volta
- Silenzio/rumore puro con allucinazione -> candidato rifiutato
- Distribuzione temporale dei retry su file lunghi
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.models.audio_artifacts import SpeechRegion
from src.processing.coverage import (
    detect_speech_activity,
    find_uncovered_regions,
    is_hallucination_or_noise,
    plan_distributed_recovery,
)
from src.processing.transcribe import TSegment, _stitch_retry_segments, _suspect


class CoverageRecoveryTests(unittest.TestCase):
    def test_empty_asr_generates_recovery_suspect(self):
        # Genera un audio sintetico con parlato acustico tra 1.0s e 2.5s
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        audio = np.zeros(sr * 3, dtype=np.float32)
        # Speech burst tra 1.0 e 2.5 s (-18 dBFS)
        burst = (0.35 * np.sin(2 * np.pi * 300 * t[: int(sr * 1.5)])).astype(np.float32)
        audio[sr : sr + int(sr * 1.5)] = burst

        regions = detect_speech_activity(audio, sr=sr)
        self.assertTrue(len(regions) >= 1)
        speech_reg = regions[0]
        self.assertAlmostEqual(speech_reg.start_s, 1.0, delta=0.25)
        self.assertAlmostEqual(speech_reg.end_s, 2.5, delta=0.25)

        # ASR iniziale completamente vuoto (0 segmenti)
        asr_segments = []
        suspects = find_uncovered_regions(regions, asr_segments)
        self.assertEqual(len(suspects), 1)
        self.assertIn("empty_asr", suspects[0].suspect_reasons)

        # Pianificazione retry mirato
        planned = plan_distributed_recovery(suspects, total_duration_s=3.0)
        self.assertEqual(len(planned), 1)
        start_pad, end_pad, reg = planned[0]
        self.assertLessEqual(start_pad, speech_reg.start_s)
        self.assertGreaterEqual(end_pad, speech_reg.end_s)

    def test_boundary_word_expansion_without_duplicates(self):
        word = lambda start, end, text: SimpleNamespace(start=start, end=end, word=text, probability=0.9)
        # Segmento originario termina a 12.0s
        original = TSegment(10.0, 12.0, "vecchio", [], avg_logprob=-1.2)

        # Retry candidate include una parola 'dopo' che attraversa il bordo fino a 12.35s
        candidates = [
            SimpleNamespace(
                start=10.0,
                end=12.5,
                text="vecchio testo dopo",
                words=[
                    word(10.0, 10.6, "vecchio"),
                    word(10.7, 11.5, "testo"),
                    word(11.8, 12.35, "dopo"),  # a cavallo del bordo (midpoint 12.075)
                ],
                avg_logprob=-0.2,
                compression_ratio=1.0,
                no_speech_prob=0.05,
            )
        ]

        # Con expand_margin=0.0 (legacy), 'dopo' con midpoint 12.075 > 12.0 viene scartata
        strict_result = _stitch_retry_segments(candidates, original, expand_margin=0.0)
        self.assertNotIn("dopo", strict_result.text)

        # Con expand_margin=0.4 (P04 recupero bordi), 'dopo' viene inclusa correttamente
        expanded_result = _stitch_retry_segments(candidates, original, expand_margin=0.4)
        self.assertIn("dopo", expanded_result.text)
        self.assertEqual(expanded_result.text, "vecchio testo dopo")
        self.assertGreater(expanded_result.end, 12.2)

    def test_silence_hallucination_is_rejected(self):
        sr = 16000
        # Silenzio puro (-80 dBFS)
        silent_slice = np.zeros(sr * 2, dtype=np.float32)

        # Allucinazione tipica con ripetizioni o frasi note
        self.assertTrue(
            is_hallucination_or_noise(
                "Sottotitoli a cura di...",
                silent_slice,
                compression_ratio=1.0,
                no_speech_prob=0.9,
            )
        )
        self.assertTrue(
            is_hallucination_or_noise(
                "grazie per la visione",
                silent_slice,
                compression_ratio=1.0,
                no_speech_prob=0.2,
            )
        )
        # Testo apparentemente plausibile ma su silenzio acustico assoluto
        self.assertTrue(
            is_hallucination_or_noise(
                "Parola inventata",
                silent_slice,
                compression_ratio=1.0,
                no_speech_prob=0.1,
            )
        )

        # Parlato reale con energia acustica
        audio_speech = (0.2 * np.random.randn(sr * 2)).astype(np.float32)
        self.assertFalse(
            is_hallucination_or_noise(
                "Buongiorno a tutti",
                audio_speech,
                compression_ratio=1.1,
                no_speech_prob=0.1,
            )
        )

    def test_short_interjection_is_suspected(self):
        # Turno molto breve con alto no_speech_prob
        short_seg = TSegment(
            start=5.0,
            end=5.8,
            text="Sì",
            words=[],
            avg_logprob=-0.8,
            compression_ratio=1.0,
            no_speech_prob=0.75,
        )
        self.assertTrue(_suspect(short_seg))

        # Turno molto breve pulito (basso no_speech, buon logprob)
        good_short = TSegment(
            start=5.0,
            end=5.8,
            text="Sì",
            words=[],
            avg_logprob=-0.3,
            compression_ratio=1.0,
            no_speech_prob=0.05,
        )
        self.assertFalse(_suspect(good_short))

    def test_distributed_recovery_budget_across_long_recording(self):
        # File di 300 secondi (5 minuti) con sospetti sparsi lungo tutta la registrazione
        suspects = [
            SpeechRegion(start_s=5.0, end_s=7.0, suspect_reasons=["empty_asr"]),
            SpeechRegion(start_s=15.0, end_s=17.0, suspect_reasons=["empty_asr"]),
            SpeechRegion(start_s=25.0, end_s=27.0, suspect_reasons=["empty_asr"]),
            SpeechRegion(start_s=150.0, end_s=152.0, suspect_reasons=["empty_asr"]),
            SpeechRegion(start_s=280.0, end_s=283.0, suspect_reasons=["empty_asr"]),
        ]
        # Con max_retries=3, il campionamento per slot garantisce che anche i minuti successivi ricevano budget
        planned = plan_distributed_recovery(suspects, total_duration_s=300.0, max_retries=3, slot_duration_s=60.0)
        self.assertEqual(len(planned), 3)
        starts = [item[0] for item in planned]
        # Non devono essere tutti nei primi 30 secondi
        self.assertTrue(any(s > 100.0 for s in starts))


if __name__ == "__main__":
    unittest.main()
