"""Test per il modulo di allineamento forzato e validazione timestamp (src.processing.alignment).

Verifica che:
1. I timestamp parola non vengano mai inventati/fabbricati quando un segmento ne e' privo.
2. I timestamp fuori ordine o fuori dai confini del segmento vengano riordinati/clampati monotonicamente.
3. I metadati di provenienza dell'allineamento (AlignmentReport) siano serializzabili e salvati nella Session.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.processing.alignment import (
    AlignmentReport,
    align_segments,
    validate_word_timestamps,
)
from src.processing.audio_profile import AudioProfile
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.transcribe import TSegment


class AlignmentValidationTests(unittest.TestCase):
    def test_never_fabricates_missing_word_timestamps(self):
        seg = TSegment(start=1.0, end=3.5, text="segmento senza parole allineate", words=[])
        aligned, report = validate_word_timestamps([seg], language="it")

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].words, [])
        self.assertEqual(report.total_segments, 1)
        self.assertEqual(report.aligned_segments, 0)
        self.assertEqual(report.unaligned_segments, 1)
        self.assertEqual(report.total_words, 0)

    def test_clamps_and_repairs_non_monotonic_word_timestamps(self):
        seg = TSegment(
            start=2.0,
            end=4.0,
            text="ciao come stai",
            words=[
                (1.5, 2.5, "ciao", 0.9),   # inizia prima del segmento (1.5 < 2.0)
                (2.3, 2.2, "come", 0.8),   # invertito e sovrapposto al precedente
                (3.5, 4.8, "stai", 0.95),  # finisce dopo il segmento (4.8 > 4.0)
            ],
        )
        aligned, report = validate_word_timestamps([seg], language="it")

        words = aligned[0].words
        self.assertEqual(len(words), 3)
        self.assertGreaterEqual(words[0][0], 2.0)
        self.assertLessEqual(words[-1][1], 4.0)
        for prev, curr in zip(words, words[1:]):
            self.assertLessEqual(prev[0], prev[1])
            self.assertLessEqual(prev[1], curr[0] + 1e-6)
        self.assertGreater(report.repaired_words, 0)
        self.assertEqual(report.aligned_words, 3)

    def test_align_segments_auto_falls_back_cleanly_when_mms_not_cached(self):
        seg = TSegment(
            start=0.0,
            end=1.2,
            text="buongiorno a tutti",
            words=[
                (0.0, 0.5, "buongiorno", 0.92),
                (0.5, 0.6, "a", 0.88),
                (0.6, 1.2, "tutti", 0.95),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "test.wav"
            sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
            with patch("src.processing.alignment._mms_fa_cached_locally", return_value=False):
                aligned, report = align_segments([seg], wav, language="it", backend="auto", device="cpu")

        self.assertEqual(len(aligned), 1)
        self.assertEqual(len(aligned[0].words), 3)
        self.assertEqual(report.backend, "faster-whisper-cross-attention")
        payload = report.to_dict()
        self.assertEqual(payload["aligned_words"], 3)
        self.assertEqual(payload["unaligned_segments"], 0)

    def test_pipeline_records_alignment_provenance(self):
        seg = TSegment(
            start=0.0,
            end=1.0,
            text="prova",
            words=[(0.1, 0.9, "prova", 0.91)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "input.wav"
            sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
            with patch("src.processing.transcribe.Transcriber") as mock_asr, patch(
                "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
            ):
                mock_asr.return_value.transcribe.return_value = ([seg], "it")
                session = process_recording(
                    mic_path=wav,
                    options=PipelineOptions(
                        denoise=False,
                        diarize=False,
                        audio_profile=AudioProfile.STEREO_AMBIENT.value,
                        analysis_mode="off",
                    ),
                    work_dir=root / "work",
                )

        self.assertIn("alignment", session.provenance)
        self.assertEqual(session.provenance["alignment"]["aligned_words"], 1)


if __name__ == "__main__":
    unittest.main()

