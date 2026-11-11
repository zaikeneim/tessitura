"""Test per la trascrizione a sorgenti indipendenti e bleed check (Passo 3).

Verifica:
1. Trascrizione indipendente di traccia mic e sistema con offset globale.
2. Formazione di Utterance con overlap_group_id e stream_id corretti.
3. Bleed Check diagnostico con attenuazione RMS (> 12 dB) e similarità testuale.
4. Non-distruttività: gli interventi sospetti restano sia in Utterance
   sia in Segment, senza soppressioni basate su correlazione e livello.
5. Integrazione nella pipeline con strategy="source_single_speaker".
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src.models.session import Session
from src.processing.audio_profile import AudioProfile
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.source_transcription import (
    _compute_segment_rms_dbfs,
    transcribe_sources,
)
from src.processing.transcribe import TSegment


def _create_sine_wav(path: Path, duration_s: float, freq: float = 440.0, amp: float = 0.5, sr: int = 16000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr)


class SourceTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp_dir.name)
        self.mic_wav = self.work / "mic.wav"
        self.sys_wav = self.work / "sys.wav"
        # Mic: segnale forte (amp = 0.5, ~ -6 dBFS)
        _create_sine_wav(self.mic_wav, 5.0, freq=440.0, amp=0.5)
        # Sys: segnale molto attenuato (amp = 0.05, ~ -26 dBFS, diff > 12 dB)
        _create_sine_wav(self.sys_wav, 5.0, freq=440.0, amp=0.04)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_compute_segment_rms_dbfs(self):
        rms_mic = _compute_segment_rms_dbfs(self.mic_wav, 0.0, 2.0)
        rms_sys = _compute_segment_rms_dbfs(self.sys_wav, 0.0, 2.0)
        self.assertTrue(np.isfinite(rms_mic))
        self.assertTrue(np.isfinite(rms_sys))
        # Differenza > 12 dB
        self.assertGreater(rms_mic - rms_sys, 15.0)

    @patch("src.processing.source_transcription.Transcriber")
    def test_transcribe_sources_headphones_confirmed(self, mock_transcriber_cls):
        """Con cuffie confermate, non viene eseguito alcun Bleed Check."""
        mock_instance = MagicMock()
        mock_transcriber_cls.return_value = mock_instance

        # Mic restituisce un segmento a [1.0, 3.0]
        # Sys restituisce un segmento a [1.2, 3.2] con testo simile
        mock_instance.transcribe.side_effect = [
            ([TSegment(start=1.0, end=3.0, text="Buongiorno a tutti", words=[(1.0, 3.0, "Buongiorno a tutti", 0.95)])], "it"),
            ([TSegment(start=1.2, end=3.2, text="Buongiorno a tutti quanti", words=[(1.2, 3.2, "Buongiorno a tutti quanti", 0.95)])], "it"),
        ]

        class DummyOpts:
            diarize = False
            model_size = "tiny"
            device = "cpu"
            compute_type = "float32"
            language = "it"
            initial_prompt = None
            quality_retries = 0
            headphones_confirmed = True

        utterances, segments, lang, meta = transcribe_sources(
            self.mic_wav,
            self.sys_wav,
            offsets_s=[0.5, 0.0],  # mic ha 0.5s di offset
            options=DummyOpts(),
        )

        # Con offset 0.5s, il mic va da 1.5s a 3.5s
        self.assertEqual(len(utterances), 2)
        self.assertEqual(len(segments), 2)
        self.assertEqual(meta["suppressed_bleed_count"], 0)

        # Verifica offset applicato
        mic_utt = next(u for u in utterances if u.speaker == "SPEAKER_MIC")
        self.assertAlmostEqual(mic_utt.start_s, 1.5, places=2)
        self.assertAlmostEqual(mic_utt.end_s, 3.5, places=2)
        self.assertAlmostEqual(mic_utt.words[0].start, 1.5, places=2)

        # Entrambi appartengono allo stesso gruppo di overlap
        self.assertIsNotNone(mic_utt.overlap_group_id)
        sys_utt = next(u for u in utterances if u.speaker == "SPEAKER_SYS")
        self.assertEqual(mic_utt.overlap_group_id, sys_utt.overlap_group_id)

    @patch("src.processing.source_transcription.Transcriber")
    def test_transcribe_sources_bleed_check_active(self, mock_transcriber_cls):
        """Il possibile rientro viene segnalato senza eliminare parlato simultaneo."""
        mock_instance = MagicMock()
        mock_transcriber_cls.return_value = mock_instance

        # Mic forte, Sys debole, testo identico sovrapposto
        mock_instance.transcribe.side_effect = [
            ([TSegment(start=1.0, end=3.0, text="Prova rientro microfono", words=[(1.0, 3.0, "Prova", 0.99)])], "it"),
            ([TSegment(start=1.0, end=3.0, text="Prova rientro microfono", words=[(1.0, 3.0, "Prova", 0.99)])], "it"),
        ]

        class DummyOpts:
            diarize = False
            model_size = "tiny"
            device = "cpu"
            compute_type = "float32"
            language = "it"
            initial_prompt = None
            quality_retries = 0
            headphones_confirmed = False

        utterances, segments, lang, meta = transcribe_sources(
            self.mic_wav,
            self.sys_wav,
            offsets_s=[0.0, 0.0],
            options=DummyOpts(),
        )

        # Sys è debole (>12 dB più basso), quindi segnalato per revisione.
        self.assertEqual(len(utterances), 2)
        self.assertEqual(meta["suppressed_bleed_count"], 0)
        self.assertEqual(meta["suspected_echo_count"], 1)

        sys_utt = next(u for u in utterances if u.speaker == "SPEAKER_SYS")
        self.assertFalse(getattr(sys_utt, "suppressed_bleed", False))
        self.assertEqual(sys_utt.candidate_speakers[-1]["reason"], "possible_acoustic_echo_review")

        # Entrambi i Segment restano disponibili anche con testo uguale.
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "SPEAKER_MIC")

    @patch("src.processing.source_transcription.Transcriber")
    def test_pipeline_integration_source_single_speaker(self, mock_transcriber_cls):
        """Test end-to-end con process_recording e strategy=source_single_speaker."""
        mock_instance = MagicMock()
        mock_transcriber_cls.return_value = mock_instance

        mock_instance.transcribe.side_effect = [
            ([TSegment(start=0.5, end=2.5, text="Ciao dal microfono", words=[(0.5, 2.5, "Ciao", 0.95)])], "it"),
            ([TSegment(start=1.0, end=3.0, text="Ciao dal sistema remoto", words=[(1.0, 3.0, "Ciao", 0.95)])], "it"),
        ]

        opts = PipelineOptions(
            audio_profile=AudioProfile.MIC_SYSTEM,
            strategy="source_single_speaker",
            headphones_confirmed=True,
            denoise=False,
            diarize=False,
            analysis_mode="off",
        )

        session = process_recording(
            self.mic_wav,
            self.sys_wav,
            options=opts,
            work_dir=self.work,
            session_name="Sessione Test Duale",
        )

        self.assertIsInstance(session, Session)
        self.assertEqual(session.pipeline_strategy, "source_single_speaker")
        self.assertIsNotNone(session.capture_manifest)
        self.assertEqual(session.capture_manifest["strategy"], "source_single_speaker")
        self.assertEqual(len(session.utterances), 2)
        self.assertEqual(len(session.segments), 2)
        self.assertIsNotNone(session.audio.mixed)
        self.assertTrue(Path(session.audio.mixed).exists() or Path(self.work / f"{self.mic_wav.stem}_mixed.wav").exists())


if __name__ == "__main__":
    unittest.main()
