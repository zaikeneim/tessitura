"""Test per contratti dati, timeline multicanale, preservazione canali e serializzazione.

Verifica i requisiti del pacchetto P02 (docs/plans/2026-09-17-gitnexus-plan-recupero-voci-ambientali.md):
- AudioAsset, AudioView, SpeechRegion, SpeakerProfile
- ASRHypothesis, Utterance (con speaker simultanei), TranscriptRevision, StageResult
- load_multichannel preserva stereo e phase coherence senza forzare il downmix mono prematuro
- Mappatura temporale e ritagli vista coerenti
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.models.audio_artifacts import (
    AudioAsset,
    AudioView,
    SpeakerProfile,
    SpeechRegion,
)
from src.models.session import Word
from src.models.transcript_artifacts import (
    ASRHypothesis,
    StageResult,
    TranscriptRevision,
    Utterance,
)
from src.processing.audio_utils import (
    create_audio_asset,
    create_audio_view,
    extract_view_audio,
    load_mono_16k,
    load_multichannel,
)


class AudioTimelineContractsTests(unittest.TestCase):
    def test_audio_asset_and_view_roundtrip(self):
        asset = AudioAsset(
            asset_id="asset_123",
            relative_path="recordings/test.wav",
            sha256="abcdef123456",
            sample_rate=16000,
            num_channels=2,
            num_samples=32000,
            duration_s=2.0,
            provenance={"format": "WAV"},
        )
        d = asset.to_dict()
        restored = AudioAsset.from_dict(d)
        self.assertEqual(asset, restored)

        view = AudioView(
            view_id="view_456",
            asset_id="asset_123",
            channel_indices=[0],
            sample_rate=16000,
            time_offset_s=0.5,
            duration_s=1.0,
            transformations=[{"type": "resample", "sr": 16000}],
            delay_compensation_s=-0.02,
        )
        vd = view.to_dict()
        v_restored = AudioView.from_dict(vd)
        self.assertEqual(view, v_restored)
        self.assertEqual(view.end_s, 1.5)

    def test_speech_region_and_profile_validation(self):
        region = SpeechRegion(
            start_s=1.0,
            end_s=2.5,
            vad_confidence=0.95,
            overlap_probability=0.4,
            suspect_reasons=["boundary_cutoff"],
        )
        self.assertEqual(region.duration_s, 1.5)
        rd = region.to_dict()
        r_restored = SpeechRegion.from_dict(rd)
        self.assertEqual(region.region_id, r_restored.region_id)

        # Negative duration must raise
        with self.assertRaises(ValueError):
            SpeechRegion(start_s=2.0, end_s=1.0)

        # NaN must raise
        with self.assertRaises(ValueError):
            SpeechRegion(start_s=float("nan"), end_s=2.0)

        profile = SpeakerProfile(
            speaker_id="SPEAKER_01",
            normalized_embeddings=[[0.1, 0.2, 0.3]],
            quality_score=0.88,
            is_verified=True,
        )
        pd = profile.to_dict()
        p_restored = SpeakerProfile.from_dict(pd)
        self.assertEqual(profile.speaker_id, p_restored.speaker_id)
        self.assertTrue(p_restored.is_verified)

    def test_simultaneous_utterances_survive_serialization(self):
        # Two speakers speaking simultaneously in the exact same interval [2.0, 4.5]
        w1 = Word(start=2.0, end=2.5, text="Ciao", probability=0.98)
        w2 = Word(start=2.0, end=2.6, text="Buongiorno", probability=0.92)

        utt1 = Utterance(
            start_s=2.0,
            end_s=4.5,
            stream_id=0,
            overlap_group_id="overlap_grp_1",
            speaker="SPEAKER_01",
            text="Ciao come stai",
            words=[w1],
            hypothesis_ids=["hyp_1"],
        )
        utt2 = Utterance(
            start_s=2.0,
            end_s=4.5,
            stream_id=1,
            overlap_group_id="overlap_grp_1",
            speaker="SPEAKER_02",
            text="Buongiorno a tutti",
            words=[w2],
            hypothesis_ids=["hyp_2"],
        )

        list_utts = [utt1.to_dict(), utt2.to_dict()]
        restored_utts = [Utterance.from_dict(x) for x in list_utts]

        self.assertEqual(len(restored_utts), 2)
        self.assertEqual(restored_utts[0].start_s, restored_utts[1].start_s)
        self.assertEqual(restored_utts[0].overlap_group_id, restored_utts[1].overlap_group_id)
        self.assertNotEqual(restored_utts[0].stream_id, restored_utts[1].stream_id)
        self.assertNotEqual(restored_utts[0].speaker, restored_utts[1].speaker)
        self.assertEqual(restored_utts[0].words[0].text, "Ciao")
        self.assertEqual(restored_utts[1].words[0].text, "Buongiorno")

    def test_transcript_revision_and_stage_result(self):
        rev = TranscriptRevision(
            operations=[{"type": "correct_word", "before": "casa", "after": "cosa"}],
            author="human",
            status="proposed",
            source_fingerprint="fp123",
        )
        rev_dict = rev.to_dict()
        rev_restored = TranscriptRevision.from_dict(rev_dict)
        self.assertEqual(rev.operations, rev_restored.operations)

        with self.assertRaises(ValueError):
            TranscriptRevision(status="invalid_status")

        stage = StageResult(
            stage_name="diarization",
            stage_version="2.1.0",
            input_hash="hash999",
            actual_model="pyannote/speaker-diarization-community-1",
            status="success",
            duration_s=4.2,
        )
        st_dict = stage.to_dict()
        st_restored = StageResult.from_dict(st_dict)
        self.assertEqual(stage.actual_model, st_restored.actual_model)


class AudioMultichannelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create a synthetic stereo 44.1 kHz file where Left and Right are distinct
        # Left channel: 440 Hz tone, Right channel: 880 Hz tone
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo_data = np.column_stack([left, right]).astype(np.float32)

        self.stereo_path = self.root / "stereo_44k.wav"
        sf.write(str(self.stereo_path), stereo_data, sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_multichannel_preserves_stereo_channels(self):
        data, sr = load_multichannel(self.stereo_path, target_sr=16000)
        self.assertEqual(sr, 16000)
        self.assertEqual(data.shape[1], 2)
        self.assertEqual(data.shape[0], 16000)

        # Check left and right channels are distinct
        self.assertFalse(np.allclose(data[:, 0], data[:, 1]))
        # Peak of left channel should be around 0.5, right around 0.3
        self.assertAlmostEqual(float(np.max(np.abs(data[:, 0]))), 0.5, delta=0.05)
        self.assertAlmostEqual(float(np.max(np.abs(data[:, 1]))), 0.3, delta=0.05)

    def test_load_mono_16k_preserves_legacy_downmix(self):
        mono = load_mono_16k(self.stereo_path)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(len(mono), 16000)

        # Legacy downmix is the average of the two channels
        data_multi, _ = load_multichannel(self.stereo_path, target_sr=16000)
        expected_mono = data_multi.mean(axis=1)
        self.assertTrue(np.allclose(mono, expected_mono, atol=1e-5))

    def test_create_audio_asset_and_views(self):
        asset = create_audio_asset(self.stereo_path, base_dir=self.root)
        self.assertEqual(asset.relative_path, "stereo_44k.wav")
        self.assertEqual(asset.num_channels, 2)
        self.assertEqual(asset.sample_rate, 44100)
        self.assertTrue(len(asset.sha256) > 0)

        # Create left channel view
        view_left = create_audio_view(asset, channel_indices=[0], time_offset_s=0.2, duration_s=0.5)
        self.assertEqual(view_left.channel_indices, [0])
        self.assertEqual(view_left.time_offset_s, 0.2)
        self.assertEqual(view_left.duration_s, 0.5)

        data_16k, sr = load_multichannel(self.stereo_path, target_sr=16000)
        extracted = extract_view_audio(data_16k, view_left, source_sr=16000)
        self.assertEqual(extracted.ndim, 1)
        self.assertEqual(len(extracted), int(0.5 * 16000))


if __name__ == "__main__":
    unittest.main()
