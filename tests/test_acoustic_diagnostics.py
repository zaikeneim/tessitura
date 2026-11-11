"""Test per la diagnostica acustica e il trattamento conservativo (P03)."""

import unittest
import numpy as np

from src.models.audio_artifacts import AudioAsset
from src.processing.acoustic_diagnostics import analyze_acoustics, create_conservative_view


class AcousticDiagnosticsTests(unittest.TestCase):
    def test_silent_audio_produces_safe_profile(self):
        silent = np.zeros((16000, 2), dtype=np.float32)
        profile = analyze_acoustics(silent, sr=16000)
        self.assertTrue(profile.is_silent)
        self.assertEqual(profile.clipping_ratio, 0.0)
        self.assertFalse(profile.has_reliable_noise_profile)
        for val in profile.rms_dbfs_per_channel:
            self.assertTrue(np.isfinite(val))

    def test_clipping_detection(self):
        data = np.zeros(16000, dtype=np.float32)
        data[:800] = 1.0  # 5% clipped samples
        profile = analyze_acoustics(data, sr=16000)
        self.assertAlmostEqual(profile.clipping_ratio, 0.05, places=2)
        self.assertTrue(any("clipping" in w.lower() for w in profile.warnings))

    def test_stereo_channel_correlation(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        # Stereo identico -> correlazione vicina a 1.0
        stereo_identical = np.column_stack([sig, sig])
        prof_id = analyze_acoustics(stereo_identical, sr=sr)
        self.assertIsNotNone(prof_id.channel_correlation)
        self.assertAlmostEqual(prof_id.channel_correlation, 1.0, places=2)

        # Stereo con canali opposti / ortogonali
        sig2 = (0.5 * np.cos(2 * np.pi * 300 * t)).astype(np.float32)
        stereo_ortho = np.column_stack([sig, sig2])
        prof_ortho = analyze_acoustics(stereo_ortho, sr=sr)
        self.assertIsNotNone(prof_ortho.channel_correlation)
        self.assertAlmostEqual(prof_ortho.channel_correlation, 0.0, delta=0.05)

    def test_conservative_view_preserves_levels_and_ceiling(self):
        sr = 16000
        # Segnale con rumore stazionario (-30 dBFS) e bursts di parlato (-15 dBFS)
        np.random.seed(42)
        noise = (np.random.randn(sr * 2) * 0.03).astype(np.float32)
        t = np.linspace(0, 1.0, sr, endpoint=False)
        speech_burst = (0.4 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        signal = noise.copy()
        signal[sr // 2 : sr // 2 + sr] += speech_burst

        profile = analyze_acoustics(signal, sr=sr)
        self.assertFalse(profile.is_silent)

        asset = AudioAsset(
            asset_id="asset_test",
            sample_rate=sr,
            num_channels=1,
            num_samples=len(signal),
            duration_s=2.0,
        )

        view, treated = create_conservative_view(asset, signal, profile)
        self.assertEqual(view.asset_id, "asset_test")
        self.assertLessEqual(float(np.max(np.abs(treated))), 0.951)
        self.assertTrue(len(view.transformations) > 0)


if __name__ == "__main__":
    unittest.main()
