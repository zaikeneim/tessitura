"""Test per il contratto e le protezioni della separazione mirata (P06).

Scenari obbligatori dal piano esecutivo:
- Separatore restituisce latenza, tracce e qualita'
- Tracce duplicate (correlazione eccessiva) -> rifiutate, fallback su raw
- Collasso energetico di una sorgente -> rifiutato
- Interruzione tramite cancel_event rispettata
"""

import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.models.audio_artifacts import AudioView, SpeechRegion
from src.processing.separation import cross_correlation, separate_region


class SeparationContractTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.view = AudioView(
            view_id="view_sep",
            asset_id="asset_sep",
            sample_rate=self.sr,
            duration_s=2.0,
        )
        self.region = SpeechRegion(start_s=0.0, end_s=2.0)

    @patch("src.processing.separation._resolve_backend")
    def test_fallback_to_raw_when_backend_inactive(self, mock_resolve):
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = False
        mock_backend.unavailable_reason = "separation_model_not_configured"
        mock_backend.name = "mock_inactive"
        mock_resolve.return_value = mock_backend

        data = (0.2 * np.random.randn(self.sr * 2)).astype(np.float32)
        res = separate_region(data, self.view, self.region, backend="auto")
        self.assertEqual(res.status, "fallback_raw")
        self.assertEqual(res.decision, "accepted")
        self.assertEqual(len(res.tracks), 1)

    def test_duplicate_tracks_are_rejected(self):
        # Due canali con audio quasi identico (correlazione ~ 1.0)
        t = np.linspace(0, 2.0, self.sr * 2, endpoint=False)
        sig = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        duplicate_data = np.column_stack([sig, sig + 0.001 * np.random.randn(len(sig)).astype(np.float32)])

        res = separate_region(
            duplicate_data,
            self.view,
            self.region,
            backend="simulated_split_test",
            duplicate_threshold=0.88,
        )
        self.assertEqual(res.decision, "rejected_duplicate")
        self.assertEqual(res.status, "fallback_raw")
        self.assertEqual(len(res.tracks), 1)  # Fallback su raw audio

    def test_energy_collapse_is_rejected(self):
        # Canale 0 con segnale, canale 1 collassato a zero
        sig = (0.3 * np.ones(self.sr * 2, dtype=np.float32))
        silent = np.zeros(self.sr * 2, dtype=np.float32)
        collapsed_data = np.column_stack([sig, silent])

        res = separate_region(
            collapsed_data,
            self.view,
            self.region,
            backend="simulated_split_test",
            silence_threshold_db=-60.0,
        )
        self.assertEqual(res.decision, "rejected_energy_collapse")
        self.assertEqual(res.status, "fallback_raw")

    def test_cancel_event_halts_immediately(self):
        cancel = threading.Event()
        cancel.set()
        data = np.zeros((self.sr * 2, 2), dtype=np.float32)
        res = separate_region(data, self.view, self.region, cancel_event=cancel)
        self.assertEqual(res.status, "cancelled")
        self.assertEqual(res.decision, "rejected_cancelled")

    def test_cross_correlation_calculation(self):
        sig1 = np.sin(np.linspace(0, 10, 1000))
        sig2 = np.sin(np.linspace(0, 10, 1000))
        self.assertAlmostEqual(cross_correlation(sig1, sig2), 1.0, places=3)

        sig_ortho = np.cos(np.linspace(0, 10, 1000))
        self.assertAlmostEqual(cross_correlation(sig1, sig_ortho), 0.0, delta=0.08)


if __name__ == "__main__":
    unittest.main()
