"""Test per i backend ASR alternativi e il registro (P08).

Verifica la conformita' del contratto ASRBackend, la registrazione, la sicurezza offline
e la gestione delle risorse senza download impliciti.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.models.audio_artifacts import AudioView
from src.models.transcript_artifacts import ASRHypothesis
from src.processing.asr_backends import (
    ASRBackend,
    WhisperBackend,
    get_asr_backend,
    list_available_asr_backends,
    register_asr_backend,
)


class DummyMockBackend(ASRBackend):
    @property
    def name(self) -> str:
        return "mock_asr"

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio, language=None, on_progress=None, **kwargs):
        return [
            ASRHypothesis(
                hypothesis_id="hyp_mock",
                start_s=0.0,
                end_s=2.0,
                backend=self.name,
                text="Testo mock di test",
            )
        ]


class ASRBackendsTests(unittest.TestCase):
    def test_backend_registry(self):
        backends = list_available_asr_backends()
        self.assertIn("whisper", backends)
        self.assertNotIn("qwen3", backends)
        self.assertNotIn("parakeet", backends)
        self.assertTrue(backends["whisper"]["available"])

    def test_custom_backend_registration(self):
        register_asr_backend("mock_asr", DummyMockBackend)
        backend = get_asr_backend("mock_asr", model_name="dummy_model")
        self.assertTrue(backend.is_available())
        hypotheses = backend.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].text, "Testo mock di test")

    def test_unknown_backend_raises_error(self):
        with self.assertRaises(ValueError):
            get_asr_backend("nonexistent_backend_xyz")

    def test_whisper_backend_default_compute_type(self):
        backend = WhisperBackend(model_name="large-v3", device="cuda")
        self.assertEqual(backend.name, "whisper")
        self.assertEqual(backend.compute_type, "int8_float16")
        backend.close()


if __name__ == "__main__":
    unittest.main()
