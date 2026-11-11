"""Backend di separazione pluggabile: contratto, regola offline e ripieghi (piano §6.4).

Scenari obbligatori:
- Interfaccia SeparationBackend e registro dei backend
- Validazione offline: nessun peso caricato senza voce nel manifest e SHA-256 corrispondente
- Ripiego deterministico su traccia grezza quando il backend manca, fallisce o rende poco
- Guardrail di perdita parlato sulle tracce separate
- Riconciliazione dell'identita' delle tracce fra finestre successive

Esecuzione: python -m tests.test_separation_backend_onnx
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.audio_artifacts import AudioView, SpeechRegion
from src.processing.separation import separate_region
from src.processing.separation_backends import (
    MODEL_SAMPLE_RATE,
    NullSeparationBackend,
    OnnxSeparationBackend,
    SeparationBackend,
    _as_tracks,
    file_sha256,
    get_separation_backend,
    list_available_separation_backends,
    verify_model,
)


class _StubBackend(SeparationBackend):
    """Separatore controllabile, per esercitare i guardrail senza pesi reali."""

    def __init__(self, tracks=None, error=None, available=True):
        super().__init__("stub")
        self._tracks = tracks or []
        self._error = error
        self._available = available

    @property
    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return self._available

    def separate_region(self, audio_data, sample_rate, num_speakers=2):
        if self._error is not None:
            raise self._error
        return list(self._tracks)


class _SwappingBackend(OnnxSeparationBackend):
    """Finge un separatore che inverte l'ordine delle uscite a blocchi alterni."""

    def __init__(self, **kwargs):
        super().__init__(model_path="mai-caricato.onnx", **kwargs)
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def _infer_chunk(self, chunk):
        self.calls += 1
        pair = np.stack([chunk.astype(np.float32), (0.5 * chunk[::-1]).astype(np.float32)])
        return pair[::-1] if self.calls % 2 == 0 else pair


def _correlation(a, b) -> float:
    width = min(len(a), len(b))
    return float(np.corrcoef(a[:width], b[:width])[0, 1])


class SeparationBackendRegistryTests(unittest.TestCase):
    def test_registro_espone_onnx_e_null(self):
        listed = list_available_separation_backends()

        self.assertIn("onnx", listed)
        self.assertIn("null", listed)
        self.assertTrue(all(entry["offline_safe"] for entry in listed.values()))

    def test_backend_sconosciuto_solleva_errore(self):
        with self.assertRaises(ValueError):
            get_separation_backend("inesistente")

    def test_null_backend_non_separa_mai(self):
        backend = NullSeparationBackend()

        self.assertTrue(backend.is_available())
        self.assertEqual(backend.separate_region(np.zeros(1000, np.float32), 16000), [])


class OfflineWeightValidationTests(unittest.TestCase):
    """La regola offline: nessun peso non dichiarato entra in inferenza."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.model = self.root / "separatore.onnx"
        self.model.write_bytes(b"pesi-finti-per-verifica")
        self.addCleanup(self._tmp.cleanup)

    def _write_manifest(self, sha: str) -> None:
        payload = {"models": {self.model.name: {"sha256": sha}}}
        (self.root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_peso_assente_e_dichiarato_mancante(self):
        verified, reason = verify_model(self.root / "assente.onnx")

        self.assertFalse(verified)
        self.assertEqual(reason, "model_file_missing")

    def test_peso_senza_voce_nel_manifest_e_rifiutato(self):
        verified, reason = verify_model(self.model)

        self.assertFalse(verified)
        self.assertEqual(reason, "manifest_entry_missing")

    def test_impronta_diversa_e_rifiutata(self):
        self._write_manifest("0" * 64)

        verified, reason = verify_model(self.model)

        self.assertFalse(verified)
        self.assertEqual(reason, "sha256_mismatch")

    def test_impronta_corrispondente_e_accettata(self):
        self._write_manifest(file_sha256(self.model))

        verified, reason = verify_model(self.model)

        self.assertTrue(verified)
        self.assertEqual(reason, "verified")

    def test_backend_onnx_indisponibile_senza_manifest(self):
        backend = OnnxSeparationBackend(model_path=self.model)

        self.assertFalse(backend.is_available())
        self.assertEqual(backend.unavailable_reason, "manifest_entry_missing")

    def test_backend_onnx_disponibile_con_impronta_valida(self):
        self._write_manifest(file_sha256(self.model))
        backend = OnnxSeparationBackend(model_path=self.model)

        self.assertTrue(backend.is_available())
        self.assertEqual(backend.unavailable_reason, "")

    def test_providers_ripiegano_su_cpu(self):
        backend = OnnxSeparationBackend(model_path=self.model)

        self.assertIn("CPUExecutionProvider", backend.resolve_providers())


class SeparationFallbackTests(unittest.TestCase):
    """Ogni anomalia del backend deve lasciare intatta la traccia grezza."""

    def setUp(self):
        self.sr = MODEL_SAMPLE_RATE
        self.view = AudioView(
            view_id="view_onnx", asset_id="asset_onnx", sample_rate=self.sr, duration_s=2.0
        )
        self.region = SpeechRegion(start_s=0.0, end_s=2.0)
        rng = np.random.default_rng(7)
        self.data = (0.2 * rng.standard_normal(self.sr * 2)).astype(np.float32)

    def test_backend_non_disponibile_ripiega_su_raw(self):
        result = separate_region(
            self.data, self.view, self.region, backend=_StubBackend(available=False)
        )

        self.assertEqual(result.status, "fallback_raw")
        self.assertEqual(result.decision, "accepted")
        self.assertEqual(len(result.tracks), 1)

    def test_errore_del_backend_ripiega_su_raw(self):
        result = separate_region(
            self.data, self.view, self.region, backend=_StubBackend(error=RuntimeError("boom"))
        )

        self.assertEqual(result.status, "fallback_raw")
        self.assertEqual(result.metrics["reason"], "backend_error")
        self.assertEqual(result.metrics["error"], "RuntimeError")

    def test_tracce_insufficienti_ripiegano_su_raw(self):
        result = separate_region(
            self.data, self.view, self.region, backend=_StubBackend(tracks=[self.data])
        )

        self.assertEqual(result.status, "fallback_raw")
        self.assertEqual(result.metrics["reason"], "backend_returned_insufficient_tracks")

    def test_perdita_di_parlato_e_rifiutata(self):
        rng = np.random.default_rng(11)
        faint = (0.002 * rng.standard_normal(self.data.size)).astype(np.float32)
        other = (0.002 * rng.standard_normal(self.data.size)).astype(np.float32)

        result = separate_region(
            self.data, self.view, self.region, backend=_StubBackend(tracks=[faint, other])
        )

        self.assertEqual(result.decision, "rejected_speech_loss")
        self.assertEqual(result.status, "fallback_raw")
        self.assertEqual(len(result.tracks), 1)

    def test_separazione_valida_e_accettata(self):
        rng = np.random.default_rng(13)
        first = (0.2 * rng.standard_normal(self.data.size)).astype(np.float32)
        second = (0.2 * rng.standard_normal(self.data.size)).astype(np.float32)

        result = separate_region(
            self.data, self.view, self.region, backend=_StubBackend(tracks=[first, second])
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.decision, "accepted")
        self.assertEqual(len(result.tracks), 2)


class ChunkedSeparationTests(unittest.TestCase):
    """Regioni lunghe: ricucitura a finestre senza scambiare i parlanti."""

    def test_identita_conservata_fra_finestre(self):
        rng = np.random.default_rng(3)
        signal = (0.2 * rng.standard_normal(MODEL_SAMPLE_RATE * 40)).astype(np.float32)
        backend = _SwappingBackend(chunk_seconds=15.0, overlap_seconds=1.0)

        tracks = backend.separate_region(signal, MODEL_SAMPLE_RATE)

        self.assertGreater(backend.calls, 1, "la regione deve essere spezzata in finestre")
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0]), signal.size)
        self.assertGreater(_correlation(tracks[0], signal), 0.9)

    def test_regione_breve_resta_una_sola_finestra(self):
        rng = np.random.default_rng(5)
        signal = (0.2 * rng.standard_normal(MODEL_SAMPLE_RATE * 2)).astype(np.float32)
        backend = _SwappingBackend(chunk_seconds=15.0, overlap_seconds=1.0)

        tracks = backend.separate_region(signal, MODEL_SAMPLE_RATE)

        self.assertEqual(backend.calls, 1)
        self.assertEqual(len(tracks), 2)


class TrackLayoutTests(unittest.TestCase):
    """Gli export ONNX differiscono nella forma delle uscite: vanno normalizzati."""

    def test_tensore_batch_parlanti_campioni(self):
        raw = np.arange(2 * 100, dtype=np.float32).reshape(1, 2, 100)

        tracks = _as_tracks([raw], 100)

        self.assertEqual(tracks.shape, (2, 100))

    def test_tensore_campioni_per_parlante_viene_trasposto(self):
        raw = np.arange(100 * 2, dtype=np.float32).reshape(100, 2)

        tracks = _as_tracks([raw], 100)

        self.assertEqual(tracks.shape, (2, 100))

    def test_uscita_separata_per_parlante(self):
        first = np.ones(100, dtype=np.float32)
        second = np.zeros(100, dtype=np.float32)

        tracks = _as_tracks([first, second], 100)

        self.assertEqual(tracks.shape, (2, 100))

    def test_uscita_vuota_non_produce_tracce(self):
        self.assertEqual(_as_tracks([], 100).shape[0], 0)


if __name__ == "__main__":
    unittest.main()
