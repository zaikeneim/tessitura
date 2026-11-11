"""Percorso `speaker_isolated`: cablaggio, ripiego dichiarato e limiti d'ingresso.

Scenari obbligatori:
- La separazione riuscita produce una traccia per parlante e Utterance ricomposte
- Il separatore assente o rifiutato non viene spacciato per separazione riuscita
- Il ripiego resta tracciabile nei metadati della sessione
- Il backend non supera mai la lunghezza d'ingresso dichiarata nel manifest
- Il CLI espone la strategia senza cambiare il default

Esecuzione: python -m tests.test_speaker_isolated_routing
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.models.audio_artifacts import AudioView, SpeechRegion
from src.processing.separation import separate_region
from src.processing.separation_backends import (
    MODEL_SAMPLE_RATE,
    OnnxSeparationBackend,
    SeparationBackend,
)
from src.processing.speaker_tracks import (
    SpeakerTracksResult,
    extract_speaker_tracks,
    transcribe_speaker_tracks,
)


class _StubBackend(SeparationBackend):
    """Separatore pilotabile: evita di dipendere dai pesi presenti sulla macchina."""

    def __init__(self, tracks=None, available=True):
        super().__init__("stub")
        self._tracks = tracks
        self._available = available
        self.requested_speakers = None

    @property
    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return self._available

    def separate_region(self, audio_data, sample_rate, num_speakers=2):
        self.requested_speakers = num_speakers
        if self._tracks is None:
            return []
        return list(self._tracks)


class _FakeSegment:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


def _voice(seed: int, seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(int(seconds * MODEL_SAMPLE_RATE))).astype(np.float32)


def _write_wav(path: Path, signal: np.ndarray) -> Path:
    import soundfile as sf

    sf.write(str(path), signal, MODEL_SAMPLE_RATE)
    return path


class ExtractSpeakerTracksTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.audio = _write_wav(self.root / "mix.wav", _voice(1) + _voice(2))

    def test_separazione_riuscita_produce_una_traccia_per_voce(self):
        backend = _StubBackend(tracks=[_voice(3), _voice(4)])

        result = extract_speaker_tracks(self.audio, num_speakers=2, backend=backend)

        self.assertTrue(result.separated)
        self.assertEqual(result.status, "separated")
        self.assertEqual(len(result.tracks), 2)

    def test_numero_di_voci_richiesto_arriva_al_backend(self):
        backend = _StubBackend(tracks=[_voice(3), _voice(4), _voice(5)])

        extract_speaker_tracks(self.audio, num_speakers=3, backend=backend)

        self.assertEqual(backend.requested_speakers, 3)

    def test_backend_indisponibile_non_e_separazione(self):
        result = extract_speaker_tracks(self.audio, backend=_StubBackend(available=False))

        self.assertFalse(result.separated)
        self.assertEqual(result.status, "fallback_raw")

    def test_traccia_unica_non_e_separazione(self):
        result = extract_speaker_tracks(self.audio, backend=_StubBackend(tracks=[_voice(3)]))

        self.assertFalse(result.separated)
        self.assertEqual(result.status, "fallback_raw")

    def test_il_ripiego_resta_dichiarato_nei_metadati(self):
        result = extract_speaker_tracks(self.audio, backend=_StubBackend(available=False))

        metadata = result.as_metadata()

        self.assertEqual(metadata["separation_status"], "fallback_raw")
        self.assertEqual(metadata["track_count"], 0)
        self.assertIn("reason", metadata)

    def test_tracce_duplicate_rifiutate_dai_guardrail(self):
        same = _voice(9)
        backend = _StubBackend(tracks=[same, same.copy()])

        result = extract_speaker_tracks(self.audio, backend=backend)

        self.assertFalse(result.separated)
        self.assertEqual(result.decision, "rejected_duplicate")


class TranscribeSpeakerTracksTests(unittest.TestCase):
    def test_senza_separazione_non_si_trascrive(self):
        with self.assertRaises(ValueError):
            transcribe_speaker_tracks(SpeakerTracksResult())

    def test_ogni_traccia_diventa_un_parlante_stabile(self):
        result = SpeakerTracksResult(
            tracks=[_voice(1), _voice(2)],
            sample_rate=MODEL_SAMPLE_RATE,
            status="separated",
        )
        fake = [
            _FakeSegment(0.0, 1.0, "prima voce", [(0.0, 1.0, "prima voce", 0.9)]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.processing.speaker_tracks.Transcriber") as transcriber, patch(
                "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
            ):
                transcriber.return_value.transcribe.return_value = (fake, "it")
                utterances, segments, language, metadata = transcribe_speaker_tracks(
                    result, work_dir=Path(tmp)
                )

        self.assertEqual(language, "it")
        self.assertEqual(metadata["strategy"], "speaker_isolated")
        self.assertEqual(metadata["separation_status"], "separated")
        self.assertEqual(
            sorted({segment.speaker for segment in segments}), ["SPEAKER_00", "SPEAKER_01"]
        )
        self.assertEqual(len(utterances), 2)

    def test_voci_simultanee_raggruppate(self):
        result = SpeakerTracksResult(
            tracks=[_voice(1), _voice(2)],
            sample_rate=MODEL_SAMPLE_RATE,
            status="separated",
        )
        fake = [_FakeSegment(0.0, 2.0, "parlato", [(0.0, 2.0, "parlato", 0.9)])]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.processing.speaker_tracks.Transcriber") as transcriber, patch(
                "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
            ):
                transcriber.return_value.transcribe.return_value = (fake, "it")
                utterances, _segments, _language, _metadata = transcribe_speaker_tracks(
                    result, work_dir=Path(tmp)
                )

        groups = {utterance.overlap_group_id for utterance in utterances}
        self.assertEqual(len(groups), 1, "le due voci sovrapposte devono condividere il gruppo")
        self.assertIsNotNone(next(iter(groups)))
        self.assertEqual({utterance.stream_id for utterance in utterances}, {0, 1})


class ManifestInputCeilingTests(unittest.TestCase):
    """Un export che congela la lunghezza tracciata non va superato."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.model = self.root / "separatore.onnx"
        self.model.write_bytes(b"pesi-finti")

    def _manifest(self, **extra):
        payload = {"models": {self.model.name: {"sha256": "0" * 64, **extra}}}
        (self.root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_il_tetto_del_manifest_riduce_la_finestra(self):
        self._manifest(max_input_seconds=3.0)
        backend = OnnxSeparationBackend(model_path=self.model, chunk_seconds=15.0)

        self.assertEqual(backend.effective_chunk_seconds(), 3.0)

    def test_senza_tetto_la_finestra_resta_quella_richiesta(self):
        self._manifest()
        backend = OnnxSeparationBackend(model_path=self.model, chunk_seconds=15.0)

        self.assertEqual(backend.effective_chunk_seconds(), 15.0)

    def test_il_tetto_non_allarga_una_finestra_piu_stretta(self):
        self._manifest(max_input_seconds=10.0)
        backend = OnnxSeparationBackend(model_path=self.model, chunk_seconds=4.0)

        self.assertEqual(backend.effective_chunk_seconds(), 4.0)


class SeparateRegionSpeakerCountTests(unittest.TestCase):
    def test_num_speakers_ha_precedenza_sui_profili(self):
        backend = _StubBackend(tracks=[_voice(1), _voice(2)])
        view = AudioView(
            view_id="v", asset_id="a", sample_rate=MODEL_SAMPLE_RATE, duration_s=2.0
        )

        separate_region(
            _voice(5),
            view,
            SpeechRegion(start_s=0.0, end_s=2.0),
            backend=backend,
            num_speakers=4,
        )

        self.assertEqual(backend.requested_speakers, 4)


class PipelineRoutingTests(unittest.TestCase):
    """Il dispatcher ha cinque chiamanti: il ramo nuovo non deve toccarne nessuno."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.audio = _write_wav(self.root / "ambientale.wav", _voice(1))

    def _options(self, strategy: str):
        from src.processing.audio_profile import AudioProfile
        from src.processing.pipeline import PipelineOptions

        return PipelineOptions(
            strategy=strategy,
            denoise=False,
            diarize=False,
            audio_profile=AudioProfile.STEREO_AMBIENT.value,
            analysis_mode="off",
            profile_first=False,
        )

    def _run(self, strategy: str, tracks_result: SpeakerTracksResult):
        from src.processing.pipeline import process_recording

        transcribed = [_FakeSegment(0.0, 1.0, "ciao", [(0.0, 1.0, "ciao", 0.9)])]
        with patch(
            "src.processing.speaker_tracks.extract_speaker_tracks", return_value=tracks_result
        ), patch("src.processing.transcribe.Transcriber") as transcriber, patch(
            "src.processing.speaker_tracks.Transcriber"
        ) as track_transcriber, patch(
            "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
        ):
            transcriber.return_value.transcribe.return_value = (transcribed, "it")
            track_transcriber.return_value.transcribe.return_value = (transcribed, "it")
            return process_recording(
                mic_path=self.audio,
                options=self._options(strategy),
                work_dir=self.root / f"work_{strategy}",
            )

    def test_separazione_riuscita_produce_una_sessione_isolata(self):
        session = self._run(
            "speaker_isolated",
            SpeakerTracksResult(
                tracks=[_voice(1), _voice(2)], status="separated", decision="accepted"
            ),
        )

        self.assertEqual(session.pipeline_strategy, "speaker_isolated")
        self.assertEqual(session.capture_manifest["separation_status"], "separated")
        self.assertEqual(
            sorted({segment.speaker for segment in session.segments}),
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_senza_separatore_si_ripiega_sul_percorso_standard(self):
        session = self._run(
            "speaker_isolated",
            SpeakerTracksResult(
                status="fallback_raw", metrics={"reason": "model_file_missing"}
            ),
        )

        # Il ripiego resta leggibile: non si dichiara una separazione mai avvenuta.
        self.assertEqual(session.capture_manifest["separation_status"], "fallback_raw")
        self.assertEqual(session.capture_manifest["reason"], "model_file_missing")
        self.assertTrue(session.segments)

    def test_il_percorso_standard_resta_intatto(self):
        from src.processing.pipeline import process_recording

        transcribed = [_FakeSegment(0.0, 1.0, "ciao", [(0.0, 1.0, "ciao", 0.9)])]
        with patch("src.processing.speaker_tracks.extract_speaker_tracks") as separator, patch(
            "src.processing.transcribe.Transcriber"
        ) as transcriber, patch(
            "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
        ):
            transcriber.return_value.transcribe.return_value = (transcribed, "it")
            session = process_recording(
                mic_path=self.audio,
                options=self._options("standard"),
                work_dir=self.root / "work_standard",
            )

        separator.assert_not_called()
        self.assertEqual(session.pipeline_strategy, "standard")
        self.assertIsNone(session.capture_manifest)


class CliStrategyTests(unittest.TestCase):
    def test_il_cli_espone_speaker_isolated_senza_cambiare_il_default(self):
        source = Path("scripts/run_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("speaker_isolated", source)
        self.assertIn('default="standard"', source)


if __name__ == "__main__":
    unittest.main()
