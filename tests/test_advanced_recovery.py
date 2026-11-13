"""Recovery proposals preserve the baseline, timing, concurrent voices and review state."""
import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.models.session import Session, Word
from src.models.storage import load_session, save_session
from src.models.timeline import TimelineEditor
from src.models.transcript_artifacts import Utterance
from src.processing.enhancement import analyze_region, bounded_gain, frame_levels_db
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.progress import RECOVERY_STAGES, StageReporter
from src.processing.region_recovery import plan_recovery, recover_region, recover_session_regions
from src.processing.separation import SeparationResult
from src.processing.transcribe import TSegment


class FakeASR:
    calls = []
    closed = 0

    def __init__(self, **kwargs):
        self.options = kwargs

    def transcribe(self, path, **kwargs):
        self.calls.append(Path(path).stem)
        text = "voce nuova" if "source-1" in str(path) or "bounded_gain" in str(path) else "prima voce"
        words = [(2.2, 2.4, text.split()[0], .9), (2.4, 2.6, text.split()[1], .9)]
        return [TSegment(2.2, 2.6, text, words)], "it"

    def close(self):
        type(self).closed += 1


def separated(audio, *_args, **_kwargs):
    return SeparationResult(region_id="test", tracks=[audio * .6, audio * -.4], status="success")


class AdvancedRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "native.wav"
        wave = np.sin(np.arange(16000 * 12) / 14).astype(np.float32) * .006
        wave *= np.tile(np.r_[np.ones(12000), np.zeros(4000)], 12)
        sf.write(self.source, np.c_[wave, wave * .8], 16000, subtype="FLOAT")
        self.baseline = Utterance(2.2, 2.6, speaker="A", text="prima voce",
                                  words=[Word(2.2, 2.4, "prima", .9), Word(2.4, 2.6, "voce", .9)])
        self.baseline.record_user_correction("text", "errato", "prima voce")
        FakeASR.calls, FakeASR.closed = [], 0
        self.enterContext(patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")))

    def recover(self, **changes):
        kwargs = dict(existing_events=[self.baseline], separate=True, potential_speakers=2,
                      transcriber_factory=FakeASR, separator=separated, compare_baseline=True, enhance=True)
        kwargs.update(changes)
        return recover_region(self.source, 2., 4., self.root, **kwargs)

    def session(self):
        return Session(utterances=[copy.deepcopy(self.baseline)], model="test", language="it", schema_version=6,
                       diarization={"turns": [{"start": 2., "end": 4., "speaker": "A"},
                                              {"start": 2., "end": 4., "speaker": "B"}]})

    def test_views_use_lossless_native_context_and_leave_verified_baseline_untouched(self):
        original, baseline = self.source.read_bytes(), self.baseline.to_dict()
        result = self.recover()
        strategies = {view["strategy"] for view in result["views"]}
        self.assertEqual(strategies, {"original", "normalized", "bounded_gain", "acoustic_separation", "enhanced_separation"})
        self.assertEqual(len(FakeASR.calls), 7)
        self.assertEqual(FakeASR.closed, 1)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.baseline.to_dict(), baseline)
        for candidate in result["candidates"]:
            event = candidate["event"]
            self.assertIsNone(event["speaker"])
            self.assertEqual(event["text"], "voce nuova")
            self.assertGreaterEqual(event["start_s"], 2)
            provenance = event["asr_provenance"]
            self.assertEqual(sf.info(provenance["original_context_audio"]).channels, 2)
            self.assertEqual(sf.info(provenance["candidate_audio"]).subtype, "FLOAT")
            self.assertEqual(len(provenance["source_sha256"]), 64)
            self.assertIn("model_revision", provenance)
        json.loads((Path(result["output_dir"]) / "proposals.json").read_text(encoding="utf-8"))

    def test_failed_and_incomplete_separation_remain_explicit(self):
        failed = lambda *_a, **_kw: SeparationResult(region_id="test", tracks=[], status="fallback_raw", decision="backend_error")
        result = self.recover(separator=failed)
        self.assertEqual(result["separation"]["status"], "fallback_raw")
        self.assertFalse(any(v["strategy"].endswith("separation") for v in result["views"]))
        result = self.recover(potential_speakers=3)
        self.assertEqual(result["separation"]["incomplete_source_count"], {"potential": 3, "returned": 2})

    def test_duplicates_removed_by_word_and_punctuation_and_context_excluded(self):
        result = self.recover(separate=False, enhance=False)
        self.assertFalse(result["candidates"])
        self.assertEqual(len(result["predictions"]), 2)

    def test_low_confidence_predictions_are_retained_but_not_proposed(self):
        class UncertainASR(FakeASR):
            def transcribe(self, *args, **kwargs):
                return [TSegment(2.2, 2.6, "inventato", [(2.2, 2.6, "inventato", .2)], avg_logprob=-2.)], "it"
        result = self.recover(transcriber_factory=UncertainASR)
        self.assertFalse(result["candidates"])
        self.assertTrue(all(p["validation"] == "rejected_asr_quality" for p in result["predictions"]))

    def test_gain_is_static_bounded_and_keeps_quiet_samples(self):
        audio = np.r_[np.ones(3200) * .001, np.ones(3200) * .002, np.zeros(3200), np.ones(320) * .95].astype(np.float32)
        enhanced, config = bounded_gain(audio)
        self.assertLessEqual(config["gain_db"], 12)
        self.assertGreater(enhanced[0], audio[0])
        self.assertAlmostEqual(float(enhanced[3200] / enhanced[0]), 2.)
        self.assertLessEqual(float(np.max(np.abs(enhanced))), .89 + 1e-6)
        self.assertTrue(np.all(enhanced[6400:9600] == 0))
        self.assertFalse(config["noise_gate"])
        self.assertFalse(config["time_varying_gain"])
        json.dumps(analyze_region(np.zeros(16000)).to_dict(), allow_nan=False)
        self.assertEqual(len(frame_levels_db(np.zeros(48000), 48000)), 50)

    def test_planner_prioritizes_overlap_but_never_claims_words_missing(self):
        session = self.session()
        plan = plan_recovery(self.source, session.utterances, session.diarization["turns"], max_regions=2, max_audio_s=20., controls=0)
        self.assertLessEqual(plan["queued"], 2)
        self.assertLessEqual(plan["audio_budget_s"], 20.)
        self.assertIn("overlap_2", plan["regions"][0]["labels"])
        self.assertIn("possible_recognition_omission", plan["regions"][0]["labels"])
        self.assertNotIn("recognition_omission", plan["regions"][0]["labels"])

    def test_automatic_pass_is_review_only_checkpointed_and_resumable(self):
        session = self.session()
        baseline = [e.to_dict() for e in session.utterances]
        reporter = StageReporter(RECOVERY_STAGES)
        result = recover_session_regions(self.source, session, self.root, stages=reporter,
            transcriber_factory=FakeASR, separator=separated, max_regions=1)
        reporter.assert_complete()
        self.assertEqual(result["counts"]["processed"], 1)
        self.assertGreater(result["counts"]["candidates"], 0)
        self.assertEqual([e.to_dict() for e in session.utterances], baseline)
        before = copy.deepcopy(session.recovery_candidates)
        with patch.object(FakeASR, "transcribe", side_effect=AssertionError("checkpoint should be reused")):
            replay = recover_session_regions(self.source, session, self.root,
                transcriber_factory=FakeASR, separator=separated, max_regions=1)
        self.assertTrue(replay["regions"][0]["resumed"])
        self.assertEqual(session.recovery_candidates, before)
        target = self.root / "session.json"
        save_session(session, target)
        restored = load_session(target)
        self.assertEqual(restored.recovery_candidates, before)
        self.assertEqual(restored.provenance["advanced_recovery"]["counts"], replay["counts"])

    def test_runtime_budget_and_cancellation_preserve_the_baseline(self):
        session = self.session()
        result = recover_session_regions(self.source, session, self.root,
            transcriber_factory=FakeASR, separator=separated, max_elapsed_s=0)
        self.assertEqual(result["counts"]["processed"], 0)
        self.assertEqual(result["stopped_reason"], "runtime_budget")
        self.assertFalse(session.recovery_candidates)
        event = threading.Event()
        event.set()
        with self.assertRaises(InterruptedError):
            self.recover(cancel_event=event)
        self.assertEqual(len(session.utterances), 1)

    def test_corrupted_checkpoint_audio_is_not_reused(self):
        session = self.session()
        report = recover_session_regions(self.source, session, self.root,
            transcriber_factory=FakeASR, separator=separated, max_regions=1)
        saved = json.loads(Path(report["regions"][0]["checkpoint"]).read_text(encoding="utf-8"))
        sf.write(saved["views"][0]["path"], np.zeros(16000), 16000)
        replay = recover_session_regions(self.source, session, self.root,
            transcriber_factory=FakeASR, separator=separated, max_regions=1)
        self.assertFalse(replay["regions"][0]["resumed"])

    def test_nonfinite_collapsed_and_wrong_length_tracks_fall_back_safely(self):
        from src.models.audio_artifacts import AudioView, SpeechRegion
        from src.processing.separation import separate_region
        from src.processing.separation_backends import NullSeparationBackend
        audio = np.sin(np.arange(32000) / 13).astype(np.float32) * .1
        for invalid in (np.full(32000, np.nan), np.zeros(32000), np.ones(1000)):
            with self.subTest(samples=invalid.size):
                backend = NullSeparationBackend()
                with patch.object(backend, "separate_region", return_value=[audio, invalid]), patch(
                        "src.processing.separation._resolve_backend", return_value=backend):
                    result = separate_region(audio, AudioView(sample_rate=16000, duration_s=2.), SpeechRegion(0., 2.))
                self.assertEqual(result.status, "fallback_raw")
                np.testing.assert_array_equal(result.tracks[0], audio)
                json.dumps(result.to_dict(), allow_nan=False)

    def test_sliced_tracks_cannot_be_reported_as_acoustic_separation(self):
        from src.processing.speaker_tracks import SpeakerTracksResult
        result = SpeakerTracksResult(tracks=[np.ones(32), np.ones(32)], status="sliced",
                                     metrics={"method": "acoustic_separation", "acoustic_separation": True})
        self.assertTrue(result.separated)
        self.assertFalse(result.acoustic)
        self.assertEqual(result.method, "diarization_slicing")
        self.assertFalse(result.as_metadata()["acoustic_separation"])
        self.assertEqual(result.as_metadata()["method"], "diarization_slicing")
        self.assertIn("non è separazione acustica", result.describe())

    def test_cuda_model_memory_failure_falls_back_to_cpu_without_download(self):
        calls = []
        def factory(**kwargs):
            calls.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("CUDA out of memory")
            return FakeASR(**kwargs)
        with patch("src.processing.pipeline.resolve_device", return_value=("cuda", "int8_float16")):
            result = self.recover(transcriber_factory=factory)
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertTrue(all(c["event"]["asr_provenance"]["device"] == "cpu" for c in result["candidates"]))

    def test_possible_intertrack_leakage_is_not_an_accepted_recovery(self):
        class LeakedASR(FakeASR):
            def transcribe(self, *args, **kwargs):
                return [TSegment(2.2, 2.8, "parole spurie duplicate", [(2.2, 2.8, "parole spurie duplicate", .9)])], "it"
        session = self.session()
        recover_session_regions(self.source, session, self.root, max_regions=1,
                                transcriber_factory=LeakedASR, separator=separated)
        sources = [c for c in session.recovery_candidates if c["event"]["asr_provenance"]["strategy"].endswith("separation")]
        self.assertTrue(sources)
        self.assertTrue(all(c["status"] == "rejected_validation" for c in sources))
        self.assertEqual(len(session.utterances), 1)

    def test_accept_reject_undo_redo_preserve_concurrent_verified_events(self):
        session = self.session()
        session.recovery_candidates = self.recover()["candidates"]
        baseline = session.utterances[0].to_dict()
        editor = TimelineEditor(session)
        candidate_id = session.recovery_candidates[0]["candidate_id"]
        editor.reject_candidate(candidate_id)
        self.assertEqual(session.utterances[0].to_dict(), baseline)
        self.assertTrue(session.undo())
        self.assertEqual(session.recovery_candidates[0]["status"], "proposed")
        event = editor.accept_candidate(candidate_id, "B")
        self.assertEqual(event.speaker, "B")
        self.assertEqual(len(session.utterances), 2)
        self.assertTrue(event.is_overlap)
        self.assertEqual(next(e for e in session.utterances if e.event_id == self.baseline.event_id).text, "prima voce")
        self.assertTrue(session.undo())
        self.assertEqual(len(session.utterances), 1)
        self.assertTrue(session.redo())
        self.assertEqual(len(session.utterances), 2)


class RecoveryPipelineTests(unittest.TestCase):
    def fixture(self):
        import tests.test_processing_progress as fixtures
        case = fixtures.ProcessingProgressTests("test_standard_executes_asr_alignment_diarization_review_reconstruction")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_optional_recovery_failure_returns_the_successful_baseline(self):
        case = self.fixture()
        def failed(*args, stages, **kwargs):
            stages.start("recovery_detect")
            raise RuntimeError("recovery failed")
        with self.assertLogs("src.processing.pipeline", level="ERROR"), patch("src.processing.region_recovery.recover_session_regions", side_effect=failed):
            session = case.run_pipeline(recovery_mode="adaptive")
        self.assertTrue(session.utterances)
        self.assertEqual(session.provenance["advanced_recovery_error"], "recovery failed")
        rows = {r["stage_id"]: r for r in session.provenance["processing_stages"]}
        self.assertEqual(rows["recovery_detect"]["status"], "failed")
        self.assertEqual(rows["finalize"]["status"], "completed")

    def test_disabled_mode_never_invokes_second_pass(self):
        case = self.fixture()
        with patch("src.processing.region_recovery.recover_session_regions") as recovery:
            session = case.run_pipeline()
        recovery.assert_not_called()
        self.assertNotIn("advanced_recovery", session.provenance)

    def test_standard_and_source_aware_recovery_execute_after_the_first_pass(self):
        for source in (False, True):
            with self.subTest(source=source):
                case = self.fixture()
                model = case.root / "local-model"
                model.mkdir()
                (model / "config.json").write_text("{}", encoding="utf-8")
                with patch("src.processing.transcribe._ensure_model_path", return_value=str(model)), patch(
                        "src.processing.separation.separate_region", side_effect=separated):
                    session = case.run_pipeline(source=source, recovery_mode="adaptive")
                self.assertNotIn("advanced_recovery_error", session.provenance)
                self.assertIn("advanced_recovery", session.provenance)
                rows = {r["stage_id"]: r for r in session.provenance["processing_stages"]}
                self.assertEqual(rows["recovery_validate"]["status"], "completed")
                self.assertEqual(rows["finalize"]["status"], "completed")
                self.assertEqual(len(session.utterances), 2)
                self.assertEqual(sum(len(e.words) for e in session.utterances), 2 if source else 1)
                case.doCleanups()


class RecoveryReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from src.gui.speaker_timeline_view import SpeakerTimelineView
        from src.gui.transcript_controller import TranscriptController
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "context.wav"
        sf.write(self.path, np.zeros(16000 * 6), 16000)
        event = Utterance(10., 11., speaker=None, text="risposta breve", processing_state="candidate",
            asr_provenance={"original_context_audio": str(self.path), "context_interval": [8., 14.],
                "comparison_views": [{"path": str(self.path), "offset_s": 8., "strategy": "bounded_gain", "source_index": 0}]})
        self.session = Session(schema_version=6, utterances=[Utterance(10., 12., speaker="A", text="già verificato")],
                               recovery_candidates=[{"candidate_id": "review", "status": "proposed", "event": event.to_dict()}])
        self.controller = TranscriptController()
        self.view = SpeakerTimelineView(self.controller)
        self.addCleanup(self.view.close)
        self.controller.load_session(self.session)
        self.view.candidates.setCurrentRow(0)

    def test_preview_uses_local_context_offsets_and_keeps_identity_unassigned(self):
        self.assertEqual(self.view.candidate_views.count(), 2)
        self.assertIsNone(self.view.speaker_combo.currentData())
        with patch.object(self.view.candidate_player, "load") as load, patch.object(
                self.view.candidate_player, "play_region") as play:
            self.view._play_candidate()
        load.assert_called_once_with(self.path)
        play.assert_called_once_with(2., 3.)
        self.assertEqual(len(self.session.utterances), 1)

    def test_reject_button_undo_and_readonly_leave_verified_events_intact(self):
        before = self.session.utterances[0].to_dict()
        self.view._reject_candidate()
        self.assertEqual(self.session.recovery_candidates[0]["status"], "rejected")
        self.assertEqual(self.session.utterances[0].to_dict(), before)
        self.assertTrue(self.controller.undo())
        self.assertEqual(self.view.candidates.count(), 1)
        self.controller.set_read_only(True)
        self.view.candidates.setCurrentRow(0)
        self.view._reject_candidate()
        self.assertEqual(self.session.recovery_candidates[0]["status"], "proposed")

    def test_player_defers_seek_until_context_media_duration_is_known(self):
        from PySide6.QtMultimedia import QMediaPlayer
        player = self.view.candidate_player
        with patch.object(QMediaPlayer, "duration", return_value=0), patch.object(QMediaPlayer, "play"), patch.object(
                QMediaPlayer, "setPosition") as position:
            player.play_region(2., 3.)
            self.assertEqual(player._pending_seek_ms, 2000)
            player._on_duration(6000)
            self.assertEqual(player._pending_seek_ms, 0)
            position.assert_called_with(2000)
        player.clear()
        self.assertEqual(player._pending_seek_ms, 0)


if __name__ == "__main__":
    unittest.main()
