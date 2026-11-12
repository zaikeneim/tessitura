"""Regression proof for real stage execution and worker/UI event delivery."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.processing.alignment import AlignmentReport
from src.processing.diarize import DiarizationResult, Turn
from src.processing.pipeline import PipelineOptions, process_recording
from src.processing.progress import StageReporter
from src.processing.transcribe import TSegment


class ProcessingProgressTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.mic, self.system = self.root / "mic.wav", self.root / "system.wav"
        for source in (self.mic, self.system):
            sf.write(source, np.sin(np.arange(48000) / 20).astype("float32") * .1, 16000)
        self.events, self.progress, self.operations = [], [], []
        self.enterContext(patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")))
        self.asr = self.enterContext(patch("src.processing.transcribe.Transcriber"))
        self.source_asr = self.enterContext(patch("src.processing.source_transcription.Transcriber"))
        self.asr.return_value.transcribe.side_effect = self.decode
        self.source_asr.return_value.transcribe.side_effect = self.decode
        for target in ("src.processing.pipeline.align_segments", "src.processing.source_transcription.align_segments"):
            self.enterContext(patch(target, side_effect=self.align))
        self.diar = self.enterContext(patch("src.processing.diarize.Diarizer"))
        self.source_diar = self.enterContext(patch("src.processing.source_transcription.Diarizer"))
        self.diar.return_value.diarize_detailed.side_effect = self.diarize
        self.source_diar.return_value.diarize_detailed.side_effect = self.diarize

    def decode(self, path, *, on_progress=None, on_status=None, **kwargs):
        self.operations.append("asr")
        if on_progress:
            on_progress(.5, 3.)
            on_progress(2., 3.)
        if on_status:
            on_status("Recupero omissione 2.0–3.0 s")
        # Completion must wait for the decoder, including its recovery pass.
        self.assertFalse(any(e.status == "completed" and e.stage_id.endswith("asr")
                             for e in self.events[-1:]))
        return [TSegment(0., 1., "ciao", [(0., 1., "ciao", .9)])], "it"

    def align(self, segments, *args, **kwargs):
        self.operations.append("alignment")
        return segments, AlignmentReport()

    def diarize(self, *args, on_progress=None, **kwargs):
        self.operations.append("diarization")
        if on_progress:
            on_progress("segmentation", .5)
            on_progress("embedding", None)
        turns = [Turn(0., 3., "A")]
        return DiarizationResult(turns, turns, [])

    def run_pipeline(self, **changes):
        source = changes.pop("source", False)
        options = dict(audio_profile="mic_system" if source else "stereo_ambient", device="cpu",
                       alignment_backend="whisper", quality_retries=2, profile_first=False)
        if source:
            options.update(strategy="source_aware", recording_mode="headset")
        options.update(changes)
        return process_recording(self.mic, self.system if source else None,
                                 options=PipelineOptions(**options), work_dir=self.root,
                                 on_stage=self.events.append,
                                 progress=lambda msg, fraction: self.progress.append((msg, fraction)))

    def assert_complete(self, session):
        rows = session.provenance["processing_stages"]
        self.assertTrue(all(row["status"] in ("completed", "skipped") for row in rows))
        fractions = [fraction for _, fraction in self.progress]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.)
        self.assertTrue(all(e.overall_fraction < 1 for e in self.events[:-1]))
        self.assertEqual(self.events[-1].stage_id, "finalize")
        self.assertEqual(type(session).from_dict(session.to_dict()).provenance["processing_stages"], rows)

    def test_standard_executes_asr_alignment_diarization_review_reconstruction(self):
        session = self.run_pipeline()
        self.assert_complete(session)
        self.assertEqual(self.operations, ["asr", "alignment", "diarization"])
        started = [e.stage_id for e in self.events if e.status == "running"]
        self.assertLess(started.index("asr"), started.index("alignment"))
        self.assertLess(started.index("alignment"), started.index("diarization"))
        self.assertTrue(any(e.stage_id == "asr" and e.fraction == 2 / 3 for e in self.events))
        self.assertTrue(any("Recupero omissione" in e.detail for e in self.events))
        for stage in ("review", "reconstruct"):
            self.assertIn(stage, started)

    def test_profile_first_executes_diarization_before_asr(self):
        session = self.run_pipeline(profile_first=True)
        self.assert_complete(session)
        self.assertEqual(self.operations, ["diarization", "asr", "alignment"])

    def test_disabled_options_show_real_skip_reasons(self):
        session = self.run_pipeline(diarize=False, analysis_mode="off")
        self.assert_complete(session)
        rows = {row["stage_id"]: row for row in session.provenance["processing_stages"]}
        for stage in ("denoise", "diarization", "profiles", "review", "separation"):
            self.assertEqual(rows[stage]["status"], "skipped")
            self.assertTrue(rows[stage]["detail"])

    def test_source_progress_is_live_and_verified_mic_skips_only_its_diarization(self):
        session = self.run_pipeline(source=True, mic_single_speaker_verified=True)
        self.assert_complete(session)
        rows = {row["stage_id"]: row for row in session.provenance["processing_stages"]}
        self.assertEqual(rows["mic_diarization"]["status"], "skipped")
        self.assertEqual(rows["system_diarization"]["status"], "completed")
        for source in ("mic", "system"):
            self.assertTrue(any(e.stage_id == f"{source}_asr" and e.fraction == 2 / 3 for e in self.events))
        self.assertEqual(len(session.utterances), 2)
        self.assertEqual(len({u.overlap_group_id for u in session.utterances}), 1)

    def test_asr_failure_keeps_alignment_pending_and_never_emits_success(self):
        self.asr.return_value.transcribe.side_effect = RuntimeError("decoder failed")
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            self.run_pipeline()
        self.assertEqual(self.events[-1].status, "failed")
        self.assertEqual(self.events[-1].stage_id, "asr")
        self.assertTrue(all(fraction < 1 for _, fraction in self.progress))
        self.assertFalse(any(e.stage_id == "alignment" and e.status != "pending" for e in self.events))
        self.asr.return_value.close.assert_called_once()

    def test_source_model_release_failure_marks_the_active_asr_failed(self):
        self.source_asr.return_value.close.side_effect = RuntimeError("release failed")
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            self.run_pipeline(source=True)
        self.assertEqual((self.events[-1].stage_id, self.events[-1].status), ("system_asr", "failed"))
        self.assertFalse(any(e.stage_id == "system_asr" and e.status == "completed" for e in self.events))
        self.assertTrue(all(fraction < 1 for _, fraction in self.progress))

    def test_alignment_failure_is_reported_before_diarization(self):
        with patch("src.processing.pipeline.align_segments", side_effect=RuntimeError("align failed")):
            with self.assertRaisesRegex(RuntimeError, "align failed"):
                self.run_pipeline()
        self.assertEqual((self.events[-1].stage_id, self.events[-1].status), ("alignment", "failed"))
        self.assertEqual(self.operations, ["asr"])

    def test_invalid_profile_is_a_failed_validation_stage(self):
        with self.assertRaises(ValueError):
            self.run_pipeline(audio_profile=None)
        self.assertEqual((self.events[-1].stage_id, self.events[-1].status), ("validate", "failed"))
        self.asr.assert_not_called()


class DiagnosticProgressTests(unittest.TestCase):
    def test_experiment_errors_remain_warnings_and_successful_variants_resume_from_cache(self):
        from src.processing.diagnostic_experiments import run_diagnostic_experiments
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sample.wav"
            sf.write(source, np.zeros(16000, dtype="float32"), 16000)
            model = root / "model"
            model.mkdir()
            (model / "model.bin").write_bytes(b"fixed model")
            report = root / "report.json"
            report.write_text(json.dumps({"source_sha256": "hash", "regions": [
                {"region_id": "sample", "audio_file": "sample.wav", "sample_origin_s": 0}]}))
            events = []
            def decode(path, **kwargs):
                kwargs["on_progress"](.5, 1.)
                if "raw_no_vad" in str(path):
                    raise RuntimeError("variant failed")
                return [], "en"
            with patch("src.processing.transcribe._ensure_model_path", return_value=str(model)), \
                    patch("src.processing.transcribe.Transcriber") as transcriber, \
                    patch("src.processing.denoise.denoise_array", side_effect=lambda data, *args, **kw: data), \
                    patch("src.processing.denoise.denoise_file", side_effect=lambda source, target, **kw: shutil.copyfile(source, target)):
                transcriber.return_value.transcribe.side_effect = decode
                result = run_diagnostic_experiments(report, count=1, device="cpu", on_stage=events.append)
                self.assertTrue(any(e.stage_id == "compare" and e.warning for e in events))
                self.assertEqual(sum("error" in r for r in json.loads(result.read_text())["results"]), 1)
                transcriber.return_value.transcribe.reset_mock()
                run_diagnostic_experiments(report, count=1, device="cpu")
                self.assertEqual(transcriber.return_value.transcribe.call_count, 1)

    def test_scan_reports_observed_frames_then_cache_skips_and_missing_source_fails(self):
        from src.processing.long_diagnostics import DiagnosticOptions, diagnose_recording
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.wav"
            sf.write(source, np.zeros(16000 * 25, dtype="float32"), 16000)
            events = []
            options = DiagnosticOptions(segment_s=20, overlap_regions=0, degraded_regions=0, control_regions=1)
            report = diagnose_recording(source, output_dir=root / "out", options=options, on_stage=events.append)
            self.assertEqual(json.loads(report.read_text())["scan"]["decoded_frames"], 400000)
            self.assertTrue(any(e.stage_id == "scan" and e.fraction and 0 < e.fraction < 1 for e in events))
            self.assertEqual(events[-1].overall_fraction, 1.)
            events.clear()
            diagnose_recording(source, output_dir=root / "out", options=options, on_stage=events.append)
            self.assertTrue(any(e.stage_id == "scan" and e.status == "skipped" for e in events))
            events.clear()
            with self.assertRaises(FileNotFoundError):
                diagnose_recording(root / "missing.wav", on_stage=events.append)
            self.assertEqual(events[-1].status, "failed")


class StageDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_delivers_real_pipeline_stages_to_visible_rows(self):
        from src.gui.stage_view import StageView
        from src.gui.workers import ProcessWorker
        view = StageView()
        view.begin_run()
        worker = ProcessWorker(None, None, PipelineOptions())
        worker.stage.connect(view.on_stage)
        errors = []
        worker.failed.connect(errors.append)
        worker.run()
        self.assertTrue(errors)
        self.assertEqual(view.topLevelItem(0).text(1), "Fallita")
        self.assertIn("Profilo audio obbligatorio", view.topLevelItem(0).text(2))
        self.assertFalse(view.isHidden())
        view.close()

    def test_batch_keeps_failures_file_context_and_stops_after_current_file(self):
        from src.gui.stage_view import StageView
        from src.gui.workers import BatchProcessWorker
        view = StageView()
        worker = BatchProcessWorker(["bad.wav", "good.wav", "pending.wav"], PipelineOptions())
        worker.stage.connect(view.on_batch_stage)
        failed, completed = [], []
        worker.file_failed.connect(lambda source, error: failed.append((source, error)))
        worker.file_done.connect(lambda source, session: completed.append(source))

        def process(source, _, *, on_stage, **kwargs):
            stages = StageReporter([("asr", "Trascrizione")], on_stage=on_stage)
            with stages.guard():
                stages.start("asr")
                if source.name == "bad.wav":
                    raise ValueError("broken audio")
                stages.complete("asr")
                worker.stop()
            return object()

        with patch("src.gui.workers.process_recording", side_effect=process):
            worker.run()
        self.assertEqual([source.name for source in completed], ["good.wav"])
        self.assertEqual(failed[0][1], "broken audio")
        self.assertIn("file 2/3", view.run_label)
        self.assertEqual(view.topLevelItemCount(), 1)
        self.assertEqual(view.topLevelItem(0).text(1), "Completata")
        view.close()

    def test_guard_and_late_callbacks_cannot_report_false_success(self):
        events = []
        reporter = StageReporter([("a", "A"), ("b", "B")], on_stage=events.append)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with reporter.guard():
                reporter.start("a")
                reporter.update("a", .5)
                raise RuntimeError("failed")
        reporter.update("a", 1.)
        self.assertEqual(events[-1].status, "failed")
        self.assertLess(events[-1].overall_fraction, 1.)
        with self.assertRaises(RuntimeError):
            reporter.assert_complete()


class TranscriberStageStatusTests(unittest.TestCase):
    def transcriber(self):
        from src.processing.transcribe import Transcriber
        instance = Transcriber.__new__(Transcriber)
        instance.batch_size = 1
        instance._batched_model = None
        word = SimpleNamespace(start=0., end=1., word="hello", probability=.9)
        segment = SimpleNamespace(start=0., end=1., text="hello", words=[word], avg_logprob=-2.)
        instance._model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([segment]), SimpleNamespace(duration=3., language="en")))
        return instance

    def test_actual_transcriber_reports_retry_and_independent_coverage_execution(self):
        from src.processing.transcribe import Transcriber
        statuses, progress = [], []
        instance = self.transcriber()
        improved = TSegment(0., 1., "hello", [(0., 1., "hello", .9)], avg_logprob=-.1)
        with patch.object(Transcriber, "_retry_segment", return_value=improved) as retry, \
                patch("src.processing.audio_utils.load_mono_16k", return_value=np.zeros(48000)), \
                patch("src.processing.coverage.detect_speech_activity", return_value=[]) as coverage:
            instance.transcribe("test.wav", quality_retries=2, on_status=statuses.append,
                                on_progress=lambda done, total: progress.append((done, total)))
        retry.assert_called_once()
        coverage.assert_called_once()
        self.assertTrue(any("Retry segmento" in status for status in statuses))
        self.assertTrue(any("copertura vocale" in status for status in statuses))
        self.assertEqual(progress, [(1., 3.)])

    def test_coverage_failure_warns_without_discarding_recognized_speech(self):
        instance = self.transcriber()
        reporter = StageReporter([("asr", "ASR")])
        reporter.start("asr")
        with patch("src.processing.transcribe._suspect", return_value=False), \
                patch("src.processing.audio_utils.load_mono_16k", side_effect=OSError("coverage unavailable")):
            segments, _ = instance.transcribe("test.wav", quality_retries=1,
                                              on_status=reporter.asr_status("asr"))
        event = reporter.transcribed("asr")
        self.assertEqual(segments[0].text, "hello")
        self.assertIn("coverage unavailable", event.warning)


if __name__ == "__main__":
    unittest.main()
