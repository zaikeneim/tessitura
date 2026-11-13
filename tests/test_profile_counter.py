"""Voice profile counter in the processing stages (count, persistence, UI).

Esecuzione: python -m tests.test_profile_counter
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from src.gui.stage_view import StageView
from src.models.session import Session
from src.models.storage import save_session, load_session
from src.processing.progress import StageReporter
from src.processing.voice_profile import VoiceProfile, format_profile_count, profile_stage_detail


def profiles(n, *, embedding=True, seconds=12.0, known=0):
    return {f"S{i}": VoiceProfile(speaker_id=f"S{i}", embedding=[1.0, 0.0] if embedding else None,
                                  total_duration=seconds, is_known_profile=i < known) for i in range(n)}


class ProfileCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_wording_for_zero_one_and_many(self):
        self.assertEqual(format_profile_count(0), "0 profili creati")
        self.assertEqual(format_profile_count(1), "1 profilo creato")
        self.assertEqual(format_profile_count(4), "4 profili creati")

    def test_counts_come_from_the_returned_collection(self):
        detail, extra = profile_stage_detail({**profiles(3, known=1), "X": VoiceProfile("X", total_duration=2.0)})
        self.assertEqual(detail, "4 profili creati")
        self.assertEqual(extra, {"profiles": 4, "with_embedding": 3, "known_matches": 1, "tse_eligible": 0})
        self.assertEqual(profile_stage_detail({})[0], "0 profili creati")

    def test_count_persists_in_session_and_renders_with_duration(self):
        for n in (0, 1, 4):
            with self.subTest(n=n):
                reporter = StageReporter([("profiles", "Profili vocali")])
                reporter.start("profiles")
                detail, extra = profile_stage_detail(profiles(n))
                reporter.complete("profiles", detail=detail, extra=extra)
                session = Session(provenance={"processing_stages": reporter.snapshot()})
                rows = Session.from_dict(session.to_dict()).provenance["processing_stages"]
                self.assertEqual(rows[0]["detail"], format_profile_count(n))
                self.assertEqual(rows[0]["extra"]["profiles"], n)
                view = StageView()
                self.assertTrue(view.show_saved(rows))
                item = view.topLevelItem(0)
                self.assertEqual(item.text(1), "Completata")
                self.assertEqual(item.text(2), format_profile_count(n))
                self.assertTrue(item.text(3).endswith(" s"))
                self.assertIn("Profili creati", item.toolTip(2))
                view.deleteLater()

    def test_embedding_validation_and_target_eligibility_require_real_evidence(self):
        data = {"A": VoiceProfile("A", embedding=[float("nan")], total_duration=100.),
                "B": VoiceProfile("B", embedding=[0., 0.]),
                "C": VoiceProfile("C", embedding=[1., 0.], total_duration=100.),
                "D": VoiceProfile("D", embedding=[1., 0.], metadata={"tse_enrollment_verified": True})}
        _, extra = profile_stage_detail(data)
        self.assertEqual(extra, {"profiles": 4, "with_embedding": 2, "known_matches": 0, "tse_eligible": 1})

    def test_profile_count_survives_disk_save_and_reopens_visibly(self):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QToolButton
        from src.gui.main_window import MainWindow
        for count in (0, 1, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                detail, extra = profile_stage_detail(profiles(count))
                reporter = StageReporter([("profiles", "Profili vocali")])
                reporter.complete("profiles", detail=detail, extra=extra)
                session = Session(provenance={"processing_stages": reporter.snapshot()})
                path = Path(directory) / "session.json"
                save_session(session, path)
                view, toggle = StageView(), QToolButton()
                toggle.setCheckable(True)
                toggle.toggled.connect(view.setVisible)
                MainWindow._show_saved_stages(SimpleNamespace(stage_view=view, stage_toggle=toggle), load_session(path))
                self.assertTrue(toggle.isChecked())
                self.assertFalse(view.isHidden())
                self.assertEqual(view.topLevelItem(0).text(2), format_profile_count(count))
                self.assertEqual(view.topLevelItem(0).text(3), "0,0 s")
                view.close()

    def test_completed_run_keeps_profile_row_inside_the_visible_table(self):
        view = StageView()
        view.resize(850, 150)
        view.show()
        stages = [("profiles", "Profili vocali")] + [(f"phase_{i}", "Elaborazione") for i in range(20)] + [("finalize", "Sessione")]
        reporter = StageReporter(stages, on_stage=view.on_stage)
        detail, extra = profile_stage_detail(profiles(4))
        reporter.complete("profiles", detail=detail, extra=extra)
        for stage, _ in stages[1:]:
            reporter.start(stage)
            reporter.complete(stage)
        self.app.processEvents()
        self.assertTrue(view.visualItemRect(view._rows["profiles"]).intersects(view.viewport().rect()))
        view.close()


class ProfileCounterPipelineTests(unittest.TestCase):
    def fixture(self):
        import tests.test_processing_progress as fixtures
        case = fixtures.ProcessingProgressTests("test_standard_executes_asr_alignment_diarization_review_reconstruction")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_standard_and_profile_first_count_actual_build_results_not_speaker_option(self):
        from src.processing.voice_profile import VoiceProfile
        for first in (False, True):
            for count in (0, 1, 4):
                with self.subTest(profile_first=first, profiles=count):
                    case = self.fixture()
                    returned = {chr(65 + i): VoiceProfile(chr(65 + i)) for i in range(count)}
                    with patch("src.processing.voice_profile.build_voice_profiles", return_value=returned) as build:
                        session = case.run_pipeline(profile_first=first, num_speakers=7)
                    build.assert_called_once()
                    row = next(r for r in session.provenance["processing_stages"] if r["stage_id"] == "profiles")
                    self.assertEqual(row["detail"], format_profile_count(count))
                    self.assertEqual(row["extra"]["profiles"], count)
                    case.doCleanups()

    def test_source_aware_counts_profiles_created_after_recomposition(self):
        from src.processing.diarize import DiarizationResult, Turn
        from src.processing.transcribe import TSegment
        for count in (0, 1, 4):
            with self.subTest(profiles=count):
                case = self.fixture()
                def decode(path, **kwargs):
                    amount = (1 if "mic" in str(path) else 0) if count == 1 else count // 2
                    return [TSegment(i * .8, i * .8 + .5, "ciao", [(i * .8, i * .8 + .5, "ciao", .9)])
                            for i in range(amount)], "it"
                def diarize(*args, **kwargs):
                    turns = [Turn(i * .8, i * .8 + .5, chr(65 + i)) for i in range(max(1, count // 2))]
                    return DiarizationResult(turns, turns, [])
                case.source_asr.return_value.transcribe.side_effect = decode
                case.source_diar.return_value.diarize_detailed.side_effect = diarize
                session = case.run_pipeline(source=True, num_speakers=7)
                row = next(r for r in session.provenance["processing_stages"] if r["stage_id"] == "profiles")
                self.assertEqual(row["detail"], format_profile_count(count))
                self.assertEqual(row["extra"]["profiles"], len(session.diarization["voice_profiles"]))
                case.doCleanups()


if __name__ == "__main__":
    unittest.main()
