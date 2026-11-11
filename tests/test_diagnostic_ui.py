"""Desktop diagnostics retain playback candidates and keep active workers alive."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()
from PySide6.QtWidgets import QApplication, QDialog
from src.gui.diagnostic_dialog import DiagnosticDialog, DiagnosticWorker


class DiagnosticUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_scan_candidates_playback_comparison_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root / "sample.wav"
            sample.write_bytes(b"lossless audio fixture")
            report = {"source_sha256": "fixture", "metadata": {"duration_s": 600},
                      "scan": {"decoded_duration_s": 600, "channels": 2, "decode_errors": []},
                      "regions": [{"region_id": "one", "start_s": 20, "end_s": 50,
                                   "selection_reason": "overlap_candidate", "audio_file": sample.name,
                                   "existing_transcript": [{"text": "unverified candidate", "start_s": 21, "end_s": 23}]}]}
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report))
            for name in ("timeline.json", "reference-review.json"):
                (root / name).write_text("{}")
            with patch.object(DiagnosticDialog, "_start"):
                dialog = DiagnosticDialog(root / "source.mp3")
            self.addCleanup(dialog.deleteLater)
            with patch.object(dialog.player, "load") as load:
                dialog._done(report_path)
                self.assertEqual(dialog.regions.count(), 1)
                self.assertEqual(load.call_args.args[0], str(sample))
                self.assertIn("unverified candidate", dialog.transcript.toPlainText())
                comparison_path = root / "comparison.json"
                comparison_path.write_text(json.dumps({"results": [{"configuration": {
                    "region_id": "one", "channel": 1, "variant": "raw_no_vad"},
                    "audio": str(sample), "segments": [{"text": "second candidate"}]}]}))
                dialog._done(comparison_path)
                self.assertEqual(dialog.variants.count(), 2)
                dialog.variants.setCurrentIndex(1)
                self.assertIn("second candidate", dialog.transcript.toPlainText())
            export = root / "export"
            export.mkdir()
            with patch("src.gui.diagnostic_dialog.QFileDialog.getExistingDirectory", return_value=str(export)):
                dialog._export()
            output = next(export.iterdir())
            self.assertEqual((output / "sample.wav").read_bytes(), sample.read_bytes())
            self.assertTrue((output / "reference-review.json").is_file())
            dialog.player.clear()

    def test_incomplete_scan_is_labeled_partial_and_worker_blocks_close(self):
        with patch.object(DiagnosticDialog, "_start"):
            dialog = DiagnosticDialog("source.mp3")
        self.addCleanup(dialog.deleteLater)
        dialog._worker = object()
        with patch.object(QDialog, "reject") as reject:
            dialog.reject()
        reject.assert_not_called()
        self.assertIn("Attendi", dialog.status.text())
        dialog._worker = None

    def test_worker_runs_model_free_full_scan_and_reports_failures(self):
        worker = DiagnosticWorker("original.mp3")
        results, errors = [], []
        worker.done.connect(results.append)
        worker.failed.connect(errors.append)
        with patch("src.processing.long_diagnostics.diagnose_recording", return_value=Path("report.json")) as scan:
            worker.run()
        scan.assert_called_once_with("original.mp3")
        self.assertEqual(results, [Path("report.json")])
        with patch("src.processing.long_diagnostics.diagnose_recording", side_effect=OSError("decode failed")):
            worker.run()
        self.assertEqual(errors, ["decode failed"])


if __name__ == "__main__":
    unittest.main()
