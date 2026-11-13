"""Hugging Face settings section, diarization model assistant and workflow resumption.

Offscreen Qt. The vault is in memory, the Hub is never contacted: the
verification and download entry points of ``hf_access`` are replaced.

Esecuzione: python -m tests.test_hf_settings_gui
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QMessageBox

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.gui import hf_settings
from src.gui.hf_settings import DiarizationModelDialog, HuggingFaceSettingsWidget
from src.hardware import Hardware
from src.processing import hf_access
from src.processing.hf_access import AccessIssue, ModelAccessError
from src.secure_store import MemoryCredentialStore

SECRET = "hf_" + "Zq9" * 6
CLEAN_ENV = {name: "" for name in hf_access.ENV_VARS}


def wait_until(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def all_text(widget) -> str:
    return "\n".join(label.text() for label in widget.findChildren(QLabel))


class GuiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._created = []
        for target in (patch.dict(os.environ, CLEAN_ENV),
                       patch("src.paths.hf_token_candidates", return_value=[])):
            target.start()
            self.addCleanup(target.stop)

    def track(self, widget):
        """Close only what this test created: other modules may leave windows behind."""
        self._created.append(widget)
        return widget

    def tearDown(self):
        for widget in reversed(self._created):
            widget.close()
            widget.deleteLater()
        self._created.clear()
        self.app.processEvents()


class SettingsWidgetTests(GuiTestCase):
    def _widget(self, store):
        widget = self.track(HuggingFaceSettingsWidget(store))
        self.addCleanup(lambda: wait_until(self.app, lambda: not widget.busy()))
        return widget

    def test_controls_masking_and_initial_state(self):
        widget = self._widget(MemoryCredentialStore())
        self.assertEqual(widget.state, "not_configured")
        self.assertIn("Non configurato", widget.state_label.text())
        self.assertEqual(widget.token_edit.echoMode(), QLineEdit.Password)
        self.assertFalse(widget.save_button.isEnabled())
        self.assertFalse(widget.remove_button.isEnabled())
        self.assertIn(hf_access.TOKENS_URL, widget.tokens_link.text())
        self.assertTrue(widget.tokens_link.openExternalLinks())
        widget.token_edit.setText(SECRET)
        self.assertTrue(widget.save_button.isEnabled())
        widget.show_button.setChecked(True)
        self.assertEqual(widget.token_edit.echoMode(), QLineEdit.Normal)
        self.assertEqual(widget.show_button.text(), "Nascondi")
        widget.show_button.setChecked(False)
        self.assertEqual(widget.token_edit.echoMode(), QLineEdit.Password)

    def test_save_stores_in_vault_clears_field_and_verifies(self):
        store = MemoryCredentialStore()
        widget = self._widget(store)
        seen_states = []
        valid = {"state": "valid", "account": "tester", "model_access": True, "kind": None,
                 "url": "", "message": "Accesso confermato.", "typed": False}
        with patch.object(hf_access, "verify_configuration", side_effect=lambda *a, **k: (seen_states.append(widget.state), valid)[1]):
            widget.token_edit.setText(SECRET)
            widget.save_button.click()
            self.assertEqual(store.read(), SECRET)
            self.assertEqual(widget.token_edit.text(), "")
            self.assertTrue(wait_until(self.app, lambda: widget.state == "valid"))
        self.assertEqual(seen_states, ["checking"])
        self.assertIn("Valido", widget.state_label.text())
        self.assertIn("tester", widget.message_label.text())
        self.assertTrue(widget.remove_button.isEnabled())
        self.assertNotIn(SECRET, all_text(widget))

    def test_invalid_and_network_results_are_shown_differently(self):
        store = MemoryCredentialStore(SECRET)
        widget = self._widget(store)
        self.assertEqual(widget.state, "unverified")
        results = [
            {"state": "invalid", "kind": AccessIssue.TOKEN_INVALID.value,
             "message": AccessIssue.TOKEN_INVALID.message, "url": hf_access.TOKENS_URL},
            {"state": "unverified", "kind": AccessIssue.NETWORK.value,
             "message": AccessIssue.NETWORK.message, "url": ""},
            {"state": "valid", "account": "tester", "model_access": False,
             "kind": AccessIssue.TERMS_NOT_ACCEPTED.value,
             "message": AccessIssue.TERMS_NOT_ACCEPTED.message, "url": hf_access.model_url()},
        ]
        shown = []
        for result in results:
            with patch.object(hf_access, "verify_configuration", return_value=result):
                widget.verify_button.click()
                self.assertTrue(wait_until(self.app, lambda: not widget.busy() and widget.state != "checking"))
            shown.append((widget.state, widget.state_label.text(), widget.message_label.text()))
        self.assertIn("Non valido", shown[0][1])
        self.assertIn("Non verificato", shown[1][1])
        self.assertIn("raggiungibile", shown[1][2])
        self.assertIn("Valido", shown[2][1])
        self.assertIn(hf_access.model_url(), shown[2][2])
        self.assertEqual(len({text for _s, _l, text in shown}), 3)

    def test_remove_after_confirmation(self):
        store = MemoryCredentialStore(SECRET)
        widget = self._widget(store)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            widget.remove_button.click()
        self.assertEqual(store.read(), SECRET)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            widget.remove_button.click()
        self.assertIsNone(store.read())
        self.assertEqual(widget.state, "not_configured")

    def test_environment_variable_is_reported_as_taking_precedence(self):
        with patch.dict(os.environ, {"HF_TOKEN": SECRET}):
            widget = self._widget(MemoryCredentialStore())
        self.assertIn("HF_TOKEN", widget.detail_label.text())
        self.assertIn("precedenza", widget.detail_label.text())
        self.assertFalse(widget.remove_button.isEnabled())
        self.assertNotIn(SECRET, all_text(widget))

    def test_legacy_plain_text_file_can_be_moved_into_the_vault(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / ".hf_token"
            legacy.write_text(SECRET, encoding="utf-8")
            store = MemoryCredentialStore()
            with patch("src.paths.hf_token_candidates", return_value=[legacy]):
                widget = self._widget(store)
                self.assertFalse(widget.legacy_button.isHidden())
                widget.legacy_button.click()
                self.assertFalse(legacy.exists())
                self.assertEqual(store.read(), SECRET)
                self.assertTrue(widget.legacy_button.isHidden())
                self.assertNotIn(SECRET, all_text(widget))


class FakeEnsure:
    """Stand-in for hf_access.ensure_model driving the dialog's steps."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def __call__(self, repo=hf_access.DIARIZATION_REPO, *, on_step=None, on_download=None, repair=False, **kwargs):
        self.calls.append(repair)
        on_step("cache", "failed", "Modello non presente in cache")
        on_step("token", "done", "Account tester")
        if self.error is not None:
            on_step("access", "done", "")
            on_step("terms", "failed", self.error.user_message)
            raise self.error
        for name in ("access", "terms", "download"):
            on_step(name, "done", "")
        on_download("plda.npz", 100)
        on_step("integrity", "done", "Checksum verificati")
        return "snapshot"


class DiarizationDialogTests(GuiTestCase):
    def _dialog(self, store, reason=None, *, cache="missing"):
        targets = [patch.object(hf_access, "local_cache_state", return_value=cache)]
        for target in targets:
            target.start()
            self.addCleanup(target.stop)
        dialog = self.track(DiarizationModelDialog(reason, store))
        self.addCleanup(lambda: wait_until(self.app, lambda: dialog._worker is None))
        return dialog

    def test_missing_token_is_explained_and_download_disabled(self):
        dialog = self._dialog(MemoryCredentialStore())
        self.assertEqual(dialog.step_state("cache"), "failed")
        self.assertEqual(dialog.step_state("token"), "failed")
        self.assertFalse(dialog.start_button.isEnabled())
        self.assertTrue(dialog.configure_button.isDefault())
        self.assertIn("non è presente", dialog.reason_label.text())

    def test_configure_then_download_accepts_and_reports_progress(self):
        store = MemoryCredentialStore()
        dialog = self._dialog(store)

        def configure(_self):
            store.write(SECRET)
            return 0
        fake = FakeEnsure()
        with patch.object(hf_settings.HuggingFaceDialog, "exec", configure), \
                patch.object(hf_access, "ensure_model", fake):
            dialog.configure_button.click()
            self.assertTrue(dialog.start_button.isEnabled())
            dialog.start_button.click()
            self.assertTrue(wait_until(self.app, lambda: dialog.outcome is not None and dialog._worker is None))
        self.assertTrue(dialog.outcome["ok"])
        self.assertTrue(wait_until(self.app, lambda: dialog.isHidden() and dialog.result() == 1))
        self.assertEqual(dialog.step_state("integrity"), "done")
        self.assertEqual(dialog.progress.value(), 100)

    def test_terms_not_accepted_offers_model_page_and_retry(self):
        dialog = self._dialog(MemoryCredentialStore(SECRET))
        fake = FakeEnsure(ModelAccessError(AccessIssue.TERMS_NOT_ACCEPTED))
        with patch.object(hf_access, "ensure_model", fake):
            dialog.start_button.click()
            self.assertTrue(wait_until(self.app, lambda: dialog.outcome is not None and dialog._worker is None))
        self.assertFalse(dialog.outcome["ok"])
        self.assertEqual(dialog.step_state("terms"), "failed")
        self.assertFalse(dialog.model_page_button.isHidden())
        self.assertEqual(dialog.start_button.text(), "Riprova")
        self.assertIn(hf_access.model_url(), dialog.reason_label.text())
        self.assertNotEqual(dialog.result(), 1)

    def test_corrupt_cache_offers_confirmed_repair(self):
        reason = ModelAccessError(AccessIssue.CACHE_CORRUPT).to_dict()
        dialog = self._dialog(MemoryCredentialStore(SECRET), reason, cache="incomplete")
        self.assertFalse(dialog.repair_button.isHidden())
        fake = FakeEnsure()
        with patch.object(hf_access, "ensure_model", fake), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            dialog.repair_button.click()
            self.assertTrue(wait_until(self.app, lambda: dialog.outcome is not None and dialog._worker is None))
        self.assertEqual(fake.calls, [True])

    def test_ready_model_needs_no_dialog(self):
        with patch.object(hf_access, "diarization_model_ready", return_value=True), \
                patch.object(hf_settings, "DiarizationModelDialog") as dialog:
            self.assertTrue(hf_settings.ensure_diarization_model())
        dialog.assert_not_called()


class ModelSettingsTabTests(GuiTestCase):
    def test_hugging_face_tab_is_part_of_the_settings(self):
        from src.gui.model_settings import ModelSettingsDialog
        hw = Hardware("cpu", "int8", "small", None, 0)
        with patch.object(ModelSettingsDialog, "_refresh", lambda *_a, **_k: None):
            dialog = self.track(ModelSettingsDialog(hw, current_transcription="small"))
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        self.assertIn("Hugging Face", titles)
        dialog.tabs.setCurrentIndex(titles.index("Hugging Face"))
        self.assertIsNone(dialog._selected_entry())
        self.assertIn("Nessun motore", dialog.active_info.text())


class FakeIdentityWorker(QObject):
    done = Signal(object)
    failed = Signal(str)
    access_failed = Signal(object)
    finished = Signal()
    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        FakeIdentityWorker.instances.append(self)

    def start(self):
        if len(FakeIdentityWorker.instances) == 1:
            self.access_failed.emit(ModelAccessError(AccessIssue.MODEL_NOT_CACHED).to_dict())
        else:
            self.failed.emit("seconda esecuzione")
        self.finished.emit()


class WorkflowResumeTests(GuiTestCase):
    def setUp(self):
        super().setUp()
        from src.gui.main_window import MainWindow
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "small", None, 0),
        ):
            self.window = MainWindow()

    def tearDown(self):
        self.window._dirty = False
        self.window.player.clear()
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_settings_menu_has_a_hugging_face_entry(self):
        self.assertEqual(self.window.hf_settings_action.text(), "Hugging Face…")

    def test_declined_setup_does_not_start_processing(self):
        window = self.window
        window._diarization_setup = staticmethod(lambda parent, reason=None: False)
        window.audio_profile_combo.setCurrentIndex(max(1, window.audio_profile_combo.findData("stereo_ambient")))
        window.diarize_check.setChecked(True)
        with patch("src.gui.main_window.ProcessWorker") as worker:
            window._run_pipeline(Path("a.wav"), None)
        worker.assert_not_called()
        self.assertIn("non disponibile", window.status_label.text())

    def test_access_failure_reopens_setup_and_resumes_the_same_request(self):
        window = self.window
        prompts = []
        window._diarization_setup = staticmethod(lambda parent, reason=None: (prompts.append(reason), True)[1])
        window._pipeline_request = (Path("a.wav"), None, [0.0], False)
        info = ModelAccessError(AccessIssue.TOKEN_MISSING).to_dict()
        window._on_access_failed(info)
        self.assertFalse(window._busy)
        self.assertIn("token", window.status_label.text().lower())
        with patch.object(type(window), "_run_pipeline") as run:
            window._resume_pipeline_after_setup()
        self.assertEqual(prompts, [info])
        run.assert_called_once_with(Path("a.wav"), None, [0.0], False, resume=True)
        self.assertIsNone(window._pending_access_error)

    def test_batch_queue_resumes_after_setup(self):
        window = self.window
        window._diarization_setup = staticmethod(lambda parent, reason=None: True)
        window._pending_batch_access_error = ModelAccessError(AccessIssue.NETWORK).to_dict()
        with patch.object(type(window), "_start_batch_processing") as start:
            window._on_batch_finished()
            self.assertTrue(wait_until(self.app, lambda: start.called))

    def test_global_review_is_repeated_after_setup_without_restart(self):
        from src.models.session import AudioSources, Session
        window = self.window
        window.session = Session(audio=AudioSources(mic="a.wav"), audio_profile="stereo_ambient")
        window._showing_raw = False
        setups = []
        window._diarization_setup = staticmethod(lambda parent, reason=None: (setups.append(reason), True)[1])
        FakeIdentityWorker.instances = []
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"")
            with patch("src.gui.main_window.paths.resolve_media", return_value=audio), \
                    patch("src.gui.identity_worker.GlobalIdentityWorker", FakeIdentityWorker):
                window._review_global_identities()
                self.assertTrue(wait_until(self.app, lambda: len(FakeIdentityWorker.instances) == 2))
        self.assertEqual(setups[0], None)           # pre-check before the first run
        self.assertEqual(setups[1]["kind"], AccessIssue.MODEL_NOT_CACHED.value)
        self.assertIn("seconda esecuzione", window.status_label.text())


class WorkerSignalTests(GuiTestCase):
    def test_process_worker_emits_access_failure_and_redacts_other_errors(self):
        from src.gui.workers import ProcessWorker
        worker = ProcessWorker("a.wav", None, MagicMock())
        access, failed = [], []
        worker.access_failed.connect(access.append)
        worker.failed.connect(failed.append)
        with patch("src.gui.workers.process_recording",
                   side_effect=ModelAccessError(AccessIssue.TERMS_NOT_ACCEPTED)):
            worker.run()
        with patch("src.gui.workers.process_recording", side_effect=RuntimeError(f"rifiutato {SECRET}")):
            worker.run()
        self.assertEqual(access[0]["kind"], AccessIssue.TERMS_NOT_ACCEPTED.value)
        self.assertEqual(len(failed), 1)
        self.assertNotIn(SECRET, failed[0])

    def test_batch_stops_at_the_first_access_failure(self):
        from src.gui.workers import BatchProcessWorker
        worker = BatchProcessWorker([Path("a.wav"), Path("b.wav")], MagicMock())
        access, failed_files = [], []
        worker.access_failed.connect(access.append)
        worker.file_failed.connect(lambda path, message: failed_files.append(path))
        with patch("src.gui.workers.process_recording",
                   side_effect=ModelAccessError(AccessIssue.MODEL_NOT_CACHED)) as run:
            worker.run()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(failed_files, [Path("a.wav")])
        self.assertEqual(len(access), 1)


if __name__ == "__main__":
    unittest.main()
