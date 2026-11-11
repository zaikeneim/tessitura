"""Recording controls preserve modes, endpoints, source provenance and lifecycle."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from PySide6.QtWidgets import QApplication
from src.audio.capture import RecordingResult
from src.audio.devices import DeviceCatalog, DeviceInfo
from src.gui.main_window import MainWindow
from src.hardware import Hardware
from src.processing.audio_profile import AudioProfile


class RecordingUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.mic = DeviceInfo(0, "Room mic", 2, 44100, False, 0)
        self.headphones = DeviceInfo(2, "Headphones [Loopback]", 2, 48000, True, 0)
        self.catalog = DeviceCatalog([self.mic], [self.headphones], self.mic, self.headphones)
        with patch("src.gui.main_window.get_devices", return_value=self.catalog), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "small", None, 0)
        ):
            self.window = MainWindow()

    def tearDown(self):
        self.window._timer.stop()
        self.window.recorder.is_recording = False
        self.window._dirty = False
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_all_modes_keep_output_and_system_capture_selectable(self):
        window = self.window
        for index, mode in enumerate(("ambient", "headset", "speakers")):
            window.recording_mode_combo.setCurrentIndex(index)
            self.assertEqual(window.recording_mode_combo.currentData(), mode)
            self.assertTrue(window.sys_combo.isEnabled())
            self.assertTrue(window.sys_check.isEnabled())
            self.assertFalse(window.mic_verified_check.isChecked())
        window.recording_mode_combo.setCurrentIndex(0)
        self.assertTrue(window.sys_check.isChecked())  # Speakers are allowed in ambient mode.

    def test_refresh_preserves_endpoint_and_requires_reselection_after_disconnect(self):
        window = self.window
        new_headphones = DeviceInfo(8, self.headphones.name, 2, 44100, True, 0)
        catalog = DeviceCatalog([self.mic], [new_headphones], self.mic, new_headphones)
        with patch("src.gui.main_window.get_devices", return_value=catalog):
            window._populate_devices()
            window._populate_devices()
        self.assertEqual(window.sys_combo.count(), 1)
        self.assertEqual(window.sys_combo.currentData().index, 8)
        speakers = DeviceInfo(9, "Speakers [Loopback]", 2, 48000, True, 0)
        with patch("src.gui.main_window.get_devices", return_value=DeviceCatalog([self.mic], [speakers], self.mic, speakers)):
            window._populate_devices()
        self.assertIsNone(window.sys_combo.currentData())
        with patch.object(window.recorder, "start") as start, patch("src.gui.main_window.QMessageBox.warning"):
            window._start_record()
        start.assert_not_called()

    def test_headset_start_sends_explicit_topology_without_speaker_count_guess(self):
        window = self.window
        window.recording_mode_combo.setCurrentIndex(1)
        window.speakers_spin.setValue(5)
        with patch.object(window.recorder, "start") as start:
            window._start_record()
        kwargs = start.call_args.kwargs
        self.assertEqual(kwargs["recording_mode"], "headset")
        self.assertEqual(kwargs["system_device"], self.headphones)
        self.assertFalse(kwargs["mic_single_speaker_verified"])
        self.assertEqual(window._pipeline_strategy, "source_aware")
        self.assertEqual(window.speakers_spin.value(), 5)

    def test_changing_capture_endpoint_invalidates_single_speaker_verification(self):
        window = self.window
        window.mic_verified_check.setChecked(True)
        window.mic_combo.addItem("Room microphone", DeviceInfo(5, "Room microphone", 2, 48000, False, 0))
        window.mic_combo.setCurrentIndex(1)
        self.assertFalse(window.mic_verified_check.isChecked())
        window.mic_verified_check.setChecked(True)
        window.sys_combo.addItem("Speakers", DeviceInfo(6, "Speakers [Loopback]", 2, 48000, True, 0))
        window.sys_combo.setCurrentIndex(1)
        self.assertFalse(window.mic_verified_check.isChecked())

    def test_batch_options_do_not_inherit_previous_capture_topology(self):
        window = self.window
        window._set_audio_profile(AudioProfile.STEREO_AMBIENT)
        window._pipeline_strategy = "source_aware"
        window._last_recording_config = dict(recording_mode="headset", mic_single_speaker_verified=True,
                                             echo_cancellation=True, recording_metadata={"private": "capture"})
        options = window._build_options(for_recording=False)
        self.assertEqual(options.strategy, "standard")
        self.assertIsNone(options.recording_mode)
        self.assertIsNone(options.recording_metadata)
        self.assertFalse(options.mic_single_speaker_verified)
        self.assertFalse(options.echo_cancellation)

    def test_mp3_export_keeps_original_sources_and_offsets(self):
        window = self.window
        window.last_mic, window.last_system = Path("mic.wav"), Path("system.wav")
        window.last_offsets = [.125, 0]
        window._on_encoded(Path("copy.mp3"), True, "")
        self.assertEqual((window.last_mic, window.last_system), (Path("mic.wav"), Path("system.wav")))
        self.assertEqual(window.last_offsets, [.125, 0])

    def test_ambient_capture_stays_transcribable_without_single_speaker_assumption(self):
        window = self.window
        window._on_record_stopped(RecordingResult(mic_path=Path("room.wav")))
        self.assertEqual(window.audio_profile_combo.currentData(), AudioProfile.STEREO_AMBIENT.value)
        options = window._build_options()
        self.assertFalse(options.mic_single_speaker_verified)
        self.assertTrue(window.process_btn.isEnabled())

    def test_import_restores_originals_and_absolute_offsets_from_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mic, system, copy = root/"sample_mic.wav", root/"sample_system.wav", root/"sample.mp3"
            for path in (mic, system, copy):
                path.write_bytes(b"private fixture")
            metadata = {"version": 1, "state": "complete", "recording_mode": "headset", "mic_single_speaker_verified": True,
                        "echo_cancellation": False, "tracks": {
                            "mic": {"path": mic.name, "offset_seconds": 0},
                            "system": {"path": system.name, "offset_seconds": .185}}}
            (root/"sample.recording.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch("src.gui.main_window.QFileDialog.getOpenFileNames", return_value=([str(copy)], "")):
                self.window._process_file()
            self.assertEqual((self.window.last_mic, self.window.last_system), (mic, system))
            self.assertEqual(self.window.last_offsets, [0, .185])
            options = self.window._build_options()
            self.assertEqual(options.recording_mode, "headset")
            self.assertTrue(options.mic_single_speaker_verified)
            self.assertEqual(options.recording_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
