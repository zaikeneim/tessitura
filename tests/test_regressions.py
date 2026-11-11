"""CPU-only regression checks: python -m unittest discover -s tests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QMessageBox

from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from src.models.session import Session, Segment, Word
from src.models import storage
from src.processing.merge import _fill_gaps


class SessionTests(unittest.TestCase):
    def test_group_adjacent_speakers_preserves_raw_segments_and_words(self):
        raw = [
            Segment(0, 1, "Ciao", "A", [Word(0, 0.5, "Ciao")]),
            Segment(1, 2, "mondo.", "A", [Word(1, 1.5, "mondo.")]),
            Segment(2, 3, "Risposta", "B", [Word(2, 2.5, "Risposta")]),
        ]
        session = Session(segments=raw.copy(), raw_segments=[Segment(
            item.start, item.end, item.text, item.speaker,
            [Word(word.start, word.end, word.text, word.probability) for word in item.words],
        ) for item in raw])

        self.assertEqual(session.adjacent_same_speaker_groups(), 1)
        self.assertEqual(session.group_adjacent_speaker_segments(), 1)
        self.assertEqual(len(session.segments), 2)
        self.assertEqual(session.segments[0].text, "Ciao mondo.")
        self.assertEqual(session.segments[0].start, 0)
        self.assertEqual(session.segments[0].end, 2)
        self.assertEqual([word.text for word in session.segments[0].words], ["Ciao", "mondo."])
        self.assertEqual(len(session.raw_segments), 3)
        self.assertEqual(session.raw_segments[0].text, "Ciao")
        self.assertEqual(session.transcript_review["grouped_count"], 1)
        self.assertEqual(session.group_adjacent_speaker_segments(), 0)

    def test_group_adjacent_speakers_does_not_merge_alternating_voices(self):
        session = Session(segments=[
            Segment(0, 1, "uno", "A"),
            Segment(1, 2, "due", "B"),
            Segment(2, 3, "tre", "A"),
        ])
        self.assertEqual(session.adjacent_same_speaker_groups(), 0)
        self.assertEqual(session.group_adjacent_speaker_segments(), 0)
        self.assertEqual([item.text for item in session.segments], ["uno", "due", "tre"])

    def test_windowed_speaker_matching_matches_full_scan(self):
        import random
        from src.processing import merge
        from src.processing.diarize import Turn
        rng = random.Random(42)
        for _ in range(50):
            units = [merge.WordUnit(i*0.5, i*0.5+0.4, " word.") for i in range(80)]
            turns = [Turn(start, start+rng.uniform(0.1, 8), rng.choice(["A", "B"]))
                     for start in [rng.uniform(0, 40) for _ in range(30)]]
            expected = [None]*len(units)
            for a, b in merge._phrases(units):
                expected[a:b] = merge._phrase_labels(units, a, b, turns)
            expected = merge._fill_gaps(merge._smooth(units, expected))
            self.assertEqual(merge.assign_speakers(units, turns), expected)

    def test_duration_includes_overlapping_segments(self):
        session = Session(segments=[Segment(0, 20, "a"), Segment(2, 3, "b")])
        self.assertEqual(session.duration, 20)

    def test_audio_profile_round_trips_and_old_sessions_still_load(self):
        session = Session(audio_profile="stereo_ambient")
        restored = Session.from_dict(session.to_dict())
        self.assertEqual(restored.audio_profile, "stereo_ambient")
        self.assertIsNone(Session.from_dict({}).audio_profile)

    def test_atomic_save_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            storage.save_session(Session(name="original"), path)
            with patch("src.models.storage.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    storage.save_session(Session(name="changed"), path)
            self.assertEqual(storage.load_session(path).name, "original")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_gap_fill_preserves_nearest_and_left_ties(self):
        import random
        rng = random.Random(4)
        for _ in range(100):
            labels = [rng.choice([None, None, "A", "B"]) for _ in range(100)]
            known = [i for i, value in enumerate(labels) if value is not None]
            expected = [value if value is not None else
                        labels[min(known, key=lambda k: abs(k-i))]
                        for i, value in enumerate(labels)]
            self.assertEqual(_fill_gaps(labels), expected)


class EncodeTests(unittest.TestCase):
    def test_unequal_mono_tracks_preserve_offset_and_tail(self):
        import numpy as np
        import soundfile as sf
        from src.audio.encode import merge_to_stereo_mp3
        with tempfile.TemporaryDirectory() as folder:
            mic = Path(folder) / "record_mic.wav"
            system = Path(folder) / "record_system.wav"
            sr = 48000
            tone = lambda n: (0.3*np.sin(2*np.pi*440*np.arange(n)/sr)).astype("float32")
            sf.write(mic, tone(sr), sr)
            sf.write(system, tone(3*sr), sr)
            result = merge_to_stereo_mp3(mic, system, offsets_s=[0, 0.5], delete_sources=False)
            data, rate = sf.read(result, always_2d=True)
            self.assertEqual(data.shape[1], 2)
            self.assertAlmostEqual(len(data)/rate, 3.5, delta=0.1)
            self.assertGreater(np.std(data[:rate//2, 0]), 0.1)
            self.assertLess(np.std(data[:rate//3, 1]), 0.01)
            self.assertGreater(np.std(data[3*rate:, 1]), 0.1)
            self.assertTrue(mic.exists() and system.exists())


class RecordingSyncRegressionTests(unittest.TestCase):
    def test_recording_result_offsets_preserve_start_alignment(self):
        import threading
        from types import SimpleNamespace
        from src.audio.capture import Recorder, RecordingResult
        recorder = Recorder()
        recorder._mic_rec = SimpleNamespace(
            _stop=threading.Event(), stopped_at=None,
            started_at=100.25, error=None, dropped_seconds=lambda: 0.0,
            stop=lambda: Path("mic.wav"),
        )
        recorder._sys_rec = SimpleNamespace(
            _stop=threading.Event(), stopped_at=None,
            started_at=100.00, error=None, dropped_seconds=lambda: 0.0,
            stop=lambda: Path("sys.wav"),
        )
        recorder.is_recording = True
        res = recorder.stop()
        self.assertEqual([res.mic_path, res.system_path], [Path("mic.wav"), Path("sys.wav")])
        offsets = res.offsets()
        self.assertEqual(len(offsets), 2)
        self.assertAlmostEqual(offsets[0], 0.25, places=6)
        self.assertAlmostEqual(offsets[1], 0.0, places=6)

        single = RecordingResult(mic_path=Path("mic.wav"), mic_offset=0.0)
        self.assertEqual(single.offsets(), [0.0])

    def test_mix_sources_preserves_original_tracks_and_peak_ceiling(self):
        import numpy as np
        import soundfile as sf
        from src.processing import audio_utils
        with tempfile.TemporaryDirectory() as folder:
            mic = Path(folder) / "mic.wav"
            sys_wav = Path(folder) / "sys.wav"
            sr = 16000
            sig_mic = np.full(sr, 0.8, dtype="float32")
            sig_sys = np.full(sr, 0.8, dtype="float32")
            sf.write(mic, sig_mic, sr)
            sf.write(sys_wav, sig_sys, sr)
            mic_bytes = mic.read_bytes()
            sys_bytes = sys_wav.read_bytes()

            out = Path(folder) / "mixed.wav"
            audio_utils.mix_sources([mic, sys_wav], out, offsets_s=[0.0, 0.25])

            self.assertEqual(mic.read_bytes(), mic_bytes)
            self.assertEqual(sys_wav.read_bytes(), sys_bytes)
            mixed, out_sr = sf.read(out, dtype="float32")
            self.assertEqual(out_sr, 16000)
            self.assertEqual(len(mixed), int(1.25 * sr))
            self.assertLessEqual(float(np.max(np.abs(mixed))), audio_utils.PEAK_CEILING + 1e-4)

    def test_split_dual_channel_and_source_dominance(self):
        import numpy as np
        import soundfile as sf
        from src.processing.audio_utils import split_dual_channel, source_dominance
        with tempfile.TemporaryDirectory() as folder:
            sr = 48000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            ch0 = np.zeros(sr, dtype="float32")
            ch1 = np.zeros(sr, dtype="float32")
            ch0[:sr // 2] = (0.4 * np.sin(2 * np.pi * 220 * t[:sr // 2])).astype("float32")
            ch1[sr // 2:] = (0.4 * np.sin(2 * np.pi * 660 * t[sr // 2:])).astype("float32")
            stereo = Path(folder) / "dual_stereo.wav"
            sf.write(stereo, np.column_stack([ch0, ch1]), sr)

            split_files = split_dual_channel(stereo, folder)
            self.assertEqual(len(split_files), 2)
            self.assertTrue(all(p.is_file() for p in split_files))

            winner, frame_s = source_dominance(split_files)
            self.assertAlmostEqual(frame_s, 0.02, places=4)
            # First half dominated by channel 0 (mic), second half by channel 1 (system)
            self.assertEqual(int(np.median(winner[:15])), 0)
            self.assertEqual(int(np.median(winner[30:45])), 1)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from src.gui.theme import apply_theme
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_highlight_resets_between_sessions_and_in_gaps(self):
        from src.gui.transcript_view import TranscriptView
        view = TranscriptView()
        session = Session(segments=[Segment(0, 1, "hello"), Segment(3, 4, "again")])
        view.set_session(session)
        view.highlight_at(0.5)
        self.assertEqual(view.currentRow(), 0)
        view.set_session(session)
        view.highlight_at(0.5)
        self.assertTrue(view.selectionModel().hasSelection())
        view.highlight_at(2)
        self.assertFalse(view.selectionModel().hasSelection())

    def test_editing_text_does_not_seek(self):
        from src.gui.transcript_view import TranscriptView
        view = TranscriptView()
        view.set_session(Session(segments=[Segment(1, 2, "hello")]))
        calls = []
        view.seekRequested.connect(calls.append)
        view._on_cell_clicked(0, 2)
        self.assertEqual(calls, [])
        view._on_cell_clicked(0, 0)
        self.assertEqual(calls, [1.0])

    def test_recording_encoding_and_missing_audio_states(self):
        from src.gui.main_window import MainWindow
        from src.hardware import Hardware
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "base")
        ):
            window = MainWindow()
        window.last_mic = Path("test.wav")
        window.recorder.is_recording = True
        window._set_busy(False)
        self.assertFalse(window.process_btn.isEnabled())
        self.assertFalse(window.process_file_btn.isEnabled())
        self.assertTrue(window.record_btn.isEnabled())
        window.recorder.is_recording = False
        window._encoding = True
        window._set_busy(False)
        self.assertFalse(window.process_file_btn.isEnabled())
        self.assertFalse(window.record_btn.isEnabled())
        window._encoding = False
        from src.processing.audio_profile import AudioProfile
        window._set_audio_profile(None)
        window._set_busy(False)
        self.assertFalse(window.process_btn.isEnabled())
        window._set_audio_profile(AudioProfile.STEREO_AMBIENT)
        window._set_busy(False)
        self.assertTrue(window.process_btn.isEnabled())
        window.player.setEnabled(True)
        self.assertFalse(window._load_audio(Session()))
        self.assertFalse(window.player.isEnabled())
        with patch.object(window, "_confirm_discard", return_value=True):
            window.close()

    def test_close_waits_for_worker_and_keeps_unsaved_session(self):
        from PySide6.QtGui import QCloseEvent
        from src.gui.main_window import MainWindow
        from src.hardware import Hardware
        from unittest.mock import Mock
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "base")
        ):
            window = MainWindow()
        window._worker = Mock()
        window._worker.isRunning.return_value = True
        event = QCloseEvent()
        window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        window._worker = None
        with patch.object(window, "_confirm_discard", return_value=False):
            event = QCloseEvent()
            window.closeEvent(event)
            self.assertFalse(event.isAccepted())
        window.close()

    def test_file_menu_groups_consecutive_speakers(self):
        from src.gui.main_window import MainWindow
        from src.hardware import Hardware
        with patch.object(MainWindow, "_populate_devices"), patch(
            "src.gui.main_window.detect", return_value=Hardware("cpu", "int8", "base")
        ):
            window = MainWindow()
        window.session = Session(segments=[
            Segment(0, 1, "prima", "A"),
            Segment(1, 2, "seconda", "A"),
            Segment(2, 3, "terza", "B"),
        ])
        window.speakers.set_session(window.session)
        with patch("src.gui.main_window.QMessageBox.question", return_value=QMessageBox.Yes):
            window._group_segments()
        self.assertEqual(len(window.session.segments), 2)
        self.assertEqual(window.session.segments[0].text, "prima seconda")
        self.assertTrue(window._dirty)
        self.assertFalse(window.group_segments_action.isEnabled())
        with patch.object(window, "_confirm_discard", return_value=True):
            window.close()


if __name__ == "__main__":
    unittest.main()
