"""Incomplete or ambiguous provenance must never invent synchronized sources."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.audio.recording_metadata import load_recording_metadata


class RecordingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / "sample_mic.wav"
        self.source.write_bytes(b"original")
        self.sidecar = self.root / "sample.recording.json"
        self.valid = {"version": 1, "state": "complete", "recording_mode": "ambient",
                      "tracks": {"mic": {"path": self.source.name, "offset_seconds": .125}}}

    def write(self, value):
        self.sidecar.write_text(json.dumps(value), encoding="utf-8")

    def test_missing_sidecar_leaves_imported_audio_unclassified(self):
        self.assertIsNone(load_recording_metadata(self.source))

    def test_unfinished_capture_cannot_claim_synchronization(self):
        self.valid["state"] = "recording"
        self.write(self.valid)
        with self.assertRaisesRegex(ValueError, "incompleta"):
            load_recording_metadata(self.source)
        self.valid["state"] = "complete"
        del self.valid["tracks"]["mic"]["offset_seconds"]
        self.write(self.valid)
        with self.assertRaisesRegex(ValueError, "Offset"):
            load_recording_metadata(self.source)

    def test_wrong_shapes_and_non_boolean_verification_fail_safely(self):
        for value in ([], {**self.valid, "tracks": {"mic": []}},
                      {**self.valid, "mic_single_speaker_verified": "false"}):
            with self.subTest(value=value):
                self.write(value)
                with self.assertRaises(ValueError):
                    load_recording_metadata(self.source)

    def test_single_source_mp3_export_restores_original_track(self):
        self.write(self.valid)
        export = self.source.with_suffix(".mp3")
        export.write_bytes(b"export")
        self.assertEqual(load_recording_metadata(export), self.valid)

    def test_metadata_cannot_redirect_outside_its_recording_folder(self):
        self.valid["tracks"]["mic"]["path"] = "../other.wav"
        self.write(self.valid)
        with self.assertRaisesRegex(ValueError, "Percorso"):
            load_recording_metadata(self.source)

    def test_encoder_defaults_preserve_originals(self):
        from src.audio.encode import to_mp3, merge_to_stereo_mp3
        with patch("src.audio.encode._duration", return_value=1), patch("src.audio.encode._encode_atomic"), patch(
            "src.audio.encode._remove_sources"
        ) as remove:
            to_mp3(self.source)
            merge_to_stereo_mp3(self.source, self.root / "sample_system.wav")
        remove.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
