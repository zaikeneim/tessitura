import tempfile
import unittest
from pathlib import Path

from scripts.build_phase0_corpus import build_manifest


class Phase0CorpusTests(unittest.TestCase):
    def test_same_start_turns_are_retained_as_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "sample.mp3"
            transcript = root / "sample.txt"
            audio.write_bytes(b"audio")
            transcript.write_text(
                "[Master 00:00:00] introduzione\n"
                "[A 00:00:05] prima\n"
                "[B 00:00:05] sovrapposta\n"
                "[A 00:00:09] fine\n",
                encoding="utf-8",
            )
            manifest, readable, stm, rttm = build_manifest(
                audio, transcript, duration=20, window_start=0,
                window_end=10, audio_profile="stereo_ambient",
            )

        self.assertEqual(len(manifest["turns"]), 4)
        self.assertEqual(manifest["turns"][1]["start"], 5)
        self.assertEqual(manifest["turns"][1]["end"], 9)
        self.assertEqual(manifest["turns"][2]["start"], 5)
        self.assertEqual(manifest["turns"][2]["end"], 9)
        self.assertIn("A 00:00:05", readable)
        self.assertIn("sample_phase0 1 A 5.000 9.000", stm)
        self.assertIn("SPEAKER sample_phase0 1 5.000 4.000", rttm)

    def test_window_and_profile_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "sample.mp3"
            transcript = root / "sample.txt"
            audio.write_bytes(b"audio")
            transcript.write_text("[A 00:00:02] testo\n", encoding="utf-8")
            manifest, *_ = build_manifest(
                audio, transcript, duration=20, window_start=1,
                window_end=5, audio_profile="mic_system",
            )
        self.assertEqual(manifest["audio_profile"], "mic_system")
        self.assertEqual(manifest["window"]["duration_s"], 4)

    def test_verification_levels_and_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "sample.mp3"
            transcript = root / "sample.txt"
            audio.write_bytes(b"audio")
            transcript.write_text("[A 00:00:02] testo\n", encoding="utf-8")

            # Default: imported_draft
            manifest_default, *_ = build_manifest(
                audio, transcript, duration=20, window_start=1,
                window_end=5, audio_profile="stereo_ambient",
            )
            self.assertEqual(manifest_default["status"], "imported_draft_v1")
            self.assertEqual(
                manifest_default["annotation"]["verification_level"], "imported_draft"
            )

            # human_verified_timing_and_speakers -> gold
            manifest_gold, *_ = build_manifest(
                audio, transcript, duration=20, window_start=1,
                window_end=5, audio_profile="stereo_ambient",
                verification_level="human_verified_timing_and_speakers",
            )
            self.assertEqual(manifest_gold["status"], "gold_reference_v1")
            self.assertEqual(
                manifest_gold["annotation"]["verification_level"],
                "human_verified_timing_and_speakers",
            )

            # human_verified_text -> verified_text_draft_timing_v1
            manifest_text, *_ = build_manifest(
                audio, transcript, duration=20, window_start=1,
                window_end=5, audio_profile="stereo_ambient",
                verification_level="human_verified_text",
            )
            self.assertEqual(manifest_text["status"], "verified_text_draft_timing_v1")

            # Invalid level raises ValueError
            with self.assertRaises(ValueError):
                build_manifest(
                    audio, transcript, duration=20, window_start=1,
                    window_end=5, audio_profile="stereo_ambient",
                    verification_level="unsupported_level",
                )


if __name__ == "__main__":
    unittest.main()
