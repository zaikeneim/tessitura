"""Test per la profilazione delle voci (VoiceProfile)."""

import unittest

from src.processing.diarize import Turn
from src.processing.voice_profile import (
    VoiceProfile,
    build_voice_profiles,
    match_with_known_profiles,
)


class VoiceProfileTests(unittest.TestCase):
    def test_build_voice_profiles_durations_and_turns(self):
        turns = [
            Turn(0.0, 5.0, "SPEAKER_00"),
            Turn(5.5, 9.0, "SPEAKER_01"),
            Turn(9.2, 14.2, "SPEAKER_00"),
        ]
        profiles = build_voice_profiles(turns)
        self.assertIn("SPEAKER_00", profiles)
        self.assertIn("SPEAKER_01", profiles)

        p0 = profiles["SPEAKER_00"]
        self.assertAlmostEqual(p0.total_duration, 10.0, places=2)
        self.assertEqual(p0.turn_count, 2)

        p1 = profiles["SPEAKER_01"]
        self.assertAlmostEqual(p1.total_duration, 3.5, places=2)
        self.assertEqual(p1.turn_count, 1)

    def test_build_voice_profiles_with_embeddings_and_matching(self):
        turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]
        # Vettori ortogonali
        emb_0 = [1.0, 0.0, 0.0, 0.0]
        emb_1 = [0.0, 1.0, 0.0, 0.0]
        known = {
            "Marco": [0.99, 0.01, 0.0, 0.0],
            "Elena": [0.01, 0.99, 0.0, 0.0],
        }
        profiles = build_voice_profiles(
            turns,
            speaker_embeddings={"SPEAKER_00": emb_0, "SPEAKER_01": emb_1},
            known_profiles=known,
        )
        self.assertEqual(profiles["SPEAKER_00"].name, "Marco")
        self.assertTrue(profiles["SPEAKER_00"].is_known_profile)
        self.assertEqual(profiles["SPEAKER_01"].name, "Elena")
        self.assertTrue(profiles["SPEAKER_01"].is_known_profile)

    def test_channel_dominance_assignment(self):
        turns = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 4.0, "SPEAKER_01")]
        frame_s = 0.5
        # 8 frame: i primi 4 canale 0 (mic), i secondi 4 canale 1 (sistema)
        dominance = [0, 0, 0, 0, 1, 1, 1, 1]
        profiles = build_voice_profiles(
            turns,
            channel_dominance=dominance,
            frame_s=frame_s,
        )
        self.assertEqual(profiles["SPEAKER_00"].channel_dominance, 0)
        self.assertEqual(profiles["SPEAKER_01"].channel_dominance, 1)

    def test_voice_profile_serialization(self):
        vp = VoiceProfile(
            speaker_id="SPEAKER_00",
            name="Alice",
            embedding=[0.5, 0.5, 0.5, 0.5],
            total_duration=12.345,
            turn_count=3,
            channel_dominance=0,
            confidence=0.95,
            is_known_profile=True,
        )
        data = vp.to_dict()
        restored = VoiceProfile.from_dict(data)
        self.assertEqual(restored.speaker_id, "SPEAKER_00")
        self.assertEqual(restored.name, "Alice")
        self.assertEqual(restored.total_duration, 12.345)
        self.assertTrue(restored.is_known_profile)


if __name__ == "__main__":
    unittest.main()
