import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.processing.checkpoints import CheckpointStore, cache_key


class CheckpointTests(unittest.TestCase):
    def test_key_changes_for_audio_options_models_and_algorithm(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio.wav"
            audio.write_bytes(b"audio")
            base = cache_key([audio], {"language": "it"}, {"asr": "a"})
            self.assertNotEqual(base, cache_key([audio], {"language": "en"}, {"asr": "a"}))
            self.assertNotEqual(base, cache_key([audio], {"language": "it"}, {"asr": "b"}))
            self.assertNotEqual(base, cache_key([audio], {"language": "it"}, {"asr": "a"}, algorithm="new"))
            audio.write_bytes(b"changed")
            self.assertNotEqual(base, cache_key([audio], {"language": "it"}, {"asr": "a"}))

    def test_atomic_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder)
            key = "a" * 64
            path = store.save(key, "asr", {"segments": [1, 2]})
            self.assertEqual(store.load(key, "asr"), {"segments": [1, 2]})
            self.assertFalse(list(Path(folder).rglob("*.tmp")))
            self.assertTrue(path.is_file())

    def test_corrupt_or_wrong_version_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder)
            key = "b" * 64
            path = store.save(key, "review", {"ok": True})
            data = json.loads(path.read_text())
            data["version"] = "old"
            path.write_text(json.dumps(data))
            self.assertIsNone(store.load(key, "review"))

    def test_invalidate_removes_later_phases_only(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder)
            key = "c" * 64
            for phase in ("prepare", "asr", "diarization", "review"):
                store.save(key, phase, {"phase": phase})
            self.assertEqual(store.invalidate(key, "diarization"), 2)
            self.assertIsNotNone(store.load(key, "asr"))
            self.assertIsNone(store.load(key, "diarization"))


if __name__ == "__main__":
    unittest.main()
