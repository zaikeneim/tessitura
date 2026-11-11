import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.processing import model_store as store


class ModelStoreSafetyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = patch.dict(os.environ, {"HF_HUB_CACHE": self.folder.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.repo = "Qwen/test"
        self.old, self.new = "a" * 40, "b" * 40

    def snapshot(self, revision):
        p = store.repo_root(self.repo) / "snapshots" / revision
        p.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (p / name).write_text("{}", encoding="utf-8")
        (p / "model.safetensors").write_bytes(b"test weights")
        return p

    def activate(self, path):
        with store.lease(self.repo):
            store.activate_loaded(self.repo, path, "analisi")

    def test_manifest_requires_every_shard(self):
        p = self.snapshot(self.old)
        (p / "model.safetensors").unlink()
        (p / "model-00001.safetensors").write_bytes(b"first")
        (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
        with self.assertRaises(ValueError):
            store.verify_snapshot(p, "analisi")

    def test_no_lexicographic_sha_guess(self):
        self.snapshot(self.old)
        self.snapshot(self.new)
        self.assertIsNone(store.local_revision(self.repo))
        self.assertIsNone(store.cached_path(self.repo, "analisi"))

    def test_corrupt_pointer_is_reported_as_unverified(self):
        pointer = store._pointer(self.repo)
        pointer.parent.mkdir(parents=True)
        pointer.write_text("{not-json", encoding="utf-8")
        self.assertIsNone(store.local_revision(self.repo))
        self.assertIsNone(store.cached_path(self.repo, "analisi"))

    def test_loaded_model_cannot_be_deleted(self):
        p = self.snapshot(self.old)
        with store.lease(self.repo):
            with self.assertRaises(RuntimeError):
                store.remove(self.repo)
        self.assertTrue(p.is_dir())

    def test_successful_activation_and_rollback(self):
        one, two = self.snapshot(self.old), self.snapshot(self.new)
        self.activate(one)
        self.activate(two)
        self.assertEqual(store.local_revision(self.repo), self.new)
        self.assertEqual(Path(store.rollback(self.repo, "analisi")), one)
        self.assertTrue(two.exists())

    def test_failed_load_does_not_activate_candidate(self):
        one, two = self.snapshot(self.old), self.snapshot(self.new)
        self.activate(one)
        info = SimpleNamespace(sha=self.new, siblings=[])
        with patch("huggingface_hub.HfApi") as api, patch(
                "huggingface_hub.snapshot_download", return_value=str(two)) as download:
            api.return_value.model_info.return_value = info
            with self.assertRaises(RuntimeError):
                store.ensure(self.repo, "analisi", update=True,
                             validate_load=lambda _: (_ for _ in ()).throw(RuntimeError("OOM")))
            self.assertEqual(download.call_args.kwargs["revision"], self.new)
        self.assertEqual(store.cached_path(self.repo, "analisi"), one)

    def test_atomic_activation_failure_keeps_old_pointer(self):
        one, two = self.snapshot(self.old), self.snapshot(self.new)
        self.activate(one)
        with patch("src.processing.model_store.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.activate(two)
        self.assertEqual(store.local_revision(self.repo), self.old)

    def test_remote_size_or_checksum_mismatch_fails(self):
        p = self.snapshot(self.old)
        bad = SimpleNamespace(rfilename="model.safetensors", size=12, lfs={"sha256": "0" * 64})
        with self.assertRaises(ValueError):
            store.verify_snapshot(p, "analisi", [bad])

    def test_offline_cache_never_contacts_hub(self):
        p = self.snapshot(self.old)
        self.activate(p)
        with patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            self.assertEqual(store.ensure(self.repo, "analisi", allow_download=False), str(p))


if __name__ == "__main__":
    unittest.main()
