"""Hugging Face credentials, access checks and the verified diarization download.

No real credential and no network: the Hub is replaced by fakes, the vault by
an in-memory store (plus one round trip in the real Windows Credential
Manager under a throw-away target). Token-like values are built at runtime and
never match real token formats.

Esecuzione: python -m tests.test_hf_access
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub import errors as hub_errors

from src import secure_store
from src.processing import hf_access, model_store
from src.processing.hf_access import AccessIssue, ModelAccessError, TokenSource
from src.secure_store import CredentialStoreUnavailable, MemoryCredentialStore

REPO = hf_access.DIARIZATION_REPO
SECRET = "hf_" + "Zq9" * 6          # token-shaped, not a real format
OTHER_SECRET = "hf_" + "Wx7" * 6
CLEAN_ENV = {name: "" for name in hf_access.ENV_VARS}
GENERIC_OLD_MESSAGE = "Serve un token HuggingFace"


def _response(status: int, url: str = "https://huggingface.co/api/models/x/auth-check", headers=None):
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("GET", url))


class FakeApi:
    """Hub stand-in recording every call; never sees the network."""

    def __init__(self, *, whoami=None, auth=None, gated="auto"):
        self.calls: list[tuple[str, object]] = []
        self._whoami = whoami
        self._auth = auth
        self._gated = gated

    def whoami(self, token=None):
        self.calls.append(("whoami", token))
        if self._whoami:
            raise self._whoami
        return {"name": "tester", "auth": {"accessToken": {"role": "read"}}}

    def model_info(self, repo, token=None, timeout=None, **_kwargs):
        self.calls.append(("model_info", token))
        return SimpleNamespace(gated=self._gated, sha="a" * 40, siblings=[])

    def auth_check(self, repo, token=None):
        self.calls.append(("auth_check", token))
        if self._auth:
            raise self._auth


class TempHub:
    """Isolated Hugging Face cache, so tests never touch the user's models."""

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "hub"
        self.root.mkdir()
        self._env = patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root)})
        self._env.start()
        return self

    def __exit__(self, *exc):
        self._env.stop()
        self._dir.cleanup()

    def write_snapshot(self, sha: str, payload: bytes = b"weights", *, files=None):
        snapshot = model_store.repo_root(REPO) / "snapshots" / sha
        for name in files or model_store.PATTERNS["diarizzazione"]:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return snapshot


def _siblings(payload: bytes, *, digest: str | None = None):
    sha = digest or hashlib.sha256(payload).hexdigest()
    return [SimpleNamespace(rfilename=name, size=len(payload), lfs={"sha256": sha})
            for name in model_store.PATTERNS["diarizzazione"]]


class StoreTests(unittest.TestCase):
    def test_memory_store_round_trip_and_info_never_contains_the_secret(self):
        store = MemoryCredentialStore()
        with patch.dict(os.environ, CLEAN_ENV), patch("src.paths.hf_token_candidates", return_value=[]):
            info = hf_access.save_token(f"  {SECRET}\n", store)
            self.assertEqual(store.read(), SECRET)
            self.assertIs(info.source, TokenSource.STORE)
            for text in (repr(info), info.describe(), str(info.__dict__)):
                self.assertNotIn(SECRET, text)
            self.assertTrue(hf_access.remove_token(store))
            self.assertFalse(hf_access.remove_token(store))
            self.assertIs(hf_access.token_info(store).source, TokenSource.NONE)

    def test_invalid_formats_are_rejected_before_storage(self):
        store = MemoryCredentialStore()
        for bad in ("", "   ", "hf_ab cd123", "hf_abc\tdefg", "hf_abc", "hf_èèèèèèèè", "x" * 3000):
            with self.subTest(bad=bad[:12]), self.assertRaises(ValueError):
                hf_access.save_token(bad, store)
        self.assertIsNone(store.read())

    def test_unavailable_store_never_falls_back_to_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / ".hf_token"
            with patch("src.paths.hf_token_candidates", return_value=[legacy]), \
                    self.assertRaises(CredentialStoreUnavailable):
                hf_access.save_token(SECRET, secure_store.UnavailableCredentialStore())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_precedence_env_then_store_then_legacy_file(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / ".hf_token"
            legacy.write_text("hf_" + "Lg5" * 6, encoding="utf-8")
            store = MemoryCredentialStore()
            with patch("src.paths.hf_token_candidates", return_value=[legacy]):
                with patch.dict(os.environ, CLEAN_ENV):
                    token, info = hf_access.resolve_token(store)
                    self.assertIs(info.source, TokenSource.LEGACY_FILE)
                    self.assertEqual(info.legacy_file, str(legacy))
                    store.write(SECRET)
                    token, info = hf_access.resolve_token(store)
                    self.assertEqual((token, info.source), (SECRET, TokenSource.STORE))
                with patch.dict(os.environ, {**CLEAN_ENV, "HF_TOKEN": OTHER_SECRET}):
                    token, info = hf_access.resolve_token(store)
                    self.assertEqual((token, info.source, info.detail), (OTHER_SECRET, TokenSource.ENV, "HF_TOKEN"))
                    self.assertTrue(info.store_has_token)
                    self.assertIn("precedenza", info.describe())

    def test_legacy_file_is_moved_into_the_vault_and_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / ".hf_token"
            legacy.write_text(SECRET + "\n", encoding="utf-8")
            store = MemoryCredentialStore()
            with patch("src.paths.hf_token_candidates", return_value=[legacy]), patch.dict(os.environ, CLEAN_ENV):
                self.assertEqual(hf_access.import_legacy_token(store), legacy)
                self.assertFalse(legacy.exists())
                self.assertEqual(store.read(), SECRET)
                self.assertIs(hf_access.token_info(store).source, TokenSource.STORE)

    def test_first_run_token_goes_to_the_vault_not_to_settings_or_files(self):
        from src.processing.first_run import configure_user_directories
        store = MemoryCredentialStore()
        secure_store.set_default_store(store)
        self.addCleanup(secure_store.set_default_store, None)
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.paths.config_dir", return_value=Path(folder) / "config"):
            settings_file = Path(folder) / "config" / "settings.json"
            configure_user_directories(hf_token=SECRET, mark_completed=False, config_file=settings_file)
            self.assertEqual(store.read(), SECRET)
            written = [p for p in Path(folder).rglob("*") if p.is_file()]
            self.assertTrue(written)
            for path in written:
                self.assertNotIn(SECRET, path.read_text(encoding="utf-8", errors="ignore"))
            self.assertFalse(any(p.name == ".hf_token" for p in written))

    def test_first_run_rejects_a_bad_token_before_changing_settings(self):
        from src.processing.first_run import configure_user_directories
        secure_store.set_default_store(MemoryCredentialStore())
        self.addCleanup(secure_store.set_default_store, None)
        with tempfile.TemporaryDirectory() as folder:
            settings_file = Path(folder) / "settings.json"
            with self.assertRaises(ValueError):
                configure_user_directories(hf_token="hf_bad value", config_file=settings_file)
            self.assertFalse(settings_file.exists())

    def test_redact_hides_token_shapes_and_explicit_secrets(self):
        text = f"errore con {SECRET} e chiave custom-secret-value"
        clean = hf_access.redact(text, "custom-secret-value")
        self.assertNotIn(SECRET, clean)
        self.assertNotIn("custom-secret-value", clean)
        self.assertIn("hf_***", clean)


@unittest.skipUnless(sys.platform.startswith("win"), "Windows Credential Manager")
class WindowsCredentialManagerTests(unittest.TestCase):
    def test_round_trip_in_the_real_vault_with_a_throwaway_target(self):
        store = secure_store.WindowsCredentialStore(f"Tessitura-test/{uuid.uuid4().hex}")
        self.addCleanup(lambda: store.delete())
        self.assertTrue(store.available())
        self.assertIsNone(store.read())
        store.write(SECRET)
        self.assertEqual(store.read(), SECRET)
        store.write(OTHER_SECRET)
        self.assertEqual(store.read(), OTHER_SECRET)
        self.assertTrue(store.delete())
        self.assertFalse(store.delete())
        self.assertIsNone(store.read())

    def test_oversized_secret_is_refused_without_echoing_it(self):
        store = secure_store.WindowsCredentialStore(f"Tessitura-test/{uuid.uuid4().hex}")
        self.addCleanup(lambda: store.delete())
        with self.assertRaises(secure_store.CredentialStoreError) as caught:
            store.write("x" * (secure_store.MAX_SECRET_BYTES + 1))
        self.assertNotIn("xxxx", str(caught.exception))


class ClassificationTests(unittest.TestCase):
    def _kind(self, exc, *, token=True, stage="access"):
        return hf_access.classify(exc, token_present=token, stage=stage).kind

    def test_each_failure_has_its_own_kind_and_message(self):
        gated = lambda status, msg="Access to model is restricted.": hub_errors.GatedRepoError(
            "x", response=_response(status), server_message=msg)
        cases = {
            AccessIssue.TOKEN_INVALID: self._kind(hub_errors.HfHubHTTPError("x", response=_response(401))),
            AccessIssue.TOKEN_MISSING: self._kind(gated(401), token=False),
            AccessIssue.TERMS_NOT_ACCEPTED: self._kind(gated(403)),
            AccessIssue.ACCESS_DENIED: self._kind(gated(403, "Your fine-grained token does not have permission")),
            AccessIssue.NETWORK: self._kind(httpx.ConnectError("offline")),
            AccessIssue.DOWNLOAD_INCOMPLETE: self._kind(ValueError("Snapshot incompleto: file vuoto"), stage="download"),
            AccessIssue.CACHE_CORRUPT: self._kind(ValueError("Checksum errato: plda/plda.npz"), stage="download"),
            AccessIssue.MODEL_INCOMPATIBLE: self._kind(hf_access._IncompatibleModel("TypeError")),
        }
        for expected, actual in cases.items():
            self.assertIs(actual, expected)
        self.assertIs(self._kind(gated(401)), AccessIssue.TOKEN_INVALID)
        self.assertIs(self._kind(hub_errors.RepositoryNotFoundError("x", response=_response(404))),
                      AccessIssue.ACCESS_DENIED)
        self.assertIs(self._kind(hub_errors.HfHubHTTPError("x", response=_response(503))), AccessIssue.NETWORK)
        self.assertIs(self._kind(httpx.ReadTimeout("slow"), stage="download"), AccessIssue.NETWORK)
        messages = {kind.message for kind in AccessIssue}
        self.assertEqual(len(messages), len(AccessIssue))
        self.assertFalse(any(GENERIC_OLD_MESSAGE in m for m in messages))

    def test_unrelated_errors_are_not_disguised(self):
        self.assertIsNone(hf_access.classify(KeyError("x"), token_present=True))
        self.assertIsNone(hf_access.classify(RuntimeError("Modello in uso"), token_present=True))

    def test_error_text_never_contains_the_token(self):
        error = ModelAccessError(AccessIssue.NETWORK, f"proxy rejected {SECRET}")
        for text in (str(error), error.user_message, str(error.to_dict())):
            self.assertNotIn(SECRET, text)


class VerificationTests(unittest.TestCase):
    def test_states_for_the_settings_screen(self):
        store = MemoryCredentialStore()
        with patch.dict(os.environ, CLEAN_ENV), patch("src.paths.hf_token_candidates", return_value=[]):
            self.assertEqual(hf_access.verify_configuration(store=store, api=FakeApi())["state"], "not_configured")
            store.write(SECRET)
            ok = hf_access.verify_configuration(store=store, api=FakeApi())
            self.assertEqual((ok["state"], ok["model_access"], ok["account"]), ("valid", True, "tester"))
            terms = hf_access.verify_configuration(store=store, api=FakeApi(
                auth=hub_errors.GatedRepoError("x", response=_response(403), server_message="restricted")))
            self.assertEqual((terms["state"], terms["model_access"], terms["kind"]),
                             ("valid", False, AccessIssue.TERMS_NOT_ACCEPTED.value))
            self.assertEqual(terms["url"], hf_access.model_url())
            invalid = hf_access.verify_configuration(store=store, api=FakeApi(
                whoami=hub_errors.HfHubHTTPError("x", response=_response(401))))
            self.assertEqual((invalid["state"], invalid["kind"]), ("invalid", AccessIssue.TOKEN_INVALID.value))
            offline = hf_access.verify_configuration(store=store, api=FakeApi(whoami=httpx.ConnectError("x")))
            self.assertEqual((offline["state"], offline["kind"]), ("unverified", AccessIssue.NETWORK.value))
            for result in (ok, terms, invalid, offline):
                self.assertNotIn(SECRET, str(result))

    def test_typed_token_is_verified_without_saving_it(self):
        store = MemoryCredentialStore()
        api = FakeApi()
        result = hf_access.verify_configuration(OTHER_SECRET, store=store, api=api)
        self.assertTrue(result["typed"])
        self.assertEqual(api.calls[0], ("whoami", OTHER_SECRET))
        self.assertIsNone(store.read())


class EnsureModelTests(unittest.TestCase):
    def setUp(self):
        self.hub = TempHub().__enter__()
        self.addCleanup(self.hub.__exit__, None, None, None)
        self.store = MemoryCredentialStore(SECRET)
        env = patch.dict(os.environ, CLEAN_ENV)
        env.start()
        self.addCleanup(env.stop)
        legacy = patch("src.paths.hf_token_candidates", return_value=[])
        legacy.start()
        self.addCleanup(legacy.stop)
        self.steps: list[tuple[str, str]] = []

    def _on_step(self, name, state, detail):
        self.assertNotIn(SECRET, detail)
        self.steps.append((name, state))

    def _hub_download(self, payload=b"weights", *, digest=None, fail_first=None):
        """Patch model_store's own HfApi/snapshot_download to write into the temp hub."""
        sha = "b" * 40
        attempts = {"n": 0}

        class Api:
            def model_info(self, repo, revision=None, token=None, timeout=None, files_metadata=False):
                return SimpleNamespace(sha=sha, siblings=_siblings(payload, digest=digest))

        def download(repo, **kwargs):
            attempts["n"] += 1
            snapshot = self.hub.write_snapshot(sha, payload, files=model_store.PATTERNS["diarizzazione"][:2])
            if fail_first is not None and attempts["n"] == 1:
                raise fail_first          # partial files stay for the next attempt
            self.hub.write_snapshot(sha, payload)
            assert kwargs["token"] == SECRET and kwargs["revision"] == sha
            return str(snapshot)

        return patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.snapshot_download", download), attempts

    def test_missing_token_stops_before_any_network_call(self):
        api = FakeApi()
        with self.assertRaises(ModelAccessError) as caught:
            hf_access.ensure_model(store=MemoryCredentialStore(), api=api, on_step=self._on_step)
        self.assertIs(caught.exception.kind, AccessIssue.TOKEN_MISSING)
        self.assertEqual(api.calls, [])
        self.assertIn(("token", "failed"), self.steps)

    def test_terms_not_accepted_never_downloads(self):
        api = FakeApi(auth=hub_errors.GatedRepoError("x", response=_response(403), server_message="restricted"))
        with patch.object(model_store, "ensure") as ensure, self.assertRaises(ModelAccessError) as caught:
            hf_access.ensure_model(store=self.store, api=api, on_step=self._on_step)
        self.assertIs(caught.exception.kind, AccessIssue.TERMS_NOT_ACCEPTED)
        ensure.assert_not_called()
        self.assertIn(("terms", "failed"), self.steps)
        self.assertIn(("access", "done"), self.steps)

    def test_verified_download_is_activated_and_found_by_the_diarizer(self):
        """Regression: a commit-pinned download has no refs/main and stayed invisible."""
        api_patch, download_patch, _ = self._hub_download()
        with api_patch, download_patch:
            path = hf_access.ensure_model(store=self.store, api=FakeApi(), on_step=self._on_step,
                                          validate_load=lambda _p: None)
        self.assertEqual(Path(path).name, "b" * 40)
        self.assertFalse((model_store.repo_root(REPO) / "refs" / "main").exists())
        self.assertEqual(hf_access.local_model(), path)
        from src.processing.diarize import _model_path
        self.assertEqual(_model_path(REPO, None, allow_download=False), path)
        self.assertEqual([s for s in self.steps if s[1] == "failed"], [("cache", "failed")])
        self.assertIn(("integrity", "done"), self.steps)

    def test_unactivated_store_download_alone_is_not_considered_available(self):
        """Documents the defect fixed by routing downloads through ensure_model."""
        api_patch, download_patch, _ = self._hub_download()
        with api_patch, download_patch:
            model_store.ensure(REPO, "diarizzazione", token=SECRET)
        self.assertIsNone(hf_access.local_model())
        self.assertEqual(hf_access.local_cache_state(), "incomplete")

    def test_interrupted_download_resumes_on_retry(self):
        api_patch, download_patch, attempts = self._hub_download(fail_first=httpx.ReadTimeout("slow"))
        with api_patch, download_patch:
            with self.assertRaises(ModelAccessError) as caught:
                hf_access.ensure_model(store=self.store, api=FakeApi(), validate_load=lambda _p: None)
            self.assertIs(caught.exception.kind, AccessIssue.NETWORK)
            self.assertIsNone(hf_access.local_model())
            self.assertEqual(hf_access.local_cache_state(), "incomplete")
            path = hf_access.ensure_model(store=self.store, api=FakeApi(), validate_load=lambda _p: None)
        self.assertEqual(attempts["n"], 2)
        self.assertEqual(hf_access.local_model(), path)

    def test_checksum_mismatch_is_a_corrupt_cache_and_is_not_activated(self):
        api_patch, download_patch, _ = self._hub_download(digest="0" * 64)
        with api_patch, download_patch, self.assertRaises(ModelAccessError) as caught:
            hf_access.ensure_model(store=self.store, api=FakeApi(), on_step=self._on_step,
                                   validate_load=lambda _p: None)
        self.assertIs(caught.exception.kind, AccessIssue.CACHE_CORRUPT)
        self.assertIn(("integrity", "failed"), self.steps)
        self.assertIsNone(hf_access.local_model())

    def test_bundle_that_does_not_load_is_incompatible_and_not_activated(self):
        def refuse(_path):
            raise hf_access._IncompatibleModel("pipeline non caricabile")
        api_patch, download_patch, _ = self._hub_download()
        with api_patch, download_patch, self.assertRaises(ModelAccessError) as caught:
            hf_access.ensure_model(store=self.store, api=FakeApi(), validate_load=refuse)
        self.assertIs(caught.exception.kind, AccessIssue.MODEL_INCOMPATIBLE)
        self.assertIsNone(hf_access.local_model())

    def test_repair_deletes_the_damaged_cache_and_downloads_again(self):
        self.hub.write_snapshot("c" * 40, b"", files=model_store.PATTERNS["diarizzazione"][:1])
        self.assertEqual(hf_access.local_cache_state(), "incomplete")
        api_patch, download_patch, _ = self._hub_download()
        with api_patch, download_patch:
            path = hf_access.ensure_model(store=self.store, api=FakeApi(), repair=True,
                                          validate_load=lambda _p: None)
        self.assertFalse((model_store.repo_root(REPO) / "snapshots" / ("c" * 40)).exists())
        self.assertEqual(hf_access.local_model(), path)

    def test_cached_model_needs_no_token_and_no_network(self):
        api_patch, download_patch, _ = self._hub_download()
        with api_patch, download_patch:
            path = hf_access.ensure_model(store=self.store, api=FakeApi(), validate_load=lambda _p: None)
        api = FakeApi()
        self.assertEqual(hf_access.ensure_model(store=MemoryCredentialStore(), api=api), path)
        self.assertEqual(api.calls, [])


class LocalOnlyPathTests(unittest.TestCase):
    def test_global_review_reports_a_missing_model_not_a_missing_token(self):
        from src.processing.diarize import _model_path
        with TempHub(), patch("huggingface_hub.snapshot_download",
                              side_effect=hub_errors.LocalEntryNotFoundError("no cache")):
            with self.assertRaises(ModelAccessError) as caught:
                _model_path(REPO, None, allow_download=False)
        self.assertIs(caught.exception.kind, AccessIssue.MODEL_NOT_CACHED)
        self.assertNotIn(GENERIC_OLD_MESSAGE, str(caught.exception))

    def test_embedding_extraction_raises_the_classified_error(self):
        from src.processing.global_workflow import extract_clean_embeddings
        with TempHub(), tempfile.TemporaryDirectory() as folder, \
                patch("huggingface_hub.snapshot_download", side_effect=hub_errors.LocalEntryNotFoundError("x")):
            with self.assertRaises(ModelAccessError) as caught:
                extract_clean_embeddings(Path(folder) / "a.wav", [], Path(folder) / "out.json", device="cpu")
        self.assertIs(caught.exception.kind, AccessIssue.MODEL_NOT_CACHED)

    def test_diarizer_reports_an_unloadable_local_bundle_as_incompatible(self):
        from pyannote.audio import Pipeline
        from src.processing.diarize import Diarizer
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.processing.diarize._model_path", return_value=folder), \
                patch.object(Pipeline, "from_pretrained", return_value=None) as load:
            with self.assertRaises(ModelAccessError) as caught:
                Diarizer(device="cpu")
        self.assertIs(caught.exception.kind, AccessIssue.MODEL_INCOMPATIBLE)
        self.assertNotIn("token", load.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
