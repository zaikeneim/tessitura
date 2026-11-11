"""Unit and integration tests for UpdateManager, GitHub REST API client, verification, and safe update guards."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from src.config import load_settings
from src.update_manager import (
    ActiveOperationState,
    AuthenticodeInfo,
    InstallerVerificationError,
    UpdateBlockedByActiveOperationError,
    UpdateManager,
    parse_sha256sums,
)


class _FakeHTTPResponse:
    def __init__(self, body: bytes | str, headers: dict[str, str] | None = None, status: int = 200) -> None:
        self._bytes = body.encode("utf-8") if isinstance(body, str) else body
        self._stream = io.BytesIO(self._bytes)
        self.headers = headers or {}
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


def _make_release_json(
    tag: str,
    *,
    prerelease: bool = False,
    draft: bool = False,
    published_at: str = "2026-11-11T10:00:00Z",
    with_installer: bool = True,
    with_checksum: bool = True,
    installer_size: int = 128,
) -> dict:
    ver = tag.lstrip("vV")
    assets = []
    if with_installer:
        assets.append(
            {
                "name": f"Tessitura-Setup-{ver}.exe",
                "browser_download_url": f"https://example.test/releases/{tag}/Tessitura-Setup-{ver}.exe",
                "size": installer_size,
                "content_type": "application/x-msdownload",
            }
        )
    if with_checksum:
        assets.append(
            {
                "name": "SHA256SUMS.txt",
                "browser_download_url": f"https://example.test/releases/{tag}/SHA256SUMS.txt",
                "size": 96,
                "content_type": "text/plain",
            }
        )
    return {
        "tag_name": tag,
        "name": f"Tessitura {ver}",
        "body": f"Changelog for {ver}",
        "html_url": f"https://github.com/aless/tessitura/releases/tag/{tag}",
        "published_at": published_at,
        "prerelease": prerelease,
        "draft": draft,
        "assets": assets,
    }


class UpdateManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache_file = self.root / "cache" / "github_releases_cache.json"
        self.downloads_dir = self.root / "cache" / "updates"
        self.config_file = self.root / "config" / "settings.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _manager(self, *, current_version: str = "1.0.0", trust_policy: str = "strict", **kwargs) -> UpdateManager:
        return UpdateManager(
            current_version=current_version,
            api_base_url="https://api.github.com/repos/aless/tessitura",
            trust_policy=trust_policy,
            cache_file=self.cache_file,
            downloads_dir=self.downloads_dir,
            config_file=self.config_file,
            **kwargs,
        )

    def test_stable_channel_selects_latest_stable_and_ignores_prereleases(self) -> None:
        payload = _make_release_json("v1.1.0", prerelease=False)
        mgr = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=10: _FakeHTTPResponse(
                json.dumps(payload),
                headers={"ETag": '"etag-stable-1"', "X-RateLimit-Remaining": "58"},
            ),
        )
        res = mgr.check_for_updates(channel="stable", force=True)
        self.assertEqual(res.status, "update_available")
        self.assertTrue(res.update_available)
        self.assertIsNotNone(res.latest_release)
        self.assertEqual(res.latest_release.version, "1.1.0")
        self.assertEqual(res.rate_limit_remaining, 58)

    def test_stable_channel_rejects_prerelease_if_returned(self) -> None:
        payload = _make_release_json("v1.2.0-beta.1", prerelease=True)
        mgr = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=10: _FakeHTTPResponse(json.dumps(payload)),
        )
        res = mgr.check_for_updates(channel="stable", force=True)
        self.assertFalse(res.update_available)
        self.assertEqual(res.status, "no_releases")

    def test_beta_channel_sorts_by_semver_not_publication_date_and_skips_drafts(self) -> None:
        # Notice v1.1.0-beta.2 has an OLDER published_at than v1.1.0-beta.1, and v1.2.0 is a draft!
        releases = [
            _make_release_json("v1.2.0", draft=True, published_at="2026-11-11T12:00:00Z"),
            _make_release_json("v1.1.0-beta.1", prerelease=True, published_at="2026-11-11T11:00:00Z"),
            _make_release_json("v1.1.0-beta.10", prerelease=True, published_at="2026-11-10T08:00:00Z"),
            _make_release_json("v1.1.0-beta.2", prerelease=True, published_at="2026-11-10T09:00:00Z"),
            _make_release_json("v1.0.0", prerelease=False, published_at="2026-11-01T00:00:00Z"),
        ]
        mgr = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=10: _FakeHTTPResponse(json.dumps(releases)),
        )
        res = mgr.check_for_updates(channel="beta", force=True)
        self.assertTrue(res.update_available)
        self.assertEqual(res.latest_release.version, "1.1.0-beta.10")

    def test_up_to_date_when_installed_matches_or_exceeds_latest(self) -> None:
        payload = _make_release_json("v1.0.0", prerelease=False)
        mgr = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=10: _FakeHTTPResponse(json.dumps(payload)),
        )
        res = mgr.check_for_updates(channel="stable", force=True)
        self.assertEqual(res.status, "up_to_date")
        self.assertFalse(res.update_available)

    def test_http_304_not_modified_uses_cached_release(self) -> None:
        payload = _make_release_json("v1.1.0", prerelease=False)
        calls = []

        def opener(req, timeout=10):
            calls.append(dict(req.header_items()))
            if len(calls) == 1:
                return _FakeHTTPResponse(
                    json.dumps(payload),
                    headers={"ETag": '"etag-v110"'},
                )
            raise urllib.error.HTTPError(
                req.full_url, 304, "Not Modified", {"ETag": '"etag-v110"'}, None
            )

        mgr = self._manager(current_version="1.0.0", http_opener=opener)
        first = mgr.check_for_updates(channel="stable", force=True)
        self.assertFalse(first.from_cache)
        self.assertTrue(first.update_available)

        second = mgr.check_for_updates(channel="stable", force=True)
        self.assertTrue(second.from_cache)
        self.assertTrue(second.update_available)
        self.assertEqual(second.latest_release.version, "1.1.0")
        self.assertIn("If-none-match", calls[1])

    def test_rate_limit_403_and_offline_handling_graceful(self) -> None:
        def rate_limited_opener(req, timeout=10):
            raise urllib.error.HTTPError(
                req.full_url,
                403,
                "rate limit exceeded",
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1900000000"},
                None,
            )

        mgr = self._manager(current_version="1.0.0", http_opener=rate_limited_opener)
        res = mgr.check_for_updates(channel="stable", force=True)
        self.assertEqual(res.status, "rate_limited")
        self.assertEqual(res.rate_limit_remaining, 0)

        def offline_opener(req, timeout=10):
            raise urllib.error.URLError("DNS lookup failed")

        mgr_offline = self._manager(current_version="1.0.0", http_opener=offline_opener)
        res_off = mgr_offline.check_for_updates(channel="stable", force=True)
        self.assertEqual(res_off.status, "offline")
        self.assertFalse(res_off.update_available)

    def test_download_and_verify_checksum_mismatch_and_corrupted_size(self) -> None:
        installer_bytes = b"MZ-FAKE-INSTALLER-BINARY-PAYLOAD-1.1.0"
        good_sha = hashlib.sha256(installer_bytes).hexdigest()
        bad_sha = "0" * 64

        rel_json = _make_release_json("v1.1.0", installer_size=len(installer_bytes))
        mgr_bad = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=30: (
                _FakeHTTPResponse(f"{bad_sha}  Tessitura-Setup-1.1.0.exe\n")
                if req.full_url.endswith("SHA256SUMS.txt")
                else _FakeHTTPResponse(installer_bytes)
            ),
            authenticode_inspector=lambda _p: AuthenticodeInfo(status="Valid", signer_subject="CN=Tessitura"),
        )
        rel = mgr_bad._select_release_from_payload(rel_json, "stable")
        inst_path, expected_sha = mgr_bad.download_release_installer(rel)
        with self.assertRaises(InstallerVerificationError):
            mgr_bad.verify_installer_package(
                inst_path,
                target_version="1.1.0",
                expected_sha256=expected_sha,
                expected_size=len(installer_bytes),
            )
        # Tampered file must be deleted on checksum mismatch
        self.assertFalse(inst_path.exists())

        # Corrupted / truncated download test
        mgr_trunc = self._manager(
            current_version="1.0.0",
            http_opener=lambda req, timeout=30: (
                _FakeHTTPResponse(f"{good_sha}  Tessitura-Setup-1.1.0.exe\n")
                if req.full_url.endswith("SHA256SUMS.txt")
                else _FakeHTTPResponse(installer_bytes[:10])
            ),
        )
        with self.assertRaises(InstallerVerificationError):
            mgr_trunc.download_release_installer(rel)

    def test_downgrade_is_blocked(self) -> None:
        installer_bytes = b"MZ-OLDER-VERSION"
        sha = hashlib.sha256(installer_bytes).hexdigest()
        f = self.downloads_dir / "Tessitura-Setup-0.9.0.exe"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(installer_bytes)

        mgr = self._manager(current_version="1.0.0")
        with self.assertRaises(InstallerVerificationError):
            mgr.verify_installer_package(
                f,
                target_version="0.9.0",
                expected_sha256=sha,
                expected_size=len(installer_bytes),
            )

    def test_authenticode_trust_policies_strict_vs_checksum_only(self) -> None:
        installer_bytes = b"MZ-UNSIGNED-INSTALLER-1.1.0"
        sha = hashlib.sha256(installer_bytes).hexdigest()
        f = self.downloads_dir / "Tessitura-Setup-1.1.0.exe"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(installer_bytes)

        # 1. Under strict policy, unsigned installer cannot auto-execute
        mgr_strict = self._manager(
            current_version="1.0.0",
            trust_policy="strict",
            authenticode_inspector=lambda _p: AuthenticodeInfo(status="NotSigned"),
        )
        ver_strict = mgr_strict.verify_installer_package(
            f,
            target_version="1.1.0",
            expected_sha256=sha,
            expected_size=len(installer_bytes),
        )
        self.assertFalse(ver_strict.verified)
        self.assertFalse(ver_strict.can_auto_execute)
        with self.assertRaises(InstallerVerificationError):
            mgr_strict.prepare_and_launch_installer(ver_strict)

        # 2. Under strict policy with Valid signature, auto-execution succeeds
        mgr_signed = self._manager(
            current_version="1.0.0",
            trust_policy="strict",
            trusted_publishers=("CN=Tessitura Project",),
            authenticode_inspector=lambda _p: AuthenticodeInfo(
                status="Valid",
                signer_subject="CN=Tessitura Project, O=Tessitura",
            ),
        )
        ver_signed = mgr_signed.verify_installer_package(
            f,
            target_version="1.1.0",
            expected_sha256=sha,
            expected_size=len(installer_bytes),
        )
        self.assertTrue(ver_signed.verified)
        self.assertTrue(ver_signed.can_auto_execute)
        self.assertFalse(ver_signed.requires_user_warning)

        # 3. Under checksum_only policy, unsigned installer with valid SHA256 is allowed with user warning
        mgr_chk = self._manager(
            current_version="1.0.0",
            trust_policy="checksum_only",
            authenticode_inspector=lambda _p: AuthenticodeInfo(status="NotSigned"),
        )
        ver_chk = mgr_chk.verify_installer_package(
            f,
            target_version="1.1.0",
            expected_sha256=sha,
            expected_size=len(installer_bytes),
        )
        self.assertTrue(ver_chk.verified)
        self.assertTrue(ver_chk.can_auto_execute)
        self.assertTrue(ver_chk.requires_user_warning)

    def test_active_operation_blocks_update_and_idle_allows_assisted_launch(self) -> None:
        installer_bytes = b"MZ-SIGNED-INSTALLER-1.1.0"
        sha = hashlib.sha256(installer_bytes).hexdigest()
        f = self.downloads_dir / "Tessitura-Setup-1.1.0.exe"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(installer_bytes)

        launched_cmds: list[list[str]] = []
        quit_called: list[bool] = []
        busy_state = {"recording": True, "processing": False}

        mgr = self._manager(
            current_version="1.0.0",
            trust_policy="strict",
            authenticode_inspector=lambda _p: AuthenticodeInfo(
                status="Valid",
                signer_subject="CN=Tessitura Project",
            ),
            activity_provider=lambda: ActiveOperationState(
                is_recording=busy_state["recording"],
                is_processing=busy_state["processing"],
            ),
            process_launcher=lambda cmd: launched_cmds.append(cmd),
            quit_callback=lambda: quit_called.append(True),
        )
        ver = mgr.verify_installer_package(
            f,
            target_version="1.1.0",
            expected_sha256=sha,
            expected_size=len(installer_bytes),
        )

        # Blocked during recording
        with self.assertRaises(UpdateBlockedByActiveOperationError):
            mgr.prepare_and_launch_installer(ver)
        self.assertEqual(launched_cmds, [])
        self.assertEqual(quit_called, [])

        # Blocked during transcription/diarization
        busy_state["recording"] = False
        busy_state["processing"] = True
        with self.assertRaises(UpdateBlockedByActiveOperationError):
            mgr.prepare_and_launch_installer(ver)
        self.assertEqual(launched_cmds, [])

        # Allowed once idle
        busy_state["processing"] = False
        backup_dir = mgr.prepare_and_launch_installer(ver, channel="stable")
        self.assertTrue(backup_dir.is_dir())
        self.assertEqual(len(launched_cmds), 1)
        self.assertEqual(Path(launched_cmds[0][0]), f.resolve())
        self.assertEqual(quit_called, [True])

        settings = load_settings(self.config_file)
        self.assertIsNotNone(settings.pending_update)
        self.assertEqual(settings.pending_update["target_version"], "1.1.0")
        self.assertEqual(settings.pending_update["previous_version"], "1.0.0")

    def test_parse_sha256sums_handles_standard_and_binary_star_formats(self) -> None:
        h1 = "a" * 64
        h2 = "b" * 64
        txt = f"# comment\n{h1}  Tessitura-Setup-1.0.0.exe\n{h2} *Tessitura-1.0.0-portable.zip\n"
        parsed = parse_sha256sums(txt)
        self.assertEqual(parsed["Tessitura-Setup-1.0.0.exe"], h1)
        self.assertEqual(parsed["Tessitura-1.0.0-portable.zip"], h2)


if __name__ == "__main__":
    unittest.main()

