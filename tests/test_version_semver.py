"""Unit tests for SemVer 2.0.0 parsing, comparison, prerelease precedence, and BuildConfig."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.version import (
    APP_NAME,
    SemVer,
    compare_versions,
    installer_asset_name,
    is_newer_version,
    load_build_config,
    parse_semver,
)


class SemVerTests(unittest.TestCase):
    def test_parse_valid_versions_and_tags(self) -> None:
        v1 = parse_semver("1.0.0")
        self.assertEqual((v1.major, v1.minor, v1.patch), (1, 0, 0))
        self.assertFalse(v1.is_prerelease)
        self.assertTrue(v1.is_stable)
        self.assertEqual(v1.tag_name, "v1.0.0")

        v2 = parse_semver("v1.2.3-beta.1+20261111.sha123")
        self.assertEqual((v2.major, v2.minor, v2.patch), (1, 2, 3))
        self.assertEqual(v2.prerelease, ("beta", "1"))
        self.assertEqual(v2.build, ("20261111", "sha123"))
        self.assertTrue(v2.is_prerelease)
        self.assertFalse(v2.is_stable)
        self.assertEqual(v2.version_without_build, "1.2.3-beta.1")
        self.assertEqual(v2.tag_name, "v1.2.3-beta.1")

    def test_prerelease_and_release_precedence_order(self) -> None:
        ordered_strings = [
            "0.9.0",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.10",
            "1.0.0-rc.1",
            "1.0.0",
            "1.0.1",
            "1.1.0-beta.1",
            "1.1.0-beta.2",
            "1.1.0",
            "2.0.0",
        ]
        parsed = [SemVer.parse(s) for s in ordered_strings]
        for i in range(len(parsed) - 1):
            left = parsed[i]
            right = parsed[i + 1]
            self.assertLess(left, right, f"Expected {left} < {right}")
            self.assertGreater(right, left, f"Expected {right} > {left}")
            self.assertEqual(compare_versions(left, right), -1)
            self.assertEqual(compare_versions(right, left), 1)

    def test_build_metadata_does_not_affect_precedence(self) -> None:
        a = SemVer.parse("1.0.0-beta.1+build.1")
        b = SemVer.parse("1.0.0-beta.1+build.99")
        self.assertEqual(a, b)
        self.assertEqual(compare_versions(a, b), 0)

    def test_invalid_semver_strings_raise_value_error(self) -> None:
        for bad in ("", "1.0", "1.0.0.0", "v01.0.0", "1.0.0-01", "1.0.0-", "1.0.0+", "not-a-version"):
            with self.assertRaises(ValueError, msg=f"Should reject {bad!r}"):
                SemVer.parse(bad)

    def test_is_newer_version_respects_channel_prerelease_flag(self) -> None:
        self.assertFalse(is_newer_version("1.1.0-beta.1", "1.0.0", allow_prerelease=False))
        self.assertTrue(is_newer_version("1.1.0-beta.1", "1.0.0", allow_prerelease=True))
        self.assertTrue(is_newer_version("1.1.0", "1.1.0-rc.1", allow_prerelease=False))
        self.assertFalse(is_newer_version("1.0.0", "1.0.0", allow_prerelease=True))

    def test_installer_asset_name(self) -> None:
        self.assertEqual(installer_asset_name("v1.2.0"), f"{APP_NAME}-Setup-1.2.0.exe")
        self.assertEqual(
            installer_asset_name("1.2.0-beta.2+sha.abc"),
            f"{APP_NAME}-Setup-1.2.0-beta.2.exe",
        )

    def test_load_build_config_from_file_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "build_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "version": "1.3.0-beta.1",
                        "github_owner": "acme",
                        "github_repo": "tessitura-app",
                        "default_channel": "beta",
                        "trust_policy": "checksum_only",
                        "trusted_publishers": ["CN=Tessitura Official"],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TESSITURA_VERSION", None)
                os.environ.pop("TESSITURA_GITHUB_OWNER", None)
                os.environ.pop("TESSITURA_GITHUB_REPO", None)
                os.environ.pop("TESSITURA_UPDATE_CHANNEL", None)
                os.environ.pop("TESSITURA_TRUST_POLICY", None)
                cfg = load_build_config(cfg_path)
                self.assertEqual(cfg.version, "1.3.0-beta.1")
                self.assertEqual(cfg.github_owner, "acme")
                self.assertEqual(cfg.github_repo, "tessitura-app")
                self.assertEqual(cfg.default_channel, "beta")
                self.assertEqual(cfg.trust_policy, "checksum_only")
                self.assertEqual(cfg.trusted_publishers, ("CN=Tessitura Official",))


if __name__ == "__main__":
    unittest.main()

