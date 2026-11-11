"""Public application metadata, build-time configuration, and SemVer 2.0.0 utilities for Tessitura."""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

APP_NAME = "Tessitura"
APP_SLUG = "tessitura"
APP_EXECUTABLE = "Tessitura.exe"
APP_ID = "com.tessitura.desktop"
INNO_APP_ID = "{{9B3F4C81-2A7E-4F1D-8E55-6D91A2B4C8E0}"
APP_PUBLISHER = "Tessitura Project"
APP_DESCRIPTION = "Registrazione, trascrizione e diarizzazione locale per Windows"

APP_VERSION = "1.0.0"

CONFIG_SCHEMA_VERSION = 1
SESSION_SCHEMA_VERSION = 6
PROJECT_SCHEMA_VERSION = 1

DEFAULT_GITHUB_OWNER = "aless"
DEFAULT_GITHUB_REPO = "tessitura"
DEFAULT_UPDATE_CHANNEL = "stable"
DEFAULT_TRUST_POLICY = "strict"  # "strict" | "checksum_only"
DEFAULT_AUTO_CHECK_INTERVAL_SECONDS = 86400  # 24 hours

_SEMVER_RE = re.compile(
    r"^[vV]?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@total_ordering
@dataclass(frozen=True)
class SemVer:
    """Semantic Versioning 2.0.0 parser and comparator."""

    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "SemVer":
        if not isinstance(value, str):
            raise ValueError(f"Version must be a string, got {type(value).__name__}")
        cleaned = value.strip()
        match = _SEMVER_RE.match(cleaned)
        if not match:
            raise ValueError(f"Invalid SemVer version string: {value!r}")

        major = int(match.group("major"))
        minor = int(match.group("minor"))
        patch = int(match.group("patch"))
        pre_raw = match.group("prerelease")
        build_raw = match.group("build")

        prerelease: tuple[str, ...] = ()
        if pre_raw:
            parts = tuple(pre_raw.split("."))
            for part in parts:
                if not part:
                    raise ValueError(f"Empty prerelease identifier in {value!r}")
                if part.isdigit() and len(part) > 1 and part.startswith("0"):
                    raise ValueError(
                        f"Numeric prerelease identifier must not have leading zeroes: {value!r}"
                    )
            prerelease = parts

        build: tuple[str, ...] = ()
        if build_raw:
            parts = tuple(build_raw.split("."))
            if any(not p for p in parts):
                raise ValueError(f"Empty build metadata identifier in {value!r}")
            build = parts

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    @property
    def is_prerelease(self) -> bool:
        return bool(self.prerelease)

    @property
    def is_stable(self) -> bool:
        return not self.prerelease and self.major >= 1

    @property
    def base_version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def normalized(self) -> str:
        version = self.base_version
        if self.prerelease:
            version += "-" + ".".join(self.prerelease)
        if self.build:
            version += "+" + ".".join(self.build)
        return version

    @property
    def version_without_build(self) -> str:
        version = self.base_version
        if self.prerelease:
            version += "-" + ".".join(self.prerelease)
        return version

    @property
    def tag_name(self) -> str:
        return f"v{self.version_without_build}"

    def __str__(self) -> str:
        return self.normalized

    def _precedence_key(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        # Per SemVer 2.0.0 section 10: build metadata does not figure into precedence.
        return (
            self._precedence_key() == other._precedence_key()
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        if self._precedence_key() != other._precedence_key():
            return self._precedence_key() < other._precedence_key()

        # Normal version has higher precedence than a pre-release version (1.0.0 > 1.0.0-alpha)
        if not self.prerelease and not other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False

        # Compare each dot-separated identifier from left to right
        for left_id, right_id in zip(self.prerelease, other.prerelease):
            if left_id == right_id:
                continue
            left_is_num = left_id.isdigit()
            right_is_num = right_id.isdigit()
            if left_is_num and right_is_num:
                return int(left_id) < int(right_id)
            if left_is_num and not right_is_num:
                # Numeric identifiers always have lower precedence than non-numeric identifiers
                return True
            if not left_is_num and right_is_num:
                return False
            return left_id < right_id

        # A larger set of pre-release fields has higher precedence than a smaller set
        return len(self.prerelease) < len(other.prerelease)


def parse_semver(value: str) -> SemVer:
    """Parse a SemVer 2.0.0 version or tag string."""
    return SemVer.parse(value)


def compare_versions(left: str | SemVer, right: str | SemVer) -> int:
    """Compare two SemVer versions; returns -1 if left < right, 0 if equal, 1 if left > right."""
    v_left = left if isinstance(left, SemVer) else SemVer.parse(left)
    v_right = right if isinstance(right, SemVer) else SemVer.parse(right)
    if v_left < v_right:
        return -1
    if v_left > v_right:
        return 1
    return 0


def is_newer_version(
    candidate: str | SemVer,
    current: str | SemVer,
    *,
    allow_prerelease: bool = False,
) -> bool:
    """Return True if candidate has strictly higher SemVer precedence than current."""
    v_cand = candidate if isinstance(candidate, SemVer) else SemVer.parse(candidate)
    v_curr = current if isinstance(current, SemVer) else SemVer.parse(current)
    if not allow_prerelease and v_cand.is_prerelease:
        return False
    return v_cand > v_curr


def installer_asset_name(version: str | SemVer) -> str:
    """Return the canonical Windows installer filename for a given version."""
    sem = version if isinstance(version, SemVer) else SemVer.parse(version)
    return f"{APP_NAME}-Setup-{sem.version_without_build}.exe"


@dataclass(frozen=True)
class BuildConfig:
    """Build-time and environment configuration for Tessitura."""

    app_name: str = APP_NAME
    app_id: str = APP_ID
    version: str = APP_VERSION
    github_owner: str = DEFAULT_GITHUB_OWNER
    github_repo: str = DEFAULT_GITHUB_REPO
    default_channel: str = DEFAULT_UPDATE_CHANNEL
    trust_policy: str = DEFAULT_TRUST_POLICY
    trusted_publishers: tuple[str, ...] = ()
    build_timestamp: str = ""
    commit_sha: str = ""

    @property
    def github_repo_slug(self) -> str:
        return f"{self.github_owner}/{self.github_repo}"

    @property
    def github_releases_html_url(self) -> str:
        return f"https://github.com/{self.github_owner}/{self.github_repo}/releases"

    @property
    def github_api_base_url(self) -> str:
        return os.environ.get(
            "TESSITURA_GITHUB_API_URL",
            f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}",
        )


def _locate_build_config_path() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        direct = exe_dir / "build_config.json"
        if direct.is_file():
            return direct
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir / "_internal"))
        return meipass / "build_config.json"
    return Path(__file__).resolve().parent.parent / "build_config.json"


def load_build_config(config_path: Path | None = None) -> BuildConfig:
    """Load build-time configuration from build_config.json and environment overrides."""
    path = config_path or _locate_build_config_path()
    raw: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                raw = loaded
        except (OSError, json.JSONDecodeError):
            raw = {}

    version = (
        os.environ.get("TESSITURA_VERSION")
        or str(raw.get("version") or "").strip()
        or APP_VERSION
    )
    github_owner = (
        os.environ.get("TESSITURA_GITHUB_OWNER")
        or str(raw.get("github_owner") or "").strip()
        or DEFAULT_GITHUB_OWNER
    )
    github_repo = (
        os.environ.get("TESSITURA_GITHUB_REPO")
        or str(raw.get("github_repo") or "").strip()
        or DEFAULT_GITHUB_REPO
    )
    default_channel = (
        os.environ.get("TESSITURA_UPDATE_CHANNEL")
        or str(raw.get("default_channel") or "").strip()
        or DEFAULT_UPDATE_CHANNEL
    ).lower()
    if default_channel not in {"stable", "beta"}:
        default_channel = DEFAULT_UPDATE_CHANNEL

    trust_policy = (
        os.environ.get("TESSITURA_TRUST_POLICY")
        or str(raw.get("trust_policy") or "").strip()
        or DEFAULT_TRUST_POLICY
    ).lower()
    if trust_policy not in {"strict", "checksum_only", "disabled"}:
        trust_policy = DEFAULT_TRUST_POLICY

    publishers_raw = raw.get("trusted_publishers", [])
    publishers: list[str] = []
    if isinstance(publishers_raw, list):
        publishers.extend(str(p).strip() for p in publishers_raw if str(p).strip())
    env_pub = os.environ.get("TESSITURA_TRUSTED_PUBLISHER", "").strip()
    if env_pub:
        publishers.extend(p.strip() for p in env_pub.split(";") if p.strip())

    return BuildConfig(
        app_name=APP_NAME,
        app_id=APP_ID,
        version=version,
        github_owner=github_owner,
        github_repo=github_repo,
        default_channel=default_channel,
        trust_policy=trust_policy,
        trusted_publishers=tuple(dict.fromkeys(publishers)),
        build_timestamp=str(raw.get("build_timestamp") or ""),
        commit_sha=str(raw.get("commit_sha") or ""),
    )


def get_app_version() -> str:
    """Return the currently running application SemVer string."""
    return load_build_config().version

