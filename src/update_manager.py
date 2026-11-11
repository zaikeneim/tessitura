"""GitHub Releases update checker, package verifier, and safe assisted update coordinator for Tessitura."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hmac
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from src import paths
from src.config import load_settings, save_settings
from src.migrations import create_state_backup
from src.version import (
    DEFAULT_AUTO_CHECK_INTERVAL_SECONDS,
    SemVer,
     installer_asset_name,
    load_build_config,
)

_SAFE_INSTALLER_NAME_RE = re.compile(r"^Tessitura-Setup-[0-9A-Za-z._-]+\.exe$")
_SHA256_LINE_RE = re.compile(r"^(?P<hash>[0-9a-fA-F]{64})\s+\*?(?P<name>.+?)\s*$")


class UpdateError(RuntimeError):
    """Base error for update operations."""


class UpdateBlockedByActiveOperationError(UpdateError):
    """Raised when an update installation is requested while recording or processing is active."""


class InstallerVerificationError(UpdateError):
    """Raised when a downloaded installer fails size, SHA-256, version, or Authenticode trust checks."""


@dataclass(frozen=True)
class ReleaseAssetInfo:
    """Metadata for a single GitHub Release asset."""

    name: str
    download_url: str
    size: int
    content_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseInfo:
    """Normalized metadata for a GitHub Release."""

    tag_name: str
    version: str
    name: str
    body: str
    html_url: str
    published_at: str
    prerelease: bool
    draft: bool
    installer_asset: ReleaseAssetInfo | None = None
    checksum_asset: ReleaseAssetInfo | None = None

    @property
    def semver(self) -> SemVer:
        return SemVer.parse(self.version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "version": self.version,
            "name": self.name,
            "body": self.body,
            "html_url": self.html_url,
            "published_at": self.published_at,
            "prerelease": self.prerelease,
            "draft": self.draft,
            "installer_asset": self.installer_asset.to_dict() if self.installer_asset else None,
            "checksum_asset": self.checksum_asset.to_dict() if self.checksum_asset else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReleaseInfo":
        inst_raw = raw.get("installer_asset")
        chk_raw = raw.get("checksum_asset")
        return cls(
            tag_name=str(raw.get("tag_name") or ""),
            version=str(raw.get("version") or ""),
            name=str(raw.get("name") or ""),
            body=str(raw.get("body") or ""),
            html_url=str(raw.get("html_url") or ""),
            published_at=str(raw.get("published_at") or ""),
            prerelease=bool(raw.get("prerelease", False)),
            draft=bool(raw.get("draft", False)),
            installer_asset=ReleaseAssetInfo(**inst_raw) if isinstance(inst_raw, dict) else None,
            checksum_asset=ReleaseAssetInfo(**chk_raw) if isinstance(chk_raw, dict) else None,
        )


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of checking GitHub Releases for an update."""

    status: str  # "update_available" | "up_to_date" | "offline" | "rate_limited" | "no_releases" | "error"
    channel: str
    current_version: str
    checked_at: str
    update_available: bool = False
    latest_release: ReleaseInfo | None = None
    from_cache: bool = False
    rate_limit_remaining: int | None = None
    rate_limit_reset_ts: int | None = None
    message: str = ""
    error_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.latest_release is not None:
            data["latest_release"] = self.latest_release.to_dict()
        return data


@dataclass(frozen=True)
class AuthenticodeInfo:
    """Result of Windows Authenticode signature inspection."""

    status: str  # "Valid" | "NotSigned" | "HashMismatch" | "UntrustedRoot" | "UnknownError"
    signer_subject: str = ""
    signer_thumbprint: str = ""
    status_message: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status.lower() == "valid"


@dataclass(frozen=True)
class PackageVerificationResult:
    """Complete result of multi-step installer verification."""

    verified: bool
    can_auto_execute: bool
    requires_user_warning: bool
    installer_path: Path
    version: str
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    signature_status: str
    signer_subject: str
    trust_policy: str
    release_html_url: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ActiveOperationState:
    """Snapshot of active operations that must never be interrupted by an update."""

    is_recording: bool = False
    is_combining: bool = False
    is_processing: bool = False
    is_exporting: bool = False

    @property
    def is_busy(self) -> bool:
        return (
            self.is_recording
            or self.is_combining
            or self.is_processing
            or self.is_exporting
        )

    @property
    def busy_reason(self) -> str:
        reasons: list[str] = []
        if self.is_recording:
            reasons.append("registrazione audio in corso")
        if self.is_combining:
            reasons.append("combinazione tracce audio in corso")
        if self.is_processing:
            reasons.append("trascrizione o diarizzazione in corso")
        if self.is_exporting:
            reasons.append("esportazione in corso")
        if not reasons:
            return ""
        return "Impossibile avviare l'aggiornamento: " + ", ".join(reasons) + "."


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse a SHA256SUMS.txt file into a mapping of {filename: lowercase_sha256}."""
    mapping: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SHA256_LINE_RE.match(line)
        if match:
            digest = match.group("hash").lower()
            name = Path(match.group("name").strip()).name
            mapping[name] = digest
    return mapping


def compute_file_sha256(path: Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file in 1 MB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def inspect_authenticode_signature(path: Path) -> AuthenticodeInfo:
    """Inspect Windows Authenticode signature on an executable using PowerShell Get-AuthenticodeSignature."""
    if not path.is_file():
        return AuthenticodeInfo(status="FileNotFound", status_message="File non trovato.")
    if not sys.platform.startswith("win"):
        return AuthenticodeInfo(
            status="NotSupportedPlatform",
            status_message="Verifica Authenticode disponibile solo su Windows.",
        )

    ps_script = (
        "$sig = Get-AuthenticodeSignature -LiteralPath $env:TESSITURA_VERIFY_PATH; "
        "$out = @{ "
        "Status = [string]$sig.Status; "
        "StatusMessage = [string]$sig.StatusMessage; "
        "Subject = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Subject } else { '' }; "
        "Thumbprint = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Thumbprint } else { '' } "
        "}; "
        "$out | ConvertTo-Json -Compress"
    )
    env = os.environ.copy()
    env["TESSITURA_VERIFY_PATH"] = str(path.resolve())
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            creationflags=creationflags,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return AuthenticodeInfo(
                status="UnknownError",
                status_message=proc.stderr.strip() or "Impossibile leggere la firma Authenticode.",
            )
        data = json.loads(proc.stdout.strip())
        return AuthenticodeInfo(
            status=str(data.get("Status") or "UnknownError"),
            signer_subject=str(data.get("Subject") or ""),
            signer_thumbprint=str(data.get("Thumbprint") or ""),
            status_message=str(data.get("StatusMessage") or ""),
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return AuthenticodeInfo(
            status="UnknownError",
            status_message=str(exc),
        )


def _parse_github_release_dict(raw: dict[str, Any]) -> ReleaseInfo | None:
    """Convert a GitHub REST API release JSON object into a ReleaseInfo if it has a valid SemVer tag."""
    if not isinstance(raw, dict):
        return None
    if bool(raw.get("draft", False)):
        return None
    tag_name = str(raw.get("tag_name") or "").strip()
    if not tag_name:
        return None
    try:
        sem = SemVer.parse(tag_name)
    except ValueError:
        return None

    expected_installer = installer_asset_name(sem)
    installer_asset: ReleaseAssetInfo | None = None
    fallback_installer: ReleaseAssetInfo | None = None
    checksum_asset: ReleaseAssetInfo | None = None

    assets_raw = raw.get("assets")
    if isinstance(assets_raw, list):
        for item in assets_raw:
            if not isinstance(item, dict):
                continue
            name = Path(str(item.get("name") or "").strip()).name
            url = str(item.get("browser_download_url") or "").strip()
            try:
                size = int(item.get("size") or 0)
            except (TypeError, ValueError):
                size = 0
            ctype = str(item.get("content_type") or "")
            if not name or not url:
                continue
            asset = ReleaseAssetInfo(
                name=name,
                download_url=url,
                size=size,
                content_type=ctype,
            )
            if name == expected_installer:
                installer_asset = asset
            elif _SAFE_INSTALLER_NAME_RE.match(name) and fallback_installer is None:
                fallback_installer = asset
            elif name.upper() == "SHA256SUMS.TXT":
                checksum_asset = asset

    return ReleaseInfo(
        tag_name=tag_name,
        version=sem.normalized,
        name=str(raw.get("name") or tag_name),
        body=str(raw.get("body") or ""),
        html_url=str(raw.get("html_url") or ""),
        published_at=str(raw.get("published_at") or ""),
        prerelease=bool(raw.get("prerelease", False)) or sem.is_prerelease,
        draft=False,
        installer_asset=installer_asset or fallback_installer,
        checksum_asset=checksum_asset,
    )


class UpdateManager:
    """Manages GitHub Release discovery, caching, download, verification, and safe assisted installation."""

    def __init__(
        self,
        *,
        current_version: str | None = None,
        api_base_url: str | None = None,
        trust_policy: str | None = None,
        trusted_publishers: tuple[str, ...] | None = None,
        cache_file: Path | None = None,
        downloads_dir: Path | None = None,
        config_file: Path | None = None,
        http_opener: Callable[..., Any] | None = None,
        authenticode_inspector: Callable[[Path], AuthenticodeInfo] | None = None,
        activity_provider: Callable[[], ActiveOperationState] | None = None,
        process_launcher: Callable[[list[str]], Any] | None = None,
        quit_callback: Callable[[], None] | None = None,
        auto_check_interval_seconds: int = DEFAULT_AUTO_CHECK_INTERVAL_SECONDS,
    ) -> None:
        build_cfg = load_build_config()
        self.current_version = current_version or build_cfg.version
        self.api_base_url = (api_base_url or build_cfg.github_api_base_url).rstrip("/")
        self.releases_html_url = build_cfg.github_releases_html_url
        self.trust_policy = (trust_policy or build_cfg.trust_policy).lower()
        self.trusted_publishers = (
            trusted_publishers
            if trusted_publishers is not None
            else build_cfg.trusted_publishers
        )
        self.cache_file = (
            cache_file
            if cache_file is not None
            else (paths.cache_dir() / "github_releases_cache.json")
        )
        self.downloads_dir = (
            downloads_dir
            if downloads_dir is not None
            else paths.updates_cache_dir()
        )
        self.config_file = config_file
        self._http_opener = http_opener or urllib.request.urlopen
        self._authenticode_inspector = authenticode_inspector or inspect_authenticode_signature
        self._activity_provider = activity_provider or (lambda: ActiveOperationState())
        self._process_launcher = process_launcher
        self._quit_callback = quit_callback
        self.auto_check_interval_seconds = auto_check_interval_seconds

    def _load_http_cache(self) -> dict[str, Any]:
        if not self.cache_file.is_file():
            return {}
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_http_cache(self, cache_data: dict[str, Any]) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_file.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(cache_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self.cache_file)
        except OSError:
            pass

    def should_auto_check(self, now_epoch: float | None = None) -> bool:
        """Return True if automatic update checking is enabled and the 24h interval has elapsed."""
        settings = load_settings(self.config_file)
        if not settings.auto_check_updates:
            return False
        now = now_epoch if now_epoch is not None else time.time()
        if settings.remind_later_until_ts:
            try:
                remind_dt = datetime.fromisoformat(
                    settings.remind_later_until_ts.replace("Z", "+00:00")
                )
                if now < remind_dt.timestamp():
                    return False
            except ValueError:
                pass

        cache = self._load_http_cache()
        channel_entry = cache.get(settings.update_channel)
        if isinstance(channel_entry, dict):
            last_epoch = channel_entry.get("checked_epoch")
            if isinstance(last_epoch, (int, float)):
                if (now - float(last_epoch)) < self.auto_check_interval_seconds:
                    return False
        return True

    def check_for_updates(
        self,
        *,
        channel: str | None = None,
        force: bool = False,
        timeout: float = 10.0,
    ) -> UpdateCheckResult:
        """Query GitHub Releases for the selected channel ('stable' or 'beta')."""
        settings = load_settings(self.config_file)
        effective_channel = (channel or settings.update_channel or "stable").lower()
        if effective_channel not in {"stable", "beta"}:
            effective_channel = "stable"

        now_epoch = time.time()
        now_iso = _utc_now_iso()
        cache = self._load_http_cache()
        cached_entry = cache.get(effective_channel) if isinstance(cache.get(effective_channel), dict) else {}

        if not force and cached_entry:
            last_epoch = cached_entry.get("checked_epoch")
            cached_release_raw = cached_entry.get("selected_release")
            if (
                isinstance(last_epoch, (int, float))
                and (now_epoch - float(last_epoch)) < self.auto_check_interval_seconds
                and isinstance(cached_release_raw, dict)
            ):
                release = ReleaseInfo.from_dict(cached_release_raw)
                return self._evaluate_release(
                    release=release,
                    channel=effective_channel,
                    checked_at=str(cached_entry.get("checked_at") or now_iso),
                    from_cache=True,
                )

        if effective_channel == "stable":
            endpoint = f"{self.api_base_url}/releases/latest"
        else:
            endpoint = f"{self.api_base_url}/releases?per_page=30"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"Tessitura/{self.current_version} (Windows; x64)",
        }
        if cached_entry.get("etag"):
            headers["If-None-Match"] = str(cached_entry["etag"])
        if cached_entry.get("last_modified"):
            headers["If-Modified-Since"] = str(cached_entry["last_modified"])

        req = urllib.request.Request(endpoint, headers=headers, method="GET")
        rate_remaining: int | None = None
        rate_reset: int | None = None

        try:
            with self._http_opener(req, timeout=timeout) as resp:
                resp_headers = getattr(resp, "headers", {}) or {}
                rem_raw = resp_headers.get("X-RateLimit-Remaining")
                rst_raw = resp_headers.get("X-RateLimit-Reset")
                if rem_raw is not None:
                    try:
                        rate_remaining = int(rem_raw)
                    except ValueError:
                        pass
                if rst_raw is not None:
                    try:
                        rate_reset = int(rst_raw)
                    except ValueError:
                        pass

                raw_bytes = resp.read()
                payload = json.loads(raw_bytes.decode("utf-8"))
                selected = self._select_release_from_payload(payload, effective_channel)

                cache[effective_channel] = {
                    "etag": resp_headers.get("ETag") or cached_entry.get("etag") or "",
                    "last_modified": resp_headers.get("Last-Modified") or cached_entry.get("last_modified") or "",
                    "checked_at": now_iso,
                    "checked_epoch": now_epoch,
                    "selected_release": selected.to_dict() if selected else None,
                }
                self._save_http_cache(cache)
                self._record_last_check(now_iso, effective_channel)

                if selected is None:
                    return UpdateCheckResult(
                        status="no_releases",
                        channel=effective_channel,
                        current_version=self.current_version,
                        checked_at=now_iso,
                        update_available=False,
                        rate_limit_remaining=rate_remaining,
                        rate_limit_reset_ts=rate_reset,
                        message="Nessuna release valida trovata sul canale selezionato.",
                    )
                return self._evaluate_release(
                    release=selected,
                    channel=effective_channel,
                    checked_at=now_iso,
                    from_cache=False,
                    rate_limit_remaining=rate_remaining,
                    rate_limit_reset_ts=rate_reset,
                )

        except urllib.error.HTTPError as exc:
            hdrs = getattr(exc, "headers", {}) or {}
            rem_raw = hdrs.get("X-RateLimit-Remaining")
            rst_raw = hdrs.get("X-RateLimit-Reset")
            if rem_raw is not None:
                try:
                    rate_remaining = int(rem_raw)
                except ValueError:
                    pass
            if rst_raw is not None:
                try:
                    rate_reset = int(rst_raw)
                except ValueError:
                    pass

            if exc.code == 304 and cached_entry:
                cached_entry["checked_at"] = now_iso
                cached_entry["checked_epoch"] = now_epoch
                cache[effective_channel] = cached_entry
                self._save_http_cache(cache)
                self._record_last_check(now_iso, effective_channel)
                rel_raw = cached_entry.get("selected_release")
                if isinstance(rel_raw, dict):
                    release = ReleaseInfo.from_dict(rel_raw)
                    return self._evaluate_release(
                        release=release,
                        channel=effective_channel,
                        checked_at=now_iso,
                        from_cache=True,
                        rate_limit_remaining=rate_remaining,
                        rate_limit_reset_ts=rate_reset,
                    )
                return UpdateCheckResult(
                    status="no_releases",
                    channel=effective_channel,
                    current_version=self.current_version,
                    checked_at=now_iso,
                    from_cache=True,
                    message="Nessun aggiornamento disponibile (cache confermata 304).",
                )

            if exc.code == 404:
                # If /releases/latest returns 404 on Stable channel, there is no stable release yet
                return UpdateCheckResult(
                    status="no_releases",
                    channel=effective_channel,
                    current_version=self.current_version,
                    checked_at=now_iso,
                    message="Nessuna release pubblicata nel repository configurato.",
                )

            if exc.code in (403, 429) or rate_remaining == 0:
                cached_rel = self._cached_release_or_none(cached_entry)
                return UpdateCheckResult(
                    status="rate_limited",
                    channel=effective_channel,
                    current_version=self.current_version,
                    checked_at=now_iso,
                    latest_release=cached_rel,
                    update_available=self._is_release_newer(cached_rel, effective_channel),
                    from_cache=cached_rel is not None,
                    rate_limit_remaining=rate_remaining,
                    rate_limit_reset_ts=rate_reset,
                    message="Limite di richieste API GitHub raggiunto. Riprova più tardi.",
                    error_detail=f"HTTP {exc.code}",
                )

            return UpdateCheckResult(
                status="error",
                channel=effective_channel,
                current_version=self.current_version,
                checked_at=now_iso,
                message=f"Errore del server GitHub (HTTP {exc.code}).",
                error_detail=str(exc),
            )

        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            cached_rel = self._cached_release_or_none(cached_entry)
            return UpdateCheckResult(
                status="offline",
                channel=effective_channel,
                current_version=self.current_version,
                checked_at=now_iso,
                latest_release=cached_rel,
                update_available=self._is_release_newer(cached_rel, effective_channel),
                from_cache=cached_rel is not None,
                message="Connessione non disponibile. Controllo aggiornamenti rimandato.",
                error_detail=str(exc),
            )

    def _record_last_check(self, checked_at: str, channel: str) -> None:
        try:
            settings = load_settings(self.config_file)
            settings.last_update_check_ts = checked_at
            settings.update_channel = channel
            save_settings(settings, self.config_file)
        except OSError:
            pass

    def _cached_release_or_none(self, cached_entry: dict[str, Any]) -> ReleaseInfo | None:
        if not isinstance(cached_entry, dict):
            return None
        raw = cached_entry.get("selected_release")
        if isinstance(raw, dict):
            try:
                return ReleaseInfo.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def _select_release_from_payload(
        self,
        payload: Any,
        channel: str,
    ) -> ReleaseInfo | None:
        """Select the highest SemVer release matching the requested channel."""
        if channel == "stable":
            if isinstance(payload, dict):
                rel = _parse_github_release_dict(payload)
                if rel and not rel.prerelease and not rel.semver.is_prerelease:
                    return rel
                return None
            if isinstance(payload, list):
                candidates = [
                    r
                    for item in payload
                    if (r := _parse_github_release_dict(item)) is not None
                    and not r.prerelease
                    and not r.semver.is_prerelease
                ]
                return max(candidates, key=lambda r: r.semver) if candidates else None
            return None

        # Beta channel: evaluate all non-draft releases (stable + prerelease) using SemVer 2.0.0 precedence
        items = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
        candidates = [
            r
            for item in items
            if (r := _parse_github_release_dict(item)) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.semver)

    def _is_release_newer(self, release: ReleaseInfo | None, channel: str) -> bool:
        if release is None:
            return False
        try:
            curr = SemVer.parse(self.current_version)
            cand = release.semver
        except ValueError:
            return False
        if channel == "stable" and (release.prerelease or cand.is_prerelease):
            return False
        return cand > curr

    def _evaluate_release(
        self,
        *,
        release: ReleaseInfo,
        channel: str,
        checked_at: str,
        from_cache: bool,
        rate_limit_remaining: int | None = None,
        rate_limit_reset_ts: int | None = None,
    ) -> UpdateCheckResult:
        newer = self._is_release_newer(release, channel)
        if newer:
            msg = f"Nuova versione disponibile: {release.version} (installata: {self.current_version})."
            status = "update_available"
        else:
            msg = f"Tessitura è aggiornata ({self.current_version})."
            status = "up_to_date"
        return UpdateCheckResult(
            status=status,
            channel=channel,
            current_version=self.current_version,
            checked_at=checked_at,
            update_available=newer,
            latest_release=release,
            from_cache=from_cache,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset_ts=rate_limit_reset_ts,
            message=msg,
        )

    def download_release_installer(
        self,
        release: ReleaseInfo,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        timeout: float = 30.0,
    ) -> tuple[Path, str]:
        """Download the installer and SHA256SUMS.txt for a release into the staging directory.

        Returns (installer_path, expected_sha256).
        """
        if release.installer_asset is None:
            raise InstallerVerificationError(
                f"La release {release.version} non contiene un pacchetto di installazione Windows valido."
            )
        if release.checksum_asset is None:
            raise InstallerVerificationError(
                f"La release {release.version} non contiene il file di verifica SHA256SUMS.txt."
            )

        asset_name = Path(release.installer_asset.name).name
        if not _SAFE_INSTALLER_NAME_RE.match(asset_name):
            raise InstallerVerificationError(
                f"Nome file installer non sicuro o non conforme: {asset_name!r}"
            )

        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch and parse SHA256SUMS.txt
        checksum_req = urllib.request.Request(
            release.checksum_asset.download_url,
            headers={"User-Agent": f"Tessitura/{self.current_version}"},
            method="GET",
        )
        try:
            with self._http_opener(checksum_req, timeout=timeout) as resp:
                checksum_text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise InstallerVerificationError(
                f"Impossibile scaricare SHA256SUMS.txt: {exc}"
            ) from exc

        checksums = parse_sha256sums(checksum_text)
        expected_sha256 = checksums.get(asset_name)
        if not expected_sha256:
            raise InstallerVerificationError(
                f"Hash SHA-256 per {asset_name} non trovato in SHA256SUMS.txt."
            )

        target_path = self.downloads_dir / asset_name
        part_path = self.downloads_dir / f"{asset_name}.part"

        # If target_path already exists and matches size + SHA256, reuse it!
        if target_path.is_file():
            if (
                release.installer_asset.size <= 0
                or target_path.stat().st_size == release.installer_asset.size
            ):
                if hmac.compare_digest(compute_file_sha256(target_path), expected_sha256):
                    if progress_callback:
                        sz = target_path.stat().st_size
                        progress_callback(sz, sz)
                    return target_path, expected_sha256

        # 2. Stream download to .part file
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass

        inst_req = urllib.request.Request(
            release.installer_asset.download_url,
            headers={"User-Agent": f"Tessitura/{self.current_version}"},
            method="GET",
        )
        expected_size = release.installer_asset.size
        downloaded = 0
        try:
            with self._http_opener(inst_req, timeout=timeout) as resp:
                if expected_size <= 0:
                    cl = getattr(resp, "headers", {}).get("Content-Length")
                    if cl and str(cl).isdigit():
                        expected_size = int(cl)
                with part_path.open("wb") as fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, expected_size)
        except Exception as exc:
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            raise InstallerVerificationError(
                f"Download dell'installer interrotto o non riuscito: {exc}"
            ) from exc

        if expected_size > 0 and downloaded != expected_size:
            try:
                part_path.unlink()
            except OSError:
                pass
            raise InstallerVerificationError(
                f"Dimensione download incompleta: ricevuti {downloaded} byte su {expected_size} attesi."
            )

        os.replace(part_path, target_path)
        return target_path, expected_sha256

    def verify_installer_package(
        self,
        installer_path: Path,
        *,
        target_version: str,
        expected_sha256: str,
        expected_size: int = 0,
        release_html_url: str = "",
        allow_same_version: bool = False,
    ) -> PackageVerificationResult:
        """Perform multi-step security and integrity verification on a downloaded installer."""
        if not installer_path.is_file():
            raise InstallerVerificationError(f"File installer non trovato: {installer_path}")

        # 1. Version / Downgrade protection
        curr_sem = SemVer.parse(self.current_version)
        target_sem = SemVer.parse(target_version)
        if target_sem < curr_sem:
            raise InstallerVerificationError(
                f"Downgrade bloccato: impossibile installare la versione {target_version} "
                f"sulla versione più recente {self.current_version}."
            )
        if target_sem == curr_sem and not allow_same_version:
            raise InstallerVerificationError(
                f"La versione {target_version} è già installata."
            )

        # 2. File size verification
        actual_size = installer_path.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            try:
                installer_path.unlink()
            except OSError:
                pass
            raise InstallerVerificationError(
                f"Dimensione installer non valida: {actual_size} byte (attesi {expected_size} byte)."
            )

        # 3. SHA-256 checksum verification
        actual_sha256 = compute_file_sha256(installer_path)
        clean_expected = expected_sha256.strip().lower()
        if not hmac.compare_digest(actual_sha256, clean_expected):
            try:
                installer_path.unlink()
            except OSError:
                pass
            raise InstallerVerificationError(
                f"Verifica SHA-256 fallita per {installer_path.name}: "
                f"atteso {clean_expected[:16]}…, calcolato {actual_sha256[:16]}…. "
                "Il file è stato rimosso per sicurezza."
            )

        # 4. Authenticode signature & publisher trust policy check
        auth = self._authenticode_inspector(installer_path)
        policy = self.trust_policy

        if policy == "disabled":
            return PackageVerificationResult(
                verified=True,
                can_auto_execute=False,
                requires_user_warning=True,
                installer_path=installer_path,
                version=target_version,
                expected_sha256=clean_expected,
                actual_sha256=actual_sha256,
                size_bytes=actual_size,
                signature_status=auth.status,
                signer_subject=auth.signer_subject,
                trust_policy=policy,
                release_html_url=release_html_url or self.releases_html_url,
                reason="L'esecuzione automatica degli aggiornamenti è disabilitata dalla policy di sicurezza.",
            )

        if auth.is_valid:
            if self.trusted_publishers:
                subj_lower = auth.signer_subject.lower()
                matched = any(pub.lower() in subj_lower for pub in self.trusted_publishers)
                if not matched:
                    return PackageVerificationResult(
                        verified=False,
                        can_auto_execute=False,
                        requires_user_warning=True,
                        installer_path=installer_path,
                        version=target_version,
                        expected_sha256=clean_expected,
                        actual_sha256=actual_sha256,
                        size_bytes=actual_size,
                        signature_status="UntrustedPublisher",
                        signer_subject=auth.signer_subject,
                        trust_policy=policy,
                        release_html_url=release_html_url or self.releases_html_url,
                        reason=(
                            f"Firma Authenticode valida ma firmatario ({auth.signer_subject!r}) "
                            "non presente nella lista dei publisher attendibili configurati."
                        ),
                    )
            return PackageVerificationResult(
                verified=True,
                can_auto_execute=True,
                requires_user_warning=False,
                installer_path=installer_path,
                version=target_version,
                expected_sha256=clean_expected,
                actual_sha256=actual_sha256,
                size_bytes=actual_size,
                signature_status=auth.status,
                signer_subject=auth.signer_subject,
                trust_policy=policy,
                release_html_url=release_html_url or self.releases_html_url,
                reason="Pacchetto verificato con SHA-256 e firma digitale Authenticode valida.",
            )

        # Signature is NOT valid (e.g. NotSigned)
        if policy == "strict":
            return PackageVerificationResult(
                verified=False,
                can_auto_execute=False,
                requires_user_warning=True,
                installer_path=installer_path,
                version=target_version,
                expected_sha256=clean_expected,
                actual_sha256=actual_sha256,
                size_bytes=actual_size,
                signature_status=auth.status,
                signer_subject=auth.signer_subject,
                trust_policy=policy,
                release_html_url=release_html_url or self.releases_html_url,
                reason=(
                    f"Il pacchetto ha superato la verifica SHA-256 ma non possiede una firma "
                    f"Authenticode valida (stato: {auth.status}). La policy 'strict' impedisce "
                    "l'esecuzione automatica di installer non firmati."
                ),
            )

        # policy == "checksum_only": SHA-256 is verified, warn user about unsigned binary
        return PackageVerificationResult(
            verified=True,
            can_auto_execute=True,
            requires_user_warning=True,
            installer_path=installer_path,
            version=target_version,
            expected_sha256=clean_expected,
            actual_sha256=actual_sha256,
            size_bytes=actual_size,
            signature_status=auth.status,
            signer_subject=auth.signer_subject,
            trust_policy=policy,
            release_html_url=release_html_url or self.releases_html_url,
            reason=(
                "Integrità SHA-256 verificata rispetto a SHA256SUMS.txt. "
                "L'installer non è firmato con certificato Authenticode."
            ),
        )

    def can_start_update(self) -> tuple[bool, str]:
        """Check whether the application is idle and safe to close for an update."""
        state = self._activity_provider()
        if state.is_busy:
            return False, state.busy_reason
        return True, ""

    def prepare_and_launch_installer(
        self,
        verification: PackageVerificationResult,
        *,
        channel: str = "stable",
        installer_args: list[str] | None = None,
    ) -> Path:
        """Prepare state backup, record pending_update, launch verified installer externally, and close app."""
        ok, busy_reason = self.can_start_update()
        if not ok:
            raise UpdateBlockedByActiveOperationError(busy_reason)

        if not verification.can_auto_execute or not verification.verified:
            raise InstallerVerificationError(
                verification.reason
                or "L'installer non soddisfa i requisiti della policy di sicurezza per l'avvio automatico."
            )

        if not verification.installer_path.is_file():
            raise InstallerVerificationError(
                f"Il file installer verificato non esiste più: {verification.installer_path}"
            )

        # Re-verify SHA-256 right before execution (TOCTOU guard)
        current_hash = compute_file_sha256(verification.installer_path)
        if not hmac.compare_digest(current_hash, verification.expected_sha256):
            raise InstallerVerificationError(
                "L'hash SHA-256 dell'installer è cambiato prima dell'esecuzione. Avvio bloccato."
            )

        cfg_root = self.config_file.parent if self.config_file is not None else paths.config_dir()
        backup = create_state_backup(
            config_root=cfg_root,
            previous_version=self.current_version,
            target_version=verification.version,
        )

        settings = load_settings(self.config_file)
        settings.pending_update = {
            "target_version": verification.version,
            "previous_version": self.current_version,
            "installer_path": str(verification.installer_path.resolve()),
            "sha256": verification.actual_sha256,
            "backup_dir": str(backup.backup_dir),
            "started_at": _utc_now_iso(),
            "channel": channel,
        }
        save_settings(settings, self.config_file)

        cmd = [str(verification.installer_path.resolve())]
        if installer_args:
            cmd.extend(installer_args)

        if self._process_launcher is not None:
            self._process_launcher(cmd)
        else:
            creationflags = 0
            if sys.platform.startswith("win"):
                creationflags = (
                    getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                )
            subprocess.Popen(
                cmd,
                cwd=str(verification.installer_path.parent),
                close_fds=True,
                creationflags=creationflags,
            )

        if self._quit_callback is not None:
            self._quit_callback()

        return backup.backup_dir

