"""First-run model provisioning, disk space verification, and offline readiness checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

from src import paths
from src.config import AppSettings, load_settings, save_settings
from src.hardware import Hardware, detect
from src.processing.model_catalog import diarization_catalog, transcription_catalog
from src.processing.model_store import cached_path


@dataclass(frozen=True)
class FirstRunStatus:
    """Summary of local AI model availability, disk space, and hardware readiness."""

    models_dir: Path
    recordings_dir: Path
    config_dir: Path
    device: str
    gpu_name: str | None
    vram_gb: float | None
    recommended_asr_model: str
    asr_model_cached: bool
    diarization_model_cached: bool
    hf_token_present: bool
    free_disk_gb: float
    estimated_download_gb: float
    has_sufficient_disk: bool
    first_run_completed: bool

    @property
    def is_offline_ready(self) -> bool:
        """True if at least the recommended ASR model is already cached locally."""
        return self.asr_model_cached

    @property
    def needs_setup_prompt(self) -> bool:
        """True if first-run setup has not been completed and models are not yet cached."""
        return not self.first_run_completed and not self.asr_model_cached

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["models_dir"] = str(self.models_dir)
        data["recordings_dir"] = str(self.recordings_dir)
        data["config_dir"] = str(self.config_dir)
        return data


def _disk_free_gb(target_dir: Path) -> float:
    probe = target_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        return round(usage.free / (1024**3), 2)
    except OSError:
        return 0.0


def _has_hf_token() -> bool:
    from src.processing.hf_access import token_info

    return token_info().present


def _estimate_model_gb(model_id: str, include_diarization: bool = True) -> float:
    asr_gb = 3.1
    for entry in transcription_catalog():
        if entry.key == model_id and entry.approx_gb is not None:
            asr_gb = float(entry.approx_gb)
            break
    diar_gb = 0.45 if include_diarization else 0.0
    return round(asr_gb + diar_gb, 2)


def check_first_run_status(
    hw: Hardware | None = None,
    *,
    config_file: Path | None = None,
) -> FirstRunStatus:
    """Inspect local model cache, disk space, and hardware without downloading anything."""
    hardware = hw if hw is not None else detect()
    settings = load_settings(config_file)
    m_dir = paths.models_dir()
    r_dir = paths.data_dir()
    c_dir = paths.config_dir()

    recommended = hardware.model or "large-v3"
    asr_entries = {e.key: e for e in transcription_catalog()}
    asr_entry = asr_entries.get(recommended)
    asr_cached = False
    if asr_entry is not None:
        try:
            asr_cached = cached_path(asr_entry.repo, "trascrizione") is not None
        except Exception:
            asr_cached = False

    diar_entries = diarization_catalog()
    diar_cached = False
    if diar_entries:
        try:
            diar_cached = cached_path(diar_entries[0].repo, "diarizzazione") is not None
        except Exception:
            diar_cached = False

    est_gb = 0.0
    if not asr_cached:
        est_gb += _estimate_model_gb(recommended, include_diarization=False)
    if not diar_cached:
        est_gb += 0.45
    est_gb = round(est_gb, 2)

    free_gb = _disk_free_gb(m_dir)
    sufficient = free_gb >= (est_gb + 1.0) if est_gb > 0 else True

    return FirstRunStatus(
        models_dir=m_dir,
        recordings_dir=r_dir,
        config_dir=c_dir,
        device=hardware.device,
        gpu_name=hardware.gpu_name,
        vram_gb=hardware.vram_gb,
        recommended_asr_model=recommended,
        asr_model_cached=asr_cached,
        diarization_model_cached=diar_cached,
        hf_token_present=_has_hf_token(),
        free_disk_gb=free_gb,
        estimated_download_gb=est_gb,
        has_sufficient_disk=sufficient,
        first_run_completed=settings.first_run_completed,
    )


def configure_user_directories(
    *,
    models_dir: Path | str | None = None,
    recordings_dir: Path | str | None = None,
    hf_token: str | None = None,
    mark_completed: bool = True,
    config_file: Path | None = None,
) -> AppSettings:
    """Persist custom models/recordings directories outside the install folder.

    An optional HF token goes to the OS credential vault, never to a file;
    it is validated and stored before any setting changes.
    """
    if hf_token is not None and hf_token.strip():
        from src.processing.hf_access import save_token

        save_token(hf_token)
    settings = load_settings(config_file)
    if models_dir is not None:
        m_str = str(models_dir).strip()
        if m_str:
            m_path = Path(m_str).expanduser().resolve()
            m_path.mkdir(parents=True, exist_ok=True)
            settings.models_dir = str(m_path)
        else:
            settings.models_dir = None

    if recordings_dir is not None:
        r_str = str(recordings_dir).strip()
        if r_str:
            r_path = Path(r_str).expanduser().resolve()
            r_path.mkdir(parents=True, exist_ok=True)
            settings.recordings_dir = str(r_path)
        else:
            settings.recordings_dir = None

    if mark_completed:
        settings.first_run_completed = True

    save_settings(settings, config_file)
    paths.reset_caches()
    return settings
