"""Diagnostica acustica e trattamento conservativo dell'audio.

Implementa i requisiti del pacchetto P03 (sezione 6.2 del piano esecutivo):
- Analisi temporale lungo l'intero audio (non limitata ai primi millisecondi)
- Valutazione livelli per canale, clipping, fondo di rumore e correlazione inter-canale
- Riconoscimento dell'affidabilita' del profilo di fondo per evitare gating distruttivo
- Creazione di viste trattate conservative che preservano le voci deboli e non clippano
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..models.audio_artifacts import AudioAsset, AudioView
from .audio_utils import PEAK_CEILING, rms_dbfs


def _safe_float(val: float, default: float = 0.0) -> float:
    if not math.isfinite(val):
        return default
    return float(val)


@dataclass
class AcousticProfile:
    """Riepilogo delle caratteristiche acustiche del segnale su tutta la durata."""

    rms_dbfs_per_channel: list[float] = field(default_factory=list)
    peak_per_channel: list[float] = field(default_factory=list)
    clipping_ratio: float = 0.0
    estimated_noise_floor_dbfs: float = -np.inf
    estimated_speech_level_dbfs: float = -np.inf
    estimated_snr_db: float = 0.0
    channel_correlation: float | None = None
    has_reliable_noise_profile: bool = False
    is_silent: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_acoustics(
    data: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 20,
) -> AcousticProfile:
    """Esegue un'analisi acustica globale su array float32 (mono o multicanale).

    Esamina l'intera traccia suddividendola in frame per stimare sia l'attivita'
    vocale che il fondo di rumore, evitando di considerare rumore puro il primo frame.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape
    if n_samples == 0:
        return AcousticProfile(is_silent=True, warnings=["Segnale vuoto"])

    # Metriche per canale
    rms_channels = [rms_dbfs(data[:, ch]) for ch in range(n_channels)]
    peak_channels = [float(np.max(np.abs(data[:, ch]))) for ch in range(n_channels)]

    # Clipping detection (campioni che toccano o superano il fondo scala)
    clipping_samples = int(np.sum(np.abs(data) >= 0.999))
    clipping_ratio = float(clipping_samples / max(1, data.size))
    warnings: list[str] = []
    if clipping_ratio > 0.001:
        warnings.append(f"Rilevato clipping su {clipping_ratio * 100:.2f}% dei campioni")

    # Se tutti i canali sono silenzio assoluto
    if all(not np.isfinite(r) or r < -80.0 for r in rms_channels):
        return AcousticProfile(
            rms_dbfs_per_channel=[_safe_float(r, -100.0) for r in rms_channels],
            peak_per_channel=peak_channels,
            clipping_ratio=clipping_ratio,
            is_silent=True,
            warnings=warnings + ["Audio essenzialmente silenzioso (< -80 dBFS)"],
        )

    # Correlazione inter-canale (per stereo o piu' canali)
    channel_corr: float | None = None
    if n_channels == 2:
        ch0, ch1 = data[:, 0], data[:, 1]
        std0, std1 = np.std(ch0), np.std(ch1)
        if std0 > 1e-6 and std1 > 1e-6:
            corr_mat = np.corrcoef(ch0, ch1)
            channel_corr = float(corr_mat[0, 1]) if math.isfinite(corr_mat[0, 1]) else None

    # Calcolo energia frame-by-frame su media mono
    mono = data.mean(axis=1) if n_channels > 1 else data[:, 0]
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = n_samples // frame_len

    if n_frames < 2:
        # Troppo breve per stima statistica del rumore
        return AcousticProfile(
            rms_dbfs_per_channel=[_safe_float(r, -100.0) for r in rms_channels],
            peak_per_channel=peak_channels,
            clipping_ratio=clipping_ratio,
            channel_correlation=channel_corr,
            has_reliable_noise_profile=False,
            warnings=warnings + ["Registrazione troppo breve per profilazione temporale affidabile"],
        )

    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_energies = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    # Converti in dBFS escludendo zeri
    nonzero = frame_energies[frame_energies > 1e-7]
    if len(nonzero) < 2:
        return AcousticProfile(
            rms_dbfs_per_channel=[_safe_float(r, -100.0) for r in rms_channels],
            peak_per_channel=peak_channels,
            clipping_ratio=clipping_ratio,
            is_silent=True,
            warnings=warnings + ["Energia insufficiente nei frame"],
        )

    frame_db = 20.0 * np.log10(nonzero)

    # Fondo di rumore stimato sul 15-esimo percentile dei frame non nulli
    noise_floor_db = float(np.percentile(frame_db, 15))
    # Livello del parlato stimato sui frame sopra il 70-esimo percentile
    speech_level_db = float(np.percentile(frame_db, 75))
    snr_est = max(0.0, speech_level_db - noise_floor_db)

    # Affidabilita' del profilo di fondo:
    # Serve una dinamica sufficiente (> 6 dB) e un rumore non eccessivamente alto (> -15 dBFS)
    # ne' irrilevante (< -65 dBFS dove il denoise rischia solo di creare artefatti).
    reliable = bool(6.0 <= snr_est <= 45.0 and -65.0 <= noise_floor_db <= -15.0)

    if not reliable:
        if snr_est < 6.0:
            warnings.append("SNR basso o segnale continuo senza pause distinte")
        elif noise_floor_db < -65.0:
            warnings.append("Fondo estremamente pulito; denoise non necessario")

    return AcousticProfile(
        rms_dbfs_per_channel=[_safe_float(r, -100.0) for r in rms_channels],
        peak_per_channel=peak_channels,
        clipping_ratio=clipping_ratio,
        estimated_noise_floor_dbfs=noise_floor_db,
        estimated_speech_level_dbfs=speech_level_db,
        estimated_snr_db=snr_est,
        channel_correlation=channel_corr,
        has_reliable_noise_profile=reliable,
        is_silent=False,
        warnings=warnings,
    )


def create_conservative_view(
    asset: AudioAsset,
    audio_data: np.ndarray,
    profile: AcousticProfile,
    *,
    base_view: AudioView | None = None,
    prop_decrease: float = 0.4,
) -> tuple[AudioView, np.ndarray]:
    """Produce una vista e un array trattato conservativamente.

    Se il profilo di rumore non e' affidabile, preserva l'audio immutato (vista raw).
    Se affidabile, applica spectral gating prudente con compensazione di guadagno e tetto
    anti-clipping, evitando di degradare le voci deboli.
    """
    transformations: list[dict[str, Any]] = []

    if not profile.has_reliable_noise_profile or profile.is_silent:
        # Mantieni vista passante senza filtraggio distruttivo
        view = AudioView(
            asset_id=asset.asset_id,
            channel_indices=base_view.channel_indices if base_view else list(range(asset.num_channels)),
            sample_rate=asset.sample_rate,
            time_offset_s=base_view.time_offset_s if base_view else 0.0,
            duration_s=base_view.duration_s if base_view else asset.duration_s,
            transformations=[{"type": "passthrough", "reason": "unreliable_or_clean_noise_profile"}],
            parent_view_id=base_view.view_id if base_view else None,
        )
        return view, audio_data

    # Applicazione denoise conservativo
    try:
        from .denoise import denoise_array
        treated = denoise_array(
            audio_data,
            asset.sample_rate,
            stationary=True,
            prop_decrease=prop_decrease,
        )
        transformations.append({
            "type": "spectral_gating_conservative",
            "prop_decrease": prop_decrease,
            "estimated_snr_db": round(profile.estimated_snr_db, 2),
        })
    except Exception as exc:
        treated = audio_data
        transformations.append({
            "type": "fallback_raw",
            "error": str(exc),
        })

    # Verifica energetica: assicurarsi che il picco non superi PEAK_CEILING
    peak = float(np.max(np.abs(treated))) if treated.size else 0.0
    if peak > PEAK_CEILING:
        scale = PEAK_CEILING / peak
        treated = (treated * scale).astype(np.float32)
        transformations.append({"type": "ceiling_limit", "scale": round(scale, 4)})

    view = AudioView(
        asset_id=asset.asset_id,
        channel_indices=base_view.channel_indices if base_view else list(range(asset.num_channels)),
        sample_rate=asset.sample_rate,
        time_offset_s=base_view.time_offset_s if base_view else 0.0,
        duration_s=base_view.duration_s if base_view else asset.duration_s,
        transformations=transformations,
        parent_view_id=base_view.view_id if base_view else None,
    )
    return view, treated
