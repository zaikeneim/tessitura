"""Pipeline batch: audio registrato -> Session trascritta e diarizzata.

Passi: (1) validazione obbligatoria del profilo e missaggio in un unico mono
16 kHz normalizzato, (2) denoise opzionale *solo per la trascrizione*,
(3) trascrizione con timestamp per parola, (4) diarizzazione sull'audio NON
denoised, (5) merge parola-per-parola con gestione dei turni brevi.

Due scelte non ovvie, entrambe nate da bug misurati su registrazioni reali:

- La **diarizzazione gira sull'audio originale**, non su quello denoised. Lo
  spectral gating altera il timbro, che e' esattamente cio' su cui pyannote
  costruisce le impronte vocali: diarizzare il denoised faceva confondere le voci.
  pyannote e' gia' addestrato su audio rumoroso, il denoise non gli serve.

- Con **due sorgenti separate** (microfono e audio di sistema) si sa da quale
  canale arriva ogni parola. Quel dato separa "io" dai partecipanti remoti molto
  meglio del solo timbro, e viene usato per correggere le etichette.

- Con il profilo **ambientale stereo** i canali non vengono considerati sorgenti
  indipendenti: il channel hint resta disattivato per evitare di inventare una
  distinzione mic/remoto che il recorder non fornisce.
"""

from __future__ import annotations

import tempfile
import logging
import copy
import gc
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import paths
from ..models.session import Session, Segment
from . import audio_utils, merge
from .alignment import align_segments
from .audio_profile import AudioProfile
from .diarize import Turn
from .transcript_review import review_labels

# Compatibilita': la funzione di merge vive ora in merge.py.
merge_transcript_and_diarization = merge.merge_transcript_and_diarization

ProgressCb = Callable[[str, float], None]  # (messaggio, frazione 0..1)


def resolve_device(device: str = "auto") -> tuple[str, str]:
    """(device, compute_type) effettivi: "auto" sceglie in base all'hardware."""
    from ..hardware import detect

    hw = detect(force=device if device in ("cuda", "cpu") else None)
    return hw.device, hw.compute_type


@dataclass
class PipelineOptions:
    denoise: bool = False
    diarize: bool = True
    # None = Whisper rileva la lingua da solo (registrazioni it/en/pt miste).
    language: str | None = None
    model_size: str = "large-v3"
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    compute_type: str | None = None  # None = dedotto dal device
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Quanto rumore togliere (0..1). Sopra ~0.6 il gating inizia a mangiare voce.
    denoise_strength: float = 0.5
    # Usa il canale (mic vs sistema) per correggere l'attribuzione delle voci.
    use_channel_hint: bool = True
    # Diarizza sull'audio originale invece che sul denoised (consigliato).
    diarize_on_raw: bool = True
    initial_prompt: str | None = None
    # Retry mirati per segmenti degeneri; 0 disattiva la seconda passata.
    quality_retries: int = 24
    # Obbligatorio: senza questo dato un file stereo potrebbe essere trattato
    # erroneamente come mic+audio di sistema.
    audio_profile: AudioProfile | str | None = None
    # Controllo rapido dell'attribuzione prima di restituire la sessione.
    # "safe" non usa rete o modelli aggiuntivi; "off" mantiene il merge grezzo.
    analysis_mode: str = "safe"
    # Profilazione preventiva delle voci e correlazione con profili noti del progetto.
    known_speaker_profiles: dict[str, list[float]] | None = None
    profile_first: bool = False
    # Autocorrezione per troncamento frasi e sovrapposizione voci.
    autocorrect_truncation: bool = True
    autocorrect_overlap: bool = True
    # Batch size per faster-whisper (1 = sequenziale con beam search; >1 = BatchedInferencePipeline)
    batch_size: int = 1
    # Backend di allineamento forzato ("auto" | "mms_fa" | "whisper")
    alignment_backend: str = "auto"
    strategy: str = "standard"
    headphones_confirmed: bool | None = None
    # Capture topology is explicit; neither endpoint names nor channel count imply it.
    recording_mode: str | None = None  # "ambient" | "headset" | "speakers"
    mic_single_speaker_verified: bool = False
    echo_cancellation: bool = False
    recording_metadata: dict | None = None


# Fasi interne di pyannote, per stimare l'avanzamento prima di averle viste tutte.
DIARIZATION_STEPS = (
    "analisi dei turni", "conteggio voci", "impronte vocali", "assegnazione voci",
)


def _noop(_msg: str, _frac: float) -> None:
    pass


def _portable_source_metadata(metadata: dict) -> dict:
    """Keep stored capture views portable without changing operational file paths."""
    result = copy.deepcopy(metadata)
    file_fields = {"original", "processing_audio", "playback_audio", "original_mic",
                   "original_system", "derivative", "asr_input", "audio_path", "processing_audio_path"}

    def convert(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in file_fields and isinstance(item, str) and Path(item).is_absolute():
                    portable = paths.to_portable(item)
                    # Preserve existing spelling for external imported media.
                    if portable is not None and not Path(portable).is_absolute():
                        value[key] = portable
                else:
                    convert(item)
        elif isinstance(value, list):
            for item in value:
                convert(item)

    convert(result)
    return result


def _attach_recording_metadata(session: Session, options: PipelineOptions) -> None:
    """Preserve native capture provenance in every processing branch."""
    if options.recording_metadata or options.recording_mode:
        session.capture_manifest = {
            **(session.capture_manifest or {}),
            "recording_mode": options.recording_mode,
            "mic_single_speaker_verified": options.mic_single_speaker_verified,
            "echo_cancellation": options.echo_cancellation,
            "recording": copy.deepcopy(options.recording_metadata),
        }
    if options.recording_metadata:
        session.provenance["recording"] = copy.deepcopy(options.recording_metadata)


def _validate_profile_inputs(
    mic_path: str | Path | None,
    system_path: str | Path | None,
    profile: AudioProfile | str | None,
    dual_channel: bool,
) -> tuple[AudioProfile, bool]:
    """Valida il profilo prima di iniziare qualsiasi elaborazione costosa.

    Un singolo file stereo può essere un recorder ambientale oppure un bundle
    mic/sistema. La distinzione è semantica e non può essere dedotta dal numero
    di canali, quindi il profilo deve essere esplicito.
    """
    parsed = AudioProfile.parse(profile)
    if parsed is None:
        raise ValueError(
            "Profilo audio obbligatorio: scegli 'Microfono + audio di sistema' "
            "oppure 'Ambientale stereo' prima di trascrivere."
        )
    if mic_path is None and system_path is None:
        raise ValueError("Serve almeno una sorgente audio.")

    if parsed is AudioProfile.STEREO_AMBIENT:
        if dual_channel or (mic_path is not None and system_path is not None):
            raise ValueError(
                "Il profilo 'Ambientale stereo' richiede un solo file sorgente; "
                "non usare due tracce mic/sistema."
            )
        return parsed, False

    # MIC_SYSTEM: due file mono/stereo distinti oppure un unico file stereo
    # dichiarato esplicitamente come bundle mic/sistema.
    if dual_channel:
        if mic_path is None or system_path is not None:
            raise ValueError("Il bundle mic+sistema deve essere un unico file stereo.")
        if audio_utils.channel_count(mic_path) < 2:
            raise ValueError("Il file mic+sistema deve contenere almeno due canali.")
        return parsed, True
    if mic_path is not None and system_path is not None:
        return parsed, False
    if mic_path is not None and audio_utils.channel_count(mic_path) >= 2:
        # Importazione di un unico stereo esplicitamente dichiarato mic+sistema.
        return parsed, True
    raise ValueError(
        "Il profilo 'Microfono + audio di sistema' richiede due sorgenti "
        "oppure un unico file stereo."
    )


def process_recording(
    mic_path: str | Path | None = None,
    system_path: str | Path | None = None,
    *,
    options: PipelineOptions | None = None,
    work_dir: str | Path | None = None,
    session_name: str | None = None,
    progress: ProgressCb | None = None,
    offsets_s: list[float] | None = None,
    dual_channel: bool = False,
) -> Session:
    """Esegue la pipeline completa e restituisce una Session pronta da salvare.

    ``options.audio_profile`` è obbligatorio: la pipeline non presume che un
    file stereo rappresenti mic/sistema, perché molti recorder ambientali
    producono lo stesso layout tecnico.

    Con dual_channel il primo percorso e' un unico file stereo mic/sistema, che
    viene riportato alle due sorgenti separate (gia' allineate).
    """
    opts = options or PipelineOptions()
    if not 0.0 <= float(opts.denoise_strength) <= 1.0:
        raise ValueError("denoise_strength deve essere compreso tra 0 e 1.")
    if opts.quality_retries < 0:
        raise ValueError("quality_retries non può essere negativo.")
    if opts.num_speakers is not None and opts.num_speakers < 1:
        raise ValueError("num_speakers deve essere positivo oppure None.")
    if opts.analysis_mode not in ("safe", "off"):
        raise ValueError("analysis_mode deve essere 'safe' oppure 'off'.")
    if opts.recording_mode not in (None, "ambient", "headset", "speakers"):
        raise ValueError("Modalità di registrazione non valida.")
    profile, dual_channel = _validate_profile_inputs(
        mic_path, system_path, opts.audio_profile, dual_channel
    )
    report = progress or _noop
    report(f"Profilo audio: {profile.label}", 0.01)
    work = Path(work_dir) if work_dir is not None else paths.work_dir()
    work.mkdir(parents=True, exist_ok=True)
    # Immutable per-run media paths preserve old sessions, even when an input
    # file is replaced or two recordings in different folders share a name.
    work = Path(tempfile.mkdtemp(prefix="session-", dir=work))

    recording = mic_path or system_path
    if dual_channel and mic_path is not None:
        mic_path, system_path = audio_utils.split_dual_channel(mic_path, work)
        offsets_s = None

    sources = [p for p in (mic_path, system_path) if p is not None]
    if not sources:
        raise ValueError("Serve almeno una sorgente audio (mic o sistema)")

    device, compute_type = resolve_device(opts.device)
    if opts.compute_type:
        compute_type = opts.compute_type

    # Nome unico per i file intermedi: con un "mixed.wav" fisso, elaborare una
    # nuova registrazione cancellava l'audio delle sessioni gia' salvate.
    stem = Path(sources[0]).stem[:40] or "audio"
    tag = stem

    # 1. audio unico 16 kHz mono, normalizzato (registrazioni basse facevano
    #    scartare il parlato dal VAD di Whisper).
    report("Preparazione audio...", 0.05)
    source_strategy = (
        opts.strategy in ("source_aware", "source_single_speaker")
        or opts.recording_mode in ("headset", "speakers")
        or (opts.recording_mode is not None and mic_path is not None and system_path is not None)
    )
    synchronized = None
    processing_mic, processing_system = mic_path, system_path
    if source_strategy:
        from .capture_sync import synchronize_capture_sources
        synchronized = synchronize_capture_sources(
            mic_path, system_path, offsets_s=offsets_s,
            metadata=opts.recording_metadata, work_dir=work,
        )
        processing_mic, processing_system = synchronized.mic_path, synchronized.system_path
    processing_sources = [path for path in (processing_mic, processing_system) if path is not None]
    channel_hint = (
        profile is AudioProfile.MIC_SYSTEM
        and opts.use_channel_hint
        and mic_path is not None
        and system_path is not None
    )
    dominance_result = [] if channel_hint else None
    mixed = audio_utils.mix_sources(processing_sources, work / f"{tag}_mixed.wav", offsets_s=offsets_s,
                                    dominance_result=dominance_result)
    audio_for_asr: Path = mixed
    audio_for_diar: Path = mixed

    if source_strategy:
        from .source_transcription import transcribe_sources
        utterances, segments, detected_lang, meta = transcribe_sources(
            mic_path=processing_mic,
            system_path=processing_system,
            offsets_s=offsets_s,
            options=opts,
            progress=report,
            work_dir=work,
            original_paths={source: path for source, path in (("mic", mic_path), ("system", system_path)) if path is not None},
            synchronization=synchronized.report,
        )
        stored_meta = _portable_source_metadata(meta)
        session = Session(
            name=session_name or "Sessione",
            language=detected_lang,
            model=opts.model_size,
            audio_profile=profile.value,
            segments=segments,
            utterances=utterances,
            pipeline_strategy=opts.strategy if opts.strategy == "source_single_speaker" else "source_aware",
            capture_manifest={
                **stored_meta,
                "strategy": opts.strategy if opts.strategy == "source_single_speaker" else "source_aware",
                "recording_mode": opts.recording_mode,
                "mic_single_speaker_verified": opts.mic_single_speaker_verified,
                "recording": copy.deepcopy(opts.recording_metadata),
                "headphones_confirmed": getattr(opts, "headphones_confirmed", True),
                "offsets_s": offsets_s or [0.0, 0.0],
                "suppressed_bleed_count": meta.get("suppressed_bleed_count", 0),
            },
        )
        from .voice_profile import enrich_voice_profiles_with_transcript
        dual_profiles = enrich_voice_profiles_with_transcript(
            {},
            session.utterances,
            total_duration=session.duration,
            speaker_names=session.speaker_names,
        )
        session.diarization["voice_profiles"] = {
            spk: p.to_dict() for spk, p in dual_profiles.items()
        }
        if meta.get("alignment"):
            session.provenance["alignment"] = meta["alignment"]
        session.diarization["sources"] = meta.get("diarization", {})
        session.provenance["source_transcription"] = stored_meta
        session.audio.mixed = paths.to_portable(mixed)
        if dual_channel:
            session.audio.mic = paths.to_portable(recording)
        else:
            if mic_path:
                session.audio.mic = paths.to_portable(mic_path)
            if system_path:
                session.audio.system = paths.to_portable(system_path)

        _attach_recording_metadata(session, opts)
        report("Completato", 1.0)
        return session

    # Separazione per parlante: percorso esplicito, mai automatico. Il benchmark
    # §9.05 non ha trovato un modello che batta il percorso standard, quindi se la
    # separazione non riesce si prosegue di sotto invece di degradare il risultato.
    separation_note: dict | None = None
    if getattr(opts, "strategy", "standard") == "speaker_isolated":
        from .speaker_tracks import extract_speaker_tracks, transcribe_speaker_tracks

        report("Separazione delle voci...", 0.2)
        tracks = extract_speaker_tracks(mixed, num_speakers=opts.num_speakers or 2)
        if tracks.separated:
            utterances, segments, detected_lang, meta = transcribe_speaker_tracks(
                tracks, options=opts, progress=report, work_dir=work
            )
            session = Session(
                name=session_name or "Sessione",
                language=detected_lang,
                model=opts.model_size,
                audio_profile=profile.value,
                segments=segments,
                utterances=utterances,
                pipeline_strategy="speaker_isolated",
                capture_manifest=meta,
            )
            if meta.get("alignment"):
                session.provenance["alignment"] = meta["alignment"]
            session.audio.mixed = paths.to_portable(mixed)
            if mic_path:
                session.audio.mic = paths.to_portable(mic_path)
            if system_path:
                session.audio.system = paths.to_portable(system_path)
            _attach_recording_metadata(session, opts)
            report("Completato", 1.0)
            return session
        separation_note = tracks.as_metadata()

    # 2. denoise: serve alla trascrizione, non alla diarizzazione.
    if opts.denoise:
        report("Riduzione rumore...", 0.15)
        from .denoise import denoise_file

        audio_for_asr = denoise_file(
            mixed, work / f"{tag}_denoised.wav", prop_decrease=opts.denoise_strength
        )
        if not opts.diarize_on_raw:
            audio_for_diar = audio_for_asr

    # 3 & 4. Trascrizione, Allineamento Forzato e Diarizzazione / Profilazione
    turns: list[Turn] = []
    overlaps: list[tuple[float, float]] = []
    diarization = None
    voice_profiles: dict = {}
    matched_speaker_names: dict[str, str] = {}
    audio_assets: list[dict] = []
    effective_prompt = opts.initial_prompt
    alignment_report = None
    raw_asr_segments = []

    if opts.diarize and opts.profile_first:
        # FASE A: Diarizzazione e profilazione preventiva delle voci
        report("Diarizzazione e profilazione voci...", 0.20)
        from .diarize import Diarizer
        from .voice_profile import build_voice_profiles

        diarizer = Diarizer(device=device)
        order: list[str] = []
        done: dict[str, float] = {}

        def _on_diar_first(step: str, frac: float | None) -> None:
            if step not in done:
                for prev in order:
                    done[prev] = 1.0
                order.append(step)
            done[step] = frac if frac is not None else done.get(step, 0.0)
            expected = max(len(DIARIZATION_STEPS), len(order))
            overall = min(1.0, sum(done.values()) / expected)
            report(f"Profilazione voci: {step} ({overall:.0%})", 0.20 + 0.25 * overall)

        try:
            diarization = diarizer.diarize_detailed(
                audio_for_diar,
                num_speakers=opts.num_speakers,
                min_speakers=opts.min_speakers,
                max_speakers=opts.max_speakers,
                on_progress=_on_diar_first,
            )
        finally:
            diarizer.close()
            del diarizer

        turns = diarization.exclusive
        overlaps = diarization.overlaps

        # Costruzione dei profili vocali
        voice_profiles = build_voice_profiles(
            diarization.turns,
            speaker_embeddings=diarization.speaker_embeddings,
            channel_dominance=dominance_result[0][0] if dominance_result else None,
            frame_s=dominance_result[0][1] if dominance_result else 0.02,
            known_profiles=opts.known_speaker_profiles,
        )
        matched_speaker_names = {
            spk: p.name for spk, p in voice_profiles.items() if p.name and p.is_known_profile
        }
        if matched_speaker_names:
            names_str = ", ".join(sorted(matched_speaker_names.values()))
            prefix = f"Partecipanti: {names_str}. "
            effective_prompt = (prefix + effective_prompt) if effective_prompt else prefix.strip()

        # FASE B: Se sono presenti più voci, estrai le voci in file separati e trascrivi ciascuna
        speaker_ids = sorted(list({t.speaker for t in diarization.turns if getattr(t, "speaker", None)}))
        if len(speaker_ids) >= 2:
            report("Estrazione delle voci in file audio separati...", 0.45)
            from .speaker_tracks import extract_speaker_tracks, transcribe_speaker_tracks
            from .transcript_reconciler import reconcile_utterances, reconcile_segments

            tracks_result = extract_speaker_tracks(
                audio_for_asr,
                turns=diarization.turns,
                num_speakers=len(speaker_ids),
            )
            if tracks_result.separated:
                utterances, segments, detected_lang, meta = transcribe_speaker_tracks(
                    tracks_result,
                    options=opts,
                    progress=report,
                    work_dir=work,
                )
                utterances = reconcile_utterances(utterances)
                segments = reconcile_segments(segments)

                for spk in speaker_ids:
                    spk_file = work / f"voice_{spk}.wav"
                    if spk_file.is_file():
                        audio_assets.append({
                            "asset_id": f"asset_voice_{spk}_{uuid.uuid4().hex[:6]}",
                            "speaker_id": spk,
                            "path": paths.to_portable(spk_file),
                            "sample_rate": 16000,
                        })

                from .voice_profile import enrich_voice_profiles_with_transcript
                duration_s = audio_utils.audio_info(mixed).duration if mixed else 0.0
                voice_profiles = enrich_voice_profiles_with_transcript(
                    voice_profiles,
                    utterances or segments,
                    total_duration=duration_s,
                    speaker_names=matched_speaker_names,
                )

                session = Session(
                    name=session_name or "Sessione",
                    language=detected_lang,
                    model=opts.model_size,
                    audio_profile=profile.value,
                    segments=segments,
                    utterances=utterances,
                    raw_utterances=copy.deepcopy(utterances),
                    audio_assets=audio_assets,
                    speaker_names=matched_speaker_names,
                    # La strategia e' quella che il chiamante ha chiesto: questo
                    # ramo e' un dettaglio interno del percorso standard, e va
                    # dichiarato nel manifest, non spacciato per una scelta
                    # dell'utente che nessuno ha fatto.
                    pipeline_strategy=getattr(opts, "strategy", "standard"),
                    capture_manifest={
                        **meta,
                        "extraction_method": "turn_slicing",
                    },
                    diarization={
                        "ordinary_turns": len(diarization.turns),
                        "exclusive_turns": len(diarization.exclusive),
                        "assignment": "overlap_aware",
                        "overlaps": [[round(a, 3), round(b, 3)] for a, b in overlaps],
                        "overlap_seconds": round(diarization.overlap_seconds(), 3),
                        "turns": [
                            {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
                            for t in diarization.turns
                        ],
                        "exclusive": [
                            {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
                            for t in diarization.exclusive
                        ],
                        "voice_profiles": {spk: vp.to_dict() for spk, vp in voice_profiles.items()},
                    },
                )
                if meta.get("alignment"):
                    session.provenance["alignment"] = meta["alignment"]
                session.audio.mixed = paths.to_portable(mixed)
                if mic_path:
                    session.audio.mic = paths.to_portable(mic_path)
                if system_path:
                    session.audio.system = paths.to_portable(system_path)

                _attach_recording_metadata(session, opts)
                report("Completato", 1.0)
                return session

        # FASE B (singolo locutore): Trascrizione Whisper diretta
        report("Trascrizione: preparazione modello...", 0.48)
        from .transcribe import Transcriber

        def _on_download_first(name: str, pct: int) -> None:
            report(f"Scaricamento modello {name}: {pct}%", 0.48 + 0.07 * (pct / 100))

        transcriber = Transcriber(
            model_size=opts.model_size,
            device=device,
            compute_type=compute_type,
            batch_size=opts.batch_size,
            on_download=_on_download_first,
        )

        def _on_asr_first(done: float, total: float) -> None:
            dm, ds = divmod(int(done), 60)
            tm, ts = divmod(int(total), 60)
            frac = 0.55 + 0.30 * (done / total if total else 0.0)
            report(f"Trascrizione {dm:02d}:{ds:02d} / {tm:02d}:{ts:02d}", frac)

        try:
            tsegments, detected_lang = transcriber.transcribe(
                audio_for_asr,
                language=opts.language,
                on_progress=_on_asr_first,
                initial_prompt=effective_prompt,
                quality_retries=opts.quality_retries,
            )
        finally:
            transcriber.close()
            del transcriber

        raw_asr_segments = copy.deepcopy(tsegments)
        report("Allineamento forzato timestamp parole...", 0.86)
        tsegments, alignment_report = align_segments(
            tsegments,
            audio_for_asr,
            language=detected_lang or opts.language,
            backend=opts.alignment_backend,
            device=device,
        )

    else:
        # Percorso standard: Trascrizione Whisper full-context -> Allineamento -> Diarizzazione
        report("Trascrizione: preparazione modello...", 0.2)
        from .transcribe import Transcriber

        def _on_download(name: str, pct: int) -> None:
            report(f"Scaricamento modello {name}: {pct}%", 0.20 + 0.10 * (pct / 100))

        transcriber = Transcriber(
            model_size=opts.model_size,
            device=device,
            compute_type=compute_type,
            batch_size=opts.batch_size,
            on_download=_on_download,
        )

        def _on_asr(done: float, total: float) -> None:
            dm, ds = divmod(int(done), 60)
            tm, ts = divmod(int(total), 60)
            frac = 0.30 + 0.35 * (done / total if total else 0.0)
            report(f"Trascrizione {dm:02d}:{ds:02d} / {tm:02d}:{ts:02d}", frac)

        try:
            tsegments, detected_lang = transcriber.transcribe(
                audio_for_asr,
                language=opts.language,
                on_progress=_on_asr,
                initial_prompt=opts.initial_prompt,
                quality_retries=opts.quality_retries,
            )
        finally:
            transcriber.close()
            del transcriber

        raw_asr_segments = copy.deepcopy(tsegments)
        report("Allineamento forzato timestamp parole...", 0.66)
        tsegments, alignment_report = align_segments(
            tsegments,
            audio_for_asr,
            language=detected_lang or opts.language,
            backend=opts.alignment_backend,
            device=device,
        )

        if opts.diarize:
            report("Diarizzazione (chi parla quando)...", 0.70)
            from .diarize import Diarizer
            from .voice_profile import build_voice_profiles

            diarizer = Diarizer(device=device)
            order = []
            done = {}

            def _on_diar(step: str, frac: float | None) -> None:
                if step not in done:
                    for prev in order:
                        done[prev] = 1.0
                    order.append(step)
                done[step] = frac if frac is not None else done.get(step, 0.0)
                expected = max(len(DIARIZATION_STEPS), len(order))
                overall = min(1.0, sum(done.values()) / expected)
                report(f"Diarizzazione: {step} ({overall:.0%})", 0.70 + 0.18 * overall)

            try:
                diarization = diarizer.diarize_detailed(
                    audio_for_diar,
                    num_speakers=opts.num_speakers,
                    min_speakers=opts.min_speakers,
                    max_speakers=opts.max_speakers,
                    on_progress=_on_diar,
                )
            finally:
                diarizer.close()
                del diarizer

            turns = diarization.exclusive
            overlaps = diarization.overlaps
            voice_profiles = build_voice_profiles(
                diarization.turns,
                speaker_embeddings=diarization.speaker_embeddings,
                channel_dominance=dominance_result[0][0] if dominance_result else None,
                frame_s=dominance_result[0][1] if dominance_result else 0.02,
                known_profiles=opts.known_speaker_profiles,
            )
            matched_speaker_names = {
                spk: p.name for spk, p in voice_profiles.items() if p.name and p.is_known_profile
            }

    # 5. Merge parola -> voce, guidato dai profili vocali e autocorrezione
    report("Unione trascrizione e voci...", 0.88)
    units = merge.flatten_words(tsegments)
    preserve_short_turns = (
        (opts.num_speakers is not None and opts.num_speakers >= 3)
        or len(voice_profiles) >= 3
    )
    labels = merge.assign_speakers(
        units, turns,
        preserve_short_turns=preserve_short_turns,
        voice_profiles=voice_profiles,
        overlaps=overlaps,
    )

    # 5b. Correzione con il canale di provenienza se presente
    if channel_hint:
        report("Attribuzione voci per canale (mic/sistema)...", 0.94)
        try:
            dominance, frame_s = dominance_result[0]
            labels = merge.apply_channel_hint(
                units, labels, dominance, frame_s, {0: "IO", 1: "REMOTO"}
            )
        except Exception:
            logging.getLogger(__name__).exception("Attribuzione per canale non riuscita")
            raise

    autocorrect_active = opts.autocorrect_truncation or opts.autocorrect_overlap
    raw_segments: list[Segment] = merge.build_segments(
        units, labels,
        voice_profiles=voice_profiles,
        overlaps=overlaps,
        autocorrect_enabled=autocorrect_active,
    )
    segments = raw_segments
    transcript_review: dict | None = None
    if opts.analysis_mode == "safe":
        report("Controllo qualità trascrizione e attribuzione...", 0.96)
        result = review_labels(
            units,
            turns,
            labels,
            profile=profile,
            preserve_short_turns=preserve_short_turns,
            speaker_count_known=opts.num_speakers is not None or bool(voice_profiles),
        )
        segments = result.segments
        transcript_review = result.metadata

        # Assicura autocorrezione coerente anche sui segmenti revisionati
        if autocorrect_active and segments:
            if opts.autocorrect_truncation:
                from .autocorrect import autocorrect_truncations
                segments = autocorrect_truncations(segments)
            if opts.autocorrect_overlap and overlaps:
                from .autocorrect import autocorrect_overlaps
                segments = autocorrect_overlaps(segments, overlaps=overlaps, profiles=voice_profiles)

    ordinary_turns = list(diarization.turns) if diarization is not None else []
    exclusive_turns = list(diarization.exclusive) if diarization is not None else []
    if diarization is not None and not channel_hint:
        from .global_diarization import reconstruct_events
        utterances = reconstruct_events(tsegments, ordinary_turns, exclusive_turns,
            source_audio_ref=paths.to_portable(mixed) if mixed else None, asr_model=opts.model_size)
    else:
        utterances = merge.build_utterances_from_segments(
            segments,
            overlaps=overlaps,
            ordinary_turns=ordinary_turns,
            exclusive_turns=exclusive_turns,
            source_audio_ref=paths.to_portable(mixed) if mixed else None,
            asr_model=opts.model_size,
        )
    # Le utterance materializzate (incluse quelle simultanee recuperate dai
    # turni non esclusivi) sono la fonte autorevole: i segmenti ne sono la proiezione.
    segments = merge.utterances_to_segments(utterances)

    session = Session(
        name=session_name or "Sessione",
        language=detected_lang,
        model=opts.model_size,
        audio_profile=profile.value,
        segments=segments,
        raw_segments=copy.deepcopy(raw_segments) if transcript_review is not None else None,
        transcript_review=transcript_review,
        speaker_names=matched_speaker_names,
        utterances=utterances,
        raw_utterances=copy.deepcopy(utterances),
        pipeline_strategy=getattr(opts, "strategy", "standard"),
        capture_manifest=separation_note,
        schema_version=6,
        media_duration_s=audio_utils.audio_info(mixed).duration if mixed else 0.,
    )
    if diarization is not None:
        session.diarization = {
            "ordinary_turns": len(diarization.turns),
            "exclusive_turns": len(diarization.exclusive),
            # L'esclusiva resta un ausilio d'allineamento; la sovrapposizione
            # e' descritta dai turni ordinari, conservati integralmente.
            "assignment": "exclusive_aligned_overlap_aware",
            "overlaps": [[round(a, 3), round(b, 3)] for a, b in overlaps],
            "overlap_seconds": round(diarization.overlap_seconds(), 3),
            "turns": [
                {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
                for t in diarization.turns
            ],
            "exclusive": [
                {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
                for t in diarization.exclusive
            ],
        }
        if diarization.speaker_embeddings:
            session.diarization["speaker_embeddings"] = diarization.speaker_embeddings
            session.diarization["embedding_model"] = {
                "repo": diarization.model_repo,
                "revision": diarization.model_revision,
            }
        if voice_profiles:
            from .voice_profile import enrich_voice_profiles_with_transcript
            voice_profiles = enrich_voice_profiles_with_transcript(
                voice_profiles,
                session.utterances or session.segments,
                total_duration=session.duration,
                speaker_names=matched_speaker_names,
            )
            session.diarization["voice_profiles"] = {
                spk: p.to_dict() for spk, p in voice_profiles.items()
            }
    if alignment_report is not None:
        session.provenance["alignment"] = alignment_report.to_dict()
    session.audio.mixed = paths.to_portable(mixed)
    if dual_channel:
        # le due tracce sono intermedi in work/: la sessione punta alla registrazione
        session.audio.mic = paths.to_portable(recording)
    else:
        if mic_path:
            session.audio.mic = paths.to_portable(mic_path)
        if system_path:
            session.audio.system = paths.to_portable(system_path)

    # These artifacts retain pre-assignment observations. Assigned words are
    # never presented as if they were raw recognition or independent VAD.
    from .global_workflow import write_json
    intermediate = {
        "raw_asr": {"segments": [{"start": s.start, "end": s.end, "text": s.text, "words": list(s.words or [])} for s in raw_asr_segments]},
        "alignment": {"segments": [{"start": s.start, "end": s.end, "text": s.text, "words": list(s.words or [])} for s in tsegments]},
        "diarization": session.diarization,
        "assignment": {"words": [{"start": w.start, "end": w.end, "text": w.text, "speaker": u.speaker_id}
                                   for u in session.utterances for w in u.words]},
    }
    references = {}
    for stage, payload in intermediate.items():
        artifact = work / f"{stage}.json"
        write_json(artifact, payload)
        references[stage] = paths.to_portable(artifact)
    session.provenance["intermediate_outputs"] = references
    session.provenance["diarization_scope"] = "entire_recording"
    session.provenance["reconstruction_policy"] = "preserve_words_and_unresolved_activity"
    _attach_recording_metadata(session, opts)
    report("Completato", 1.0)
    return session
