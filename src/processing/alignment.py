"""Allineamento forzato e validazione dei timestamp per parola (Fase 4).

Questo modulo costituisce il confine architetturale tra la trascrizione ASR
(faster-whisper) e l'assegnazione speaker-parola (merge.py):

1. Supporta l'allineamento forzato acustico CTC tramite ``torchaudio.pipelines.MMS_FA``
   quando i pesi sono disponibili localmente (o scaricabili su richiesta esplicita)
   e la lingua usa alfabeto latino compatibile col dizionario MMS_FA.
2. Valida e normalizza rigorosamente i timestamp parola per parola (provenienti da
   MMS_FA o dall'attenzione incrociata DTW di faster-whisper):
   - garantisce che i tempi siano finiti, non negativi e monotoni non decrescenti;
   - vincola ciascuna parola all'interno dell'intervallo ``[segment.start, segment.end]``;
   - NON inventa né interpola mai timestamp per parole prive di allineamento reale
     (un segmento privo di timestamp parola mantiene ``words=[]`` così che a valle
     ``Segment.words_valid`` resti ``False``).
3. Rilascia immediatamente eventuale memoria GPU allocata dall'allineatore prima
   dello stadio di diarizzazione.
"""

from __future__ import annotations

import gc
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .transcribe import TSegment


@dataclass
class AlignmentReport:
    """Metadati strutturati di provenienza per lo stadio di forced alignment."""

    backend: str = "faster-whisper-cross-attention"
    status: str = "validated_cross_attention"
    language: str | None = None
    total_segments: int = 0
    aligned_segments: int = 0
    unaligned_segments: int = 0
    total_words: int = 0
    aligned_words: int = 0
    monotonic_repairs: int = 0
    repaired_words: int = 0
    clamped_words: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def validate_word_timestamps(
    segments: list[TSegment],
    *,
    language: str | None = None,
    backend: str = "faster-whisper-cross-attention",
    notes: list[str] | None = None,
) -> tuple[list[TSegment], AlignmentReport]:
    """Valida e normalizza i timestamp per parola senza mai fabbricare tempi mancanti."""
    if not segments:
        return [], AlignmentReport(
            backend=backend,
            status="empty",
            language=language,
            notes=list(notes or []),
        )

    cleaned_segments: list[TSegment] = []
    aligned_count = 0
    unaligned_count = 0
    total_words = 0
    monotonic_repairs = 0
    clamped_words = 0

    for seg in segments:
        s_start = _valid_float(seg.start)
        s_end = _valid_float(seg.end)
        if s_start is None or s_end is None or s_start < 0.0 or s_end < s_start:
            continue
        text = str(seg.text or "")
        if not text.strip():
            continue

        raw_words = list(seg.words or [])
        if not raw_words:
            # Regola fondamentale: non fabbricare mai timestamp mancanti.
            unaligned_count += 1
            cleaned_segments.append(
                TSegment(
                    start=s_start,
                    end=s_end,
                    text=text,
                    words=[],
                    avg_logprob=getattr(seg, "avg_logprob", 0.0),
                    compression_ratio=getattr(seg, "compression_ratio", 0.0),
                    no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
                )
            )
            continue

        norm_words: list[tuple[float, float, str, float]] = []
        cursor = s_start
        for item in raw_words:
            if len(item) < 3:
                continue
            w_start = _valid_float(item[0])
            w_end = _valid_float(item[1])
            w_text = str(item[2] or "")
            w_prob = _valid_float(item[3]) if len(item) > 3 else 1.0
            if w_start is None or w_end is None or not w_text.strip():
                continue
            if w_prob is None:
                w_prob = 0.0
            w_prob = max(0.0, min(1.0, w_prob))

            clamped = False
            if w_start < s_start:
                w_start = s_start
                clamped = True
            if w_end > s_end:
                w_end = s_end
                clamped = True
            if w_start > s_end:
                w_start = s_end
                clamped = True
            if clamped:
                clamped_words += 1

            repaired = False
            if w_start < cursor - 1e-4:
                w_start = cursor
                repaired = True
            if w_end < w_start:
                w_end = w_start
                repaired = True
            if repaired:
                monotonic_repairs += 1

            cursor = w_end
            norm_words.append((round(w_start, 3), round(w_end, 3), w_text, round(w_prob, 4)))

        if norm_words:
            aligned_count += 1
            total_words += len(norm_words)
            eff_start = min(s_start, norm_words[0][0])
            eff_end = max(s_end, norm_words[-1][1])
        else:
            unaligned_count += 1
            eff_start = s_start
            eff_end = s_end

        cleaned_segments.append(
            TSegment(
                start=round(eff_start, 3),
                end=round(eff_end, 3),
                text=text,
                words=norm_words,
                avg_logprob=getattr(seg, "avg_logprob", 0.0),
                compression_ratio=getattr(seg, "compression_ratio", 0.0),
                no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
            )
        )

    status = "aligned" if backend == "torchaudio_mms_fa" else "validated_cross_attention"
    if unaligned_count > 0 and aligned_count > 0:
        status = "partial_unaligned"
    elif aligned_count == 0 and unaligned_count > 0:
        status = "unaligned"

    report = AlignmentReport(
        backend=backend,
        status=status,
        language=language,
        total_segments=len(cleaned_segments),
        aligned_segments=aligned_count,
        unaligned_segments=unaligned_count,
        total_words=total_words,
        aligned_words=total_words,
        monotonic_repairs=monotonic_repairs,
        repaired_words=monotonic_repairs + clamped_words,
        clamped_words=clamped_words,
        notes=list(notes or []),
    )
    return cleaned_segments, report


def _mms_fa_cached_locally() -> bool:
    """Verifica se i pesi di torchaudio MMS_FA sono gia' presenti nella cache locale."""
    try:
        import torch

        hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
        if hub_dir.is_dir() and (any(hub_dir.glob("*mms*")) or any(hub_dir.glob("*align*"))):
            return True
    except Exception:
        pass
    return False


_mms_fa_weights_cached = _mms_fa_cached_locally


def _align_with_mms_fa(
    segments: list[TSegment],
    audio_path: str | Path,
    *,
    device: str = "cpu",
) -> list[TSegment]:
    """Esegue forced alignment acustico CTC con torchaudio.pipelines.MMS_FA."""
    import torch
    import torchaudio
    from . import audio_utils

    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels(star=None)
    dictionary = bundle.get_dict(star=None)
    valid_chars = set(dictionary.keys())

    model = bundle.get_model(with_star=False)
    target_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(target_device)
    model.eval()

    try:
        waveform_np = audio_utils.load_mono_16k(audio_path)
        sr = int(bundle.sample_rate)
        out_segments: list[TSegment] = []

        for seg in segments:
            s_idx = max(0, int(round(seg.start * sr)))
            e_idx = min(len(waveform_np), int(round(seg.end * sr)))
            if e_idx - s_idx < int(0.08 * sr):
                out_segments.append(seg)
                continue

            raw_words = [w for w in re.split(r"\s+", seg.text.strip()) if w]
            if not raw_words:
                out_segments.append(seg)
                continue

            norm_tokens: list[str] = []
            orig_words: list[str] = []
            for w in raw_words:
                cleaned = "".join(ch for ch in w.lower() if ch in valid_chars)
                if cleaned:
                    norm_tokens.append(cleaned)
                    orig_words.append(w)

            if not norm_tokens or len(norm_tokens) != len(raw_words):
                # Se il segmento contiene cifre o simboli fuori dizionario, conserva
                # i timestamp cross-attention di Whisper per non perdere parole.
                out_segments.append(seg)
                continue

            transcript_tokens = [dictionary[c] for word in norm_tokens for c in word]
            if not transcript_tokens:
                out_segments.append(seg)
                continue

            chunk = torch.from_numpy(waveform_np[s_idx:e_idx]).unsqueeze(0).to(target_device)
            with torch.inference_mode():
                emission, _ = model(chunk)
                token_spans = torchaudio.functional.forced_align(
                    emission.cpu(),
                    torch.tensor([transcript_tokens], dtype=torch.int32),
                    blank=0,
                )
            # Ricostruisci i confini parola dagli span carattere
            frame_dur = float(e_idx - s_idx) / float(sr) / float(max(1, emission.size(1)))
            spans = token_spans[0]
            aligned_words: list[tuple[float, float, str, float]] = []
            # Raggruppa frame contigui appartenenti a ciascuna parola
            # usando gli indici non-blank
            non_blank_frames = [idx for idx, tok in enumerate(spans.tolist()) if tok != 0]
            if len(non_blank_frames) < len(transcript_tokens):
                out_segments.append(seg)
                continue

            # Usa torchaudio merge_tokens per ottenere gli span per carattere
            char_spans = torchaudio.functional.merge_tokens(spans, emission.cpu()[0].exp())
            if len(char_spans) != len(transcript_tokens):
                out_segments.append(seg)
                continue

            char_cursor = 0
            for orig_w, norm_w in zip(orig_words, norm_tokens):
                w_spans = char_spans[char_cursor : char_cursor + len(norm_w)]
                char_cursor += len(norm_w)
                w_start = seg.start + w_spans[0].start * frame_dur
                w_end = seg.start + w_spans[-1].end * frame_dur
                w_score = sum(float(sp.score) for sp in w_spans) / max(1, len(w_spans))
                aligned_words.append((
                    round(w_start, 3),
                    round(max(w_start, w_end), 3),
                    " " + orig_w if aligned_words else orig_w,
                    round(max(0.0, min(1.0, w_score)), 4),
                ))

            out_segments.append(
                TSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=aligned_words,
                    avg_logprob=getattr(seg, "avg_logprob", 0.0),
                    compression_ratio=getattr(seg, "compression_ratio", 0.0),
                    no_speech_prob=getattr(seg, "no_speech_prob", 0.0),
                )
            )
        return out_segments
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def align_segments(
    segments: list[TSegment],
    audio_path: str | Path | None = None,
    *,
    language: str | None = None,
    backend: str = "auto",
    device: str = "cpu",
    use_mms_fa: bool = False,
    allow_download: bool = False,
) -> tuple[list[TSegment], AlignmentReport]:
    """Esegue lo stadio di allineamento forzato e validazione dei timestamp parola."""
    notes: list[str] = []
    effective_backend = "faster-whisper-cross-attention"
    working = list(segments or [])

    want_mms_fa = use_mms_fa or backend in ("mms_fa", "torchaudio_mms_fa") or (
        backend == "auto" and _mms_fa_cached_locally()
    )

    if want_mms_fa and audio_path is not None and Path(audio_path).is_file():
        if allow_download or _mms_fa_cached_locally():
            try:
                working = _align_with_mms_fa(working, audio_path, device=device)
                effective_backend = "torchaudio_mms_fa"
            except Exception as exc:
                notes.append(f"fallback_whisper_cross_attention: {type(exc).__name__}")
                effective_backend = "faster-whisper-cross-attention"
        else:
            notes.append("mms_fa_not_cached_offline_using_whisper_cross_attention")

    return validate_word_timestamps(
        working,
        language=language,
        backend=effective_backend,
        notes=notes,
    )

