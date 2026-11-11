"""Trascrizione con faster-whisper (Whisper ottimizzato, locale, GPU).

Nota Windows: faster-whisper usa CTranslate2, che a runtime carica le DLL di
cuDNN/cuBLAS per CUDA. Su Windows non le trova da solo; qui le rendiamo
caricabili aggiungendo al percorso di ricerca DLL la cartella `torch/lib`
(le build CUDA di PyTorch includono gia' cuDNN 9 e cuBLAS 12 compatibili) e
le eventuali cartelle dei pacchetti `nvidia-*`.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..bootstrap import retain_dll_directory
from .hf_progress import make_progress_tqdm


def _ensure_cuda_libs() -> None:
    """Aggiunge al percorso di ricerca DLL le librerie CUDA gia' installate da torch."""
    if not sys.platform.startswith("win"):
        return
    candidates: list[Path] = []
    try:
        import torch  # le DLL CUDA sono in <torch>/lib

        candidates.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass
    # Pacchetti nvidia-* (cudnn/cublas) se installati separatamente
    for entry in sys.path:
        nv = Path(entry) / "nvidia"
        if nv.is_dir():
            candidates.extend(p for p in nv.glob("*/bin") if p.is_dir())
    for d in candidates:
        if d.is_dir():
            retain_dll_directory(d)


def _ensure_model_path(model_size: str, on_download: Callable[[str, int], None] | None,
                       *, local_only: bool = False):
    """Scarica il modello (se non in cache) mostrando il progresso; ritorna path/id.

    Se il modello e' gia' in cache, snapshot_download non scarica nulla e ritorna
    subito. Se model_size non e' una taglia nota (id custom o path locale), lascia
    gestire il download a WhisperModel.
    """
    if Path(model_size).is_dir():
        return model_size
    try:
        from faster_whisper.utils import _MODELS  # mappa taglia -> repo HF
    except Exception:
        if local_only:
            raise RuntimeError("Impossibile risolvere il modello in modalita offline")
        return model_size
    repo = _MODELS.get(model_size, model_size if "/" in model_size else None)
    if repo is None:
        if local_only:
            raise RuntimeError(f"Modello locale sconosciuto: {model_size}")
        return model_size
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    patterns = ["config.json", "preprocessor_config.json", "model.bin",
                "tokenizer.json", "vocabulary.*"]
    try:
        cached = Path(snapshot_download(repo, local_files_only=True, allow_patterns=patterns))
        required = [cached / name for name in ("config.json", "model.bin", "tokenizer.json")]
        required += list(cached.glob("vocabulary.*"))
        if len(required) >= 4 and all(path.is_file() and path.stat().st_size > 0 for path in required):
            return str(cached)
    except LocalEntryNotFoundError:
        pass
    if local_only:
        raise RuntimeError(f"Modello {model_size} non disponibile completo in cache; nessun download eseguito.")
    kwargs = {"allow_patterns": patterns}
    if on_download is not None:
        kwargs["tqdm_class"] = make_progress_tqdm(on_download)
    return snapshot_download(repo, **kwargs)


@dataclass
class TSegment:
    start: float
    end: float
    text: str
    words: list[tuple[float, float, str, float]]  # (start, end, word, prob)
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


def _repetition_count(text: str) -> int:
    """Conteggio leggero di ripetizioni consecutive o di bigrammi duplicati."""
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    adjacent = sum(a == b for a, b in zip(tokens, tokens[1:]))
    bigrams = list(zip(tokens, tokens[1:]))
    duplicated = sum(a == b for a, b in zip(bigrams, bigrams[1:]))
    return adjacent + duplicated


def _quality_score(segment: TSegment) -> float:
    """Punteggio comparabile tra prima passata e retry."""
    score = segment.avg_logprob
    if segment.compression_ratio > 2.4:
        score -= (segment.compression_ratio - 2.4) * 0.5
    score -= _repetition_count(segment.text) * 0.45
    if len(segment.text.split()) >= 3 and segment.no_speech_prob > 0.7:
        score -= (segment.no_speech_prob - 0.7) * 0.5
    return score


def _suspect(segment: TSegment) -> bool:
    """Individua segmenti da ritentare (inclusi casi brevi con alta incertezza)."""
    words_count = len(segment.text.split())
    if words_count == 0:
        return True
    if words_count < 3:
        return (
            float(getattr(segment, "no_speech_prob", 0.0)) > 0.65
            or float(getattr(segment, "avg_logprob", 0.0)) < -1.1
            or _repetition_count(segment.text) >= 2
        )
    return (
        float(getattr(segment, "avg_logprob", 0.0)) < -1.0
        or float(getattr(segment, "compression_ratio", 0.0)) > 2.6
        or _repetition_count(segment.text) >= 2
    )


def _stitch_retry_segments(
    candidates,
    original: TSegment,
    *,
    expand_margin: float = 0.0,
) -> TSegment | None:
    """Stitch a retry without duplicating context, allowing controlled boundary expansion."""
    accepted = [seg for seg in candidates
                if seg.end >= (original.start - 0.2 - expand_margin)
                and seg.start <= (original.end + 0.2 + expand_margin)]
    if not accepted:
        return None
    words = []
    seen = set()
    metrics = []
    for seg in accepted:
        metrics.append((float(getattr(seg, "avg_logprob", 0.0)),
                        float(getattr(seg, "compression_ratio", 0.0)),
                        float(getattr(seg, "no_speech_prob", 0.0))))
        for word in seg.words or []:
            start, end = float(word.start), float(word.end)
            # A retry is decoded with padding around the suspect segment.
            # Midpoint is checked against the window (expanded by expand_margin when allowed).
            midpoint = (start + end) / 2.0
            if midpoint < (original.start - expand_margin) or midpoint > (original.end + expand_margin):
                continue
            key = (round(start, 3), round(end, 3), word.word.strip().casefold())
            if key not in seen:
                seen.add(key)
                words.append((start, end, word.word, float(word.probability)))
    words.sort(key=lambda item: (item[0], item[1]))
    if words:
        pieces: list[str] = []
        for _start, _end, raw, _prob in words:
            token = str(raw).strip()
            if not token:
                continue
            if pieces and token[0] in ",.;:!?)]}%":
                pieces[-1] += token
            else:
                pieces.append(token)
        text = " ".join(pieces).strip()
        start = max(original.start - 0.2 - expand_margin, words[0][0])
        end = min(original.end + 0.2 + expand_margin, words[-1][1])
    else:
        # Without word timings we cannot safely cut a padded segment.  Only
        # choose a retry whose centre is in the source window (or that covers
        # its centre); context-only candidates are discarded entirely.
        centre = (original.start + original.end) / 2.0
        fallback = [seg for seg in accepted
                    if (original.start - expand_margin) <= (float(seg.start) + float(seg.end)) / 2.0 <= (original.end + expand_margin)
                    or float(seg.start) <= centre <= float(seg.end)]
        if not fallback:
            return None
        best = max(fallback, key=lambda seg: _quality_score(TSegment(
            float(seg.start), float(seg.end), str(seg.text), [],
            float(getattr(seg, "avg_logprob", 0.0)),
            float(getattr(seg, "compression_ratio", 0.0)),
            float(getattr(seg, "no_speech_prob", 0.0)))))
        text = str(best.text).strip()
        if not text:
            return None
        start = max(original.start - 0.2 - expand_margin, float(best.start))
        end = min(original.end + 0.2 + expand_margin, float(best.end))
    if not text or end <= start:
        return None
    average = tuple(sum(item[i] for item in metrics) / len(metrics) for i in range(3))
    return TSegment(start=start, end=end, text=text, words=words,
                    avg_logprob=average[0], compression_ratio=average[1],
                    no_speech_prob=average[2])


class Transcriber:
    """Wrapper su faster-whisper. Carica il modello una volta e lo riusa."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        on_download: Callable[[str, int], None] | None = None,
        batch_size: int = 1,
    ):
        from ..bootstrap import install_system_certs

        install_system_certs()  # il primo load scarica il modello da HuggingFace
        _ensure_cuda_libs()
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = max(1, int(batch_size or 1))
        self._batched_model = None
        # Scarica il modello (con progresso) se non e' gia' in cache.
        model_ref = _ensure_model_path(model_size, on_download)
        if device == "cuda":
            candidates = []
            for ct in (compute_type, "int8_float16", "float16", "int8"):
                if ct and ct not in candidates:
                    candidates.append(ct)
            last_exc: Exception | None = None
            for ct in candidates:
                try:
                    self._model = WhisperModel(model_ref, device="cuda", compute_type=ct)
                    self.compute_type = ct
                    last_exc = None
                    break
                except ValueError as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
        else:
            cpu_ct = compute_type if compute_type in ("int8", "float32", "int8_float32") else "int8"
            self.compute_type = cpu_ct
            self._model = WhisperModel(model_ref, device=device, compute_type=cpu_ct)

    def close(self) -> None:
        """Rilascia esplicitamente il modello CTranslate2 e svuota la cache CUDA."""
        import gc

        self._batched_model = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    unload = close

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        vad_filter: bool = True,
        on_progress: Callable[[float, float], None] | None = None,
        initial_prompt: str | None = None,
        quality_retries: int = 24,
        batch_size: int | None = None,
    ) -> tuple[list[TSegment], str]:
        """Trascrive un file audio.

        Ritorna (segmenti, lingua_rilevata). I timestamp sono per-parola, utili
        sia per la GUI sincronizzata sia per allineare la diarizzazione.

        Scelte dei parametri (erano i default, e causavano bug reali):
        - condition_on_previous_text=False: con True Whisper si aggancia al testo
          gia' prodotto ed entra in loop di ripetizione ("Kenia. Kenia Alvarcos.
          Kenia. Kenia Alvarcos."), oppure trascina errori in avanti.
        - vad_parameters: il default (min_silence 2000 ms, pad 400 ms, soglia 0.5)
          taglia il parlato basso o le pause brevi. Soglia piu' bassa, silenzio
          minimo piu' corto e piu' padding = molto meno parlato perso.
        - temperature a scalini + compression_ratio_threshold: se un blocco esce
          degenere (ripetuto), viene ritentato con temperatura piu' alta.
        - hallucination_silence_threshold=5.0 (era 2.0): a 2.0 buttava via parlato
          vero. Misurato su 10 minuti di lezione, parlato non trascritto: 19.6%
          a 2.0, 12.4% a 5.0, 7.9% disattivandolo. Disattivarlo pero' riporta i
          frammenti ("Un...", "Una..."): 12 contro 0. A 5.0 si recupera il blocco
          lungo che mancava e non compaiono frammenti (ed e' il 40% piu' veloce).

        on_progress(secondi_trascritti, secondi_totali) viene chiamato dopo ogni
        segmento: faster-whisper e' un generatore, quindi il tempo raggiunto
        (seg.end) rispetto alla durata totale e' un avanzamento reale.
        """
        vad_parameters = {
            "threshold": 0.35,             # default 0.5: troppo severo su voci basse
            "neg_threshold": 0.25,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 400,  # default 2000: fondeva/tagliava turni
            "speech_pad_ms": 600,            # non tronca inizio/fine parola
        }
        effective_batch = max(1, int(batch_size if batch_size is not None else self.batch_size))
        common_kwargs = dict(
            language=language,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.05,
            no_repeat_ngram_size=0,
            hallucination_silence_threshold=5.0,
            initial_prompt=initial_prompt,
        )
        if effective_batch > 1 and vad_filter:
            from faster_whisper import BatchedInferencePipeline

            if self._batched_model is None:
                self._batched_model = BatchedInferencePipeline(model=self._model)
            segments, info = self._batched_model.transcribe(
                str(audio_path),
                batch_size=effective_batch,
                **common_kwargs,
            )
        else:
            segments, info = self._model.transcribe(
                str(audio_path),
                **common_kwargs,
            )
        total = float(info.duration or 0.0)
        out: list[TSegment] = []
        for seg in segments:  # generatore: l'iterazione esegue la trascrizione
            words = [
                (w.start, w.end, w.word, w.probability)
                for w in (seg.words or [])
            ]
            out.append(TSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
                compression_ratio=float(getattr(seg, "compression_ratio", 0.0)),
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
            ))
            if on_progress is not None and total > 0:
                on_progress(min(seg.end, total), total)

        # Retry mirato: una seconda decodifica dell'intero file è costosa e
        # tende a cambiare anche segmenti già buoni. Si ritentano solo i casi
        # degeneri, entro un limite stabile, con un po' di contesto acustico.
        retries_left = max(0, int(quality_retries))
        if retries_left and out:
            for index, original in sorted(
                ((i, seg) for i, seg in enumerate(out) if _suspect(seg)),
                key=lambda item: _quality_score(item[1]),
            )[:retries_left]:
                retry = self._retry_segment(
                    audio_path,
                    original,
                    language=language or info.language,
                    total=total,
                    initial_prompt=initial_prompt,
                )
                if retry is not None and _quality_score(retry) > _quality_score(original) + 0.05:
                    out[index] = retry
                retries_left -= 1

        # 2. Recupero omissioni complete da copertura vocale indipendente (Piano §6.C)
        if retries_left > 0 and total > 0.5:
            try:
                import numpy as np
                from . import audio_utils
                from .coverage import (
                    detect_speech_activity,
                    find_uncovered_regions,
                    plan_distributed_recovery,
                    is_hallucination_or_noise,
                )

                audio_mono = audio_utils.load_mono_16k(audio_path)
                speech_regions = detect_speech_activity(audio_mono, sr=16000)
                uncovered = find_uncovered_regions(speech_regions, out, min_omission_s=0.4)
                if uncovered:
                    existing_spans = [(float(s.start), float(s.end)) for s in out]
                    planned_gaps = plan_distributed_recovery(uncovered, total, max_retries=min(6, retries_left))
                    for start_pad, end_pad, _reg in planned_gaps:
                        try:
                            gap_segs, _ = self._model.transcribe(
                                str(audio_path),
                                language=language or info.language,
                                clip_timestamps=[start_pad, end_pad],
                                vad_filter=False,
                                word_timestamps=True,
                                beam_size=5,
                                initial_prompt=initial_prompt,
                            )
                            for gseg in gap_segs:
                                raw_gwords = list(gseg.words or [])
                                if raw_gwords:
                                    filtered_gwords = []
                                    for w in raw_gwords:
                                        w_mid = 0.5 * (float(w.start) + float(w.end))
                                        if any(es0 + 0.04 <= w_mid <= es1 - 0.04 for es0, es1 in existing_spans):
                                            continue
                                        filtered_gwords.append((float(w.start), float(w.end), w.word, float(w.probability)))
                                    if not filtered_gwords:
                                        continue
                                    gwords = filtered_gwords
                                    gstart = gwords[0][0]
                                    gend = gwords[-1][1]
                                    gtext = " ".join(str(w[2]).strip() for w in gwords if str(w[2]).strip()).strip()
                                else:
                                    gstart = float(gseg.start)
                                    gend = float(gseg.end)
                                    g_mid = 0.5 * (gstart + gend)
                                    if any(es0 + 0.04 <= g_mid <= es1 - 0.04 for es0, es1 in existing_spans):
                                        continue
                                    gtext = str(gseg.text or "").strip()
                                    gwords = []
                                if not gtext:
                                    continue
                                s_idx = int(round(max(0.0, gstart) * 16000))
                                e_idx = int(round(min(total, gend) * 16000))
                                audio_chunk = audio_mono[s_idx:e_idx] if e_idx > s_idx else np.array([])
                                if is_hallucination_or_noise(
                                    gtext,
                                    audio_chunk,
                                    compression_ratio=float(getattr(gseg, "compression_ratio", 1.0)),
                                    no_speech_prob=float(getattr(gseg, "no_speech_prob", 0.0)),
                                ):
                                    continue
                                out.append(
                                    TSegment(
                                        start=gstart,
                                        end=gend,
                                        text=gtext,
                                        words=gwords,
                                        avg_logprob=float(getattr(gseg, "avg_logprob", 0.0)),
                                        compression_ratio=float(getattr(gseg, "compression_ratio", 0.0)),
                                        no_speech_prob=float(getattr(gseg, "no_speech_prob", 0.0)),
                                    )
                                )
                                existing_spans.append((gstart, gend))
                                retries_left -= 1
                                if retries_left <= 0:
                                    break
                        except Exception:
                            pass
                        if retries_left <= 0:
                            break
                    out.sort(key=lambda s: (s.start, s.end))
            except Exception:
                pass

        return out, info.language

    def _retry_segment(
        self,
        audio_path: str | Path,
        original: TSegment,
        *,
        language: str | None,
        total: float,
        initial_prompt: str | None,
    ) -> TSegment | None:
        """Riconosce un segmento sospetto con contesto e decoding più stabile."""
        pad = 0.75
        start = max(0.0, original.start - pad)
        end = min(total, original.end + pad) if total else original.end + pad
        try:
            retry_segments, _ = self._model.transcribe(
                str(audio_path),
                language=language,
                clip_timestamps=[start, end],
                vad_filter=False,
                word_timestamps=True,
                beam_size=8,
                temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=True,
                compression_ratio_threshold=None,
                log_prob_threshold=-1.2,
                no_speech_threshold=0.6,
                repetition_penalty=1.05,
                initial_prompt=initial_prompt,
            )
            candidates = list(retry_segments)
        except Exception:
            # Un retry non deve mai rendere indisponibile la trascrizione già
            # riuscita; il segmento originale resta il fallback sicuro.
            return None
        if not candidates:
            return None

        return _stitch_retry_segments(candidates, original, expand_margin=0.35)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trascrivi un file audio con faster-whisper")
    parser.add_argument("audio")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    t = Transcriber(model_size=args.model)
    segs, lang = t.transcribe(args.audio, language=args.lang)
    print(f"Lingua: {lang} | segmenti: {len(segs)}")
    for s in segs[:20]:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text}")
