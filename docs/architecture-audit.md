# Architecture Audit — Tessitura (Pre-Refactoring)

**Date**: 2026-11-11  
**Base Commit**: `d6fea5e` (`feature/affidabilita-audio`)  
**Refactoring Branch**: `refactor/local-asr-diarization`

---

## 1. Current Architecture Overview

Tessitura is a local Windows x64 desktop application written in Python 3.11 with a PySide6 GUI, `pyaudiowpatch` for microphone and WASAPI loopback audio capture, `ffmpeg` / `soundfile` / `scipy` for audio composition and preprocessing, `faster-whisper` (`CTranslate2`) for speech recognition, `pyannote.audio` (`pyannote/speaker-diarization-community-1`) for speaker diarization, and an added layer of `transformers`/`accelerate`/`sentencepiece` for local Qwen LLM semantic analysis and text refinement.

### Entry Points
- **Desktop GUI**: `main.py` → `src/app.py:run()` → `src/gui/main_window.py:MainWindow`
- **Launcher scripts**: `Avvia.bat` → `scripts/launch.ps1` (activates local Conda prefix `./env` via `scripts/runtime.ps1`)
- **CLI Pipeline**: `scripts/run_pipeline.py`
- **CLI LLM Analysis (to be removed)**: `scripts/analyze_session.py`
- **Verification & Diagnostics**: `scripts/verify.py`, `scripts/check_env.py`, `scripts/runtime_health.py`, `scripts/evaluate_sessions.py`, `scripts/benchmark_pipeline.py`, `scripts/benchmark_dual_source.py`

---

## 2. Components to Preserve (Working Core)

| Area | Files | Why Preserve |
| :--- | :--- | :--- |
| **Audio Acquisition** | `src/audio/capture.py`, `src/audio/devices.py` | Producer-consumer `_StreamRecorder` per track (mic + WASAPI loopback), explicit `started_at` monotonic clock offsets, frame drop tracking, clean shutdown. |
| **Audio Merging & Encoding** | `src/audio/encode.py`, `src/processing/audio_utils.py` | Atomic `ffmpeg` MP3 encoding (`to_mp3`, `merge_to_stereo_mp3` with `adelay` offset compensation, `combine_to_mp3` in `concat` and `mix` modes), `mix_sources` at 16 kHz float32 with peak-ceiling protection (`PEAK_CEILING = 0.891`), `source_dominance`, and `channels_are_duplicate`. |
| **Acoustic Diagnostics & Denoising** | `src/processing/acoustic_diagnostics.py`, `src/processing/denoise.py`, `src/processing/audio_profile.py` | Multi-window SNR and noise-floor estimation, conservative spectral gating (`noisereduce` with `prop_decrease=0.5`), explicit `AudioProfile` (`MIC_SYSTEM`, `STEREO_AMBIENT`). |
| **Multi-Track Audio Player** | `src/gui/audio_player.py` | Supports switching dynamically between `Mix completo`, `Solo Microfono`, and `Solo Sistema` while preserving playback position. |
| **Multi-View Transcript UI & Editing** | `src/gui/transcript_controller.py`, `src/gui/conversation_view.py`, `src/gui/speaker_timeline_view.py`, `src/gui/transcript_view.py`, `src/gui/speaker_panel.py` | Conversation, Timeline, and Table views; simultaneous overlap highlighting during playback; per-speaker renaming and colors; in-place text and speaker editing with full undo/redo (`Session.begin_edit`, `undo`, `redo`). |
| **Session Persistence & Exports** | `src/models/session.py`, `src/models/storage.py`, `src/models/transcript_artifacts.py`, `src/models/audio_artifacts.py` | Atomic JSON persistence (`.tmp` + `os.replace`), schema migration (v1–v4), TXT export, chronological multi-speaker SRT export, and per-speaker TXT export. |
| **Project Glossary & Voice Profiles** | `src/models/project.py`, `src/processing/adaptation.py`, `src/processing/voice_profile.py`, `src/gui/project_actions.py`, `src/gui/project_dialog.py` | Local, deterministic project glossary for Whisper `initial_prompt`, and cosine-similarity speaker profile matching across sessions using pyannote embeddings. |

---

## 3. Components to Remove (Unnecessary Qwen / LLM Complexity)

| Component | Files / Symbols | Technical Debt & Reason for Removal |
| :--- | :--- | :--- |
| **Local Qwen LLM Engine** | `src/processing/llm.py` (`LocalLLM`, `CATALOG`, `ModelSpec`, `spec_for`, `recommended_spec`) | Loads 1.5B–4B Qwen models (`Qwen3-4B-Instruct-2507`, `Qwen2.5-3B-Instruct`, etc.) via `transformers` + `accelerate` in VRAM/RAM, competing with ASR/diarization on an 8 GB Laptop GPU and adding fragile regex/JSON parsing. |
| **LLM Semantic Analysis & Refinement** | `src/processing/analysis.py` (`analyze_session`, `suggest_speaker_names`, `review_speaker_assignments`, `apply_speaker_review`, `refine_transcription_text`, `apply_transcription_refinement`, `answer_question`), `src/processing/llm_refinement.py`, `src/processing/llm_backends.py`, `src/processing/ollama_backend.py`, `src/processing/retrieval.py` | Generative summarization, speaker guessing from text alone (without audio), and LLM rewriting of transcripts introduce hallucination risk, high latency, and unnecessary complexity outside the core ASR + diarization scope. |
| **Pipeline LLM Post-Processing Hook** | `src/processing/pipeline.py` (`_refine_with_local_llm`, `PipelineOptions.llm_refine`, `PipelineOptions.analysis_model_repo`) | Runs an LLM pass at the end of every transcription pipeline when `llm_refine=True`, delaying completion and risking text alteration. |
| **GUI LLM Dialogs & Actions** | `src/gui/analysis_chat.py` (`AnalysisChatDialog`), `src/gui/speaker_review_dialog.py` (`SpeakerReviewDialog`), `src/gui/workers.py` (`AnalysisWorker`), `src/gui/main_window.py` (`Analisi` menu, `Chat locale`, `Riassunto`, `Rivedi voci`, `Raffina trascrizione con modello locale`, `review_check`) | UI clutter tied exclusively to Qwen/LLM analysis and refinement. |
| **Unused Experimental ASR Backends** | `src/processing/asr_backends.py` (`Qwen3ASRBackend`, `ParakeetBackend`), `src/processing/model_catalog.py`, `src/processing/model_store.py` | Unverified/placeholder backends for `Qwen/Qwen3-ASR-1.7B`, `Parakeet`, and `MossFormer2` that increase surface area without improving the production `faster-whisper` pipeline. |
| **CLI LLM Script** | `scripts/analyze_session.py` | Standalone CLI wrapper for `src/processing/analysis.py` and `LocalLLM`. |
| **LLM Test Suites** | `tests/test_analysis.py`, `tests/test_llm_catalog.py`, `tests/test_llm_refinement.py`, `tests/test_ollama_backend.py`, `tests/test_retrieval.py` | Tests exclusively targeting removed LLM/Qwen modules. |

---

## 4. Components to Modify & Architectural Flaws to Fix

1. **`src/processing/pipeline.py` — Multi-Speaker Track Slicing Anti-Pattern**:
   - **Current Flaw**: In the `standard` strategy, `profile_first=True` runs `Diarizer` first, and whenever `len(speaker_ids) >= 2`, slices the mixed audio by diarization turns (`extract_speaker_tracks`) into separate `voice_SPEAKER_XX.wav` files and runs `Transcriber.transcribe()` N times (once per speaker). This destroys sentence context across speaker turns, multiplies Whisper inference time by N, and produces boundary artifacts requiring `transcript_reconciler.py`.
   - **Fix**: Make the primary `standard` pipeline execute in clean, sequential stages:
     1. Audio acquisition & preprocessing (offset alignment, conservative denoise, stereo dominance analysis)
     2. Full-context ASR (`faster-whisper` `large-v3` with VAD, gap recovery, and targeted retries) — unloading ASR from GPU immediately after
     3. Forced word-level alignment (`src/processing/alignment.py`) — refining word timestamps using acoustic forced alignment when available or preserving verified Whisper word timestamps without fabricating missing timestamps, then unloading from GPU
     4. Speaker diarization (`pyannote/speaker-diarization-community-1`) — extracting both ordinary overlapping turns (`turns`), `exclusive_speaker_diarization` (`exclusive`), `overlaps`, and speaker embeddings, then unloading from GPU
     5. Speaker-to-word assignment (`src/processing/merge.py`) — combining `exclusive` and ordinary turns with stereo/dual-source dominance hints, preserving overlapping speech (`Utterance` streams + `overlap_group_id`), and flagging uncertain assignments (`attribution_status`, `speaker_ambiguous`)
     6. Deterministic post-processing (`autocorrect.py`, `transcript_review.py`) and persistence (`Session` v4).
   - Meanwhile, preserve the fast dual-source path (`source_single_speaker` in `src/processing/source_transcription.py`) when microphone and system tracks are recorded separately and selected as 1-speaker-per-track, as well as exploiting separate mic/system tracks (`use_channel_hint` / `source_dominance`) during multi-speaker diarization.

2. **`src/hardware.py` & `src/processing/transcribe.py` — 8 GB VRAM Optimization**:
   - **Current Flaw**: On an 8 GB GPU (`vram_gb >= 6.0`), `hardware.py` defaults to `compute_type = "float16"`. As seen in `artifacts/benchmark_dual_source.json`, peak VRAM in the standard pipeline reached `8,378.8 MB` on the 8 GB RTX 4070 Laptop GPU, causing shared-memory spilling and severe slowdowns (`727s` for 15m audio).
   - **Fix**: Use `int8_float16` as the default CUDA `compute_type` on 6–10 GB GPUs (with automatic fallback to `float16` / `int8` / CPU if unsupported), enforce conservative batching, and ensure strict sequential GPU lifecycle (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) between ASR, alignment, and diarization so two heavy models are never resident in VRAM simultaneously.

3. **`src/processing/alignment.py` — Dedicated Forced Alignment Stage**:
   - **Current Flaw**: Word timestamps come only from Whisper's internal cross-attention DTW (`word_timestamps=True`), with no dedicated forced-alignment module boundary.
   - **Fix**: Introduce `src/processing/alignment.py` with a clean interface that refines word timestamps using wav2vec2 CTC forced alignment (`torchaudio` / `whisperx` style) when a local alignment model is available, or falls back transparently to Whisper's cross-attention word timestamps with explicit provenance metadata (`alignment_backend`, `alignment_status`), never fabricating timestamps for unaligned words.

4. **`src/gui/main_window.py`, `src/gui/model_settings.py`, `src/gui/workers.py`, `src/models/project.py`**:
   - Remove all Qwen/LLM controls, menus, workers, and settings while keeping ASR model selection, audio profile selection, recording mode selection (`source_single_speaker` vs `standard`), speaker panel renaming, conversation/timeline/table views, project glossary/voice profiles, and TXT/SRT/JSON exports intact.

---

## 5. Dependencies & Environment Audit

- **Runtime**: Local Conda environment at `./env` (Python 3.11, Windows x64).
- **Core Packages Used**:
  - `torch`, `torchaudio`, `torchcodec` (CUDA runtime & audio decoding)
  - `faster-whisper`, `ctranslate2` (ASR inference)
  - `pyannote.audio` 4.x (`pyannote/speaker-diarization-community-1`)
  - `PySide6` (Desktop GUI)
  - `PyAudioWPatch`, `sounddevice`, `soundfile`, `scipy`, `numpy`, `noisereduce` (Audio capture & DSP)
- **Packages to Remove from `requirements.txt`**:
  - `accelerate` and `sentencepiece` (installed solely for `LocalLLM` Qwen inference; verify whether any retained package requires `transformers` before removing `transformers` from `requirements.txt`).

