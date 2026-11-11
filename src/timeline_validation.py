"""Installed-runtime validation using a copy of a real timeline session."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def validate_timeline_runtime(session_path, *, inference=False):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QThreadPool
    from .gui.main_window import MainWindow
    from .models.storage import load_session, save_session, export_txt, export_srt
    from .processing.global_diarization import reconstruct_events
    from .processing.transcribe import TSegment

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Tessitura runtime validation")
    session = load_session(session_path)
    original_words = sum(len(u.words) for u in session.utterances)
    segments = [TSegment(u.start_s, u.end_s, u.text,
                          [(w.start, w.end, w.text, w.probability) for w in u.words])
                for u in session.utterances if u.words]
    rebuilt = reconstruct_events(segments, session.diarization.get("turns", []), source_audio_ref=session.audio.primary())
    if sum(len(u.words) for u in rebuilt) != original_words:
        raise RuntimeError("Installed reconstruction discarded words")
    window = MainWindow()
    window.session = session
    window._display_transcript()
    if window.controller.session is not session:
        raise RuntimeError("GUI detached the canonical session")
    window.timeline_view._waveform_timer.stop()
    window.timeline_view.canvas.set_window(94, 18)
    active = [u for u in session.utterances if u.start_s < 99.47 and u.end_s > 99.24]
    if len(active) < 2:
        raise RuntimeError("Concurrent reference events missing")
    event = window.controller.create_event(99.24, 99.47, "runtime validation", None)
    window.controller.reassign_speaker(event.event_id, "RUNTIME_CHECK")
    with tempfile.TemporaryDirectory(prefix="tessitura-runtime-") as directory:
        target = save_session(session, Path(directory) / "session.json")
        restored = load_session(target)
        if not restored.undo() or not restored.redo():
            raise RuntimeError("Installed persistence lost undo/redo")
        if "runtime validation" not in export_txt(restored) or "runtime validation" not in export_srt(restored):
            raise RuntimeError("Installed exports lost manual event")
        result = {"frozen": bool(getattr(__import__("sys"), "frozen", False)), "schema": restored.schema_version,
                  "events": len(restored.utterances), "source_words_preserved": original_words,
                  "concurrent_events_at_99": len(active), "canonical_gui": True,
                  "save_reload_undo_redo_export": True, "media_duration_s": restored.duration}
        if inference:
            from .processing.region_recovery import recover_region
            from .processing.resources import acquire_inference
            source = session.audio.resolved_primary()
            with acquire_inference():
                recovered = recover_region(source, 123.8, 125.18, directory, existing_events=session.utterances,
                    model_size=session.model or "large-v3", language=session.language, device="auto")
            result["offline_region_inference"] = {"candidates": len(recovered["candidates"]),
                                                   "texts": [c["event"]["text"] for c in recovered["candidates"]]}
    window._dirty = False
    window.timeline_view._waveform_timer.stop()
    window.player.clear()
    window.close()
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if os.environ.get("TESSITURA_TIMELINE_SMOKE_OUT"):
        Path(os.environ["TESSITURA_TIMELINE_SMOKE_OUT"]).write_text(output, encoding="utf-8")
    print(output, flush=True)
    return 0
