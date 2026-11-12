"""Full-pipeline regression: a reviewed speaker assignment must survive.

Ambient profile, two voices. The phrase-level merge absorbs a short real reply
("sì") into the surrounding voice; the quality review restores it to voice B.
Before the fix, `reconstruct_events` rebuilt every word from the raw turns and
the reviewed decision never reached the Session: the word came out unassigned
while `transcript_review` still claimed one changed word.

The decision is followed through event reconstruction, Session creation,
serialization, reloading, undo/redo, timeline display and export.

Esecuzione: python -m tests.test_review_persistence
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.models.storage import export_srt, export_txt, load_session, save_session
from src.processing.diarize import DiarizationResult, Turn
from src.processing.global_diarization import reconstruct_events
from src.processing.transcribe import TSegment

WORDS = [(0.00, 0.60, " una", 0.9), (0.60, 1.20, " frase.", 0.9),
         (1.20, 1.50, " sì", 0.9), (1.50, 2.00, " continua.", 0.9)]
TSEGMENTS = [TSegment(0.0, 2.0, " una frase. sì continua.", WORDS)]
# "sì" straddles the A/B boundary: two active candidates, no overlap. The
# merge absorbs it into A (phrase vote); the review keeps B (turn support).
TURNS = [Turn(0.0, 1.22, "A"), Turn(1.22, 1.50, "B"), Turn(1.50, 2.0, "A")]


def _run_pipeline(folder: Path):
    from src.processing.pipeline import PipelineOptions, process_recording

    source = folder / "room.wav"
    rng = np.random.default_rng(0)
    sf.write(source, (rng.standard_normal((16000 * 3, 2)) * 0.05).astype("float32"), 16000)
    opts = PipelineOptions(denoise=False, diarize=True, device="cpu", quality_retries=0,
                           audio_profile="stereo_ambient", analysis_mode="safe")
    diarization = DiarizationResult(turns=list(TURNS), exclusive=list(TURNS), overlaps=[],
                                    speaker_embeddings={}, model_repo="test", model_revision="r0")
    with patch("src.processing.transcribe.Transcriber") as transcriber, \
            patch("src.processing.diarize.Diarizer") as diarizer, \
            patch("src.processing.pipeline.align_segments", return_value=(TSEGMENTS, None)), \
            patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")):
        transcriber.return_value.transcribe.return_value = (TSEGMENTS, "it")
        diarizer.return_value.diarize_detailed.return_value = diarization
        return process_recording(source, options=opts, work_dir=folder)


def _event_for(session, text: str):
    return next(u for u in session.utterances if u.text == text)


class ReviewedAssignmentPersistenceTests(unittest.TestCase):
    def test_reviewer_decision_survives_reconstruction_session_reload_timeline_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            session = _run_pipeline(Path(folder))

            # The review really changed something, and the metadata says so.
            review = session.transcript_review
            self.assertEqual(review["selected"], "preserve_short_turns")
            self.assertEqual(review["changed_words"], 1)
            # Pre-review output is kept as the raw comparison copy.
            self.assertEqual([s.speaker for s in session.raw_segments], ["A"])

            # 1. Event reconstruction keeps the reviewed label and every word.
            event = _event_for(session, "sì")
            self.assertEqual(event.speaker_id, "B", "la decisione della revisione deve raggiungere gli eventi canonici")
            self.assertEqual(sum(len(u.words) for u in session.utterances), len(WORDS))
            self.assertEqual([u.speaker for u in session.utterances if u.text], ["A", "B", "A"])
            self.assertTrue(event.diarization_provenance["reviewed_labels"])
            self.assertEqual({c["speaker"] for c in event.candidate_speakers}, {"A", "B"})
            self.assertEqual(event.original_prediction["speaker"], "B")
            # Projection and the untranscribed activity of the other voices are intact.
            segment = next(s for s in session.segments if s.text == "sì")
            self.assertEqual(segment.speaker, "B")
            self.assertNotIn("UNASSIGNED", {s.speaker for s in session.segments})

            # 2./3. Serialization and reloading.
            target = save_session(session, Path(folder) / "session.json")
            restored = load_session(target)
            self.assertEqual(_event_for(restored, "sì").speaker_id, "B")
            self.assertEqual(restored.transcript_review["changed_words"], 1)

            # 4. Timeline display: the block sits in lane B, not in "Non assegnato".
            from PySide6.QtWidgets import QApplication
            from src.gui.speaker_timeline_view import TimelineCanvas
            QApplication.instance() or QApplication([])
            canvas = TimelineCanvas()
            canvas.resize(900, 400)
            canvas.set_session_data(restored)
            canvas.set_window(0.0, 5.0)
            lanes = {lane["speaker"]: lane for lane in canvas._lanes}
            block = next(b for b in canvas._blocks if b["utterance"].text == "sì")
            lane_b = lanes["B"]
            self.assertTrue(lane_b["top"] <= block["rect"].top() < lane_b["top"] + lane_b["height"])

            # 5. Export shows the reviewed voice.
            self.assertIn("[B 00:00:01] sì", export_txt(restored))
            self.assertIn("sì", export_srt(restored))

            # 6. Manual correction on the reloaded session, undo and redo.
            from src.models.timeline import TimelineEditor
            editor = TimelineEditor(restored)
            editor.edit_event(_event_for(restored, "sì").event_id, speaker_id="A")
            self.assertEqual(_event_for(restored, "sì").speaker_id, "A")
            self.assertTrue(restored.undo())
            self.assertEqual(_event_for(restored, "sì").speaker_id, "B")
            self.assertTrue(restored.redo())
            self.assertEqual(_event_for(restored, "sì").speaker_id, "A")

    def test_review_metadata_describes_the_session_content(self):
        """`changed_words` must count words that differ from raw_segments in the session itself."""
        with tempfile.TemporaryDirectory() as folder:
            session = _run_pipeline(Path(folder))
        raw = {(w.start, w.end): s.speaker for s in session.raw_segments for w in s.words}
        final = {(w.start, w.end): u.speaker for u in session.utterances for w in u.words}
        changed = sum(1 for key, speaker in final.items() if raw[key] != speaker)
        self.assertEqual(changed, session.transcript_review["changed_words"])


class ReconstructWithLabelsTests(unittest.TestCase):
    def test_without_labels_behaviour_is_unchanged(self):
        events = reconstruct_events(TSEGMENTS, TURNS, TURNS)
        self.assertEqual([(u.text, u.speaker) for u in events if u.text],
                         [("una frase.", "A"), ("sì", None), ("continua.", "A")])

    def test_labels_decide_speaker_activity_decides_uncertainty(self):
        events = reconstruct_events(TSEGMENTS, TURNS, TURNS, word_labels=["A", "A", "B", "A"])
        textual = [u for u in events if u.text]
        self.assertEqual([(u.text, u.speaker, u.uncertain) for u in textual],
                         [("una frase.", "A", False), ("sì", "B", False), ("continua.", "A", False)])
        self.assertEqual(textual[1].diarization_provenance["assignment"], "reviewed_label_supported")
        # Word order and count are preserved exactly.
        self.assertEqual([w.text.strip() for u in textual for w in u.words], ["una", "frase.", "sì", "continua."])

    def test_label_without_acoustic_support_or_in_overlap_is_flagged_not_hidden(self):
        overlapping = [Turn(0.0, 2.0, "A"), Turn(1.22, 1.50, "B")]
        events = reconstruct_events(TSEGMENTS, overlapping, overlapping, word_labels=["A", "A", "C", "A"])
        event = next(u for u in events if u.text == "sì")
        self.assertEqual(event.speaker_id, "C")
        self.assertTrue(event.uncertain)
        self.assertEqual(event.processing_state, "attribution_uncertain")
        self.assertIn(event.diarization_provenance["assignment"],
                      ("reviewed_label_in_overlap", "reviewed_label_without_activity"))
        # The other voice's simultaneous activity is still materialized.
        self.assertTrue(any(u.speaker == "B" and not u.text for u in events))

    def test_phrase_is_not_split_on_candidate_changes_when_labelled(self):
        # w2 and w3 straddle the A|B|A boundaries: two active candidates each,
        # no simultaneous speech. Word-level reconstruction leaves them
        # unassigned and splits the phrase; the reviewed label keeps it whole.
        words = [(i * 0.4, i * 0.4 + 0.4, f" w{i}", 0.9) for i in range(6)]
        tsegments = [TSegment(0.0, 2.4, " ".join(w[2] for w in words), words)]
        turns = [Turn(0.0, 1.0, "A"), Turn(1.0, 1.3, "B"), Turn(1.3, 2.4, "A")]
        labelled = reconstruct_events(tsegments, turns, turns, word_labels=["A"] * 6)
        self.assertEqual([u.text for u in labelled if u.text], [" ".join(f"w{i}" for i in range(6))])
        self.assertFalse(labelled[0].uncertain)
        unlabelled = reconstruct_events(tsegments, turns, turns)
        self.assertGreater(len([u for u in unlabelled if u.text]), 1)
        self.assertTrue(any(u.text and u.speaker is None for u in unlabelled))
        # Candidate activity is accumulated over the whole event, not the last word.
        self.assertEqual({c["speaker"] for c in labelled[0].candidate_speakers}, {"A", "B"})

    def test_unsupported_label_inside_a_phrase_is_isolated_and_flagged(self):
        words = [(i * 0.4, i * 0.4 + 0.4, f" w{i}", 0.9) for i in range(6)]
        tsegments = [TSegment(0.0, 2.4, " ".join(w[2] for w in words), words)]
        turns = [Turn(0.0, 1.2, "A"), Turn(1.2, 1.6, "B"), Turn(1.6, 2.4, "A")]
        events = [u for u in reconstruct_events(tsegments, turns, turns, word_labels=["A"] * 6) if u.text]
        self.assertEqual([(u.text, u.uncertain) for u in events],
                         [("w0 w1 w2", False), ("w3", True), ("w4 w5", False)])
        self.assertEqual(events[1].speaker_id, "A")
        self.assertEqual(events[1].diarization_provenance["assignment"], "reviewed_label_without_activity")

    def test_zero_length_first_word_before_any_activity_stays_editable(self):
        """Real case (2h reference): first word " Gli" at 0.0–0.0, first turn at 0.03 s.

        The isolated word became a 0-length event that TimelineEditor refused to
        edit ("Intervallo non valido"), so the user could never correct it.
        """
        from src.models.session import Session
        from src.models.timeline import TimelineEditor
        words = [(0.0, 0.0, " Gli", 0.9), (0.0, 0.4, " istanti", 0.9), (0.4, 0.9, " che", 0.9)]
        tsegments = [TSegment(0.0, 0.9, " Gli istanti che", words)]
        turns = [Turn(0.03, 0.9, "A")]
        for labels in (None, ["A", "A", "A"]):
            events = reconstruct_events(tsegments, turns, turns, word_labels=labels)
            self.assertEqual([w.text.strip() for u in events for w in u.words], ["Gli", "istanti", "che"])
            self.assertTrue(all(u.end_s > u.start_s for u in events))
            session = Session(utterances=events, schema_version=6)
            session.sync_segments()
            first = next(u for u in session.utterances if "Gli" in u.text)
            self.assertTrue(TimelineEditor(session).edit_event(first.event_id, speaker_id="B"))

    def test_label_count_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            reconstruct_events(TSEGMENTS, TURNS, TURNS, word_labels=["A"])


if __name__ == "__main__":
    unittest.main()
