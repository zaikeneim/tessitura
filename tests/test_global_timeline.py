"""Non-destructive timeline regression cases, including the real cached MP3."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.models.session import AudioSources, Segment, Session, Word
from src.models.storage import export_srt, export_txt, load_session, save_session
from src.models.timeline import TimelineEditor
from src.models.transcript_artifacts import Utterance
from src.processing.diarize import Turn
from src.processing.global_diarization import activity_spans, clean_identity_windows, reconstruct_events, identity_review, review_confirmed_identities, annotate_identity_uncertainty
from src.processing.merge import materialize_overlap_utterances
from src.processing.transcribe import TSegment


class CanonicalTimelineTests(unittest.TestCase):
    def session(self):
        session = Session(audio=AudioSources(mic="original.mp3"), utterances=[
            Utterance(99, 102, utterance_id="one", speaker="A", text="prima voce", words=[Word(99, 100, "prima"), Word(101, 102, "voce")]),
            Utterance(99, 101, utterance_id="two", speaker="B", text="seconda voce")])
        session.sync_segments()
        return session

    def test_identical_starts_edit_exact_id_save_reload_and_export(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.edit_event("two", text="correzione", speaker_id="C")
        with tempfile.TemporaryDirectory() as directory:
            target = save_session(session, Path(directory) / "session.json")
            restored = load_session(target)
        self.assertEqual([(u.event_id, u.text, u.speaker) for u in restored.utterances],
                         [("two", "correzione", "C"), ("one", "prima voce", "A")])
        self.assertIn("correzione", export_txt(restored))
        self.assertIn("prima voce", export_srt(restored))
        self.assertEqual(restored.utterances[0].original_prediction["text"], "seconda voce")
        self.assertTrue(restored.utterances[0].user_corrections)
        self.assertTrue(restored.undo())
        self.assertEqual(next(u for u in restored.utterances if u.event_id == "two").text, "seconda voce")
        self.assertTrue(restored.redo())

    def test_manual_missing_utterance_and_undo_redo(self):
        session = self.session()
        editor = TimelineEditor(session)
        event = editor.create_event(99.24, 99.48, "interruzione", None)
        self.assertIsNone(event.speaker_id)
        self.assertEqual(len(session.utterances), 3)
        editor.edit_event(event.event_id, speaker_id="D", uncertain=True)
        self.assertEqual(len(event.overlapping_event_ids), 2)
        self.assertEqual(event.edit_status, "user_created")
        self.assertTrue(session.undo())
        self.assertIsNone(next(u for u in session.utterances if u.event_id == event.event_id).speaker_id)
        self.assertTrue(session.undo())
        self.assertEqual(len(session.utterances), 2)
        self.assertTrue(session.redo())
        self.assertEqual(len(session.utterances), 3)

    def test_split_and_merge_keep_lineage_original_and_content(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.split_event("one", 100.5)
        parts = [u for u in session.utterances if u.speaker == "A"]
        self.assertEqual([u.text for u in parts], ["prima", "voce"])
        self.assertTrue(all(u.parent_event_ids == ["one"] for u in parts))
        editor.merge_events(parts[0].event_id, parts[1].event_id)
        merged = next(u for u in session.utterances if u.speaker == "A")
        self.assertEqual(merged.text, "prima voce")
        self.assertEqual(merged.original_prediction["text"], "prima voce")
        self.assertEqual(len(session.utterances), 2)

    def test_merge_rejects_concurrent_events_even_same_identity(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.edit_event("two", speaker_id="A")
        with self.assertRaises(ValueError):
            editor.merge_events("one", "two")

    def test_legacy_migration_is_idempotent_and_preserves_edits(self):
        session = Session(schema_version=4, segments=[Segment(1, 3, "corretto", "C", id="same")],
                          utterances=[Utterance(1, 3, text="predetto", speaker="A", utterance_id="same")])
        session.migrate_to_v6()
        before = session.to_dict()
        session.migrate_to_v6()
        self.assertEqual(before, session.to_dict())
        self.assertEqual(session.utterances[0].text, "corretto")
        self.assertEqual(session.utterances[0].original_prediction["text"], "predetto")
        self.assertEqual(session.raw_utterances[0].text, "predetto")

    def test_projection_never_wins_over_canonical_data(self):
        session = self.session()
        data = session.to_dict()
        data["segments"][0]["text"] = "stale legacy copy"
        restored = Session.from_dict(data)
        self.assertNotIn("stale legacy copy", export_txt(restored))
        with self.assertRaises(ValueError):
            Session.from_dict({"schema_version": 6, "utterances": [data["utterances"][0]] * 2})

    def test_reconstruction_never_moves_words_or_user_corrections(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.edit_event("one", text="verificato")
        before = next(u for u in session.utterances if u.event_id == "one").to_dict()
        turns = [Turn(99, 102, "A"), Turn(99.24, 99.48, "D"), Turn(99.25, 99.46, "E")]
        events = materialize_overlap_utterances(session.utterances, ordinary_turns=turns)
        self.assertEqual(next(u for u in events if u.event_id == "one").text, before["text"])
        missing = [u for u in events if u.speaker in ("D", "E")]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(not u.text and u.processing_state == "missing_transcription" for u in missing))
        ids = [u.event_id for u in events]
        self.assertEqual(ids, [u.event_id for u in materialize_overlap_utterances(events, ordinary_turns=turns)])

    def test_global_words_cross_chunk_boundaries_without_loss(self):
        chunks = [TSegment(29, 31, "uno due", [(29.8, 30.1, "uno", .9), (30.1, 30.5, "due", .8)]),
                  TSegment(30.5, 32, "tre", [(30.5, 31.5, "tre", .9)])]
        turns = [Turn(28, 32, "A"), Turn(30, 31.8, "B")]
        events = reconstruct_events(chunks, turns)
        self.assertEqual(Counter(w.text.strip() for u in events for w in u.words), Counter(["uno", "due", "tre"]))
        self.assertTrue(any(u.speaker is None and u.text for u in events))
        self.assertTrue(any(u.speaker == "B" and not u.text for u in events))

    def test_distinct_active_speakers_not_track_count_and_clean_samples(self):
        turns = [Turn(0, 8, "A"), Turn(1, 2, "A"), Turn(4, 6, "B"), Turn(5, 6, "C")]
        spans = activity_spans(turns)
        self.assertFalse(next(s for s in spans if s["start_s"] == 1)["overlap"])
        self.assertEqual(max(s["potential_speaker_count"] for s in spans), 3)
        for window in clean_identity_windows(turns):
            self.assertFalse(any(t.speaker != window["speaker"] and min(t.end, window["end_s"]) > max(t.start, window["start_s"]) for t in turns))

    def test_short_noisy_identity_samples_never_merge(self):
        review = identity_review([{"start_s": 0, "end_s": 1, "speaker": "A", "embedding": [1, 0]}])
        self.assertEqual(review["centroids"], {})
        self.assertEqual(review["policy"], "review_only_no_automatic_merges")

    def test_candidate_acceptance_preserves_verified_text(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.edit_event("one", text="verificato")
        session.recovery_candidates.append({"candidate_id": "c1", "status": "proposed",
                                            "event": Utterance(99, 101, text="candidato").to_dict()})
        editor.accept_candidate("c1", "D")
        self.assertEqual(next(u for u in session.utterances if u.event_id == "one").text, "verificato")
        self.assertEqual(len(session.utterances), 3)
        session.undo()
        self.assertEqual(session.recovery_candidates[0]["status"], "proposed")

    def test_confirmed_short_numbers_do_not_blindly_relabel_recording(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.confirm_identity("two", "CONFIRMED_NUMBER")
        samples = [{"start_s": 200, "end_s": 205, "speaker": "A", "embedding": [1., 0.]}]
        review = review_confirmed_identities(session, samples)
        self.assertEqual(review["candidates"], [])
        self.assertTrue(review["suspected_regions"])

    def test_long_confirmed_clean_evidence_generates_guarded_global_proposal(self):
        session = Session(utterances=[Utterance(0, 15, utterance_id="anchor", speaker="USER_A", text="confermato"),
                                      Utterance(100, 105, utterance_id="elsewhere", speaker="MODEL_B", text="altrove")])
        editor = TimelineEditor(session)
        editor.confirm_identity("anchor", "USER_A")
        samples = [{"start_s": a, "end_s": b, "speaker": "MODEL_A", "embedding": [1., 0.], "overlap": False}
                   for a, b in ((0, 5), (5, 10), (10, 15), (100, 105))]
        review = review_confirmed_identities(session, samples)
        self.assertEqual(len(review["candidates"]), 1)
        self.assertEqual(review["candidates"][0]["proposed_speaker"], "USER_A")
        session.recovery_candidates = review["candidates"]
        editor.edit_event("elsewhere", text="correzione successiva")
        with self.assertRaises(ValueError):
            editor.accept_candidate(review["candidates"][0]["candidate_id"])
        self.assertEqual(next(u for u in session.utterances if u.event_id == "elsewhere").text, "correzione successiva")

    def test_global_identity_uncertainty_visible_without_overwriting_confirmed_event(self):
        session = self.session()
        editor = TimelineEditor(session)
        editor.confirm_identity("two", "B")
        annotate_identity_uncertainty(session.utterances, {"suspected_regions": [{"start_s": 99, "end_s": 102, "kind": "identity_uncertain"}]})
        self.assertTrue(next(u for u in session.utterances if u.event_id == "one").uncertain)
        self.assertFalse(next(u for u in session.utterances if u.event_id == "two").uncertain)


class ReferenceRecordingTests(unittest.TestCase):
    def test_cached_full_recording_preserves_every_word_and_regressions(self):
        artifact = Path("artifacts/global-timeline/session.json")
        checkpoint = Path.home() / "AppData/Local/Tessitura/cache/benchmarks/full-raw-260824-2105/checkpoints/a4f1f95c16488036aaeb692102c0f71177c88c286cb99f1107ab87194715e733/alignment.json"
        if not artifact.exists() or not checkpoint.exists():
            self.skipTest("Reference MP3 checkpoints unavailable")
        session = load_session(artifact)
        aligned = json.loads(checkpoint.read_text(encoding="utf-8"))["payload"]["segments"]
        self.assertEqual(sum(len(s["words"]) for s in aligned), sum(len(u.words) for u in session.utterances))
        for start, end, speakers in ((99.24, 99.47, {"SPEAKER_01", "SPEAKER_03"}),
                                     (102.13, 102.71, {"SPEAKER_00", "SPEAKER_03"})):
            active = [u for u in session.utterances if u.start_s < end and u.end_s > start]
            self.assertTrue(speakers <= {u.speaker for u in active})
            self.assertGreaterEqual(len(active), 2)
        number = next(u for u in session.utterances for w in u.words if w.text.strip() == "27" and w.start < 180)
        self.assertIsNone(number.speaker_id)
        self.assertTrue(number.uncertain)
        self.assertGreater(session.duration, 5890)
