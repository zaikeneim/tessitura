import unittest
from src.models.session import Session, Segment, Word


class SessionHistoryTests(unittest.TestCase):
    def test_legacy_migration_assigns_repeatable_ids_and_roundtrips(self):
        old = {"id": "old", "segments": [{"start": 0, "end": 1, "text": "ciao"}]}
        one, two = Session.from_dict(old), Session.from_dict(old)
        self.assertEqual(one.segments[0].id, two.segments[0].id)
        self.assertEqual(one.schema_version, 2)
        self.assertEqual(Session.from_dict(one.to_dict()).to_dict(), one.to_dict())

    def test_undo_redo_preserves_original_word_timing(self):
        s = Session(segments=[Segment(0, 1, "ciao", "A", [Word(0, 1, "ciao")])])
        original = s.segments[0].id
        s.edit_segment(0, text="salve", speaker="B")
        self.assertFalse(s.segments[0].words_valid)
        self.assertEqual(s.raw_segments[0].text, "ciao")
        s = Session.from_dict(s.to_dict())
        self.assertTrue(s.undo())
        self.assertEqual(s.segments[0].text, "ciao")
        self.assertEqual(s.segments[0].id, original)
        self.assertTrue(s.segments[0].words_valid)
        self.assertEqual(s.revision, 2)
        self.assertTrue(s.redo())
        self.assertEqual(s.segments[0].text, "salve")
        self.assertEqual(s.revision, 3)

    def test_grouping_is_reversible_without_altering_word_timing(self):
        s = Session(segments=[Segment(0, 1, "uno", "A", [Word(0, 1, "uno")]),
                              Segment(5, 6, "due", "A", [Word(5, 6, "due")])])
        before = s.fingerprint()
        self.assertEqual(s.group_adjacent_speaker_segments(), 1)
        self.assertEqual(s.segments[0].words[1].start, 5)
        self.assertTrue(s.undo())
        self.assertEqual(len(s.segments), 2)
        self.assertNotEqual(s.fingerprint(), before)

    def test_split_keeps_every_word_exactly_once_and_undo(self):
        words = [Word(0, 1, "uno"), Word(2, 3, "due")]
        s = Session(segments=[Segment(0, 3, "uno due", "A", words)])
        self.assertTrue(s.split_segment(0, 1.5))
        self.assertEqual([w.text for seg in s.segments for w in seg.words], ["uno", "due"])
        self.assertEqual(len({seg.id for seg in s.segments}), 2)
        self.assertTrue(s.undo())
        self.assertEqual(s.segments[0].text, "uno due")

    def test_fingerprint_detects_legacy_direct_edits(self):
        s = Session(segments=[Segment(0, 1, "ciao")])
        before = s.fingerprint()
        s.segments[0].text = "testo diverso"
        self.assertNotEqual(s.fingerprint(), before)

    def test_invalid_future_schema_duplicate_ids_and_confidence_rejected(self):
        with self.assertRaises(ValueError):
            Session.from_dict({"schema_version": 999})
        seg = Segment(0, 1, "uno", words=[Word(0, 1, "uno", float("nan"))])
        with self.assertRaises(ValueError):
            Session.from_dict(Session(segments=[seg]).to_dict())
        seg.words.clear()
        with self.assertRaises(ValueError):
            Session.from_dict(Session(segments=[seg, seg]).to_dict())


if __name__ == "__main__":
    unittest.main()
