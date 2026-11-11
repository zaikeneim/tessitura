"""Test per transcript_reconciler.py (Piano 2026-09-21 §6.D)."""

import unittest
from src.models.session import Segment, Word
from src.models.transcript_artifacts import Utterance
from src.processing.transcript_reconciler import (
    reconcile_margin_overlap,
    reconcile_segments,
    reconcile_utterances,
)


class TranscriptReconcilerTests(unittest.TestCase):
    def test_reconcile_margin_overlap_deduplicates_repeated_context_words(self):
        w1 = [
            Word(start=0.0, end=1.0, text="buongiorno", probability=0.95),
            Word(start=1.1, end=2.0, text="a", probability=0.95),
            Word(start=2.1, end=3.0, text="tutti", probability=0.95),
        ]
        # Finestra successiva con overlap di margine che ripete "a tutti"
        w2 = [
            Word(start=1.12, end=2.02, text="a", probability=0.96),
            Word(start=2.12, end=3.02, text="tutti", probability=0.96),
            Word(start=3.1, end=4.0, text="presenti", probability=0.92),
        ]
        merged, deduped = reconcile_margin_overlap(w1, w2, overlap_s=1.9)
        self.assertTrue(deduped)
        texts = [w.text for w in merged]
        self.assertEqual(texts, ["buongiorno", "a", "tutti", "presenti"])

    def test_real_repetitions_preserved(self):
        """Ripetizioni lecite come 'no, no' non devono essere eliminate."""
        w1 = [Word(start=0.0, end=0.8, text="no", probability=0.95)]
        w2 = [Word(start=1.2, end=2.0, text="no", probability=0.95)]
        # Nessun overlap temporale (< 0.05s)
        merged, deduped = reconcile_margin_overlap(w1, w2, overlap_s=0.0)
        self.assertFalse(deduped)
        self.assertEqual([w.text for w in merged], ["no", "no"])

    def test_reconcile_segments_preserves_simultaneous_different_speakers(self):
        """Interventi simultanei di speaker diversi non vengono mai uniti o rimossi."""
        segs = [
            Segment(start=1.0, end=3.0, text="Parlo io", speaker="SPK_0", id="s1"),
            Segment(start=1.5, end=2.5, text="Parlo anch'io", speaker="SPK_1", id="s2"),
        ]
        reconciled = reconcile_segments(segs)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(reconciled[0].speaker, "SPK_0")
        self.assertEqual(reconciled[1].speaker, "SPK_1")

    def test_reconcile_utterances_preserves_overlap_groups(self):
        utts = [
            Utterance(start_s=1.0, end_s=3.0, text="Intervento A", speaker="SPK_0", utterance_id="u1", overlap_group_id="ov1"),
            Utterance(start_s=1.5, end_s=3.5, text="Intervento B", speaker="SPK_1", utterance_id="u2", overlap_group_id="ov1"),
        ]
        reconciled = reconcile_utterances(utts)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(reconciled[0].overlap_group_id, "ov1")
        self.assertEqual(reconciled[1].overlap_group_id, "ov1")


if __name__ == "__main__":
    unittest.main()
