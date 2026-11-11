"""Known error counts for complete references, overlap, and global identities."""
from copy import deepcopy
import unittest

from src.models.transcript_artifacts import Utterance
from src.processing.verified_evaluation import evaluate_verified_reference


SHA = "a" * 64


def event(start, end, speaker, text, words=None):
    value = {"start_s": start, "end_s": end, "speaker": speaker, "text": text}
    if words is not None:
        value["words"] = [{"start": a, "end": b, "text": token} for a, b, token in words]
    return value


def manifest(events, start=0, end=4):
    return {"verification_level": "human_verified", "reviewed_by": "Reference reviewer",
            "source_sha256": SHA, "annotation_complete": True,
            "regions": [{"start_s": start, "end_s": end, "events": events}]}


class VerifiedEvaluationTests(unittest.TestCase):
    def score(self, reference, hypothesis, **kwargs):
        return evaluate_verified_reference(reference, hypothesis, source_sha256=SHA, **kwargs)

    def test_exact_overlap_and_permuted_speaker_ids_have_zero_error(self):
        ref_events = [event(0, 2, "A", "hello there", [(0.2, 0.5, "hello"), (1.2, 1.5, "there")]),
                      event(1, 3, "B", "good morning", [(1.3, 1.5, "good"), (2.3, 2.5, "morning")])]
        hyp_events = deepcopy(ref_events)
        hyp_events[0]["speaker"], hyp_events[1]["speaker"] = "Y", "X"
        result = self.score(manifest(ref_events), hyp_events)
        self.assertEqual(result["der"]["der"], 0)
        self.assertEqual(result["der"]["mapping"], {"Y": "A", "X": "B"})
        self.assertEqual(result["jer"]["jer"], 0)
        self.assertEqual(result["speaker_attributed_wer"]["wer"], 0)
        self.assertEqual(result["overlap_wer"]["reference_words"], 2)
        self.assertEqual(result["overlap_wer"]["wer"], 0)
        self.assertEqual(result["identity_fragmentation"]["extra_labels"], 0)

    def test_missing_second_simultaneous_voice_counts_speaker_time_and_words(self):
        reference = manifest([event(0, 2, "A", "hello"), event(0, 2, "B", "yes")])
        result = self.score(reference, [Utterance(0, 2, speaker="X", text="hello")])
        self.assertEqual(result["der"]["miss_s"], 2)
        self.assertEqual(result["der"]["reference_speaker_seconds"], 4)
        self.assertEqual(result["der"]["der"], 0.5)
        self.assertEqual(result["jer"]["jer"], 0.5)
        self.assertEqual(result["speaker_attributed_wer"]["deletions"], 1)
        self.assertEqual(result["speaker_attributed_wer"]["wer"], 0.5)
        self.assertEqual(result["overlap_wer"]["wer"], 0.5)

    def test_false_alarm_in_reviewed_silence_is_not_ignored(self):
        reference = manifest([event(0, 1, "A", "hello")], end=3)
        result = self.score(reference, [event(0, 1, "X", "hello"), event(1, 2, "X", "noise")])
        self.assertEqual(result["der"]["false_alarm_s"], 1)
        self.assertEqual(result["der"]["der"], 1)
        self.assertEqual(result["jer"]["jer"], 0.5)
        self.assertEqual(result["speaker_attributed_wer"]["insertions"], 1)

    def test_optimal_global_mapping_counts_identity_fragmentation(self):
        reference = manifest([event(0, 2, "A", "one two"), event(2, 4, "B", "three")])
        hypothesis = [event(0, 1, "X", "one"), event(1, 2, "Y", "two"), event(2, 4, "Z", "three")]
        result = self.score(reference, hypothesis)
        self.assertEqual(result["der"]["confusion_s"], 1)
        self.assertEqual(result["der"]["der"], 0.25)
        self.assertEqual(result["identity_fragmentation"]["extra_labels"], 1)
        self.assertEqual(result["identity_fragmentation"]["extra_labels_by_reference"]["A"], 1)
        self.assertEqual(result["jer"]["jer"], 0.25)

    def test_hungarian_assignment_beats_greedy_largest_overlap(self):
        reference = manifest([event(0, 5, "A", "one"), event(5, 10, "B", "two")], end=10)
        hypothesis = [event(0, 4, "X", "one"), event(1, 4, "Y", "one"), event(5, 8, "X", "two")]
        result = self.score(reference, hypothesis)
        # Intersection weights are [[4, 3], [3, 0]]. A greedy X->A pairing
        # credits 4 speaker-seconds; the global pairing credits 6.
        self.assertEqual(result["der"]["mapping"], {"X": "B", "Y": "A"})
        self.assertEqual(result["der"]["miss_s"], 3)
        self.assertEqual(result["der"]["false_alarm_s"], 3)
        self.assertEqual(result["der"]["confusion_s"], 1)
        self.assertEqual(result["der"]["der"], 0.7)

    def test_reference_regions_scope_out_unreviewed_hypotheses(self):
        reference = manifest([event(0, 1, "A", "hello")], end=2)
        result = self.score(reference, [event(0, 1, "X", "hello"), event(10, 20, "Y", "unreviewed")])
        self.assertEqual(result["der"]["der"], 0)
        self.assertEqual(result["speaker_attributed_wer"]["hypothesis_words"], 1)
        self.assertEqual(result["reviewed_seconds"], 2)

    def test_partial_overlap_without_word_times_makes_overlap_wer_unavailable(self):
        reference = manifest([event(0, 2, "A", "hello there"), event(1, 3, "B", "good morning")])
        result = self.score(reference, deepcopy(reference["regions"][0]["events"]))
        self.assertEqual(result["der"]["der"], 0)
        self.assertEqual(result["speaker_attributed_wer"]["wer"], 0)
        self.assertIsNone(result["overlap_wer"]["wer"])
        self.assertEqual(result["overlap_wer"]["status"], "unavailable")

    def test_crossing_review_boundary_without_word_times_does_not_guess_words(self):
        reference = manifest([event(1, 2, "A", "hello")], start=1, end=2)
        result = self.score(reference, [event(0, 3, "X", "before hello after")])
        self.assertEqual(result["der"]["der"], 0)
        self.assertIsNone(result["speaker_attributed_wer"]["wer"])

    def test_unverified_incomplete_or_wrong_recording_references_are_rejected(self):
        base = manifest([event(0, 1, "A", "hello")])
        for change in ({"verification_level": "unverified_candidate"}, {"reviewed_by": " "},
                       {"source_sha256": "b" * 64}, {"annotation_complete": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score({**base, **change}, [])
        missing_complete = deepcopy(base)
        missing_complete.pop("annotation_complete")
        with self.assertRaises(ValueError):
            self.score(missing_complete, [])

    def test_incomplete_region_cannot_be_overridden_by_complete_manifest(self):
        reference = manifest([event(0, 1, "A", "hello")])
        reference["regions"][0]["annotation_complete"] = False
        with self.assertRaises(ValueError):
            self.score(reference, [])

    def test_reference_undefined_speaker_or_crossing_region_is_rejected(self):
        for item in (event(0, 1, None, "hello"), event(0, 5, "A", "hello"), event(0, 1, "A", "")):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.score(manifest([item]), [])

    def test_region_verification_cannot_contradict_manifest(self):
        reference = manifest([event(0, 1, "A", "hello")])
        reference["regions"][0]["verification_level"] = "unverified_candidate"
        with self.assertRaises(ValueError):
            self.score(reference, [])

    def test_unassigned_hypothesis_is_not_credited_as_known_identity(self):
        result = self.score(manifest([event(0, 1, "A", "hello")]), [event(0, 1, None, "hello")])
        self.assertEqual(result["der"]["confusion_s"], 1)
        self.assertEqual(result["der"]["mapping"], {})

    def test_no_reference_speech_has_undefined_rate_and_explicit_false_alarm(self):
        result = self.score(manifest([], end=2), [event(0, 1, "X", "noise")])
        self.assertIsNone(result["der"]["der"])
        self.assertEqual(result["der"]["false_alarm_s"], 1)
        self.assertIsNone(result["jer"]["jer"])

    def test_runtime_and_vram_metadata_do_not_affect_quality(self):
        reference = manifest([event(0, 1, "A", "hello")])
        hypothesis = [event(0, 1, "X", "hello")]
        result = self.score(reference, hypothesis, runtime_metadata={"elapsed_s": 12, "peak_vram_bytes": 1024})
        baseline = self.score(reference, hypothesis)
        self.assertEqual(result["der"], baseline["der"])
        self.assertEqual(result["performance"], {"elapsed_s": 12, "peak_vram_bytes": 1024})


if __name__ == "__main__":
    unittest.main()
