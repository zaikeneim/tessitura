"""Overlap means two or more *distinct* speakers active at once.

`_overlapping_spans` is the source of `DiarizationResult.overlaps`, which
protects short interjections from smoothing and is persisted in the session.
Counting intersecting intervals instead of distinct voices would report a
duplicated or fragmented annotation of one voice as simultaneous speech.

Esecuzione: python -m tests.test_overlap_semantics
"""

from __future__ import annotations

import unittest

from src.processing.diarize import Turn, _overlapping_spans
from src.processing.global_diarization import activity_spans


def _distinct_speaker_overlap(turns):
    """Independent reference: merged overlap pieces from `activity_spans`."""
    merged = []
    for span in activity_spans(turns):
        if not span["overlap"]:
            continue
        if merged and abs(merged[-1][1] - span["start_s"]) < 1e-9:
            merged[-1] = (merged[-1][0], span["end_s"])
        else:
            merged.append((span["start_s"], span["end_s"]))
    return merged


class OverlapSemanticsTests(unittest.TestCase):
    def test_two_different_speakers_overlapping(self):
        self.assertEqual(_overlapping_spans([Turn(0, 5, "A"), Turn(3, 8, "B")]), [(3, 5)])

    def test_adjacent_non_overlapping_turns(self):
        self.assertEqual(_overlapping_spans([Turn(0, 5, "A"), Turn(5, 9, "B"), Turn(9, 12, "A")]), [])

    def test_three_simultaneous_speakers_count_as_one_span(self):
        self.assertEqual(_overlapping_spans([Turn(0, 9, "A"), Turn(1, 8, "B"), Turn(2, 7, "C")]), [(1, 8)])

    def test_intersecting_turns_of_the_same_speaker_are_not_overlap(self):
        self.assertEqual(_overlapping_spans([Turn(0, 5, "A"), Turn(3, 8, "A")]), [])
        # ...but a different voice inside that region still is.
        self.assertEqual(_overlapping_spans([Turn(0, 5, "A"), Turn(3, 8, "A"), Turn(4, 6, "B")]), [(4, 6)])

    def test_duplicated_and_fragmented_annotation_intervals(self):
        duplicated = [Turn(0, 5, "A"), Turn(0, 5, "A"), Turn(2, 3, "A")]
        self.assertEqual(_overlapping_spans(duplicated), [])
        fragmented = [Turn(0, 2, "A"), Turn(2, 4, "A"), Turn(1.5, 2.5, "A"), Turn(3, 6, "B")]
        self.assertEqual(_overlapping_spans(fragmented), [(3, 4)])

    def test_agrees_with_activity_spans_on_distinct_speakers(self):
        turns = [Turn(0, 8, "A"), Turn(1, 2, "A"), Turn(4, 6, "B"), Turn(5, 6, "C"),
                 Turn(7, 9, "B"), Turn(9, 10, "A"), Turn(9.5, 10, "A")]
        self.assertEqual(_overlapping_spans(turns), _distinct_speaker_overlap(turns))

    def test_zero_length_and_empty_input(self):
        self.assertEqual(_overlapping_spans([]), [])
        self.assertEqual(_overlapping_spans([Turn(1, 1, "A"), Turn(1, 1, "B")]), [])


if __name__ == "__main__":
    unittest.main()
