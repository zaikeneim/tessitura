import unittest
from types import SimpleNamespace

from src.processing.transcribe import TSegment, _stitch_retry_segments


class RetryStitchTests(unittest.TestCase):
    def test_words_are_deduplicated_and_clipped_to_original_window(self):
        word = lambda start, end, text: SimpleNamespace(start=start, end=end, word=text, probability=.9)
        original = TSegment(10, 12, "vecchio", [], avg_logprob=-1.2)
        candidates = [
            SimpleNamespace(start=9.5, end=11, text=" contesto prima vecchio", words=[
                word(9.5, 9.9, " contesto"), word(10, 10.6, " vecchio")],
                avg_logprob=-.4, compression_ratio=1.1, no_speech_prob=.1),
            SimpleNamespace(start=10, end=12.4, text=" vecchio testo dopo", words=[
                word(10, 10.6, " vecchio"), word(11, 11.5, " testo"), word(12.2, 12.5, " dopo")],
                avg_logprob=-.3, compression_ratio=1.1, no_speech_prob=.1),
        ]
        result = _stitch_retry_segments(candidates, original)
        self.assertEqual(result.text, "vecchio testo")
        self.assertEqual([w[2].strip() for w in result.words], ["vecchio", "testo"])
        self.assertGreaterEqual(result.start, 9.8)
        self.assertLessEqual(result.end, 12.2)

    def test_word_joining_adds_spaces_without_breaking_punctuation(self):
        word = lambda start, end, text: SimpleNamespace(start=start, end=end, word=text, probability=.9)
        original = TSegment(0, 3, "", [])
        result = _stitch_retry_segments([
            SimpleNamespace(start=0, end=3, text="", words=[
                word(0, 1, "ciao"), word(1, 2, " a"), word(2, 3, ",")],
                avg_logprob=-.2, compression_ratio=1.0, no_speech_prob=.1),
        ], original)
        self.assertEqual(result.text, "ciao a,")

    def test_no_words_uses_best_single_segment_without_concatenating_context(self):
        original = TSegment(10, 12, "vecchio", [])
        segment = SimpleNamespace(start=10, end=12, text=" testo corretto", words=[],
                                  avg_logprob=-.2, compression_ratio=1.0, no_speech_prob=.1)
        other = SimpleNamespace(start=9, end=10, text=" contesto", words=[],
                                avg_logprob=-1.0, compression_ratio=3.0, no_speech_prob=.1)
        result = _stitch_retry_segments([segment, other], original)
        self.assertEqual(result.text, "testo corretto")

    def test_no_word_context_only_candidate_is_rejected(self):
        original = TSegment(10, 12, "vecchio", [])
        context = SimpleNamespace(start=8, end=9, text=" contesto", words=[],
                                  avg_logprob=-.2, compression_ratio=1.0, no_speech_prob=.1)
        self.assertIsNone(_stitch_retry_segments([context], original))


if __name__ == "__main__":
    unittest.main()
