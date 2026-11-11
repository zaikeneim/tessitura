import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.models.transcript_artifacts import Utterance
from src.processing.region_recovery import recover_region
from src.processing.separation import SeparationResult
from src.processing.transcribe import TSegment


class RegionRecoveryTests(unittest.TestCase):
    def test_separated_candidates_keep_absolute_offsets_and_do_not_edit_originals(self):
        class ASR:
            def __init__(self, **_kwargs):
                pass
            def transcribe(self, _path, **_kwargs):
                return [TSegment(1.5, 2.5, "seconda voce", [(1.5, 2.5, "seconda voce", .9)])], "it"
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.wav"
            sf.write(path, np.ones((16000 * 12, 2), dtype=np.float32) * .05, 16000)
            before = path.read_bytes()
            original = Utterance(9, 12, text="verificato", speaker="A", edit_status="user_edited")
            separate = lambda *_a, **_kw: SeparationResult(region_id="r", tracks=[np.ones(16000 * 4) * .1] * 3, delay_compensation_s=.1)
            with patch("src.processing.pipeline.resolve_device", return_value=("cpu", "int8")):
                result = recover_region(path, 10, 11, directory, existing_events=[original], separate=True,
                                        potential_speakers=3, transcriber_factory=ASR, separator=separate)
            self.assertEqual(len(result["candidates"]), 3)
            event = result["candidates"][0]["event"]
            self.assertAlmostEqual(event["start_s"], 9.4)
            self.assertAlmostEqual(event["end_s"], 10.4)
            self.assertIsNone(event["speaker"])
            self.assertTrue(event["diarization_provenance"]["source_index_is_not_speaker_id"])
            self.assertEqual(original.text, "verificato")
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(Path(event["asr_provenance"]["original_context_audio"]).exists())

    def test_invalid_or_entire_recording_selection_rejected(self):
        for start, end in ((0, 121), (-1, 1), (2, 1), (0, float("nan"))):
            with self.assertRaises(ValueError):
                recover_region("unused.wav", start, end, "unused")
