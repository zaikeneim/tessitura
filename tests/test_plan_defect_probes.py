"""Probe di riproduzione dei difetti specificati nel piano correttivo (2026-09-21).

Verifica:
1. Collisione grafica nella timeline per eventi brevi non sovrapposti dello stesso speaker.
2. Riassegnazione speaker nel controller che usa ID rigoroso senza fallback temporale che corrompe segmenti simultanei.
3. Distinzione tra ritaglio turni (sliced) e separazione acustica reale.
"""

from __future__ import annotations

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from src.gui.speaker_timeline_view import SpeakerTimelineView
from src.gui.transcript_controller import TranscriptController
from src.models.session import Segment, Session
from src.models.transcript_artifacts import Utterance
from src.processing.speaker_tracks import SpeakerTracksResult, extract_speaker_tracks

app = QApplication.instance() or QApplication([])


class PlanDefectProbesTests(unittest.TestCase):
    def test_probe_1_timeline_non_overlapping_short_events_no_collision(self):
        """Interventi 0.0-0.1s e 0.2-0.3s dello stesso speaker non devono intersecarsi graficamente."""
        session = Session(
            name="Test Collisioni",
            utterances=[
                Utterance(start_s=0.0, end_s=0.1, utterance_id="u_short1", speaker="SPK_0", text="A"),
                Utterance(start_s=0.2, end_s=0.3, utterance_id="u_short2", speaker="SPK_0", text="B"),
            ],
            speaker_names={"SPK_0": "Speaker Zero"},
        )
        controller = TranscriptController()
        view = SpeakerTimelineView(controller)
        controller.load_session(session)
        canvas = view.canvas

        b1 = next(b for b in canvas._blocks if b["utterance"].utterance_id == "u_short1")
        b2 = next(b for b in canvas._blocks if b["utterance"].utterance_id == "u_short2")

        # Verifica che non si intersechino
        intersects = b1["rect"].intersects(b2["rect"])
        self.assertFalse(
            intersects,
            f"I rettangoli grafici di due eventi non sovrapposti nel tempo non devono intersecarsi: {b1['rect']} vs {b2['rect']}",
        )

    def test_probe_2_controller_exact_id_matching_no_temporal_fallback_corruption(self):
        """Due interventi simultanei (0.0-2.0s) con ID distinti: riassegnare il secondo non deve toccare il primo."""
        session = Session(
            name="Test Riassegnazione Esatta",
            segments=[
                Segment(start=0.0, end=2.0, text="Intervento A", speaker="S0", id="id_A"),
                Segment(start=0.0, end=2.0, text="Intervento B", speaker="S1", id="id_B"),
            ],
            utterances=[
                Utterance(start_s=0.0, end_s=2.0, text="Intervento A", speaker="S0", utterance_id="id_A"),
                Utterance(start_s=0.0, end_s=2.0, text="Intervento B", speaker="S1", utterance_id="id_B"),
            ],
            speaker_names={"S0": "Alice", "S1": "Bob", "S2": "Charlie"},
        )
        controller = TranscriptController()
        controller.load_session(session)

        # Riassegna id_B a S2
        res = controller.reassign_speaker("id_B", "S2")
        self.assertTrue(res)

        # Segment A deve rimanere intatto S0
        seg_a = next(s for s in session.segments if s.id == "id_A")
        self.assertEqual(seg_a.speaker, "S0", "Il primo segmento non doveva essere modificato")

        # Segment B deve essere S2
        seg_b = next(s for s in session.segments if s.id == "id_B")
        self.assertEqual(seg_b.speaker, "S2")

    def test_probe_3_turn_slicing_metadata_distinct_from_acoustic_separation(self):
        """L'estrazione per turni deve dichiarare esplicitamente il metodo turn_slicing e non dichiararsi separazione acustica pura."""
        import numpy as np
        from src.processing.diarize import Turn

        # Creazione audio fittizio
        sr = 16000
        mono = np.ones(sr * 3, dtype=np.float32) * 0.1
        turns = [
            Turn(start=0.0, end=2.0, speaker="S0"),
            Turn(start=1.0, end=3.0, speaker="S1"),
        ]

        import tempfile
        from pathlib import Path
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "test.wav"
            sf.write(str(wav_path), mono, sr)

            result = extract_speaker_tracks(wav_path, turns=turns)
            self.assertEqual(result.metrics.get("method"), "turn_slicing")
            self.assertIn("sliced", result.status)


if __name__ == "__main__":
    unittest.main()
