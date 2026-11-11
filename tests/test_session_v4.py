"""Test per lo schema di sessione v4 e migrazione v3->v4 (Passo 1).

Verifica la migrazione automatica senza perdita di dati, la corretta serializzazione
dei nuovi campi tipizzati (pipeline_strategy, capture_manifest, active_view,
timeline_visible_speakers), la conservazione dell'undo/redo e la determinazione
del fingerprint.
"""

from __future__ import annotations

import unittest

from src.models.session import Segment, Session, Word
from src.models.transcript_artifacts import Utterance


class SessionV4MigrationTests(unittest.TestCase):
    def test_v3_to_v4_migration(self):
        """Una sessione v3 viene migrata a v4 con i campi predefiniti corretti."""
        v3_data = {
            "schema_version": 3,
            "id": "sess_v3_to_v4",
            "name": "Sessione Test V3",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Intervento 1",
                    "speaker": "SPEAKER_00",
                    "id": "seg_01",
                    "words": [{"start": 0.0, "end": 2.0, "text": "Intervento 1"}],
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Intervento 2",
                    "speaker": "SPEAKER_01",
                    "id": "seg_02",
                    "words": [{"start": 2.0, "end": 4.0, "text": "Intervento 2"}],
                },
            ],
            "utterances": [
                {
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "utterance_id": "seg_01",
                    "stream_id": 0,
                    "speaker": "SPEAKER_00",
                    "text": "Intervento 1",
                },
                {
                    "start_s": 2.0,
                    "end_s": 4.0,
                    "utterance_id": "seg_02",
                    "stream_id": 0,
                    "speaker": "SPEAKER_01",
                    "text": "Intervento 2",
                },
            ],
            "speaker_names": {"SPEAKER_00": "Docente", "SPEAKER_01": "Studente"},
        }

        session = Session.from_dict(v3_data)
        self.assertEqual(session.schema_version, 3)

        # Migrazione esplicita a v4
        session.upgrade_to_v4()
        self.assertEqual(session.schema_version, 4)
        self.assertEqual(session.pipeline_strategy, "standard")
        self.assertEqual(session.active_view, "conversation")
        self.assertEqual(session.timeline_visible_speakers, ["SPEAKER_00", "SPEAKER_01"])
        self.assertIsNone(session.capture_manifest)

        # Roundtrip to_dict
        dumped = session.to_dict()
        self.assertEqual(dumped["schema_version"], 4)
        self.assertEqual(dumped["pipeline_strategy"], "standard")
        self.assertEqual(dumped["active_view"], "conversation")
        self.assertEqual(dumped["timeline_visible_speakers"], ["SPEAKER_00", "SPEAKER_01"])

    def test_v4_direct_roundtrip_with_manifest(self):
        """Una sessione creata direttamente con schema v4 conserva tutti i metadati."""
        manifest = {
            "mic_path": "audio_mic.wav",
            "system_path": "audio_sys.wav",
            "offsets_s": [0.0, 0.045],
            "sample_rates": [44100, 48000],
            "headphones_confirmed": True,
        }
        session = Session(
            name="Sessione Duale v4",
            schema_version=4,
            pipeline_strategy="source_single_speaker",
            capture_manifest=manifest,
            active_view="timeline",
            timeline_visible_speakers=["SPEAKER_MIC", "SPEAKER_SYS"],
        )
        self.assertEqual(session.schema_version, 4)
        self.assertEqual(session.pipeline_strategy, "source_single_speaker")
        self.assertEqual(session.active_view, "timeline")
        self.assertEqual(session.capture_manifest["headphones_confirmed"], True)

        dumped = session.to_dict()
        restored = Session.from_dict(dumped)
        self.assertEqual(restored.schema_version, 4)
        self.assertEqual(restored.pipeline_strategy, "source_single_speaker")
        self.assertEqual(restored.active_view, "timeline")
        self.assertEqual(restored.capture_manifest, manifest)
        self.assertEqual(restored.timeline_visible_speakers, ["SPEAKER_MIC", "SPEAKER_SYS"])

    def test_undo_redo_preserves_view_and_strategy(self):
        """Undo e redo preservano le proprietà di visualizzazione e strategia della sessione."""
        session = Session(
            name="Sessione Undo Test",
            segments=[Segment(start=0.0, end=1.0, text="Ciao", speaker="S0", id="s1")],
            pipeline_strategy="source_single_speaker",
            active_view="timeline",
        )
        session.edit_segment(0, text="Modificato")
        self.assertEqual(session.segments[0].text, "Modificato")
        self.assertEqual(session.active_view, "timeline")

        # Undo
        self.assertTrue(session.undo())
        self.assertEqual(session.segments[0].text, "Ciao")
        self.assertEqual(session.pipeline_strategy, "source_single_speaker")
        self.assertEqual(session.active_view, "timeline")

        # Redo
        self.assertTrue(session.redo())
        self.assertEqual(session.segments[0].text, "Modificato")
        self.assertEqual(session.pipeline_strategy, "source_single_speaker")
        self.assertEqual(session.active_view, "timeline")

    def test_fingerprint_detects_strategy_and_view_change(self):
        """Il fingerprint riflette variazioni nella strategia o nella vista attiva."""
        s = Session(name="Fingerprint Test", schema_version=4, active_view="conversation")
        fp1 = s.fingerprint()

        s.active_view = "timeline"
        fp2 = s.fingerprint()
        self.assertNotEqual(fp1, fp2, "Cambiare active_view deve modificare il fingerprint")

        s.pipeline_strategy = "source_single_speaker"
        fp3 = s.fingerprint()
        self.assertNotEqual(fp2, fp3, "Cambiare pipeline_strategy deve modificare il fingerprint")


if __name__ == "__main__":
    unittest.main()
