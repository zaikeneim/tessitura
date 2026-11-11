"""Test per lo schema di sessione v3, migrazione v2->v3 e CheckpointStore esteso (P10).

Verifica la migrazione automatica senza perdita di dati, la proiezione coerente
fra Utterance e Segment, la persistenza delle tracce audio/revisioni e
l'invalidazione a cascata per dipendenze nei checkpoint.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.session import Segment, Session, Word
from src.models.transcript_artifacts import Utterance
from src.processing.checkpoints import CheckpointStore


class SessionV3MigrationTests(unittest.TestCase):
    def test_v2_to_v3_migration(self):
        """Una sessione con schema_version=2 viene migrata a v3 con generazione di utterances."""
        v2_data = {
            "schema_version": 2,
            "id": "sess_v2_test",
            "name": "Sessione Test V2",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Buongiorno a tutti",
                    "speaker": "SPEAKER_00",
                    "id": "seg_01",
                    "words": [{"start": 0.0, "end": 1.0, "text": "Buongiorno"}, {"start": 1.0, "end": 2.5, "text": "a tutti"}],
                    "words_valid": True,
                    "warnings": [],
                    "candidates": [],
                    "source_ids": ["hyp_01"],
                },
                {
                    "start": 2.6,
                    "end": 4.0,
                    "text": "Benvenuti",
                    "speaker": "SPEAKER_01",
                    "id": "seg_02",
                    "words": [{"start": 2.6, "end": 4.0, "text": "Benvenuti"}],
                    "words_valid": True,
                    "warnings": [],
                    "candidates": [],
                    "source_ids": [],
                },
            ],
            "speaker_names": {"SPEAKER_00": "Mario"},
        }

        session = Session.from_dict(v2_data)
        self.assertEqual(session.schema_version, 2, "La sessione caricata da v2 conserva inizialmente schema_version=2")
        self.assertEqual(len(session.utterances), 2, "Le utterances devono essere generate dai segmenti v2")

        # Migrazione esplicita a v3
        session.migrate_to_v3()
        self.assertEqual(session.schema_version, 3, "La sessione migrata deve avere schema_version=3")
        self.assertEqual(session.utterances[0].text, "Buongiorno a tutti")
        self.assertEqual(session.utterances[0].speaker, "SPEAKER_00")
        self.assertEqual(session.utterances[0].utterance_id, "seg_01")
        self.assertEqual(session.utterances[0].hypothesis_ids, ["hyp_01"])
        self.assertEqual(len(session.utterances[0].words), 2)

        # Roundtrip to_dict
        dumped = session.to_dict()
        self.assertEqual(dumped["schema_version"], 3)
        self.assertEqual(len(dumped["utterances"]), 2)
        self.assertEqual(len(dumped["segments"]), 2)

    def test_utterances_to_segments_projection(self):
        """Se vengono fornite solo utterances, segments viene proiettato fedelmente."""
        v3_data = {
            "schema_version": 3,
            "id": "sess_v3_direct",
            "name": "Sessione Diretta V3",
            "segments": [],
            "utterances": [
                {
                    "start_s": 5.0,
                    "end_s": 7.0,
                    "utterance_id": "u_direct",
                    "stream_id": 0,
                    "speaker": "SPEAKER_00",
                    "text": "Traccia parlata diretta",
                    "words": [{"start": 5.0, "end": 7.0, "text": "Traccia parlata diretta"}],
                }
            ],
        }

        session = Session.from_dict(v3_data)
        self.assertEqual(len(session.segments), 1)
        self.assertEqual(session.segments[0].text, "Traccia parlata diretta")
        self.assertEqual(session.segments[0].start, 5.0)
        self.assertEqual(session.segments[0].end, 7.0)

    def test_undo_redo_preserves_utterances(self):
        """Undo e redo preservano coerentemente sia segmenti che utterances."""
        session = Session(
            id="sess_undo",
            name="Test Undo",
            segments=[Segment(0.0, 1.0, "Prima", "SPEAKER_00")],
            utterances=[Utterance(0.0, 1.0, speaker="SPEAKER_00", text="Prima")],
        )

        session.begin_edit("Modifica testo")
        session.segments[0].text = "Dopo"
        session.utterances[0].text = "Dopo"

        self.assertEqual(session.segments[0].text, "Dopo")
        self.assertEqual(session.utterances[0].text, "Dopo")

        # Undo
        self.assertTrue(session.undo())
        self.assertEqual(session.segments[0].text, "Prima")
        self.assertEqual(session.utterances[0].text, "Prima")

        # Redo
        self.assertTrue(session.redo())
        self.assertEqual(session.segments[0].text, "Dopo")
        self.assertEqual(session.utterances[0].text, "Dopo")


class CheckpointDependenciesTests(unittest.TestCase):
    def test_stage_dependencies_invalidation(self):
        """Il caricamento di uno stadio con dipendenze fallisce se la dipendenza è mancante o invalidata."""
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder)
            key = "f" * 64

            # Salva stadio prepare
            store.save(key, "prepare", {"info": "audio_ok"})

            # Salva stadio asr dipendente da prepare
            store.save(key, "asr", {"text": "Trascrizione ok"}, depends_on=["prepare"])

            # Entrambi si caricano
            self.assertIsNotNone(store.load(key, "prepare"))
            self.assertIsNotNone(store.load(key, "asr"))

            # Se prepare viene rimosso/invalidato
            store.invalidate(key, "prepare")

            # asr non deve piu' essere valido perchè la sua dipendenza prepare non esiste più
            self.assertIsNone(store.load(key, "asr"))


if __name__ == "__main__":
    unittest.main()
