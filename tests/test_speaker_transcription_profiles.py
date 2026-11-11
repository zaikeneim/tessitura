"""Test per l'estrazione dei profili di trascrizione per voce ed esportazione."""

import tempfile
import unittest
from pathlib import Path

from src.models.session import AudioSources, Segment, Session
from src.models.storage import export_all_speaker_transcripts, export_txt_by_speaker, transcript_lines
from src.models.transcript_artifacts import Utterance
from src.processing.diarize import Turn
from src.processing.merge import utterances_to_segments
from src.processing.voice_profile import (
    VoiceProfile,
    build_voice_profiles,
    enrich_voice_profiles_with_transcript,
)


class SpeakerTranscriptionProfileTests(unittest.TestCase):
    def test_enrich_voice_profiles_ambient_audio(self):
        """Verifica estrazione del profilo di trascrizione per ogni voce su audio ambientali."""
        turns = [
            Turn(0.0, 4.0, "SPEAKER_00"),
            Turn(4.5, 8.0, "SPEAKER_01"),
            Turn(8.5, 12.0, "SPEAKER_00"),
        ]
        profiles = build_voice_profiles(turns)
        self.assertEqual(len(profiles), 2)

        segments = [
            Segment(start=0.5, end=3.5, text="Buongiorno a tutti i presenti", speaker="SPEAKER_00"),
            Segment(start=4.8, end=7.2, text="Ciao Marco buongiorno", speaker="SPEAKER_01"),
            Segment(start=9.0, end=11.5, text="Iniziamo la riunione di oggi", speaker="SPEAKER_00"),
        ]

        enrich_voice_profiles_with_transcript(profiles, segments)

        p0 = profiles["SPEAKER_00"]
        self.assertIn("Buongiorno a tutti i presenti", p0.transcript)
        self.assertIn("Iniziamo la riunione di oggi", p0.transcript)
        self.assertEqual(p0.word_count, 10)
        self.assertGreater(p0.speaking_ratio, 0.4)
        self.assertEqual(p0.turn_intervals, [[0.5, 3.5], [9.0, 11.5]])

        p1 = profiles["SPEAKER_01"]
        self.assertEqual(p1.transcript, "Ciao Marco buongiorno")
        self.assertEqual(p1.word_count, 3)
        self.assertGreater(p1.speaking_ratio, 0.1)
        self.assertEqual(p1.turn_intervals, [[4.8, 7.2]])

    def test_session_speaker_transcription_profile_methods(self):
        """Verifica speaker_transcription_profile e all_speaker_transcription_profiles in Session."""
        vp0 = VoiceProfile(
            speaker_id="SPEAKER_00",
            name="Alice",
            total_duration=10.0,
            turn_count=2,
            transcript="Prima frase Alice\nSeconda frase Alice",
            word_count=6,
            speaking_ratio=0.6,
            turn_intervals=[[0.0, 5.0], [10.0, 15.0]],
        )
        session = Session(
            audio=AudioSources(mixed="test.mp3"),
            segments=[
                Segment(start=0.5, end=4.5, text="Prima frase Alice", speaker="SPEAKER_00"),
                Segment(start=5.0, end=9.0, text="Intervento Bob", speaker="SPEAKER_01"),
                Segment(start=10.5, end=14.5, text="Seconda frase Alice", speaker="SPEAKER_00"),
            ],
            speaker_names={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
            diarization={"voice_profiles": {"SPEAKER_00": vp0.to_dict()}},
        )

        # SPEAKER_00 ha profilo esplicito
        prof0 = session.speaker_transcription_profile("SPEAKER_00")
        self.assertEqual(prof0["speaker_id"], "SPEAKER_00")
        self.assertEqual(prof0["name"], "Alice")
        self.assertEqual(prof0["word_count"], 6)
        self.assertEqual(prof0["turn_count"], 2)

        # SPEAKER_01 non era in voice_profiles ma viene ricostruito dai segmenti
        prof1 = session.speaker_transcription_profile("SPEAKER_01")
        self.assertEqual(prof1["speaker_id"], "SPEAKER_01")
        self.assertEqual(prof1["name"], "Bob")
        self.assertEqual(prof1["transcript"], "Intervento Bob")
        self.assertEqual(prof1["word_count"], 2)

        all_profs = session.all_speaker_transcription_profiles()
        self.assertIn("SPEAKER_00", all_profs)
        self.assertIn("SPEAKER_01", all_profs)

    def test_export_txt_by_speaker_and_all(self):
        """Verifica l'esportazione delle trascrizioni per singola voce in storage.py."""
        session = Session(
            audio=AudioSources(mixed="ambient.mp3"),
            segments=[
                Segment(start=1.0, end=4.0, text="Intervento introduttivo di Alice", speaker="SPEAKER_00"),
                Segment(start=5.0, end=8.0, text="Replica puntuale di Bob", speaker="SPEAKER_01"),
                Segment(start=9.0, end=12.0, text="Chiusura di Alice", speaker="SPEAKER_00"),
            ],
            speaker_names={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
        )

        txt_alice = export_txt_by_speaker(session, "SPEAKER_00")
        self.assertIn("PROFILO TRASCRIZIONE VOCE: Alice (SPEAKER_00)", txt_alice)
        self.assertIn("Parole totali: 7", txt_alice)
        self.assertIn("[00:00:01 - 00:00:04] Intervento introduttivo di Alice", txt_alice)
        self.assertIn("[00:00:09 - 00:00:12] Chiusura di Alice", txt_alice)
        self.assertNotIn("Replica puntuale di Bob", txt_alice)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir)
            files = export_all_speaker_transcripts(session, out_p)
            self.assertEqual(len(files), 2)
            self.assertTrue(any("Alice" in p.name for p in files))
            self.assertTrue(any("Bob" in p.name for p in files))

    def test_transcript_lines_theater_script_overlap(self):
        """Verifica il formato copione teatrale [In contemporanea] per segmenti sovrapposti."""
        session = Session(
            audio=AudioSources(mixed="test.wav"),
            segments=[
                # Non sovrapposto
                Segment(start=0.0, end=3.0, text="Inizio discorso", speaker="SPEAKER_00"),
                # Due segmenti sovrapposti temporalmente
                Segment(start=4.0, end=8.0, text="Parlo io mentre lui si sovrappone", speaker="SPEAKER_00"),
                Segment(start=5.0, end=7.0, text="Mi sovrappongo qui", speaker="SPEAKER_01"),
                # Non sovrapposto successivo
                Segment(start=9.0, end=12.0, text="Fine discorso", speaker="SPEAKER_01"),
            ],
            speaker_names={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
        )

        lines = transcript_lines(session)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "[Alice 00:00:00] Inizio discorso")
        self.assertIn("[In contemporanea]:", lines[1])
        self.assertIn("- Alice (00:00:04): Parlo io mentre lui si sovrappone", lines[1])
        self.assertIn("- Bob (00:00:05): Mi sovrappongo qui", lines[1])
        self.assertEqual(lines[2], "[Bob 00:00:09] Fine discorso")

    def test_utterances_to_segments_id_synchronization(self):
        """Verifica che utterance_id sia propagato deterministicamente come id del Segment."""
        u1 = Utterance(
            start_s=1.0,
            end_s=3.0,
            utterance_id="utt_fixed_id_12345",
            stream_id=0,
            speaker="SPEAKER_00",
            text="Test sincronizzazione ID",
        )
        segs = utterances_to_segments([u1])
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].id, "utt_fixed_id_12345")
        self.assertEqual(segs[0].text, "Test sincronizzazione ID")


if __name__ == "__main__":
    unittest.main()
