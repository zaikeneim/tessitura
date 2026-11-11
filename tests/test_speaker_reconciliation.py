"""Test per la riconciliazione delle identita' globali e profili robusti (P05).

Scenari obbligatori dal piano esecutivo:
- Due finestre con etichette locali permutate -> identita' globali stabili
- Matching ambiguo -> assegnazione a UNKNOWN
- Voce nuova -> creazione di nuovo profilo globale senza forzature
- Filtraggio turni puliti -> esclusione di sovrapposizioni e turni troppo brevi
"""

import unittest
from src.processing.diarize import Turn
from src.processing.voice_profile import (
    VoiceProfile,
    filter_clean_speech_turns,
    reconcile_window_speakers,
)


class SpeakerReconciliationTests(unittest.TestCase):
    def test_local_id_permutations_preserve_global_identities(self):
        # Embedding ortogonali normalizzati
        emb_alice = [1.0, 0.0, 0.0, 0.0]
        emb_bob = [0.0, 1.0, 0.0, 0.0]

        # Finestra 0: SPEAKER_00 e' Alice, SPEAKER_01 e' Bob
        win0 = {
            "SPEAKER_00": VoiceProfile(speaker_id="SPEAKER_00", embedding=emb_alice, total_duration=5.0),
            "SPEAKER_01": VoiceProfile(speaker_id="SPEAKER_01", embedding=emb_bob, total_duration=4.0),
        }

        # Finestra 1: etichette permutate! SPEAKER_00 e' Bob, SPEAKER_01 e' Alice
        win1 = {
            "SPEAKER_00": VoiceProfile(speaker_id="SPEAKER_00", embedding=emb_bob, total_duration=6.0),
            "SPEAKER_01": VoiceProfile(speaker_id="SPEAKER_01", embedding=emb_alice, total_duration=5.5),
        }

        mapping = reconcile_window_speakers([win0, win1], similarity_threshold=0.75)

        # In entrambe le finestre, Alice deve puntare allo stesso global ID
        global_alice_win0 = mapping.local_to_global[(0, "SPEAKER_00")]
        global_alice_win1 = mapping.local_to_global[(1, "SPEAKER_01")]
        self.assertEqual(global_alice_win0, global_alice_win1)

        # Bob deve puntare allo stesso global ID
        global_bob_win0 = mapping.local_to_global[(0, "SPEAKER_01")]
        global_bob_win1 = mapping.local_to_global[(1, "SPEAKER_00")]
        self.assertEqual(global_bob_win0, global_bob_win1)

        # Alice e Bob hanno global ID distinti
        self.assertNotEqual(global_alice_win0, global_bob_win0)

    def test_ambiguous_similarity_produces_unknown(self):
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0]
        win0 = {
            "SPEAKER_00": VoiceProfile(speaker_id="SPEAKER_00", embedding=emb_a),
            "SPEAKER_01": VoiceProfile(speaker_id="SPEAKER_01", embedding=emb_b),
        }

        # Voce perfettamente intermedia: sim a = 0.707, sim b = 0.707 (ambigua)
        emb_mid = [0.7071, 0.7071, 0.0, 0.0]
        win1 = {
            "SPEAKER_X": VoiceProfile(speaker_id="SPEAKER_X", embedding=emb_mid),
        }

        mapping = reconcile_window_speakers(
            [win0, win1], similarity_threshold=0.65, ambiguity_margin=0.08
        )
        self.assertEqual(mapping.local_to_global[(1, "SPEAKER_X")], "UNKNOWN")

    def test_new_speaker_creates_distinct_global_profile(self):
        emb_known = [1.0, 0.0, 0.0, 0.0]
        emb_new = [0.0, 0.0, 1.0, 0.0]  # ortogonale

        win0 = {"SPEAKER_00": VoiceProfile(speaker_id="SPEAKER_00", embedding=emb_known)}
        win1 = {"SPEAKER_NEW": VoiceProfile(speaker_id="SPEAKER_NEW", embedding=emb_new)}

        mapping = reconcile_window_speakers([win0, win1], similarity_threshold=0.70)
        gid_known = mapping.local_to_global[(0, "SPEAKER_00")]
        gid_new = mapping.local_to_global[(1, "SPEAKER_NEW")]

        self.assertNotEqual(gid_known, gid_new)
        self.assertIn(gid_new, mapping.global_speakers)

    def test_clean_speech_turns_filtering(self):
        turns = [
            Turn(start=0.0, end=0.4, speaker="A"),  # troppo breve (< 0.8s) -> escluso
            Turn(start=1.0, end=2.2, speaker="A"),  # pulito (1.2s, nessun overlap) -> tenuto
            Turn(start=3.0, end=4.5, speaker="A"),  # overlap con B (4.0 -> 4.5) -> escluso
            Turn(start=4.0, end=5.5, speaker="B"),  # overlap con A (4.0 -> 4.5) -> escluso
            Turn(start=6.0, end=8.0, speaker="B"),  # pulito (2.0s, nessun overlap) -> tenuto
        ]

        clean = filter_clean_speech_turns(turns, min_duration_s=0.8, exclude_overlaps=True)
        self.assertEqual(len(clean), 2)
        self.assertEqual(clean[0].start, 1.0)
        self.assertEqual(clean[1].start, 6.0)


if __name__ == "__main__":
    unittest.main()
