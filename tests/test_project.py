"""Progetti: stato nella cartella, coda degli audio e memoria per il modello.

Esecuzione: python -m tests.test_project
"""
import tempfile
import unittest
from pathlib import Path

from src.models.project import (
    PROJECT_FILE,
    Project,
    load_project,
    open_or_create,
)
from src.models.session import Segment, Session


class ProjectStateTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)

    def _audio(self, *names):
        for name in names:
            (self.folder / name).write_bytes(b"finto")

    def test_apre_o_crea_e_rilegge_dalla_cartella(self):
        project = open_or_create(self.folder, "Campagna")
        project.description = "Sessioni di gioco del martedì"
        project.set_glossary("Xatrag, piano dell'oblio")
        project.save()

        reread = load_project(self.folder)

        self.assertEqual(reread.name, "Campagna")
        self.assertEqual(reread.glossary, ["Xatrag", "piano dell'oblio"])
        self.assertTrue((self.folder / PROJECT_FILE).is_file())

    def test_riaprire_non_sovrascrive_il_progetto_esistente(self):
        open_or_create(self.folder, "Primo")

        again = open_or_create(self.folder, "Nome diverso")

        self.assertEqual(again.name, "Primo")

    def test_elenca_gli_audio_in_ordine_prevedibile(self):
        self._audio("b.mp3", "a.wav", "c.m4a", "appunti.txt")
        project = open_or_create(self.folder, "P")

        names = [item.name for item in project.audio_files()]

        self.assertEqual(names, ["a.wav", "b.mp3", "c.m4a"])

    def test_i_file_gia_elaborati_escono_dalla_coda(self):
        self._audio("uno.wav", "due.wav")
        project = open_or_create(self.folder, "P")
        session = Session(name="uno", segments=[Segment(0, 1, "ciao", "SPEAKER_00")])

        project.remember_session(self.folder / "uno.wav", session)

        self.assertEqual([item.name for item in project.pending_audio()], ["due.wav"])
        self.assertEqual(len(project.sessions), 1)

    def test_rielaborare_lo_stesso_audio_aggiorna_invece_di_duplicare(self):
        self._audio("uno.wav")
        project = open_or_create(self.folder, "P")
        session = Session(name="uno", segments=[Segment(0, 1, "ciao", "SPEAKER_00")])
        project.remember_session(self.folder / "uno.wav", session, summary="prima versione")

        project.remember_session(self.folder / "uno.wav", session, summary="seconda versione")

        self.assertEqual(len(project.sessions), 1)
        self.assertEqual(project.sessions[0].summary, "seconda versione")

    def test_i_nomi_assegnati_diventano_voci_del_progetto(self):
        self._audio("uno.wav")
        project = open_or_create(self.folder, "P")
        session = Session(segments=[Segment(0, 1, "ciao", "SPEAKER_00")])
        session.rename_speaker("SPEAKER_00", "Marco")
        session.rename_speaker("SPEAKER_01", "Anna")

        project.remember_session(self.folder / "uno.wav", session)

        # L'ordine e' quello in cui i nomi sono stati assegnati, non alfabetico:
        # e' lo stesso ordine usato per proporli alla seduta successiva.
        self.assertEqual(project.speakers, ["Marco", "Anna"])

    def test_i_percorsi_salvati_sono_relativi_alla_cartella(self):
        self._audio("uno.wav")
        project = open_or_create(self.folder, "P")
        session = Session(segments=[Segment(0, 1, "ciao")])

        project.remember_session(self.folder / "uno.wav", session)
        project.save()

        stored = load_project(self.folder)
        self.assertEqual(stored.sessions[0].audio, "uno.wav")
        self.assertEqual(stored.resolve("uno.wav"), self.folder / "uno.wav")


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)

    def _project(self):
        project = open_or_create(self.folder, "Campagna")
        project.description = "Gioco di ruolo, quattro giocatori"
        project.set_glossary("Xatrag, Kelial")
        project.add_speakers(["Master", "Roghur"])
        return project

    def test_il_contesto_contiene_descrizione_glossario_e_voci(self):
        memory = self._project().memory_text()

        self.assertIn("Campagna", memory)
        self.assertIn("Gioco di ruolo", memory)
        self.assertIn("Xatrag", memory)
        self.assertIn("Master", memory)

    def test_solo_le_ultime_sedute_entrano_nel_contesto(self):
        project = self._project()
        for index in range(12):
            session = Session(name=f"seduta {index}", segments=[Segment(0, 1, "x")])
            project.remember_session(self.folder / f"{index}.wav", session,
                                     summary=f"riassunto {index}")

        memory = project.memory_text(limit=3)

        self.assertIn("riassunto 11", memory)
        self.assertNotIn("riassunto 5", memory)

    def test_il_glossario_per_whisper_unisce_termini_e_voci_senza_ripetizioni(self):
        project = self._project()
        project.add_speakers(["Kelial"])  # gia' presente nel glossario

        prompt = project.glossary_prompt()

        self.assertEqual(prompt.count("Kelial"), 1)
        self.assertIn("Xatrag", prompt)

    def test_i_nomi_si_propongono_solo_se_il_numero_di_voci_coincide(self):
        project = self._project()   # due voci: Master, Roghur
        due = Session(segments=[Segment(0, 1, "a", "SPEAKER_00"),
                                Segment(1, 2, "b", "SPEAKER_01")])
        tre = Session(segments=[Segment(0, 1, "a", "SPEAKER_00"),
                                Segment(1, 2, "b", "SPEAKER_01"),
                                Segment(2, 3, "c", "SPEAKER_02")])

        self.assertEqual(project.suggested_names(due),
                         {"SPEAKER_00": "Master", "SPEAKER_01": "Roghur"})
        self.assertEqual(project.suggested_names(tre), {})


class ProjectAdaptiveLearningTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)

    def _corrected_session(self):
        session = Session(
            name="seduta",
            segments=[Segment(0, 2, "Incontriamo Chisrael", "SPEAKER_00")],
            raw_segments=[Segment(0, 2, "Incontriamo Chisrael", "SPEAKER_00")],
        )
        session.edit_segment(0, text="Incontriamo Kisrael")
        session.speaker_names["SPEAKER_00"] = "Marco"
        session.provenance["confirmed_speakers"] = ["SPEAKER_00"]
        session.diarization["speaker_embeddings"] = {"SPEAKER_00": [1.0, 0.0]}
        return session

    def test_correzione_e_voce_confermata_persistono_e_vengono_riusate(self):
        project = open_or_create(self.folder, "P")
        session = self._corrected_session()

        report = project.learn_from_session(session)
        project.save()
        reread = load_project(self.folder)
        future = Session(segments=[Segment(0, 1, "ciao", "SPEAKER_07")])
        future.diarization["speaker_embeddings"] = {"SPEAKER_07": [0.999, 0.01]}

        self.assertTrue(report["changed"])
        self.assertIn("Kisrael", reread.glossary_prompt())
        self.assertEqual(reread.suggested_names(future), {"SPEAKER_07": "Marco"})

    def test_annulla_rimuove_il_contributo_della_correzione(self):
        project = open_or_create(self.folder, "P")
        session = self._corrected_session()
        project.learn_from_session(session)

        self.assertTrue(session.undo())
        report = project.learn_from_session(session)

        self.assertTrue(report["changed"])
        self.assertNotIn("Kisrael", project.glossary_prompt())
        self.assertNotIn(session.id, project.learning_sessions)

    def test_profili_di_revisioni_diverse_non_vengono_confrontati(self):
        project = open_or_create(self.folder, "P")
        session = self._corrected_session()
        session.diarization["embedding_model"] = {
            "repo": "pyannote/community", "revision": "rev-a",
        }
        project.learn_from_session(session)
        future = Session(segments=[Segment(0, 1, "ciao", "SPEAKER_07")])
        future.diarization = {
            "speaker_embeddings": {"SPEAKER_07": [1.0, 0.0]},
            "embedding_model": {"repo": "pyannote/community", "revision": "rev-b"},
        }

        self.assertEqual(project.suggested_names(future), {})

    def test_preferenze_analisi_sono_esplicite_limitate_e_disattivabili(self):
        project = open_or_create(self.folder, "P")

        self.assertTrue(project.remember_analysis_feedback(
            "Distingui sempre fatti e ipotesi"
        ))
        self.assertIn("Distingui sempre", project.memory_text())
        project.learning_enabled = False
        self.assertFalse(project.remember_analysis_feedback("Nuova regola"))
        self.assertNotIn("Preferenze di analisi", project.memory_text())

    def test_azzera_apprendimento_senza_toccare_glossario_manuale(self):
        project = open_or_create(self.folder, "P")
        project.set_glossary("Kisrael")
        project.learn_from_session(self._corrected_session())
        project.remember_analysis_feedback("Rispondi in modo schematico")

        project.clear_learning()

        self.assertEqual(project.learning_sessions, {})
        self.assertEqual(project.analysis_preferences, [])
        self.assertEqual(project.glossary, ["Kisrael"])

    def test_vecchio_schema_e_memoria_malformata_migrano_in_sicurezza(self):
        project = Project.from_dict({
            "name": "P",
            "learning_sessions": {
                "rotta": {"speaker_embeddings": {"X": [float("nan"), 1]}},
                "tipo-errato": {"transcription_corrections": {"after": "no"}},
                "buona": {"transcription_corrections": [
                    {"before": "Chisrael", "after": "Kisrael"}
                ]},
            },
            "learning_revision": "non-un-numero",
        }, self.folder)

        self.assertTrue(project.learning_enabled)
        self.assertNotIn("rotta", project.learning_sessions)
        self.assertNotIn("tipo-errato", project.learning_sessions)
        self.assertIn("buona", project.learning_sessions)
        self.assertEqual(project.learning_revision, 0)


class ProjectValidationTests(unittest.TestCase):
    def test_un_progetto_senza_nome_viene_rifiutato(self):
        with self.assertRaises(ValueError):
            Project.from_dict({"name": "   "}, "/tmp")

    def test_una_seduta_malformata_viene_rifiutata(self):
        with self.assertRaises(ValueError):
            Project.from_dict({"name": "P", "sessions": [{"nope": 1}]}, "/tmp")

    def test_cartella_senza_progetto(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_project(folder)


if __name__ == "__main__":
    unittest.main()
