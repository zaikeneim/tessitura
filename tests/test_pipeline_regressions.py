"""Regressioni sui difetti trovati eseguendo davvero il percorso principale.

Tutti questi giravano con la suite verde: i test esistenti coprivano il percorso
a voce singola, mentre i difetti stavano nel ramo che si attiva quando la
diarizzazione trova piu' di una voce, cioe' il caso d'uso normale.

Scenari obbligatori:
- Il ramo con piu' voci non solleva NameError (audio_assets mai inizializzato)
- I profili vocali si arricchiscono senza TypeError e il risultato non si perde
- La sessione dichiara la strategia che il chiamante ha chiesto
- Nessuna parola sparisce quando etichette e parole non combaciano
- La riconciliazione non riscrive gli oggetti del chiamante

Esecuzione: python -m tests.test_pipeline_regressions
"""

import copy
import inspect
import unittest

from src.models.session import Segment, Word
from src.models.transcript_artifacts import Utterance
from src.processing.merge import WordUnit, build_segments
from src.processing.transcript_reconciler import reconcile_segments, reconcile_utterances
from src.processing.voice_profile import enrich_voice_profiles_with_transcript


def word(start: float, end: float, text: str) -> Word:
    return Word(start=start, end=end, text=text, probability=0.9)


class BuildSegmentsTests(unittest.TestCase):
    """`zip` troncava sulla lista piu' corta, facendo sparire le ultime parole."""

    def setUp(self):
        self.frase = "una frase che arriva fino in fondo".split()
        self.units = [
            WordUnit(i * 0.3, i * 0.3 + 0.28, w, 0.9) for i, w in enumerate(self.frase)
        ]

    def _parole(self, segments) -> list[str]:
        return " ".join(s.text for s in segments).split()

    def test_etichette_mancanti_non_fanno_sparire_parole(self):
        segments = build_segments(self.units, ["A"] * 4, autocorrect_enabled=False)

        self.assertEqual(self._parole(segments), self.frase)

    def test_etichette_in_eccesso_non_fanno_sparire_parole(self):
        segments = build_segments(self.units, ["A"] * 12, autocorrect_enabled=False)

        self.assertEqual(self._parole(segments), self.frase)

    def test_il_disallineamento_viene_dichiarato(self):
        with self.assertLogs("src.processing.merge", level="WARNING"):
            build_segments(self.units, ["A"] * 2, autocorrect_enabled=False)

    def test_lunghezze_pari_restano_intatte(self):
        segments = build_segments(self.units, ["A"] * len(self.units), autocorrect_enabled=False)

        self.assertEqual(self._parole(segments), self.frase)
        self.assertEqual(len(segments), 1)


class EnrichVoiceProfilesTests(unittest.TestCase):
    """La durata e' keyword-only e il risultato e' un dizionario nuovo."""

    def test_la_durata_non_si_passa_posizionalmente(self):
        parameters = inspect.signature(enrich_voice_profiles_with_transcript).parameters

        self.assertEqual(parameters["total_duration"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_i_profili_tornano_in_un_dizionario_nuovo(self):
        segments = [Segment(0.0, 1.0, "ciao", "A", [word(0.0, 1.0, "ciao")])]
        origine: dict = {}

        risultato = enrich_voice_profiles_with_transcript(
            origine, segments, total_duration=1.0
        )

        self.assertIsNot(risultato, origine, "il chiamante deve riassegnare il valore")
        self.assertIn("A", risultato)


class ReconcilerTests(unittest.TestCase):
    """La riconciliazione riscriveva testo e confini negli oggetti del chiamante."""

    def _segmenti(self) -> list[Segment]:
        return [
            Segment(0.0, 2.0, "ciao come stai", "A",
                    [word(0.0, 1.0, "ciao"), word(1.0, 1.5, "come"), word(1.5, 2.0, "stai")]),
            Segment(1.8, 3.0, "stai bene", "A",
                    [word(1.8, 2.2, "stai"), word(2.2, 3.0, "bene")]),
        ]

    def _interventi(self) -> list[Utterance]:
        return [
            Utterance(start_s=0.0, end_s=2.0, utterance_id="u1", stream_id=0,
                      overlap_group_id=None, speaker="A", candidate_speakers=[],
                      text="ciao come stai",
                      words=[word(0.0, 1.0, "ciao"), word(1.0, 1.5, "come"), word(1.5, 2.0, "stai")],
                      hypothesis_ids=["h1"]),
            Utterance(start_s=1.8, end_s=3.0, utterance_id="u2", stream_id=0,
                      overlap_group_id=None, speaker="A", candidate_speakers=[],
                      text="stai bene",
                      words=[word(1.8, 2.2, "stai"), word(2.2, 3.0, "bene")],
                      hypothesis_ids=["h2"]),
        ]

    def test_i_segmenti_in_ingresso_restano_intatti(self):
        segments = self._segmenti()
        prima = copy.deepcopy(segments)

        reconcile_segments(segments)

        self.assertEqual([s.text for s in segments], [s.text for s in prima])
        self.assertEqual([s.end for s in segments], [s.end for s in prima])

    def test_gli_interventi_in_ingresso_restano_intatti(self):
        utterances = self._interventi()
        prima = copy.deepcopy(utterances)

        reconcile_utterances(utterances)

        self.assertEqual([u.text for u in utterances], [u.text for u in prima])
        self.assertEqual([u.hypothesis_ids for u in utterances],
                         [u.hypothesis_ids for u in prima])

    def test_la_fusione_avviene_comunque(self):
        fusi = reconcile_segments(self._segmenti())

        self.assertEqual(fusi[0].text, "ciao come stai bene")


class SpeakerTracksTests(unittest.TestCase):
    """Il margine sconfinava nel turno accanto e duplicava il parlato."""

    def test_il_margine_non_invade_il_turno_altrui(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        import soundfile as sf

        from src.processing.speaker_tracks import extract_speaker_tracks

        rate, seconds = 16000, 6.0
        t = np.arange(int(seconds * rate)) / rate
        # A parla 0-2s, B parla 2-4s: confini netti, nessuna sovrapposizione vera.
        mix = np.where((t >= 0) & (t < 2.0), 0.3 * np.sin(2 * np.pi * 200 * t), 0.0)
        mix += np.where((t >= 2.0) & (t < 4.0), 0.3 * np.sin(2 * np.pi * 400 * t), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mix.wav"
            sf.write(str(path), mix.astype(np.float32), rate)
            result = extract_speaker_tracks(path, turns=[
                SimpleNamespace(speaker="A", start=0.0, end=2.0),
                SimpleNamespace(speaker="B", start=2.0, end=4.0),
            ])

        self.assertTrue(result.separated)
        # Nel turno di B la traccia di A deve tacere, altrimenti lo stesso
        # parlato viene trascritto due volte e attribuito a entrambi.
        track_a = result.tracks[result.speaker_ids.index("A")]
        intruso = track_a[int(2.3 * rate) : int(3.7 * rate)]
        self.assertLess(float(np.abs(intruso).max()), 1e-6)


class MultiVoiceBranchTests(unittest.TestCase):
    """Il ramo con piu' voci: due crash certi che nessun test copriva.

    Si attiva quando la diarizzazione trova almeno due parlanti con
    `profile_first` attivo, cioe' la configurazione predefinita.
    """

    def _esegui(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        import numpy as np
        import soundfile as sf

        from src.processing.audio_profile import AudioProfile
        from src.processing.pipeline import PipelineOptions, process_recording
        from src.processing.speaker_tracks import SpeakerTracksResult

        rate = 16000
        rumore = (0.2 * np.random.default_rng(0).standard_normal(rate * 4)).astype(np.float32)
        turni = [
            SimpleNamespace(speaker="SPEAKER_00", start=0.0, end=2.0),
            SimpleNamespace(speaker="SPEAKER_01", start=2.0, end=4.0),
        ]
        diarizzazione = MagicMock(turns=turni, exclusive=turni, overlaps=[], speaker_embeddings={})
        trascritto = [SimpleNamespace(start=0.0, end=1.0, text="ciao",
                                      words=[(0.0, 1.0, "ciao", 0.9)])]

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        audio = root / "voci.wav"
        sf.write(str(audio), rumore, rate)

        with patch("src.processing.diarize.Diarizer") as diarizer, patch(
            "src.processing.speaker_tracks.extract_speaker_tracks"
        ) as estrai, patch("src.processing.speaker_tracks.Transcriber") as asr, patch(
            "src.processing.pipeline.resolve_device", return_value=("cpu", "int8")
        ):
            diarizer.return_value.diarize_detailed.return_value = diarizzazione
            estrai.return_value = SpeakerTracksResult(
                tracks=[rumore, rumore],
                speaker_ids=["SPEAKER_00", "SPEAKER_01"],
                status="sliced",
            )
            asr.return_value.transcribe.return_value = (trascritto, "it")
            return process_recording(
                mic_path=audio,
                options=PipelineOptions(
                    strategy="standard", denoise=False, diarize=True, profile_first=True,
                    audio_profile=AudioProfile.STEREO_AMBIENT.value, analysis_mode="off",
                    num_speakers=2,
                ),
                work_dir=root / "lavoro",
            )

    def test_il_ramo_con_piu_voci_arriva_in_fondo(self):
        sessione = self._esegui()

        self.assertTrue(sessione.segments)

    def test_la_strategia_dichiarata_e_quella_richiesta(self):
        sessione = self._esegui()

        self.assertEqual(sessione.pipeline_strategy, "standard")
        self.assertEqual(sessione.capture_manifest["extraction_method"], "turn_slicing")

    def test_i_profili_vocali_finiscono_nella_sessione(self):
        sessione = self._esegui()

        self.assertTrue(sessione.diarization.get("voice_profiles"))


if __name__ == "__main__":
    unittest.main()
