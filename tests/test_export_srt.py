"""Export SRT: le battute devono restare leggibili anche dopo il raggruppamento.

Il raggruppamento per voce unisce i turni contigui: senza limiti, un turno di
cinque minuti diventava un unico sottotitolo. Qui si verifica che durata e
lunghezza restino nei limiti e che il testo non venga perso per strada.

Esecuzione: python -m tests.test_export_srt
"""
import re
import unittest

from src.models import storage
from src.models.session import Segment, Session, Word


def _cues(srt: str):
    """(inizio, fine, righe di testo) di ogni battuta."""
    parsed = []
    for block in [item for item in srt.split("\n\n") if item.strip()]:
        lines = block.strip().splitlines()
        times = re.match(
            r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1]
        )
        values = [int(part) for part in times.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        parsed.append((start, end, lines[2:]))
    return parsed


def _long_segment(words=40, step=0.5):
    items = [Word(i * step, (i + 1) * step, f"parola{i}") for i in range(words)]
    text = " ".join(word.text for word in items)
    return Segment(0.0, words * step, text, "A", items)


class ExportSrtTests(unittest.TestCase):
    def test_un_turno_lungo_diventa_piu_battute_entro_i_limiti(self):
        session = Session(segments=[_long_segment()])

        cues = _cues(storage.export_srt(session))

        self.assertGreater(len(cues), 1)
        for start, end, lines in cues:
            self.assertLessEqual(end - start, storage.SRT_MAX_SECONDS + 0.01)
            for line in lines:
                self.assertLessEqual(len(line), storage.SRT_MAX_CHARS)

    def test_nessuna_parola_persa_o_duplicata(self):
        session = Session(segments=[_long_segment()])

        testo = " ".join(" ".join(lines) for _s, _e, lines in _cues(storage.export_srt(session)))

        atteso = ["[A]"] + [f"parola{i}" for i in range(40)]
        self.assertEqual(testo.split(), atteso)

    def test_i_tempi_seguono_le_parole_allineate(self):
        session = Session(segments=[_long_segment()])

        cues = _cues(storage.export_srt(session))

        self.assertAlmostEqual(cues[0][0], 0.0, places=3)
        self.assertAlmostEqual(cues[-1][1], 20.0, places=3)
        for (_start, end, _lines), (next_start, _e, _l) in zip(cues, cues[1:]):
            self.assertLessEqual(end, next_start + 0.001)  # battute non sovrapposte

    def test_senza_allineamento_i_tempi_sono_ripartiti_sul_testo(self):
        """Dopo una modifica manuale le parole non sono piu' affidabili."""
        segment = Segment(0.0, 30.0, "frase lunghissima " * 30, "A")
        segment.words_valid = False
        session = Session(segments=[segment])

        cues = _cues(storage.export_srt(session))

        self.assertGreater(len(cues), 1)
        self.assertLessEqual(cues[-1][1], 30.0 + 0.001)
        for start, end, _lines in cues:
            self.assertLessEqual(end - start, storage.SRT_MAX_SECONDS + 0.01)

    def test_un_segmento_corto_resta_una_sola_battuta(self):
        session = Session(segments=[Segment(0.0, 2.0, "Va bene.", "A",
                                            [Word(0.0, 2.0, "Va bene.")])])

        cues = _cues(storage.export_srt(session))

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], ["[A] Va bene."])

    def test_i_limiti_sono_configurabili(self):
        session = Session(segments=[_long_segment()])

        stretti = _cues(storage.export_srt(session, max_seconds=2.0, max_chars=30))

        self.assertGreater(len(stretti), len(_cues(storage.export_srt(session))))
        for start, end, lines in stretti:
            self.assertLessEqual(end - start, 2.01)
            for line in lines:
                self.assertLessEqual(len(line), 30)

    def test_il_nome_della_voce_compare_una_volta_per_turno(self):
        session = Session(segments=[_long_segment()])
        session.rename_speaker("A", "Marco")

        cues = _cues(storage.export_srt(session))

        prime = [lines[0] for _s, _e, lines in cues]
        self.assertEqual(sum(1 for line in prime if line.startswith("[Marco]")), 1)

    def test_voci_simultanee_preservate_in_export_srt(self):
        """Due speaker che parlano contemporaneamente devono comparire entrambi nell'SRT."""
        seg_a = Segment(10.0, 15.0, "Discorso principale", "A", [Word(10.0, 15.0, "Discorso principale")])
        seg_b = Segment(11.0, 12.5, "Interruzione breve", "B", [Word(11.0, 12.5, "Interruzione breve")])
        session = Session(segments=[seg_a, seg_b])

        cues = _cues(storage.export_srt(session))
        self.assertEqual(len(cues), 2, "Entrambe le battute sovrapposte devono comparire")
        self.assertEqual(cues[0][0], 10.0)
        self.assertEqual(cues[0][2], ["[A] Discorso principale"])
        self.assertEqual(cues[1][0], 11.0)
        self.assertEqual(cues[1][2], ["[B] Interruzione breve"])


if __name__ == "__main__":
    unittest.main()
