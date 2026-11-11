"""Ricucitura delle frasi spezzate dal rumore di attribuzione (piano 2026-09-21, livello 1).

Il difetto misurato sulle sessioni reali: 8.6% dei segmenti finiva in catene dove
l'etichetta alterna su parole contigue, spezzando una frase in frammenti di una
parola ("abbiamo fatto tutta" / "la" / "camminata e la" / "tirata").

Scenari obbligatori:
- Poche parole incastonate fra due tratti della stessa voce vengono ricucite
- Un'alternanza prolungata collassa sulla voce che occupa piu' tempo
- Un turno vero, aperto da una pausa o da una frase chiusa, resta intatto
- In sovrapposizione attestata l'interiezione singola resta al suo parlante

Esecuzione: python -m tests.test_merge_alternations
"""

import unittest

from src.processing.merge import WordUnit, _absorb_alternations


def words(*spec: tuple[float, float, str]) -> list[WordUnit]:
    return [WordUnit(start, end, text) for start, end, text in spec]


class IsolaTests(unittest.TestCase):
    """Poche parole incastonate fra due tratti della stessa voce."""

    def setUp(self):
        # "abbiamo fatto tutta | la | camminata": l'articolo strappato a meta' sintagma.
        self.words = words(
            (0.0, 0.4, " abbiamo"),
            (0.4, 0.8, " fatto"),
            (0.8, 1.2, " tutta"),
            (1.2, 1.3, " la"),
            (1.3, 1.9, " camminata"),
            (1.9, 2.4, " intera."),
        )

    def test_isola_riassorbita_nella_voce_circostante(self):
        labels = ["A", "A", "A", "B", "A", "A"]

        result = _absorb_alternations(self.words, list(labels))

        self.assertEqual(result, ["A"] * 6)

    def test_pausa_prima_dell_isola_la_rende_un_turno(self):
        staccato = words(
            (0.0, 0.4, " abbiamo"),
            (0.4, 0.8, " fatto"),
            (0.8, 1.2, " tutta"),
            (2.0, 2.2, " la"),      # 0.8s di pausa: e' un intervento, non un taglio
            (2.2, 2.8, " camminata"),
            (2.8, 3.2, " intera."),
        )

        result = _absorb_alternations(staccato, ["A", "A", "A", "B", "A", "A"])

        self.assertEqual(result[3], "B")

    def test_frase_chiusa_prima_dell_isola_la_rende_un_turno(self):
        chiuso = words(
            (0.0, 0.4, " abbiamo"),
            (0.4, 0.8, " finito."),  # frase conclusa: il cambio e' plausibile
            (0.8, 1.0, " Certo"),
            (1.0, 1.6, " riprendiamo"),
            (1.6, 2.0, " dopo."),
        )

        result = _absorb_alternations(chiuso, ["A", "A", "B", "A", "A"])

        self.assertEqual(result[2], "B")

    def test_sequenza_lunga_e_un_turno_non_un_frammento(self):
        lungo = words(*[(i * 0.4, i * 0.4 + 0.4, f" parola{i}") for i in range(10)])
        labels = ["A", "A", "B", "B", "B", "B", "B", "A", "A", "A"]

        result = _absorb_alternations(lungo, list(labels))

        self.assertEqual(result, labels)

    def test_interiezione_in_sovrapposizione_resta_al_suo_parlante(self):
        # pyannote attesta la simultaneita': puo' essere un intervento reale.
        result = _absorb_alternations(
            self.words, ["A", "A", "A", "B", "A", "A"], overlaps=[(1.15, 1.35)]
        )

        self.assertEqual(result[3], "B")


class CatenaTests(unittest.TestCase):
    """Alternanza prolungata fra due voci sulla stessa frase."""

    def setUp(self):
        # A occupa 1.2s, B occupa 0.3s: A domina.
        self.words = words(
            (0.0, 0.6, " che"),
            (0.6, 0.7, " orario"),
            (0.7, 1.3, " era"),
            (1.3, 1.4, " piu'"),
            (1.4, 1.8, " o"),
            (1.8, 1.9, " meno"),
        )
        self.labels = ["A", "B", "A", "B", "A", "B"]

    def test_catena_collassa_sulla_voce_dominante(self):
        result = _absorb_alternations(self.words, list(self.labels))

        self.assertEqual(len(set(result)), 1, "la catena deve restare una voce sola")
        self.assertEqual(result[0], "A", "vince chi occupa piu' tempo")

    def test_catena_ricucita_anche_in_sovrapposizione(self):
        # Un'alternanza cosi' fitta non e' parlato simultaneo plausibile:
        # va ricucita anche dove la diarizzazione dichiara sovrapposizione.
        result = _absorb_alternations(
            self.words, list(self.labels), overlaps=[(0.0, 2.0)]
        )

        self.assertEqual(len(set(result)), 1)

    def test_tre_voci_non_sono_una_catena(self):
        labels = ["A", "B", "C", "A", "B", "C"]

        result = _absorb_alternations(self.words, list(labels))

        self.assertEqual(result, labels, "il criterio vale solo fra due voci")


class InvariantiTests(unittest.TestCase):
    def test_voce_unica_resta_invariata(self):
        uno = words((0.0, 0.5, " tutto"), (0.5, 1.0, " uguale."))

        self.assertEqual(_absorb_alternations(uno, ["A", "A"]), ["A", "A"])

    def test_sequenza_vuota_non_esplode(self):
        self.assertEqual(_absorb_alternations([], []), [])

    def test_il_numero_di_parole_non_cambia(self):
        campione = words(*[(i * 0.3, i * 0.3 + 0.3, f" w{i}") for i in range(8)])
        labels = ["A", "B", "A", "B", "A", "B", "A", "A"]

        result = _absorb_alternations(campione, list(labels))

        self.assertEqual(len(result), len(labels), "si riattribuisce, non si cancella")


if __name__ == "__main__":
    unittest.main()
