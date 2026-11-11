"""Combinazione di piu' registrazioni in un solo MP3.

Due modi che non vanno confusi: in sequenza (una registrazione spezzata in piu'
file) e sovrapposti (piu' microfoni sulla stessa scena). Se si scambiano, la
durata risultante e i tempi della trascrizione sono sbagliati.

Esecuzione: python -m tests.test_combine_audio
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.encode import CONCAT, MIX, combine_to_mp3


def _tone(folder: Path, name: str, seconds: float, freq: float,
          samplerate: int = 44100, channels: int = 1) -> Path:
    times = np.linspace(0, seconds, int(samplerate * seconds), endpoint=False)
    data = 0.3 * np.sin(2 * np.pi * freq * times)
    if channels == 2:
        data = np.column_stack([data, data])
    path = folder / name
    sf.write(path, data, samplerate)
    return path


class CombineAudioTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.addCleanup(self._folder.cleanup)

    def test_in_sequenza_la_durata_e_la_somma(self):
        first = _tone(self.folder, "a.wav", 3.0, 440)
        second = _tone(self.folder, "b.wav", 2.0, 660)

        result = combine_to_mp3([first, second], self.folder / "out.mp3", mode=CONCAT)

        self.assertTrue(result.is_file())
        self.assertAlmostEqual(sf.info(str(result)).duration, 5.0, delta=0.15)

    def test_sovrapposti_la_durata_e_quella_del_piu_lungo(self):
        first = _tone(self.folder, "a.wav", 3.0, 440)
        second = _tone(self.folder, "b.wav", 2.0, 660)

        result = combine_to_mp3([first, second], self.folder / "out.mp3", mode=MIX)

        self.assertAlmostEqual(sf.info(str(result)).duration, 3.0, delta=0.15)

    def test_formati_diversi_vengono_uniformati(self):
        """File da dispositivi diversi: frequenze e canali non coincidono."""
        first = _tone(self.folder, "a.wav", 2.0, 440, samplerate=44100, channels=1)
        second = _tone(self.folder, "b.wav", 2.0, 660, samplerate=16000, channels=2)

        result = combine_to_mp3([first, second], self.folder / "out.mp3", mode=CONCAT)

        info = sf.info(str(result))
        self.assertAlmostEqual(info.duration, 4.0, delta=0.15)
        self.assertEqual(info.channels, 2)

    def test_tre_file_in_sequenza(self):
        files = [_tone(self.folder, f"{index}.wav", 1.5, 300 + index * 100)
                 for index in range(3)]

        result = combine_to_mp3(files, self.folder / "out.mp3", mode=CONCAT)

        self.assertAlmostEqual(sf.info(str(result)).duration, 4.5, delta=0.3)

    def test_il_progresso_arriva_in_fondo(self):
        files = [_tone(self.folder, f"{index}.wav", 1.0, 440) for index in range(2)]
        steps: list[tuple[str, float]] = []

        combine_to_mp3(files, self.folder / "out.mp3", mode=CONCAT,
                       on_progress=lambda message, frac: steps.append((message, frac)))

        self.assertTrue(steps)
        self.assertEqual(steps[-1][1], 1.0)

    def test_un_solo_file_viene_rifiutato(self):
        only = _tone(self.folder, "a.wav", 1.0, 440)

        with self.assertRaises(ValueError):
            combine_to_mp3([only], self.folder / "out.mp3")

    def test_file_mancante_viene_segnalato_prima_di_elaborare(self):
        existing = _tone(self.folder, "a.wav", 1.0, 440)

        with self.assertRaises(FileNotFoundError):
            combine_to_mp3([existing, self.folder / "mai_esistito.wav"],
                           self.folder / "out.mp3")

    def test_destinazione_uguale_a_una_sorgente_viene_rifiutata(self):
        """Scrivere sopra una sorgente distruggerebbe la registrazione."""
        first = _tone(self.folder, "a.wav", 1.0, 440)
        second = _tone(self.folder, "b.mp3", 1.0, 660)

        with self.assertRaises(ValueError):
            combine_to_mp3([first, second], second, mode=CONCAT)

    def test_le_sorgenti_restano_sul_disco(self):
        """La combinazione non deve cancellare gli originali senza richiesta."""
        files = [_tone(self.folder, f"{index}.wav", 1.0, 440) for index in range(2)]

        combine_to_mp3(files, self.folder / "out.mp3", mode=CONCAT)

        self.assertTrue(all(path.is_file() for path in files))

    def test_modo_sconosciuto_viene_rifiutato(self):
        files = [_tone(self.folder, f"{index}.wav", 1.0, 440) for index in range(2)]

        with self.assertRaises(ValueError):
            combine_to_mp3(files, self.folder / "out.mp3", mode="incolla")


if __name__ == "__main__":
    unittest.main()
