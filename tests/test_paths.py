"""Test dei percorsi portabili: una sessione deve ritrovare il suo audio altrove."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fresh_paths(data_root: Path):
    """Ricarica il modulo paths puntandolo a una cartella dati temporanea."""
    os.environ["TESSITURA_DATA_DIR"] = str(data_root)
    for mod in list(sys.modules):
        if mod in ("src.paths", "src.models.session"):
            del sys.modules[mod]
    from src import paths

    paths._data_dir_cache = None
    return paths


def test_percorso_salvato_e_relativo_alla_cartella_dati():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _fresh_paths(root)
        audio = paths.recordings_dir() / "sessione_mic.wav"
        audio.write_bytes(b"RIFF")
        assert paths.to_portable(audio) == "recordings/sessione_mic.wav"


def test_audio_ritrovato_dopo_lo_spostamento_della_cartella_dati():
    """Il caso "apro la sessione su un altro PC": percorso diverso, stesso file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "altro_pc"
        paths = _fresh_paths(root)
        audio = paths.recordings_dir() / "sessione_mic.wav"
        audio.write_bytes(b"RIFF")
        # riferimento salvato su un PC Windows diverso, con separatori Windows
        found = paths.resolve_media("recordings\\sessione_mic.wav")
        assert found is not None and found.name == "sessione_mic.wav"


def test_audio_ritrovato_anche_se_spostato_in_una_sottocartella():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _fresh_paths(Path(tmp))
        nested = paths.data_dir() / "archivio" / "2026"
        nested.mkdir(parents=True)
        (nested / "vecchia.wav").write_bytes(b"RIFF")
        assert paths.resolve_media("recordings/vecchia.wav") is not None


def test_audio_inesistente_ritorna_none():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _fresh_paths(Path(tmp))
        assert paths.resolve_media("recordings/mai_esistito.wav") is None
        assert paths.resolve_media(None) is None


# `unittest discover` ignora i moduli scritti a funzioni: senza questo
# adattatore questi test risultavano "passati" solo perche' non eseguiti.
# I test qui dentro puntano `paths` a cartelle temporanee tramite la variabile
# d'ambiente: il teardown la rimuove e ricarica il modulo, altrimenti i test
# eseguiti dopo nella stessa sessione scriverebbero in una cartella cancellata.
class FunctionStyleTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.environ.pop("TESSITURA_DATA_DIR", None)
        for module in ("src.paths", "src.models.session"):
            sys.modules.pop(module, None)
        from src import paths as reloaded

        reloaded._data_dir_cache = None


for _name, _func in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_func):
        setattr(FunctionStyleTests, _name, (lambda f: lambda self: f())(_func))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FALLITO {name}: {exc}")
    os.environ.pop("TESSITURA_DATA_DIR", None)
    print(f"\n{'tutti i test passano' if not failures else f'{failures} test falliti'}")
    raise SystemExit(1 if failures else 0)
