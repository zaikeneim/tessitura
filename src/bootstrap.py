"""Inizializzazione da eseguire PRIMA di qualsiasi codice che faccia rete.

Su questa macchina AVG Antivirus intercetta il traffico HTTPS e lo re-firma con
una CA propria presente solo nello store certificati di Windows (non in certifi).
Senza questo fix, i download dei modelli (HuggingFace / faster-whisper / pyannote)
falliscono con CERTIFICATE_VERIFY_FAILED. `truststore.inject_into_ssl()` fa usare
a Python (e quindi a requests/urllib) lo store di sistema, che contiene quella CA.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# CPython removes DLL search directories when their handle is closed/collected.
_dll_handles: dict[str, object] = {}


def retain_dll_directory(path: Path) -> None:
    key = str(path.resolve())
    if key not in _dll_handles and path.is_dir() and sys.platform.startswith("win"):
        try:
            _dll_handles[key] = os.add_dll_directory(key)
        except OSError:
            pass


class _NullStream(io.TextIOBase):
    """Stream che scarta tutto: sostituisce stdout/stderr quando non esistono."""

    def write(self, s: str) -> int:  # noqa: D102
        return len(s)

    def flush(self) -> None:  # noqa: D102
        pass

    def isatty(self) -> bool:  # noqa: D102
        return False


def ensure_std_streams() -> None:
    """Garantisce che sys.stdout/sys.stderr non siano None.

    Lanciata con pythonw.exe (GUI senza console) Python mette a None stdout e
    stderr: qualunque libreria che ci scriva esplode. tqdm (usato da
    huggingface_hub per il progresso dei download) falliva cosi' con
    "'NoneType' object has no attribute 'write'".
    """
    for name in ("stdout", "stderr", "__stdout__", "__stderr__"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, _NullStream())


def add_conda_dll_dirs() -> None:
    """Rende caricabili le DLL dell'ambiente conda o del pacchetto PyInstaller onedir.

    Le DLL native di conda-forge (Qt6*, icu, ...) stanno in <env>/Library/bin.
    Quando si attiva l'ambiente sono nel PATH, ma se l'app viene lanciata senza
    attivazione non verrebbero trovate: qui le aggiungiamo esplicitamente.
    """
    if not sys.platform.startswith("win"):
        return
    for sub in ("Library/bin", "Library/mingw-w64/bin", "DLLs"):
        d = Path(sys.prefix) / sub
        if d.is_dir():
            retain_dll_directory(d)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir / "_internal"))
        for candidate in (
            exe_dir,
            meipass,
            meipass / "torch" / "lib",
            meipass / "ctranslate2",
            meipass / "PySide6",
        ):
            if candidate.is_dir():
                retain_dll_directory(candidate)


def use_bundled_models() -> None:
    """Fa scaricare e cercare i modelli nella cartella modelli configurata.

    Va chiamata prima di importare huggingface_hub (quindi prima di faster-whisper
    e pyannote): le variabili vengono lette all'import. Se l'utente le ha gia'
    impostate a mano, si rispetta la sua scelta.
    """
    from . import paths

    target = paths.models_dir()
    os.environ.setdefault("HF_HOME", str(target))
    # La cache dei repo sta in <HF_HOME>/hub: puntandola alla cartella padre i
    # modelli gia' scaricati non verrebbero trovati e si riscaricherebbe tutto.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(target / "hub"))


def silence_noisy_warnings() -> None:
    """Zittisce avvisi innocui ma rumorosi di torch/pyannote/HF.

    - torch flop_counter: 'triton not found' (triton non esiste su Windows).
    - pyannote: TF32 disabilitato (informativo) e std() con dof<=0 su segmenti corti.
    """
    import logging
    import warnings

    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    for pattern in (
        ".*TensorFloat-32.*",
        ".*degrees of freedom is <= 0.*",
        ".*does not have many workers.*",
    ):
        warnings.filterwarnings("ignore", message=pattern)


def install_system_certs() -> bool:
    """Fa usare a Python lo store certificati del sistema operativo.

    Ritorna True se l'iniezione e' andata a buon fine, False altrimenti
    (in tal caso si prosegue comunque: su reti senza interception funziona lo stesso).
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False
