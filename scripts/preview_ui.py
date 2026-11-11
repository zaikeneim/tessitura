"""Render synthetic desktop previews without recording or loading AI models."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from src.bootstrap import add_conda_dll_dirs
add_conda_dll_dirs()

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from src.gui.theme import apply_theme
from src.gui.main_window import MainWindow
from src.hardware import Hardware
from src.models.session import Session, Segment


def main():
    app = QApplication([])
    # Qt's offscreen Windows backend needs explicit font registration.
    for font in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / font
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    apply_theme(app)
    with patch.object(MainWindow, "_populate_devices"), patch(
        "src.gui.main_window.detect",
        return_value=Hardware("cuda", "float16", "large-v3", "NVIDIA GeForce RTX 4070", 8),
    ):
        window = MainWindow()
    window.mic_combo.addItem("Microfono predefinito")
    window.sys_combo.addItem("Altoparlanti di sistema")
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    window.show()
    QTest.qWait(150)
    window.grab().save(str(out / "interface-empty.png"))
    session = Session(name="Riunione di progetto", language="it", segments=[
        Segment(0, 7, "Partiamo dagli obiettivi della settimana. Vorrei semplificare il percorso dalla registrazione alla trascrizione.", "SPEAKER_00"),
        Segment(8, 15, "Sono d’accordo. Possiamo dare spazio al testo e tenere le impostazioni di registrazione sempre a portata di mano.", "SPEAKER_01"),
        Segment(16, 22, "Per le voci, aggiungiamo nomi riconoscibili e colori coerenti. Sarà facile seguire la conversazione.", "SPEAKER_00"),
        Segment(23, 29, "Perfetto. Controlliamo anche che le registrazioni lunghe mantengano tutto l’audio fino alla fine.", "SPEAKER_02"),
    ], speaker_names={"SPEAKER_00": "Alessandro", "SPEAKER_01": "Giulia", "SPEAKER_02": "Marco"})
    window.session = session
    window.transcript.set_session(session)
    window.speakers.set_session(session)
    window._describe_session()
    window._set_busy(False)
    window.transcript.highlight_at(9)
    QTest.qWait(150)
    window.grab().save(str(out / "interface-review.png"))
    window.resize(1000, 700)
    QTest.qWait(200)
    window.grab().save(str(out / "interface-compact.png"))
    window.close()
    print(f"Previews saved to {out}")


if __name__ == "__main__":
    main()
