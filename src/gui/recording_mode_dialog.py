"""Dialog a due passi per la selezione della modalità di elaborazione e verifica cuffie (Passo 4).

Guida l'utente nella scelta tra:
1. Trascrizione indipendente delle sorgenti ('source_aware') per call/interviste online
   (Zoom, Meet, Teams), con diarizzazione per sorgente e verifica dell'ascolto.
2. Diarizzazione acustica classica ('standard') per registrazioni ambientali o con più
   interlocutori per traccia.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class _OptionCard(QFrame):
    """Card cliccabile per la selezione visiva con hover ed evidenziazione."""
    clicked = Signal()

    def __init__(self, title: str, description: str, badge: str | None = None, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("optionCard")
        self.setStyleSheet("""
            QFrame#optionCard {
                background: #182332;
                border: 1px solid #2a394d;
                border-radius: 10px;
                padding: 14px;
            }
            QFrame#optionCard:hover {
                background: #1e2c3f;
                border-color: #5ee0ca;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #edf3fa;")
        header.addWidget(title_label)
        header.addStretch()

        if badge:
            badge_label = QLabel(badge)
            badge_label.setStyleSheet("""
                background: #193632;
                color: #5ee0ca;
                font-size: 11px;
                font-weight: 700;
                padding: 3px 8px;
                border-radius: 6px;
                border: 1px solid #2d5d57;
            """)
            header.addWidget(badge_label)
        layout.addLayout(header)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #9aacc2; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class RecordingModeDialog(QDialog):
    """Wizard a due passi per configurazione sorgenti e verifica cuffie."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurazione Sorgenti Audio")
        self.resize(580, 420)
        self.setModal(True)

        self.strategy: str = "standard"
        self.headphones_confirmed: bool = True

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        self.stack = QStackedWidget(self)
        main_layout.addWidget(self.stack)

        self._init_step1()
        self._init_step2()
        self.stack.setCurrentIndex(0)

    def _init_step1(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        title = QLabel("Configurazione Sorgenti Audio")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #edf3fa;")
        layout.addWidget(title)

        subtitle = QLabel("Come sono organizzate le voci nelle tracce audio?")
        subtitle.setStyleSheet("color: #9aacc2; font-size: 13px;")
        layout.addWidget(subtitle)

        # Card 1: sorgenti di acquisizione indipendenti, ciascuna con una o più voci.
        card_dual = _OptionCard(
            title="Microfono e uscita separati (cuffie o altoparlanti)",
            description=(
                "Ottimizzato per call online (Zoom, Google Meet, Teams). "
                "Trascrive ciascuna sorgente indipendentemente e identifica le voci presenti. "
                "Ogni traccia può contenere più persone; gli interventi simultanei restano separati."
            ),
            badge="CONSIGLIATO PER CALL",
        )
        card_dual.clicked.connect(self._on_select_source_single_speaker)
        layout.addWidget(card_dual)

        # Card 2: Multi-speaker standard
        card_std = _OptionCard(
            title="Più persone per traccia / Mix standard",
            description=(
                "Ideale per registrazioni ambientali o stanze con più partecipanti. "
                "Utilizza l'analisi acustica avanzata per identificare e separare "
                "le voci all'interno del mix audio."
            ),
        )
        card_std.clicked.connect(self._on_select_standard)
        layout.addWidget(card_std)

        layout.addStretch()

        # Bottone annulla
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Annulla")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.stack.addWidget(page)

    def _init_step2(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        title = QLabel("Verifica Dispositivo di Ascolto")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #edf3fa;")
        layout.addWidget(title)

        subtitle = QLabel("Hai utilizzato le cuffie durante la registrazione?")
        subtitle.setStyleSheet("color: #9aacc2; font-size: 13px;")
        layout.addWidget(subtitle)

        # Card A: Cuffie confermate
        card_headphones = _OptionCard(
            title="Sì, ho usato le cuffie",
            description=(
                "L'audio remoto arriva dal loopback dell'uscita cuffie. "
                "Il microfono può comunque contenere più voci e viene verificato separatamente."
            ),
            badge="MASSIMA FEDELTÀ",
        )
        card_headphones.clicked.connect(self._on_confirm_headphones)
        layout.addWidget(card_headphones)

        # Card B: Altoparlanti / Non sicuro
        card_speakers = _OptionCard(
            title="No, ho usato altoparlanti / Non sono sicuro",
            description=(
                "Valuta il rientro acustico usando il loopback come riferimento. "
                "Conserva le sorgenti e il parlato simultaneo; nessuna voce viene eliminata "
                "solo perché coincide con la riproduzione."
            ),
            badge="BLEED CHECK ATTIVO",
        )
        card_speakers.clicked.connect(self._on_unconfirmed_headphones)
        layout.addWidget(card_speakers)

        layout.addStretch()

        # Bottoni navigazione
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("← Indietro")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        nav_layout.addWidget(back_btn)
        nav_layout.addStretch()
        cancel_btn = QPushButton("Annulla")
        cancel_btn.clicked.connect(self.reject)
        nav_layout.addWidget(cancel_btn)
        layout.addLayout(nav_layout)

        self.stack.addWidget(page)

    def _on_select_source_single_speaker(self) -> None:
        self.stack.setCurrentIndex(1)

    def _on_select_standard(self) -> None:
        self.strategy = "standard"
        self.headphones_confirmed = True
        self.accept()

    def _on_confirm_headphones(self) -> None:
        self.strategy = "source_aware"
        self.headphones_confirmed = True
        self.accept()

    def _on_unconfirmed_headphones(self) -> None:
        self.strategy = "source_aware"
        self.headphones_confirmed = False
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            if self.stack.currentIndex() == 1:
                self.stack.setCurrentIndex(0)
                return
        super().keyPressEvent(event)

    def reject(self) -> None:
        if self.stack.currentIndex() == 1:
            # Fallback conservativo al passo 2 (decisione piano §1.33):
            # La scelta di trascrivere separatamente resta confermata,
            # assumendo cuffie non verificate (rientro da sottoporre a revisione).
            self.strategy = "source_aware"
            self.headphones_confirmed = False
            self.accept()
            return
        super().reject()


def ask_recording_mode(parent=None) -> tuple[str, bool]:
    """Mostra il dialog modale a 2 passi.
    
    Ritorna una tupla (strategy, headphones_confirmed).
    In caso di chiusura/annullamento, adotta il fallback sicuro standard.
    """
    dialog = RecordingModeDialog(parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.strategy, dialog.headphones_confirmed
    return "standard", False
