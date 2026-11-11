"""Finestra del progetto: cosa ricorda e cosa verra' passato al modello.

La memoria di un progetto e' testo, e l'utente deve poterlo leggere e correggere:
un contesto che influenza trascrizione e risposte senza essere ispezionabile e'
un comportamento inspiegabile, non una funzione.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


def describe_project(project) -> str:
    """Riga di stato: nome, sedute registrate e file ancora da elaborare."""
    pending = len(project.pending_audio())
    done = len(project.sessions)
    parts = [f"Progetto: {project.name}", f"{done} sedute in memoria"]
    parts.append(f"{pending} audio da elaborare" if pending else "nessun audio in attesa")
    return " · ".join(parts)


class ProjectDetailsDialog(QDialog):
    """Descrizione, glossario, voci ricorrenti e memoria delle sedute."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self._clear_learning_requested = False
        self.setWindowTitle(f"Progetto — {project.name}")
        self.resize(760, 620)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Questi dati restano nella cartella del progetto e vengono passati "
            "al modello locale a ogni elaborazione. Le correzioni confermate "
            "alimentano glossario, profili vocali e preferenze di analisi; "
            "i pesi dei modelli non vengono modificati."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit(project.name)
        form.addRow("Nome", self.name_edit)
        self.description_edit = QLineEdit(project.description)
        self.description_edit.setPlaceholderText("Di cosa si parla in questo progetto")
        form.addRow("Descrizione", self.description_edit)
        self.glossary_edit = QLineEdit(", ".join(project.glossary))
        self.glossary_edit.setPlaceholderText("Nomi propri e termini, separati da virgole")
        form.addRow("Glossario", self.glossary_edit)
        self.speakers_edit = QLineEdit(", ".join(project.speakers))
        self.speakers_edit.setPlaceholderText("Nomi delle voci ricorrenti, in ordine")
        form.addRow("Voci", self.speakers_edit)
        layout.addLayout(form)

        layout.addWidget(QLabel("Note libere"))
        self.notes_edit = QPlainTextEdit(project.notes)
        self.notes_edit.setPlaceholderText(
            "Contesto che vuoi ricordare: obiettivi, persone, decisioni aperte."
        )
        self.notes_edit.setMaximumHeight(120)
        layout.addWidget(self.notes_edit)

        self.learning_check = QCheckBox("Usa l'apprendimento locale del progetto")
        self.learning_check.setChecked(project.learning_enabled)
        self.learning_check.toggled.connect(self._refresh_preview)
        layout.addWidget(self.learning_check)

        learning_row = QHBoxLayout()
        self.learning_status = QLabel()
        self.learning_status.setObjectName("muted")
        self.learning_status.setWordWrap(True)
        learning_row.addWidget(self.learning_status, 1)
        self.clear_learning_button = QPushButton("Azzera apprendimento")
        self.clear_learning_button.setToolTip(
            "La cancellazione viene applicata soltanto premendo Salva."
        )
        self.clear_learning_button.clicked.connect(self._request_clear_learning)
        learning_row.addWidget(self.clear_learning_button)
        layout.addLayout(learning_row)

        layout.addWidget(QLabel("Memoria passata al modello"))
        self.memory_view = QPlainTextEdit(project.memory_text())
        self.memory_view.setReadOnly(True)
        layout.addWidget(self.memory_view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (self.name_edit, self.description_edit, self.glossary_edit,
                       self.speakers_edit):
            widget.textChanged.connect(self._refresh_preview)
        self.notes_edit.textChanged.connect(self._refresh_preview)
        self._refresh_preview()

    # --- anteprima -----------------------------------------------------------

    def _values(self) -> dict:
        return {
            "name": self.name_edit.text().strip() or self._project.name,
            "description": self.description_edit.text().strip(),
            "notes": self.notes_edit.toPlainText().strip(),
            "glossary": self.glossary_edit.text(),
            "speakers": [item.strip() for item in self.speakers_edit.text().split(",")
                         if item.strip()],
            "learning_enabled": self.learning_check.isChecked(),
        }

    def _refresh_preview(self) -> None:
        """L'anteprima si aggiorna mentre si scrive, senza toccare il progetto."""
        import copy

        preview = copy.deepcopy(self._project)
        values = self._values()
        preview.name = values["name"]
        preview.description = values["description"]
        preview.notes = values["notes"]
        preview.set_glossary(values["glossary"])
        preview.speakers = values["speakers"]
        preview.learning_enabled = values["learning_enabled"]
        if self._clear_learning_requested:
            preview.clear_learning()
        self.memory_view.setPlainText(preview.memory_text())
        stats = preview.learning_summary()
        state = "attivo" if preview.learning_enabled else "in pausa"
        self.learning_status.setText(
            f"Apprendimento {state} · revisione {stats['revision']} · "
            f"{stats['corrections']} correzioni · {stats['speakers']} profili vocali · "
            f"{stats['preferences']} preferenze"
        )

    def _request_clear_learning(self) -> None:
        self._clear_learning_requested = True
        self.clear_learning_button.setEnabled(False)
        self.clear_learning_button.setText("Verrà azzerato al salvataggio")
        self._refresh_preview()

    def apply_to_project(self) -> None:
        """Scrive i valori nel progetto: chiamare solo se il dialogo e' accettato."""
        values = self._values()
        self._project.name = values["name"]
        self._project.description = values["description"]
        self._project.notes = values["notes"]
        self._project.set_glossary(values["glossary"])
        self._project.speakers = values["speakers"]
        if self._clear_learning_requested:
            self._project.clear_learning()
        self._project.learning_enabled = values["learning_enabled"]
