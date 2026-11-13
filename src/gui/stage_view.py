"""Ordered, observed processing state; no time-based completion estimates."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem

STATUS_NAMES = {"pending": "In attesa", "running": "In corso", "completed": "Completata",
                "skipped": "Saltata", "failed": "Fallita"}

# Diagnostic keys stored in StageEvent.extra, shown in the detail tooltip.
EXTRA_LABELS = {
    "profiles": "Profili creati",
    "with_embedding": "Con impronta vocale valida",
    "known_matches": "Riconosciuti fra le voci note",
    "tse_eligible": "Idonei all'estrazione del parlante target (indicativo)",
    "detected": "Ipotesi di regioni rilevate",
    "queued": "Regioni in coda (entro i limiti)",
    "processed": "Regioni elaborate",
    "recovered": "Regioni con parole nuove proposte (non verificate)",
    "unresolved": "Regioni non risolte",
    "candidates": "Candidati da revisionare",
    "controls": "Regioni di controllo in coda",
    "controls_processed": "Regioni di controllo elaborate",
    "rejected_predictions": "Predizioni respinte dai controlli ASR",
}


def format_seconds(value: float) -> str:
    """Italian decimal comma, e.g. '0,0 s'. The duration stays its own column."""
    return f"{float(value):.1f} s".replace(".", ",")


def describe_extra(extra: dict | None) -> str:
    lines = []
    for key, value in (extra or {}).items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"{EXTRA_LABELS.get(key, key)}: {value}")
    return "\n".join(lines)


class StageView(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Fase", "Stato", "Dettaglio", "Tempo"])
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setMaximumHeight(180)
        self.setMinimumHeight(100)
        self.setColumnWidth(0, 280)
        self.setColumnWidth(1, 110)
        self.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setColumnWidth(3, 65)
        self._rows = {}
        self.run_label = ""
        self.hide()

    def begin_run(self, label=""):
        self.clear()
        self._rows.clear()
        self.run_label = str(label)
        self.setHeaderLabels([f"Fase · {label}" if label else "Fase", "Stato", "Dettaglio", "Tempo"])
        self.show()

    def on_stage(self, event):
        if isinstance(event, dict):  # snapshot saved in the session
            event = SimpleNamespace(**{"extra": {}, "warning": None, "fraction": None, **event})
        row = self._rows.get(event.stage_id)
        if row is None:
            row = QTreeWidgetItem(self)
            self._rows[event.stage_id] = row
        state = STATUS_NAMES.get(event.status, event.status)
        if event.status == "running" and event.fraction is not None:
            state += f" {event.fraction:.0%}"
        detail = event.detail
        if event.warning:
            state += " · Avviso"
            detail = f"{detail} — {event.warning}" if detail else event.warning
        values = (event.label, state, detail, format_seconds(event.elapsed_s))
        extra_text = describe_extra(getattr(event, "extra", None))
        for column, value in enumerate(values):
            row.setText(column, value)
            tooltip = f"{value}\n\n{extra_text}" if column == 2 and extra_text else value
            row.setToolTip(column, tooltip)
        if event.status in ("running", "failed"):
            self.scrollToItem(row)
        elif event.stage_id == "finalize" and event.status == "completed":
            profiles = self._rows.get("profiles")
            if profiles is not None:
                self.scrollToItem(profiles)

    def show_saved(self, rows, label="ultima elaborazione salvata") -> bool:
        """Rebuild the table from ``provenance.processing_stages`` of a reopened session."""
        if not rows:
            return False
        self.begin_run(label)
        for row in rows:
            if isinstance(row, dict) and row.get("stage_id"):
                self.on_stage(row)
        return True

    def on_batch_stage(self, source, event):
        if self.run_label != str(source):
            self.begin_run(source)
        self.on_stage(event)
