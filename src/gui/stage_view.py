"""Ordered, observed processing state; no time-based completion estimates."""
from __future__ import annotations

from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem


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
        row = self._rows.get(event.stage_id)
        if row is None:
            row = QTreeWidgetItem(self)
            self._rows[event.stage_id] = row
        names = {"pending": "In attesa", "running": "In corso", "completed": "Completata",
                 "skipped": "Saltata", "failed": "Fallita"}
        state = names.get(event.status, event.status)
        if event.status == "running" and event.fraction is not None:
            state += f" {event.fraction:.0%}"
        detail = event.detail
        if event.warning:
            state += " · Avviso"
            detail = f"{detail} — {event.warning}" if detail else event.warning
        for column, value in enumerate((event.label, state, detail, f"{event.elapsed_s:.1f} s")):
            row.setText(column, value)
            row.setToolTip(column, value)
        if event.status in ("running", "failed"):
            self.scrollToItem(row)

    def on_batch_stage(self, source, event):
        if self.run_label != str(source):
            self.begin_run(source)
        self.on_stage(event)
