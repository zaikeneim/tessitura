"""Targeted recovery runs outside the UI thread and returns review proposals."""
from PySide6.QtCore import QThread, Signal


class RegionRecoveryWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, source, start_s, end_s, output_dir, *, options=None, parent=None):
        super().__init__(parent)
        self.arguments = (source, start_s, end_s, output_dir)
        self.options = options or {}

    def run(self):
        try:
            from ..processing.resources import acquire_inference
            from ..processing.region_recovery import recover_region
            with acquire_inference():
                self.done.emit(recover_region(*self.arguments, **self.options))
        except Exception as exc:
            self.failed.emit(str(exc))
