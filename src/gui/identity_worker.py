"""Global identity evidence is computed locally outside the GUI thread."""
from PySide6.QtCore import QThread, Signal


class GlobalIdentityWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    # Modello di diarizzazione assente o non utilizzabile: la finestra propone
    # la configurazione e ripete la revisione, senza riavviare l'applicazione.
    access_failed = Signal(object)

    def __init__(self, session, source, output_dir, parent=None):
        super().__init__(parent)
        self.session, self.source, self.output_dir = session, source, output_dir

    def run(self):
        from ..processing.hf_access import ModelAccessError, redact
        try:
            from pathlib import Path
            from ..processing.resources import acquire_inference
            from ..processing.pipeline import resolve_device
            from ..processing.global_workflow import extract_clean_embeddings
            from ..processing.global_diarization import identity_review, annotate_identity_uncertainty, review_confirmed_identities
            device, _ = resolve_device()
            turns = self.session.diarization.get("turns", [])
            if not turns:
                raise ValueError("Turni ordinari non disponibili. Apri una timeline con diagnostica completa.")
            with acquire_inference():
                samples, metrics = extract_clean_embeddings(self.source, turns, Path(self.output_dir) / "identity-embeddings.json", device=device)
            review = identity_review(samples)
            annotate_identity_uncertainty(self.session.utterances, review)
            confirmed = review_confirmed_identities(self.session, samples)
            review["confirmed_identity_review"] = {k: v for k, v in confirmed.items() if k != "candidates"}
            review["suspected_regions"].extend(confirmed["suspected_regions"])
            self.done.emit({"review": review, "candidates": confirmed["candidates"], "metrics": metrics})
        except ModelAccessError as exc:
            self.access_failed.emit(exc.to_dict())
        except Exception as exc:
            self.failed.emit(redact(str(exc)))
