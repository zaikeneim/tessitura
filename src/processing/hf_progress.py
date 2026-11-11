"""Progresso dei download da HuggingFace, riportato alla UI invece che a stdout.

huggingface_hub mostra l'avanzamento con tqdm su stderr: in una GUI non lo vede
nessuno (e con pythonw stderr non esiste nemmeno). Qui si costruisce una
sottoclasse di tqdm che chiama una callback dell'app e non scrive su console.
"""

from __future__ import annotations

from typing import Callable

from ..bootstrap import _NullStream

# (nome file, percentuale 0..100)
DownloadCb = Callable[[str, int], None]


def make_progress_tqdm(on_download: DownloadCb, step_pct: int = 5):
    """tqdm che invoca on_download(nome_file, percentuale) circa ogni step_pct%."""
    from tqdm.auto import tqdm as _Tqdm

    step = max(1, int(step_pct))

    class _ProgressTqdm(_Tqdm):
        def __init__(self, *args, **kwargs):
            self._last_pct = -step
            # La barra testuale non serve (il progresso va in GUI) e con pythonw
            # non c'e' una console su cui scriverla: la mandiamo su un sink.
            kwargs["file"] = _NullStream()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if self.total:
                pct = int(self.n * 100 / self.total)
                if pct >= self._last_pct + step or pct >= 100:
                    self._last_pct = pct - (pct % step)
                    name = (self.desc or "modello").split("/")[-1]
                    on_download(name, min(pct, 100))
            return result

    return _ProgressTqdm
