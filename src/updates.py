"""Controllo aggiornamenti: librerie Python e modelli scaricati.

Solo diagnosi, nessuna installazione: aggiornare l'ambiente da dentro l'app
mentre l'app gira e' un ottimo modo per romperlo. Qui si dice cosa e' invecchiato
e con quale comando aggiornarlo (scripts\\update.ps1).
"""

from __future__ import annotations

import json
import subprocess
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

# Pacchetti che contano davvero per l'app: il resto sono dipendenze indirette,
# che si aggiornano da sole al seguito di questi.
DIRECT_PACKAGES = {
    "faster-whisper", "pyannote.audio", "pyaudiowpatch", "soundfile",
    "sounddevice", "noisereduce", "numpy", "scipy", "truststore",
    "torch", "torchaudio", "torchcodec", "pyside6", "ctranslate2",
}


@dataclass
class Report:
    direct: list[tuple[str, str, str]] = field(default_factory=list)   # nome, ora, disponibile
    indirect_count: int = 0
    models: list[tuple[str, str]] = field(default_factory=list)        # repo, stato
    errors: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.direct and not self.models and not self.indirect_count


def _pip_outdated() -> tuple[list[tuple[str, str, str]], int, str | None]:
    """Pacchetti con una versione piu' recente su PyPI."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json",
             "--timeout", "15", "--retries", "1", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return [], 0, "controllo librerie scaduto (rete lenta?)"
    except OSError as exc:
        return [], 0, f"impossibile avviare pip: {exc}"
    if out.returncode != 0:
        return [], 0, f"pip non ha risposto: {out.stderr.strip()[:200]}"

    direct: list[tuple[str, str, str]] = []
    indirect = 0
    try:
        packages = json.loads(out.stdout or "[]")
        if not isinstance(packages, list) or any(
            not isinstance(pkg, dict) or not all(isinstance(pkg.get(key), str)
                for key in ("name", "version", "latest_version")) for pkg in packages
        ):
            raise ValueError("formato non valido")
    except (ValueError, TypeError):
        return [], 0, "risposta pip non valida"
    normalized = {re.sub(r"[-_.]+", "-", name).lower() for name in DIRECT_PACKAGES}
    for pkg in packages:
        row = (pkg["name"], pkg["version"], pkg["latest_version"])
        if re.sub(r"[-_.]+", "-", pkg["name"]).lower() in normalized:
            direct.append(row)
        else:
            indirect += 1
    return direct, indirect, None


def _cached_repos() -> list[tuple[str, Path]]:
    """Repo HuggingFace presenti nella cartella models: (nome, cartella)."""
    hub = paths.models_dir() / "hub"
    out: list[tuple[str, Path]] = []
    for d in sorted(hub.glob("models--*")):
        parts = d.name.split("--")[1:]
        if parts:
            out.append(("/".join(parts), d))
    return out


def _model_status(repo: str, folder: Path, token: str | None) -> tuple[str, str]:
    """Confronta la revisione scaricata con quella pubblicata."""
    from huggingface_hub import HfApi

    ref = folder / "refs" / "main"
    local = ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
    remote = HfApi().model_info(repo, token=token, timeout=15).sha or ""
    if not local:
        return repo, "revisione locale sconosciuta"
    if remote and remote != local:
        return repo, f"aggiornato a monte ({local[:7]} -> {remote[:7]})"
    return repo, "aggiornato"


def check(progress=None) -> Report:
    """Controlla librerie e modelli. progress(messaggio) per la UI."""
    report = Report()
    say = progress or (lambda _m: None)

    say("Controllo le librerie Python…")
    direct, indirect, err = _pip_outdated()
    report.direct, report.indirect_count = direct, indirect
    if err:
        report.errors.append(err)

    from .processing.diarize import read_hf_token

    token = read_hf_token()
    repos = _cached_repos()
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(_model_status, repo, folder, token): repo for repo, folder in repos}
        for i, future in enumerate(as_completed(pending), start=1):
            repo = pending[future]
            say(f"Controllo i modelli… ({i}/{len(repos)}) {repo}")
            try:
                name, status = future.result()
                if status != "aggiornato":
                    report.models.append((name, status))
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{repo}: {type(exc).__name__}")
    report.models.sort()
    report.errors.sort()
    return report


def format_report(report: Report) -> tuple[str, str]:
    """(riassunto, dettaglio) da mostrare nella finestra."""
    lines: list[str] = []
    if report.direct:
        lines.append("Librerie con una versione piu' recente:")
        for name, now, latest in report.direct:
            lines.append(f"  {name}: {now} -> {latest}")
        lines.append("")
    if report.indirect_count:
        lines.append(f"Altre {report.indirect_count} dipendenze indirette aggiornabili.")
        lines.append("")
    if report.models:
        lines.append("Modelli aggiornati a monte:")
        for repo, status in report.models:
            lines.append(f"  {repo}: {status}")
        lines.append("")
    if report.errors:
        lines.append("Non verificato:")
        lines.extend(f"  {e}" for e in report.errors)
        lines.append("")

    if report.is_empty() and not report.errors:
        return "Tutto aggiornato: librerie e modelli sono all'ultima versione.", ""

    lines.append("Per aggiornare, chiudi l'app e lancia:")
    lines.append("    powershell -ExecutionPolicy Bypass -File scripts\\update.ps1")
    lines.append("")
    lines.append("L'aggiornamento viene preparato in una copia indipendente e attivato")
    lines.append("solo dopo le verifiche. L'ambiente precedente resta disponibile.")
    lines.append("Ripristino: scripts\\update.ps1 -Rollback")
    lines.append("Il motore di inferenza e le sue dipendenze sono vincolati alle versioni")
    lines.append("attuali; i modelli non vengono rimossi o aggiornati automaticamente.")

    n = len(report.direct) + len(report.models)
    summary = f"{n} aggiornamenti disponibili." if n else "Controllo completato."
    return summary, "\n".join(lines)
