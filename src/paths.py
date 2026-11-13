"""Percorsi dell'app, indipendenti dalla cartella da cui viene lanciata.

Separazione delle directory per installazioni Windows (Tessitura):
  - Binari applicazione: %LOCALAPPDATA%\\Programs\\Tessitura (immutabile a runtime)
  - Configurazione:      %APPDATA%\\Tessitura
  - Modelli AI:          %LOCALAPPDATA%\\Tessitura\\models (o percorso scelto dall'utente)
  - Cache e update:      %LOCALAPPDATA%\\Tessitura\\cache
  - Log diagnostici:     %LOCALAPPDATA%\\Tessitura\\logs
  - Registrazioni/Dati:  %LOCALAPPDATA%\\Tessitura\\data (o percorso scelto dall'utente;
                         in modalita' sviluppo/portabile usa <root>/data se scrivibile)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def app_root() -> Path:
    """Cartella di installazione o root del repository."""
    if getattr(sys, "frozen", False):  # eseguibile PyInstaller
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def install_dir() -> Path:
    """Percorso standard di installazione per-utente su Windows (%LOCALAPPDATA%\\Programs\\Tessitura)."""
    env = os.environ.get("TESSITURA_INSTALL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / "Programs" / "Tessitura"
    return app_root()


def is_installed_mode() -> bool:
    """Ritorna True quando l'app gira come pacchetto installato (non dalla repo di sviluppo).

    In modalita' installata nessun dato utente, modello, cache o log viene mai scritto
    dentro `app_root()`, cosi' aggiornamenti e disinstallazioni non toccano i file utente.
    """
    env_flag = os.environ.get("TESSITURA_INSTALLED", "").strip().lower()
    if env_flag in {"1", "true", "yes"}:
        return True
    if env_flag in {"0", "false", "no"}:
        return False

    root = app_root()
    if (root / ".portable").is_file() or os.environ.get("TESSITURA_PORTABLE") == "1":
        return False
    if getattr(sys, "frozen", False):
        return True
    if (root / ".installed").is_file():
        return True
    try:
        inst = install_dir().resolve()
        if root.resolve() == inst or inst in root.resolve().parents:
            return True
    except OSError:
        pass
    return False


def _user_data_dir() -> Path:
    env = os.environ.get("TESSITURA_USER_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    preferred_cap = Path(base) / "Tessitura"
    preferred_low = Path(base) / "tessitura"
    legacy = Path(base) / "kisp-like"
    # Su Windows il file system e' case-insensitive, ma preserviamo eventuale cartella
    # esistente; per nuove installazioni usiamo "Tessitura" (compatibile con "tessitura").
    if preferred_cap.exists():
        return preferred_cap
    if preferred_low.exists():
        return preferred_low
    if legacy.exists():
        return legacy
    return preferred_cap


def config_dir() -> Path:
    """Cartella configurazione utente (%APPDATA%\\Tessitura su Windows)."""
    env = os.environ.get("TESSITURA_CONFIG_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        d = base / "Tessitura"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Preferences" / "Tessitura"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        d = base / "tessitura"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Cartella cache e download temporanei (%LOCALAPPDATA%\\Tessitura\\cache)."""
    env = os.environ.get("TESSITURA_CACHE_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        d = _user_data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def updates_cache_dir() -> Path:
    """Cartella temporanea dedicata per il download e la verifica degli installer di aggiornamento."""
    d = cache_dir() / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    """Cartella log diagnostici (%LOCALAPPDATA%\\Tessitura\\logs)."""
    env = os.environ.get("TESSITURA_LOGS_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        d = _user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_configured_dir(key: str) -> Path | None:
    """Legge un percorso personalizzato da settings.json senza creare dipendenze circolari."""
    cfg_file = config_dir() / "settings.json"
    if not cfg_file.is_file():
        return None
    try:
        raw = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            return None
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return Path(val.strip()).expanduser().resolve()
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


_data_dir_cache: Path | None = None


def reset_caches() -> None:
    """Azzera la cache dei percorsi (utile nei test o dopo cambio impostazioni)."""
    global _data_dir_cache
    _data_dir_cache = None


def data_dir() -> Path:
    """Cartella dati (registrazioni, sessioni, file temporanei)."""
    global _data_dir_cache
    if _data_dir_cache is not None:
        return _data_dir_cache
    env = os.environ.get("TESSITURA_DATA_DIR") or os.environ.get("KISP_DATA_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        configured = _read_configured_dir("recordings_dir")
        if configured is not None and _writable(configured):
            d = configured
        elif is_installed_mode():
            # In modalita' installata non scriviamo MAI dentro %LOCALAPPDATA%\Programs\Tessitura.
            user_root = _user_data_dir()
            # Se esistono gia' sessioni/registrazioni direttamente in user_root, manteniamolo;
            # altrimenti usiamo %LOCALAPPDATA%\Tessitura\data.
            if (user_root / "sessions").exists() or (user_root / "recordings").exists():
                d = user_root
            else:
                d = user_root / "data"
        else:
            d = app_root() / "data"
            if not _writable(d):
                d = _user_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    _data_dir_cache = d
    return d


def models_dir() -> Path:
    """Cache dei modelli (Whisper, pyannote).

    Ordine di priorita':
      1. variabile d'ambiente TESSITURA_MODELS_DIR
      2. percorso personalizzato `models_dir` in %APPDATA%\\Tessitura\\settings.json
      3. in modalita' installata: %LOCALAPPDATA%\\Tessitura\\models
      4. in modalita' sviluppo/portabile: <root>/models se scrivibile, altrimenti %LOCALAPPDATA%\\Tessitura\\models
    """
    env = os.environ.get("TESSITURA_MODELS_DIR")
    if env:
        d = Path(env).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    configured = _read_configured_dir("models_dir")
    if configured is not None and _writable(configured):
        return configured

    if is_installed_mode():
        d = _user_data_dir() / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    d = app_root() / "models"
    if not _writable(d):
        d = _user_data_dir() / "models"
        d.mkdir(parents=True, exist_ok=True)
    return d


def recordings_dir() -> Path:
    return _sub("recordings")


def sessions_dir() -> Path:
    return _sub("sessions")


def work_dir() -> Path:
    return _sub("work")


def _sub(name: str) -> Path:
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- riferimenti audio portabili -------------------------------------------

def to_portable(path: str | Path | None) -> str | None:
    """Percorso da salvare in una sessione: relativo alla cartella dati se possibile."""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return p.relative_to(data_dir().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_media(stored: str | None) -> Path | None:
    """Ritrova un file audio riferito da una sessione, anche se e' stato spostato."""
    if not stored:
        return None
    raw = Path(stored.replace("\\", "/"))
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        data_dir() / raw,
        app_root() / raw,
        Path.cwd() / raw,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    matches = sorted(data_dir().rglob(raw.name))
    return matches[0].resolve() if matches else None


# --- token HuggingFace ------------------------------------------------------

def hf_token_candidates() -> list[Path]:
    """Vecchi file in chiaro dove cercare il token HF, in ordine di priorita'.

    Solo lettura, per compatibilita': il token nuovo va nel Gestore credenziali
    (``processing.hf_access.save_token``) e nessuna funzione lo scrive su file.
    """
    names = (".hf_token", ".huggingface_token")
    roots = (
        config_dir(),
        _user_data_dir(),
        data_dir(),
        Path.cwd(),
        app_root(),
        Path.home(),
    )
    seen: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p not in seen:
                seen.append(p)
    return seen
