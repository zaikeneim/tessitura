"""Hugging Face credentials and access to the gated diarization model.

Token precedence (first match wins):

1. environment variables ``HF_TOKEN``, ``HUGGING_FACE_HUB_TOKEN``,
   ``HUGGINGFACE_TOKEN`` (same convention as huggingface_hub: an explicit
   environment overrides stored credentials);
2. the operating system's credential vault (Windows Credential Manager);
3. a legacy plain-text ``.hf_token`` file from older versions, read only and
   reported as insecure so the user can move it into the vault.

The token value never leaves this module except towards huggingface_hub
calls that download or check models. ``TokenInfo`` describes where a token
comes from without containing it, and every message is passed through
``redact``.
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .. import paths
from ..secure_store import (
    MAX_SECRET_BYTES,
    CredentialStore,
    CredentialStoreError,
    CredentialStoreUnavailable,
    default_store,
)

DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
TOKENS_URL = "https://huggingface.co/settings/tokens"
ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
DIARIZATION_CATEGORY = "diarizzazione"


def model_url(repo: str = DIARIZATION_REPO) -> str:
    return f"https://huggingface.co/{repo}"


# --- redaction ---------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")


def redact(text: str, *secrets: str | None) -> str:
    """Remove token-like strings and the given secrets from user-facing text."""
    result = str(text)
    for secret in secrets:
        if secret and len(secret) >= 4:
            result = result.replace(secret, "***")
    return _TOKEN_PATTERN.sub("hf_***", result)


# --- token sources -----------------------------------------------------------

class TokenSource(str, Enum):
    ENV = "env"
    STORE = "credential_store"
    LEGACY_FILE = "legacy_file"
    NONE = "none"


@dataclass(frozen=True)
class TokenInfo:
    """Where the effective token comes from. Never contains the token."""

    source: TokenSource
    detail: str = ""
    store_name: str = ""
    store_available: bool = True
    store_has_token: bool = False
    store_error: str = ""
    legacy_file: str | None = None

    @property
    def present(self) -> bool:
        return self.source is not TokenSource.NONE

    def describe(self) -> str:
        if self.source is TokenSource.ENV:
            text = f"Variabile d'ambiente {self.detail} (ha la precedenza sul token salvato)"
        elif self.source is TokenSource.STORE:
            text = f"Salvato in: {self.store_name}"
        elif self.source is TokenSource.LEGACY_FILE:
            text = f"File in chiaro {self.detail} (sconsigliato: spostalo nel {self.store_name or 'Gestore credenziali'})"
        else:
            text = "Nessun token configurato"
        if not self.store_available:
            text += " · Gestore credenziali non disponibile"
        elif self.store_error:
            text += f" · {self.store_error}"
        return text


def _env_token() -> tuple[str | None, str]:
    import os
    for name in ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, name
    return None, ""


def _legacy_token() -> tuple[str | None, Path | None]:
    for candidate in paths.hf_token_candidates():
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value, candidate
        except OSError:
            continue
    return None, None


def _store_token(store: CredentialStore) -> tuple[str | None, bool, str]:
    if not store.available():
        return None, False, ""
    try:
        return store.read(), True, ""
    except CredentialStoreUnavailable:
        return None, False, ""
    except CredentialStoreError as exc:
        return None, True, str(exc)


def resolve_token(store: CredentialStore | None = None) -> tuple[str | None, TokenInfo]:
    """Effective token and its description. Backend use only."""
    store = store or default_store()
    stored, available, error = _store_token(store)
    legacy, legacy_path = _legacy_token()
    common = dict(store_name=store.name, store_available=available,
                  store_has_token=bool(stored), store_error=error,
                  legacy_file=str(legacy_path) if legacy_path else None)
    env, env_name = _env_token()
    if env:
        return env, TokenInfo(TokenSource.ENV, env_name, **common)
    if stored:
        return stored, TokenInfo(TokenSource.STORE, store.name, **common)
    if legacy:
        return legacy, TokenInfo(TokenSource.LEGACY_FILE, str(legacy_path), **common)
    return None, TokenInfo(TokenSource.NONE, **common)


def token_info(store: CredentialStore | None = None) -> TokenInfo:
    """Safe for the GUI: describes the configuration without the secret."""
    return resolve_token(store)[1]


def read_token(store: CredentialStore | None = None) -> str | None:
    return resolve_token(store)[0]


def validate_token_format(token: str) -> str:
    value = (token or "").strip()
    if not value:
        raise ValueError("Inserisci un token Hugging Face.")
    if any(ch.isspace() or not ch.isprintable() or ord(ch) > 126 for ch in value):
        raise ValueError("Il token contiene spazi o caratteri non validi: copialo di nuovo da Hugging Face.")
    if len(value) < 8:
        raise ValueError("Il token è troppo corto.")
    if len(value.encode("utf-8")) > MAX_SECRET_BYTES:
        raise ValueError("Il token è troppo lungo.")
    return value


def save_token(token: str, store: CredentialStore | None = None) -> TokenInfo:
    """Store the token in the OS vault. Never falls back to a file."""
    store = store or default_store()
    value = validate_token_format(token)
    if not store.available():
        raise CredentialStoreUnavailable(AccessIssue.STORE_UNAVAILABLE.message)
    store.write(value)
    return token_info(store)


def remove_token(store: CredentialStore | None = None) -> bool:
    store = store or default_store()
    if not store.available():
        raise CredentialStoreUnavailable(AccessIssue.STORE_UNAVAILABLE.message)
    return store.delete()


def import_legacy_token(store: CredentialStore | None = None) -> Path:
    """Move a legacy plain-text token file into the vault, then delete the file."""
    store = store or default_store()
    value, path = _legacy_token()
    if value is None or path is None:
        raise ValueError("Nessun file .hf_token da importare.")
    save_token(value, store)
    try:
        path.unlink()
    except OSError as exc:
        raise CredentialStoreError(
            f"Token salvato nel {store.name}, ma il file {path} non è stato eliminato: eliminalo a mano."
        ) from exc
    return path


# --- access errors -----------------------------------------------------------

class AccessIssue(str, Enum):
    TOKEN_MISSING = "token_missing"
    TOKEN_INVALID = "token_invalid"
    ACCESS_DENIED = "access_denied"
    TERMS_NOT_ACCEPTED = "terms_not_accepted"
    NETWORK = "network"
    DOWNLOAD_INCOMPLETE = "download_incomplete"
    CACHE_CORRUPT = "cache_corrupt"
    MODEL_INCOMPATIBLE = "model_incompatible"
    MODEL_NOT_CACHED = "model_not_cached"
    STORE_UNAVAILABLE = "store_unavailable"

    @property
    def message(self) -> str:
        return _MESSAGES[self]

    @property
    def needs_token_setup(self) -> bool:
        return self in (AccessIssue.TOKEN_MISSING, AccessIssue.TOKEN_INVALID,
                        AccessIssue.ACCESS_DENIED, AccessIssue.STORE_UNAVAILABLE)


_MESSAGES = {
    AccessIssue.TOKEN_MISSING: (
        "Nessun token Hugging Face configurato. Serve solo per scaricare la prima volta "
        "il modello di diarizzazione."),
    AccessIssue.TOKEN_INVALID: (
        "Il token Hugging Face non è valido, è scaduto o è stato revocato. "
        "Creane uno nuovo di tipo «Read» e salvalo."),
    AccessIssue.ACCESS_DENIED: (
        "Il token è valido ma non può leggere il modello di diarizzazione. Con un token "
        "«fine-grained» abilita la lettura dei repository gated pubblici, oppure usa un token «Read»."),
    AccessIssue.TERMS_NOT_ACCEPTED: (
        "Le condizioni d'uso del modello di diarizzazione non sono ancora state accettate "
        "su Hugging Face con l'account di questo token."),
    AccessIssue.NETWORK: (
        "Hugging Face non è raggiungibile: rete assente, proxy o firewall. "
        "Riprova: i file già scaricati vengono riutilizzati."),
    AccessIssue.DOWNLOAD_INCOMPLETE: (
        "Download interrotto o incompleto. Riprova: il download riprende dai file già scaricati."),
    AccessIssue.CACHE_CORRUPT: (
        "La cache del modello di diarizzazione è danneggiata: dimensioni o checksum non corrispondono. "
        "Elimina la cache del modello e scaricalo di nuovo."),
    AccessIssue.MODEL_INCOMPATIBLE: (
        "Il modello di diarizzazione non si carica con la versione di pyannote.audio installata."),
    AccessIssue.MODEL_NOT_CACHED: (
        "Il modello di diarizzazione non è presente completo nella cache locale."),
    AccessIssue.STORE_UNAVAILABLE: (
        "Il Gestore credenziali del sistema non è disponibile. Imposta la variabile d'ambiente HF_TOKEN."),
}


class ModelAccessError(RuntimeError):
    """A classified, user-presentable failure. Contains no secret."""

    def __init__(self, kind: AccessIssue | str, detail: str = "", *, repo: str = DIARIZATION_REPO):
        self.kind = AccessIssue(kind)
        self.detail = redact(detail)
        self.repo = repo
        super().__init__(self.user_message)

    @property
    def user_message(self) -> str:
        return self.kind.message + (f" ({self.detail})" if self.detail else "")

    @property
    def url(self) -> str:
        if self.kind in (AccessIssue.TERMS_NOT_ACCEPTED, AccessIssue.ACCESS_DENIED):
            return model_url(self.repo)
        if self.kind in (AccessIssue.TOKEN_MISSING, AccessIssue.TOKEN_INVALID):
            return TOKENS_URL
        return ""

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "message": self.user_message, "url": self.url, "repo": self.repo}


class _IncompatibleModel(Exception):
    pass


def classify(exc: BaseException, *, token_present: bool, stage: str = "access") -> ModelAccessError | None:
    """Map huggingface_hub, network and cache errors to a distinct issue.

    Returns None for errors that are not about model access, which the caller
    re-raises unchanged.
    """
    if isinstance(exc, ModelAccessError):
        return exc
    if isinstance(exc, CredentialStoreUnavailable):
        return ModelAccessError(AccessIssue.STORE_UNAVAILABLE)
    if isinstance(exc, _IncompatibleModel):
        return ModelAccessError(AccessIssue.MODEL_INCOMPATIBLE, str(exc))
    from huggingface_hub import errors as hub_errors
    import httpx

    offline = getattr(hub_errors, "OfflineModeIsEnabled", ())
    if (offline and isinstance(exc, offline)) or isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return ModelAccessError(AccessIssue.NETWORK, type(exc).__name__)
    if isinstance(exc, hub_errors.LocalEntryNotFoundError):
        # Raised by snapshot_download when the Hub cannot be reached.
        return ModelAccessError(AccessIssue.NETWORK)
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    server = (getattr(exc, "server_message", None) or "").lower()
    if isinstance(exc, hub_errors.GatedRepoError):
        if status == 401:
            return ModelAccessError(AccessIssue.TOKEN_INVALID if token_present else AccessIssue.TOKEN_MISSING)
        if "fine-grained" in server or "permission" in server or "scope" in server:
            return ModelAccessError(AccessIssue.ACCESS_DENIED, "permessi del token insufficienti")
        if "awaiting" in server or "pending" in server:
            return ModelAccessError(AccessIssue.ACCESS_DENIED, "richiesta di accesso in attesa di approvazione")
        return ModelAccessError(AccessIssue.TERMS_NOT_ACCEPTED)
    if isinstance(exc, hub_errors.RepositoryNotFoundError):
        return ModelAccessError(AccessIssue.ACCESS_DENIED if token_present else AccessIssue.TOKEN_MISSING)
    if isinstance(exc, hub_errors.HfHubHTTPError):
        if status == 401:
            return ModelAccessError(AccessIssue.TOKEN_INVALID if token_present else AccessIssue.TOKEN_MISSING)
        if status == 403:
            return ModelAccessError(AccessIssue.ACCESS_DENIED, f"HTTP {status}")
        if status is not None and (status == 429 or status >= 500):
            return ModelAccessError(AccessIssue.NETWORK, f"servizio Hugging Face non disponibile, HTTP {status}")
        if stage == "download":
            return ModelAccessError(AccessIssue.DOWNLOAD_INCOMPLETE, f"HTTP {status}" if status else "")
        return ModelAccessError(AccessIssue.NETWORK, f"HTTP {status}" if status else "")
    if stage == "download" and isinstance(exc, ValueError):
        text = str(exc)
        if "incompleto" in text.lower():
            return ModelAccessError(AccessIssue.DOWNLOAD_INCOMPLETE, text)
        return ModelAccessError(AccessIssue.CACHE_CORRUPT, text)
    if stage == "download" and isinstance(exc, OSError):
        return ModelAccessError(AccessIssue.DOWNLOAD_INCOMPLETE, type(exc).__name__)
    return None


# --- verification ------------------------------------------------------------

@dataclass(frozen=True)
class AccessReport:
    account: str
    token_role: str
    gated: str
    repo: str = DIARIZATION_REPO


def _api():
    from ..bootstrap import install_system_certs
    install_system_certs()
    from huggingface_hub import HfApi
    return HfApi()


def check_token(token: str, *, api=None) -> tuple[str, str]:
    """Account name and token role. Raises ModelAccessError."""
    api = api or _api()
    try:
        info = api.whoami(token=token)
    except Exception as exc:  # noqa: BLE001 - classified below
        error = classify(exc, token_present=True, stage="token")
        if error is None:
            raise
        raise error from None
    role = ((info.get("auth") or {}).get("accessToken") or {}).get("role", "") if isinstance(info, dict) else ""
    return (info.get("name", "") if isinstance(info, dict) else ""), str(role)


def check_repo_access(token: str, *, repo: str = DIARIZATION_REPO, api=None) -> str:
    """Gating mode of the repository once access is confirmed. Raises ModelAccessError."""
    api = api or _api()
    try:
        gated = api.model_info(repo, token=token, timeout=20).gated
        api.auth_check(repo, token=token)
    except Exception as exc:  # noqa: BLE001 - classified below
        error = classify(exc, token_present=True, stage="access")
        if error is None:
            raise
        error.repo = repo
        raise error from None
    return str(gated or "no")


def check_access(token: str | None = None, *, repo: str = DIARIZATION_REPO,
                 store: CredentialStore | None = None, api=None) -> AccessReport:
    """Token validity, then repository access and accepted conditions."""
    token = token if token is not None else read_token(store)
    if not token:
        raise ModelAccessError(AccessIssue.TOKEN_MISSING, repo=repo)
    api = api or _api()
    account, role = check_token(token, api=api)
    gated = check_repo_access(token, repo=repo, api=api)
    return AccessReport(account=account, token_role=role, gated=gated, repo=repo)


def verify_configuration(token: str | None = None, *, store: CredentialStore | None = None,
                         repo: str = DIARIZATION_REPO, api=None) -> dict:
    """Result for the settings screen; contains no secret.

    ``state``: ``not_configured`` | ``valid`` | ``invalid`` | ``unverified``.
    A valid token without access to the model stays ``valid`` with
    ``model_access=False``: the token works, the account must accept the
    conditions or widen the token permissions.
    """
    typed = token is not None
    if token is None:
        token = read_token(store)
    if not token:
        error = ModelAccessError(AccessIssue.TOKEN_MISSING, repo=repo)
        return {"state": "not_configured", "kind": error.kind.value, "message": error.user_message,
                "url": error.url, "typed": typed}
    try:
        api = api or _api()
        account, role = check_token(token, api=api)
    except ModelAccessError as error:
        state = "invalid" if error.kind in (AccessIssue.TOKEN_INVALID, AccessIssue.TOKEN_MISSING) else "unverified"
        return {"state": state, "kind": error.kind.value, "message": error.user_message,
                "url": error.url, "typed": typed}
    except Exception as exc:  # noqa: BLE001 - the screen stays usable
        return {"state": "unverified", "kind": None, "typed": typed,
                "message": redact(f"Verifica non riuscita: {type(exc).__name__}: {exc}", token), "url": ""}
    result = {"state": "valid", "account": account, "role": role, "typed": typed, "repo": repo}
    try:
        gated = check_repo_access(token, repo=repo, api=api)
    except ModelAccessError as error:
        result.update(model_access=False, kind=error.kind.value, message=error.user_message, url=error.url)
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(model_access=None, kind=None, url="",
                      message=redact(f"Accesso al modello non verificato: {type(exc).__name__}", token))
        return result
    terms = "condizioni accettate" if gated != "no" else "modello senza condizioni"
    result.update(model_access=True, kind=None, url="", gated=gated,
                  message=f"Accesso a {repo} confermato · {terms}.")
    return result


# --- local cache and download ------------------------------------------------

def local_model(repo: str = DIARIZATION_REPO) -> str | None:
    """Verified local snapshot (activation pointer or refs/main), else None."""
    from . import model_store
    try:
        found = model_store.cached_path(repo, DIARIZATION_CATEGORY)
    except (OSError, ValueError):
        return None
    return str(found) if found is not None else None


def local_cache_state(repo: str = DIARIZATION_REPO) -> str:
    """'ready', 'missing' or 'incomplete' (files present but not verifiable)."""
    if local_model(repo):
        return "ready"
    from . import model_store
    try:
        snapshots = model_store.repo_root(repo) / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return "incomplete"
    except (OSError, ValueError):
        pass
    return "missing"


def _validate_pipeline_load(path: str) -> None:
    """The downloaded bundle must load on CPU before it is activated."""
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(str(path))
    except Exception as exc:  # noqa: BLE001 - any load failure means incompatible
        raise _IncompatibleModel(type(exc).__name__) from exc
    if pipeline is None:
        raise _IncompatibleModel("pipeline non caricabile")
    del pipeline
    gc.collect()


StepCallback = Callable[[str, str, str], None]  # step, state, detail
STEPS = ("cache", "token", "access", "terms", "download", "integrity")


def ensure_model(repo: str = DIARIZATION_REPO, *, token: str | None = None,
                 store: CredentialStore | None = None,
                 on_download: Callable[[str, int], None] | None = None,
                 on_step: StepCallback | None = None, repair: bool = False,
                 api=None, ensure_fn=None, validate_load=None) -> str:
    """Make the diarization model available, verified and activated.

    Steps: local cache, credentials, repository access, accepted conditions,
    download (resumed by huggingface_hub from partial files), integrity
    (sizes and SHA-256 against Hub metadata, then a CPU load test). Only a
    verified snapshot is activated, so the diarizer and the catalog find it.
    ``repair`` deletes an unusable cache before downloading again.
    """
    from . import model_store

    def step(name: str, state: str, detail: str = "") -> None:
        if on_step is not None:
            on_step(name, state, redact(detail, token))

    step("cache", "running", "Controllo della cache locale")
    if repair:
        model_store.remove(repo)
        step("cache", "done", "Cache eliminata: il modello verrà scaricato di nuovo")
    else:
        found = local_model(repo)
        if found:
            step("cache", "done", "Modello completo in cache")
            for name in STEPS[1:-1]:
                step(name, "skipped", "Non necessario: modello già in cache")
            step("integrity", "done", f"Revisione {Path(found).name[:12]}")
            return found
        state = local_cache_state(repo)
        step("cache", "failed", "Cache incompleta: verrà completata" if state == "incomplete"
             else "Modello non presente in cache")

    step("token", "running", "Ricerca del token")
    if token is None:
        token, info = resolve_token(store)
    else:
        info = None
    if not token:
        error = ModelAccessError(AccessIssue.TOKEN_MISSING, repo=repo)
        step("token", "failed", error.user_message)
        raise error
    api = api or _api()
    try:
        account, _role = check_token(token, api=api)
    except ModelAccessError as error:
        step("token", "failed", error.user_message)
        raise
    source = f" · {info.describe()}" if info is not None else ""
    step("token", "done", f"Account {account}{source}" if account else f"Token valido{source}")

    step("access", "running", "Verifica accesso al repository")
    step("terms", "running", "Verifica delle condizioni d'uso")
    try:
        gated = check_repo_access(token, repo=repo, api=api)
    except ModelAccessError as error:
        if error.kind is AccessIssue.TERMS_NOT_ACCEPTED:
            step("access", "done", "Repository raggiungibile")
            step("terms", "failed", error.user_message)
        else:
            step("access", "failed", error.user_message)
            step("terms", "skipped", "")
        raise
    step("access", "done", "Accesso consentito")
    step("terms", "done", "Condizioni accettate" if gated != "no" else "Modello senza condizioni")

    step("download", "running", "Download in corso")
    try:
        path = (ensure_fn or model_store.ensure)(
            repo, DIARIZATION_CATEGORY, token=token, on_download=on_download,
            validate_load=validate_load or _validate_pipeline_load)
    except Exception as exc:  # noqa: BLE001 - classified below
        error = classify(exc, token_present=True, stage="download")
        if error is None:
            step("download", "failed", redact(str(exc), token))
            raise
        error.repo = repo
        failed_step = "integrity" if error.kind in (AccessIssue.CACHE_CORRUPT,
                                                     AccessIssue.MODEL_INCOMPATIBLE) else "download"
        if failed_step == "integrity":
            step("download", "done", "File scaricati")
        step(failed_step, "failed", error.user_message)
        raise error from None
    step("download", "done", "File scaricati")
    step("integrity", "running", "Verifica dell'integrità")
    final = local_model(repo)
    if final is None or Path(final).resolve() != Path(path).resolve():
        error = ModelAccessError(AccessIssue.CACHE_CORRUPT, "verifica finale della cache non superata", repo=repo)
        step("integrity", "failed", error.user_message)
        raise error
    step("integrity", "done", f"Checksum verificati · revisione {Path(final).name[:12]}")
    return final


def diarization_model_ready() -> bool:
    return local_model(DIARIZATION_REPO) is not None
