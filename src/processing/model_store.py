"""Transactional Hugging Face snapshots, shared by all three model families.

Downloads never change the application pointer until integrity and optional
runtime loading succeed. A per-repository OS lock protects updates and deletion
while a model is loaded, including other Tessitura processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .hf_progress import make_progress_tqdm

PATTERNS = {
    "analisi": ["*.json", "*.safetensors", "*.model", "*.txt", "pytorch_model*.bin"],
    "trascrizione": ["config.json", "preprocessor_config.json", "model.bin",
                      "tokenizer.json", "vocabulary.*"],
    "diarizzazione": ["config.yaml", "segmentation/pytorch_model.bin",
                       "embedding/pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"],
}


def hub_root() -> Path:
    from .. import paths
    custom = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if custom:
        return Path(custom).expanduser().resolve()
    home = os.environ.get("HF_HOME")
    return ((Path(home).expanduser() if home else paths.models_dir()) / "hub").resolve()


def repo_root(repo: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", repo):
        raise ValueError(f"Repository modello non valido: {repo!r}")
    return hub_root() / ("models--" + repo.replace("/", "--"))


def _pointer(repo: str) -> Path:
    return hub_root() / ".tessitura" / (repo_root(repo).name + ".json")


def _state(repo: str) -> dict:
    path = _pointer(repo)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Riferimento modello danneggiato")
    return data


def _revision(value) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{40,64}", value) else None


def local_revision(repo: str) -> str | None:
    # A half-written/corrupt pointer must make the model unverifiable, not
    # crash the settings worker or accidentally select an arbitrary snapshot.
    try:
        state = _state(repo)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if state:
        return _revision(state.get("active"))
    ref = repo_root(repo) / "refs" / "main"
    try:
        return _revision(ref.read_text(encoding="utf-8").strip()) if ref.is_file() else None
    except OSError:
        return None


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".model-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def lease(repo: str):
    """Acquire before loading; release only after freeing model resources."""
    from filelock import FileLock, Timeout
    location = hub_root() / ".tessitura" / "locks" / (repo_root(repo).name + ".lock")
    location.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(location))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError("Modello in uso o aggiornamento in corso: riprova a lavoro terminato.") from exc
    return lock


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def verify_snapshot(path: Path, category: str, siblings=None) -> None:
    """Verify manifest completeness and remote hashes/sizes when available."""
    path = Path(path)
    if category == "diarizzazione":
        required = [path / p for p in PATTERNS[category]]
    elif category == "trascrizione":
        required = [path / name for name in ("config.json", "model.bin", "tokenizer.json")]
        vocab = list(path.glob("vocabulary.*"))
        if not vocab:
            raise ValueError("Vocabolario Whisper assente")
        required += vocab
    elif category == "analisi":
        required = [path / name for name in ("config.json", "tokenizer_config.json")]
        tokenizers = [p for p in (path / "tokenizer.json", path / "tokenizer.model") if _nonempty(p)]
        if not tokenizers:
            raise ValueError("Tokenizer assente o incompleto")
        required += tokenizers
        manifests = list(path.glob("*.index.json"))
        weight_files = []
        for manifest in manifests:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            mapping = data.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Manifest shard non valido")
            for name in set(mapping.values()):
                if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("Percorso shard non valido")
                weight_files.append(path / name)
        if not manifests:
            weight_files = [p for p in (path / "model.safetensors", path / "pytorch_model.bin")
                            if p.is_file()]
        if not weight_files:
            raise ValueError("Pesi o manifest dei pesi mancanti")
        required += weight_files + manifests
    else:
        raise ValueError(f"Categoria modello sconosciuta: {category}")
    if not all(_nonempty(p) for p in required):
        raise ValueError("Snapshot incompleto: uno o più file richiesti sono assenti o vuoti")
    for file in required:
        if file.suffix == ".json":
            json.loads(file.read_text(encoding="utf-8"))
    for sibling in siblings or ():
        name = getattr(sibling, "rfilename", "")
        # Metadata is remote input; only verify descendants of this snapshot.
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name:
            raise ValueError("Percorso remoto non valido")
        file = path / relative
        if not file.is_file():
            continue
        size = getattr(sibling, "size", None)
        if size is not None and file.stat().st_size != size:
            raise ValueError(f"Dimensione inattesa: {name}")
        lfs = getattr(sibling, "lfs", None)
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if expected:
            with file.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum errato: {name}")


def cached_path(repo: str, category: str) -> Path | None:
    try:
        revision = local_revision(repo)
        if revision is None:
            return None
        candidate = repo_root(repo) / "snapshots" / revision
        verify_snapshot(candidate, category)
        return candidate
    except (OSError, ValueError, TypeError):
        return None


def ensure(repo: str, category: str, *, update=False, on_download=None,
           token=None, validate_load=None, allow_download=True) -> str:
    old = cached_path(repo, category)
    if old is not None and not update:
        return str(old)
    if not allow_download:
        raise RuntimeError(f"Modello {repo} non disponibile completo in cache; download disattivato.")
    from huggingface_hub import HfApi, snapshot_download
    from ..bootstrap import install_system_certs
    install_system_certs()
    with lease(repo):
        api = HfApi()
        try:
            info = api.model_info(repo, revision="main", token=token,
                                  timeout=20, files_metadata=True)
        except TypeError:  # older huggingface_hub without files_metadata
            info = api.model_info(repo, revision="main", token=token, timeout=20)
        sha = _revision(info.sha)
        if sha is None:
            raise ValueError("Revisione remota non valida")
        kwargs = {"revision": sha, "cache_dir": str(hub_root()), "token": token,
                  "allow_patterns": PATTERNS[category]}
        if on_download:
            kwargs["tqdm_class"] = make_progress_tqdm(on_download)
        candidate = Path(snapshot_download(repo, **kwargs))
        if candidate.resolve() != (repo_root(repo) / "snapshots" / sha).resolve():
            raise ValueError("Snapshot fuori dalla cache prevista")
        verify_snapshot(candidate, category, info.siblings)
        if validate_load is not None:
            validate_load(str(candidate))
            activate_loaded(repo, candidate, category)
        return str(candidate)


def activate_loaded(repo: str, candidate: str | Path, category: str) -> None:
    """Call only after successful runtime load, while holding the repo lease."""
    candidate = Path(candidate)
    sha = _revision(candidate.name)
    if sha is None or candidate.resolve() != (repo_root(repo) / "snapshots" / sha).resolve():
        raise ValueError("Snapshot da attivare non valido")
    verify_snapshot(candidate, category)
    previous = local_revision(repo)
    _atomic_json(_pointer(repo), {"active": sha, "previous": previous if previous != sha
                                 else _state(repo).get("previous"),
                                 "category": category, "load_verified": True})


def rollback(repo: str, category: str) -> str:
    with lease(repo):
        state = _state(repo)
        previous = _revision(state.get("previous"))
        if previous is None:
            raise RuntimeError("Nessuna revisione precedente disponibile")
        candidate = repo_root(repo) / "snapshots" / previous
        verify_snapshot(candidate, category)
        _atomic_json(_pointer(repo), {**state, "active": previous, "previous": state["active"]})
        return str(candidate)


def disk_bytes(repo: str) -> int:
    # HF snapshots link to blobs: count each physical target only once.
    files = {p.resolve() for p in repo_root(repo).rglob("*") if p.is_file()}
    return sum(p.stat().st_size for p in files)


def remove(repo: str) -> int:
    with lease(repo):
        root = repo_root(repo).resolve()
        if root.parent != hub_root().resolve() or not root.name.startswith("models--"):
            raise ValueError("Percorso cache modello non sicuro")
        size = disk_bytes(repo)
        if root.exists():
            shutil.rmtree(root)
        _pointer(repo).unlink(missing_ok=True)
        return size
