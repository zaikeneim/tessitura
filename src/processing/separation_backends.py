"""Backend pluggabili per la separazione di voci simultanee (piano §6.4, guardrail P06).

Rispecchia il contratto di ASRBackend: `name`, `is_available()` e il metodo di
lavoro `separate_region`. La priorita' va a ONNX Runtime (DirectML dove esiste,
CPU altrimenti) per non contendere la VRAM a CUDA/PyTorch durante la pipeline.

Regola offline rigorosa: i pesi vivono nella cartella locale dei modelli e devono
corrispondere allo SHA-256 dichiarato nel manifest. Un peso assente dal manifest,
o con impronta diversa, vale come backend non disponibile: nessun download
implicito e nessun modello non verificato caricato a runtime.
"""

from __future__ import annotations

import abc
import hashlib
import itertools
import json
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..paths import models_dir

SEPARATION_DIRNAME = "separation"
MANIFEST_NAME = "manifest.json"
MODEL_SAMPLE_RATE = 16000
MAX_PERMUTATION_TRACKS = 4


def separation_dir() -> Path:
    """Cartella locale dei pesi di separazione."""
    return models_dir() / SEPARATION_DIRNAME


def file_sha256(path: str | Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory: str | Path | None = None) -> dict[str, Any]:
    """Voci del manifest dei pesi verificati, vuoto se assente o illeggibile."""
    root = Path(directory) if directory is not None else separation_dir()
    path = root / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = data.get("models")
    return models if isinstance(models, dict) else {}


def manifest_entry(path: str | Path) -> dict[str, Any]:
    """Voce di manifest del peso indicato, vuota se assente."""
    path = Path(path)
    entry = load_manifest(path.parent).get(path.name)
    return entry if isinstance(entry, dict) else {}


def verify_model(path: str | Path, directory: str | Path | None = None) -> tuple[bool, str]:
    """Esito e motivo della verifica offline di un peso: (verificato, motivo)."""
    path = Path(path)
    if not path.exists():
        return False, "model_file_missing"
    entry = load_manifest(directory if directory is not None else path.parent).get(path.name)
    if not isinstance(entry, dict) or not entry.get("sha256"):
        return False, "manifest_entry_missing"
    if file_sha256(path) != str(entry["sha256"]).strip().lower():
        return False, "sha256_mismatch"
    return True, "verified"


def _resample(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    divisor = gcd(int(source_sr), int(target_sr))
    resampled = resample_poly(signal, target_sr // divisor, int(source_sr) // divisor)
    return resampled.astype(np.float32, copy=False)


def _fit_length(signal: np.ndarray, length: int) -> np.ndarray:
    if signal.size == length:
        return signal
    if signal.size > length:
        return signal[:length]
    return np.pad(signal, (0, length - signal.size))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    width = min(a.size, b.size)
    if width < 2:
        return 0.0
    left, right = a[:width], b[:width]
    if float(np.std(left)) < 1e-9 or float(np.std(right)) < 1e-9:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _as_tracks(outputs: list[Any], expected_len: int) -> np.ndarray:
    """Normalizza l'uscita del modello in una matrice (tracce, campioni).

    Gli export ONNX dei separatori differiscono: alcuni emettono un tensore
    (batch, parlanti, campioni), altri un'uscita per parlante. Entrambe le forme
    devono arrivare al chiamante come righe di una matrice.
    """
    arrays = [np.squeeze(np.asarray(item, dtype=np.float32)) for item in outputs]
    arrays = [item for item in arrays if item.size]
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)

    if len(arrays) > 1 and all(item.ndim == 1 for item in arrays):
        width = min(item.size for item in arrays)
        return np.stack([item[:width] for item in arrays])

    first = arrays[0]
    if first.ndim == 1:
        return first[np.newaxis, :]
    if first.ndim == 2:
        rows, cols = first.shape
        # L'asse dei campioni e' quello di lunghezza confrontabile con il blocco.
        return first if abs(cols - expected_len) <= abs(rows - expected_len) else first.T
    return first.reshape(-1, first.shape[-1])


def _crossfade_window(width: int, fade: int) -> np.ndarray:
    window = np.ones(width, dtype=np.float32)
    fade = min(fade, width)
    if fade > 1:
        window[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    return window


def _best_permutation(reference: np.ndarray, candidate: np.ndarray) -> list[int]:
    """Ordine delle tracce del blocco corrente che combacia con quelle precedenti.

    Senza questo passaggio un separatore, che non garantisce l'ordine delle uscite,
    scambierebbe i parlanti da una finestra all'altra spezzando l'identita' (§6.4).
    """
    count = candidate.shape[0]
    identity = list(range(count))
    if count != reference.shape[0] or count > MAX_PERMUTATION_TRACKS:
        return identity
    best_order, best_score = identity, -np.inf
    for order in itertools.permutations(range(count)):
        score = sum(abs(_corr(reference[i], candidate[j])) for i, j in enumerate(order))
        if score > best_score:
            best_score, best_order = score, list(order)
    return best_order


class SeparationBackend(abc.ABC):
    """Interfaccia astratta per un separatore di voci simultanee."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.options = kwargs

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Nome identificativo del backend (es. 'onnx', 'null')."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Verifica librerie e pesi locali verificati, senza scaricare nulla."""

    @abc.abstractmethod
    def separate_region(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: int = 2,
    ) -> list[np.ndarray]:
        """Tracce separate della regione; lista vuota se non ne produce."""

    @property
    def unavailable_reason(self) -> str:
        return ""


class NullSeparationBackend(SeparationBackend):
    """Ripiego deterministico: disponibile ovunque, non separa nulla."""

    def __init__(self, model_name: str = "null", **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)

    @property
    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True

    def separate_region(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: int = 2,
    ) -> list[np.ndarray]:
        return []

    @property
    def unavailable_reason(self) -> str:
        return "null_backend_never_separates"


class OnnxSeparationBackend(SeparationBackend):
    """Separatore su ONNX Runtime con pesi locali verificati.

    Modelli di riferimento: MossFormer2 o Conv-TasNet esportati in ONNX a 16 kHz.
    Le regioni lunghe vengono elaborate a finestre sovrapposte e ricucite con
    riconciliazione delle identita', perche' un separatore non garantisce lo
    stesso ordine di uscita fra una finestra e la successiva.
    """

    def __init__(
        self,
        model_name: str = "mossformer2_ss_16k",
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        chunk_seconds: float = 15.0,
        overlap_seconds: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.model_path = (
            Path(model_path) if model_path else separation_dir() / f"{model_name}.onnx"
        )
        self.providers = providers
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self._session: Any | None = None
        self._reason = ""

    @property
    def name(self) -> str:
        return "onnx"

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self._reason = "onnxruntime_missing"
            return False
        verified, reason = verify_model(self.model_path)
        self._reason = "" if verified else reason
        return verified

    def resolve_providers(self) -> list[str]:
        import onnxruntime as ort

        wanted = self.providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        return [item for item in wanted if item in available] or ["CPUExecutionProvider"]

    def _get_session(self) -> Any:
        if self._session is None:
            import onnxruntime as ort

            if not self.is_available():
                raise RuntimeError(
                    f"Separatore ONNX non disponibile ({self._reason}): "
                    f"{self.model_path}"
                )
            self._session = ort.InferenceSession(
                str(self.model_path), providers=self.resolve_providers()
            )
        return self._session

    def _infer_chunk(self, chunk: np.ndarray) -> np.ndarray:
        session = self._get_session()
        spec = session.get_inputs()[0]
        payload = chunk.astype(np.float32, copy=False)
        rank = len([dim for dim in (spec.shape or [])]) or 2
        while payload.ndim < rank:
            payload = payload[np.newaxis, ...]
        outputs = session.run(None, {spec.name: payload})
        return _as_tracks(outputs, chunk.size)

    def effective_chunk_seconds(self) -> float:
        """Finestra di lavoro, limitata a quanto il grafo esportato regge davvero.

        Alcuni export congelano la lunghezza d'ingresso negli embedding
        posizionali e falliscono oltre quella tracciata: il manifest la dichiara
        con `max_input_seconds` e qui non la si supera mai.
        """
        ceiling = manifest_entry(self.model_path).get("max_input_seconds")
        if isinstance(ceiling, (int, float)) and ceiling > 0:
            return min(self.chunk_seconds, float(ceiling))
        return self.chunk_seconds

    def _separate_16k(self, work: np.ndarray) -> np.ndarray:
        size = max(1, int(self.effective_chunk_seconds() * MODEL_SAMPLE_RATE))
        if work.size <= size:
            return self._infer_chunk(work)

        overlap = min(size - 1, max(0, int(self.overlap_seconds * MODEL_SAMPLE_RATE)))
        hop = size - overlap
        buffers: np.ndarray | None = None
        weights = np.zeros(work.size, dtype=np.float32)

        for start in range(0, work.size, hop):
            block = work[start : start + size]
            if block.size < MODEL_SAMPLE_RATE // 10:
                break
            produced = self._infer_chunk(block)
            if produced.shape[0] == 0:
                return np.zeros((0, 0), dtype=np.float32)

            width = min(produced.shape[1], work.size - start)
            if buffers is None:
                buffers = np.zeros((produced.shape[0], work.size), dtype=np.float32)
            else:
                settled = weights[start : start + width] > 1e-6
                if settled.any():
                    reference = (
                        buffers[:, start : start + width][:, settled]
                        / weights[start : start + width][settled]
                    )
                    order = _best_permutation(reference, produced[:, :width][:, settled])
                    produced = produced[order]

            window = _crossfade_window(width, overlap if start else 0)
            buffers[:, start : start + width] += produced[:, :width] * window
            weights[start : start + width] += window

        if buffers is None:
            return np.zeros((0, 0), dtype=np.float32)
        return buffers / np.maximum(weights, 1e-6)

    def separate_region(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: int = 2,
    ) -> list[np.ndarray]:
        signal = np.asarray(audio_data, dtype=np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1).astype(np.float32, copy=False)
        if signal.size == 0:
            return []

        work = _resample(signal, int(sample_rate), MODEL_SAMPLE_RATE)
        tracks = self._separate_16k(work)
        if tracks.shape[0] == 0:
            return []
        return [
            _fit_length(_resample(row, MODEL_SAMPLE_RATE, int(sample_rate)), signal.size)
            for row in tracks
        ]


_REGISTRY: dict[str, type[SeparationBackend]] = {
    "onnx": OnnxSeparationBackend,
    "null": NullSeparationBackend,
}


def register_separation_backend(name: str, backend_cls: type[SeparationBackend]) -> None:
    _REGISTRY[name.lower()] = backend_cls


def get_separation_backend(name: str = "onnx", **kwargs: Any) -> SeparationBackend:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Backend di separazione sconosciuto: '{name}'. "
            f"Registrati: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](**kwargs)


def list_available_separation_backends() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, cls in _REGISTRY.items():
        instance = cls()
        available = instance.is_available()
        result[name] = {
            "name": name,
            "class": cls.__name__,
            "available": available,
            "reason": instance.unavailable_reason,
            "offline_safe": True,
        }
    return result
