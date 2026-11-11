"""Enumerazione dei dispositivi audio (microfoni e loopback di sistema).

Su Windows l'audio "che esce dalle casse" si cattura via WASAPI loopback: per ogni
dispositivo di output esiste un dispositivo di loopback corrispondente che si puo'
aprire come fosse un input. Qui usiamo PyAudioWPatch, il fork di PyAudio con
supporto al loopback WASAPI.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pyaudiowpatch as pyaudio
except ImportError as _exc:  # backend audio assente (non-Windows, o pacchetto mancante)
    pyaudio = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


def audio_backend_available() -> bool:
    """False se la cattura audio non e' utilizzabile su questo PC."""
    return pyaudio is not None


def require_backend() -> None:
    if pyaudio is None:
        raise RuntimeError(
            "Cattura audio non disponibile su questo sistema "
            f"(PyAudioWPatch richiede Windows): {_IMPORT_ERROR}"
        )


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool
    host_api: int | None = None

    def short(self) -> str:
        kind = "loopback" if self.is_loopback else "input"
        return f"[{self.index}] {self.name} ({self.channels}ch @ {self.sample_rate}Hz, {kind})"


def _to_info(d: dict, *, loopback: bool) -> DeviceInfo:
    return DeviceInfo(
        index=int(d["index"]),
        name=str(d["name"]),
        channels=int(d["maxInputChannels"]) or int(d.get("maxOutputChannels", 0)),
        sample_rate=int(d["defaultSampleRate"]),
        is_loopback=loopback,
        host_api=int(d["hostApi"]) if "hostApi" in d else None,
    )


def list_input_devices(p: pyaudio.PyAudio) -> list[DeviceInfo]:
    """Microfoni / dispositivi di ingresso reali (esclude i loopback)."""
    out: list[DeviceInfo] = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if int(d["maxInputChannels"]) > 0 and not d.get("isLoopbackDevice", False):
            out.append(_to_info(d, loopback=False))
    return out


def list_loopback_devices(p: pyaudio.PyAudio) -> list[DeviceInfo]:
    """Dispositivi di loopback WASAPI (audio in uscita dal PC)."""
    out: list[DeviceInfo] = []
    for d in p.get_loopback_device_info_generator():
        out.append(_to_info(d, loopback=True))
    return out


def default_microphone(p: pyaudio.PyAudio) -> DeviceInfo | None:
    try:
        d = p.get_default_input_device_info()
        if not d.get("isLoopbackDevice", False):
            return _to_info(d, loopback=False)
    except OSError:
        pass
    inputs = list_input_devices(p)
    return inputs[0] if inputs else None


def default_system_loopback(p: pyaudio.PyAudio) -> DeviceInfo | None:
    """Loopback corrispondente al dispositivo di output predefinito."""
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        name = default_out["name"]
        matches = [lb for lb in p.get_loopback_device_info_generator()
                   if lb["name"].removesuffix(" [Loopback]") == name]
        return _to_info(matches[0], loopback=True) if len(matches) == 1 else None
    except (OSError, IndexError, KeyError):
        return None


def resolve_device(p: pyaudio.PyAudio, selected: DeviceInfo) -> DeviceInfo:
    """Resolve a refreshed endpoint without silently recording another device."""
    candidates = list_loopback_devices(p) if selected.is_loopback else list_input_devices(p)
    matches = [device for device in candidates
               if device.name == selected.name
               and (selected.host_api is None or device.host_api == selected.host_api)]
    if len(matches) != 1:
        raise RuntimeError(f"Dispositivo scollegato o ambiguo: {selected.name}. Aggiorna i dispositivi.")
    # Use its current native format: USB/Bluetooth reconnects may change both
    # PortAudio indices and supported sample rates.
    return matches[0]


@dataclass
class DeviceCatalog:
    microphones: list[DeviceInfo]
    loopbacks: list[DeviceInfo]
    default_mic: DeviceInfo | None
    default_loopback: DeviceInfo | None


def get_devices() -> DeviceCatalog:
    """Elenca microfoni e loopback + i predefiniti, gestendo PyAudio internamente."""
    require_backend()
    p = pyaudio.PyAudio()
    try:
        return DeviceCatalog(
            microphones=list_input_devices(p),
            loopbacks=list_loopback_devices(p),
            default_mic=default_microphone(p),
            default_loopback=default_system_loopback(p),
        )
    finally:
        p.terminate()


def describe_all() -> str:
    """Stringa diagnostica con tutti i dispositivi (utile da CLI)."""
    require_backend()
    p = pyaudio.PyAudio()
    try:
        lines = ["== Microfoni =="]
        for d in list_input_devices(p):
            lines.append("  " + d.short())
        lines.append("== Loopback di sistema ==")
        for d in list_loopback_devices(p):
            lines.append("  " + d.short())
        mic = default_microphone(p)
        sys_lb = default_system_loopback(p)
        lines.append(f"-> mic predefinito:      {mic.short() if mic else 'nessuno'}")
        lines.append(f"-> loopback predefinito: {sys_lb.short() if sys_lb else 'nessuno'}")
        return "\n".join(lines)
    finally:
        p.terminate()


if __name__ == "__main__":
    print(describe_all())
