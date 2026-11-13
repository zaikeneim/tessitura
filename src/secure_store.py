"""Secret storage in the operating system's credential vault.

Windows: Credential Manager (``advapi32`` ``CredWriteW``/``CredReadW``/
``CredDeleteW``), generic credential, persisted for the current user on this
computer only (``CRED_PERSIST_LOCAL_MACHINE``: it does not roam). No extra
package is needed and nothing is written to a file.

Other platforms have no backend here: the store reports itself unavailable and
callers fall back to environment variables. Error messages never contain the
secret.
"""
from __future__ import annotations

import sys
from typing import Protocol

DEFAULT_TARGET = "Tessitura/HuggingFace"
# CRED_MAX_CREDENTIAL_BLOB_SIZE = 5 * 512 bytes.
MAX_SECRET_BYTES = 5 * 512


class CredentialStoreError(RuntimeError):
    """The vault refused an operation. The message never contains the secret."""


class CredentialStoreUnavailable(CredentialStoreError):
    """No secure vault is available on this system."""


class CredentialStore(Protocol):
    name: str

    def available(self) -> bool: ...
    def read(self) -> str | None: ...
    def write(self, secret: str) -> None: ...
    def delete(self) -> bool: ...


class MemoryCredentialStore:
    """In-process store for tests. Never persisted."""

    name = "memoria (test)"

    def __init__(self, secret: str | None = None, *, available: bool = True):
        self._secret = secret
        self._available = available

    def available(self) -> bool:
        return self._available

    def _check(self) -> None:
        if not self._available:
            raise CredentialStoreUnavailable("Archivio credenziali non disponibile.")

    def read(self) -> str | None:
        self._check()
        return self._secret

    def write(self, secret: str) -> None:
        self._check()
        self._secret = secret

    def delete(self) -> bool:
        self._check()
        existed = self._secret is not None
        self._secret = None
        return existed


class UnavailableCredentialStore(MemoryCredentialStore):
    name = "non disponibile"

    def __init__(self):
        super().__init__(available=False)


class WindowsCredentialStore:
    """Generic credential in Windows Credential Manager."""

    name = "Gestore credenziali di Windows"
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    _ERROR_NOT_FOUND = 1168

    def __init__(self, target: str = DEFAULT_TARGET):
        self.target = target
        self._api = None

    def available(self) -> bool:
        try:
            self._load()
            return True
        except CredentialStoreUnavailable:
            return False

    def _load(self):
        if self._api is not None:
            return self._api
        if not sys.platform.startswith("win"):
            raise CredentialStoreUnavailable("Gestore credenziali disponibile solo su Windows.")
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class CREDENTIALW(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        try:
            advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        except OSError as exc:
            raise CredentialStoreUnavailable("Gestore credenziali di Windows non raggiungibile.") from exc
        pcred = ctypes.POINTER(CREDENTIALW)
        advapi.CredWriteW.argtypes = [pcred, wintypes.DWORD]
        advapi.CredWriteW.restype = wintypes.BOOL
        advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pcred)]
        advapi.CredReadW.restype = wintypes.BOOL
        advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        advapi.CredDeleteW.restype = wintypes.BOOL
        advapi.CredFree.argtypes = [ctypes.c_void_p]
        advapi.CredFree.restype = None
        self._api = (ctypes, advapi, CREDENTIALW, pcred)
        return self._api

    def read(self) -> str | None:
        ctypes, advapi, _cred, pcred = self._load()
        out = pcred()
        if not advapi.CredReadW(self.target, self._CRED_TYPE_GENERIC, 0, ctypes.byref(out)):
            code = ctypes.get_last_error()
            if code == self._ERROR_NOT_FOUND:
                return None
            raise CredentialStoreError(f"Lettura dal Gestore credenziali non riuscita (errore Windows {code}).")
        try:
            size = out.contents.CredentialBlobSize
            raw = ctypes.string_at(out.contents.CredentialBlob, size) if size else b""
        finally:
            advapi.CredFree(out)
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CredentialStoreError("Credenziale salvata illeggibile: rimuovila e salvala di nuovo.") from exc
        return value or None

    def write(self, secret: str) -> None:
        ctypes, advapi, cred_type, _pcred = self._load()
        blob = secret.encode("utf-8")
        if not blob or len(blob) > MAX_SECRET_BYTES:
            raise CredentialStoreError("Credenziale vuota o troppo lunga per il Gestore credenziali.")
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        credential = cred_type()
        credential.Type = self._CRED_TYPE_GENERIC
        credential.TargetName = self.target
        credential.Comment = "Token Hugging Face usato da Tessitura per scaricare i modelli"
        credential.CredentialBlobSize = len(blob)
        credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = self._CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "huggingface"
        try:
            if not advapi.CredWriteW(ctypes.byref(credential), 0):
                code = ctypes.get_last_error()
                raise CredentialStoreError(f"Salvataggio nel Gestore credenziali non riuscito (errore Windows {code}).")
        finally:
            ctypes.memset(buffer, 0, len(blob))

    def delete(self) -> bool:
        ctypes, advapi, _cred, _pcred = self._load()
        if advapi.CredDeleteW(self.target, self._CRED_TYPE_GENERIC, 0):
            return True
        code = ctypes.get_last_error()
        if code == self._ERROR_NOT_FOUND:
            return False
        raise CredentialStoreError(f"Rimozione dal Gestore credenziali non riuscita (errore Windows {code}).")


_default: CredentialStore | None = None


def default_store() -> CredentialStore:
    """Process-wide store. Tests replace it with ``set_default_store``."""
    global _default
    if _default is None:
        _default = WindowsCredentialStore() if sys.platform.startswith("win") else UnavailableCredentialStore()
    return _default


def set_default_store(store: CredentialStore | None) -> None:
    global _default
    _default = store
