"""Cooperative ownership of inference memory across workers and processes."""
from threading import Event


class CancelledError(RuntimeError):
    pass


def check_cancel(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise CancelledError("Operazione annullata")


def acquire_inference(cancel_event=None, on_wait=None):
    from filelock import FileLock, Timeout
    from .model_store import hub_root
    path = hub_root() / ".tessitura" / "inference.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    informed = False
    while True:
        check_cancel(cancel_event)
        try:
            lock.acquire(timeout=0.1)
            return lock
        except Timeout:
            if on_wait and not informed:
                on_wait("In coda: un altro motore sta usando le risorse di inferenza")
                informed = True
