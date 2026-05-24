from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from types import TracebackType

try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - POSIX path.
    msvcrt = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
DEFAULT_LOCK_PATH = Path("/tmp/yorkie_hailo_device.lock")
DEFAULT_TIMEOUT_SECONDS = 120.0


class HailoDeviceLockError(RuntimeError):
    """Raised when the shared Hailo device lock cannot be acquired."""


class HailoDeviceLock:
    """Cross-process lock for the single Hailo device.

    Linux uses fcntl.flock on /tmp/yorkie_hailo_device.lock. A small Windows
    fallback keeps unit tests portable, but the production Pi path is fcntl.
    """

    def __init__(
        self,
        *,
        lock_path: str | Path | None = None,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.1,
        clock: object = time.monotonic,
        sleep: object = time.sleep,
    ) -> None:
        self.lock_path = Path(lock_path or os.getenv("HAILO_DEVICE_LOCK_PATH") or default_lock_path())
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else _float_env("HAILO_DEVICE_LOCK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.poll_interval_seconds = max(0.01, poll_interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._file: object | None = None
        self.acquired = False

    def __enter__(self) -> "HailoDeviceLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @classmethod
    def from_env(cls) -> "HailoDeviceLock":
        return cls()

    def acquire(self) -> None:
        if self.acquired:
            return

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+b")
        if msvcrt is not None and fcntl is None:
            try:
                if self.lock_path.stat().st_size == 0:
                    lock_file.seek(0)
                    lock_file.write(b"0")
                    lock_file.flush()
            except OSError:
                lock_file.close()
                raise

        start = self._clock()  # type: ignore[operator]
        while True:
            try:
                self._try_lock(lock_file)
            except BlockingIOError:
                if self.timeout_seconds >= 0 and self._clock() - start >= self.timeout_seconds:  # type: ignore[operator]
                    lock_file.close()
                    raise HailoDeviceLockError(
                        f"Timed out waiting for Hailo device lock after {self.timeout_seconds:g}s."
                    )
                self._sleep(self.poll_interval_seconds)  # type: ignore[operator]
                continue

            self._file = lock_file
            self.acquired = True
            LOGGER.debug("Acquired Hailo device lock: %s", self.lock_path)
            return

    def release(self) -> None:
        lock_file = self._file
        if lock_file is None:
            return
        try:
            self._unlock(lock_file)
        finally:
            lock_file.close()  # type: ignore[attr-defined]
            self._file = None
            self.acquired = False
            LOGGER.debug("Released Hailo device lock: %s", self.lock_path)

    def _try_lock(self, lock_file: object) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined, union-attr]
            return
        if msvcrt is not None:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined, union-attr]
            except OSError as exc:
                raise BlockingIOError(str(exc)) from exc
            return
        raise HailoDeviceLockError("No supported file locking module is available.")

    def _unlock(self, lock_file: object) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined, union-attr]
            return
        if msvcrt is not None:
            try:
                lock_file.seek(0)  # type: ignore[attr-defined]
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined, union-attr]
            except OSError:
                LOGGER.debug("Windows Hailo lock was already released.")


def default_lock_path() -> Path:
    if os.name == "nt" and str(DEFAULT_LOCK_PATH).startswith("\\"):
        return Path(tempfile.gettempdir()) / DEFAULT_LOCK_PATH.name
    return DEFAULT_LOCK_PATH


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise HailoDeviceLockError(f"{name} must be a number.") from exc
