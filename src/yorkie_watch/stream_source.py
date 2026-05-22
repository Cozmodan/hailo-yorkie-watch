from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from .config import ConfigError, StreamConfig

LOGGER = logging.getLogger(__name__)
SUPPORTED_STREAM_BACKENDS = {"opencv"}


class StreamSourceError(RuntimeError):
    """Raised when live stream frame capture cannot continue."""


class StreamFrameSource(Protocol):
    """Context-managed source that yields sampled image paths."""

    def __enter__(self) -> "StreamFrameSource": ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool: ...

    def __iter__(self) -> Iterator[Path]: ...


class OpenCVSubprocessFrameSource:
    """Use a Python/OpenCV helper process to sample frames from a live stream."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "OpenCVSubprocessFrameSource":
        helper_path = Path(__file__).resolve().parents[2] / "scripts" / "opencv_stream_frames.py"
        argv = [
            self.config.python_executable,
            str(helper_path),
            "--url",
            self.config.url,
            "--output-dir",
            self.config.debug_dir,
            "--frame-interval",
            str(self.config.frame_interval_seconds),
        ]
        try:
            self.process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise StreamSourceError(f"Stream Python executable was not found: {argv[0]}") from exc
        except OSError as exc:
            raise StreamSourceError(f"Could not start stream frame helper: {exc}") from exc
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[Path]:
        process = self.process
        if process is None or process.stdout is None:
            raise StreamSourceError("Stream frame helper has not been started.")

        for line in process.stdout:
            event = _parse_event(line)
            event_type = event.get("type")
            if event_type == "connected":
                LOGGER.info("Stream connected.")
                continue
            if event_type == "frame":
                frame_path = event.get("path")
                if not frame_path:
                    raise StreamSourceError("Stream frame helper emitted a frame without a path.")
                yield Path(str(frame_path))
                continue
            if event_type == "error":
                raise StreamSourceError(str(event.get("error") or "Stream frame helper failed."))
            LOGGER.debug("Ignoring unknown stream helper event type %r.", event_type)

        returncode = process.wait()
        stderr = self._stderr()
        if returncode != 0:
            raise StreamSourceError(
                f"Stream frame helper exited with returncode={returncode}: {_short_output(stderr)!r}"
            )
        raise StreamSourceError("Stream frame helper stopped before the watch loop ended.")

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None

    def _stderr(self) -> str:
        process = self.process
        if process is None or process.stderr is None:
            return ""
        return _redact_value(process.stderr.read(), self.config.url)


def create_stream_source(config: StreamConfig) -> StreamFrameSource:
    """Create the configured live stream frame source."""
    if not config.enabled:
        raise ConfigError("YORKIE_STREAM_ENABLED=1 is required for --watch-stream.")
    if not config.url:
        raise ConfigError("YORKIE_STREAM_URL is required for --watch-stream.")
    if config.backend not in SUPPORTED_STREAM_BACKENDS:
        raise ValueError(f"YORKIE_STREAM_BACKEND must be one of: {', '.join(sorted(SUPPORTED_STREAM_BACKENDS))}")
    return OpenCVSubprocessFrameSource(config)


def _parse_event(line: str) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise StreamSourceError(f"Stream frame helper emitted invalid JSON: {_short_output(line)!r}") from exc
    if not isinstance(event, dict):
        raise StreamSourceError("Stream frame helper emitted a non-object JSON event.")
    return event


def _redact_value(output: str, sensitive_value: str) -> str:
    return output.replace(sensitive_value, "<redacted-stream-url>") if sensitive_value else output


def _short_output(output: str | bytes | None, *, max_chars: int = 800) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output.strip()
    if len(output) <= max_chars:
        return output
    return f"{output[:max_chars]}..."
