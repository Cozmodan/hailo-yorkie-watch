from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from .config import ConfigError, StreamConfig

LOGGER = logging.getLogger(__name__)
DIRECT_STREAM_BACKEND = "opencv"
HOME_ASSISTANT_STREAM_BACKENDS = {"home_assistant", "ha_hls"}
SUPPORTED_STREAM_BACKENDS = {DIRECT_STREAM_BACKEND, *HOME_ASSISTANT_STREAM_BACKENDS}


class StreamSourceError(RuntimeError):
    """Raised when live stream frame capture cannot continue."""


class StreamFrameSource(Protocol):
    """Context-managed source that yields sampled image paths."""

    def __enter__(self) -> "StreamFrameSource": ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool: ...

    def __iter__(self) -> Iterator[Path]: ...


class OpenCVSubprocessFrameSource:
    """Use a Python/OpenCV helper process to sample frames from a live stream."""

    def __init__(self, config: StreamConfig, *, frame_limit: int = 0) -> None:
        self.config = config
        self.frame_limit = max(0, frame_limit)
        self.stream_url = resolve_stream_url(config)
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "OpenCVSubprocessFrameSource":
        argv = self._build_helper_argv()
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

    def _build_helper_argv(self) -> list[str]:
        helper_path = Path(__file__).resolve().parents[2] / "scripts" / "opencv_stream_frames.py"
        return [
            self.config.python_executable,
            str(helper_path),
            "--url",
            self.stream_url,
            "--output-dir",
            self.config.debug_dir,
            "--frame-interval",
            str(self.config.frame_interval_seconds),
        ]

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
                frame_path = event.get("frame_path") or event.get("path")
                if not frame_path:
                    raise StreamSourceError("Stream frame helper emitted a frame without a path.")
                yield Path(str(frame_path))
                continue
            if event_type == "error":
                error = str(event.get("error") or "Stream frame helper failed.")
                raise StreamSourceError(redact_stream_output(self.config, error, resolved_url=self.stream_url))
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
        return redact_stream_output(self.config, process.stderr.read(), resolved_url=self.stream_url)


class FFmpegSubprocessFrameSource(OpenCVSubprocessFrameSource):
    """Use an ffmpeg helper for Home Assistant authenticated camera streams."""

    def __init__(self, config: StreamConfig, *, frame_limit: int = 0) -> None:
        super().__init__(config, frame_limit=frame_limit)
        self.bearer_token = home_assistant_bearer_token(config)

    def _build_helper_argv(self) -> list[str]:
        helper_path = Path(__file__).resolve().parents[2] / "scripts" / "ffmpeg_stream_frames.py"
        argv = [
            self.config.python_executable,
            str(helper_path),
            "--url",
            self.stream_url,
            "--output-dir",
            self.config.debug_dir,
            "--frame-interval",
            str(self.config.frame_interval_seconds),
            "--backend",
            self.config.backend,
        ]
        if self.bearer_token:
            argv.extend(["--bearer-token", self.bearer_token])
        if self.frame_limit:
            argv.extend(["--frames", str(self.frame_limit)])
        return argv


def create_stream_source(config: StreamConfig, *, frame_limit: int = 0) -> StreamFrameSource:
    """Create the configured live stream frame source."""
    if not config.enabled:
        raise ConfigError("YORKIE_STREAM_ENABLED=1 is required for --watch-stream.")
    if config.backend not in SUPPORTED_STREAM_BACKENDS:
        raise ValueError(f"YORKIE_STREAM_BACKEND must be one of: {', '.join(sorted(SUPPORTED_STREAM_BACKENDS))}")
    if uses_home_assistant_stream(config):
        return FFmpegSubprocessFrameSource(config, frame_limit=frame_limit)
    return OpenCVSubprocessFrameSource(config, frame_limit=frame_limit)


def resolve_stream_url(config: StreamConfig) -> str:
    """Resolve the local runtime URL consumed by the frame helper."""
    if uses_home_assistant_stream(config):
        if config.ha_stream_url:
            return config.ha_stream_url
        if not config.ha_base_url:
            raise ConfigError("Home Assistant stream mode needs YORKIE_HA_BASE_URL when YORKIE_HA_STREAM_URL is empty.")
        if not config.ha_stream_entity:
            raise ConfigError(
                "Home Assistant stream mode needs YORKIE_HA_STREAM_ENTITY when YORKIE_HA_STREAM_URL is empty."
            )
        entity = quote(config.ha_stream_entity, safe="")
        return f"{config.ha_base_url.rstrip('/')}/api/camera_proxy_stream/{entity}"

    if config.url:
        return config.url
    raise ConfigError("YORKIE_STREAM_URL is required for --watch-stream.")


def uses_home_assistant_stream(config: StreamConfig) -> bool:
    """Return whether a stream config consumes a Home Assistant HLS URL."""
    return config.use_home_assistant or config.backend in HOME_ASSISTANT_STREAM_BACKENDS


def home_assistant_bearer_token(config: StreamConfig) -> str:
    """Validate Home Assistant stream auth and return the bearer token when used."""
    auth_mode = config.ha_stream_auth_mode or "bearer"
    if auth_mode not in {"bearer", "none"}:
        raise ValueError("YORKIE_HA_STREAM_AUTH_MODE must be one of: bearer, none")
    if auth_mode == "none":
        return ""
    if config.ha_long_lived_token:
        return config.ha_long_lived_token
    raise ConfigError(
        "Home Assistant bearer stream auth needs YORKIE_HA_LONG_LIVED_TOKEN. "
        "Set YORKIE_HA_STREAM_AUTH_MODE=none only for unauthenticated stream URLs."
    )


def redact_stream_output(config: StreamConfig, output: str, *, resolved_url: str = "") -> str:
    """Remove configured stream secrets from output intended for logs or errors."""
    redacted = output
    for sensitive_value in (resolved_url, config.url, config.ha_stream_url, config.ha_long_lived_token):
        redacted = _redact_value(redacted, sensitive_value)
    redacted = re.sub(
        r"(?i)(Authorization:\s*Bearer\s+)[^\r\n]+",
        r"\1<redacted-stream-value>",
        redacted,
    )
    return _redact_url_queries(redacted)


def _parse_event(line: str) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise StreamSourceError(f"Stream frame helper emitted invalid JSON: {_short_output(line)!r}") from exc
    if not isinstance(event, dict):
        raise StreamSourceError("Stream frame helper emitted a non-object JSON event.")
    return event


def _redact_value(output: str, sensitive_value: str) -> str:
    return output.replace(sensitive_value, "<redacted-stream-value>") if sensitive_value else output


def _redact_url_queries(output: str) -> str:
    return re.sub(
        r"((?:https?|rtsp)://[^\s'\"?]+\?)[^\s'\"]+",
        r"\1<redacted-query>",
        output,
        flags=re.IGNORECASE,
    )


def _short_output(output: str | bytes | None, *, max_chars: int = 800) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output.strip()
    if len(output) <= max_chars:
        return output
    return f"{output[:max_chars]}..."
