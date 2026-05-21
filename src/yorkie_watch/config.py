from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class OpenClawConfig:
    """Runtime settings for OpenClaw notification delivery."""

    notify_mode: str
    url: str
    token: str
    event_endpoint: str
    ssh_host: str
    ssh_user: str
    ssh_port: int
    binary: str
    whatsapp_account: str
    whatsapp_target: str


@dataclass(frozen=True)
class DetectorConfig:
    """Runtime settings for object detection."""

    enabled: bool
    backend: str
    hef_path: str
    hailo_apps_root: str
    confidence_threshold: float
    target_classes: tuple[str, ...]
    timeout_seconds: float
    python_executable: str
    command_template: str


def load_environment(env_path: str | Path | None = None) -> None:
    """Load environment variables from `.env` without overriding existing values."""
    loaded = load_dotenv(dotenv_path=env_path, override=False)
    if not loaded:
        LOGGER.debug("No .env file was loaded; using existing environment variables only.")


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Read an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if value is None or not value.strip():
        if required:
            message = f"Missing required environment variable: {name}"
            LOGGER.error(message)
            raise ConfigError(message)
        return ""
    return value.strip()


def get_int_env(name: str, default: int, *, required: bool = False) -> int:
    """Read an integer environment variable with validation."""
    raw_value = get_env(name, str(default), required=required) or str(default)
    try:
        return int(raw_value)
    except ValueError as exc:
        message = f"Environment variable {name} must be an integer."
        LOGGER.error(message)
        raise ConfigError(message) from exc


def get_float_env(name: str, default: float, *, required: bool = False) -> float:
    """Read a float environment variable with validation."""
    raw_value = get_env(name, str(default), required=required) or str(default)
    try:
        return float(raw_value)
    except ValueError as exc:
        message = f"Environment variable {name} must be a number."
        LOGGER.error(message)
        raise ConfigError(message) from exc


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean-like environment variable."""
    raw_value = get_env(name)
    if not raw_value:
        return default
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    message = f"Environment variable {name} must be a boolean value."
    LOGGER.error(message)
    raise ConfigError(message)


def get_csv_env(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated list of values."""
    raw_value = get_env(name, default) or default
    values = tuple(value.strip().lower() for value in raw_value.split(",") if value.strip())
    return values or tuple(value.strip().lower() for value in default.split(",") if value.strip())


def load_openclaw_config() -> OpenClawConfig:
    """Load OpenClaw notification settings from `.env` / process environment."""
    load_environment()
    notify_mode = (get_env("OPENCLAW_NOTIFY_MODE", "http") or "http").lower()
    if notify_mode not in {"http", "ssh", "disabled"}:
        message = "OPENCLAW_NOTIFY_MODE must be one of: http, ssh, disabled"
        LOGGER.error(message)
        raise ValueError(message)

    event_endpoint = get_env("OPENCLAW_EVENT_ENDPOINT", "/api/events/yorkie-watch") or "/api/events/yorkie-watch"

    if notify_mode == "http":
        return OpenClawConfig(
            notify_mode=notify_mode,
            url=get_env("OPENCLAW_URL", required=True),
            token=get_env("OPENCLAW_TOKEN", required=True),
            event_endpoint=event_endpoint,
            ssh_host="",
            ssh_user="",
            ssh_port=22,
            binary="openclaw",
            whatsapp_account="business",
            whatsapp_target=get_env("OPENCLAW_WHATSAPP_TARGET", required=True),
        )

    if notify_mode == "ssh":
        return OpenClawConfig(
            notify_mode=notify_mode,
            url="",
            token="",
            event_endpoint=event_endpoint,
            ssh_host=get_env("OPENCLAW_SSH_HOST", required=True),
            ssh_user=get_env("OPENCLAW_SSH_USER", required=True),
            ssh_port=get_int_env("OPENCLAW_SSH_PORT", 22),
            binary=get_env("OPENCLAW_BINARY", "openclaw") or "openclaw",
            whatsapp_account=get_env("OPENCLAW_WHATSAPP_ACCOUNT", "business") or "business",
            whatsapp_target=get_env("OPENCLAW_WHATSAPP_TARGET", required=True),
        )

    return OpenClawConfig(
        notify_mode=notify_mode,
        url="",
        token="",
        event_endpoint=event_endpoint,
        ssh_host="",
        ssh_user="",
        ssh_port=22,
        binary="openclaw",
        whatsapp_account="business",
        whatsapp_target="",
    )


def load_detector_config() -> DetectorConfig:
    """Load object detector settings from `.env` / process environment."""
    load_environment()
    return DetectorConfig(
        enabled=get_bool_env("YORKIE_DETECTOR_ENABLED", False),
        backend=(get_env("YORKIE_DETECTOR_BACKEND", "hailo_apps") or "hailo_apps").lower(),
        hef_path=get_env("YORKIE_HAILO_HEF", "/usr/share/hailo-models/yolov8m_h10.hef")
        or "/usr/share/hailo-models/yolov8m_h10.hef",
        hailo_apps_root=get_env("YORKIE_HAILO_APPS_ROOT", "/home/pi/hailo-apps") or "/home/pi/hailo-apps",
        confidence_threshold=get_float_env("YORKIE_DOG_CONFIDENCE", 0.35),
        target_classes=get_csv_env("YORKIE_TARGET_CLASSES", "dog"),
        timeout_seconds=get_float_env("YORKIE_DETECTOR_TIMEOUT", 60.0),
        python_executable=get_env("YORKIE_HAILO_PYTHON", "python3") or "python3",
        command_template=get_env("YORKIE_HAILO_DETECT_COMMAND"),
    )
