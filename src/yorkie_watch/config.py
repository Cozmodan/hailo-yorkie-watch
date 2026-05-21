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
