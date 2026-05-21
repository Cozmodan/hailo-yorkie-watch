from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


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
