from __future__ import annotations

import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import get_env, load_environment

LOGGER = logging.getLogger(__name__)


class HomeAssistantError(RuntimeError):
    """Raised when a Home Assistant snapshot cannot be fetched or saved."""


class HomeAssistantClient:
    """Small client for Home Assistant camera snapshot plumbing."""

    def __init__(self, base_url: str, token: str, camera_entity: str, *, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.camera_entity = camera_entity
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "HomeAssistantClient":
        """Create a client from `.env` / process environment settings."""
        load_environment()
        return cls(
            base_url=get_env("HOME_ASSISTANT_URL", required=True),
            token=get_env("HOME_ASSISTANT_TOKEN", required=True),
            camera_entity=get_env("HOME_ASSISTANT_CAMERA_ENTITY", required=True),
        )

    @property
    def camera_proxy_url(self) -> str:
        """Build the Home Assistant camera proxy URL for the configured entity."""
        entity = quote(self.camera_entity, safe="")
        return f"{self.base_url}/api/camera_proxy/{entity}"

    def fetch_snapshot(self) -> bytes:
        """Fetch one camera image from Home Assistant and return the raw bytes."""
        request = Request(
            self.camera_proxy_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "image/*",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                image_bytes = response.read()
        except HTTPError as exc:
            message = f"Home Assistant snapshot request failed with HTTP {exc.code}: {exc.reason}"
            LOGGER.error(message)
            raise HomeAssistantError(message) from exc
        except URLError as exc:
            message = f"Could not reach Home Assistant at {self.base_url}: {exc.reason}"
            LOGGER.error(message)
            raise HomeAssistantError(message) from exc
        except TimeoutError as exc:
            message = f"Timed out fetching Home Assistant snapshot from {self.base_url}"
            LOGGER.error(message)
            raise HomeAssistantError(message) from exc

        if not image_bytes:
            message = "Home Assistant returned an empty snapshot response."
            LOGGER.error(message)
            raise HomeAssistantError(message)

        LOGGER.info("Fetched Home Assistant snapshot from %s (%d bytes).", self.camera_proxy_url, len(image_bytes))
        return image_bytes

    def save_snapshot(self, path: str | Path) -> Path:
        """Fetch one camera image and save it to `path`."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = self.fetch_snapshot()
        output_path.write_bytes(image_bytes)
        LOGGER.info("Saved Home Assistant snapshot to %s (%d bytes).", output_path, len(image_bytes))
        return output_path
