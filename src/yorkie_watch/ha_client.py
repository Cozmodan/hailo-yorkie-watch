from __future__ import annotations

import logging
import time
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

    def _camera_proxy_url_with_cache_buster(self) -> str:
        """Build a camera proxy URL with a timestamp query parameter."""
        unix_ms = int(time.time() * 1000)
        return f"{self.camera_proxy_url}?ts={unix_ms}"

    def _fetch_snapshot_once(self) -> bytes:
        """Fetch one camera image from Home Assistant without retrying."""
        snapshot_url = self._camera_proxy_url_with_cache_buster()
        request = Request(
            snapshot_url,
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
            if exc.code == 500:
                message = f"{message}. The camera backend may be temporarily unavailable."
            raise HomeAssistantError(message) from exc
        except URLError as exc:
            message = f"Could not reach Home Assistant at {self.base_url}: {exc.reason}"
            raise HomeAssistantError(message) from exc
        except TimeoutError as exc:
            message = f"Timed out fetching Home Assistant snapshot from {self.base_url}"
            raise HomeAssistantError(message) from exc

        if not image_bytes:
            message = "Home Assistant returned an empty snapshot response."
            raise HomeAssistantError(message)

        LOGGER.info("Fetched Home Assistant snapshot from %s (%d bytes).", snapshot_url, len(image_bytes))
        return image_bytes

    def fetch_snapshot(self, *, attempts: int = 3, delay_seconds: float = 2.0) -> bytes:
        """Fetch one camera image from Home Assistant and return the raw bytes, retrying transient failures."""
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must not be negative")

        for attempt in range(1, attempts + 1):
            try:
                return self._fetch_snapshot_once()
            except HomeAssistantError as exc:
                LOGGER.warning("Home Assistant snapshot attempt %d/%d failed: %s", attempt, attempts, exc)
                if attempt == attempts:
                    LOGGER.error("Home Assistant snapshot failed after %d attempt(s).", attempts)
                    raise
                time.sleep(delay_seconds)

        raise HomeAssistantError("Home Assistant snapshot failed without a captured error.")

    def save_snapshot(self, path: str | Path, *, attempts: int = 3, delay_seconds: float = 2.0) -> Path:
        """Fetch one camera image and save it to `path`."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = self.fetch_snapshot(attempts=attempts, delay_seconds=delay_seconds)
        output_path.write_bytes(image_bytes)
        LOGGER.info("Saved Home Assistant snapshot to %s (%d bytes).", output_path, len(image_bytes))
        return output_path
