from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_env, load_environment

LOGGER = logging.getLogger(__name__)
DEFAULT_EVENT_ENDPOINT = "/api/events/yorkie-watch"


class OpenClawClient:
    """Small client for sending Yorkie Watch events to OpenClaw."""

    def __init__(
        self,
        base_url: str,
        token: str,
        whatsapp_target: str,
        *,
        event_endpoint: str = DEFAULT_EVENT_ENDPOINT,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.whatsapp_target = whatsapp_target
        self.event_endpoint = event_endpoint if event_endpoint.startswith("/") else f"/{event_endpoint}"
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenClawClient":
        """Create a client from `.env` / process environment settings."""
        load_environment()
        return cls(
            base_url=get_env("OPENCLAW_URL", required=True),
            token=get_env("OPENCLAW_TOKEN", required=True),
            whatsapp_target=get_env("OPENCLAW_WHATSAPP_TARGET", required=True),
            event_endpoint=get_env("OPENCLAW_EVENT_ENDPOINT", DEFAULT_EVENT_ENDPOINT),
        )

    @property
    def event_url(self) -> str:
        """Build the configured OpenClaw event endpoint URL."""
        return f"{self.base_url}{self.event_endpoint}"

    def send_event(self, event: Mapping[str, Any]) -> bool:
        """Send one JSON event to OpenClaw and return whether it was accepted."""
        payload = dict(event)
        payload.setdefault("whatsapp_target", self.whatsapp_target)
        body = json.dumps(payload).encode("utf-8")

        request = Request(
            self.event_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                response.read()
        except HTTPError as exc:
            LOGGER.error("OpenClaw event request failed with HTTP %s: %s", exc.code, exc.reason)
            return False
        except URLError as exc:
            LOGGER.error("Could not reach OpenClaw at %s: %s", self.base_url, exc.reason)
            return False
        except TimeoutError:
            LOGGER.error("Timed out sending event to OpenClaw at %s", self.base_url)
            return False

        if 200 <= status < 300:
            LOGGER.info("Sent OpenClaw event to %s with status %s.", self.event_url, status)
            return True

        LOGGER.error("OpenClaw event request returned unexpected status %s.", status)
        return False
