from __future__ import annotations

import json
import logging
import shlex
import subprocess
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ConfigError, load_openclaw_config

LOGGER = logging.getLogger(__name__)
DEFAULT_EVENT_ENDPOINT = "/api/events/yorkie-watch"
SUPPORTED_NOTIFY_MODES = {"http", "ssh", "disabled"}


class OpenClawClient:
    """Small client for sending Yorkie Watch events to OpenClaw."""

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        whatsapp_target: str = "",
        *,
        notify_mode: str = "http",
        event_endpoint: str = DEFAULT_EVENT_ENDPOINT,
        ssh_host: str = "",
        ssh_user: str = "",
        ssh_port: int = 22,
        binary: str = "openclaw",
        whatsapp_account: str = "business",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.notify_mode = notify_mode.lower()
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.whatsapp_target = whatsapp_target
        self.event_endpoint = event_endpoint if event_endpoint.startswith("/") else f"/{event_endpoint}"
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.binary = binary or "openclaw"
        self.whatsapp_account = whatsapp_account or "business"
        self.timeout_seconds = timeout_seconds
        self._validate_config()

    @classmethod
    def from_env(cls) -> "OpenClawClient":
        """Create a client from `.env` / process environment settings."""
        config = load_openclaw_config()
        return cls(
            base_url=config.url,
            token=config.token,
            whatsapp_target=config.whatsapp_target,
            notify_mode=config.notify_mode,
            event_endpoint=config.event_endpoint,
            ssh_host=config.ssh_host,
            ssh_user=config.ssh_user,
            ssh_port=config.ssh_port,
            binary=config.binary,
            whatsapp_account=config.whatsapp_account,
        )

    @property
    def event_url(self) -> str:
        """Build the configured OpenClaw event endpoint URL."""
        return f"{self.base_url}{self.event_endpoint}"

    def _validate_config(self) -> None:
        if self.notify_mode not in SUPPORTED_NOTIFY_MODES:
            message = "OPENCLAW_NOTIFY_MODE must be one of: http, ssh, disabled"
            LOGGER.error(message)
            raise ValueError(message)

        if self.notify_mode == "disabled":
            return

        if self.notify_mode == "http":
            self._require_config("OPENCLAW_URL", self.base_url)
            self._require_config("OPENCLAW_TOKEN", self.token)
            self._require_config("OPENCLAW_WHATSAPP_TARGET", self.whatsapp_target)
            return

        self._require_config("OPENCLAW_SSH_HOST", self.ssh_host)
        self._require_config("OPENCLAW_SSH_USER", self.ssh_user)
        self._require_config("OPENCLAW_WHATSAPP_TARGET", self.whatsapp_target)
        if not 1 <= self.ssh_port <= 65535:
            message = "OPENCLAW_SSH_PORT must be between 1 and 65535."
            LOGGER.error(message)
            raise ConfigError(message)

    def _require_config(self, name: str, value: str) -> None:
        if value.strip():
            return
        message = f"Missing required environment variable for OPENCLAW_NOTIFY_MODE={self.notify_mode}: {name}"
        LOGGER.error(message)
        raise ConfigError(message)

    def send_message(self, message: str, *, event_type: str = "yorkie_watch_test", confidence: float = 0.0) -> bool:
        """Send one WhatsApp message through the configured OpenClaw notification path."""
        return self.send_event(
            {
                "event_type": event_type,
                "message": message,
                "confidence": confidence,
            }
        )

    def send_event(self, event: Mapping[str, Any]) -> bool:
        """Send one event through the configured OpenClaw notification path."""
        if self.notify_mode == "disabled":
            LOGGER.info("OpenClaw notifications are disabled; skipping event.")
            return True
        if self.notify_mode == "ssh":
            return self._send_ssh_message(self._event_message(event))
        return self._send_http_event(event)

    def _send_http_event(self, event: Mapping[str, Any]) -> bool:
        """Send one JSON event to OpenClaw over HTTP and return whether it was accepted."""
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

    def _event_message(self, event: Mapping[str, Any]) -> str:
        message = event.get("message")
        if message is not None:
            return str(message)
        return json.dumps(dict(event), sort_keys=True)

    def _build_ssh_argv(self, message: str) -> list[str]:
        remote_args = [
            self.binary,
            "message",
            "send",
            "--channel",
            "whatsapp",
            "--account",
            self.whatsapp_account,
            "--target",
            self.whatsapp_target,
            "--message",
            message,
        ]
        remote_command = " ".join(shlex.quote(arg) for arg in remote_args)
        return [
            "ssh",
            "-p",
            str(self.ssh_port),
            f"{self.ssh_user}@{self.ssh_host}",
            remote_command,
        ]

    def _send_ssh_message(self, message: str) -> bool:
        """Send one WhatsApp message by invoking OpenClaw over SSH."""
        argv = self._build_ssh_argv(message)
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            LOGGER.error("ssh executable was not found. Install an SSH client or use OPENCLAW_NOTIFY_MODE=http.")
            return False
        except subprocess.TimeoutExpired:
            LOGGER.error("Timed out sending OpenClaw SSH notification.")
            return False
        except OSError as exc:
            LOGGER.error("Failed to start OpenClaw SSH notification: %s", exc)
            return False

        if completed.returncode == 0:
            LOGGER.info("Sent OpenClaw SSH WhatsApp notification.")
            return True

        stderr = completed.stderr.strip()
        if stderr:
            LOGGER.error("OpenClaw SSH notification failed with exit code %s: %s", completed.returncode, stderr)
        else:
            LOGGER.error("OpenClaw SSH notification failed with exit code %s.", completed.returncode)
        return False
