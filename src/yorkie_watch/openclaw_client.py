from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
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
        ssh_media_remote_dir: str = "",
        ssh_media_command_template: str = "",
        timeout_seconds: float = 60.0,
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
        self.ssh_media_remote_dir = ssh_media_remote_dir.rstrip("/")
        self.ssh_media_command_template = ssh_media_command_template
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
            ssh_media_remote_dir=config.ssh_media_remote_dir,
            ssh_media_command_template=config.ssh_media_command_template,
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

    def send_message(
        self,
        message: str,
        *,
        event_type: str = "yorkie_watch_test",
        confidence: float = 0.0,
        attachment_path: str | Path | None = None,
    ) -> bool:
        """Send one WhatsApp message through the configured OpenClaw notification path."""
        event: dict[str, Any] = {
            "event_type": event_type,
            "message": message,
            "confidence": confidence,
        }
        if attachment_path is not None:
            event["attachment_path"] = str(attachment_path)
        return self.send_event(event)

    def send_event(self, event: Mapping[str, Any]) -> bool:
        """Send one event through the configured OpenClaw notification path."""
        if self.notify_mode == "disabled":
            LOGGER.info("OpenClaw notifications are disabled; skipping event.")
            return True
        if self.notify_mode == "ssh":
            return self._send_ssh_message(self._event_message(event), attachment_path=self._event_attachment_path(event))
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

    def _event_attachment_path(self, event: Mapping[str, Any]) -> Path | None:
        attachment_path = event.get("attachment_path")
        if not attachment_path:
            return None
        return Path(str(attachment_path))

    def _ssh_base_argv(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(self.ssh_port),
            f"{self.ssh_user}@{self.ssh_host}",
        ]

    def _scp_base_argv(self) -> list[str]:
        return [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-P",
            str(self.ssh_port),
        ]

    def _build_ssh_argv(self, message: str) -> list[str]:
        remote_argv = [
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
        remote_command = shlex.join(remote_argv)
        return [*self._ssh_base_argv(), remote_command]

    def _build_openclaw_help_argv(self, openclaw_args: list[str]) -> list[str]:
        remote_command = shlex.join([self.binary, *openclaw_args])
        return [*self._ssh_base_argv(), remote_command]

    def _remote_media_path_for(self, attachment_path: Path) -> str:
        remote_dir = self.ssh_media_remote_dir or "/tmp/yorkie-watch"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", attachment_path.name).strip("._")
        if not safe_name:
            safe_name = "snapshot.jpg"
        timestamp = int(time.time())
        return f"{remote_dir.rstrip('/')}/{timestamp}-{safe_name}"

    def _build_scp_argv(self, local_path: Path, remote_path: str) -> list[str]:
        return [
            *self._scp_base_argv(),
            str(local_path),
            f"{self.ssh_user}@{self.ssh_host}:{remote_path}",
        ]

    def _build_remote_mkdir_argv(self) -> list[str]:
        remote_dir = self.ssh_media_remote_dir or "/tmp/yorkie-watch"
        remote_command = shlex.join(["mkdir", "-p", remote_dir])
        return [*self._ssh_base_argv(), remote_command]

    def _build_remote_media_command(self, *, message: str, remote_media_path: str) -> str:
        if not self.ssh_media_command_template:
            raise ConfigError("OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE is required for SSH media attachments.")
        quoted_values = {
            "binary": shlex.quote(self.binary),
            "channel": "whatsapp",
            "account": shlex.quote(self.whatsapp_account),
            "target": shlex.quote(self.whatsapp_target),
            "message": shlex.quote(message),
            "media_path": shlex.quote(remote_media_path),
        }
        return self.ssh_media_command_template.format(**quoted_values)

    def _redact_ssh_output(self, output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        redacted = output.strip()
        if self.whatsapp_target:
            redacted = redacted.replace(self.whatsapp_target, "<redacted-target>")
            target_digits = re.sub(r"\D", "", self.whatsapp_target)
            if len(target_digits) >= 4:
                formatted_target_pattern = r"[\s().+-]*".join(re.escape(digit) for digit in target_digits)
                redacted = re.sub(formatted_target_pattern, "<redacted-target>", redacted)
        if self.ssh_host:
            redacted = redacted.replace(self.ssh_host, "<redacted-host>")
        if self.ssh_user:
            redacted = redacted.replace(self.ssh_user, "<redacted-user>")
        return redacted

    def _run_ssh_subprocess(self, argv: list[str], *, action: str) -> bool:
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            LOGGER.error("%s executable was not found.", argv[0])
            return False
        except subprocess.TimeoutExpired as exc:
            stdout = self._redact_ssh_output(exc.stdout)
            stderr = self._redact_ssh_output(exc.stderr)
            LOGGER.error("%s timed out: returncode=timeout stdout=%r stderr=%r", action, stdout, stderr)
            return False
        except OSError as exc:
            LOGGER.error("%s could not start: %s", action, exc)
            return False

        if completed.returncode == 0:
            return True

        stdout = self._redact_ssh_output(completed.stdout)
        stderr = self._redact_ssh_output(completed.stderr)
        LOGGER.error(
            "%s failed: returncode=%s stdout=%r stderr=%r",
            action,
            completed.returncode,
            stdout,
            stderr,
        )
        return False

    def _send_ssh_message(self, message: str, *, attachment_path: Path | None = None) -> bool:
        """Send one WhatsApp message by invoking OpenClaw over SSH."""
        if attachment_path is not None:
            return self._send_ssh_message_with_attachment(message, attachment_path)

        argv = self._build_ssh_argv(message)
        if self._run_ssh_subprocess(argv, action="OpenClaw SSH notification"):
            LOGGER.info("Sent OpenClaw SSH WhatsApp notification.")
            return True
        return False

    def _send_ssh_message_with_attachment(self, message: str, attachment_path: Path) -> bool:
        if not self.ssh_media_command_template:
            LOGGER.warning(
                "Snapshot attachment requested, but OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE is not set; sending text only."
            )
            return self._send_ssh_message(message)

        if not attachment_path.exists():
            LOGGER.error("Snapshot attachment does not exist: %s", attachment_path)
            return False

        remote_media_path = self._remote_media_path_for(attachment_path)
        if not self._run_ssh_subprocess(self._build_remote_mkdir_argv(), action="OpenClaw SSH media directory setup"):
            return False
        if not self._run_ssh_subprocess(
            self._build_scp_argv(attachment_path, remote_media_path),
            action="OpenClaw SSH media copy",
        ):
            return False

        remote_command = self._build_remote_media_command(message=message, remote_media_path=remote_media_path)
        argv = [*self._ssh_base_argv(), remote_command]
        if self._run_ssh_subprocess(argv, action="OpenClaw SSH media notification"):
            LOGGER.info("Sent OpenClaw SSH WhatsApp notification with media attachment.")
            return True
        return False

    def run_openclaw_help(self, openclaw_args: list[str]) -> tuple[int, str, str]:
        """Run one OpenClaw help command over SSH and return redacted output."""
        argv = self._build_openclaw_help_argv(openclaw_args)
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            return 127, "", "ssh executable was not found"
        except subprocess.TimeoutExpired as exc:
            stdout = self._redact_ssh_output(exc.stdout)
            stderr = self._redact_ssh_output(exc.stderr)
            return 124, stdout, f"timed out: {stderr}"
        except OSError as exc:
            return 1, "", f"could not start ssh: {exc}"
        return (
            completed.returncode,
            self._redact_ssh_output(completed.stdout),
            self._redact_ssh_output(completed.stderr),
        )
