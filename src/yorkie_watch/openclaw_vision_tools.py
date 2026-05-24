from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .config import YorkieVisionConfig, load_yorkie_vision_config

LOGGER = logging.getLogger(__name__)
VisionRoute = Literal["camera_snapshot", "latest_alert"]
CAMERA_SNAPSHOT_PATH = "/vision/camera-snapshot"
LATEST_ALERT_PATH = "/vision/latest-alert"
SECRET_HEADER = "X-OpenClaw-Secret"
DEFAULT_CAMERA_PROMPT = "Describe the camera view briefly. Is a dog or Yorkie visible?"
DEFAULT_LATEST_ALERT_PROMPT = "Was the last Yorkie Watch alert a real dog or Yorkie? Mention uncertainty."


@dataclass(frozen=True)
class VisionToolResult:
    """Normalized response returned to the OpenClaw LLM/agent."""

    ok: bool
    source: str
    description: str
    error: str
    image_path: str
    status_code: int
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, status_code: int) -> "VisionToolResult":
        return cls(
            ok=bool(payload.get("ok", False)),
            source=str(payload.get("source", "")),
            description=str(payload.get("description", "")).strip(),
            error=str(payload.get("error", "")).strip(),
            image_path=str(payload.get("image_path", "")).strip(),
            status_code=status_code,
            payload=dict(payload),
        )


@dataclass(frozen=True)
class VisionInteraction:
    """One routed WhatsApp question and its resulting reply text."""

    route: VisionRoute | None
    result: VisionToolResult | None
    reply_text: str
    sent: bool


class OpenClawVisionToolClient:
    """OpenClaw-side client for the Raspberry Pi Yorkie Watch vision tool."""

    def __init__(
        self,
        config: YorkieVisionConfig,
        *,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self.config = config
        self.opener = opener

    @classmethod
    def from_env(cls) -> "OpenClawVisionToolClient":
        return cls(load_yorkie_vision_config())

    def camera_snapshot(self, prompt: str = DEFAULT_CAMERA_PROMPT) -> VisionToolResult:
        return self._post(CAMERA_SNAPSHOT_PATH, prompt=prompt, route="camera_snapshot")

    def latest_alert(self, prompt: str = DEFAULT_LATEST_ALERT_PROMPT) -> VisionToolResult:
        return self._post(LATEST_ALERT_PATH, prompt=prompt, route="latest_alert")

    def _post(self, path: str, *, prompt: str, route: VisionRoute) -> VisionToolResult:
        body = json.dumps({"prompt": prompt}).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.shared_secret:
            headers[SECRET_HEADER] = self.config.shared_secret

        request = Request(url, data=body, headers=headers, method="POST")
        status_code = 0
        try:
            with self.opener(request, timeout=self.config.timeout_seconds) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                status_code = int(status)
                raw_body = response.read()
        except HTTPError as exc:
            status_code = exc.code
            raw_body = exc.read()
            LOGGER.warning("Yorkie vision route selected: %s; HTTP status from Pi: %s", route, status_code)
            return self._result_from_body(raw_body, status_code=status_code, route=route)
        except TimeoutError as exc:
            message = f"Timed out waiting for Yorkie vision tool after {self.config.timeout_seconds:g}s"
            safe_message = redact_vision_text(message, self.config)
            LOGGER.warning("%s", safe_message)
            return _failure_result(route, safe_message, status_code=0)
        except URLError as exc:
            message = f"Could not reach Yorkie vision tool: {exc.reason}"
            safe_message = redact_vision_text(message, self.config)
            LOGGER.warning("%s", safe_message)
            return _failure_result(route, safe_message, status_code=0)
        except OSError as exc:
            message = f"Yorkie vision tool request failed: {exc}"
            safe_message = redact_vision_text(message, self.config)
            LOGGER.warning("%s", safe_message)
            return _failure_result(route, safe_message, status_code=0)

        LOGGER.info("Yorkie vision route selected: %s; HTTP status from Pi: %s", route, status_code)
        return self._result_from_body(raw_body, status_code=status_code, route=route)

    def _result_from_body(self, raw_body: bytes, *, status_code: int, route: VisionRoute) -> VisionToolResult:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _failure_result(route, "Yorkie vision tool returned a non-JSON response.", status_code=status_code)
        if not isinstance(payload, dict):
            return _failure_result(route, "Yorkie vision tool returned a JSON value that was not an object.", status_code=status_code)
        result = VisionToolResult.from_payload(payload, status_code=status_code)
        if 200 <= status_code < 300:
            return result
        if result.error:
            return result
        return _failure_result(route, f"Yorkie vision tool returned HTTP {status_code}.", status_code=status_code)


def select_vision_route(message: str) -> VisionRoute | None:
    """Select the Yorkie vision tool route implied by an inbound WhatsApp message."""
    normalized = _normalize_text(message)
    if not normalized:
        return None

    latest_phrases = (
        "last alert",
        "latest alert",
        "previous alert",
        "alert real",
        "was the alert real",
        "false trigger",
        "false alarm",
        "was that a dog",
        "was that real",
        "previous dog detection",
        "last dog detection",
        "last detection",
    )
    if any(phrase in normalized for phrase in latest_phrases):
        return "latest_alert"

    camera_phrases = (
        "what do you see",
        "what can you see",
        "check the camera",
        "check camera",
        "camera view",
        "current camera",
        "live image",
        "live view",
        "yorkie there",
        "dog there",
        "dog visible",
        "yorkie visible",
        "is the yorkie there",
        "is a yorkie there",
        "is the dog there",
        "is a dog there",
    )
    if any(phrase in normalized for phrase in camera_phrases):
        return "camera_snapshot"

    if ("camera" in normalized or "see" in normalized) and any(
        word in normalized for word in ("dog", "yorkie", "view", "there", "visible")
    ):
        return "camera_snapshot"
    return None


def handle_whatsapp_vision_message(
    message: str,
    *,
    client: OpenClawVisionToolClient,
    send_reply: Callable[[str], None] | None = None,
    debug: bool = False,
) -> VisionInteraction:
    """Route one inbound WhatsApp text to a Yorkie vision tool and optionally send the reply."""
    LOGGER.info("Incoming WhatsApp text received: %s", redact_vision_text(message.strip(), client.config))
    route = select_vision_route(message)
    if route is None:
        return VisionInteraction(
            route=None,
            result=None,
            reply_text="I did not find a Yorkie Watch vision request in that message.",
            sent=False,
        )

    LOGGER.info("Vision route selected: %s", route)
    if route == "latest_alert":
        result = client.latest_alert(prompt=message.strip() or DEFAULT_LATEST_ALERT_PROMPT)
    else:
        result = client.camera_snapshot(prompt=message.strip() or DEFAULT_CAMERA_PROMPT)

    reply = format_vision_reply(result, debug=debug)
    sent = False
    if send_reply is not None:
        send_reply(reply)
        sent = True
        LOGGER.info("Reply sent to WhatsApp.")
    return VisionInteraction(route=route, result=result, reply_text=reply, sent=sent)


def format_vision_reply(result: VisionToolResult, *, debug: bool = False) -> str:
    """Format a Yorkie vision response for WhatsApp without raw JSON unless debug is enabled."""
    if result.ok and result.description:
        reply = result.description
    elif result.ok:
        reply = "Yorkie Watch returned successfully, but did not include a description."
    else:
        error = result.error or "Yorkie Watch could not complete the vision request."
        reply = f"I could not get a clear vision answer from Yorkie Watch: {error}"

    if debug:
        reply = f"{reply}\n\nDebug: {json.dumps(result.payload, sort_keys=True)}"
    return reply


def redact_vision_text(text: str, config: YorkieVisionConfig) -> str:
    """Redact shared secret and private-ish vision URL details from logs/errors."""
    redacted = text
    if config.shared_secret:
        redacted = redacted.replace(config.shared_secret, "<redacted-vision-secret>")
    if config.base_url:
        redacted = redacted.replace(config.base_url, _redacted_base_url(config.base_url))
    redacted = re.sub(r"https?://[^\s'\"<>]+", _redact_url_match, redacted)
    return redacted


def _failure_result(route: VisionRoute, error: str, *, status_code: int) -> VisionToolResult:
    return VisionToolResult(
        ok=False,
        source=route,
        description="",
        error=error,
        image_path="",
        status_code=status_code,
        payload={"ok": False, "source": route, "error": error},
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _redacted_base_url(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return "<redacted-yorkie-vision-url>"
    return urlunsplit((parsed.scheme, "<redacted-yorkie-vision-host>", "", "", ""))


def _redact_url_match(match: re.Match[str]) -> str:
    return _redacted_base_url(match.group(0))
