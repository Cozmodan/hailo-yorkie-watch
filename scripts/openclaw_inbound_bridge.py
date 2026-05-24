from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError, load_environment  # noqa: E402
from yorkie_watch.event_state import LATEST_EVENT_PATH, load_latest_event  # noqa: E402
from yorkie_watch.openclaw_client import OpenClawClient  # noqa: E402
from yorkie_watch.openclaw_vision_tools import (  # noqa: E402
    OpenClawVisionToolClient,
    VisionRoute,
    VisionToolResult,
    format_vision_reply,
    select_vision_route,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8020
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
PAUSE_FLAG_PATH = RUNTIME_DIR / "alerts_paused.flag"
SECRET_HEADER = "X-OpenClaw-Secret"
SMOKE_TEST_HEADER = "X-OpenClaw-Smoke-Test"
KNOWN_COMMANDS = (
    "status, what do you see, what can you see, check camera, is the Yorkie there, "
    "is the dog there, last alert, was the last alert real, false trigger, pause alerts, resume alerts"
)


@dataclass(frozen=True)
class InboundConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    shared_secret: str = ""
    allowed_senders: tuple[str, ...] = ()
    latest_event_path: Path = LATEST_EVENT_PATH
    pause_flag_path: Path = PAUSE_FLAG_PATH
    smoke_send: bool = False


@dataclass(frozen=True)
class InboundMessage:
    sender: str
    message: str
    timestamp: str = ""


@dataclass(frozen=True)
class BridgeResponse:
    status: HTTPStatus
    payload: dict[str, object]


@dataclass(frozen=True)
class RoutedReply:
    ok: bool
    route: str
    message: str


def load_inbound_config() -> InboundConfig:
    load_environment()
    return InboundConfig(
        host=os.getenv("OPENCLAW_INBOUND_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=_parse_port(os.getenv("OPENCLAW_INBOUND_PORT", str(DEFAULT_PORT))),
        shared_secret=os.getenv("OPENCLAW_INBOUND_SHARED_SECRET", "").strip(),
        allowed_senders=parse_allowed_senders(os.getenv("OPENCLAW_ALLOWED_SENDERS", "")),
        smoke_send=_bool_env("OPENCLAW_INBOUND_SMOKE_SEND", False),
    )


def parse_allowed_senders(raw_value: str) -> tuple[str, ...]:
    return tuple(_normalize_sender(value) for value in raw_value.split(",") if _normalize_sender(value))


def parse_inbound_payload(payload: Mapping[str, Any]) -> InboundMessage:
    sender = _find_first_value(payload, ("sender", "from", "phone"))
    message = _find_first_value(payload, ("message", "text", "body"))
    timestamp = _find_first_value(payload, ("timestamp",))
    return InboundMessage(sender=sender.strip(), message=message.strip(), timestamp=timestamp.strip())


def validate_shared_secret(headers: Mapping[str, str], config: InboundConfig) -> bool:
    if not config.shared_secret:
        return True
    supplied = _header_value(headers, SECRET_HEADER)
    return bool(supplied) and compare_digest(supplied, config.shared_secret)


def is_allowed_sender(sender: str, config: InboundConfig) -> bool:
    if not config.allowed_senders:
        return True
    return _normalize_sender(sender) in config.allowed_senders


def handle_inbound(
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    config: InboundConfig,
    vision_client_factory: Callable[[], OpenClawVisionToolClient] | None = None,
    notifier_factory: Callable[[], OpenClawClient] | None = None,
) -> BridgeResponse:
    if not validate_shared_secret(headers, config):
        return _error(HTTPStatus.UNAUTHORIZED, "Invalid or missing OpenClaw shared secret.", route="auth")

    inbound = parse_inbound_payload(payload)
    smoke_test = _is_smoke_test(headers)
    LOGGER.info(
        "Incoming WhatsApp text received from %s: %s",
        _redact_sender(inbound.sender, config),
        _redact_text(inbound.message, config),
    )

    if not inbound.message:
        return _error(HTTPStatus.BAD_REQUEST, "Inbound OpenClaw message was empty.", route="validation")
    if inbound.sender and not smoke_test and not is_allowed_sender(inbound.sender, config):
        LOGGER.warning("Rejected inbound sender: %s", _redact_sender(inbound.sender, config))
        return _error(HTTPStatus.FORBIDDEN, "Inbound OpenClaw sender is not allowed.", route="validation")

    reply = route_command(
        inbound,
        config=config,
        vision_client_factory=vision_client_factory or OpenClawVisionToolClient.from_env,
    )
    send_replies = not (smoke_test and not config.smoke_send)
    reply_sent = False
    if send_replies:
        reply_sent = send_reply_to_sender(
            inbound,
            reply.message,
            notifier_factory=notifier_factory or OpenClawClient.from_env,
            config=config,
        )
    else:
        LOGGER.info("Smoke-test inbound request handled without sending WhatsApp reply.")

    status = HTTPStatus.OK if reply.ok else HTTPStatus.OK
    return BridgeResponse(
        status,
        {
            "ok": reply.ok,
            "route": reply.route,
            "reply_sent": reply_sent,
            "message": reply.message,
        },
    )


def route_command(
    inbound: InboundMessage,
    *,
    config: InboundConfig,
    vision_client_factory: Callable[[], OpenClawVisionToolClient],
) -> RoutedReply:
    command = _normalize_command(inbound.message)
    if command == "status":
        return RoutedReply(True, "status", build_status_summary(config))
    if command == "pause alerts":
        config.pause_flag_path.parent.mkdir(parents=True, exist_ok=True)
        config.pause_flag_path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        return RoutedReply(True, "pause_alerts", "Yorkie Watch alert pause flag is set.")
    if command == "resume alerts":
        config.pause_flag_path.unlink(missing_ok=True)
        return RoutedReply(True, "resume_alerts", "Yorkie Watch alert pause flag is cleared.")

    route = select_vision_route(inbound.message)
    if route == "camera_snapshot":
        LOGGER.info("Vision route selected: camera_snapshot")
        return _run_vision_route(route, inbound.message, vision_client_factory)
    if route == "latest_alert":
        LOGGER.info("Vision route selected: latest_alert")
        return _run_vision_route(route, inbound.message, vision_client_factory)

    return RoutedReply(False, "unknown", f"Unknown Yorkie Watch command. Use one of: {KNOWN_COMMANDS}.")


def build_status_summary(config: InboundConfig) -> str:
    paused = config.pause_flag_path.exists()
    status = "paused" if paused else "running"
    details = (
        f"Yorkie Watch status: {status}. "
        "Stack hint: run scripts/yorkie_stack_tmux.sh status on the Pi to inspect the tmux processes."
    )
    event = load_latest_event(config.latest_event_path)
    if event is None:
        return f"{details} No latest alert event is available yet."

    timestamp = str(event.get("timestamp", "unknown time"))
    detector_class = str(event.get("detector_class", "unknown object"))
    confidence = event.get("confidence", "unknown confidence")
    region = str(event.get("region", "unknown region"))
    summary = str(event.get("vlm_summary", "")).strip()
    details = (
        f"{details} Last alert: {detector_class} at confidence {confidence} "
        f"in {region} at {timestamp}."
    )
    if summary:
        details = f"{details} VLM summary: {summary}"
    return details


def send_reply_to_sender(
    inbound: InboundMessage,
    message: str,
    *,
    notifier_factory: Callable[[], OpenClawClient],
    config: InboundConfig,
) -> bool:
    try:
        notifier = notifier_factory()
    except (ConfigError, ValueError) as exc:
        LOGGER.warning("Could not create OpenClaw notifier for inbound reply: %s", _redact_text(str(exc), config))
        return False

    target = inbound.sender.strip() or getattr(notifier, "whatsapp_target", "")
    if target:
        notifier.whatsapp_target = target
    try:
        sent = notifier.send_message(message, event_type="yorkie_watch_inbound_reply")
    except Exception as exc:  # noqa: BLE001 - inbound bridge should return JSON instead of crashing.
        LOGGER.warning("OpenClaw inbound reply send failed: %s", _redact_text(str(exc), config))
        return False
    if sent:
        LOGGER.info("Reply sent to WhatsApp target %s.", _redact_sender(target, config))
    return sent


def make_handler(
    *,
    config: InboundConfig,
    vision_client_factory: Callable[[], OpenClawVisionToolClient] | None = None,
    notifier_factory: Callable[[], OpenClawClient] | None = None,
) -> type[BaseHTTPRequestHandler]:
    class OpenClawInboundHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(_error(HTTPStatus.NOT_FOUND, "Not found.", route="not_found"))
                return
            self._send_json(BridgeResponse(HTTPStatus.OK, {"ok": True, "status": "ok"}))

        def do_POST(self) -> None:
            if self.path != "/openclaw/inbound":
                self._send_json(_error(HTTPStatus.NOT_FOUND, "Not found.", route="not_found"))
                return
            try:
                payload = self._read_json_payload()
            except ValueError as exc:
                self._send_json(_error(HTTPStatus.BAD_REQUEST, str(exc), route="validation"))
                return

            response = handle_inbound(
                payload,
                self.headers,
                config=config,
                vision_client_factory=vision_client_factory,
                notifier_factory=notifier_factory,
            )
            self._send_json(response)

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json_payload(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            if length <= 0:
                raise ValueError("Request body must contain a JSON object.")

            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_json(self, response: BridgeResponse) -> None:
            body = json.dumps(response.payload, sort_keys=True).encode("utf-8")
            self.send_response(response.status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OpenClawInboundHandler


def run_server(config: InboundConfig | None = None) -> None:
    config = config or load_inbound_config()
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config=config))
    LOGGER.info("OpenClaw deterministic inbound bridge listening on http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_server()
    return 0


def _reply_from_vision_result(route: VisionRoute, result: VisionToolResult) -> RoutedReply:
    return RoutedReply(
        ok=result.ok,
        route=route,
        message=format_vision_reply(result),
    )


def _run_vision_route(
    route: VisionRoute,
    prompt: str,
    vision_client_factory: Callable[[], OpenClawVisionToolClient],
) -> RoutedReply:
    try:
        vision_client = vision_client_factory()
    except (ConfigError, ValueError) as exc:
        return RoutedReply(False, route, f"I could not configure the Yorkie Watch vision tool: {exc}")
    if route == "latest_alert":
        return _reply_from_vision_result(route, vision_client.latest_alert(prompt=prompt))
    return _reply_from_vision_result(route, vision_client.camera_snapshot(prompt=prompt))


def _error(status: HTTPStatus, message: str, *, route: str) -> BridgeResponse:
    return BridgeResponse(status, {"ok": False, "route": route, "reply_sent": False, "message": message})


def _parse_port(raw_value: str | None) -> int:
    raw_value = (raw_value or str(DEFAULT_PORT)).strip()
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError("OPENCLAW_INBOUND_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("OPENCLAW_INBOUND_PORT must be between 1 and 65535.")
    return port


def _find_first_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    stack: list[Mapping[str, Any]] = [payload]
    while stack:
        current = stack.pop(0)
        for key in keys:
            if key in current and current[key] is not None:
                return str(current[key])
        for value in current.values():
            if isinstance(value, Mapping):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(item for item in value if isinstance(item, Mapping))
    return ""


def _header_value(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value.strip()
    return ""


def _normalize_sender(sender: str) -> str:
    return "".join(str(sender).strip().lower().split())


def _normalize_command(message: str) -> str:
    return " ".join(message.strip().lower().split())


def _is_smoke_test(headers: Mapping[str, str]) -> bool:
    return _header_value(headers, SMOKE_TEST_HEADER).lower() in {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _redact_sender(sender: str, config: InboundConfig) -> str:
    if not sender:
        return "<missing-sender>"
    normalized = _normalize_sender(sender)
    if not normalized:
        return "<missing-sender>"
    return "<redacted-sender>"


def _redact_text(text: str, config: InboundConfig) -> str:
    redacted = str(text)
    if config.shared_secret:
        redacted = redacted.replace(config.shared_secret, "<redacted-inbound-secret>")
    for sender in config.allowed_senders:
        redacted = redacted.replace(sender, "<redacted-sender>")
    redacted = re.sub(r"\+?\d[\d\s().-]{6,}\d", "<redacted-phone>", redacted)
    return redacted


if __name__ == "__main__":
    raise SystemExit(main())
