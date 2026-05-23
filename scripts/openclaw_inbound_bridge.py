from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.event_state import LATEST_EVENT_PATH, load_latest_event  # noqa: E402
from yorkie_watch.main import run_chat  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8020
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
PAUSE_FLAG_PATH = RUNTIME_DIR / "alerts_paused.flag"
SECRET_HEADER = "X-OpenClaw-Secret"
CHAT_COMMANDS = {"check last alert", "last alert", "is that a dog", "false trigger"}
KNOWN_COMMANDS = "status, check last alert, last alert, is that a dog, false trigger, pause alerts, resume alerts"


@dataclass(frozen=True)
class InboundConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    shared_secret: str = ""
    allowed_senders: tuple[str, ...] = ()
    latest_event_path: Path = LATEST_EVENT_PATH
    pause_flag_path: Path = PAUSE_FLAG_PATH


@dataclass(frozen=True)
class InboundMessage:
    sender: str
    message: str
    timestamp: str = ""


@dataclass(frozen=True)
class BridgeResponse:
    status: HTTPStatus
    payload: dict[str, object]


def load_inbound_config() -> InboundConfig:
    return InboundConfig(
        host=os.getenv("OPENCLAW_INBOUND_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=_parse_port(os.getenv("OPENCLAW_INBOUND_PORT", str(DEFAULT_PORT))),
        shared_secret=os.getenv("OPENCLAW_INBOUND_SHARED_SECRET", "").strip(),
        allowed_senders=parse_allowed_senders(os.getenv("OPENCLAW_ALLOWED_SENDERS", "")),
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
    chat_runner: Callable[..., int] = run_chat,
) -> BridgeResponse:
    if not validate_shared_secret(headers, config):
        return _error(HTTPStatus.UNAUTHORIZED, "Invalid or missing OpenClaw shared secret.")

    inbound = parse_inbound_payload(payload)
    if not inbound.message:
        return _error(HTTPStatus.BAD_REQUEST, "Inbound OpenClaw message was empty.")
    if not is_allowed_sender(inbound.sender, config):
        return _error(HTTPStatus.FORBIDDEN, "Inbound OpenClaw sender is not allowed.")

    return route_command(inbound, config=config, chat_runner=chat_runner)


def route_command(
    inbound: InboundMessage,
    *,
    config: InboundConfig,
    chat_runner: Callable[..., int] = run_chat,
) -> BridgeResponse:
    command = _normalize_command(inbound.message)
    if command == "status":
        return _ok("status", build_status_summary(config))
    if command in CHAT_COMMANDS:
        return _run_latest_alert_chat(inbound.message, config=config, chat_runner=chat_runner)
    if command == "pause alerts":
        config.pause_flag_path.parent.mkdir(parents=True, exist_ok=True)
        config.pause_flag_path.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        return _ok("pause_alerts", "Yorkie Watch alert pause flag is set.")
    if command == "resume alerts":
        try:
            config.pause_flag_path.unlink()
        except FileNotFoundError:
            pass
        return _ok("resume_alerts", "Yorkie Watch alert pause flag is cleared.")

    return _ok("unknown", f"Unknown Yorkie Watch command. Use one of: {KNOWN_COMMANDS}.")


def build_status_summary(config: InboundConfig) -> str:
    paused = config.pause_flag_path.exists()
    status = "paused" if paused else "running"
    event = load_latest_event(config.latest_event_path)
    if event is None:
        return f"Yorkie Watch status: {status}. No latest alert event is available yet."

    timestamp = str(event.get("timestamp", "unknown time"))
    detector_class = str(event.get("detector_class", "unknown object"))
    confidence = event.get("confidence", "unknown confidence")
    region = str(event.get("region", "unknown region"))
    summary = str(event.get("vlm_summary", "")).strip()
    details = (
        f"Yorkie Watch status: {status}. Last alert: {detector_class} "
        f"at confidence {confidence} in {region} at {timestamp}."
    )
    if summary:
        details = f"{details} VLM summary: {summary}"
    return details


def make_handler(
    *,
    config: InboundConfig,
    chat_runner: Callable[..., int] = run_chat,
) -> type[BaseHTTPRequestHandler]:
    class OpenClawInboundHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/openclaw/inbound":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                return
            try:
                payload = self._read_json_payload()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            response = handle_inbound(payload, self.headers, config=config, chat_runner=chat_runner)
            self._send_json(response.status, response.payload)

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

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OpenClawInboundHandler


def run_server(config: InboundConfig | None = None) -> None:
    config = config or load_inbound_config()
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config=config))
    LOGGER.info("OpenClaw inbound bridge listening on http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_server()
    return 0


def _run_latest_alert_chat(
    question: str,
    *,
    config: InboundConfig,
    chat_runner: Callable[..., int] = run_chat,
) -> BridgeResponse:
    output = StringIO()
    with redirect_stdout(output):
        returncode = chat_runner(question, send_reply=True, latest_event_path=config.latest_event_path)
    message = output.getvalue().strip()
    if returncode == 0:
        return _ok("chat_reply", message or "Yorkie Watch chat reply was sent.")
    return BridgeResponse(
        HTTPStatus.OK,
        {
            "ok": False,
            "command": "chat_reply",
            "message": message or "Yorkie Watch could not answer the latest alert question.",
        },
    )


def _ok(command: str, message: str) -> BridgeResponse:
    return BridgeResponse(HTTPStatus.OK, {"ok": True, "command": command, "message": message})


def _error(status: HTTPStatus, message: str) -> BridgeResponse:
    return BridgeResponse(status, {"ok": False, "error": message})


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


if __name__ == "__main__":
    raise SystemExit(main())