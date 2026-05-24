from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError, get_env, get_float_env, load_environment  # noqa: E402
from yorkie_watch.event_state import LATEST_EVENT_PATH, latest_event_image_path, load_latest_event  # noqa: E402
from yorkie_watch.ha_client import HomeAssistantClient, HomeAssistantError  # noqa: E402
from yorkie_watch.vlm_client import VLMClient, VLMResult, redact_vlm_text  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8021
DEFAULT_PROMPT = "Describe what you can see. Mention whether a dog or Yorkie is visible and include uncertainty."
SECRET_HEADER = "X-OpenClaw-Secret"
VISION_TOOL_DIR = PROJECT_ROOT / "data" / "vision_tool"


@dataclass(frozen=True)
class VisionToolConfig:
    """Runtime settings for OpenClaw's local vision tool API."""

    host: str
    port: int
    shared_secret: str
    default_prompt: str
    latest_event_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ToolResponse:
    """HTTP status and JSON payload for one vision tool request."""

    status: HTTPStatus
    payload: dict[str, object]


def load_vision_tool_config() -> VisionToolConfig:
    load_environment()
    return VisionToolConfig(
        host=get_env("OPENCLAW_VISION_TOOL_HOST", DEFAULT_HOST) or DEFAULT_HOST,
        port=_parse_port(os.getenv("OPENCLAW_VISION_TOOL_PORT", str(DEFAULT_PORT)), "OPENCLAW_VISION_TOOL_PORT"),
        shared_secret=get_env("OPENCLAW_VISION_TOOL_SHARED_SECRET"),
        default_prompt=get_env("OPENCLAW_VISION_DEFAULT_PROMPT", DEFAULT_PROMPT) or DEFAULT_PROMPT,
        latest_event_path=LATEST_EVENT_PATH,
        output_dir=VISION_TOOL_DIR,
    )


class VisionTool:
    """JSON-only OpenClaw vision tool backed by Yorkie Watch and local VLM."""

    def __init__(
        self,
        *,
        config: VisionToolConfig,
        vlm_client_factory: Callable[[], object] | None = None,
        ha_client_factory: Callable[[], HomeAssistantClient] = HomeAssistantClient.from_env,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.vlm_client_factory = vlm_client_factory or create_vlm_client_from_env
        self.ha_client_factory = ha_client_factory
        self.clock = clock

    def handle(self, path: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> ToolResponse:
        if not validate_shared_secret(headers, self.config):
            return _error(HTTPStatus.UNAUTHORIZED, "Invalid or missing OpenClaw vision tool shared secret.")
        if path == "/vision/latest-alert":
            return self.latest_alert(payload)
        if path == "/vision/camera-snapshot":
            return self.camera_snapshot(payload)
        if path == "/vision/describe-image":
            return self.describe_image(payload)
        return _error(HTTPStatus.NOT_FOUND, "Not found.")

    def health(self) -> ToolResponse:
        return ToolResponse(HTTPStatus.OK, {"ok": True, "status": "ok"})

    def latest_alert(self, payload: Mapping[str, Any]) -> ToolResponse:
        event = load_latest_event(self.config.latest_event_path)
        if event is None:
            return _ok_false("latest_alert", "Latest event state was not found.")

        image_path = latest_event_image_path(event)
        if image_path is None:
            return _ok_false("latest_alert", "Latest event image path is missing.")
        if not image_path.exists():
            return _ok_false("latest_alert", f"Latest event image does not exist: {_display_path(image_path)}")

        return self._describe_path(
            source="latest_alert",
            image_path=image_path,
            prompt=_prompt_from_payload(payload, self.config.default_prompt),
        )

    def camera_snapshot(self, payload: Mapping[str, Any]) -> ToolResponse:
        output_path = self.config.output_dir / f"camera_snapshot_{_timestamp(self.clock())}.jpg"
        try:
            snapshot_path = self.ha_client_factory().save_snapshot(output_path, attempts=3, delay_seconds=2.0)
        except (ConfigError, HomeAssistantError, OSError, ValueError) as exc:
            LOGGER.warning("Vision tool camera snapshot failed: %s", exc)
            return _ok_false("camera_snapshot", f"Could not fetch Home Assistant snapshot: {exc}")

        return self._describe_path(
            source="camera_snapshot",
            image_path=snapshot_path,
            prompt=_prompt_from_payload(payload, self.config.default_prompt),
        )

    def describe_image(self, payload: Mapping[str, Any]) -> ToolResponse:
        raw_image = payload.get("image_base64")
        if not isinstance(raw_image, str) or not raw_image.strip():
            return _error(HTTPStatus.BAD_REQUEST, "JSON body must include image_base64.")
        try:
            image_path = save_base64_image(
                raw_image,
                output_dir=self.config.output_dir,
                stem=f"provided_image_{_timestamp(self.clock())}",
            )
        except ValueError as exc:
            return _error(HTTPStatus.BAD_REQUEST, str(exc))

        return self._describe_path(
            source="provided_image",
            image_path=image_path,
            prompt=_prompt_from_payload(payload, self.config.default_prompt),
        )

    def _describe_path(self, *, source: str, image_path: Path, prompt: str) -> ToolResponse:
        try:
            client = self.vlm_client_factory()
            result = client.describe_image(image_path, prompt)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - HTTP tool must always return JSON.
            LOGGER.warning("Vision tool VLM description failed: %s", redact_vlm_text(str(exc)))
            return _ok_false(source, f"VLM description failed: {redact_vlm_text(str(exc))}", image_path=image_path)
        if not isinstance(result, VLMResult):
            result = VLMResult(
                ok=bool(getattr(result, "ok", False)),
                text=str(getattr(result, "text", "")),
                error=str(getattr(result, "error", "")),
                model=str(getattr(result, "model", "")),
            )
        if not result.ok:
            return _ok_false(source, result.error or "VLM description failed.", image_path=image_path)
        return ToolResponse(
            HTTPStatus.OK,
            {
                "ok": True,
                "source": source,
                "description": result.text,
                "image_path": _display_path(image_path),
            },
        )


def create_vlm_client_from_env() -> VLMClient:
    load_environment()
    base_url = get_env("YORKIE_VLM_BASE_URL", "http://127.0.0.1:8010") or "http://127.0.0.1:8010"
    model = get_env("YORKIE_VLM_MODEL", "Qwen2-VL-2B-Instruct") or "Qwen2-VL-2B-Instruct"
    timeout_seconds = max(1.0, get_float_env("YORKIE_VLM_TIMEOUT_SECONDS", 60.0))
    return VLMClient(base_url=base_url, model=model, timeout_seconds=timeout_seconds)


def validate_shared_secret(headers: Mapping[str, str], config: VisionToolConfig) -> bool:
    if not config.shared_secret:
        return True
    supplied = _header_value(headers, SECRET_HEADER)
    return bool(supplied) and compare_digest(supplied, config.shared_secret)


def save_base64_image(encoded_image: str, *, output_dir: str | Path, stem: str) -> Path:
    image_bytes = _decode_base64_image(encoded_image)
    suffix = _image_suffix(image_bytes)
    if suffix is None:
        raise ValueError("image_base64 must be a JPEG or PNG image.")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{stem}{suffix}"
    output_path.write_bytes(image_bytes)
    return output_path


def make_handler(*, tool: VisionTool) -> type[BaseHTTPRequestHandler]:
    class OpenClawVisionToolHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(_error(HTTPStatus.NOT_FOUND, "Not found."))
                return
            self._send_json(tool.health())

        def do_POST(self) -> None:
            if self.path not in {"/vision/latest-alert", "/vision/camera-snapshot", "/vision/describe-image"}:
                self._send_json(_error(HTTPStatus.NOT_FOUND, "Not found."))
                return
            try:
                payload = self._read_json_payload()
            except ValueError as exc:
                self._send_json(_error(HTTPStatus.BAD_REQUEST, str(exc)))
                return
            self._send_json(tool.handle(self.path, payload, self.headers))

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json_payload(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            if length <= 0:
                return {}
            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_json(self, response: ToolResponse) -> None:
            body = json.dumps(response.payload, sort_keys=True).encode("utf-8")
            self.send_response(response.status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OpenClawVisionToolHandler


def run_server(config: VisionToolConfig | None = None) -> None:
    config = config or load_vision_tool_config()
    tool = VisionTool(config=config)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(tool=tool))
    LOGGER.info("OpenClaw vision tool listening on http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_server()
    return 0


def _decode_base64_image(encoded_image: str) -> bytes:
    raw = encoded_image.strip()
    if "," in raw and raw.lower().startswith("data:image/"):
        raw = raw.split(",", 1)[1].strip()
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64.") from exc


def _image_suffix(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return None


def _prompt_from_payload(payload: Mapping[str, Any], default_prompt: str) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return default_prompt


def _header_value(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value).strip()
    return ""


def _ok_false(source: str, error: str, *, image_path: Path | None = None) -> ToolResponse:
    payload: dict[str, object] = {"ok": False, "source": source, "error": error}
    if image_path is not None:
        payload["image_path"] = _display_path(image_path)
    return ToolResponse(HTTPStatus.OK, payload)


def _error(status: HTTPStatus, error: str) -> ToolResponse:
    return ToolResponse(status, {"ok": False, "error": error})


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _timestamp(value: float) -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(value)) + f"_{int(value * 1000) % 1000:03d}"


def _parse_port(raw_value: str | None, env_name: str) -> int:
    raw_value = (raw_value or "").strip() or str(DEFAULT_PORT)
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{env_name} must be between 1 and 65535.")
    return port


if __name__ == "__main__":
    raise SystemExit(main())
