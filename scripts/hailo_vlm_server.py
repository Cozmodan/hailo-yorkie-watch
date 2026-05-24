#!/usr/bin/python3
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.hailo_lock import HailoDeviceLock, HailoDeviceLockError  # noqa: E402

LOGGER = logging.getLogger("hailo_vlm_server")
DEFAULT_HEF = "/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef"
DEFAULT_MODEL = "Qwen2-VL-2B-Instruct"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
DEFAULT_MAX_TOKENS = 80
DEFAULT_INPUT_SHAPE = (336, 336, 3)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")


class RequestError(ValueError):
    """Raised for client-side request validation errors."""


@dataclass(frozen=True)
class ServerConfig:
    """Runtime settings for the local Hailo VLM HTTP wrapper."""

    hef_path: str
    host: str
    port: int
    max_tokens: int
    optimize_memory: bool
    clear_context: bool
    unload_after_request: bool
    model: str


@dataclass(frozen=True)
class VLMRequest:
    """Normalized image prompt request for the Hailo VLM runtime."""

    model: str
    prompt: list[dict[str, Any]]
    images: tuple[str, ...]


class HailoVLMRuntime:
    """Owns the Hailo VLM instance and serializes generation calls."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        vdevice: object,
        vlm: object,
        cv2_module: object,
        numpy_module: object,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    ) -> None:
        self.config = config
        self.vdevice = vdevice
        self.vlm = vlm
        self.cv2 = cv2_module
        self.np = numpy_module
        self.input_shape = input_shape
        self.lock = threading.Lock()

    def generate(self, request: VLMRequest) -> str:
        frames = [
            decode_image_to_frame(
                encoded_image,
                cv2_module=self.cv2,
                numpy_module=self.np,
                input_shape=self.input_shape,
            )
            for encoded_image in request.images
        ]
        with self.lock:
            if self.config.clear_context and hasattr(self.vlm, "clear_context"):
                self.vlm.clear_context()
            output = call_vlm_generate(
                self.vlm,
                request.prompt,
                frames,
                max_tokens=self.config.max_tokens,
            )
        return clean_response_text(output)

    def close(self) -> None:
        """Release Hailo runtime resources when supported by the loaded objects."""
        close_resource(self.vlm)
        close_resource(self.vdevice)


class HailoVLMRuntimeManager:
    """Loads the Hailo VLM permanently or per request based on configuration."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        runtime_loader: Callable[[ServerConfig], HailoVLMRuntime] | None = None,
        runtime: HailoVLMRuntime | None = None,
    ) -> None:
        self.config = config
        self._runtime_loader = runtime_loader or load_hailo_runtime
        self._runtime = runtime
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def generate(self, request: VLMRequest) -> str:
        if self.config.unload_after_request:
            return self._generate_with_temporary_runtime(request)
        with self._lock:
            with HailoDeviceLock.from_env():
                runtime = self._runtime
                if runtime is None:
                    self._runtime = self._load_runtime()
                    runtime = self._runtime
                return runtime.generate(request)

    def close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.close()

    def _generate_with_temporary_runtime(self, request: VLMRequest) -> str:
        with self._lock:
            try:
                with HailoDeviceLock.from_env():
                    runtime = self._load_runtime()
                    self._runtime = runtime
                    try:
                        return runtime.generate(request)
                    finally:
                        self._runtime = None
                        runtime.close()
            except HailoDeviceLockError:
                raise

    def _load_runtime(self) -> HailoVLMRuntime:
        return self._runtime_loader(self.config)


def load_config() -> ServerConfig:
    """Load Hailo VLM wrapper settings from environment variables."""
    hef_path = get_env("HAILO_VLM_HEF", DEFAULT_HEF)
    return ServerConfig(
        hef_path=hef_path,
        host=get_env("HAILO_VLM_HOST", DEFAULT_HOST),
        port=get_int_env("HAILO_VLM_PORT", DEFAULT_PORT),
        max_tokens=max(1, get_int_env("HAILO_VLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        optimize_memory=get_bool_env("HAILO_VLM_OPTIMIZE_MEMORY", True),
        clear_context=get_bool_env("HAILO_VLM_CLEAR_CONTEXT", True),
        unload_after_request=get_bool_env("HAILO_VLM_UNLOAD_AFTER_REQUEST", True),
        model=Path(hef_path).stem or DEFAULT_MODEL,
    )


def get_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if value and value.strip() else default


def get_int_env(name: str, default: int) -> int:
    raw_value = get_env(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def load_hailo_runtime(config: ServerConfig) -> HailoVLMRuntime:
    """Import system Hailo dependencies and load the HEF once at startup."""
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from hailo_platform import VDevice  # type: ignore[import-not-found]
    from hailo_platform.genai import VLM  # type: ignore[import-not-found]

    vdevice = VDevice()
    vlm = instantiate_vlm(
        vlm_class=VLM,
        vdevice=vdevice,
        hef_path=config.hef_path,
        optimize_memory=config.optimize_memory,
    )
    LOGGER.info("Loaded Hailo VLM model %s.", Path(config.hef_path).name)
    return HailoVLMRuntime(
        config=config,
        vdevice=vdevice,
        vlm=vlm,
        cv2_module=cv2,
        numpy_module=np,
    )


def build_runtime_manager(config: ServerConfig) -> HailoVLMRuntimeManager:
    """Build the runtime manager without monopolizing Hailo when unloading is enabled."""
    if config.unload_after_request:
        LOGGER.info(
            "Hailo VLM unload-after-request mode enabled; the model will load only during requests."
        )
        return HailoVLMRuntimeManager(config=config)
    with HailoDeviceLock.from_env():
        return HailoVLMRuntimeManager(config=config, runtime=load_hailo_runtime(config))


def close_resource(resource: object) -> None:
    """Call the first supported close-style method on a runtime resource."""
    for method_name in ("close", "release", "shutdown"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception as exc:
            LOGGER.warning("Could not release Hailo VLM resource via %s(): %s", method_name, exc)
        return


def instantiate_vlm(*, vlm_class: object, vdevice: object, hef_path: str, optimize_memory: bool) -> object:
    """Try common Hailo VLM constructor signatures without importing in tests."""
    attempts = [
        lambda: vlm_class(vdevice=vdevice, hef_path=hef_path, optimize_memory=optimize_memory),  # type: ignore[misc]
        lambda: vlm_class(hef_path=hef_path, vdevice=vdevice, optimize_memory=optimize_memory),  # type: ignore[misc]
        lambda: vlm_class(hef_path=hef_path, optimize_memory=optimize_memory),  # type: ignore[misc]
        lambda: vlm_class(vdevice, hef_path, optimize_memory=optimize_memory),  # type: ignore[misc]
        lambda: vlm_class(hef_path, vdevice, optimize_memory=optimize_memory),  # type: ignore[misc]
        lambda: vlm_class(vdevice, hef_path),  # type: ignore[misc]
        lambda: vlm_class(hef_path, vdevice),  # type: ignore[misc]
        lambda: vlm_class(hef_path),  # type: ignore[misc]
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Could not instantiate Hailo VLM with known signatures: {errors[-1] if errors else ''}")


def parse_chat_payload(payload: dict[str, Any], *, default_model: str = DEFAULT_MODEL) -> VLMRequest:
    """Parse Ollama-style /api/chat JSON into a structured Hailo prompt."""
    model = str(payload.get("model") or default_model)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("JSON body must include a non-empty messages list.")
    prompt, images = build_structured_prompt(messages)
    return VLMRequest(model=model, prompt=prompt, images=tuple(images))


def parse_generate_payload(payload: dict[str, Any], *, default_model: str = DEFAULT_MODEL) -> VLMRequest:
    """Parse Ollama-style /api/generate JSON into a structured Hailo prompt."""
    model = str(payload.get("model") or default_model)
    prompt_text = payload.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise RequestError("JSON body must include a non-empty prompt string.")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise RequestError("JSON body must include a non-empty images list.")
    image_values = validate_image_list(images)
    prompt = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _encoded in image_values),
                {"type": "text", "text": prompt_text.strip()},
            ],
        }
    ]
    return VLMRequest(model=model, prompt=prompt, images=tuple(image_values))


def build_structured_prompt(messages: list[object]) -> tuple[list[dict[str, Any]], list[str]]:
    """Build Hailo structured prompts and return matching base64 image inputs."""
    structured: list[dict[str, Any]] = []
    images: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            raise RequestError("Each message must be a JSON object.")
        role = str(item.get("role") or "user")
        text = extract_message_text(item.get("content"))
        message_images = validate_image_list(item.get("images") or [])
        content: list[dict[str, str]] = []
        for encoded_image in message_images:
            content.append({"type": "image"})
            images.append(encoded_image)
        if text:
            content.append({"type": "text", "text": text})
        if content:
            structured.append({"role": role, "content": content})

    if not images:
        raise RequestError("At least one base64 image is required.")
    if not structured:
        raise RequestError("At least one message with text or image content is required.")
    return structured, images


def extract_message_text(content: object) -> str:
    """Extract plain text from Ollama/OpenAI-style message content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for entry in content:
            if isinstance(entry, dict) and entry.get("type") == "text" and isinstance(entry.get("text"), str):
                parts.append(entry["text"].strip())
            elif isinstance(entry, str):
                parts.append(entry.strip())
        return "\n".join(part for part in parts if part)
    return ""


def validate_image_list(images: object) -> list[str]:
    if not isinstance(images, list):
        raise RequestError("images must be a list.")
    values: list[str] = []
    for image in images:
        if not isinstance(image, str) or not image.strip():
            raise RequestError("Each image must be a non-empty base64 string.")
        values.append(image.strip())
    return values


def decode_image_to_frame(
    encoded_image: str,
    *,
    cv2_module: object,
    numpy_module: object,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
) -> object:
    """Decode a base64 JPEG/PNG into RGB uint8 HWC frame for Hailo VLM."""
    encoded = strip_data_uri_prefix(encoded_image)
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError("Image is not valid base64.") from exc
    if not image_bytes:
        raise RequestError("Image base64 decoded to empty bytes.")

    np_buffer = numpy_module.frombuffer(image_bytes, dtype=numpy_module.uint8)
    decoded = cv2_module.imdecode(np_buffer, cv2_module.IMREAD_COLOR)
    if decoded is None:
        raise RequestError("Image could not be decoded as JPEG or PNG.")

    rgb = cv2_module.cvtColor(decoded, cv2_module.COLOR_BGR2RGB)
    height, width, channels = input_shape
    if channels != 3:
        raise RequestError("Hailo VLM input shape must have 3 channels.")
    resized = cv2_module.resize(rgb, (width, height), interpolation=getattr(cv2_module, "INTER_AREA", 3))
    return resized.astype(numpy_module.uint8, copy=False)


def strip_data_uri_prefix(encoded_image: str) -> str:
    if "," in encoded_image and encoded_image.lower().startswith("data:image/"):
        return encoded_image.split(",", 1)[1].strip()
    return encoded_image.strip()


def call_vlm_generate(vlm: object, prompt: list[dict[str, Any]], frames: list[object], *, max_tokens: int) -> str:
    """Call the Hailo VLM generation method with common signature variants."""
    attempts = [
        lambda: vlm.generate(prompt, frames, max_tokens=max_tokens),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt, frames, max_new_tokens=max_tokens),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt, frames, max_generated_tokens=max_tokens),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt=prompt, images=frames, max_tokens=max_tokens),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt=prompt, images=frames, max_new_tokens=max_tokens),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt=prompt, images=frames),  # type: ignore[attr-defined]
        lambda: vlm.generate(prompt, frames),  # type: ignore[attr-defined]
    ]
    if len(frames) == 1:
        frame = frames[0]
        attempts.extend(
            [
                lambda: vlm.generate(prompt, frame, max_tokens=max_tokens),  # type: ignore[attr-defined]
                lambda: vlm.generate(prompt, frame, max_new_tokens=max_tokens),  # type: ignore[attr-defined]
                lambda: vlm.generate(prompt, frame),  # type: ignore[attr-defined]
            ]
        )

    errors: list[str] = []
    for attempt in attempts:
        try:
            return normalize_generation_output(attempt())
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Could not call Hailo VLM generate with known signatures: {errors[-1] if errors else ''}")


def normalize_generation_output(output: object) -> str:
    """Normalize Hailo generation outputs that may be strings, chunks, or mappings."""
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if isinstance(output, dict):
        for key in ("text", "response", "content"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    if hasattr(output, "__iter__"):
        chunks: list[str] = []
        for chunk in output:  # type: ignore[union-attr]
            chunks.append(normalize_generation_output(chunk))
        return "".join(chunks)
    return str(output)


def clean_response_text(text: object) -> str:
    """Remove model special tokens and trim whitespace from the final answer."""
    cleaned = SPECIAL_TOKEN_RE.sub("", str(text))
    cleaned = cleaned.replace("<|im_end|>", "")
    return " ".join(cleaned.strip().split())


def ollama_response(*, model: str, content: str) -> dict[str, Any]:
    """Build an Ollama-compatible non-streaming response."""
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "response": content,
        "done": True,
    }


class HailoVLMHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for an Ollama-style local Hailo VLM server."""

    server_version = "HailoVLMServer/0.1"

    @property
    def runtime(self) -> HailoVLMRuntimeManager:
        return self.server.runtime  # type: ignore[attr-defined, no-any-return]

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(
            {
                "ok": True,
                "model": self.runtime.config.model,
                "loaded": self.runtime.loaded,
                "unload_after_request": self.runtime.config.unload_after_request,
            }
        )

    def do_POST(self) -> None:
        if self.path not in {"/api/chat", "/api/generate"}:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._read_json_body()
            if self.path == "/api/chat":
                request = parse_chat_payload(payload, default_model=self.runtime.config.model)
            else:
                request = parse_generate_payload(payload, default_model=self.runtime.config.model)
            response_text = self.runtime.generate(request)
        except RequestError as exc:
            self._send_json({"error": str(exc), "done": True}, status=400)
            return
        except Exception as exc:
            LOGGER.exception("Hailo VLM generation failed.")
            self._send_json({"error": str(exc), "done": True}, status=500)
            return

        self._send_json(ollama_response(model=request.model, content=response_text))

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise RequestError("Request body is empty.")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise RequestError("Request body must be a JSON object.")
        return payload

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HailoVLMHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server with an attached Hailo runtime."""

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], runtime: HailoVLMRuntimeManager) -> None:
        super().__init__(server_address, handler_class)
        self.runtime = runtime


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    runtime = build_runtime_manager(config)
    server = HailoVLMHTTPServer((config.host, config.port), HailoVLMHandler, runtime)
    LOGGER.info(
        "Serving Hailo VLM wrapper on http://%s:%d using model %s.",
        config.host,
        config.port,
        config.model,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Hailo VLM wrapper stopped.")
    finally:
        runtime.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
