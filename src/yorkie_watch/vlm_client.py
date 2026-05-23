from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import VLMConfig

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only when Pillow is unavailable.
    Image = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
VLM_TEMP_DIR = Path("data") / "vlm_tmp"


class VLMError(RuntimeError):
    """Raised when a local VLM request cannot be completed."""


@dataclass(frozen=True)
class VLMResult:
    """Normalized VLM response used by alerts and chat mode."""

    ok: bool
    text: str
    error: str
    model: str

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "ok": self.ok,
            "text": self.text,
            "error": self.error,
            "model": self.model,
        }


class VLMClient:
    """Small Ollama-compatible local VLM client."""

    def __init__(self, base_url: str, model: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: VLMConfig) -> "VLMClient":
        return cls(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )

    def describe_image(self, image_path: str | Path, prompt: str) -> VLMResult:
        """Ask the local VLM about one image, falling back from chat to generate API."""
        image = Path(image_path)
        if not image.exists():
            return VLMResult(False, "", f"VLM image does not exist: {image}", self.model)

        try:
            encoded_image = base64.b64encode(image.read_bytes()).decode("ascii")
        except OSError as exc:
            error = f"Could not read VLM image: {exc}"
            return VLMResult(False, "", redact_vlm_text(error, self.base_url), self.model)

        chat_error = ""
        try:
            chat_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [encoded_image],
                    }
                ],
                "stream": False,
            }
            text = _extract_response_text(self._post_json("/api/chat", chat_payload))
            if text:
                return VLMResult(True, shorten_vlm_text(text), "", self.model)
            chat_error = "VLM /api/chat returned no text."
        except VLMError as exc:
            chat_error = str(exc)
            LOGGER.info("VLM /api/chat failed; falling back to /api/generate: %s", chat_error)

        try:
            generate_payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [encoded_image],
                "stream": False,
            }
            text = _extract_response_text(self._post_json("/api/generate", generate_payload))
            if text:
                return VLMResult(True, shorten_vlm_text(text), "", self.model)
            error = "VLM /api/generate returned no text."
        except VLMError as exc:
            error = str(exc)

        if chat_error:
            error = f"{chat_error}; {error}"
        return VLMResult(False, "", redact_vlm_text(error, self.base_url), self.model)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            details = _short_error_body(exc.read())
            message = f"VLM request failed with HTTP {exc.code}: {exc.reason}; body={details!r}"
            raise VLMError(redact_vlm_text(message, self.base_url)) from exc
        except URLError as exc:
            message = f"Could not reach VLM service: {exc.reason}"
            raise VLMError(redact_vlm_text(message, self.base_url)) from exc
        except TimeoutError as exc:
            message = f"Timed out waiting for VLM service after {self.timeout_seconds:g}s"
            raise VLMError(message) from exc
        except OSError as exc:
            message = f"Could not start VLM request: {exc}"
            raise VLMError(redact_vlm_text(message, self.base_url)) from exc

        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VLMError("VLM response was not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise VLMError("VLM response JSON was not an object.")
        return parsed


def create_vlm_image_copy(
    image_path: str | Path,
    *,
    max_width: int,
    output_dir: str | Path = VLM_TEMP_DIR,
) -> Path:
    """Create a temporary VLM-ready copy without modifying the evidence image."""
    source = Path(image_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{source.stem}_vlm_{int(time.time() * 1000)}.jpg"

    if Image is None:
        shutil.copy2(source, output_path)
        return output_path

    try:
        with Image.open(source) as image:
            image.load()
            prepared = image.convert("RGB")
            if prepared.width > max_width:
                ratio = max_width / float(prepared.width)
                height = max(1, int(prepared.height * ratio))
                prepared = prepared.resize((max_width, height))
            prepared.save(output_path, quality=85, optimize=True)
            return output_path
    except OSError as exc:
        LOGGER.warning("Pillow could not prepare VLM image copy; using raw copy: %s", exc)
        shutil.copy2(source, output_path)
        return output_path


def cleanup_vlm_image_copy(path: str | Path, *, output_dir: str | Path = VLM_TEMP_DIR) -> bool:
    """Delete one temporary VLM image copy if it is inside the configured temp directory."""
    candidate = Path(path).resolve()
    root = Path(output_dir).resolve()
    if not _is_within(candidate, root):
        LOGGER.warning("Skipping VLM temp image cleanup outside temp directory: %s", candidate)
        return False
    candidate.unlink(missing_ok=True)
    return True


def shorten_vlm_text(text: str, *, max_chars: int = 360) -> str:
    """Keep model output short enough for WhatsApp alerts."""
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def redact_vlm_text(text: str, base_url: str = "") -> str:
    """Redact VLM URLs from logs/errors before they reach public output."""
    redacted = text
    if base_url:
        redacted = redacted.replace(base_url, "<redacted-vlm-url>")
    return re.sub(r"https?://[^\s'\"<>]+", "<redacted-vlm-url>", redacted)


def _extract_response_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

    for key in ("response", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            choice_message = first.get("message")
            if isinstance(choice_message, dict) and isinstance(choice_message.get("content"), str):
                return choice_message["content"].strip()
            if isinstance(first.get("text"), str):
                return first["text"].strip()
    return ""


def _short_error_body(body: bytes, *, max_chars: int = 500) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
