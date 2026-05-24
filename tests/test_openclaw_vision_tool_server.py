from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openclaw_vision_tool_server import (  # noqa: E402
    VisionTool,
    VisionToolConfig,
    save_base64_image,
    validate_shared_secret,
)
from yorkie_watch.event_state import write_latest_event  # noqa: E402
from yorkie_watch.config import ConfigError  # noqa: E402
from yorkie_watch.vlm_client import VLMResult  # noqa: E402


JPEG_BYTES = b"\xff\xd8\xffplaceholder-jpeg"


class FakeVLMClient:
    def __init__(self, text: str = "A dog-like animal may be visible.") -> None:
        self.text = text
        self.calls: list[tuple[Path, str]] = []

    def describe_image(self, image_path: str | Path, prompt: str) -> VLMResult:
        self.calls.append((Path(image_path), prompt))
        return VLMResult(True, self.text, "", "mock-vlm")


class FakeHomeAssistantClient:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save_snapshot(self, path: str | Path, *, attempts: int = 3, delay_seconds: float = 2.0) -> Path:
        del attempts, delay_seconds
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(JPEG_BYTES)
        self.saved_paths.append(output_path)
        return output_path


def encoded_jpeg() -> str:
    return base64.b64encode(JPEG_BYTES).decode("ascii")


def vision_config(root: Path, *, shared_secret: str = "") -> VisionToolConfig:
    return VisionToolConfig(
        host="127.0.0.1",
        port=8021,
        shared_secret=shared_secret,
        default_prompt="Describe what you can see.",
        latest_event_path=root / "latest_event.json",
        output_dir=root / "vision_tool",
    )


class OpenClawVisionToolTests(unittest.TestCase):
    def test_shared_secret_validation_is_optional_and_strict_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            open_config = vision_config(Path(tmp_dir))
            locked_config = vision_config(Path(tmp_dir), shared_secret="secret-placeholder")

            self.assertTrue(validate_shared_secret({}, open_config))
            self.assertFalse(validate_shared_secret({}, locked_config))
            self.assertFalse(validate_shared_secret({"X-OpenClaw-Secret": "wrong-placeholder"}, locked_config))
            self.assertTrue(validate_shared_secret({"x-openclaw-secret": "secret-placeholder"}, locked_config))

    def test_latest_alert_success_describes_persistent_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "evidence.jpg"
            image_path.write_bytes(JPEG_BYTES)
            write_latest_event(
                image_path=image_path,
                detector_class="dog",
                confidence=0.72,
                region="lower_half",
                state_path=root / "latest_event.json",
            )
            fake_vlm = FakeVLMClient("The latest alert image shows a possible dog.")
            tool = VisionTool(config=vision_config(root), vlm_client_factory=lambda: fake_vlm)

            response = tool.handle("/vision/latest-alert", {"prompt": "Is there a dog?"}, {})

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertEqual(response.payload["source"], "latest_alert")
        self.assertIn("possible dog", str(response.payload["description"]))
        self.assertEqual(fake_vlm.calls[0][0], image_path)
        self.assertEqual(fake_vlm.calls[0][1], "Is there a dog?")

    def test_latest_alert_missing_image_returns_ok_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = root / "missing.jpg"
            write_latest_event(
                image_path=missing,
                detector_class="dog",
                confidence=0.72,
                region="lower_half",
                state_path=root / "latest_event.json",
            )
            tool = VisionTool(config=vision_config(root), vlm_client_factory=lambda: FakeVLMClient())

            response = tool.handle("/vision/latest-alert", {}, {})

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertFalse(response.payload["ok"])
        self.assertIn("does not exist", str(response.payload["error"]))

    def test_camera_snapshot_uses_home_assistant_and_vlm_without_whatsapp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_ha = FakeHomeAssistantClient()
            fake_vlm = FakeVLMClient("Fresh camera snapshot shows no clear dog.")
            tool = VisionTool(
                config=vision_config(root),
                vlm_client_factory=lambda: fake_vlm,
                ha_client_factory=lambda: fake_ha,  # type: ignore[arg-type]
                clock=lambda: 10.123,
            )

            response = tool.handle("/vision/camera-snapshot", {}, {})

            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertTrue(response.payload["ok"])
            self.assertEqual(response.payload["source"], "camera_snapshot")
            self.assertEqual(len(fake_ha.saved_paths), 1)
            self.assertTrue(fake_ha.saved_paths[0].exists())
            self.assertEqual(fake_vlm.calls[0][0], fake_ha.saved_paths[0])

    def test_camera_snapshot_returns_json_when_home_assistant_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = VisionTool(
                config=vision_config(Path(tmp_dir)),
                vlm_client_factory=lambda: FakeVLMClient(),
                ha_client_factory=lambda: (_ for _ in ()).throw(ConfigError("missing placeholder config")),
            )

            response = tool.handle("/vision/camera-snapshot", {}, {})

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertFalse(response.payload["ok"])
        self.assertIn("Could not fetch Home Assistant snapshot", str(response.payload["error"]))

    def test_describe_image_saves_base64_and_calls_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_vlm = FakeVLMClient("The provided image is dark, with uncertainty.")
            tool = VisionTool(
                config=vision_config(root),
                vlm_client_factory=lambda: fake_vlm,
                clock=lambda: 20.456,
            )

            response = tool.handle(
                "/vision/describe-image",
                {"prompt": "What can you see?", "image_base64": encoded_jpeg()},
                {},
            )

            self.assertEqual(response.status, HTTPStatus.OK)
            self.assertTrue(response.payload["ok"])
            self.assertEqual(response.payload["source"], "provided_image")
            self.assertTrue(fake_vlm.calls[0][0].exists())
            self.assertEqual(fake_vlm.calls[0][1], "What can you see?")

    def test_describe_image_returns_json_when_vlm_client_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = VisionTool(
                config=vision_config(Path(tmp_dir)),
                vlm_client_factory=lambda: (_ for _ in ()).throw(RuntimeError("http://private-host-placeholder")),
                clock=lambda: 30.789,
            )

            response = tool.handle("/vision/describe-image", {"image_base64": encoded_jpeg()}, {})

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertFalse(response.payload["ok"])
        self.assertIn("VLM description failed", str(response.payload["error"]))
        self.assertNotIn("private-host-placeholder", str(response.payload["error"]))

    def test_describe_image_rejects_invalid_base64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = VisionTool(config=vision_config(Path(tmp_dir)), vlm_client_factory=lambda: FakeVLMClient())

            response = tool.handle("/vision/describe-image", {"image_base64": "not-valid-base64"}, {})

        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(response.payload["ok"])
        self.assertIn("base64", str(response.payload["error"]))

    def test_save_base64_image_rejects_non_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            encoded = base64.b64encode(b"not-an-image").decode("ascii")

            with self.assertRaisesRegex(ValueError, "JPEG or PNG"):
                save_base64_image(encoded, output_dir=Path(tmp_dir), stem="bad")

    def test_response_payloads_are_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = VisionTool(config=vision_config(Path(tmp_dir)), vlm_client_factory=lambda: FakeVLMClient())
            response = tool.handle("/vision/latest-alert", {}, {})

        json.dumps(response.payload)


if __name__ == "__main__":
    unittest.main()
