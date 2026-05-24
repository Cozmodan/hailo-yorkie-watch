from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import YorkieVisionConfig  # noqa: E402
from yorkie_watch.openclaw_vision_tools import (  # noqa: E402
    OpenClawVisionToolClient,
    VisionToolResult,
    format_vision_reply,
    handle_whatsapp_vision_message,
    select_vision_route,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def vision_config(*, secret: str = "secret-placeholder") -> YorkieVisionConfig:
    return YorkieVisionConfig(
        base_url="http://pi-vision-placeholder:8021",
        shared_secret=secret,
        timeout_seconds=180.0,
    )


class OpenClawVisionToolsTests(unittest.TestCase):
    def test_camera_questions_route_to_camera_snapshot(self) -> None:
        self.assertEqual(select_vision_route("What do you see?"), "camera_snapshot")
        self.assertEqual(select_vision_route("Can you check the camera?"), "camera_snapshot")
        self.assertEqual(select_vision_route("Is the Yorkie there?"), "camera_snapshot")

    def test_last_alert_question_routes_to_latest_alert(self) -> None:
        self.assertEqual(select_vision_route("Was the last alert real?"), "latest_alert")
        self.assertEqual(select_vision_route("Could that previous dog detection be a false trigger?"), "latest_alert")

    def test_camera_snapshot_posts_to_pi_with_secret_header(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_opener(request: object, timeout: float) -> FakeResponse:
            calls.append(
                {
                    "url": request.full_url,  # type: ignore[attr-defined]
                    "timeout": timeout,
                    "secret": request.get_header("X-openclaw-secret"),  # type: ignore[attr-defined]
                    "content_type": request.get_header("Content-type"),  # type: ignore[attr-defined]
                    "body": json.loads(request.data.decode("utf-8")),  # type: ignore[attr-defined]
                }
            )
            return FakeResponse({"ok": True, "source": "camera_snapshot", "description": "A doorway is visible."})

        client = OpenClawVisionToolClient(vision_config(), opener=fake_opener)

        result = client.camera_snapshot("What do you see?")

        self.assertTrue(result.ok)
        self.assertEqual(result.description, "A doorway is visible.")
        self.assertEqual(calls[0]["url"], "http://pi-vision-placeholder:8021/vision/camera-snapshot")
        self.assertEqual(calls[0]["timeout"], 180.0)
        self.assertEqual(calls[0]["secret"], "secret-placeholder")
        self.assertEqual(calls[0]["content_type"], "application/json")
        self.assertEqual(calls[0]["body"], {"prompt": "What do you see?"})

    def test_latest_alert_posts_to_pi_without_secret_header_when_unset(self) -> None:
        calls: list[str | None] = []

        def fake_opener(request: object, timeout: float) -> FakeResponse:
            del timeout
            calls.append(request.get_header("X-openclaw-secret"))  # type: ignore[attr-defined]
            self.assertEqual(request.full_url, "http://pi-vision-placeholder:8021/vision/latest-alert")  # type: ignore[attr-defined]
            return FakeResponse({"ok": True, "source": "latest_alert", "description": "The alert looks plausible."})

        client = OpenClawVisionToolClient(vision_config(secret=""), opener=fake_opener)

        result = client.latest_alert("Was the last alert real?")

        self.assertTrue(result.ok)
        self.assertEqual(result.description, "The alert looks plausible.")
        self.assertEqual(calls, [None])

    def test_handle_whatsapp_message_formats_reply_and_marks_sent(self) -> None:
        sent: list[str] = []
        client = OpenClawVisionToolClient(
            vision_config(),
            opener=lambda request, timeout: FakeResponse(  # noqa: ARG005
                {"ok": True, "source": "camera_snapshot", "description": "A small dog-like animal is visible."}
            ),
        )

        interaction = handle_whatsapp_vision_message(
            "Is the Yorkie there?",
            client=client,
            send_reply=sent.append,
        )

        self.assertEqual(interaction.route, "camera_snapshot")
        self.assertTrue(interaction.sent)
        self.assertEqual(sent, ["A small dog-like animal is visible."])

    def test_handle_whatsapp_message_logs_route_status_and_reply_without_secret(self) -> None:
        sent: list[str] = []
        client = OpenClawVisionToolClient(
            vision_config(),
            opener=lambda request, timeout: FakeResponse(  # noqa: ARG005
                {"ok": True, "source": "camera_snapshot", "description": "The camera view is quiet."}
            ),
        )

        with self.assertLogs("yorkie_watch.openclaw_vision_tools", level="INFO") as captured:
            interaction = handle_whatsapp_vision_message(
                "What do you see secret-placeholder?",
                client=client,
                send_reply=sent.append,
            )

        output = "\n".join(captured.output)
        self.assertEqual(interaction.route, "camera_snapshot")
        self.assertIn("Incoming WhatsApp text received", output)
        self.assertIn("Vision route selected: camera_snapshot", output)
        self.assertIn("HTTP status from Pi: 200", output)
        self.assertIn("Reply sent to WhatsApp", output)
        self.assertNotIn("secret-placeholder", output)

    def test_missing_pi_response_produces_friendly_whatsapp_error(self) -> None:
        def failing_opener(request: object, timeout: float) -> FakeResponse:
            del request, timeout
            raise URLError("connection refused")

        client = OpenClawVisionToolClient(vision_config(), opener=failing_opener)

        interaction = handle_whatsapp_vision_message("What do you see?", client=client)

        self.assertEqual(interaction.route, "camera_snapshot")
        self.assertIsNotNone(interaction.result)
        self.assertFalse(interaction.result.ok)  # type: ignore[union-attr]
        self.assertIn("could not get a clear vision answer", interaction.reply_text.lower())
        self.assertNotIn("{", interaction.reply_text)

    def test_debug_mode_is_required_for_raw_json(self) -> None:
        result = VisionToolResult(
            ok=False,
            source="camera_snapshot",
            description="",
            error="temporary failure",
            image_path="",
            status_code=200,
            payload={"ok": False, "source": "camera_snapshot", "error": "temporary failure"},
        )

        normal = format_vision_reply(result)
        debug = format_vision_reply(result, debug=True)

        self.assertNotIn('"source"', normal)
        self.assertIn('"source"', debug)

    def test_secret_is_redacted_from_logs(self) -> None:
        logger = logging.getLogger("yorkie_watch.openclaw_vision_tools")

        def failing_opener(request: object, timeout: float) -> FakeResponse:
            del request, timeout
            raise URLError("secret-placeholder was rejected by http://pi-vision-placeholder:8021/private")

        client = OpenClawVisionToolClient(vision_config(), opener=failing_opener)

        with self.assertLogs(logger, level="WARNING") as captured:
            result = client.camera_snapshot()

        output = "\n".join(captured.output)
        self.assertFalse(result.ok)
        self.assertNotIn("secret-placeholder", result.error)
        self.assertNotIn("pi-vision-placeholder", result.error)
        self.assertNotIn("secret-placeholder", output)
        self.assertNotIn("pi-vision-placeholder", output)
        self.assertIn("<redacted-vision-secret>", output)
        self.assertIn("<redacted-yorkie-vision-host>", output)


if __name__ == "__main__":
    unittest.main()
