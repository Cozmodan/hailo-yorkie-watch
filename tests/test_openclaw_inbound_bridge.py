from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openclaw_inbound_bridge import (  # noqa: E402
    InboundConfig,
    InboundMessage,
    handle_inbound,
    is_allowed_sender,
    make_handler,
    parse_allowed_senders,
    parse_inbound_payload,
    route_command,
    validate_shared_secret,
)
from yorkie_watch.event_state import write_latest_event  # noqa: E402
from yorkie_watch.openclaw_vision_tools import VisionToolResult  # noqa: E402


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def camera_snapshot(self, prompt: str) -> VisionToolResult:
        self.calls.append(("camera_snapshot", prompt))
        return VisionToolResult(
            ok=True,
            source="camera_snapshot",
            description="The current camera view shows a quiet doorway.",
            error="",
            image_path="",
            status_code=200,
            payload={"ok": True, "source": "camera_snapshot", "description": "The current camera view shows a quiet doorway."},
        )

    def latest_alert(self, prompt: str) -> VisionToolResult:
        self.calls.append(("latest_alert", prompt))
        return VisionToolResult(
            ok=True,
            source="latest_alert",
            description="The latest alert looks like a real dog with moderate uncertainty.",
            error="",
            image_path="",
            status_code=200,
            payload={"ok": True, "source": "latest_alert", "description": "The latest alert looks like a real dog."},
        )


class FakeNotifier:
    def __init__(self) -> None:
        self.whatsapp_target = "<configured-target-placeholder>"
        self.sent_messages: list[tuple[str, str]] = []

    def send_message(self, message: str, *, event_type: str = "yorkie_watch_test", **kwargs: object) -> bool:
        del event_type, kwargs
        self.sent_messages.append((self.whatsapp_target, message))
        return True


def bridge_config(root: Path, **kwargs: object) -> InboundConfig:
    return InboundConfig(
        shared_secret=str(kwargs.get("shared_secret", "")),
        allowed_senders=kwargs.get("allowed_senders", ()),  # type: ignore[arg-type]
        latest_event_path=root / "latest_event.json",
        pause_flag_path=root / "runtime" / "alerts_paused.flag",
        smoke_send=bool(kwargs.get("smoke_send", False)),
    )


class OpenClawInboundBridgeTests(unittest.TestCase):
    def test_health_endpoint_returns_ok_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = bridge_config(Path(tmp_dir))
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config=config))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(payload, {"ok": True, "status": "ok"})

    def test_payload_parsing_accepts_nested_flexible_fields(self) -> None:
        inbound = parse_inbound_payload(
            {
                "data": {
                    "from": " sender-placeholder ",
                    "body": " status ",
                    "timestamp": "2026-05-23T23:22:00",
                }
            }
        )

        self.assertEqual(inbound.sender, "sender-placeholder")
        self.assertEqual(inbound.message, "status")
        self.assertEqual(inbound.timestamp, "2026-05-23T23:22:00")

    def test_shared_secret_rejects_and_accepts(self) -> None:
        open_config = InboundConfig(shared_secret="")
        locked_config = InboundConfig(shared_secret="secret-placeholder")

        self.assertTrue(validate_shared_secret({}, open_config))
        self.assertFalse(validate_shared_secret({}, locked_config))
        self.assertFalse(validate_shared_secret({"X-OpenClaw-Secret": "wrong-placeholder"}, locked_config))
        self.assertTrue(validate_shared_secret({"x-openclaw-secret": "secret-placeholder"}, locked_config))

    def test_allowed_sender_rejects_and_accepts(self) -> None:
        open_config = InboundConfig(allowed_senders=())
        locked_config = InboundConfig(allowed_senders=parse_allowed_senders("+61000000000, sender-placeholder"))

        self.assertTrue(is_allowed_sender("any-sender-placeholder", open_config))
        self.assertTrue(is_allowed_sender("+61 000 000 000", locked_config))
        self.assertTrue(is_allowed_sender("sender-placeholder", locked_config))
        self.assertFalse(is_allowed_sender("blocked-placeholder", locked_config))

    def test_empty_message_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "sender-placeholder", "message": "   "},
                {},
                config=bridge_config(Path(tmp_dir)),
                vision_client_factory=FakeVisionClient,
                notifier_factory=FakeNotifier,
            )

        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(response.payload["ok"])
        self.assertFalse(response.payload["reply_sent"])

    def test_unknown_sender_returns_403(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "blocked-placeholder", "message": "status"},
                {},
                config=bridge_config(Path(tmp_dir), allowed_senders=("sender-placeholder",)),
                vision_client_factory=FakeVisionClient,
                notifier_factory=FakeNotifier,
            )

        self.assertEqual(response.status, HTTPStatus.FORBIDDEN)
        self.assertFalse(response.payload["ok"])

    def test_missing_sender_uses_configured_target(self) -> None:
        fake_notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"message": "status"},
                {},
                config=bridge_config(Path(tmp_dir), allowed_senders=("sender-placeholder",)),
                vision_client_factory=FakeVisionClient,
                notifier_factory=lambda: fake_notifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["reply_sent"])
        self.assertEqual(fake_notifier.sent_messages[0][0], "<configured-target-placeholder>")

    def test_what_do_you_see_routes_to_camera_snapshot_and_sends_reply(self) -> None:
        fake_vision = FakeVisionClient()
        fake_notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "sender-placeholder", "message": "What do you see?"},
                {},
                config=bridge_config(Path(tmp_dir)),
                vision_client_factory=lambda: fake_vision,
                notifier_factory=lambda: fake_notifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertEqual(response.payload["route"], "camera_snapshot")
        self.assertTrue(response.payload["reply_sent"])
        self.assertEqual(fake_vision.calls, [("camera_snapshot", "What do you see?")])
        self.assertEqual(fake_notifier.sent_messages[0][0], "sender-placeholder")
        self.assertIn("quiet doorway", fake_notifier.sent_messages[0][1])

    def test_last_alert_question_routes_to_latest_alert(self) -> None:
        fake_vision = FakeVisionClient()
        fake_notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "sender-placeholder", "message": "Was the last alert real?"},
                {},
                config=bridge_config(Path(tmp_dir)),
                vision_client_factory=lambda: fake_vision,
                notifier_factory=lambda: fake_notifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertEqual(response.payload["route"], "latest_alert")
        self.assertEqual(fake_vision.calls, [("latest_alert", "Was the last alert real?")])
        self.assertIn("real dog", fake_notifier.sent_messages[0][1])

    def test_was_that_a_dog_routes_to_latest_alert(self) -> None:
        fake_vision = FakeVisionClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = route_command(
                InboundMessage("sender-placeholder", "Was that a dog?"),
                config=bridge_config(Path(tmp_dir)),
                vision_client_factory=lambda: fake_vision,  # type: ignore[return-value]
            )

        self.assertTrue(response.ok)
        self.assertEqual(response.route, "latest_alert")

    def test_status_includes_stack_hint_and_latest_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "snapshot.jpg"
            image_path.write_bytes(b"placeholder image bytes")
            latest_event_path = write_latest_event(
                image_path=image_path,
                detector_class="dog",
                confidence=0.72,
                region="lower_half",
                vlm_summary="Dog-like shape visible.",
                state_path=root / "latest_event.json",
            )
            config = InboundConfig(latest_event_path=latest_event_path, pause_flag_path=root / "alerts_paused.flag")

            response = route_command(
                InboundMessage("sender-placeholder", "status"),
                config=config,
                vision_client_factory=FakeVisionClient,
            )

        self.assertTrue(response.ok)
        self.assertEqual(response.route, "status")
        self.assertIn("scripts/yorkie_stack_tmux.sh status", response.message)
        self.assertIn("Last alert: dog", response.message)

    def test_pause_and_resume_alerts_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            flag_path = Path(tmp_dir) / "runtime" / "alerts_paused.flag"
            config = InboundConfig(pause_flag_path=flag_path)

            pause_response = route_command(
                InboundMessage("sender-placeholder", "pause alerts"),
                config=config,
                vision_client_factory=FakeVisionClient,
            )
            self.assertTrue(flag_path.exists())
            resume_response = route_command(
                InboundMessage("sender-placeholder", "resume alerts"),
                config=config,
                vision_client_factory=FakeVisionClient,
            )

        self.assertTrue(pause_response.ok)
        self.assertTrue(resume_response.ok)
        self.assertFalse(flag_path.exists())

    def test_smoke_test_mode_does_not_send_whatsapp_unless_enabled(self) -> None:
        fake_notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "test", "message": "status"},
                {"X-OpenClaw-Smoke-Test": "1"},
                config=bridge_config(Path(tmp_dir), allowed_senders=("sender-placeholder",), smoke_send=False),
                vision_client_factory=FakeVisionClient,
                notifier_factory=lambda: fake_notifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertFalse(response.payload["reply_sent"])
        self.assertEqual(fake_notifier.sent_messages, [])

    def test_smoke_test_mode_sends_whatsapp_when_enabled(self) -> None:
        fake_notifier = FakeNotifier()
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"sender": "test", "message": "status"},
                {"X-OpenClaw-Smoke-Test": "1"},
                config=bridge_config(Path(tmp_dir), allowed_senders=("sender-placeholder",), smoke_send=True),
                vision_client_factory=FakeVisionClient,
                notifier_factory=lambda: fake_notifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["reply_sent"])
        self.assertEqual(fake_notifier.sent_messages[0][0], "test")

    def test_secrets_and_phone_numbers_are_redacted_in_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = bridge_config(
                Path(tmp_dir),
                shared_secret="secret-placeholder",
                allowed_senders=parse_allowed_senders("+61000000000"),
            )
            with self.assertLogs("openclaw_inbound_bridge", level="INFO") as captured:
                response = handle_inbound(
                    {"sender": "+61000000000", "message": "status secret-placeholder +61000000000"},
                    {"X-OpenClaw-Secret": "secret-placeholder"},
                    config=config,
                    vision_client_factory=FakeVisionClient,
                    notifier_factory=FakeNotifier,
                )

        output = "\n".join(captured.output)
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertNotIn("secret-placeholder", output)
        self.assertNotIn("+61000000000", output)
        self.assertIn("<redacted-inbound-secret>", output)
        self.assertIn("<redacted-sender>", output)

    def test_response_payload_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = handle_inbound(
                {"phone": "sender-placeholder", "text": "unknown command"},
                {},
                config=bridge_config(Path(tmp_dir)),
                vision_client_factory=FakeVisionClient,
                notifier_factory=FakeNotifier,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertFalse(response.payload["ok"])
        self.assertEqual(response.payload["route"], "unknown")
        json.dumps(response.payload)


if __name__ == "__main__":
    unittest.main()
