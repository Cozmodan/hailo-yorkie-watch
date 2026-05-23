from __future__ import annotations

import json
import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openclaw_inbound_bridge import (  # noqa: E402
    InboundConfig,
    InboundMessage,
    handle_inbound,
    is_allowed_sender,
    parse_allowed_senders,
    parse_inbound_payload,
    route_command,
    validate_shared_secret,
)
from yorkie_watch.event_state import write_latest_event  # noqa: E402


class OpenClawInboundBridgeTests(unittest.TestCase):
    def test_payload_parsing_accepts_flexible_fields(self) -> None:
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

    def test_shared_secret_validation_is_optional_and_strict_when_set(self) -> None:
        open_config = InboundConfig(shared_secret="")
        locked_config = InboundConfig(shared_secret="secret-placeholder")

        self.assertTrue(validate_shared_secret({}, open_config))
        self.assertFalse(validate_shared_secret({}, locked_config))
        self.assertFalse(validate_shared_secret({"X-OpenClaw-Secret": "wrong-placeholder"}, locked_config))
        self.assertTrue(validate_shared_secret({"x-openclaw-secret": "secret-placeholder"}, locked_config))

    def test_allowed_sender_filtering_is_optional_and_normalized(self) -> None:
        open_config = InboundConfig(allowed_senders=())
        locked_config = InboundConfig(allowed_senders=parse_allowed_senders(" Sender-Placeholder , other-placeholder "))

        self.assertTrue(is_allowed_sender("any-sender-placeholder", open_config))
        self.assertTrue(is_allowed_sender("sender-placeholder", locked_config))
        self.assertFalse(is_allowed_sender("blocked-placeholder", locked_config))

    def test_rejects_empty_messages(self) -> None:
        response = handle_inbound(
            {"sender": "sender-placeholder", "message": "   "},
            {},
            config=InboundConfig(),
        )

        self.assertEqual(response.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("empty", str(response.payload["error"]))

    def test_command_routing_status_includes_latest_event(self) -> None:
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

            response = route_command(InboundMessage("sender-placeholder", "status"), config=config)

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertIn("Last alert: dog", str(response.payload["message"]))
        self.assertIn("Dog-like shape", str(response.payload["message"]))

    def test_command_routing_pause_and_resume_alerts_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            flag_path = Path(tmp_dir) / "runtime" / "alerts_paused.flag"
            config = InboundConfig(pause_flag_path=flag_path)

            pause_response = route_command(InboundMessage("sender-placeholder", "pause alerts"), config=config)
            self.assertTrue(flag_path.exists())
            resume_response = route_command(InboundMessage("sender-placeholder", "resume alerts"), config=config)

        self.assertEqual(pause_response.status, HTTPStatus.OK)
        self.assertEqual(resume_response.status, HTTPStatus.OK)
        self.assertFalse(flag_path.exists())

    def test_command_routing_chat_reuses_existing_flow_and_mocks_outbound(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_chat_runner(question: str, **kwargs: object) -> int:
            calls.append({"question": question, **kwargs})
            print("Mocked VLM answer sent through OpenClaw.")
            return 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = InboundConfig(latest_event_path=Path(tmp_dir) / "latest_event.json")
            response = route_command(
                InboundMessage("sender-placeholder", "is that a dog"),
                config=config,
                chat_runner=fake_chat_runner,
            )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertEqual(calls[0]["question"], "is that a dog")
        self.assertTrue(calls[0]["send_reply"])
        self.assertIn("Mocked VLM answer", str(response.payload["message"]))

    def test_missing_latest_event_handling_returns_no_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = InboundConfig(latest_event_path=Path(tmp_dir) / "missing_latest_event.json")
            response = route_command(InboundMessage("sender-placeholder", "last alert"), config=config)

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertFalse(response.payload["ok"])
        self.assertIn("No latest Yorkie Watch event", str(response.payload["message"]))

    def test_http_response_payload_is_json_serializable(self) -> None:
        response = handle_inbound(
            {"phone": "sender-placeholder", "text": "unknown command"},
            {},
            config=InboundConfig(),
        )

        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertTrue(response.payload["ok"])
        self.assertIn("known", str(response.payload["message"]).lower())
        json.dumps(response.payload)


if __name__ == "__main__":
    unittest.main()