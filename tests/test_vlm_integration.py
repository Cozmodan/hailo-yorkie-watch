from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import DogAlertConfig, VLMConfig  # noqa: E402
from yorkie_watch.detector import Detection, DetectionResult  # noqa: E402
from yorkie_watch.event_state import write_latest_event  # noqa: E402
from yorkie_watch.main import _notify_detection_result, run_chat  # noqa: E402
from yorkie_watch.vlm_client import VLMResult, cleanup_vlm_image_copy, create_vlm_image_copy  # noqa: E402


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send_message(
        self,
        message: str,
        *,
        event_type: str = "yorkie_watch_test",
        confidence: float = 0.0,
        attachment_path: str | Path | None = None,
    ) -> bool:
        self.messages.append(
            {
                "message": message,
                "event_type": event_type,
                "confidence": confidence,
                "attachment_path": attachment_path,
            }
        )
        return True


class FakeVLMClient:
    def __init__(self, result: VLMResult) -> None:
        self.result = result
        self.calls: list[tuple[Path, str]] = []

    def describe_image(self, image_path: str | Path, prompt: str) -> VLMResult:
        self.calls.append((Path(image_path), prompt))
        return self.result


def dog_alert_config(evidence_dir: Path) -> DogAlertConfig:
    return DogAlertConfig(
        min_confidence=0.45,
        cooldown_seconds=180.0,
        confirmation_frames=1,
        min_box_area_ratio=0.01,
        save_debug_frames=False,
        evidence_dir=str(evidence_dir),
        image_retention_seconds=3600.0,
        max_evidence_images=100,
    )


def vlm_config(*, enabled: bool = True) -> VLMConfig:
    return VLMConfig(
        enabled=enabled,
        base_url="http://vlm-host-placeholder:8000",
        model="vlm-model-placeholder",
        timeout_seconds=60.0,
        max_image_width=1280,
        prompt="Look at this image and briefly describe whether a dog is visible.",
    )


def make_image(directory: Path, name: str = "snapshot.jpg", *, size: tuple[int, int] = (200, 100)) -> Path:
    image_path = directory / name
    Image.new("RGB", size, color=(80, 90, 100)).save(image_path)
    return image_path


def dog_result(image_path: Path) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend="fake",
        image=str(image_path),
        detections=(Detection(class_name="dog", class_id=16, confidence=0.72, bbox=(20, 20, 120, 80)),),
        matched=True,
        matched_reason="dog matched",
    )


class VLMIntegrationTests(unittest.TestCase):
    def test_vlm_disabled_does_not_call_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(VLMResult(True, "dog visible", "", "mock"))

            with redirect_stdout(StringIO()):
                sent = _notify_detection_result(
                    image_path,
                    dog_result(image_path),
                    notifier=notifier,  # type: ignore[arg-type]
                    dog_alert_config=dog_alert_config(root / "evidence"),
                    vlm_config=vlm_config(enabled=False),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    latest_event_path=root / "latest_event.json",
                )

        self.assertTrue(sent)
        self.assertEqual(vlm_client.calls, [])
        self.assertNotIn("VLM:", str(notifier.messages[0]["message"]))

    def test_vlm_failure_does_not_block_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(VLMResult(False, "", "service unavailable", "mock"))

            with redirect_stdout(StringIO()):
                sent = _notify_detection_result(
                    image_path,
                    dog_result(image_path),
                    notifier=notifier,  # type: ignore[arg-type]
                    dog_alert_config=dog_alert_config(root / "evidence"),
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    latest_event_path=root / "latest_event.json",
                )

        self.assertTrue(sent)
        self.assertEqual(len(vlm_client.calls), 1)
        self.assertEqual(len(notifier.messages), 1)
        self.assertNotIn("VLM:", str(notifier.messages[0]["message"]))

    def test_vlm_success_appends_summary_to_alert_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(
                VLMResult(True, "A small dog-like animal is visible near the doorway.", "", "mock")
            )

            with redirect_stdout(StringIO()):
                sent = _notify_detection_result(
                    image_path,
                    dog_result(image_path),
                    notifier=notifier,  # type: ignore[arg-type]
                    dog_alert_config=dog_alert_config(root / "evidence"),
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    latest_event_path=root / "latest_event.json",
                )

        self.assertTrue(sent)
        message = str(notifier.messages[0]["message"])
        self.assertIn("Detector: dog confidence 0.72 >= 0.45", message)
        self.assertIn("VLM: A small dog-like animal is visible near the doorway.", message)

    def test_latest_event_json_is_written_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            state_path = root / "latest_event.json"
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(VLMResult(True, "Dog visible with some uncertainty.", "", "mock"))

            with redirect_stdout(StringIO()):
                _notify_detection_result(
                    image_path,
                    dog_result(image_path),
                    notifier=notifier,  # type: ignore[arg-type]
                    dog_alert_config=dog_alert_config(root / "evidence"),
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    latest_event_path=state_path,
                )

            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["detector_class"], "dog")
        self.assertEqual(payload["confidence"], 0.72)
        self.assertEqual(payload["vlm_summary"], "Dog visible with some uncertainty.")
        text = json.dumps(payload)
        self.assertNotIn("token", text.lower())
        self.assertNotIn("vlm-host-placeholder", text)
        self.assertNotIn("whatsapp", text.lower())

    def test_chat_mode_returns_vlm_answer_for_latest_image_without_sending_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            state_path = write_latest_event(
                image_path=image_path,
                detector_class="dog",
                confidence=0.72,
                region="lower_half",
                vlm_summary="Earlier summary.",
                state_path=root / "latest_event.json",
            )
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(VLMResult(True, "It looks like a dog, but I am not fully certain.", "", "mock"))
            output = StringIO()

            with redirect_stdout(output):
                returncode = run_chat(
                    "Was that actually my dog or just a shadow?",
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    notifier=notifier,  # type: ignore[arg-type]
                    latest_event_path=state_path,
                )

        self.assertEqual(returncode, 0)
        self.assertIn("It looks like a dog", output.getvalue())
        self.assertEqual(notifier.messages, [])
        self.assertIn("most recent Yorkie Watch alert", vlm_client.calls[0][1])
        self.assertIn("Was that actually my dog", vlm_client.calls[0][1])

    def test_chat_mode_handles_missing_latest_image_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_path = root / "latest_event.json"
            state_path.write_text(
                json.dumps({"image_path": str(root / "missing.jpg"), "detector_class": "dog"}),
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                returncode = run_chat(
                    "What was in the image?",
                    vlm_config=vlm_config(),
                    vlm_client=FakeVLMClient(VLMResult(True, "unused", "", "mock")),  # type: ignore[arg-type]
                    latest_event_path=state_path,
                )

        self.assertEqual(returncode, 1)
        self.assertIn("image is unavailable", output.getvalue())
        self.assertIn("missing.jpg", output.getvalue())
        self.assertIn("Wait for the next real alert", output.getvalue())

    def test_vlm_image_resize_helper_does_not_overwrite_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root, size=(2000, 1000))
            with Image.open(image_path) as original:
                original_size = original.size
            output_path = create_vlm_image_copy(image_path, max_width=128, output_dir=root / "vlm_tmp")
            with Image.open(output_path) as resized:
                resized_size = resized.size

            with Image.open(image_path) as original_after:
                self.assertEqual(original_after.size, original_size)
            self.assertLessEqual(resized_size[0], 128)
            self.assertNotEqual(output_path, image_path)
            self.assertTrue(cleanup_vlm_image_copy(output_path, output_dir=root / "vlm_tmp"))

    def test_openclaw_chat_reply_is_sent_only_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = make_image(root)
            state_path = write_latest_event(
                image_path=image_path,
                detector_class="dog",
                confidence=0.72,
                region="lower_half",
                state_path=root / "latest_event.json",
            )
            notifier = FakeNotifier()
            vlm_client = FakeVLMClient(VLMResult(True, "The latest alert likely shows a dog.", "", "mock"))

            with redirect_stdout(StringIO()):
                run_chat(
                    "What is there?",
                    send_reply=False,
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    notifier=notifier,  # type: ignore[arg-type]
                    latest_event_path=state_path,
                )
            with redirect_stdout(StringIO()):
                returncode = run_chat(
                    "What is there?",
                    send_reply=True,
                    vlm_config=vlm_config(),
                    vlm_client=vlm_client,  # type: ignore[arg-type]
                    notifier=notifier,  # type: ignore[arg-type]
                    latest_event_path=state_path,
                )

        self.assertEqual(returncode, 0)
        self.assertEqual(len(notifier.messages), 1)
        self.assertEqual(notifier.messages[0]["event_type"], "chat_reply")


if __name__ == "__main__":
    unittest.main()
