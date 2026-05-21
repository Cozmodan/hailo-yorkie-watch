from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.detector import Detection, DetectionResult, DetectorError, evaluate_detections
from yorkie_watch.main import run_detection_and_maybe_notify


class FakeDetector:
    def __init__(self, result: DetectionResult | None = None, error: str = "") -> None:
        self.result = result
        self.error = error

    def detect(self, image_path: Path) -> DetectionResult:
        if self.error:
            raise DetectorError(self.error)
        if self.result is None:
            raise AssertionError("FakeDetector result was not configured.")
        return self.result


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


class DetectionDecisionTests(unittest.TestCase):
    def test_dog_above_threshold_sends_notification(self) -> None:
        result = evaluate_detections(
            image_path="snapshot.jpg",
            backend="mock",
            detections=(
                Detection(class_name="person", class_id=0, confidence=0.99),
                Detection(class_name="dog", class_id=16, confidence=0.72),
            ),
            target_classes=("dog",),
            confidence_threshold=0.35,
        )
        notifier = FakeNotifier()

        with redirect_stdout(StringIO()), patch.dict("os.environ", {"YORKIE_ENABLE_CROP_SCAN": "0"}):
            sent = run_detection_and_maybe_notify(
                Path("snapshot.jpg"),
                detector=FakeDetector(result),
                notifier=notifier,  # type: ignore[arg-type]
            )

        self.assertTrue(sent)
        self.assertEqual(len(notifier.messages), 1)
        self.assertEqual(notifier.messages[0]["event_type"], "dog_detected")
        self.assertEqual(notifier.messages[0]["confidence"], 0.72)
        self.assertEqual(notifier.messages[0]["attachment_path"], Path("snapshot.jpg"))

    def test_no_dog_does_not_send_notification(self) -> None:
        result = evaluate_detections(
            image_path="snapshot.jpg",
            backend="mock",
            detections=(Detection(class_name="person", class_id=0, confidence=0.99),),
            target_classes=("dog",),
            confidence_threshold=0.35,
        )
        notifier = FakeNotifier()

        with redirect_stdout(StringIO()), patch.dict("os.environ", {"YORKIE_ENABLE_CROP_SCAN": "0"}):
            sent = run_detection_and_maybe_notify(
                Path("snapshot.jpg"),
                detector=FakeDetector(result),
                notifier=notifier,  # type: ignore[arg-type]
            )

        self.assertFalse(sent)
        self.assertEqual(notifier.messages, [])

    def test_detector_failure_does_not_send_notification(self) -> None:
        notifier = FakeNotifier()

        with (
            redirect_stdout(StringIO()),
            patch.dict("os.environ", {"YORKIE_ENABLE_CROP_SCAN": "0"}),
            self.assertLogs("yorkie_watch.main", level="ERROR"),
        ):
            sent = run_detection_and_maybe_notify(
                Path("snapshot.jpg"),
                detector=FakeDetector(error="mock detector failed"),
                notifier=notifier,  # type: ignore[arg-type]
            )

        self.assertFalse(sent)
        self.assertEqual(notifier.messages, [])


if __name__ == "__main__":
    unittest.main()
