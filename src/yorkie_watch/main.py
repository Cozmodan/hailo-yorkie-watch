from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from .config import ConfigError, load_detector_config
from .detector import COCO_DOG_CLASS_ID, DetectionResult, DetectorError, create_detector, print_result
from .ha_client import HomeAssistantClient, HomeAssistantError
from .openclaw_client import OpenClawClient

LOGGER = logging.getLogger(__name__)
SNAPSHOT_DIR = Path("data") / "snapshots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hailo Yorkie Watch plumbing CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Fetch one Home Assistant snapshot and save it locally.")
    mode.add_argument("--test-openclaw", action="store_true", help="Send one test event to OpenClaw.")
    mode.add_argument("--test-detect", metavar="IMAGE", help="Run detector once against an existing image.")
    mode.add_argument(
        "--what-see",
        action="store_true",
        help="Fetch one snapshot, run detection, and send a WhatsApp summary with the snapshot.",
    )
    return parser


def run_once() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
    client = HomeAssistantClient.from_env()
    saved_path = client.save_snapshot(output_path, attempts=3, delay_seconds=2.0)
    print(f"Saved snapshot to {saved_path} ({saved_path.stat().st_size} bytes)")
    detector_config = load_detector_config()
    if detector_config.enabled:
        run_detection_and_maybe_notify(saved_path, detector=create_detector(detector_config))
    return 0


def run_test_openclaw() -> int:
    client = OpenClawClient.from_env()
    if client.notify_mode == "disabled":
        print("OpenClaw notifications are disabled; no test event sent.")
        return 0

    success = client.send_message("Test alert from Hailo Yorkie Watch")
    if success:
        print(f"OpenClaw test event sent successfully via {client.notify_mode}.")
        return 0

    print("OpenClaw test event failed. Check logs and OpenClaw connectivity.")
    return 1


def run_test_detect(image_path: str) -> int:
    detector = create_detector(load_detector_config())
    try:
        result = detector.detect(Path(image_path))
    except DetectorError as exc:
        LOGGER.error("Detector failed: %s", exc)
        return 1
    print_result(result)
    return 0 if result.ok else 1


def run_what_see() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SNAPSHOT_DIR / f"what_see_{timestamp}.jpg"
    client = HomeAssistantClient.from_env()
    saved_path = client.save_snapshot(output_path, attempts=3, delay_seconds=2.0)
    print(f"Saved snapshot to {saved_path} ({saved_path.stat().st_size} bytes)")

    detector = create_detector(load_detector_config())
    try:
        result = detector.detect(saved_path)
    except DetectorError as exc:
        LOGGER.error("Detector failed for %s: %s", saved_path, exc)
        message = f"What I see: detector failed: {exc}"
        confidence = 0.0
    else:
        print_result(result)
        message = f"What I see: {_detection_summary(result)}"
        confidence = _matched_confidence(result)

    notifier = OpenClawClient.from_env()
    if notifier.send_message(
        message,
        event_type="what_see",
        confidence=confidence,
        attachment_path=saved_path,
    ):
        print("What-see response sent.")
        return 0

    print("What-see response failed. Check logs and OpenClaw connectivity.")
    return 1


def run_detection_and_maybe_notify(
    image_path: Path,
    *,
    detector: object,
    notifier: OpenClawClient | None = None,
) -> bool:
    """Run detection and send a notification only when the alert condition matches."""
    try:
        result = detector.detect(image_path)  # type: ignore[attr-defined]
    except DetectorError as exc:
        LOGGER.error("Detector failed for %s: %s", image_path, exc)
        print("No alert sent: detector failed.")
        return False

    print_result(result)
    if not result.matched:
        print(f"No alert sent: {result.matched_reason}.")
        return False

    notifier = notifier or OpenClawClient.from_env()
    confidence = _matched_confidence(result)
    message = f"Dog detected by Hailo Yorkie Watch: {result.matched_reason}"
    if notifier.send_message(message, event_type="dog_detected", confidence=confidence, attachment_path=image_path):
        print("Alert sent: dog detected.")
        return True

    print("Alert condition matched, but OpenClaw notification failed.")
    return False


def _matched_confidence(result: DetectionResult) -> float:
    return max(
        (
            detection.confidence
            for detection in result.detections
            if detection.class_name == "dog" or detection.class_id == COCO_DOG_CLASS_ID
        ),
        default=0.0,
    )


def _detection_summary(result: DetectionResult) -> str:
    if not result.ok:
        return f"detector failed: {result.error or result.matched_reason}"
    if not result.detections:
        return result.matched_reason
    detections = sorted(result.detections, key=lambda detection: detection.confidence, reverse=True)
    parts = [
        f"{detection.class_name or 'unknown'} {detection.confidence:.2f}"
        for detection in detections[:5]
    ]
    summary = ", ".join(parts)
    if result.matched:
        return f"{result.matched_reason}. Top detections: {summary}"
    return f"{result.matched_reason}. Top detections: {summary}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    try:
        if args.once:
            return run_once()
        if args.test_openclaw:
            return run_test_openclaw()
        if args.test_detect:
            return run_test_detect(args.test_detect)
        if args.what_see:
            return run_what_see()
    except (ConfigError, HomeAssistantError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
