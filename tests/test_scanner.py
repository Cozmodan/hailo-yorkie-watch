from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ScanConfig  # noqa: E402
from yorkie_watch.detector import Detection, DetectionResult  # noqa: E402
from yorkie_watch.scanner import scan_confirmed_snapshots, scan_image, scanner_summary  # noqa: E402


def scan_config(**overrides: object) -> ScanConfig:
    values = {
        "night_mode": "auto",
        "scan_tiles": "2x2",
        "enable_crop_scan": True,
        "enable_person_roi_scan": True,
        "full_frame_dog_confidence": 0.35,
        "crop_dog_confidence": 0.20,
        "person_confidence": 0.35,
        "confirm_frames": 2,
        "confirm_interval_seconds": 1.0,
        "max_crops_per_image": 8,
        "save_debug_crops": True,
    }
    values.update(overrides)
    return ScanConfig(**values)  # type: ignore[arg-type]


def result(image_path: Path, detections: tuple[Detection, ...]) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend="fake",
        image=str(image_path),
        detections=detections,
        matched=False,
        matched_reason="fake result",
    )


class PathAwareDetector:
    def __init__(self, detections_by_name: dict[str, tuple[Detection, ...]]) -> None:
        self.detections_by_name = detections_by_name
        self.calls: list[Path] = []

    def detect(self, image_path: Path) -> DetectionResult:
        self.calls.append(image_path)
        detections = self.detections_by_name.get(image_path.stem, ())
        return result(image_path, detections)


class ScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image_path = self.root / "snapshot.jpg"
        Image.new("RGB", (200, 100), color=(5, 10, 15)).save(self.image_path)
        self.crop_dir = self.root / "debug_crops"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_frame_dog_match_preserves_existing_alert_behavior(self) -> None:
        detector = PathAwareDetector(
            {
                "snapshot": (Detection(class_name="dog", class_id=16, confidence=0.72),),
            }
        )

        frame_scan = scan_image(
            self.image_path,
            detector=detector,
            config=scan_config(),
            debug_crop_dir=self.crop_dir,
        )

        self.assertTrue(frame_scan.result.matched)
        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(frame_scan.result.detections[0].source, "full_frame")

    def test_tile_crop_dog_is_mapped_back_to_original_coordinates(self) -> None:
        detector = PathAwareDetector(
            {
                "snapshot_crop_tile_lower_right": (
                    Detection(class_name="dog", class_id=16, confidence=0.24, bbox=(5, 6, 20, 21)),
                ),
            }
        )

        frame_scan = scan_image(
            self.image_path,
            detector=detector,
            config=scan_config(enable_person_roi_scan=False),
            debug_crop_dir=self.crop_dir,
        )

        self.assertTrue(frame_scan.result.matched)
        dog = max(frame_scan.result.detections, key=lambda detection: detection.confidence)
        self.assertEqual(dog.source, "tile")
        self.assertEqual(dog.crop_id, "tile_lower_right")
        self.assertEqual(dog.bbox, (105, 56, 120, 71))
        self.assertTrue(Path(dog.crop_path).exists())

    def test_person_roi_crop_can_match_lower_confidence_dog(self) -> None:
        detector = PathAwareDetector(
            {
                "snapshot": (
                    Detection(class_name="person", class_id=0, confidence=0.91, bbox=(80, 20, 120, 80)),
                ),
                "snapshot_crop_person_roi_1": (
                    Detection(class_name="dog", class_id=16, confidence=0.21, bbox=(10, 50, 30, 70)),
                ),
            }
        )

        frame_scan = scan_image(
            self.image_path,
            detector=detector,
            config=scan_config(),
            debug_crop_dir=self.crop_dir,
        )

        self.assertTrue(frame_scan.result.matched)
        roi_dogs = [detection for detection in frame_scan.result.detections if detection.source == "person_roi"]
        self.assertEqual(len(roi_dogs), 1)
        self.assertEqual(roi_dogs[0].crop_id, "person_roi_1")

    def test_confirmation_requires_dog_in_each_configured_frame(self) -> None:
        first = self.image_path
        second = self.root / "snapshot_frame2.jpg"
        Image.new("RGB", (200, 100), color=(5, 10, 15)).save(second)
        detector = PathAwareDetector(
            {
                "snapshot": (Detection(class_name="dog", class_id=16, confidence=0.45),),
                "snapshot_frame2": (Detection(class_name="dog", class_id=16, confidence=0.46),),
            }
        )
        sleep = Mock()

        confirmed = scan_confirmed_snapshots(
            capture_snapshot=lambda frame_index: first if frame_index == 0 else second,
            detector=detector,
            config=scan_config(confirm_frames=2, confirm_interval_seconds=0.5),
            debug_crop_dir=self.crop_dir,
            sleep=sleep,
        )

        self.assertTrue(confirmed.matched)
        self.assertIn("2/2", confirmed.matched_reason)
        sleep.assert_called_once_with(0.5)

    def test_summary_includes_requested_plain_english_lines(self) -> None:
        summary = scanner_summary(
            DetectionResult(
                ok=True,
                backend="fake",
                image=str(self.image_path),
                detections=(
                    Detection(class_name="person", class_id=0, confidence=0.9, source="full_frame"),
                    Detection(
                        class_name="dog",
                        class_id=16,
                        confidence=0.24,
                        source="tile",
                        crop_id="tile_lower_right",
                    ),
                ),
                matched=True,
                matched_reason="tile dog confidence 0.24 >= 0.20",
            ),
            best_crop_path=self.crop_dir / "snapshot_crop_tile_lower_right.jpg",
        )

        self.assertIn("Yorkie Watch view:", summary)
        self.assertIn("Full-frame scan: 1 person(s), 0 dog(s).", summary)
        self.assertIn("Zoom scan: dog candidate found in tile lower right crop at 0.24 confidence.", summary)
        self.assertIn("Alert condition: matched.", summary)
        self.assertIn("Best crop: snapshot_crop_tile_lower_right.jpg", summary)


if __name__ == "__main__":
    unittest.main()