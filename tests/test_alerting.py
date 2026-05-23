from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.alerting import annotate_dog_alert_image, evaluate_dog_alert
from yorkie_watch.cleanup import cleanup_evidence_artifacts, cleanup_image_directory
from yorkie_watch.config import DogAlertConfig, StreamConfig
from yorkie_watch.detector import Detection, DetectionResult
from yorkie_watch.main import run_stream_watch_loop


def dog_alert_config(**overrides: object) -> DogAlertConfig:
    values = {
        "min_confidence": 0.45,
        "cooldown_seconds": 180.0,
        "confirmation_frames": 2,
        "min_box_area_ratio": 0.01,
        "save_debug_frames": False,
        "evidence_dir": "data/evidence",
        "image_retention_seconds": 3600.0,
        "max_evidence_images": 100,
    }
    values.update(overrides)
    return DogAlertConfig(**values)  # type: ignore[arg-type]


def stream_config(**overrides: object) -> StreamConfig:
    values = {
        "enabled": True,
        "url": "<stream-url>",
        "backend": "opencv",
        "use_home_assistant": False,
        "ha_base_url": "",
        "ha_stream_entity": "",
        "ha_stream_url": "",
        "ha_long_lived_token": "",
        "ha_stream_auth_mode": "bearer",
        "frame_interval_seconds": 5.0,
        "reconnect_seconds": 0.0,
        "max_failures": 0,
        "keep_frames": True,
        "save_debug_frames": True,
        "debug_dir": "data/stream_frames",
        "retention_minutes": 60.0,
        "max_frame_files": 500,
        "debug_crop_retention_minutes": 60.0,
        "debug_crop_max_files": 500,
        "alert_cooldown_seconds": 300.0,
        "python_executable": "python3",
    }
    values.update(overrides)
    return StreamConfig(**values)  # type: ignore[arg-type]


def detection_result(image_path: Path, *, confidence: float = 0.72, bbox: tuple[float, float, float, float] = (20, 20, 120, 80)) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend="fake",
        image=str(image_path),
        detections=(Detection(class_name="dog", class_id=16, confidence=confidence, bbox=bbox),),
        matched=True,
        matched_reason="dog matched",
    )


class FakeFrameSource:
    def __init__(self, frames: list[Path]) -> None:
        self.frames = frames

    def __enter__(self) -> "FakeFrameSource":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def __iter__(self):
        yield from self.frames


class DogAlertTests(unittest.TestCase):
    def make_image(self, directory: Path, name: str = "frame.jpg") -> Path:
        image_path = directory / name
        Image.new("RGB", (200, 100), color=(20, 30, 40)).save(image_path)
        return image_path

    def test_confidence_below_dog_min_confidence_does_not_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = self.make_image(Path(tmp_dir))
            evaluation = evaluate_dog_alert(
                image_path,
                detection_result(image_path, confidence=0.44),
                dog_alert_config(min_confidence=0.45),
            )

        self.assertFalse(evaluation.matched)
        self.assertIn("no dog met alert confidence 0.45", evaluation.reason)

    def test_one_detection_frame_does_not_alert_when_two_confirmations_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame = self.make_image(Path(tmp_dir), "frame1.jpg")
            sent: list[Path] = []

            state = run_stream_watch_loop(
                config=stream_config(),
                source_factory=lambda: FakeFrameSource([frame]),
                scan_frame=lambda path: detection_result(path),
                notify_alert=lambda path, _result: sent.append(path) is None,
                max_frames=1,
                dog_alert_config=dog_alert_config(confirmation_frames=2),
            )

        self.assertEqual(sent, [])
        self.assertEqual(state.dog_confirmation_count, 1)

    def test_two_consecutive_valid_detections_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames = [self.make_image(root, "frame1.jpg"), self.make_image(root, "frame2.jpg")]
            sent: list[Path] = []

            state = run_stream_watch_loop(
                config=stream_config(),
                source_factory=lambda: FakeFrameSource(frames),
                scan_frame=lambda path: detection_result(path),
                notify_alert=lambda path, _result: sent.append(path) is None,
                max_frames=2,
                dog_alert_config=dog_alert_config(confirmation_frames=2),
            )

        self.assertEqual(sent, [frames[1]])
        self.assertEqual(state.dog_confirmation_count, 2)

    def test_cooldown_suppresses_repeat_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames = [self.make_image(root, f"frame{index}.jpg") for index in range(1, 4)]
            times = iter([10.0, 11.0, 12.0])
            sent: list[Path] = []

            with self.assertLogs("yorkie_watch.main", level="INFO") as logs:
                run_stream_watch_loop(
                    config=stream_config(),
                    source_factory=lambda: FakeFrameSource(frames),
                    scan_frame=lambda path: detection_result(path),
                    notify_alert=lambda path, _result: sent.append(path) is None,
                    max_frames=3,
                    clock=lambda: next(times),
                    dog_alert_config=dog_alert_config(confirmation_frames=1, cooldown_seconds=180.0),
                )

        self.assertEqual(sent, [frames[0]])
        self.assertIn("stream alert matched but cooldown active; no message sent", "\n".join(logs.output))

    def test_tiny_bbox_below_min_area_ratio_does_not_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = self.make_image(Path(tmp_dir))
            evaluation = evaluate_dog_alert(
                image_path,
                detection_result(image_path, confidence=0.90, bbox=(10, 10, 14, 14)),
                dog_alert_config(min_box_area_ratio=0.01),
            )

        self.assertFalse(evaluation.matched)
        self.assertIn("min box area ratio", evaluation.reason)

    def test_annotated_image_function_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = self.make_image(root)
            evaluation = evaluate_dog_alert(image_path, detection_result(image_path), dog_alert_config())
            self.assertIsNotNone(evaluation.candidate)

            output_path = annotate_dog_alert_image(
                image_path,
                evaluation.candidate,  # type: ignore[arg-type]
                output_dir=root / "evidence",
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_cleanup_does_not_delete_outside_allowed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            outside_dir = root / "outside"
            outside_dir.mkdir()
            outside_file = outside_dir / "evidence.jpg"
            outside_file.write_bytes(b"evidence")

            stats = cleanup_image_directory(
                outside_dir,
                retention_minutes=0.01,
                max_files=0,
                allowed_data_dir=data_dir,
                now=10_000.0,
            )

            self.assertTrue(stats.skipped)
            self.assertTrue(outside_file.exists())

    def test_evidence_cleanup_limits_count_inside_allowed_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            evidence_dir = data_dir / "evidence"
            evidence_dir.mkdir(parents=True)
            for index in range(3):
                path = evidence_dir / f"alert_{index}.jpg"
                path.write_bytes(b"evidence")

            stats = cleanup_evidence_artifacts(
                dog_alert_config(evidence_dir=str(evidence_dir), max_evidence_images=1),
                allowed_data_dir=data_dir,
            )

            self.assertEqual(stats.count_deleted, 2)
            self.assertEqual(len(list(evidence_dir.glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
