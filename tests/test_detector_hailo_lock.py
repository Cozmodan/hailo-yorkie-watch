from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import DetectorConfig  # noqa: E402
from yorkie_watch.detector import HailoAppsDetector  # noqa: E402


class FakeLock:
    events: list[str] = []

    @classmethod
    def from_env(cls) -> "FakeLock":
        return cls()

    def __enter__(self) -> "FakeLock":
        self.events.append("enter")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.events.append("exit")


def detector_config() -> DetectorConfig:
    return DetectorConfig(
        enabled=True,
        backend="hailo_apps",
        hef_path="/usr/share/hailo-models/yolov8m_h10.hef",
        hailo_apps_root="<hailo-apps-root>",
        confidence_threshold=0.45,
        target_classes=("dog",),
        timeout_seconds=60.0,
        python_executable="python3",
        command_template="",
    )


class DetectorHailoLockTests(unittest.TestCase):
    def test_hailo_apps_detector_uses_lock_around_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "snapshot.jpg"
            Image.new("RGB", (32, 32), color="gray").save(image_path)
            completed = Mock(
                returncode=0,
                stdout='{"ok": true, "backend": "hailo_apps", "detections": [], "matched": false}',
                stderr="",
            )
            FakeLock.events = []

            with (
                patch("yorkie_watch.detector.HailoDeviceLock", FakeLock),
                patch("yorkie_watch.detector.subprocess.run", return_value=completed) as run,
            ):
                result = HailoAppsDetector(detector_config()).detect(image_path)

        self.assertTrue(result.ok)
        self.assertEqual(FakeLock.events, ["enter", "exit"])
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
