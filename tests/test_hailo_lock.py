from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.hailo_lock import HailoDeviceLock, HailoDeviceLockError  # noqa: E402


class HailoLockTests(unittest.TestCase):
    def test_lock_context_acquires_and_releases_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "hailo.lock"
            lock = HailoDeviceLock(lock_path=lock_path, timeout_seconds=1.0)

            with lock:
                self.assertTrue(lock.acquired)
                self.assertTrue(lock_path.exists())

            self.assertFalse(lock.acquired)

    def test_lock_timeout_when_already_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "hailo.lock"
            first = HailoDeviceLock(lock_path=lock_path, timeout_seconds=1.0)
            second = HailoDeviceLock(lock_path=lock_path, timeout_seconds=0.0)

            with first:
                with self.assertRaises(HailoDeviceLockError):
                    second.acquire()


if __name__ == "__main__":
    unittest.main()
