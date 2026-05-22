from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import WatchConfig  # noqa: E402
from yorkie_watch.detector import DetectionResult, DetectorError  # noqa: E402
from yorkie_watch.ha_client import HomeAssistantError  # noqa: E402
from yorkie_watch.main import build_parser, run_watch_loop  # noqa: E402


def watch_config(**overrides: object) -> WatchConfig:
    values = {
        "interval_seconds": 0.0,
        "cooldown_seconds": 300.0,
        "max_iterations": 1,
        "send_no_match_log": True,
        "heartbeat_every": 0,
        "reuse_last_snapshot_on_ha_fail": False,
        "stop_on_error": False,
    }
    values.update(overrides)
    return WatchConfig(**values)  # type: ignore[arg-type]


def scan_result(image_path: Path, *, matched: bool) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend="fake",
        image=str(image_path),
        detections=(),
        matched=matched,
        matched_reason="dog matched" if matched else "no match",
    )


class WatchModeTests(unittest.TestCase):
    def test_watch_loop_sends_alert_on_match(self) -> None:
        sent: list[Path] = []

        state = run_watch_loop(
            config=watch_config(),
            capture_snapshot=lambda iteration: Path(f"snapshot-{iteration}.jpg"),
            scan_snapshot=lambda image_path, _iteration: scan_result(image_path, matched=True),
            notify_alert=lambda image_path, _result: sent.append(image_path) is None,
            clock=lambda: 10.0,
        )

        self.assertEqual(state.iterations, 1)
        self.assertEqual(sent, [Path("snapshot-1.jpg")])
        self.assertEqual(state.last_alert_at, 10.0)

    def test_watch_loop_does_not_send_during_cooldown(self) -> None:
        sent: list[Path] = []
        alert_times = iter([10.0, 11.0])

        with self.assertLogs("yorkie_watch.main", level="INFO") as logs:
            run_watch_loop(
                config=watch_config(max_iterations=2),
                capture_snapshot=lambda iteration: Path(f"snapshot-{iteration}.jpg"),
                scan_snapshot=lambda image_path, _iteration: scan_result(image_path, matched=True),
                notify_alert=lambda image_path, _result: sent.append(image_path) is None,
                clock=lambda: next(alert_times),
            )

        self.assertEqual(sent, [Path("snapshot-1.jpg")])
        self.assertIn("alert matched but cooldown active; no message sent", "\n".join(logs.output))

    def test_watch_loop_continues_after_snapshot_failure(self) -> None:
        captures: list[int] = []
        scans: list[Path] = []

        def capture_snapshot(iteration: int) -> Path:
            captures.append(iteration)
            if iteration == 1:
                raise HomeAssistantError("temporary Home Assistant failure")
            return Path(f"snapshot-{iteration}.jpg")

        with self.assertLogs("yorkie_watch.main", level="WARNING"):
            state = run_watch_loop(
                config=watch_config(max_iterations=2),
                capture_snapshot=capture_snapshot,
                scan_snapshot=lambda image_path, _iteration: scans.append(image_path)
                or scan_result(image_path, matched=False),
                notify_alert=lambda _image_path, _result: True,
            )

        self.assertEqual(state.iterations, 2)
        self.assertEqual(captures, [1, 2])
        self.assertEqual(scans, [Path("snapshot-2.jpg")])

    def test_watch_loop_continues_after_detector_failure(self) -> None:
        scanned_iterations: list[int] = []

        def scan_snapshot(image_path: Path, iteration: int) -> DetectionResult:
            scanned_iterations.append(iteration)
            if iteration == 1:
                raise DetectorError("temporary detector failure")
            return scan_result(image_path, matched=False)

        with self.assertLogs("yorkie_watch.main", level="WARNING"):
            state = run_watch_loop(
                config=watch_config(max_iterations=2),
                capture_snapshot=lambda iteration: Path(f"snapshot-{iteration}.jpg"),
                scan_snapshot=scan_snapshot,
                notify_alert=lambda _image_path, _result: True,
            )

        self.assertEqual(state.iterations, 2)
        self.assertEqual(scanned_iterations, [1, 2])

    def test_watch_loop_max_iterations_exits_cleanly(self) -> None:
        captures: list[int] = []

        state = run_watch_loop(
            config=watch_config(max_iterations=2),
            capture_snapshot=lambda iteration: captures.append(iteration) or Path(f"snapshot-{iteration}.jpg"),
            scan_snapshot=lambda image_path, _iteration: scan_result(image_path, matched=False),
            notify_alert=lambda _image_path, _result: True,
        )

        self.assertEqual(state.iterations, 2)
        self.assertEqual(captures, [1, 2])

    def test_existing_once_and_what_see_cli_modes_remain_available(self) -> None:
        parser = build_parser()

        self.assertTrue(parser.parse_args(["--once"]).once)
        self.assertTrue(parser.parse_args(["--what-see"]).what_see)
        args = parser.parse_args(["--watch", "--watch-iterations", "2"])
        self.assertTrue(args.watch)
        self.assertEqual(args.watch_iterations, 2)


if __name__ == "__main__":
    unittest.main()
