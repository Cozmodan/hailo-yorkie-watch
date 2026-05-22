from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import StreamConfig  # noqa: E402
from yorkie_watch.detector import DetectionResult  # noqa: E402
from yorkie_watch.main import build_parser, run_stream_watch_loop  # noqa: E402
from yorkie_watch.stream_source import (  # noqa: E402
    OpenCVSubprocessFrameSource,
    StreamSourceError,
    create_stream_source,
    redact_stream_output,
    resolve_stream_url,
)


def stream_config(**overrides: object) -> StreamConfig:
    values = {
        "enabled": True,
        "url": "<stream-url>",
        "backend": "opencv",
        "use_home_assistant": False,
        "ha_stream_entity": "",
        "ha_stream_url": "",
        "ha_stream_token": "",
        "frame_interval_seconds": 5.0,
        "reconnect_seconds": 0.0,
        "max_failures": 0,
        "save_debug_frames": True,
        "debug_dir": "data/stream_frames",
        "alert_cooldown_seconds": 300.0,
        "python_executable": "python3",
    }
    values.update(overrides)
    return StreamConfig(**values)  # type: ignore[arg-type]


def scan_result(frame_path: Path, *, matched: bool) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend="fake",
        image=str(frame_path),
        detections=(),
        matched=matched,
        matched_reason="dog matched" if matched else "no dog matched",
    )


class FakeFrameSource:
    def __init__(self, frames: list[Path] | None = None, *, error: str = "") -> None:
        self.frames = frames or []
        self.error = error

    def __enter__(self) -> "FakeFrameSource":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def __iter__(self):
        yield from self.frames
        if self.error:
            raise StreamSourceError(self.error)


class StreamWatchTests(unittest.TestCase):
    def test_stream_loop_samples_frames(self) -> None:
        scanned: list[Path] = []

        state = run_stream_watch_loop(
            config=stream_config(),
            source_factory=lambda: FakeFrameSource([Path("frame-1.jpg"), Path("frame-2.jpg")]),
            scan_frame=lambda frame_path: scanned.append(frame_path) or scan_result(frame_path, matched=False),
            notify_alert=lambda _frame_path, _result: True,
            max_frames=2,
        )

        self.assertEqual(state.sampled_frames, 2)
        self.assertEqual(scanned, [Path("frame-1.jpg"), Path("frame-2.jpg")])

    def test_stream_match_sends_alert(self) -> None:
        sent: list[Path] = []

        state = run_stream_watch_loop(
            config=stream_config(),
            source_factory=lambda: FakeFrameSource([Path("frame-1.jpg")]),
            scan_frame=lambda frame_path: scan_result(frame_path, matched=True),
            notify_alert=lambda frame_path, _result: sent.append(frame_path) is None,
            max_frames=1,
            clock=lambda: 10.0,
        )

        self.assertEqual(sent, [Path("frame-1.jpg")])
        self.assertEqual(state.last_alert_at, 10.0)

    def test_stream_cooldown_suppresses_repeated_alert(self) -> None:
        sent: list[Path] = []
        times = iter([10.0, 11.0])

        with self.assertLogs("yorkie_watch.main", level="INFO") as logs:
            run_stream_watch_loop(
                config=stream_config(),
                source_factory=lambda: FakeFrameSource([Path("frame-1.jpg"), Path("frame-2.jpg")]),
                scan_frame=lambda frame_path: scan_result(frame_path, matched=True),
                notify_alert=lambda frame_path, _result: sent.append(frame_path) is None,
                max_frames=2,
                clock=lambda: next(times),
            )

        self.assertEqual(sent, [Path("frame-1.jpg")])
        self.assertIn("stream alert matched but cooldown active; no message sent", "\n".join(logs.output))

    def test_stream_failure_reconnects_for_bounded_test_run(self) -> None:
        sleep = Mock()
        sources = iter(
            [
                FakeFrameSource(error="stream read failed"),
                FakeFrameSource([Path("frame-1.jpg")]),
            ]
        )

        with self.assertLogs("yorkie_watch.main", level="INFO") as logs:
            state = run_stream_watch_loop(
                config=stream_config(reconnect_seconds=0.25, max_failures=2),
                source_factory=lambda: next(sources),
                scan_frame=lambda frame_path: scan_result(frame_path, matched=False),
                notify_alert=lambda _frame_path, _result: True,
                max_frames=1,
                sleep=sleep,
            )

        self.assertEqual(state.sampled_frames, 1)
        self.assertEqual(state.failures, 1)
        sleep.assert_called_once_with(0.25)
        self.assertIn("Reconnecting stream", "\n".join(logs.output))

    def test_snapshot_watch_and_stream_watch_cli_modes_remain_available(self) -> None:
        parser = build_parser()

        self.assertTrue(parser.parse_args(["--once"]).once)
        self.assertTrue(parser.parse_args(["--watch"]).watch)
        args = parser.parse_args(["--watch-stream", "--stream-frames", "3", "--stream-save-debug-frame"])
        self.assertTrue(args.watch_stream)
        self.assertEqual(args.stream_frames, 3)
        self.assertTrue(args.stream_save_debug_frame)

    def test_home_assistant_hls_url_is_passed_to_frame_helper(self) -> None:
        hls_url = "http://<home-assistant-host>:8123/api/hls/<placeholder>/master_playlist.m3u8"
        source = OpenCVSubprocessFrameSource(
            stream_config(
                url="",
                backend="ha_hls",
                use_home_assistant=True,
                ha_stream_url=hls_url,
                ha_stream_token="<ha-stream-token>",
            )
        )

        argv = source._build_helper_argv()

        self.assertEqual(argv[argv.index("--url") + 1], hls_url)

    def test_stream_redaction_removes_hls_url_and_token(self) -> None:
        hls_url = "http://<home-assistant-host>:8123/api/hls/<placeholder>/master_playlist.m3u8"
        token = "<ha-stream-token>"
        config = stream_config(url="", backend="home_assistant", ha_stream_url=hls_url, ha_stream_token=token)

        output = redact_stream_output(
            config,
            f"failed opening {hls_url} using {token}",
            resolved_url=hls_url,
        )

        self.assertNotIn(hls_url, output)
        self.assertNotIn(token, output)
        self.assertIn("<redacted-stream-value>", output)

    def test_direct_url_mode_still_uses_stream_url(self) -> None:
        direct_url = "rtsp://<camera-stream-host>/<placeholder>"

        self.assertEqual(resolve_stream_url(stream_config(url=direct_url)), direct_url)

    def test_stream_mode_fails_clearly_without_a_url(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "YORKIE_STREAM_URL"):
            create_stream_source(stream_config(url=""))
        with self.assertRaisesRegex(RuntimeError, "YORKIE_HA_STREAM_URL or YORKIE_STREAM_URL"):
            create_stream_source(stream_config(url="", backend="ha_hls", use_home_assistant=True))


if __name__ == "__main__":
    unittest.main()
