from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import StreamConfig  # noqa: E402
from yorkie_watch.cleanup import cleanup_image_directory  # noqa: E402
from yorkie_watch.detector import DetectionResult  # noqa: E402
from yorkie_watch.main import build_parser, run_stream_watch_loop  # noqa: E402
from yorkie_watch.stream_source import (  # noqa: E402
    FFmpegSubprocessFrameSource,
    OpenCVSubprocessFrameSource,
    StreamSourceError,
    create_stream_source,
    redact_stream_output,
    resolve_stream_url,
)
from scripts import ffmpeg_stream_frames as ffmpeg_helper  # noqa: E402


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
        "keep_frames": False,
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

    def test_processed_stream_frame_is_deleted_when_keep_and_debug_are_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "data" / "stream_frames" / "frame.jpg"
            frame_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"frame")

            run_stream_watch_loop(
                config=stream_config(keep_frames=False, save_debug_frames=False, debug_dir=str(frame_path.parent)),
                source_factory=lambda: FakeFrameSource([frame_path]),
                scan_frame=lambda frame: scan_result(frame, matched=False),
                notify_alert=lambda _frame_path, _result: True,
                max_frames=1,
            )

            self.assertFalse(frame_path.exists())

    def test_processed_stream_frame_is_kept_when_keep_or_debug_is_true(self) -> None:
        for overrides in ({"keep_frames": True, "save_debug_frames": False}, {"keep_frames": False, "save_debug_frames": True}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp_dir:
                frame_path = Path(tmp_dir) / "data" / "stream_frames" / "frame.jpg"
                frame_path.parent.mkdir(parents=True)
                frame_path.write_bytes(b"frame")

                run_stream_watch_loop(
                    config=stream_config(debug_dir=str(frame_path.parent), **overrides),
                    source_factory=lambda: FakeFrameSource([frame_path]),
                    scan_frame=lambda frame: scan_result(frame, matched=False),
                    notify_alert=lambda _frame_path, _result: True,
                    max_frames=1,
                )

                self.assertTrue(frame_path.exists())

    def test_alert_stream_frame_is_available_for_notification_and_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "data" / "stream_frames" / "alert.jpg"
            frame_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"frame")
            notified: list[Path] = []

            def notify_alert(path: Path, _result: DetectionResult) -> bool:
                self.assertTrue(path.exists())
                notified.append(path)
                return True

            run_stream_watch_loop(
                config=stream_config(keep_frames=False, save_debug_frames=False, debug_dir=str(frame_path.parent)),
                source_factory=lambda: FakeFrameSource([frame_path]),
                scan_frame=lambda frame: scan_result(frame, matched=True),
                notify_alert=notify_alert,
                max_frames=1,
                clock=lambda: 10.0,
            )

            self.assertEqual(notified, [frame_path])
            self.assertTrue(frame_path.exists())

    def test_cleanup_deletes_old_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            stream_dir = data_dir / "stream_frames"
            stream_dir.mkdir(parents=True)
            old_file = stream_dir / "old.jpg"
            new_file = stream_dir / "new.jpg"
            old_file.write_bytes(b"old")
            new_file.write_bytes(b"new")
            now = 10_000.0
            os.utime(old_file, (now - 7_200, now - 7_200))
            os.utime(new_file, (now, now))

            stats = cleanup_image_directory(
                stream_dir,
                retention_minutes=60.0,
                max_files=500,
                allowed_data_dir=data_dir,
                now=now,
            )

            self.assertEqual(stats.age_deleted, 1)
            self.assertFalse(old_file.exists())
            self.assertTrue(new_file.exists())

    def test_cleanup_enforces_max_file_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            debug_dir = data_dir / "debug_crops"
            debug_dir.mkdir(parents=True)
            now = 10_000.0
            files: list[Path] = []
            for index in range(5):
                path = debug_dir / f"crop_{index}.jpg"
                path.write_bytes(b"crop")
                os.utime(path, (now + index, now + index))
                files.append(path)

            stats = cleanup_image_directory(
                debug_dir,
                retention_minutes=60.0,
                max_files=2,
                allowed_data_dir=data_dir,
                now=now,
            )

            self.assertEqual(stats.count_deleted, 3)
            self.assertEqual(sorted(path.name for path in debug_dir.iterdir()), ["crop_3.jpg", "crop_4.jpg"])
            self.assertFalse(any(path.exists() for path in files[:3]))

    def test_cleanup_never_deletes_outside_configured_data_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            outside_dir = Path(tmp_dir) / "outside"
            outside_dir.mkdir()
            outside_file = outside_dir / "outside.jpg"
            outside_file.write_bytes(b"outside")

            stats = cleanup_image_directory(
                outside_dir,
                retention_minutes=1.0,
                max_files=0,
                allowed_data_dir=data_dir,
                now=10_000.0,
            )

            self.assertTrue(stats.skipped)
            self.assertTrue(outside_file.exists())

    def test_snapshot_watch_and_stream_watch_cli_modes_remain_available(self) -> None:
        parser = build_parser()

        self.assertTrue(parser.parse_args(["--once"]).once)
        self.assertTrue(parser.parse_args(["--watch"]).watch)
        args = parser.parse_args(["--watch-stream", "--stream-frames", "3", "--stream-save-debug-frame"])
        self.assertTrue(args.watch_stream)
        self.assertEqual(args.stream_frames, 3)
        self.assertTrue(args.stream_save_debug_frame)

    def test_home_assistant_backend_builds_camera_proxy_stream_url(self) -> None:
        config = stream_config(
            url="",
            backend="home_assistant",
            use_home_assistant=True,
            ha_base_url="http://<home-assistant-host>:8123/",
            ha_stream_entity="camera.<placeholder>",
            ha_long_lived_token="<ha-long-lived-token>",
        )

        source = create_stream_source(config)

        self.assertIsInstance(source, FFmpegSubprocessFrameSource)
        self.assertEqual(
            resolve_stream_url(config),
            "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.%3Cplaceholder%3E",
        )

    def test_bearer_token_selects_ffmpeg_helper_header_command(self) -> None:
        token = "<ha-long-lived-token>"
        source = FFmpegSubprocessFrameSource(
            stream_config(
                url="",
                backend="home_assistant",
                use_home_assistant=True,
                ha_base_url="http://<home-assistant-host>:8123",
                ha_stream_entity="camera.<placeholder>",
                ha_long_lived_token=token,
            )
        )

        helper_argv = source._build_helper_argv()
        ffmpeg_argv = ffmpeg_helper.build_ffmpeg_argv(
            stream_url=source.stream_url,
            output_pattern="data/stream_frames/frame_%06d.jpg",
            frame_interval=5.0,
            bearer_token=token,
        )

        self.assertIn("ffmpeg_stream_frames.py", helper_argv[1])
        self.assertEqual(helper_argv[helper_argv.index("--bearer-token") + 1], token)
        self.assertEqual(ffmpeg_argv[ffmpeg_argv.index("-headers") + 1], f"Authorization: Bearer {token}\r\n")

    def test_stream_redaction_removes_authorization_token_and_url_query(self) -> None:
        token = "<ha-long-lived-token>"
        stream_url = "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder?token=<query-token>"
        config = stream_config(
            url="",
            backend="home_assistant",
            ha_stream_url=stream_url,
            ha_long_lived_token=token,
        )

        output = redact_stream_output(
            config,
            f"Authorization: Bearer {token}\r\nfailed opening {stream_url}",
            resolved_url="",
        )

        self.assertNotIn("Authorization: Bearer " + token, output)
        self.assertNotIn(token, output)
        self.assertNotIn("<query-token>", output)
        self.assertIn("Authorization: Bearer <redacted-stream-value>", output)

    def test_direct_url_mode_still_uses_stream_url(self) -> None:
        direct_url = "rtsp://<camera-stream-host>/<placeholder>"

        self.assertEqual(resolve_stream_url(stream_config(url=direct_url)), direct_url)

    def test_direct_opencv_backend_does_not_add_home_assistant_headers(self) -> None:
        source = create_stream_source(stream_config(url="rtsp://<camera-stream-host>/<placeholder>"))

        self.assertIsInstance(source, OpenCVSubprocessFrameSource)
        self.assertNotIsInstance(source, FFmpegSubprocessFrameSource)
        self.assertNotIn("--bearer-token", source._build_helper_argv())

    def test_stream_mode_fails_clearly_without_a_url(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "YORKIE_STREAM_URL"):
            create_stream_source(stream_config(url=""))

    def test_home_assistant_backend_requires_base_and_entity_without_override(self) -> None:
        config = stream_config(
            url="",
            backend="home_assistant",
            use_home_assistant=True,
            ha_long_lived_token="<ha-long-lived-token>",
        )

        with self.assertRaisesRegex(RuntimeError, "YORKIE_HA_BASE_URL"):
            create_stream_source(config)
        with self.assertRaisesRegex(RuntimeError, "YORKIE_HA_STREAM_ENTITY"):
            create_stream_source(
                stream_config(
                    url="",
                    backend="home_assistant",
                    use_home_assistant=True,
                    ha_base_url="http://<home-assistant-host>:8123",
                    ha_long_lived_token="<ha-long-lived-token>",
                )
            )

    def test_bearer_token_is_required_only_for_home_assistant_bearer_auth(self) -> None:
        bearer_config = stream_config(
            url="",
            backend="home_assistant",
            use_home_assistant=True,
            ha_base_url="http://<home-assistant-host>:8123",
            ha_stream_entity="camera.<placeholder>",
        )

        with self.assertRaisesRegex(RuntimeError, "YORKIE_HA_LONG_LIVED_TOKEN"):
            create_stream_source(bearer_config)
        self.assertIsInstance(
            create_stream_source(
                stream_config(
                    url="",
                    backend="home_assistant",
                    use_home_assistant=True,
                    ha_base_url="http://<home-assistant-host>:8123",
                    ha_stream_entity="camera.<placeholder>",
                    ha_stream_auth_mode="none",
                )
            ),
            FFmpegSubprocessFrameSource,
        )
        self.assertIsInstance(
            create_stream_source(stream_config(url="rtsp://<camera-stream-host>/<placeholder>")),
            OpenCVSubprocessFrameSource,
        )

    def test_ffmpeg_helper_accepts_frame_limit(self) -> None:
        args = ffmpeg_helper.build_parser().parse_args(
            [
                "--url",
                "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder",
                "--output-dir",
                "data/stream_frames",
                "--frames",
                "3",
            ]
        )

        self.assertEqual(args.frames, 3)

    def test_bounded_ffmpeg_helper_argv_uses_frame_limit_without_fps_filter(self) -> None:
        argv = ffmpeg_helper.build_ffmpeg_argv(
            stream_url="http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder",
            output_pattern="data/stream_frames/frame_%06d.jpg",
            frame_interval=5.0,
            frames=3,
        )

        self.assertIn("-frames:v", argv)
        self.assertEqual(argv[argv.index("-frames:v") + 1], "3")
        self.assertNotIn("-vf", argv)
        self.assertNotIn("-fflags", argv)

    def test_continuous_ffmpeg_helper_argv_uses_timestamp_fps_sampling(self) -> None:
        argv = ffmpeg_helper.build_ffmpeg_argv(
            stream_url="http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder",
            output_pattern="data/stream_frames/frame_%06d.jpg",
            frame_interval=5.0,
        )

        self.assertIn("-fflags", argv)
        self.assertEqual(argv[argv.index("-fflags") + 1], "+genpts")
        self.assertIn("-use_wallclock_as_timestamps", argv)
        self.assertEqual(argv[argv.index("-vf") + 1], "fps=1/5")
        self.assertNotIn("-frames:v", argv)

    def test_ffmpeg_helper_bounded_capture_emits_output_files_after_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = StringIO()
            output_pattern = Path(temp_dir) / "run_%06d.jpg"
            for index in range(1, 4):
                (Path(temp_dir) / f"run_{index:06d}.jpg").write_bytes(b"jpeg")

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "ffmpeg_stream_frames.py",
                        "--url",
                        "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder",
                        "--output-dir",
                        temp_dir,
                        "--frame-interval",
                        "0",
                        "--frames",
                        "3",
                    ],
                ),
                patch("scripts.ffmpeg_stream_frames.build_output_pattern", return_value=output_pattern),
                patch(
                    "scripts.ffmpeg_stream_frames.subprocess.run",
                    return_value=Mock(returncode=0, stderr=""),
                ) as run,
                redirect_stdout(output),
            ):
                returncode = ffmpeg_helper.main()

        lines = [line for line in output.getvalue().splitlines() if '"type": "frame"' in line]
        self.assertEqual(returncode, 0)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("-vf", run.call_args.args[0])
        self.assertEqual(len(lines), 3)
        self.assertIn('"frame_index": 3', lines[-1])

    def test_ffmpeg_helper_bounded_capture_collects_partial_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = StringIO()
            first_pattern = Path(temp_dir) / "first_%06d.jpg"
            second_pattern = Path(temp_dir) / "second_%06d.jpg"
            (Path(temp_dir) / "first_000001.jpg").write_bytes(b"jpeg")
            (Path(temp_dir) / "first_000002.jpg").write_bytes(b"jpeg")
            (Path(temp_dir) / "second_000001.jpg").write_bytes(b"jpeg")

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "ffmpeg_stream_frames.py",
                        "--url",
                        "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder",
                        "--output-dir",
                        temp_dir,
                        "--frames",
                        "3",
                    ],
                ),
                patch(
                    "scripts.ffmpeg_stream_frames.build_output_pattern",
                    side_effect=[first_pattern, second_pattern],
                ),
                patch(
                    "scripts.ffmpeg_stream_frames.subprocess.run",
                    side_effect=[Mock(returncode=0, stderr=""), Mock(returncode=0, stderr="")],
                ) as run,
                redirect_stdout(output),
            ):
                returncode = ffmpeg_helper.main()

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        frame_events = [event for event in events if event["type"] == "frame"]
        self.assertEqual(returncode, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([event["frame_index"] for event in frame_events], [1, 2, 3])
        self.assertFalse([event for event in events if event["type"] == "error"])
        first_argv = run.call_args_list[0].args[0]
        second_argv = run.call_args_list[1].args[0]
        self.assertEqual(first_argv[first_argv.index("-frames:v") + 1], "3")
        self.assertEqual(second_argv[second_argv.index("-frames:v") + 1], "1")

    def test_ffmpeg_helper_bounded_no_output_error_is_clear_and_redacted(self) -> None:
        token = "<ha-long-lived-token>"
        url = "http://<home-assistant-host>:8123/api/camera_proxy_stream/camera.placeholder?token=<query-token>"
        output = StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "ffmpeg_stream_frames.py",
                        "--url",
                        url,
                        "--output-dir",
                        temp_dir,
                        "--frames",
                        "1",
                        "--bearer-token",
                        token,
                    ],
                ),
                patch(
                    "scripts.ffmpeg_stream_frames.subprocess.run",
                    return_value=Mock(
                        returncode=0,
                        stderr=f"Authorization: Bearer {token}\nfinished reading {url}",
                    ),
                ),
                redirect_stdout(output),
            ):
                returncode = ffmpeg_helper.main()

        text = output.getvalue()
        self.assertNotEqual(returncode, 0)
        self.assertIn('"type": "error"', text)
        self.assertIn("ffmpeg exited 0 but no output frames were found", text)
        self.assertIn('"found_count": 0', text)
        self.assertIn('"requested_count": 1', text)
        self.assertNotIn(token, text)
        self.assertNotIn(url, text)
        self.assertNotIn("<query-token>", text)
        self.assertNotIn(f"Authorization: Bearer {token}", text)

    def test_stream_frame_limit_is_passed_to_ffmpeg_helper_and_loop_finishes(self) -> None:
        config = stream_config(
            url="",
            backend="home_assistant",
            use_home_assistant=True,
            ha_base_url="http://<home-assistant-host>:8123",
            ha_stream_entity="camera.<placeholder>",
            ha_long_lived_token="<ha-long-lived-token>",
        )
        source = create_stream_source(config, frame_limit=3)

        state = run_stream_watch_loop(
            config=config,
            source_factory=lambda: FakeFrameSource([Path("frame-1.jpg"), Path("frame-2.jpg"), Path("frame-3.jpg")]),
            scan_frame=lambda frame_path: scan_result(frame_path, matched=False),
            notify_alert=lambda _frame_path, _result: True,
            max_frames=3,
        )

        self.assertIn("--frames", source._build_helper_argv())  # type: ignore[attr-defined]
        self.assertEqual(source._build_helper_argv()[-1], "3")  # type: ignore[attr-defined]
        self.assertEqual(state.sampled_frames, 3)


if __name__ == "__main__":
    unittest.main()
