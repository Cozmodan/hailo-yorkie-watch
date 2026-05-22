from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .config import (
    ConfigError,
    StreamConfig,
    WatchConfig,
    load_detector_config,
    load_scan_config,
    load_stream_config,
    load_watch_config,
)
from .detector import COCO_DOG_CLASS_ID, DetectionResult, DetectorError, create_detector, print_result
from .ha_client import HomeAssistantClient, HomeAssistantError
from .openclaw_client import OpenClawClient
from .scanner import best_crop_path, best_dog_confidence, scan_confirmed_snapshots, scan_image, scanner_summary
from .stream_source import StreamFrameSource, StreamSourceError, create_stream_source

LOGGER = logging.getLogger(__name__)
SNAPSHOT_DIR = Path("data") / "snapshots"


@dataclass
class WatchState:
    """In-memory state for one continuous watch process."""

    iterations: int = 0
    last_alert_at: float | None = None
    last_snapshot_path: Path | None = None


@dataclass
class StreamWatchState:
    """In-memory state for one continuous stream watch process."""

    sampled_frames: int = 0
    failures: int = 0
    last_alert_at: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hailo Yorkie Watch plumbing CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Fetch one Home Assistant snapshot and save it locally.")
    mode.add_argument("--watch", action="store_true", help="Continuously scan Home Assistant snapshots for alerts.")
    mode.add_argument("--watch-stream", action="store_true", help="Continuously scan sampled live stream frames for alerts.")
    mode.add_argument("--test-openclaw", action="store_true", help="Send one test event to OpenClaw.")
    mode.add_argument("--test-detect", metavar="IMAGE", help="Run detector once against an existing image.")
    mode.add_argument(
        "--what-see",
        action="store_true",
        help="Fetch one snapshot, run detection, and send a WhatsApp summary with the snapshot.",
    )
    parser.add_argument(
        "--watch-iterations",
        metavar="N",
        type=int,
        help="Override YORKIE_WATCH_MAX_ITERATIONS for bounded watch-mode test runs.",
    )
    parser.add_argument(
        "--stream-frames",
        metavar="N",
        type=int,
        help="Stop live stream watch mode after N sampled frames.",
    )
    parser.add_argument(
        "--stream-save-debug-frame",
        action="store_true",
        help="Keep sampled live stream frames for this run.",
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
        scan_config = load_scan_config()
        detector = create_detector(detector_config)
        if scan_config.confirm_frames > 1:
            result = scan_confirmed_snapshots(
                capture_snapshot=lambda frame_index: saved_path
                if frame_index == 0
                else client.save_snapshot(
                    SNAPSHOT_DIR / f"snapshot_{timestamp}_frame{frame_index + 1}.jpg",
                    attempts=3,
                    delay_seconds=2.0,
                ),
                detector=detector,
                config=scan_config,
            )
            _notify_detection_result(saved_path, result)
        else:
            run_detection_and_maybe_notify(saved_path, detector=detector)
    return 0


def run_watch(*, max_iterations: int | None = None) -> int:
    """Continuously capture, scan, and notify until stopped or iteration-limited."""
    watch_config = load_watch_config()
    if max_iterations is not None:
        if max_iterations < 0:
            raise ValueError("--watch-iterations must be zero or greater.")
        watch_config = replace(watch_config, max_iterations=max_iterations)

    client = HomeAssistantClient.from_env()
    detector = create_detector(load_detector_config())
    scan_config = load_scan_config()
    notifier: OpenClawClient | None = None

    def capture_snapshot(iteration: int) -> Path:
        return client.save_snapshot(_watch_snapshot_path(iteration), attempts=3, delay_seconds=2.0)

    def scan_snapshot(snapshot_path: Path, iteration: int) -> DetectionResult:
        if scan_config.confirm_frames > 1:
            return scan_confirmed_snapshots(
                capture_snapshot=lambda frame_index: snapshot_path
                if frame_index == 0
                else client.save_snapshot(
                    _watch_snapshot_path(iteration, frame_index=frame_index),
                    attempts=3,
                    delay_seconds=2.0,
                ),
                detector=detector,
                config=scan_config,
            )
        return scan_image(snapshot_path, detector=detector, config=scan_config).result

    def notify_alert(snapshot_path: Path, result: DetectionResult) -> bool:
        nonlocal notifier
        notifier = notifier or OpenClawClient.from_env()
        return _notify_detection_result(snapshot_path, result, notifier=notifier)

    def send_heartbeat(iteration: int) -> bool:
        nonlocal notifier
        notifier = notifier or OpenClawClient.from_env()
        return notifier.send_message(
            f"Yorkie Watch heartbeat: iteration {iteration} complete.",
            event_type="watch_heartbeat",
        )

    LOGGER.info("Starting watch mode.")
    try:
        state = run_watch_loop(
            config=watch_config,
            capture_snapshot=capture_snapshot,
            scan_snapshot=scan_snapshot,
            notify_alert=notify_alert,
            send_heartbeat=send_heartbeat,
        )
    except KeyboardInterrupt:
        LOGGER.info("Watch mode stopped.")
        print("Watch mode stopped.")
        return 0

    LOGGER.info("Watch mode finished after %d iteration(s).", state.iterations)
    return 0


def run_watch_loop(
    *,
    config: WatchConfig,
    capture_snapshot: Callable[[int], Path],
    scan_snapshot: Callable[[Path, int], DetectionResult],
    notify_alert: Callable[[Path, DetectionResult], bool],
    send_heartbeat: Callable[[int], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchState:
    """Run the watch loop with injectable IO boundaries for tests."""
    state = WatchState()
    while config.max_iterations == 0 or state.iterations < config.max_iterations:
        state.iterations += 1
        iteration = state.iterations
        try:
            snapshot_path = capture_snapshot(iteration)
        except HomeAssistantError as exc:
            LOGGER.warning("Watch snapshot fetch failed: %s", exc)
            if config.stop_on_error:
                raise
            if config.reuse_last_snapshot_on_ha_fail and state.last_snapshot_path is not None:
                snapshot_path = state.last_snapshot_path
                LOGGER.info("Reusing last snapshot after Home Assistant fetch failure.")
            else:
                _watch_iteration_finished(config, iteration, send_heartbeat, sleep)
                continue
        else:
            state.last_snapshot_path = snapshot_path

        try:
            result = scan_snapshot(snapshot_path, iteration)
        except HomeAssistantError as exc:
            LOGGER.warning("Watch snapshot fetch failed during scan confirmation: %s", exc)
            if config.stop_on_error:
                raise
        except DetectorError as exc:
            LOGGER.warning("Watch detector failed: %s", exc)
            if config.stop_on_error:
                raise
        else:
            if result.matched:
                alert_time = clock()
                if _watch_cooldown_active(state, config, alert_time):
                    LOGGER.info("alert matched but cooldown active; no message sent")
                elif notify_alert(snapshot_path, result):
                    state.last_alert_at = alert_time
                else:
                    LOGGER.warning("Alert condition matched, but notification was not sent.")
            elif config.send_no_match_log:
                LOGGER.info("watch no alert: %s", result.matched_reason)

        _watch_iteration_finished(config, iteration, send_heartbeat, sleep)

    return state


def run_watch_stream(*, max_frames: int | None = None, keep_debug_frame: bool = False) -> int:
    """Continuously scan sampled live stream frames until stopped or limited."""
    if max_frames is not None and max_frames < 0:
        raise ValueError("--stream-frames must be zero or greater.")

    stream_config = load_stream_config()
    if keep_debug_frame:
        stream_config = replace(stream_config, save_debug_frames=True)

    detector = create_detector(load_detector_config())
    scan_config = load_scan_config()
    notifier: OpenClawClient | None = None

    def scan_frame(frame_path: Path) -> DetectionResult:
        return scan_image(frame_path, detector=detector, config=scan_config).result

    def notify_alert(frame_path: Path, result: DetectionResult) -> bool:
        nonlocal notifier
        notifier = notifier or OpenClawClient.from_env()
        return _notify_detection_result(frame_path, result, notifier=notifier)

    LOGGER.info("Starting live stream watch mode.")
    try:
        state = run_stream_watch_loop(
            config=stream_config,
            source_factory=lambda: create_stream_source(stream_config, frame_limit=max_frames or 0),
            scan_frame=scan_frame,
            notify_alert=notify_alert,
            max_frames=max_frames or 0,
        )
    except KeyboardInterrupt:
        LOGGER.info("Live stream watch mode stopped.")
        print("Live stream watch mode stopped.")
        return 0

    LOGGER.info(
        "Live stream watch mode finished after %d sampled frame(s) and %d stream failure(s).",
        state.sampled_frames,
        state.failures,
    )
    return 0


def run_stream_watch_loop(
    *,
    config: StreamConfig,
    source_factory: Callable[[], StreamFrameSource],
    scan_frame: Callable[[Path], DetectionResult],
    notify_alert: Callable[[Path, DetectionResult], bool],
    max_frames: int = 0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StreamWatchState:
    """Run stream reconnect, scan, notification, and cooldown behavior."""
    state = StreamWatchState()
    while max_frames == 0 or state.sampled_frames < max_frames:
        try:
            with source_factory() as source:
                for frame_path in source:
                    state.sampled_frames += 1
                    LOGGER.info("Stream frame sampled: %s", frame_path)
                    try:
                        result = scan_frame(frame_path)
                    except DetectorError as exc:
                        LOGGER.warning("Stream detector failed: %s", exc)
                    else:
                        _handle_stream_scan_result(
                            frame_path=frame_path,
                            result=result,
                            state=state,
                            config=config,
                            notify_alert=notify_alert,
                            alert_time=clock(),
                        )
                    finally:
                        if not config.save_debug_frames:
                            frame_path.unlink(missing_ok=True)

                    if max_frames and state.sampled_frames >= max_frames:
                        return state
        except StreamSourceError as exc:
            state.failures += 1
            LOGGER.warning("Stream failure: %s", exc)
            if config.max_failures and state.failures >= config.max_failures:
                LOGGER.error("Stream failure limit reached after %d failure(s).", state.failures)
                return state
            LOGGER.info("Reconnecting stream after %.1f second(s).", config.reconnect_seconds)
            if config.reconnect_seconds > 0:
                sleep(config.reconnect_seconds)

    return state


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
        frame_scan = scan_image(saved_path, detector=detector, config=load_scan_config())
        result = frame_scan.result
    except DetectorError as exc:
        LOGGER.error("Detector failed for %s: %s", saved_path, exc)
        message = f"What I see: detector failed: {exc}"
        confidence = 0.0
    else:
        print_result(result)
        crop_path = frame_scan.best_crop_path or best_crop_path(result)
        message = scanner_summary(result, best_crop_path=crop_path)
        confidence = best_dog_confidence(result)

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
        result = scan_image(image_path, detector=detector, config=load_scan_config()).result
    except DetectorError as exc:
        LOGGER.error("Detector failed for %s: %s", image_path, exc)
        print("No alert sent: detector failed.")
        return False

    print_result(result)
    if not result.matched:
        print(f"No alert sent: {result.matched_reason}.")
        return False

    return _notify_detection_result(image_path, result, notifier=notifier)


def _notify_detection_result(
    image_path: Path,
    result: DetectionResult,
    *,
    notifier: OpenClawClient | None = None,
) -> bool:
    """Send the dog alert for an already computed scanner result."""
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


def _watch_snapshot_path(iteration: int, *, frame_index: int = 0) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_suffix = "" if frame_index == 0 else f"_frame{frame_index + 1}"
    return SNAPSHOT_DIR / f"watch_{timestamp}_iter{iteration}{frame_suffix}.jpg"


def _watch_cooldown_active(state: WatchState, config: WatchConfig, alert_time: float) -> bool:
    if state.last_alert_at is None:
        return False
    return alert_time - state.last_alert_at < config.cooldown_seconds


def _watch_iteration_finished(
    config: WatchConfig,
    iteration: int,
    send_heartbeat: Callable[[int], bool] | None,
    sleep: Callable[[float], None],
) -> None:
    if send_heartbeat is not None and config.heartbeat_every > 0 and iteration % config.heartbeat_every == 0:
        if send_heartbeat(iteration):
            LOGGER.info("Watch heartbeat sent after iteration %d.", iteration)
        else:
            LOGGER.warning("Watch heartbeat notification failed after iteration %d.", iteration)

    if config.max_iterations and iteration >= config.max_iterations:
        return
    if config.interval_seconds > 0:
        sleep(config.interval_seconds)


def _handle_stream_scan_result(
    *,
    frame_path: Path,
    result: DetectionResult,
    state: StreamWatchState,
    config: StreamConfig,
    notify_alert: Callable[[Path, DetectionResult], bool],
    alert_time: float,
) -> None:
    if not result.matched:
        LOGGER.info("stream no alert: %s", result.matched_reason)
        return

    LOGGER.info("Stream detector matched: %s", result.matched_reason)
    if state.last_alert_at is not None and alert_time - state.last_alert_at < config.alert_cooldown_seconds:
        LOGGER.info("stream alert matched but cooldown active; no message sent")
        return

    if notify_alert(frame_path, result):
        state.last_alert_at = alert_time
        return
    LOGGER.warning("Stream alert condition matched, but notification was not sent.")


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
        if args.watch:
            return run_watch(max_iterations=args.watch_iterations)
        if args.watch_stream:
            return run_watch_stream(max_frames=args.stream_frames, keep_debug_frame=args.stream_save_debug_frame)
        if args.test_openclaw:
            return run_test_openclaw()
        if args.test_detect:
            return run_test_detect(args.test_detect)
        if args.what_see:
            return run_what_see()
    except (ConfigError, DetectorError, HomeAssistantError, StreamSourceError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
