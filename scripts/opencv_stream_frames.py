from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample JPEG frames from one OpenCV camera stream.")
    parser.add_argument("--url", required=True, help="Stream URL loaded from local runtime configuration.")
    parser.add_argument("--output-dir", required=True, help="Directory where sampled JPEG frames are written.")
    parser.add_argument("--frame-interval", type=float, default=5.0, help="Seconds between sampled frames.")
    return parser


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def next_frame_path(output_dir: Path, sample_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"stream_{timestamp}_{sample_index}.jpg"


def main() -> int:
    args = build_parser().parse_args()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        emit({"type": "error", "error": "OpenCV import failed in stream helper. Install cv2 for the stream Python."})
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.url)
    if not capture.isOpened():
        emit({"type": "error", "error": "OpenCV could not open the configured stream."})
        return 3

    emit({"type": "connected"})
    sample_index = 0
    last_sample_at: float | None = None
    frame_interval = max(0.0, args.frame_interval)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                emit({"type": "error", "error": "OpenCV could not read the next stream frame."})
                return 4

            now = time.monotonic()
            if last_sample_at is not None and frame_interval > 0 and now - last_sample_at < frame_interval:
                continue

            sample_index += 1
            frame_path = next_frame_path(output_dir, sample_index)
            if not cv2.imwrite(str(frame_path), frame):
                emit({"type": "error", "error": "OpenCV could not save a sampled stream frame."})
                return 5
            last_sample_at = now
            emit({"type": "frame", "path": str(frame_path), "sample_index": sample_index})
    except KeyboardInterrupt:
        return 0
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
