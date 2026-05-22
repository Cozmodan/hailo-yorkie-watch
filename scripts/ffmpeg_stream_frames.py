from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample JPEG frames from one ffmpeg-readable stream.")
    parser.add_argument("--url", required=True, help="Stream URL loaded from local runtime configuration.")
    parser.add_argument("--output-dir", required=True, help="Directory where sampled JPEG frames are written.")
    parser.add_argument("--frame-interval", type=float, default=5.0, help="Seconds between sampled frames.")
    parser.add_argument("--backend", default="home_assistant", help="Stream backend name emitted with JSON events.")
    parser.add_argument("--bearer-token", help="Home Assistant bearer token passed to ffmpeg as an HTTP header.")
    return parser


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_output_pattern(output_dir: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(output_dir / f"stream_{timestamp}_%08d.jpg")


def build_ffmpeg_argv(
    *,
    stream_url: str,
    output_pattern: str,
    frame_interval: float,
    bearer_token: str = "",
) -> list[str]:
    """Build the ffmpeg frame-sampling command without invoking a shell."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if bearer_token:
        argv.extend(["-headers", f"Authorization: Bearer {bearer_token}\r\n"])
    argv.extend(["-i", stream_url, "-map", "0:v:0", "-an"])
    if frame_interval > 0:
        argv.extend(["-vf", f"fps={1.0 / frame_interval:g}"])
    argv.extend(["-q:v", "2", "-f", "image2", output_pattern])
    return argv


def emit_frame_events(output_dir: Path, pattern_prefix: str, emitted: set[Path], backend: str) -> None:
    frame_paths = sorted(output_dir.glob(f"{pattern_prefix}*.jpg"))
    for frame_path in frame_paths:
        if frame_path in emitted or not frame_path.exists() or frame_path.stat().st_size == 0:
            continue
        emitted.add(frame_path)
        emit(
            {
                "type": "frame",
                "ok": True,
                "source": "ffmpeg",
                "backend": backend,
                "frame_index": len(emitted),
                "sample_index": len(emitted),
                "frame_path": str(frame_path),
                "path": str(frame_path),
            }
        )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = build_output_pattern(output_dir)
    pattern_prefix = Path(output_pattern).name.split("%", maxsplit=1)[0]
    argv = build_ffmpeg_argv(
        stream_url=args.url,
        output_pattern=output_pattern,
        frame_interval=max(0.0, args.frame_interval),
        bearer_token=args.bearer_token or "",
    )

    try:
        process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        emit({"type": "error", "ok": False, "source": "ffmpeg", "backend": args.backend, "error": "ffmpeg was not found."})
        return 2
    except OSError as exc:
        emit(
            {
                "type": "error",
                "ok": False,
                "source": "ffmpeg",
                "backend": args.backend,
                "error": f"ffmpeg could not start: {exc}",
            }
        )
        return 3

    emit({"type": "connected", "ok": True, "source": "ffmpeg", "backend": args.backend})
    emitted: set[Path] = set()
    try:
        while process.poll() is None:
            emit_frame_events(output_dir, pattern_prefix, emitted, args.backend)
            time.sleep(0.1)
        emit_frame_events(output_dir, pattern_prefix, emitted, args.backend)
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        return 0

    returncode = process.wait()
    if returncode == 0:
        return 0
    emit(
        {
            "type": "error",
            "ok": False,
            "source": "ffmpeg",
            "backend": args.backend,
            "error": f"ffmpeg stream frame helper exited with returncode={returncode}.",
        }
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
