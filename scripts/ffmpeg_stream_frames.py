from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--frames", type=int, default=0, help="Stop after N sampled frames; zero runs forever.")
    return parser


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_frame_path(output_dir: Path, frame_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"stream_{timestamp}_{frame_index}.jpg"


def build_ffmpeg_argv(
    *,
    stream_url: str,
    output_file: str | Path,
    bearer_token: str = "",
) -> list[str]:
    """Build one ffmpeg single-frame capture command without invoking a shell."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if bearer_token:
        argv.extend(["-headers", f"Authorization: Bearer {bearer_token}\r\n"])
    argv.extend(["-i", stream_url, "-frames:v", "1", "-y", str(output_file)])
    return argv


def redact_error(output: str, *, stream_url: str, bearer_token: str) -> str:
    """Redact stream credentials from ffmpeg stderr before emitting JSON."""
    redacted = output
    for sensitive_value in (stream_url, bearer_token):
        if sensitive_value:
            redacted = redacted.replace(sensitive_value, "<redacted-stream-value>")
    redacted = re.sub(
        r"(?i)(Authorization:\s*Bearer\s+)[^\r\n]+",
        r"\1<redacted-stream-value>",
        redacted,
    )
    return re.sub(
        r"((?:https?|rtsp)://[^\s'\"?]+\?)[^\s'\"]+",
        r"\1<redacted-query>",
        redacted,
        flags=re.IGNORECASE,
    )


def emit_frame(frame_path: Path, *, frame_index: int, backend: str) -> None:
    emit(
        {
            "type": "frame",
            "ok": True,
            "source": "ffmpeg",
            "backend": backend,
            "frame_index": frame_index,
            "sample_index": frame_index,
            "frame_path": str(frame_path),
            "path": str(frame_path),
        }
    )


def emit_capture_error(
    *,
    backend: str,
    returncode: int | str,
    stderr: str,
    stream_url: str,
    bearer_token: str,
) -> None:
    safe_stderr = redact_error(stderr, stream_url=stream_url, bearer_token=bearer_token)
    emit(
        {
            "type": "error",
            "ok": False,
            "source": "ffmpeg",
            "backend": backend,
            "error": f"ffmpeg single-frame capture failed with returncode={returncode}: {safe_stderr}",
        }
    )


def capture_frame(
    *,
    stream_url: str,
    frame_path: Path,
    bearer_token: str,
    backend: str,
) -> int:
    """Run one ffmpeg capture and emit a redacted error on failure."""
    argv = build_ffmpeg_argv(stream_url=stream_url, output_file=frame_path, bearer_token=bearer_token)
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        emit_capture_error(
            backend=backend,
            returncode="not-found",
            stderr="ffmpeg was not found.",
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return 2
    except OSError as exc:
        emit_capture_error(
            backend=backend,
            returncode="start-error",
            stderr=f"ffmpeg could not start: {exc}",
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return 3

    if completed.returncode != 0:
        emit_capture_error(
            backend=backend,
            returncode=completed.returncode,
            stderr=completed.stderr,
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return 4
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        emit_capture_error(
            backend=backend,
            returncode=completed.returncode,
            stderr=completed.stderr or "ffmpeg did not create a non-empty JPEG frame.",
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return 5
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < 0:
        emit(
            {
                "type": "error",
                "ok": False,
                "source": "ffmpeg",
                "backend": args.backend,
                "error": "--frames must be zero or greater.",
            }
        )
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emit({"type": "connected", "ok": True, "source": "ffmpeg", "backend": args.backend})
    frame_interval = max(0.0, args.frame_interval)
    frame_index = 0
    try:
        while args.frames == 0 or frame_index < args.frames:
            frame_index += 1
            frame_path = build_frame_path(output_dir, frame_index)
            returncode = capture_frame(
                stream_url=args.url,
                frame_path=frame_path,
                bearer_token=args.bearer_token or "",
                backend=args.backend,
            )
            if returncode != 0:
                return returncode
            emit_frame(frame_path, frame_index=frame_index, backend=args.backend)
            if (args.frames == 0 or frame_index < args.frames) and frame_interval > 0:
                time.sleep(frame_interval)
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
