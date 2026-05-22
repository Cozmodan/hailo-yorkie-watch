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

POLL_SECONDS = 0.1
STABLE_CHECK_SECONDS = 0.02


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


def build_output_pattern(output_dir: Path) -> Path:
    run_id = datetime.now().strftime("stream_%Y%m%d_%H%M%S_%f")
    return output_dir / f"{run_id}_%06d.jpg"


def build_ffmpeg_argv(
    *,
    stream_url: str,
    output_pattern: str | Path,
    frame_interval: float,
    frames: int = 0,
    bearer_token: str = "",
) -> list[str]:
    """Build one ffmpeg stream-sampling command without invoking a shell."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if bearer_token:
        argv.extend(["-headers", f"Authorization: Bearer {bearer_token}\r\n"])
    argv.extend(["-i", stream_url])
    if frame_interval > 0:
        argv.extend(["-vf", f"fps=1/{frame_interval:g}"])
    argv.extend(["-q:v", "2"])
    if frames > 0:
        argv.extend(["-frames:v", str(frames)])
    argv.extend(["-y", str(output_pattern)])
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


def emit_process_error(
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
            "error": f"ffmpeg stream sampling failed with returncode={returncode}: {safe_stderr}",
        }
    )


def start_ffmpeg(
    *,
    stream_url: str,
    output_pattern: Path,
    frame_interval: float,
    frames: int,
    bearer_token: str,
    backend: str,
) -> subprocess.Popen[str] | None:
    """Start the long-running ffmpeg sampler and emit a start error when needed."""
    argv = build_ffmpeg_argv(
        stream_url=stream_url,
        output_pattern=output_pattern,
        frame_interval=frame_interval,
        frames=frames,
        bearer_token=bearer_token,
    )
    try:
        return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        emit_process_error(
            backend=backend,
            returncode="not-found",
            stderr="ffmpeg was not found.",
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return None
    except OSError as exc:
        emit_process_error(
            backend=backend,
            returncode="start-error",
            stderr=f"ffmpeg could not start: {exc}",
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
        return None


def stable_frame_paths(output_pattern: Path, emitted: set[Path]) -> list[Path]:
    """Return newly written JPGs after a quick size stability check."""
    candidates = sorted(output_pattern.parent.glob(output_pattern.name.replace("%06d", "*")))
    stable: list[Path] = []
    for candidate in candidates:
        if candidate in emitted or not candidate.exists():
            continue
        first_size = candidate.stat().st_size
        if first_size <= 0:
            continue
        time.sleep(STABLE_CHECK_SECONDS)
        if candidate.exists() and candidate.stat().st_size == first_size:
            stable.append(candidate)
    return stable


def process_stderr(process: subprocess.Popen[str]) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read()


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


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
    frame_interval = max(0.0, args.frame_interval)
    output_pattern = build_output_pattern(output_dir)
    bearer_token = args.bearer_token or ""
    process = start_ffmpeg(
        stream_url=args.url,
        output_pattern=output_pattern,
        frame_interval=frame_interval,
        frames=args.frames,
        bearer_token=bearer_token,
        backend=args.backend,
    )
    if process is None:
        return 3

    emit({"type": "connected", "ok": True, "source": "ffmpeg", "backend": args.backend})
    emitted: set[Path] = set()
    try:
        while True:
            for frame_path in stable_frame_paths(output_pattern, emitted):
                emitted.add(frame_path)
                emit_frame(frame_path, frame_index=len(emitted), backend=args.backend)
                if args.frames and len(emitted) >= args.frames:
                    process.wait()
                    return 0

            returncode = process.poll()
            if returncode is not None:
                if args.frames == 0 and returncode == 0:
                    return 0
                emit_process_error(
                    backend=args.backend,
                    returncode=returncode,
                    stderr=process_stderr(process),
                    stream_url=args.url,
                    bearer_token=bearer_token,
                )
                return 4
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        stop_process(process)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
