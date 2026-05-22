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
from uuid import uuid4

POLL_SECONDS = 0.1
STABLE_CHECK_SECONDS = 0.02
CONTINUOUS_NO_OUTPUT_TIMEOUT_SECONDS = 15.0
FALLBACK_BATCH_FRAMES = 3


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
    run_id = f"{datetime.now().strftime('stream_%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
    return output_dir / f"{run_id}_%06d.jpg"


def build_ffmpeg_argv(
    *,
    stream_url: str,
    output_pattern: str | Path,
    frame_interval: float,
    frames: int = 0,
    bearer_token: str = "",
) -> list[str]:
    """Build a finite batch capture or continuous ffmpeg sampling command."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if bearer_token:
        argv.extend(["-headers", f"Authorization: Bearer {bearer_token}\r\n"])
    if frames == 0:
        argv.extend(["-fflags", "+genpts", "-use_wallclock_as_timestamps", "1"])
    argv.extend(["-i", stream_url])
    if frames > 0:
        argv.extend(["-frames:v", str(frames)])
    elif frame_interval > 0:
        argv.extend(["-vf", f"fps=1/{frame_interval:g}"])
    argv.extend(["-q:v", "2"])
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


def output_glob(output_pattern: Path) -> str:
    return str(output_pattern).replace("%06d", "*")


def emit_frame_count_error(
    *,
    backend: str,
    output_pattern: Path,
    found_count: int,
    emitted_count: int,
    requested_count: int,
    stderr: str,
    stream_url: str,
    bearer_token: str,
) -> None:
    """Emit one redacted error when ffmpeg did not create enough frame files."""
    expected_glob = redact_error(output_glob(output_pattern), stream_url=stream_url, bearer_token=bearer_token)
    safe_stderr = redact_error(stderr.strip(), stream_url=stream_url, bearer_token=bearer_token)
    if found_count == 0:
        error = "ffmpeg exited 0 but no output frames were found"
    else:
        error = "ffmpeg exited 0 before the requested output frame count was found"
    detail = (
        f"{error}; expected_glob={expected_glob!r}; found_count={found_count}; "
        f"emitted_count={emitted_count}; requested_count={requested_count}"
    )
    if safe_stderr:
        detail = f"{detail}; stderr={safe_stderr}"
    emit(
        {
            "type": "error",
            "ok": False,
            "source": "ffmpeg",
            "backend": backend,
            "error": detail,
            "expected_glob": expected_glob,
            "found_count": found_count,
            "emitted_count": emitted_count,
            "requested_count": requested_count,
        }
    )


def bounded_max_attempts(frames: int) -> int:
    """Limit finite reconnect capture attempts for short Home Assistant streams."""
    return max(3, frames * 2)


def emit_attempt_limit_error(
    *,
    backend: str,
    output_pattern: Path,
    attempt_count: int,
    found_count: int,
    emitted_count: int,
    requested_count: int,
    stderr: str,
    stream_url: str,
    bearer_token: str,
) -> None:
    """Emit one redacted error when bounded capture stays short after retries."""
    expected_glob = redact_error(output_glob(output_pattern), stream_url=stream_url, bearer_token=bearer_token)
    safe_stderr = redact_error(stderr.strip(), stream_url=stream_url, bearer_token=bearer_token)
    detail = (
        f"bounded ffmpeg capture reached max attempts before requested output frame count was found; "
        f"attempt_count={attempt_count}; expected_glob={expected_glob!r}; found_count={found_count}; "
        f"emitted_count={emitted_count}; requested_count={requested_count}"
    )
    if safe_stderr:
        detail = f"{detail}; stderr={safe_stderr}"
    emit(
        {
            "type": "error",
            "ok": False,
            "source": "ffmpeg",
            "backend": backend,
            "error": detail,
            "attempt_count": attempt_count,
            "expected_glob": expected_glob,
            "found_count": found_count,
            "emitted_count": emitted_count,
            "requested_count": requested_count,
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


def run_ffmpeg_batch(
    *,
    stream_url: str,
    output_pattern: Path,
    frames: int,
    bearer_token: str,
    backend: str,
) -> subprocess.CompletedProcess[str] | None:
    """Capture a bounded ffmpeg frame batch and emit start errors."""
    argv = build_ffmpeg_argv(
        stream_url=stream_url,
        output_pattern=output_pattern,
        frame_interval=0.0,
        frames=frames,
        bearer_token=bearer_token,
    )
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
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


def frame_paths(output_pattern: Path) -> list[Path]:
    """Return complete non-empty frames written for one ffmpeg output run."""
    candidates = sorted(output_pattern.parent.glob(output_pattern.name.replace("%06d", "*")))
    return [candidate for candidate in candidates if candidate.exists() and candidate.stat().st_size > 0]


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


def emit_batch_frames(
    output_pattern: Path,
    *,
    backend: str,
    start_index: int = 0,
    max_frames: int = 0,
) -> list[Path]:
    batch_paths = frame_paths(output_pattern)
    if max_frames > 0:
        batch_paths = batch_paths[:max_frames]
    for frame_index, frame_path in enumerate(batch_paths, start=start_index + 1):
        emit_frame(frame_path, frame_index=frame_index, backend=backend)
    return batch_paths


def run_bounded_capture(
    *,
    stream_url: str,
    output_dir: Path,
    frames: int,
    bearer_token: str,
    backend: str,
) -> int:
    emit({"type": "connected", "ok": True, "source": "ffmpeg", "backend": backend})
    emitted_count = 0
    last_found_count = 0
    last_stderr = ""
    last_output_pattern: Path | None = None
    max_attempts = bounded_max_attempts(frames)
    for attempt in range(1, max_attempts + 1):
        output_pattern = build_output_pattern(output_dir)
        last_output_pattern = output_pattern
        completed = run_ffmpeg_batch(
            stream_url=stream_url,
            output_pattern=output_pattern,
            frames=frames - emitted_count,
            bearer_token=bearer_token,
            backend=backend,
        )
        if completed is None:
            return 3

        last_stderr = completed.stderr
        batch_paths = emit_batch_frames(
            output_pattern,
            backend=backend,
            start_index=emitted_count,
            max_frames=frames - emitted_count,
        )
        last_found_count = len(batch_paths)
        emitted_count += last_found_count
        if emitted_count >= frames:
            return 0
        if not batch_paths:
            if completed.returncode != 0:
                emit_process_error(
                    backend=backend,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                    stream_url=stream_url,
                    bearer_token=bearer_token,
                )
                return 4
            emit_frame_count_error(
                backend=backend,
                output_pattern=output_pattern,
                found_count=0,
                emitted_count=emitted_count,
                requested_count=frames,
                stderr=completed.stderr,
                stream_url=stream_url,
                bearer_token=bearer_token,
            )
            return 5
        if attempt == max_attempts:
            emit_attempt_limit_error(
                backend=backend,
                output_pattern=output_pattern,
                attempt_count=attempt,
                found_count=last_found_count,
                emitted_count=emitted_count,
                requested_count=frames,
                stderr=completed.stderr,
                stream_url=stream_url,
                bearer_token=bearer_token,
            )
            return 6

    if last_output_pattern is not None:
        emit_attempt_limit_error(
            backend=backend,
            output_pattern=last_output_pattern,
            attempt_count=max_attempts,
            found_count=last_found_count,
            emitted_count=emitted_count,
            requested_count=frames,
            stderr=last_stderr,
            stream_url=stream_url,
            bearer_token=bearer_token,
        )
    return 6


def no_output_timeout(frame_interval: float) -> float:
    return max(CONTINUOUS_NO_OUTPUT_TIMEOUT_SECONDS, frame_interval * 2)


def run_reconnect_batch_loop(
    *,
    stream_url: str,
    output_dir: Path,
    frame_interval: float,
    bearer_token: str,
    backend: str,
    start_index: int,
) -> int:
    """Fallback to small bounded captures when fps sampling produces no files."""
    frame_index = start_index
    try:
        while True:
            output_pattern = build_output_pattern(output_dir)
            completed = run_ffmpeg_batch(
                stream_url=stream_url,
                output_pattern=output_pattern,
                frames=FALLBACK_BATCH_FRAMES,
                bearer_token=bearer_token,
                backend=backend,
            )
            if completed is None:
                return 3
            if completed.returncode != 0:
                emit_process_error(
                    backend=backend,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                    stream_url=stream_url,
                    bearer_token=bearer_token,
                )
                return 4

            batch_paths = emit_batch_frames(output_pattern, backend=backend, start_index=frame_index)
            if not batch_paths:
                emit_frame_count_error(
                    backend=backend,
                    output_pattern=output_pattern,
                    found_count=0,
                    emitted_count=0,
                    requested_count=FALLBACK_BATCH_FRAMES,
                    stderr=completed.stderr,
                    stream_url=stream_url,
                    bearer_token=bearer_token,
                )
                return 5
            frame_index += len(batch_paths)
            if frame_interval > 0:
                time.sleep(frame_interval)
    except KeyboardInterrupt:
        return 0


def run_continuous_capture(
    *,
    stream_url: str,
    output_dir: Path,
    frame_interval: float,
    bearer_token: str,
    backend: str,
) -> int:
    output_pattern = build_output_pattern(output_dir)
    process = start_ffmpeg(
        stream_url=stream_url,
        output_pattern=output_pattern,
        frame_interval=frame_interval,
        frames=0,
        bearer_token=bearer_token,
        backend=backend,
    )
    if process is None:
        return 3

    emit({"type": "connected", "ok": True, "source": "ffmpeg", "backend": backend})
    emitted: set[Path] = set()
    started_at = time.monotonic()
    try:
        while True:
            for frame_path in stable_frame_paths(output_pattern, emitted):
                emitted.add(frame_path)
                emit_frame(frame_path, frame_index=len(emitted), backend=backend)

            returncode = process.poll()
            if returncode is not None:
                if returncode == 0 and not emitted:
                    emit_frame_count_error(
                        backend=backend,
                        output_pattern=output_pattern,
                        found_count=0,
                        emitted_count=0,
                        requested_count=0,
                        stderr=process_stderr(process),
                        stream_url=stream_url,
                        bearer_token=bearer_token,
                    )
                    return 5
                emit_process_error(
                    backend=backend,
                    returncode=returncode,
                    stderr=process_stderr(process),
                    stream_url=stream_url,
                    bearer_token=bearer_token,
                )
                return 4
            if not emitted and time.monotonic() - started_at >= no_output_timeout(frame_interval):
                stop_process(process)
                return run_reconnect_batch_loop(
                    stream_url=stream_url,
                    output_dir=output_dir,
                    frame_interval=frame_interval,
                    bearer_token=bearer_token,
                    backend=backend,
                    start_index=0,
                )
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        stop_process(process)
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
    frame_interval = max(0.0, args.frame_interval)
    bearer_token = args.bearer_token or ""
    if args.frames:
        return run_bounded_capture(
            stream_url=args.url,
            output_dir=output_dir,
            frames=args.frames,
            bearer_token=bearer_token,
            backend=args.backend,
        )
    return run_continuous_capture(
        stream_url=args.url,
        output_dir=output_dir,
        frame_interval=frame_interval,
        bearer_token=bearer_token,
        backend=args.backend,
    )


if __name__ == "__main__":
    raise SystemExit(main())
