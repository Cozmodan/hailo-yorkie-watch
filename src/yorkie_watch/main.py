from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from .config import ConfigError
from .ha_client import HomeAssistantClient, HomeAssistantError
from .openclaw_client import OpenClawClient

LOGGER = logging.getLogger(__name__)
SNAPSHOT_DIR = Path("data") / "snapshots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hailo Yorkie Watch plumbing CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Fetch one Home Assistant snapshot and save it locally.")
    mode.add_argument("--test-openclaw", action="store_true", help="Send one test event to OpenClaw.")
    return parser


def run_once() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
    client = HomeAssistantClient.from_env()
    saved_path = client.save_snapshot(output_path)
    print(f"Saved snapshot to {saved_path} ({saved_path.stat().st_size} bytes)")
    return 0


def run_test_openclaw() -> int:
    client = OpenClawClient.from_env()
    success = client.send_event(
        {
            "event_type": "yorkie_watch_test",
            "message": "Test alert from Hailo Yorkie Watch",
            "confidence": 0.0,
        }
    )
    if success:
        print("OpenClaw test event sent successfully.")
        return 0

    print("OpenClaw test event failed. Check logs and OpenClaw connectivity.")
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    try:
        if args.once:
            return run_once()
        if args.test_openclaw:
            return run_test_openclaw()
    except (ConfigError, HomeAssistantError) as exc:
        LOGGER.error("%s", exc)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
