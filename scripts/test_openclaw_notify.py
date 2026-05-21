from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError  # noqa: E402
from yorkie_watch.openclaw_client import OpenClawClient  # noqa: E402

TEST_PAYLOAD = {
    "event_type": "yorkie_watch_test",
    "message": "Test alert from Hailo Yorkie Watch",
    "confidence": 0.0,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        client = OpenClawClient.from_env()
    except ConfigError as exc:
        print(f"OpenClaw test failed: {exc}")
        return 1

    if client.send_event(TEST_PAYLOAD):
        print("OpenClaw test event sent successfully.")
        return 0

    print("OpenClaw test event failed. Check logs and OpenClaw connectivity.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
