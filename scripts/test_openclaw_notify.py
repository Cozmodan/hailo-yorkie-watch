from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError  # noqa: E402
from yorkie_watch.openclaw_client import OpenClawClient  # noqa: E402

TEST_MESSAGE = "Test alert from Hailo Yorkie Watch"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        client = OpenClawClient.from_env()
    except (ConfigError, ValueError) as exc:
        print(f"OpenClaw test failed: {exc}")
        return 1

    if client.notify_mode == "disabled":
        print("OpenClaw notification test skipped because OPENCLAW_NOTIFY_MODE=disabled.")
        return 0

    if client.send_message(TEST_MESSAGE):
        print(f"OpenClaw test message sent successfully via {client.notify_mode}.")
        return 0

    print("OpenClaw test message failed. Check logs and OpenClaw connectivity.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
