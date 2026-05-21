from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError  # noqa: E402
from yorkie_watch.ha_client import HomeAssistantClient, HomeAssistantError  # noqa: E402

SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshots" / "test_snapshot.jpg"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        client = HomeAssistantClient.from_env()
        saved_path = client.save_snapshot(SNAPSHOT_PATH)
    except (ConfigError, HomeAssistantError) as exc:
        print(f"Failed to save Home Assistant snapshot: {exc}")
        return 1

    print(f"Saved snapshot: {saved_path}")
    print(f"File size: {saved_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
