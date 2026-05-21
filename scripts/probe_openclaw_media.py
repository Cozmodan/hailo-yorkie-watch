from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError, get_env, load_environment  # noqa: E402
from yorkie_watch.openclaw_client import OpenClawClient  # noqa: E402

HELP_COMMANDS = [
    [],
    ["message", "--help"],
    ["message", "send", "--help"],
    ["messages", "--help"],
    ["media", "--help"],
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_environment()
    notify_mode = (get_env("OPENCLAW_NOTIFY_MODE", "http") or "http").lower()
    if notify_mode != "ssh":
        print("OpenClaw media probe requires OPENCLAW_NOTIFY_MODE=ssh.")
        return 1

    try:
        client = OpenClawClient.from_env()
    except (ConfigError, ValueError) as exc:
        print(f"OpenClaw media probe failed: {exc}")
        return 1

    ok = True
    for args in HELP_COMMANDS:
        label = " ".join([client.binary, *args]) or client.binary
        print(f"\n$ {label}")
        try:
            returncode, stdout, stderr = client.run_openclaw_help(args)
        except Exception as exc:
            print(f"probe failed: {exc}")
            ok = False
            continue

        print(f"returncode: {returncode}")
        if stdout:
            print("stdout:")
            print(stdout)
        if stderr:
            print("stderr:")
            print(stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
