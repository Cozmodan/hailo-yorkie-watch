from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError  # noqa: E402
from yorkie_watch.openclaw_vision_tools import (  # noqa: E402
    DEFAULT_CAMERA_PROMPT,
    DEFAULT_LATEST_ALERT_PROMPT,
    OpenClawVisionToolClient,
    format_vision_reply,
    handle_whatsapp_vision_message,
    select_vision_route,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw-side Yorkie Watch vision tool tests")
    parser.add_argument("--debug", action="store_true", help="Include raw JSON payloads in printed replies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera = subparsers.add_parser("test-camera", help="Call /vision/camera-snapshot.")
    camera.add_argument("--prompt", default=DEFAULT_CAMERA_PROMPT)

    latest = subparsers.add_parser("test-latest-alert", help="Call /vision/latest-alert.")
    latest.add_argument("--prompt", default=DEFAULT_LATEST_ALERT_PROMPT)

    route = subparsers.add_parser("route", help="Route one inbound WhatsApp-style message.")
    route.add_argument("--message", required=True)

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    try:
        client = OpenClawVisionToolClient.from_env()
    except (ConfigError, ValueError) as exc:
        print(f"OpenClaw Yorkie vision config error: {exc}")
        return 1

    if args.command == "test-camera":
        result = client.camera_snapshot(prompt=args.prompt)
        print(format_vision_reply(result, debug=args.debug))
        return 0 if result.ok else 1

    if args.command == "test-latest-alert":
        result = client.latest_alert(prompt=args.prompt)
        print(format_vision_reply(result, debug=args.debug))
        return 0 if result.ok else 1

    if args.command == "route":
        route = select_vision_route(args.message)
        if route is None:
            print("No Yorkie Watch vision route matched that message.")
            return 2
        interaction = handle_whatsapp_vision_message(args.message, client=client, debug=args.debug)
        print(interaction.reply_text)
        if args.debug:
            print(json.dumps({"route": interaction.route, "sent": interaction.sent}, sort_keys=True))
        return 0 if interaction.result and interaction.result.ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
