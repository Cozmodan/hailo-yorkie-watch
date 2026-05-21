from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yorkie_watch.config import ConfigError, load_detector_config  # noqa: E402
from yorkie_watch.detector import (  # noqa: E402
    DetectorError,
    create_detector,
    detection_result_from_cli_error,
    print_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Yorkie Watch detector check against an image.")
    parser.add_argument("image", help="Path to a snapshot image.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    image_path = Path(args.image)

    try:
        config = load_detector_config()
        detector = create_detector(config)
        result = detector.detect(image_path)
    except (ConfigError, ValueError) as exc:
        print_result(detection_result_from_cli_error(image_path, "configured", str(exc)))
        return 1
    except DetectorError as exc:
        print_result(detection_result_from_cli_error(image_path, "hailo_apps", str(exc)))
        return 1

    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
