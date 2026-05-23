from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATEST_EVENT_PATH = PROJECT_ROOT / "data" / "latest_event.json"


def write_latest_event(
    *,
    image_path: str | Path,
    detector_class: str,
    confidence: float,
    region: str,
    vlm_summary: str = "",
    state_path: str | Path = LATEST_EVENT_PATH,
    timestamp: datetime | None = None,
) -> Path:
    """Write non-secret metadata for the latest alert event."""
    output_path = Path(state_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": (timestamp or datetime.now()).isoformat(timespec="seconds"),
        "image_path": _display_path(Path(image_path)),
        "detector_class": detector_class,
        "confidence": round(float(confidence), 4),
        "region": region,
        "vlm_summary": vlm_summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Updated latest Yorkie Watch event state: %s", output_path)
    return output_path


def load_latest_event(state_path: str | Path = LATEST_EVENT_PATH) -> dict[str, Any] | None:
    """Load latest alert event metadata, returning None when it is unavailable."""
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read latest event state: %s", exc)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Latest event state was not a JSON object: %s", path)
        return None
    return payload


def latest_event_image_path(event: dict[str, Any]) -> Path | None:
    """Resolve the latest event image path from stored metadata."""
    raw_path = event.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    return image_path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
