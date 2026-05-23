from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import DogAlertConfig
from .detector import COCO_DOG_CLASS_ID, Detection, DetectionResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DogAlertCandidate:
    """The dog detection that is strong enough to trigger an alert."""

    detection: Detection
    region: str
    confidence: float
    threshold: float
    area_ratio: float
    bbox_pixels: tuple[int, int, int, int]

    @property
    def reason(self) -> str:
        return f"dog confidence {self.confidence:.2f} >= {self.threshold:.2f}"


@dataclass(frozen=True)
class DogAlertEvaluation:
    """Dog alert policy result for one scanned image."""

    matched: bool
    reason: str
    candidate: DogAlertCandidate | None = None


def evaluate_dog_alert(
    image_path: str | Path,
    result: DetectionResult,
    config: DogAlertConfig,
) -> DogAlertEvaluation:
    """Select the best dog detection that passes confidence, region, and box-size filters."""
    if not result.ok:
        return DogAlertEvaluation(False, f"detector failed: {result.error or result.matched_reason}")

    dog_detections = sorted(
        (detection for detection in result.detections if _is_dog(detection)),
        key=lambda detection: detection.confidence,
        reverse=True,
    )
    if not dog_detections:
        return DogAlertEvaluation(False, "no dog detections found")

    image_size = _image_size(image_path)
    for detection in dog_detections:
        region = detection_region(detection)
        if detection.confidence < config.min_confidence:
            LOGGER.info(
                "Ignoring dog detection below DOG_MIN_CONFIDENCE: confidence %.2f < %.2f, region=%s",
                detection.confidence,
                config.min_confidence,
                region,
            )
            continue
        bbox = detection.bbox
        if bbox is None:
            LOGGER.warning(
                "Ignoring dog detection because bbox was unavailable: confidence %.2f, region=%s",
                detection.confidence,
                region,
            )
            continue
        bbox_pixels = bbox_to_pixels(bbox, image_size)
        area_ratio = bbox_area_ratio(bbox, image_size)
        if bbox_pixels is None or area_ratio is None:
            LOGGER.warning(
                "Ignoring dog detection because bbox/image dimensions were unavailable: confidence %.2f, region=%s",
                detection.confidence,
                region,
            )
            continue
        if area_ratio < config.min_box_area_ratio:
            LOGGER.info(
                "Ignoring small dog detection: confidence %.2f, area_ratio %.4f < %.4f, region=%s",
                detection.confidence,
                area_ratio,
                config.min_box_area_ratio,
                region,
            )
            continue
        if image_size is not None and not bbox_center_is_in_region(detection, bbox_pixels, image_size):
            LOGGER.info(
                "Ignoring dog detection outside active region: confidence %.2f, region=%s",
                detection.confidence,
                region,
            )
            continue

        candidate = DogAlertCandidate(
            detection=detection,
            region=region,
            confidence=detection.confidence,
            threshold=config.min_confidence,
            area_ratio=area_ratio,
            bbox_pixels=bbox_pixels,
        )
        return DogAlertEvaluation(True, f"{region}: {candidate.reason}", candidate)

    return DogAlertEvaluation(
        False,
        f"no dog met alert confidence {config.min_confidence:.2f} and min box area ratio "
        f"{config.min_box_area_ratio:.4f}",
    )


def format_dog_alert_message(candidate: DogAlertCandidate, *, vlm_summary: str = "") -> str:
    """Build the WhatsApp alert text for one confirmed dog candidate."""
    lines = [
        f"Dog detected by Hailo Yorkie Watch: {candidate.region}",
        f"Detector: {candidate.reason}",
    ]
    if vlm_summary:
        lines.append(f"VLM: {vlm_summary}")
    return "\n".join(lines)


def annotate_dog_alert_image(
    image_path: str | Path,
    candidate: DogAlertCandidate,
    *,
    output_dir: str | Path,
    timestamp: datetime | None = None,
) -> Path:
    """Write one annotated alert evidence image and return its path."""
    source_path = Path(image_path)
    output_path = _evidence_path(source_path, output_dir=output_dir, timestamp=timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now()

    with Image.open(source_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size
        box = _clip_box(candidate.bbox_pixels, width, height)
        line_width = max(2, min(width, height) // 160)
        draw.rectangle(box, outline=(255, 40, 40), width=line_width)
        label_lines = [
            f"dog {candidate.confidence:.2f}",
            f"{candidate.region} {stamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        _draw_label(draw, box, label_lines)
        annotated.save(output_path, quality=92)

    LOGGER.info("Saved annotated dog alert evidence image: %s", output_path)
    return output_path


def detection_region(detection: Detection) -> str:
    """Return a short region or zone label for logs and alert text."""
    return detection.crop_id or detection.source or "full_frame"


def bbox_area_ratio(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int] | None,
) -> float | None:
    """Return the box area as a ratio of the full image area."""
    if _is_normalized_bbox(bbox):
        left, top, right, bottom = _clip_normalized_box(bbox)
        return max(0.0, right - left) * max(0.0, bottom - top)
    if image_size is None:
        return None
    width, height = image_size
    left, top, right, bottom = _clip_box(_bbox_to_ints(bbox), width, height)
    return ((right - left) * (bottom - top)) / float(width * height)


def bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Convert normalized or pixel bbox values to clipped pixel coordinates."""
    if image_size is None:
        if _is_normalized_bbox(bbox):
            left, top, right, bottom = _clip_normalized_box(bbox)
            return (
                int(left * 1000),
                int(top * 1000),
                max(int(right * 1000), int(left * 1000) + 1),
                max(int(bottom * 1000), int(top * 1000) + 1),
            )
        return None
    width, height = image_size
    left, top, right, bottom = bbox
    if _is_normalized_bbox(bbox):
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    return _clip_box((int(left), int(top), int(right), int(bottom)), width, height)


def bbox_center_is_in_region(
    detection: Detection,
    bbox_pixels: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    """Validate simple scanner regions when a detection carries a known crop label."""
    region = detection_region(detection)
    if region in {"", "full_frame", "person_roi"} or region.startswith("person_roi"):
        return True
    width, height = image_size
    left, top, right, bottom = bbox_pixels
    center_x = (left + right) / 2.0 / width
    center_y = (top + bottom) / 2.0 / height

    if region == "lower_half":
        return center_y >= 0.5
    if region == "center" or region == "center_zoom":
        return 0.175 <= center_x <= 0.825 and 0.175 <= center_y <= 0.825
    if region.startswith("tile_"):
        return _center_matches_tile_region(region, center_x, center_y)
    return True


def _is_dog(detection: Detection) -> bool:
    return detection.class_name == "dog" or detection.class_id == COCO_DOG_CLASS_ID


def _image_size(image_path: str | Path) -> tuple[int, int] | None:
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError as exc:
        LOGGER.warning("Could not read image dimensions for dog alert policy: %s", exc)
        return None


def _is_normalized_bbox(bbox: tuple[float, float, float, float]) -> bool:
    return max(abs(value) for value in bbox) <= 1.0


def _clip_normalized_box(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    left = min(max(0.0, left), 1.0)
    top = min(max(0.0, top), 1.0)
    right = min(max(left, right), 1.0)
    bottom = min(max(top, bottom), 1.0)
    return left, top, right, bottom


def _bbox_to_ints(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return int(left), int(top), int(right), int(bottom)


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = min(max(0, left), max(0, width - 1))
    top = min(max(0, top), max(0, height - 1))
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    return left, top, right, bottom


def _draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label_lines: list[str]) -> None:
    font = ImageFont.load_default()
    left, top, _right, _bottom = box
    line_sizes = [_text_size(draw, text, font) for text in label_lines]
    padding = 4
    label_width = max(width for width, _height in line_sizes) + padding * 2
    label_height = sum(height for _width, height in line_sizes) + padding * 2 + max(0, len(label_lines) - 1) * 2
    label_left = left
    label_top = max(0, top - label_height)
    label_right = label_left + label_width
    label_bottom = label_top + label_height
    draw.rectangle((label_left, label_top, label_right, label_bottom), fill=(0, 0, 0))
    y = label_top + padding
    for text, (_width, line_height) in zip(label_lines, line_sizes):
        draw.text((label_left + padding, y), text, fill=(255, 255, 255), font=font)
        y += line_height + 2


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def _center_matches_tile_region(region: str, center_x: float, center_y: float) -> bool:
    parts = region.split("_")
    vertical = next((part for part in parts if part in {"upper", "middle", "lower"}), "")
    horizontal = next((part for part in parts if part in {"left", "center", "right"}), "")
    if vertical == "upper" and center_y > 0.5:
        return False
    if vertical == "middle" and not 0.33 <= center_y <= 0.67:
        return False
    if vertical == "lower" and center_y < 0.5:
        return False
    if horizontal == "left" and center_x > 0.5:
        return False
    if horizontal == "center" and not 0.33 <= center_x <= 0.67:
        return False
    if horizontal == "right" and center_x < 0.5:
        return False
    return True


def _evidence_path(source_path: Path, *, output_dir: str | Path, timestamp: datetime | None) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return Path(output_dir) / f"{source_path.stem}_dog_alert_{stamp}.jpg"
