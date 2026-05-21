from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from .config import ScanConfig
from .detector import COCO_DOG_CLASS_ID, COCO_PERSON_CLASS_ID, Detection, DetectionResult

LOGGER = logging.getLogger(__name__)
DEBUG_CROP_DIR = Path("data") / "debug_crops"


@dataclass(frozen=True)
class CropSpec:
    """One image crop to run through the detector."""

    crop_id: str
    source: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameScan:
    """Scanner output for one source image."""

    image_path: Path
    result: DetectionResult
    best_crop_path: Path | None = None


def scan_image(
    image_path: str | Path,
    *,
    detector: object,
    config: ScanConfig,
    debug_crop_dir: str | Path = DEBUG_CROP_DIR,
) -> FrameScan:
    """Run full-frame detection and optional zoom/crop passes for one image."""
    image = Path(image_path)
    full_result = detector.detect(image)  # type: ignore[attr-defined]
    full_detections = tuple(replace(detection, source="full_frame", crop_id="", crop_path="") for detection in full_result.detections)
    if not full_result.ok:
        return FrameScan(image, replace(full_result, detections=full_detections))

    full_match = _best_dog(full_detections, config.full_frame_dog_confidence)
    if full_match is not None or not config.enable_crop_scan:
        return FrameScan(
            image,
            _build_result(
                image_path=image,
                backend=full_result.backend,
                detections=full_detections,
                matched=full_match is not None,
                matched_reason=_match_reason(full_match, config.full_frame_dog_confidence, "full-frame dog")
                if full_match is not None
                else f"no dog met full-frame confidence threshold {config.full_frame_dog_confidence:.2f}",
            ),
        )

    crop_dir = Path(debug_crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    all_detections = list(full_detections)
    best_crop_path: Path | None = None
    generated_crop_paths: list[Path] = []

    try:
        with Image.open(image) as opened_image:
            opened_image.load()
            width, height = opened_image.size
            crop_specs = _build_crop_specs(width, height, full_detections, config)[: config.max_crops_per_image]
            for crop_spec in crop_specs:
                crop_path = _crop_path(crop_dir, image, crop_spec)
                opened_image.crop(crop_spec.box).save(crop_path)
                generated_crop_paths.append(crop_path)
                crop_result = detector.detect(crop_path)  # type: ignore[attr-defined]
                mapped = tuple(_map_crop_detection(detection, crop_spec, crop_path) for detection in crop_result.detections)
                all_detections.extend(mapped)
                if best_crop_path is None and _best_dog(mapped, config.crop_dog_confidence) is not None:
                    best_crop_path = crop_path
    finally:
        if not config.save_debug_crops:
            for generated_crop_path in generated_crop_paths:
                generated_crop_path.unlink(missing_ok=True)

    detections = tuple(all_detections)
    crop_match = _best_dog(
        (detection for detection in detections if detection.source != "full_frame"),
        config.crop_dog_confidence,
    )
    return FrameScan(
        image,
        _build_result(
            image_path=image,
            backend=full_result.backend,
            detections=detections,
            matched=crop_match is not None,
            matched_reason=_match_reason(crop_match, config.crop_dog_confidence, f"{crop_match.source} dog")
            if crop_match is not None
            else f"no dog met crop confidence threshold {config.crop_dog_confidence:.2f}",
        ),
        best_crop_path=best_crop_path,
    )


def scan_confirmed_snapshots(
    *,
    capture_snapshot: Callable[[int], Path],
    detector: object,
    config: ScanConfig,
    debug_crop_dir: str | Path = DEBUG_CROP_DIR,
    sleep: Callable[[float], None] = time.sleep,
) -> DetectionResult:
    """Capture and scan multiple frames, matching when enough frames detect a dog."""
    frame_count = max(1, config.confirm_frames)
    frame_scans: list[FrameScan] = []
    for frame_index in range(frame_count):
        if frame_index and config.confirm_interval_seconds > 0:
            sleep(config.confirm_interval_seconds)
        snapshot_path = capture_snapshot(frame_index)
        frame_scans.append(scan_image(snapshot_path, detector=detector, config=config, debug_crop_dir=debug_crop_dir))

    detections = tuple(detection for frame_scan in frame_scans for detection in frame_scan.result.detections)
    matched_frames = sum(1 for frame_scan in frame_scans if frame_scan.result.matched)
    first_result = frame_scans[0].result
    return _build_result(
        image=first_result.image,
        backend=first_result.backend,
        detections=detections,
        matched=matched_frames >= frame_count,
        matched_reason=f"dog detected in {matched_frames}/{frame_count} confirmed frame(s)",
    )


def scanner_summary(result: DetectionResult, *, best_crop_path: str | Path | None = None) -> str:
    """Build a plain-English scanner summary suitable for WhatsApp."""
    full_person_count = sum(1 for detection in result.detections if detection.source == "full_frame" and _is_person(detection, 0.0))
    full_dog_count = sum(1 for detection in result.detections if detection.source == "full_frame" and _is_dog(detection, 0.0))
    crop_dogs = [detection for detection in result.detections if detection.source != "full_frame" and _is_dog(detection, 0.0)]
    lines = [
        "Yorkie Watch view:",
        f"Full-frame scan: {full_person_count} person(s), {full_dog_count} dog(s).",
    ]
    if crop_dogs:
        best = max(crop_dogs, key=lambda detection: detection.confidence)
        label = best.crop_id.replace("_", " ") or best.source.replace("_", " ")
        lines.append(f"Zoom scan: dog candidate found in {label} crop at {best.confidence:.2f} confidence.")
    else:
        lines.append("Zoom scan: no dog candidate found.")
    lines.append(f"Alert condition: {'matched' if result.matched else 'not matched'}.")
    lines.append(f"Snapshot: {Path(result.image).name}")
    if best_crop_path is not None:
        lines.append(f"Best crop: {Path(best_crop_path).name}")
    return "\n".join(lines)


def best_dog_confidence(result: DetectionResult) -> float:
    """Return the highest dog confidence in a scanner result."""
    best = _best_dog(result.detections, 0.0)
    return best.confidence if best is not None else 0.0


def best_crop_path(result: DetectionResult) -> Path | None:
    """Return the best crop path referenced by dog detections, if any."""
    crop_dogs = [detection for detection in result.detections if detection.source != "full_frame" and _is_dog(detection, 0.0)]
    if not crop_dogs:
        return None
    best = max(crop_dogs, key=lambda detection: detection.confidence)
    return Path(best.crop_path) if best.crop_path else None


def _build_crop_specs(width: int, height: int, detections: tuple[Detection, ...], config: ScanConfig) -> list[CropSpec]:
    crop_specs = [
        CropSpec("center", "center_zoom", _center_crop(width, height)),
        CropSpec("lower_half", "lower_half", (0, height // 2, width, height)),
    ]
    crop_specs.extend(_tile_specs(width, height, config.scan_tiles))
    if config.enable_person_roi_scan:
        for index, detection in enumerate(
            (detection for detection in detections if _is_person(detection, config.person_confidence)),
            start=1,
        ):
            roi = _expanded_person_roi(detection, width, height)
            if roi is not None:
                crop_specs.append(CropSpec(f"person_roi_{index}", "person_roi", roi))
    return _deduplicate_crops(crop_specs)


def _center_crop(width: int, height: int) -> tuple[int, int, int, int]:
    crop_width = max(1, int(width * 0.65))
    crop_height = max(1, int(height * 0.65))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return (left, top, min(width, left + crop_width), min(height, top + crop_height))


def _tile_specs(width: int, height: int, tile_mode: str) -> list[CropSpec]:
    if tile_mode == "3x3":
        grid = 3
    elif tile_mode == "2x2":
        grid = 2
    else:
        return []

    vertical_names = {0: "upper", 1: "middle", 2: "lower"} if grid == 3 else {0: "upper", 1: "lower"}
    horizontal_names = {0: "left", 1: "center", 2: "right"} if grid == 3 else {0: "left", 1: "right"}
    specs: list[CropSpec] = []
    for row in range(grid):
        for column in range(grid):
            left = width * column // grid
            top = height * row // grid
            right = width * (column + 1) // grid
            bottom = height * (row + 1) // grid
            specs.append(
                CropSpec(
                    f"tile_{vertical_names[row]}_{horizontal_names[column]}",
                    "tile",
                    (left, top, right, bottom),
                )
            )
    return specs


def _expanded_person_roi(detection: Detection, width: int, height: int) -> tuple[int, int, int, int] | None:
    if detection.bbox is None:
        return None
    left, top, right, bottom = _bbox_to_pixels(detection.bbox, width, height)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    expanded_left = int(left - box_width * 0.75)
    expanded_right = int(right + box_width * 0.75)
    expanded_top = int(top - box_height * 0.20)
    expanded_bottom = int(bottom + box_height * 0.80)
    return _clip_box((expanded_left, expanded_top, expanded_right, expanded_bottom), width, height)


def _bbox_to_pixels(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    if max(abs(value) for value in bbox) <= 1.0:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    return _clip_box((int(left), int(top), int(right), int(bottom)), width, height)


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = min(max(0, left), max(0, width - 1))
    top = min(max(0, top), max(0, height - 1))
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    return (left, top, right, bottom)


def _deduplicate_crops(crop_specs: list[CropSpec]) -> list[CropSpec]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[CropSpec] = []
    for crop_spec in crop_specs:
        if crop_spec.box in seen:
            continue
        seen.add(crop_spec.box)
        unique.append(crop_spec)
    return unique


def _crop_path(crop_dir: Path, image_path: Path, crop_spec: CropSpec) -> Path:
    return crop_dir / f"{image_path.stem}_crop_{crop_spec.crop_id}.jpg"


def _map_crop_detection(detection: Detection, crop_spec: CropSpec, crop_path: Path) -> Detection:
    left, top, _right, _bottom = crop_spec.box
    bbox = detection.bbox
    if bbox is not None:
        bbox = (bbox[0] + left, bbox[1] + top, bbox[2] + left, bbox[3] + top)
    return replace(
        detection,
        bbox=bbox,
        source=crop_spec.source,
        crop_id=crop_spec.crop_id,
        crop_path=str(crop_path),
    )


def _best_dog(detections: object, confidence_threshold: float) -> Detection | None:
    dogs = [detection for detection in detections if _is_dog(detection, confidence_threshold)]  # type: ignore[union-attr]
    if not dogs:
        return None
    return max(dogs, key=lambda detection: detection.confidence)


def _is_dog(detection: Detection, confidence_threshold: float) -> bool:
    return detection.confidence >= confidence_threshold and (
        detection.class_name == "dog" or detection.class_id == COCO_DOG_CLASS_ID
    )


def _is_person(detection: Detection, confidence_threshold: float) -> bool:
    return detection.confidence >= confidence_threshold and (
        detection.class_name == "person" or detection.class_id == COCO_PERSON_CLASS_ID
    )


def _match_reason(detection: Detection, threshold: float, label: str) -> str:
    return f"{label} confidence {detection.confidence:.2f} >= {threshold:.2f}"


def _build_result(
    *,
    image_path: Path | None = None,
    image: str | None = None,
    backend: str,
    detections: tuple[Detection, ...],
    matched: bool,
    matched_reason: str,
) -> DetectionResult:
    return DetectionResult(
        ok=True,
        backend=backend,
        image=image or str(image_path),
        detections=detections,
        matched=matched,
        matched_reason=matched_reason,
    )