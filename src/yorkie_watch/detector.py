from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DetectorConfig, load_detector_config

LOGGER = logging.getLogger(__name__)
COCO_PERSON_CLASS_ID = 0
COCO_DOG_CLASS_ID = 16
DEFAULT_BACKEND = "hailo_apps"


class DetectorError(RuntimeError):
    """Raised when detector setup or execution fails."""


@dataclass(frozen=True)
class Detection:
    """One normalized object detection."""

    class_name: str
    confidence: float
    class_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    source: str = "full_frame"
    crop_id: str = ""
    crop_path: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Detection":
        class_name = str(data.get("class_name") or data.get("label") or data.get("name") or "").strip().lower()
        class_id = data.get("class_id")
        confidence = data.get("confidence", data.get("score", 0.0))
        bbox = data.get("bbox")
        return cls(
            class_name=class_name,
            class_id=int(class_id) if class_id is not None and str(class_id).strip() else None,
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox[:4]) if isinstance(bbox, list | tuple) and len(bbox) >= 4 else None,
            source=str(data.get("source") or "full_frame"),
            crop_id=str(data.get("crop_id") or ""),
            crop_path=str(data.get("crop_path") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source": self.source,
        }
        if self.crop_id:
            data["crop_id"] = self.crop_id
        if self.crop_path:
            data["crop_path"] = self.crop_path
        return data


@dataclass(frozen=True)
class DetectionResult:
    """Normalized detector output and alert decision."""

    ok: bool
    backend: str
    image: str
    detections: tuple[Detection, ...]
    matched: bool
    matched_reason: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ok": self.ok,
            "backend": self.backend,
            "image": self.image,
            "detections": [detection.to_dict() for detection in self.detections],
            "matched": self.matched,
            "matched_reason": self.matched_reason,
        }
        if self.error:
            data["error"] = self.error
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class DisabledDetector:
    """Detector implementation used when detection is not enabled."""

    backend = "disabled"

    def detect(self, image_path: str | Path) -> DetectionResult:
        image = str(image_path)
        return DetectionResult(
            ok=True,
            backend=self.backend,
            image=image,
            detections=(),
            matched=False,
            matched_reason="detector disabled",
        )


class MockDetector:
    """Detector implementation for tests and dry-run development."""

    backend = "mock"

    def __init__(
        self,
        detections: tuple[Detection, ...] = (),
        *,
        target_classes: tuple[str, ...] = ("dog",),
        confidence_threshold: float = 0.35,
        error: str = "",
    ) -> None:
        self.detections = detections
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        self.error = error

    def detect(self, image_path: str | Path) -> DetectionResult:
        if self.error:
            raise DetectorError(self.error)
        return evaluate_detections(
            image_path=image_path,
            backend=self.backend,
            detections=self.detections,
            target_classes=self.target_classes,
            confidence_threshold=self.confidence_threshold,
        )


class HailoAppsDetector:
    """Detector adapter that invokes a JSON-emitting Hailo subprocess."""

    backend = DEFAULT_BACKEND

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def detect(self, image_path: str | Path) -> DetectionResult:
        image = Path(image_path)
        if not image.exists():
            raise DetectorError(f"Input image does not exist: {image}")

        argv = self._build_command(image)
        LOGGER.info("Running detector backend %s for %s.", self.backend, image)
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise DetectorError(f"Detector executable was not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            stdout = _short_output(exc.stdout)
            stderr = _short_output(exc.stderr)
            raise DetectorError(
                f"Detector timed out after {self.config.timeout_seconds:g}s. stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        except OSError as exc:
            raise DetectorError(f"Could not start detector subprocess: {exc}") from exc

        if completed.returncode != 0:
            raise DetectorError(
                "Detector subprocess failed with "
                f"returncode={completed.returncode} stdout={_short_output(completed.stdout)!r} "
                f"stderr={_short_output(completed.stderr)!r}"
            )

        payload = _parse_json_output(completed.stdout)
        return result_from_payload(
            payload,
            image_path=image,
            backend=str(payload.get("backend") or self.backend),
            target_classes=("dog",),
            confidence_threshold=self.config.confidence_threshold,
        )

    def _build_command(self, image_path: Path) -> list[str]:
        substitutions = {
            "image": str(image_path),
            "hef": self.config.hef_path,
            "hailo_apps_root": self.config.hailo_apps_root,
            "threshold": str(self.config.confidence_threshold),
            "classes": ",".join(self.config.target_classes),
        }
        if self.config.command_template:
            command = self.config.command_template.format(**substitutions)
            return shlex.split(command, posix=os.name != "nt")

        wrapper_path = Path(__file__).resolve().parents[2] / "scripts" / "hailo_apps_detect_json.py"
        return [
            self.config.python_executable,
            str(wrapper_path),
            "--image",
            substitutions["image"],
            "--hef",
            substitutions["hef"],
            "--hailo-apps-root",
            substitutions["hailo_apps_root"],
            "--threshold",
            substitutions["threshold"],
            "--classes",
            substitutions["classes"],
        ]


def create_detector(config: DetectorConfig | None = None) -> DisabledDetector | MockDetector | HailoAppsDetector:
    """Create a detector for the configured backend."""
    config = config or load_detector_config()
    if not config.enabled:
        return DisabledDetector()
    if config.backend == "mock":
        return MockDetector(
            target_classes=config.target_classes,
            confidence_threshold=config.confidence_threshold,
        )
    if config.backend == DEFAULT_BACKEND:
        return HailoAppsDetector(config)
    raise ValueError(f"YORKIE_DETECTOR_BACKEND must be one of: {DEFAULT_BACKEND}, mock")


def evaluate_detections(
    *,
    image_path: str | Path,
    backend: str,
    detections: tuple[Detection, ...],
    target_classes: tuple[str, ...],
    confidence_threshold: float,
    ok: bool = True,
    error: str = "",
) -> DetectionResult:
    """Evaluate normalized detections against configured alert criteria."""
    targets = tuple(target.lower() for target in target_classes)
    for detection in detections:
        if _detection_matches(detection, targets, confidence_threshold):
            class_name = detection.class_name or _class_name_from_id(detection.class_id)
            reason = f"{class_name} confidence {detection.confidence:.2f} >= {confidence_threshold:.2f}"
            return DetectionResult(
                ok=ok,
                backend=backend,
                image=str(image_path),
                detections=detections,
                matched=True,
                matched_reason=reason,
                error=error,
            )

    return DetectionResult(
        ok=ok,
        backend=backend,
        image=str(image_path),
        detections=detections,
        matched=False,
        matched_reason=f"no target class met confidence threshold {confidence_threshold:.2f}",
        error=error,
    )


def result_from_payload(
    payload: dict[str, Any],
    *,
    image_path: str | Path,
    backend: str,
    target_classes: tuple[str, ...],
    confidence_threshold: float,
) -> DetectionResult:
    detections = tuple(Detection.from_mapping(item) for item in payload.get("detections", []))
    return evaluate_detections(
        image_path=str(payload.get("image") or image_path),
        backend=backend,
        detections=detections,
        target_classes=target_classes,
        confidence_threshold=confidence_threshold,
        ok=bool(payload.get("ok", True)),
        error=str(payload.get("error") or ""),
    )


def _detection_matches(detection: Detection, target_classes: tuple[str, ...], confidence_threshold: float) -> bool:
    if detection.confidence < confidence_threshold:
        return False
    if detection.class_name and detection.class_name.lower() in target_classes:
        return True
    return "dog" in target_classes and detection.class_id == COCO_DOG_CLASS_ID


def _class_name_from_id(class_id: int | None) -> str:
    if class_id == COCO_PERSON_CLASS_ID:
        return "person"
    if class_id == COCO_DOG_CLASS_ID:
        return "dog"
    return f"class {class_id}" if class_id is not None else "unknown"


def _parse_json_output(stdout: str) -> dict[str, Any]:
    output = stdout.strip()
    if not output:
        raise DetectorError("Detector did not emit JSON output.")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        start = output.rfind("\n{")
        if start >= 0:
            return json.loads(output[start + 1 :])
        raise DetectorError(f"Detector output was not valid JSON: {_short_output(output)!r}")


def _short_output(output: str | bytes | None, *, max_chars: int = 800) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output.strip()
    if len(output) <= max_chars:
        return output
    return f"{output[:max_chars]}..."


def detection_result_from_cli_error(image_path: str | Path, backend: str, error: str) -> DetectionResult:
    return DetectionResult(
        ok=False,
        backend=backend,
        image=str(image_path),
        detections=(),
        matched=False,
        matched_reason="detector failed",
        error=error,
    )


def print_result(result: DetectionResult) -> None:
    sys.stdout.write(result.to_json())
    sys.stdout.write("\n")
