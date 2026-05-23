from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DogAlertConfig, StreamConfig

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
DEBUG_CROP_DIR = Path("data") / "debug_crops"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class CleanupStats:
    """Result of one bounded image cleanup pass."""

    directory: Path
    age_deleted: int = 0
    count_deleted: int = 0
    skipped: bool = False

    @property
    def deleted(self) -> int:
        return self.age_deleted + self.count_deleted


def cleanup_stream_artifacts(
    config: StreamConfig,
    *,
    allowed_data_dir: str | Path = PROJECT_DATA_DIR,
    now: float | None = None,
) -> tuple[CleanupStats, CleanupStats]:
    """Clean stream frame and debug crop image directories within project data."""
    current_time = time.time() if now is None else now
    stream_stats = cleanup_image_directory(
        config.debug_dir,
        retention_minutes=config.retention_minutes,
        max_files=config.max_frame_files,
        allowed_data_dir=allowed_data_dir,
        now=current_time,
        label="stream frames",
    )
    crop_stats = cleanup_image_directory(
        DEBUG_CROP_DIR,
        retention_minutes=config.debug_crop_retention_minutes,
        max_files=config.debug_crop_max_files,
        allowed_data_dir=allowed_data_dir,
        now=current_time,
        label="debug crops",
    )
    return stream_stats, crop_stats


def cleanup_evidence_artifacts(
    config: DogAlertConfig,
    *,
    allowed_data_dir: str | Path = PROJECT_DATA_DIR,
    now: float | None = None,
) -> CleanupStats:
    """Clean annotated alert evidence images within project data."""
    return cleanup_image_directory(
        config.evidence_dir,
        retention_minutes=config.image_retention_seconds / 60.0,
        max_files=config.max_evidence_images,
        allowed_data_dir=allowed_data_dir,
        now=now,
        label="alert evidence",
    )


def delete_image_file(
    path: str | Path,
    *,
    allowed_data_dir: str | Path = PROJECT_DATA_DIR,
    label: str = "image",
) -> bool:
    """Delete one generated image only when it lives inside the configured data directory."""
    image_path = _resolve_project_path(path)
    if not image_path.exists():
        return False
    allowed_path = _resolve_project_path(allowed_data_dir)
    if not _is_within(image_path, allowed_path):
        LOGGER.warning("Skipping %s delete outside project data directory: %s", label, image_path)
        return False
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        LOGGER.warning("Skipping %s delete for non-image file: %s", label, image_path)
        return False
    image_path.unlink(missing_ok=True)
    return True


def cleanup_image_directory(
    directory: str | Path,
    *,
    retention_minutes: float,
    max_files: int,
    allowed_data_dir: str | Path = PROJECT_DATA_DIR,
    now: float | None = None,
    label: str = "images",
) -> CleanupStats:
    """Delete old image files by age and count without leaving project data."""
    directory_path = _resolve_project_path(directory)
    allowed_path = _resolve_project_path(allowed_data_dir)
    if not _is_within(directory_path, allowed_path):
        LOGGER.warning("Skipping %s cleanup outside project data directory: %s", label, directory_path)
        return CleanupStats(directory=directory_path, skipped=True)

    if not directory_path.exists():
        LOGGER.debug("Skipping %s cleanup; directory does not exist: %s", label, directory_path)
        return CleanupStats(directory=directory_path)

    current_time = time.time() if now is None else now
    candidates = _image_files(directory_path)
    age_deleted = _delete_by_age(candidates, retention_minutes=retention_minutes, now=current_time)
    remaining = _image_files(directory_path)
    count_deleted = _delete_by_count(remaining, max_files=max_files)
    stats = CleanupStats(directory=directory_path, age_deleted=age_deleted, count_deleted=count_deleted)
    if stats.deleted:
        LOGGER.info(
            "Cleaned %d %s file(s) from %s (%d by age, %d by count).",
            stats.deleted,
            label,
            directory_path,
            stats.age_deleted,
            stats.count_deleted,
        )
    else:
        LOGGER.debug("No %s cleanup needed in %s.", label, directory_path)
    return stats


def _resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    return path_obj.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _image_files(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _delete_by_age(files: list[Path], *, retention_minutes: float, now: float) -> int:
    if retention_minutes <= 0:
        return 0
    cutoff = now - retention_minutes * 60.0
    deleted = 0
    for path in files:
        if path.stat().st_mtime >= cutoff:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def _delete_by_count(files: list[Path], *, max_files: int) -> int:
    if max_files < 0 or len(files) <= max_files:
        return 0
    deleted = 0
    oldest_files = sorted(files, key=lambda file_path: (file_path.stat().st_mtime, file_path.name))
    for path in oldest_files[: len(files) - max_files]:
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted
