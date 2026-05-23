from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class OpenClawConfig:
    """Runtime settings for OpenClaw notification delivery."""

    notify_mode: str
    url: str
    token: str
    event_endpoint: str
    ssh_host: str
    ssh_user: str
    ssh_port: int
    binary: str
    whatsapp_account: str
    whatsapp_target: str
    ssh_media_remote_dir: str
    ssh_media_command_template: str


@dataclass(frozen=True)
class DetectorConfig:
    """Runtime settings for object detection."""

    enabled: bool
    backend: str
    hef_path: str
    hailo_apps_root: str
    confidence_threshold: float
    target_classes: tuple[str, ...]
    timeout_seconds: float
    python_executable: str
    command_template: str


@dataclass(frozen=True)
class ScanConfig:
    """Runtime settings for multi-stage image scanning."""

    night_mode: str
    scan_tiles: str
    enable_crop_scan: bool
    enable_person_roi_scan: bool
    full_frame_dog_confidence: float
    crop_dog_confidence: float
    person_confidence: float
    confirm_frames: int
    confirm_interval_seconds: float
    max_crops_per_image: int
    save_debug_crops: bool


@dataclass(frozen=True)
class WatchConfig:
    """Runtime settings for continuous watch mode."""

    interval_seconds: float
    cooldown_seconds: float
    max_iterations: int
    send_no_match_log: bool
    heartbeat_every: int
    reuse_last_snapshot_on_ha_fail: bool
    stop_on_error: bool


@dataclass(frozen=True)
class StreamConfig:
    """Runtime settings for live camera stream watch mode."""

    enabled: bool
    url: str
    backend: str
    use_home_assistant: bool
    ha_base_url: str
    ha_stream_entity: str
    ha_stream_url: str
    ha_long_lived_token: str
    ha_stream_auth_mode: str
    frame_interval_seconds: float
    reconnect_seconds: float
    max_failures: int
    keep_frames: bool
    save_debug_frames: bool
    debug_dir: str
    retention_minutes: float
    max_frame_files: int
    debug_crop_retention_minutes: float
    debug_crop_max_files: int
    alert_cooldown_seconds: float
    python_executable: str


@dataclass(frozen=True)
class DogAlertConfig:
    """Runtime settings for dog alert filtering and evidence images."""

    min_confidence: float
    cooldown_seconds: float
    confirmation_frames: int
    min_box_area_ratio: float
    save_debug_frames: bool
    evidence_dir: str
    image_retention_seconds: float
    max_evidence_images: int


@dataclass(frozen=True)
class VLMConfig:
    """Runtime settings for optional local vision-language model reasoning."""

    enabled: bool
    base_url: str
    model: str
    timeout_seconds: float
    max_image_width: int
    prompt: str


def load_environment(env_path: str | Path | None = None) -> None:
    """Load environment variables from `.env` without overriding existing values."""
    loaded = load_dotenv(dotenv_path=env_path, override=False)
    if not loaded:
        LOGGER.debug("No .env file was loaded; using existing environment variables only.")


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Read an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if value is None or not value.strip():
        if required:
            message = f"Missing required environment variable: {name}"
            LOGGER.error(message)
            raise ConfigError(message)
        return ""
    return value.strip()


def get_int_env(name: str, default: int, *, required: bool = False) -> int:
    """Read an integer environment variable with validation."""
    raw_value = get_env(name, str(default), required=required) or str(default)
    try:
        return int(raw_value)
    except ValueError as exc:
        message = f"Environment variable {name} must be an integer."
        LOGGER.error(message)
        raise ConfigError(message) from exc


def get_float_env(name: str, default: float, *, required: bool = False) -> float:
    """Read a float environment variable with validation."""
    raw_value = get_env(name, str(default), required=required) or str(default)
    try:
        return float(raw_value)
    except ValueError as exc:
        message = f"Environment variable {name} must be a number."
        LOGGER.error(message)
        raise ConfigError(message) from exc


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean-like environment variable."""
    raw_value = get_env(name)
    if not raw_value:
        return default
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    message = f"Environment variable {name} must be a boolean value."
    LOGGER.error(message)
    raise ConfigError(message)


def get_csv_env(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated list of values."""
    raw_value = get_env(name, default) or default
    values = tuple(value.strip().lower() for value in raw_value.split(",") if value.strip())
    return values or tuple(value.strip().lower() for value in default.split(",") if value.strip())


def load_openclaw_config() -> OpenClawConfig:
    """Load OpenClaw notification settings from `.env` / process environment."""
    load_environment()
    notify_mode = (get_env("OPENCLAW_NOTIFY_MODE", "http") or "http").lower()
    if notify_mode not in {"http", "ssh", "disabled"}:
        message = "OPENCLAW_NOTIFY_MODE must be one of: http, ssh, disabled"
        LOGGER.error(message)
        raise ValueError(message)

    event_endpoint = get_env("OPENCLAW_EVENT_ENDPOINT", "/api/events/yorkie-watch") or "/api/events/yorkie-watch"

    if notify_mode == "http":
        return OpenClawConfig(
            notify_mode=notify_mode,
            url=get_env("OPENCLAW_URL", required=True),
            token=get_env("OPENCLAW_TOKEN", required=True),
            event_endpoint=event_endpoint,
            ssh_host="",
            ssh_user="",
            ssh_port=22,
            binary="openclaw",
            whatsapp_account="business",
            whatsapp_target=get_env("OPENCLAW_WHATSAPP_TARGET", required=True),
            ssh_media_remote_dir="",
            ssh_media_command_template="",
        )

    if notify_mode == "ssh":
        return OpenClawConfig(
            notify_mode=notify_mode,
            url="",
            token="",
            event_endpoint=event_endpoint,
            ssh_host=get_env("OPENCLAW_SSH_HOST", required=True),
            ssh_user=get_env("OPENCLAW_SSH_USER", required=True),
            ssh_port=get_int_env("OPENCLAW_SSH_PORT", 22),
            binary=get_env("OPENCLAW_BINARY", "openclaw") or "openclaw",
            whatsapp_account=get_env("OPENCLAW_WHATSAPP_ACCOUNT", "business") or "business",
            whatsapp_target=get_env("OPENCLAW_WHATSAPP_TARGET", required=True),
            ssh_media_remote_dir=get_env("OPENCLAW_SSH_MEDIA_REMOTE_DIR", "/tmp/yorkie-watch")
            or "/tmp/yorkie-watch",
            ssh_media_command_template=get_env("OPENCLAW_SSH_MEDIA_COMMAND_TEMPLATE"),
        )

    return OpenClawConfig(
        notify_mode=notify_mode,
        url="",
        token="",
        event_endpoint=event_endpoint,
        ssh_host="",
        ssh_user="",
        ssh_port=22,
        binary="openclaw",
        whatsapp_account="business",
        whatsapp_target="",
        ssh_media_remote_dir="",
        ssh_media_command_template="",
    )


def load_detector_config() -> DetectorConfig:
    """Load object detector settings from `.env` / process environment."""
    load_environment()
    dog_min_confidence = get_float_env("DOG_MIN_CONFIDENCE", 0.45)
    full_frame_dog_confidence = get_float_env(
        "YORKIE_FULL_FRAME_DOG_CONFIDENCE",
        get_float_env("YORKIE_DOG_CONFIDENCE", dog_min_confidence),
    )
    return DetectorConfig(
        enabled=get_bool_env("YORKIE_DETECTOR_ENABLED", False),
        backend=(get_env("YORKIE_DETECTOR_BACKEND", "hailo_apps") or "hailo_apps").lower(),
        hef_path=get_env("YORKIE_HAILO_HEF", "/usr/share/hailo-models/yolov8m_h10.hef")
        or "/usr/share/hailo-models/yolov8m_h10.hef",
        hailo_apps_root=get_env("YORKIE_HAILO_APPS_ROOT", "/home/pi/hailo-apps") or "/home/pi/hailo-apps",
        confidence_threshold=full_frame_dog_confidence,
        target_classes=get_csv_env("YORKIE_TARGET_CLASSES", "dog,person"),
        timeout_seconds=get_float_env("YORKIE_DETECTOR_TIMEOUT", 60.0),
        python_executable=get_env("YORKIE_HAILO_PYTHON", "python3") or "python3",
        command_template=get_env("YORKIE_HAILO_DETECT_COMMAND"),
    )


def load_scan_config() -> ScanConfig:
    """Load multi-stage scanner settings from `.env` / process environment."""
    load_environment()
    dog_min_confidence = get_float_env("DOG_MIN_CONFIDENCE", 0.45)
    return ScanConfig(
        night_mode=(get_env("YORKIE_NIGHT_MODE", "auto") or "auto").lower(),
        scan_tiles=(get_env("YORKIE_SCAN_TILES", "2x2") or "2x2").lower(),
        enable_crop_scan=get_bool_env("YORKIE_ENABLE_CROP_SCAN", True),
        enable_person_roi_scan=get_bool_env("YORKIE_ENABLE_PERSON_ROI_SCAN", True),
        full_frame_dog_confidence=get_float_env("YORKIE_FULL_FRAME_DOG_CONFIDENCE", dog_min_confidence),
        crop_dog_confidence=get_float_env("YORKIE_CROP_DOG_CONFIDENCE", dog_min_confidence),
        person_confidence=get_float_env("YORKIE_PERSON_CONFIDENCE", 0.35),
        confirm_frames=max(1, get_int_env("YORKIE_CONFIRM_FRAMES", get_int_env("DOG_CONFIRMATION_FRAMES", 2))),
        confirm_interval_seconds=max(0.0, get_float_env("YORKIE_CONFIRM_INTERVAL_SECONDS", 1.0)),
        max_crops_per_image=max(0, get_int_env("YORKIE_MAX_CROPS_PER_IMAGE", 8)),
        save_debug_crops=get_bool_env("YORKIE_SAVE_DEBUG_CROPS", get_bool_env("SAVE_DEBUG_FRAMES", False)),
    )


def load_watch_config() -> WatchConfig:
    """Load continuous watcher settings from `.env` / process environment."""
    load_environment()
    return WatchConfig(
        interval_seconds=max(0.0, get_float_env("YORKIE_WATCH_INTERVAL_SECONDS", 5.0)),
        cooldown_seconds=max(0.0, get_float_env("YORKIE_WATCH_COOLDOWN_SECONDS", 300.0)),
        max_iterations=max(0, get_int_env("YORKIE_WATCH_MAX_ITERATIONS", 0)),
        send_no_match_log=get_bool_env("YORKIE_WATCH_SEND_NO_MATCH_LOG", True),
        heartbeat_every=max(0, get_int_env("YORKIE_WATCH_HEARTBEAT_EVERY", 0)),
        reuse_last_snapshot_on_ha_fail=get_bool_env("YORKIE_WATCH_REUSE_LAST_SNAPSHOT_ON_HA_FAIL", False),
        stop_on_error=get_bool_env("YORKIE_WATCH_STOP_ON_ERROR", False),
    )


def load_stream_config() -> StreamConfig:
    """Load live camera stream settings from `.env` / process environment."""
    load_environment()
    return StreamConfig(
        enabled=get_bool_env("YORKIE_STREAM_ENABLED", False),
        url=get_env("YORKIE_STREAM_URL"),
        backend=(get_env("YORKIE_STREAM_BACKEND", "opencv") or "opencv").lower(),
        use_home_assistant=get_bool_env("YORKIE_STREAM_USE_HOME_ASSISTANT", False),
        ha_base_url=get_env("YORKIE_HA_BASE_URL"),
        ha_stream_entity=get_env("YORKIE_HA_STREAM_ENTITY"),
        ha_stream_url=get_env("YORKIE_HA_STREAM_URL"),
        ha_long_lived_token=get_env("YORKIE_HA_LONG_LIVED_TOKEN"),
        ha_stream_auth_mode=(get_env("YORKIE_HA_STREAM_AUTH_MODE", "bearer") or "bearer").lower(),
        frame_interval_seconds=max(0.0, get_float_env("YORKIE_STREAM_FRAME_INTERVAL", 5.0)),
        reconnect_seconds=max(0.0, get_float_env("YORKIE_STREAM_RECONNECT_SECONDS", 5.0)),
        max_failures=max(0, get_int_env("YORKIE_STREAM_MAX_FAILURES", 0)),
        keep_frames=get_bool_env("YORKIE_STREAM_KEEP_FRAMES", False),
        save_debug_frames=get_bool_env("YORKIE_STREAM_SAVE_DEBUG_FRAMES", get_bool_env("SAVE_DEBUG_FRAMES", False)),
        debug_dir=get_env("YORKIE_STREAM_DEBUG_DIR", "data/stream_frames") or "data/stream_frames",
        retention_minutes=max(0.0, get_float_env("YORKIE_STREAM_RETENTION_MINUTES", 60.0)),
        max_frame_files=max(0, get_int_env("YORKIE_STREAM_MAX_FRAME_FILES", 500)),
        debug_crop_retention_minutes=max(0.0, get_float_env("YORKIE_DEBUG_CROP_RETENTION_MINUTES", 60.0)),
        debug_crop_max_files=max(0, get_int_env("YORKIE_DEBUG_CROP_MAX_FILES", 500)),
        alert_cooldown_seconds=max(0.0, get_float_env("YORKIE_STREAM_ALERT_COOLDOWN_SECONDS", 300.0)),
        python_executable=get_env("YORKIE_STREAM_PYTHON", "python3") or "python3",
    )


def load_dog_alert_config() -> DogAlertConfig:
    """Load dog alert filtering and evidence image settings."""
    load_environment()
    return DogAlertConfig(
        min_confidence=max(0.0, get_float_env("DOG_MIN_CONFIDENCE", 0.45)),
        cooldown_seconds=max(0.0, get_float_env("DOG_ALERT_COOLDOWN_SECONDS", 180.0)),
        confirmation_frames=max(1, get_int_env("DOG_CONFIRMATION_FRAMES", 2)),
        min_box_area_ratio=max(0.0, get_float_env("DOG_MIN_BOX_AREA_RATIO", 0.01)),
        save_debug_frames=get_bool_env("SAVE_DEBUG_FRAMES", False),
        evidence_dir=get_env("DOG_EVIDENCE_DIR", "data/evidence") or "data/evidence",
        image_retention_seconds=max(0.0, get_float_env("IMAGE_RETENTION_SECONDS", 3600.0)),
        max_evidence_images=max(0, get_int_env("MAX_EVIDENCE_IMAGES", 100)),
    )


def load_vlm_config() -> VLMConfig:
    """Load optional local VLM settings from `.env` / process environment."""
    load_environment()
    enabled = get_bool_env("YORKIE_VLM_ENABLED", False)
    timeout_seconds = get_float_env("YORKIE_VLM_TIMEOUT_SECONDS", 60.0)
    max_image_width = get_int_env("YORKIE_VLM_MAX_IMAGE_WIDTH", 1280)
    if timeout_seconds <= 0:
        message = "YORKIE_VLM_TIMEOUT_SECONDS must be greater than zero."
        LOGGER.error(message)
        raise ConfigError(message)
    if max_image_width <= 0:
        message = "YORKIE_VLM_MAX_IMAGE_WIDTH must be greater than zero."
        LOGGER.error(message)
        raise ConfigError(message)

    return VLMConfig(
        enabled=enabled,
        base_url=get_env("YORKIE_VLM_BASE_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000",
        model=get_env("YORKIE_VLM_MODEL", required=enabled),
        timeout_seconds=timeout_seconds,
        max_image_width=max_image_width,
        prompt=get_env(
            "YORKIE_VLM_PROMPT",
            "Look at this image. Is there a dog or Yorkie? Briefly describe what you see and mention uncertainty.",
        )
        or "Look at this image. Is there a dog or Yorkie? Briefly describe what you see and mention uncertainty.",
    )
