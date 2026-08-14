"""Validated configuration objects for the console preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    """Laptop camera settings."""

    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class PreviewConfig:
    """Visual settings used before any model is introduced."""

    window_name: str = "EMOTIANS — Stage 1 Preview"
    roi_width_ratio: float = 0.42
    roi_height_ratio: float = 0.58
    quit_keys: tuple[str, ...] = ("q", "ESC")


@dataclass(frozen=True)
class AppConfig:
    """Complete Stage 1 application configuration."""

    camera: CameraConfig = CameraConfig()
    preview: PreviewConfig = PreviewConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    A small explicit schema makes failures understandable before OpenCV opens a
    camera window. Unknown keys are rejected so configuration mistakes are not
    silently ignored.
    """

    config_path = Path(path)
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(content, dict):
        raise ValueError("Configuration root must be a mapping.")

    _reject_unknown_keys(content, {"camera", "preview"}, "root")
    camera_data = _mapping(content.get("camera", {}), "camera")
    preview_data = _mapping(content.get("preview", {}), "preview")
    _reject_unknown_keys(camera_data, {"index", "width", "height"}, "camera")
    _reject_unknown_keys(
        preview_data,
        {"window_name", "roi_width_ratio", "roi_height_ratio", "quit_keys"},
        "preview",
    )

    camera = CameraConfig(
        index=_integer(camera_data.get("index", 0), "camera.index", minimum=0),
        width=_integer(camera_data.get("width", 1280), "camera.width", minimum=1),
        height=_integer(camera_data.get("height", 720), "camera.height", minimum=1),
    )
    preview = PreviewConfig(
        window_name=_non_empty_string(
            preview_data.get("window_name", "EMOTIANS — Stage 1 Preview"),
            "preview.window_name",
        ),
        roi_width_ratio=_ratio(
            preview_data.get("roi_width_ratio", 0.42), "preview.roi_width_ratio"
        ),
        roi_height_ratio=_ratio(
            preview_data.get("roi_height_ratio", 0.58), "preview.roi_height_ratio"
        ),
        quit_keys=_quit_keys(preview_data.get("quit_keys", ["q", "ESC"])),
    )
    return AppConfig(camera=camera, preview=preview)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _reject_unknown_keys(values: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} configuration key(s): {', '.join(sorted(unknown))}")


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _ratio(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a number in the interval (0, 1].")
    return float(value)


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _quit_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("preview.quit_keys must be a non-empty list of strings.")
    keys = tuple(_non_empty_string(item, "preview.quit_keys item") for item in value)
    return keys
