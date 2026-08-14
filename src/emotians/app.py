"""Console entry point for the Stage 1 OpenCV webcam preview."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from emotians.config import AppConfig, load_config
from emotians.utils.logging import configure_logging
from emotians.vision import centered_roi, draw_preview_overlay

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/preview.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMOTIANS console prototype")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    parser.add_argument("--camera", type=int, default=None, help="Override the configured camera index")
    parser.add_argument("--mode", choices=("preview",), default="preview", help="Available Stage 1 mode")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def run_preview(config: AppConfig, camera_index: int | None = None) -> int:
    """Run the model-free camera preview until the user presses q or Escape."""

    index = config.camera.index if camera_index is None else camera_index
    if index < 0:
        raise ValueError("camera index must be zero or greater.")

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        LOGGER.error("Could not open camera %d. Check the camera index and permissions.", index)
        return 2

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    LOGGER.info("Camera %d opened. Press q or Escape to close the preview.", index)
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("No frame received from camera %d; stopping preview.", index)
                return 3

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            roi = centered_roi(
                frame.shape,
                config.preview.roi_width_ratio,
                config.preview.roi_height_ratio,
            )
            overlay = draw_preview_overlay(frame, roi, fps=1.0 / elapsed)
            cv2.imshow(config.preview.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if _should_quit(key, config.preview.quit_keys):
                return 0
    except cv2.error as error:
        LOGGER.error("OpenCV preview failed: %s", error)
        return 4
    finally:
        capture.release()
        cv2.destroyAllWindows()
        LOGGER.info("Camera preview closed.")


def _should_quit(key_code: int, quit_keys: tuple[str, ...]) -> bool:
    normalized = {key.upper() for key in quit_keys}
    return key_code == 27 and "ESC" in normalized or chr(key_code).upper() in normalized


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        config = load_config(args.config)
        return run_preview(config, args.camera)
    except ValueError as error:
        LOGGER.error("Configuration error: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
