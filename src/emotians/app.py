"""Console entry point for the EMOTIANS staged console application."""

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
    parser.add_argument(
        "--mode",
        choices=("preview", "prepare-data", "download-fer2013", "prepare-fer2013", "train", "evaluate"),
        default="preview",
        help="Application mode",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/expressions"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/stage_2"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint required for evaluate mode")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string")
    parser.add_argument("--seed", type=int, default=42)
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
        if args.mode != "preview":
            return _run_stage_two(args)
        config = load_config(args.config)
        return run_preview(config, args.camera)
    except (RuntimeError, ValueError) as error:
        LOGGER.error("Application error: %s", error)
        return 1


def _run_stage_two(args: argparse.Namespace) -> int:
    """Run the data, training or evaluation command introduced in Stage 2."""

    from emotians.training.cli import (
        evaluate_classifier,
        download_and_prepare_fer2013,
        format_report,
        prepare_data,
        prepare_fer2013,
        train_classifier,
    )

    if args.mode == "prepare-data":
        report = prepare_data(args.data_root, args.manifest, args.seed)
    elif args.mode == "download-fer2013":
        report = download_and_prepare_fer2013(args.manifest, args.seed)
    elif args.mode == "prepare-fer2013":
        report = prepare_fer2013(args.data_root, args.manifest, args.seed)
    elif args.mode == "train":
        if args.epochs < 1 or args.batch_size < 1:
            raise ValueError("--epochs and --batch-size must be positive integers.")
        report = train_classifier(
            args.data_root,
            args.manifest,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.device,
            args.seed,
        )
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be a positive integer.")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode evaluate is used.")
        report = evaluate_classifier(
            args.data_root, args.manifest, args.checkpoint, args.batch_size, args.device
        )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
