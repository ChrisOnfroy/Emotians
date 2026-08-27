"""Console entry point for the EMOTIANS staged console application."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

from emotians.config import AppConfig, load_config
from emotians.data.catalog import EMOTION_LABELS
from emotians.data.image_io import write_image
from emotians.utils.logging import configure_logging
from emotians.vision import centered_roi, draw_live_overlay, draw_preview_overlay

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/preview.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMOTIANS console prototype")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    parser.add_argument("--camera", type=int, default=None, help="Override the configured camera index")
    parser.add_argument(
        "--mode",
        choices=(
            "preview",
            "prepare-data",
            "download-fer2013",
            "prepare-fer2013",
            "prepare-fer2013-detector",
            "collect-expressions",
            "train",
            "evaluate",
            "train-detector",
            "evaluate-detector",
            "live-demo",
        ),
        default="preview",
        help="Application mode",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/expressions"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/stage_2"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint required for evaluate mode")
    parser.add_argument("--face-data-root", type=Path, default=Path("data/raw/faces"))
    parser.add_argument("--face-manifest", type=Path, default=Path("data/processed/face_boxes.csv"))
    parser.add_argument("--detector-output-dir", type=Path, default=Path("artifacts/runs/stage_3_detector"))
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("artifacts/models/face_detection_yunet_2023mar.onnx"),
        help="Pretrained YuNet ONNX model used by live-demo",
    )
    parser.add_argument("--expression-checkpoint", type=Path, default=None)
    parser.add_argument("--capture-root", type=Path, default=Path("data/raw/expressions"))
    parser.add_argument("--subject-id", default="subject_01")
    parser.add_argument("--capture-label", choices=EMOTION_LABELS, default="happy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--expression-threshold", type=float, default=0.45)
    parser.add_argument("--smoothing-alpha", type=float, default=0.55)
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
        config = load_config(args.config)
        if args.mode == "live-demo":
            return run_live_demo(config, args)
        if args.mode == "collect-expressions":
            return run_expression_capture(config, args)
        if args.mode != "preview":
            return _run_training_command(args)
        return run_preview(config, args.camera)
    except (RuntimeError, ValueError) as error:
        LOGGER.error("Application error: %s", error)
        return 1


def run_live_demo(config: AppConfig, args: argparse.Namespace) -> int:
    """Run Stage 3 detector and expression inference from the webcam."""

    if args.face_model is None:
        raise ValueError("--face-model is required when --mode live-demo is used.")
    if args.expression_checkpoint is None:
        raise ValueError("--expression-checkpoint is required when --mode live-demo is used.")

    from emotians.inference import ExpressionCNNClassifier, LabelSmoother, YuNetFaceDetector, process_frame

    detector = YuNetFaceDetector(args.face_model, args.confidence_threshold)
    classifier = ExpressionCNNClassifier(args.expression_checkpoint, args.device)
    smoother = LabelSmoother(args.smoothing_alpha)

    index = config.camera.index if args.camera is None else args.camera
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        LOGGER.error("Could not open camera %d. Check the camera index and permissions.", index)
        return 2

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    LOGGER.info("Camera %d opened for live demo. Press q or Escape to close.", index)
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("No frame received from camera %d; stopping live demo.", index)
                return 3
            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            result = process_frame(
                frame,
                detector,
                classifier,
                smoother,
                expression_threshold=args.expression_threshold,
            )
            overlay = draw_live_overlay(frame, result, fps=1.0 / elapsed)
            cv2.imshow(config.preview.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if _should_quit(key, config.preview.quit_keys):
                return 0
    except cv2.error as error:
        LOGGER.error("OpenCV live demo failed: %s", error)
        return 4
    finally:
        capture.release()
        cv2.destroyAllWindows()
        LOGGER.info("Live demo closed.")


def run_expression_capture(config: AppConfig, args: argparse.Namespace) -> int:
    """Collect labeled expression crops from the webcam using the Stage 1 ROI."""

    if not args.subject_id.strip():
        raise ValueError("--subject-id must be a non-empty value.")

    index = config.camera.index if args.camera is None else args.camera
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        LOGGER.error("Could not open camera %d. Check the camera index and permissions.", index)
        return 2

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    destination = args.capture_root / args.subject_id / args.capture_label
    LOGGER.info(
        "Capturing '%s' images for %s into %s. Press c to save, q or Escape to exit.",
        args.capture_label,
        args.subject_id,
        destination,
    )
    previous_time = time.perf_counter()
    saved = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("No frame received from camera %d; stopping capture.", index)
                return 3

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            roi = centered_roi(
                frame.shape,
                config.preview.roi_width_ratio,
                config.preview.roi_height_ratio,
            )
            message = (
                f"Label: {args.capture_label} | Subject: {args.subject_id} | "
                f"c: capture | saved: {saved}"
            )
            overlay = draw_preview_overlay(frame, roi, fps=1.0 / elapsed, message=message)
            cv2.imshow(config.preview.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if _should_quit(key, config.preview.quit_keys):
                return 0
            if chr(key).lower() == "c":
                left, top, right, bottom = roi
                crop = frame[top:bottom, left:right]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = destination / f"{args.subject_id}_{args.capture_label}_{timestamp}.jpg"
                if not write_image(output_path, crop):
                    LOGGER.error("Could not save capture to %s.", output_path)
                    return 5
                saved += 1
                LOGGER.info("Saved capture %d: %s", saved, output_path)
    except cv2.error as error:
        LOGGER.error("OpenCV capture failed: %s", error)
        return 4
    finally:
        capture.release()
        cv2.destroyAllWindows()
        LOGGER.info("Expression capture closed after %d saved image(s).", saved)


def _run_training_command(args: argparse.Namespace) -> int:
    """Run data, training or evaluation commands introduced in Stages 2 and 3."""

    from emotians.training.cli import (
        evaluate_face_detector,
        evaluate_classifier,
        download_and_prepare_fer2013,
        format_report,
        prepare_data,
        prepare_fer2013,
        prepare_fer2013_detector,
        train_face_detector,
        train_classifier,
    )

    if args.mode == "prepare-data":
        report = prepare_data(args.data_root, args.manifest, args.seed)
    elif args.mode == "download-fer2013":
        report = download_and_prepare_fer2013(args.manifest, args.seed)
    elif args.mode == "prepare-fer2013":
        report = prepare_fer2013(args.data_root, args.manifest, args.seed)
    elif args.mode == "prepare-fer2013-detector":
        report = prepare_fer2013_detector(args.data_root, args.manifest, args.face_manifest)
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
    elif args.mode == "evaluate":
        if args.batch_size < 1:
            raise ValueError("--batch-size must be a positive integer.")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode evaluate is used.")
        report = evaluate_classifier(
            args.data_root, args.manifest, args.checkpoint, args.batch_size, args.device
        )
    elif args.mode == "train-detector":
        if args.epochs < 1 or args.batch_size < 1:
            raise ValueError("--epochs and --batch-size must be positive integers.")
        report = train_face_detector(
            args.face_data_root,
            args.face_manifest,
            args.detector_output_dir,
            args.epochs,
            args.batch_size,
            args.input_size,
            args.device,
            args.seed,
            args.confidence_threshold,
        )
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be a positive integer.")
        if args.detector_checkpoint is None:
            raise ValueError("--detector-checkpoint is required when --mode evaluate-detector is used.")
        report = evaluate_face_detector(
            args.face_data_root,
            args.face_manifest,
            args.detector_checkpoint,
            args.batch_size,
            args.input_size,
            args.device,
            args.confidence_threshold,
        )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
