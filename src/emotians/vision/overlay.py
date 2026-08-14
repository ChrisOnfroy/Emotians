"""Pure-ish OpenCV rendering helpers for the Stage 1 webcam preview."""

from __future__ import annotations

import cv2
import numpy as np


BoundingBox = tuple[int, int, int, int]


def centered_roi(frame_shape: tuple[int, ...], width_ratio: float, height_ratio: float) -> BoundingBox:
    """Return a centered ``(left, top, right, bottom)`` guidance box."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must include height and width.")
    frame_height, frame_width = frame_shape[:2]
    if frame_height < 1 or frame_width < 1:
        raise ValueError("frame dimensions must be positive.")
    if not 0 < width_ratio <= 1 or not 0 < height_ratio <= 1:
        raise ValueError("ROI ratios must be in the interval (0, 1].")

    roi_width = max(1, round(frame_width * width_ratio))
    roi_height = max(1, round(frame_height * height_ratio))
    left = (frame_width - roi_width) // 2
    top = (frame_height - roi_height) // 2
    return left, top, left + roi_width, top + roi_height


def draw_preview_overlay(
    frame_bgr: np.ndarray,
    roi: BoundingBox,
    fps: float,
    message: str = "Collecting data — center your face in the guide",
) -> np.ndarray:
    """Copy a frame and draw the model-free Stage 1 guidance overlay."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image.")

    left, top, right, bottom = roi
    canvas = frame_bgr.copy()
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 215, 255), 2)
    cv2.putText(
        canvas,
        "STAGE 1: CAMERA PREVIEW",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message,
        (16, max(60, canvas.shape[0] - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}   |   q / Esc: exit",
        (16, canvas.shape[0] - 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
