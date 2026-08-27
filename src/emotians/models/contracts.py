"""Stable model and inference contracts used by the webcam pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One detected face box in ``(left, top, right, bottom)`` pixel coordinates."""

    xyxy: BoundingBox
    confidence: float


@dataclass(frozen=True)
class Prediction:
    """A visible-expression prediction for a cropped face."""

    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class FrameResult:
    """All inference output needed by the renderer for one frame."""

    detections: list[Detection]
    prediction: Prediction | None
    status: str


class FaceDetector(Protocol):
    def detect(self, bgr_frame: np.ndarray) -> list[Detection]: ...


class ExpressionClassifier(Protocol):
    def predict(self, face_bgr: np.ndarray) -> Prediction: ...
