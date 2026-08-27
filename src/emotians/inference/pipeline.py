"""Detector-to-expression inference pipeline for live webcam frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from emotians.data.catalog import EMOTION_LABELS
from emotians.models import (
    Detection,
    EmotionCNN,
    ExpressionClassifier,
    FaceDetector,
    FrameResult,
    Prediction,
    TinyFaceDetector,
)


class TorchFaceDetector:
    """Load and run a project-trained ``TinyFaceDetector`` checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_name: str = "auto",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device = _resolve_device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.input_size = int(checkpoint.get("input_size", 128))
        self.confidence_threshold = confidence_threshold
        self.model = TinyFaceDetector(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("bgr_frame must be an HxWx3 image.")
        height, width = bgr_frame.shape[:2]
        resized = cv2.resize(bgr_frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            logit, box = self.model(tensor)
            score = float(torch.sigmoid(logit)[0].cpu())
            if score < self.confidence_threshold:
                return []
            left, top, right, bottom = _normalized_box_to_pixels(box[0].cpu().tolist(), width, height)
        if right <= left or bottom <= top:
            return []
        return [Detection((left, top, right, bottom), score)]


class YuNetFaceDetector:
    """Run OpenCV's pretrained YuNet face detector."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model was not found: {model_path}")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: cv2.FaceDetectorYN | None = None
        self._input_size: tuple[int, int] | None = None

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("bgr_frame must be an HxWx3 image.")
        height, width = bgr_frame.shape[:2]
        input_size = (width, height)
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                input_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._input_size = input_size
        elif self._input_size != input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, faces = self._detector.detect(bgr_frame)
        if faces is None:
            return []

        detections: list[Detection] = []
        for face in faces:
            left, top, box_width, box_height = face[:4]
            right = min(width, max(0, round(left + box_width)))
            bottom = min(height, max(0, round(top + box_height)))
            left = max(0, round(left))
            top = max(0, round(top))
            score = float(face[14])
            if right > left and bottom > top:
                detections.append(Detection((left, top, right, bottom), score))
        return detections


class ExpressionCNNClassifier:
    """Load and run a project-trained four-class expression checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device_name: str = "auto") -> None:
        self.device = _resolve_device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if tuple(checkpoint.get("labels", ())) != EMOTION_LABELS:
            raise ValueError("Expression checkpoint labels do not match the EMOTIANS label contract.")
        self.labels = tuple(checkpoint["labels"])
        self.model = EmotionCNN(num_classes=len(self.labels)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, face_bgr: np.ndarray) -> Prediction:
        if face_bgr.size == 0:
            raise ValueError("face_bgr crop is empty.")
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = (
            torch.from_numpy(resized.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        mapped = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        label = max(mapped, key=mapped.get)
        return Prediction(label=label, confidence=mapped[label], probabilities=mapped)


class LabelSmoother:
    """Exponential moving average over expression probabilities."""

    def __init__(self, alpha: float = 0.55) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        self.alpha = alpha
        self._state: dict[str, float] | None = None

    def update(self, prediction: Prediction) -> Prediction:
        if self._state is None:
            self._state = dict(prediction.probabilities)
        else:
            self._state = {
                label: self.alpha * prediction.probabilities[label]
                + (1.0 - self.alpha) * self._state.get(label, 0.0)
                for label in prediction.probabilities
            }
        total = sum(self._state.values()) or 1.0
        probabilities = {label: score / total for label, score in self._state.items()}
        label = max(probabilities, key=probabilities.get)
        return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)

    def reset(self) -> None:
        self._state = None


def process_frame(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    classifier: ExpressionClassifier,
    smoother: LabelSmoother | None = None,
    expression_threshold: float = 0.45,
) -> FrameResult:
    """Run detector, crop the best face, classify expression and smooth output."""

    detections = detector.detect(frame_bgr)
    if not detections:
        if smoother:
            smoother.reset()
        return FrameResult([], None, "no-face")

    detection = max(detections, key=lambda item: item.confidence)
    left, top, right, bottom = detection.xyxy
    face_crop = frame_bgr[top:bottom, left:right]
    if face_crop.size == 0:
        if smoother:
            smoother.reset()
        return FrameResult([detection], None, "invalid-crop")

    prediction = classifier.predict(face_crop)
    if smoother:
        prediction = smoother.update(prediction)
    status = "ok" if prediction.confidence >= expression_threshold else "low-confidence"
    return FrameResult([detection], prediction, status)


def _normalized_box_to_pixels(
    box: list[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    center_x, center_y, width, height = [max(0.0, min(1.0, value)) for value in box]
    left = round((center_x - width / 2.0) * frame_width)
    top = round((center_y - height / 2.0) * frame_height)
    right = round((center_x + width / 2.0) * frame_width)
    bottom = round((center_y + height / 2.0) * frame_height)
    return (
        max(0, min(frame_width - 1, left)),
        max(0, min(frame_height - 1, top)),
        max(1, min(frame_width, right)),
        max(1, min(frame_height, bottom)),
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this PyTorch installation.")
    return device
