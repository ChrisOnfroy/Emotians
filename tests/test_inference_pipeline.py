import cv2
import numpy as np
import pytest

from emotians.inference import LabelSmoother, YuNetFaceDetector, process_frame
from emotians.models import Detection, Prediction


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, bgr_frame: np.ndarray) -> list[Detection]:
        return self.detections


class FakeClassifier:
    def predict(self, face_bgr: np.ndarray) -> Prediction:
        assert face_bgr.shape[:2] == (20, 20)
        return Prediction(
            label="happy",
            confidence=0.8,
            probabilities={"happy": 0.8, "sad": 0.1, "angry": 0.05, "surprised": 0.05},
        )


def test_process_frame_detects_classifies_and_smooths() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detector = FakeDetector([Detection((10, 12, 30, 32), 0.9)])
    smoother = LabelSmoother(alpha=1.0)

    result = process_frame(frame, detector, FakeClassifier(), smoother, expression_threshold=0.5)

    assert result.status == "ok"
    assert result.prediction is not None
    assert result.prediction.label == "happy"
    assert result.detections[0].confidence == 0.9


def test_process_frame_resets_smoother_when_face_is_missing() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    smoother = LabelSmoother(alpha=1.0)
    process_frame(frame, FakeDetector([Detection((10, 12, 30, 32), 0.9)]), FakeClassifier(), smoother)

    result = process_frame(frame, FakeDetector([]), FakeClassifier(), smoother)

    assert result.status == "no-face"
    assert result.prediction is None


def test_yunet_detector_converts_and_clips_boxes(monkeypatch) -> None:
    model_path = "face_detection_yunet.onnx"
    monkeypatch.setattr("pathlib.Path.is_file", lambda _: True)

    class FakeYuNet:
        def __init__(self) -> None:
            self.input_sizes: list[tuple[int, int]] = []

        def detect(self, frame: np.ndarray) -> tuple[None, np.ndarray]:
            face = np.zeros((1, 15), dtype=np.float32)
            face[0, :4] = (-5, 4, 30, 20)
            face[0, 14] = 0.91
            return None, face

        def setInputSize(self, size: tuple[int, int]) -> None:
            self.input_sizes.append(size)

    fake_detector = FakeYuNet()
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *args: fake_detector)

    detector = YuNetFaceDetector(model_path)
    detections = detector.detect(np.zeros((40, 60, 3), dtype=np.uint8))

    assert detections[0].xyxy == (0, 4, 25, 24)
    assert detections[0].confidence == pytest.approx(0.91)
