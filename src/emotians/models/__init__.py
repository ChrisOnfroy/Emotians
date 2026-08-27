"""Neural-network definitions created and trained inside this project."""

from emotians.models.contracts import Detection, ExpressionClassifier, FaceDetector, FrameResult, Prediction
from emotians.models.emotion_cnn import EmotionCNN
from emotians.models.face_detector import TinyFaceDetector

__all__ = [
    "Detection",
    "EmotionCNN",
    "ExpressionClassifier",
    "FaceDetector",
    "FrameResult",
    "Prediction",
    "TinyFaceDetector",
]
