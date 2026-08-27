"""Runtime inference components for the Stage 3 webcam demo."""

from emotians.inference.pipeline import (
	ExpressionCNNClassifier,
	LabelSmoother,
	TorchFaceDetector,
	YuNetFaceDetector,
	process_frame,
)

__all__ = [
	"ExpressionCNNClassifier",
	"LabelSmoother",
	"TorchFaceDetector",
	"YuNetFaceDetector",
	"process_frame",
]
