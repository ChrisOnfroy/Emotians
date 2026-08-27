import numpy as np
import pytest

from emotians.models import Detection, FrameResult, Prediction
from emotians.vision import centered_roi, draw_preview_overlay
from emotians.vision.overlay import draw_live_overlay


def test_centered_roi_stays_inside_frame() -> None:
    roi = centered_roi((720, 1280, 3), width_ratio=0.42, height_ratio=0.58)

    assert roi == (371, 151, 909, 569)


def test_preview_overlay_does_not_mutate_source_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    rendered = draw_preview_overlay(frame, centered_roi(frame.shape, 0.5, 0.5), fps=30.0)

    assert rendered.shape == frame.shape
    assert np.array_equal(frame, np.zeros((120, 160, 3), dtype=np.uint8))
    assert np.any(rendered != frame)


def test_preview_overlay_rejects_non_bgr_frame() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        draw_preview_overlay(np.zeros((100, 100), dtype=np.uint8), (1, 1, 20, 20), 10.0)


def test_live_overlay_draws_detection_and_prediction() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = FrameResult(
        detections=[Detection((20, 20, 80, 90), 0.92)],
        prediction=Prediction(
            "happy",
            0.81,
            {"happy": 0.81, "sad": 0.1, "angry": 0.05, "surprised": 0.04},
        ),
        status="ok",
    )

    rendered = draw_live_overlay(frame, result, fps=24.0)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
