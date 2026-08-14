import numpy as np
import pytest

from emotians.vision import centered_roi, draw_preview_overlay


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
