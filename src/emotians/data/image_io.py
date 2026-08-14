"""OpenCV decoding helpers that also work with Unicode Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image without OpenCV's Windows narrow-path limitation.

    ``cv2.imread`` can fail when a Windows user directory includes characters
    such as ``ñ``. Reading file bytes through NumPy and then calling
    ``cv2.imdecode`` retains the same decoder while accepting Unicode paths.
    """

    try:
        encoded = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)
