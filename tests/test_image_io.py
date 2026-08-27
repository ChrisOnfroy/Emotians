from pathlib import Path

import numpy as np

from emotians.data.image_io import read_image, write_image


def test_write_image_round_trips_with_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "captura_ñ.jpg"
    image = np.full((12, 16, 3), 127, dtype=np.uint8)

    assert write_image(path, image)
    decoded = read_image(path)

    assert decoded is not None
    assert decoded.shape == image.shape
