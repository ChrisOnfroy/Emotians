import io
import numpy as np
import cv2
from fastapi.testclient import TestClient

from emotians.web import create_app, is_low_light


def make_jpeg_image(value: int, size=(64, 64)) -> bytes:
    img = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def test_is_low_light_detects_dark_image() -> None:
    dark = np.zeros((48, 48, 3), dtype=np.uint8)
    low, metrics = is_low_light(dark, mean_th=50.0, std_th=10.0, dark_pixel_th=30)
    assert low is True
    assert metrics['mean'] < 10


def test_api_analyze_returns_low_light_for_dark_image(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    # Send a very dark JPEG image
    payload = make_jpeg_image(2, size=(80, 60))
    files = {'frame': ('frame.jpg', io.BytesIO(payload), 'image/jpeg')}
    response = client.post('/api/analyze', files=files)
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'low-light'
    assert 'metrics' in data

