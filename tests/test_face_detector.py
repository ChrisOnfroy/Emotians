from pathlib import Path

import cv2
import numpy as np
import torch

from emotians.data.face_boxes import (
    FaceBoxDataset,
    create_full_image_face_manifest,
    load_face_box_manifest,
)
from emotians.models import TinyFaceDetector


def _write_image(path: Path, color: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 80, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_loads_face_box_manifest_and_dataset(tmp_path: Path) -> None:
    image_root = tmp_path / "faces"
    _write_image(image_root / "positive.jpg", 30)
    _write_image(image_root / "negative.jpg", 0)
    manifest = tmp_path / "face_boxes.csv"
    manifest.write_text(
        "\n".join(
            [
                "path,split,face_present,x_min,y_min,x_max,y_max",
                "positive.jpg,train,true,10,12,50,54",
                "negative.jpg,train,false,,,,",
            ]
        ),
        encoding="utf-8",
    )

    samples = load_face_box_manifest(image_root, manifest, "train")
    dataset = FaceBoxDataset(samples, input_size=32)
    image, objectness, box = dataset[0]
    negative_image, negative_objectness, negative_box = dataset[1]

    assert image.shape == (3, 32, 32)
    assert objectness.item() == 1.0
    assert torch.all((box >= 0) & (box <= 1))
    assert negative_image.shape == (3, 32, 32)
    assert negative_objectness.item() == 0.0
    assert torch.equal(negative_box, torch.zeros(4))


def test_tiny_face_detector_output_shapes() -> None:
    model = TinyFaceDetector(input_size=64)
    logits, boxes = model(torch.zeros((2, 3, 64, 64), dtype=torch.float32))

    assert logits.shape == (2,)
    assert boxes.shape == (2, 4)
    assert torch.all((boxes >= 0) & (boxes <= 1))


def test_creates_full_image_face_manifest_from_expression_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "fer2013"
    _write_image(image_root / "train" / "happy" / "sample.jpg", 90)
    source_manifest = tmp_path / "splits.csv"
    source_manifest.write_text(
        "\n".join(
            [
                "path,subject_id,label,split",
                "train/happy/sample.jpg,unknown_fer2013,happy,train",
            ]
        ),
        encoding="utf-8",
    )
    output_manifest = tmp_path / "face_boxes.csv"

    report = create_full_image_face_manifest(image_root, source_manifest, output_manifest)
    samples = load_face_box_manifest(image_root, output_manifest, "train")

    assert report["splits"]["train"] == 1
    assert samples[0].bbox == (0, 0, 80, 64)
