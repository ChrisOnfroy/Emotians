"""Face-box manifest loading for the Stage 3 detector."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from emotians.data.image_io import read_image

REQUIRED_FACE_COLUMNS = ("path", "split", "face_present", "x_min", "y_min", "x_max", "y_max")
SOURCE_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class FaceBoxSample:
    path: Path
    split: str
    bbox: tuple[int, int, int, int] | None


class FaceBoxDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Load detector images and normalized ``cx, cy, w, h`` targets."""

    def __init__(self, samples: list[FaceBoxSample], input_size: int = 128) -> None:
        if input_size < 16:
            raise ValueError("input_size must be at least 16 pixels.")
        self.samples = samples
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_image(sample.path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {sample.path}")

        height, width = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        objectness = torch.tensor(1.0 if sample.bbox else 0.0, dtype=torch.float32)
        box = torch.zeros(4, dtype=torch.float32)
        if sample.bbox:
            left, top, right, bottom = sample.bbox
            center_x = ((left + right) / 2.0) / width
            center_y = ((top + bottom) / 2.0) / height
            box_width = (right - left) / width
            box_height = (bottom - top) / height
            box = torch.tensor((center_x, center_y, box_width, box_height), dtype=torch.float32)
        return tensor, objectness, box


def load_face_box_manifest(
    image_root: str | Path,
    manifest_path: str | Path,
    split: str,
) -> list[FaceBoxSample]:
    """Load one detector split from a CSV manifest.

    Expected columns: ``path, split, face_present, x_min, y_min, x_max, y_max``.
    Rows with ``face_present=false`` are treated as no-face negatives and their
    coordinates are ignored.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    root = Path(image_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"Face-box manifest not found: {manifest}")

    samples: list[FaceBoxSample] = []
    with manifest.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(REQUIRED_FACE_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(
                "Face-box manifest must include: " + ", ".join(REQUIRED_FACE_COLUMNS)
            )
        for row_number, row in enumerate(reader, start=2):
            if row["split"] != split:
                continue
            image_path = _safe_image_path(root, row["path"])
            face_present = _parse_bool(row["face_present"], row_number)
            bbox = _parse_bbox(row, row_number) if face_present else None
            _validate_image_and_box(image_path, bbox, row_number)
            samples.append(FaceBoxSample(image_path, split, bbox))
    if not samples:
        raise ValueError(f"No face-box samples found for '{split}' in {manifest}.")
    return samples


def create_full_image_face_manifest(
    image_root: str | Path,
    source_manifest_path: str | Path,
    output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Create detector boxes by treating each manifest image as a full-face crop.

    This is a pragmatic bridge for cropped face datasets such as FER2013. It is
    not a substitute for real webcam-frame face-box annotations, but it lets the
    Stage 3 detector path train from project data without external detectors.
    """

    root = Path(image_root).resolve()
    source_manifest = Path(source_manifest_path)
    if not source_manifest.is_file():
        raise ValueError(f"Source manifest not found: {source_manifest}")

    output_manifest = Path(output_manifest_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    skipped: list[str] = []

    with source_manifest.open(newline="", encoding="utf-8") as source, output_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"path", "split"}.issubset(reader.fieldnames):
            raise ValueError("Source manifest must include path and split columns.")
        writer = csv.DictWriter(output, fieldnames=REQUIRED_FACE_COLUMNS)
        writer.writeheader()
        for row in reader:
            split = row["split"]
            if split not in SOURCE_SPLITS:
                skipped.append(row.get("path", "<missing path>"))
                continue
            image_path = _safe_image_path(root, row["path"])
            image = read_image(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV cannot decode image: {image_path}")
            height, width = image.shape[:2]
            writer.writerow(
                {
                    "path": row["path"],
                    "split": split,
                    "face_present": "true",
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": width,
                    "y_max": height,
                }
            )
            counts[split] += 1

    return {
        "manifest": str(output_manifest),
        "source_manifest": str(source_manifest),
        "assumption": "Each source image is already a cropped face; full image becomes the face box.",
        "splits": {split: counts[split] for split in ("train", "validation", "test")},
        "skipped_rows": len(skipped),
    }


def _safe_image_path(root: Path, raw_path: str) -> Path:
    image_path = (root / raw_path).resolve()
    if root not in image_path.parents or not image_path.is_file():
        raise ValueError(f"Manifest contains a missing or out-of-root path: {raw_path}")
    return image_path


def _parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Row {row_number}: face_present must be true or false.")


def _parse_bbox(row: dict[str, str], row_number: int) -> tuple[int, int, int, int]:
    try:
        left = int(row["x_min"])
        top = int(row["y_min"])
        right = int(row["x_max"])
        bottom = int(row["y_max"])
    except ValueError as error:
        raise ValueError(f"Row {row_number}: face box coordinates must be integers.") from error
    if right <= left or bottom <= top:
        raise ValueError(f"Row {row_number}: face box must have positive width and height.")
    return left, top, right, bottom


def _validate_image_and_box(
    image_path: Path,
    bbox: tuple[int, int, int, int] | None,
    row_number: int,
) -> None:
    image = read_image(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Row {row_number}: OpenCV cannot decode image: {image_path}")
    if bbox is None:
        return
    height, width = image.shape[:2]
    left, top, right, bottom = bbox
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"Row {row_number}: face box is outside image bounds.")
