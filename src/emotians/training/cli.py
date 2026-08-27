"""Stage 2 command handlers, kept separate from the webcam application loop."""

from __future__ import annotations

import json
from pathlib import Path

from emotians.data.catalog import create_split_manifest, scan_dataset, validate_dataset
from emotians.data.dataset import ExpressionDataset
from emotians.data.face_boxes import (
    FaceBoxDataset,
    create_full_image_face_manifest,
    load_face_box_manifest,
)
from emotians.data.fer2013 import create_fer2013_manifest, download_fer2013
from emotians.training.detector import (
    DetectorTrainingConfig,
    evaluate_detector_checkpoint,
    train_detector,
)
from emotians.training.engine import TrainingConfig, evaluate_checkpoint, train


def prepare_data(data_root: Path, manifest: Path, seed: int) -> dict[str, object]:
    samples = scan_dataset(data_root)
    report = validate_dataset(samples)
    report["subject_splits"] = create_split_manifest(data_root, manifest, seed)
    report["manifest"] = str(manifest)
    return report


def download_and_prepare_fer2013(manifest: Path, seed: int) -> dict[str, object]:
    """Download Kaggle FER2013 and generate a compatible four-class manifest."""

    data_root = download_fer2013()
    return prepare_fer2013(data_root, manifest, seed)


def prepare_fer2013(data_root: Path, manifest: Path, seed: int) -> dict[str, object]:
    """Create a four-class manifest from an already-downloaded FER2013 folder."""

    counts = create_fer2013_manifest(data_root, manifest, seed)
    return {
        "data_root": str(data_root),
        "manifest": str(manifest),
        "image_splits": counts,
        "labels_used": ["angry", "happy", "sad", "surprised"],
        "excluded_source_labels": ["disgust", "fear", "neutral"],
        "warning": "FER2013 has no subject IDs; splits cannot be subject-disjoint.",
    }


def prepare_fer2013_detector(data_root: Path, source_manifest: Path, output_manifest: Path) -> dict[str, object]:
    """Create a quick detector manifest from FER2013 cropped face images."""

    return create_full_image_face_manifest(data_root, source_manifest, output_manifest)


def train_classifier(
    data_root: Path,
    manifest: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    from emotians.data.catalog import load_manifest

    train_data = ExpressionDataset(load_manifest(data_root, manifest, "train"), augment=True)
    validation_data = ExpressionDataset(load_manifest(data_root, manifest, "validation"))
    return train(
        train_data,
        validation_data,
        output_dir,
        TrainingConfig(epochs=epochs, batch_size=batch_size, device=device, seed=seed),
    )


def evaluate_classifier(
    data_root: Path, manifest: Path, checkpoint: Path, batch_size: int, device: str
) -> dict[str, object]:
    from emotians.data.catalog import load_manifest

    test_data = ExpressionDataset(load_manifest(data_root, manifest, "test"))
    return evaluate_checkpoint(test_data, checkpoint, batch_size=batch_size, device_name=device)


def train_face_detector(
    image_root: Path,
    manifest: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    input_size: int,
    device: str,
    seed: int,
    confidence_threshold: float,
) -> dict[str, object]:
    train_data = FaceBoxDataset(
        load_face_box_manifest(image_root, manifest, "train"),
        input_size=input_size,
    )
    validation_data = FaceBoxDataset(
        load_face_box_manifest(image_root, manifest, "validation"),
        input_size=input_size,
    )
    return train_detector(
        train_data,
        validation_data,
        output_dir,
        DetectorTrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            input_size=input_size,
            device=device,
            seed=seed,
            confidence_threshold=confidence_threshold,
        ),
    )


def evaluate_face_detector(
    image_root: Path,
    manifest: Path,
    checkpoint: Path,
    batch_size: int,
    input_size: int,
    device: str,
    confidence_threshold: float,
) -> dict[str, object]:
    test_data = FaceBoxDataset(
        load_face_box_manifest(image_root, manifest, "test"),
        input_size=input_size,
    )
    return evaluate_detector_checkpoint(
        test_data,
        checkpoint,
        batch_size=batch_size,
        device_name=device,
        confidence_threshold=confidence_threshold,
    )


def format_report(report: dict[str, object]) -> str:
    """Format reports as stable, readable JSON for console and CI use."""

    return json.dumps(report, indent=2, ensure_ascii=False)
