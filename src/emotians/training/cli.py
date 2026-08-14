"""Stage 2 command handlers, kept separate from the webcam application loop."""

from __future__ import annotations

import json
from pathlib import Path

from emotians.data.catalog import create_split_manifest, scan_dataset, validate_dataset
from emotians.data.dataset import ExpressionDataset
from emotians.data.fer2013 import create_fer2013_manifest, download_fer2013
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


def format_report(report: dict[str, object]) -> str:
    """Format reports as stable, readable JSON for console and CI use."""

    return json.dumps(report, indent=2, ensure_ascii=False)
