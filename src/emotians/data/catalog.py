"""Dataset discovery, validation and subject-disjoint split manifests."""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from emotians.data.image_io import read_image

EMOTION_LABELS = ("happy", "sad", "angry", "surprised")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class Sample:
    path: Path
    subject_id: str
    label: str


def scan_dataset(dataset_root: str | Path) -> list[Sample]:
    """Read the Stage 2 layout without modifying source images.

    Expected layout: ``<root>/<subject_id>/<emotion>/<image-file>``. Grouping
    by subject is mandatory so evaluation does not learn a participant's face
    in training and report an inflated score on that same person.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"Dataset folder not found: {root}")

    samples: list[Sample] = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for label_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if label_dir.name not in EMOTION_LABELS:
                continue
            for image_path in sorted(label_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append(Sample(image_path, subject_dir.name, label_dir.name))
    return samples


def dataset_report(samples: list[Sample]) -> dict[str, object]:
    """Return serializable counts and warnings suitable for console output."""

    class_counts = Counter(sample.label for sample in samples)
    subject_counts = Counter(sample.subject_id for sample in samples)
    missing_labels = [label for label in EMOTION_LABELS if not class_counts[label]]
    warnings: list[str] = []
    if missing_labels:
        warnings.append(f"Missing label folder or images: {', '.join(missing_labels)}")
    if len(subject_counts) < 3:
        warnings.append("At least three subjects are needed for train/validation/test splits.")
    return {
        "images": len(samples),
        "subjects": len(subject_counts),
        "class_counts": {label: class_counts[label] for label in EMOTION_LABELS},
        "warnings": warnings,
    }


def validate_dataset(samples: list[Sample]) -> dict[str, object]:
    """Validate the conditions required to start training."""

    report = dataset_report(samples)
    if not samples:
        raise ValueError("No images were found in the expected dataset layout.")
    if report["warnings"]:
        raise ValueError("Dataset is not ready: " + "; ".join(report["warnings"]))
    unreadable = [str(sample.path) for sample in samples if read_image(sample.path) is None]
    if unreadable:
        preview = ", ".join(unreadable[:3])
        suffix = "..." if len(unreadable) > 3 else ""
        raise ValueError(f"OpenCV cannot decode {len(unreadable)} image(s): {preview}{suffix}")
    return report


def create_split_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    seed: int = 42,
) -> dict[str, int]:
    """Create a reproducible 70/15/15 split where a subject appears once only."""

    root = Path(dataset_root)
    samples = scan_dataset(root)
    validate_dataset(samples)
    subjects = sorted({sample.subject_id for sample in samples})
    random.Random(seed).shuffle(subjects)
    subject_splits = _split_subjects(subjects)

    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("path", "subject_id", "label", "split"))
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": sample.path.relative_to(root).as_posix(),
                    "subject_id": sample.subject_id,
                    "label": sample.label,
                    "split": subject_splits[sample.subject_id],
                }
            )
    counts = Counter(subject_splits.values())
    return {split: counts[split] for split in ("train", "validation", "test")}


def load_manifest(dataset_root: str | Path, manifest_path: str | Path, split: str) -> list[Sample]:
    """Load one split and reject malformed or unsafe manifest rows."""

    root = Path(dataset_root).resolve()
    manifest = Path(manifest_path)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if not manifest.is_file():
        raise ValueError(f"Split manifest not found: {manifest}")

    samples: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"path", "subject_id", "label", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Split manifest must include path, subject_id, label and split columns.")
        for row in reader:
            if row["split"] != split:
                continue
            if row["label"] not in EMOTION_LABELS:
                raise ValueError(f"Unsupported emotion label in manifest: {row['label']}")
            image_path = (root / row["path"]).resolve()
            if root not in image_path.parents or not image_path.is_file():
                raise ValueError(f"Manifest contains a missing or out-of-root path: {row['path']}")
            samples.append(Sample(image_path, row["subject_id"], row["label"]))
    if not samples:
        raise ValueError(f"No samples found for '{split}' in {manifest}.")
    return samples


def _split_subjects(subjects: list[str]) -> dict[str, str]:
    if len(subjects) < 3:
        raise ValueError("At least three subjects are needed to build three splits.")
    test_count = max(1, round(len(subjects) * 0.15))
    validation_count = max(1, round(len(subjects) * 0.15))
    train_count = len(subjects) - test_count - validation_count
    if train_count < 1:
        raise ValueError("Dataset needs more subjects to keep a non-empty training split.")
    assignments: dict[str, str] = {}
    for subject in subjects[:train_count]:
        assignments[subject] = "train"
    for subject in subjects[train_count : train_count + validation_count]:
        assignments[subject] = "validation"
    for subject in subjects[train_count + validation_count :]:
        assignments[subject] = "test"
    return assignments
