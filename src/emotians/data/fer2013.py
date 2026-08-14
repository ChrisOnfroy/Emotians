"""Adapter for the Kaggle ``msambare/fer2013`` image-folder dataset."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from emotians.data.catalog import EMOTION_LABELS, IMAGE_SUFFIXES

KAGGLE_DATASET = "msambare/fer2013"
FER2013_LABEL_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
}


def download_fer2013() -> Path:
    """Download FER2013 through kagglehub and return its local cache path."""

    try:
        import kagglehub
    except ImportError as error:
        raise ValueError("Install kagglehub first: python -m pip install kagglehub") from error
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def create_fer2013_manifest(dataset_root: str | Path, manifest_path: str | Path, seed: int = 42) -> dict[str, int]:
    """Create train/validation/test rows without changing downloaded data.

    The provider's ``test`` directory stays entirely held out. A deterministic,
    label-stratified 15% of its ``train`` images becomes validation. FER2013
    contains no participant IDs, so this cannot provide a subject-disjoint
    evaluation; that limitation is recorded in the generated report.
    """

    root = Path(dataset_root).resolve()
    train_root, test_root = root / "train", root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise ValueError("FER2013 must contain train/ and test/ directories.")
    training = _collect(train_root)
    testing = _collect(test_root)
    if not training or not testing:
        raise ValueError("FER2013 has no usable images for the four EMOTIANS labels.")

    validation_paths = _stratified_validation_paths(training, seed)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0}
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("path", "subject_id", "label", "split"))
        writer.writeheader()
        for image_path, label in training:
            split = "validation" if image_path in validation_paths else "train"
            writer.writerow(
                {
                    "path": image_path.relative_to(root).as_posix(),
                    "subject_id": "unknown_fer2013",
                    "label": label,
                    "split": split,
                }
            )
            counts[split] += 1
        for image_path, label in testing:
            writer.writerow(
                {
                    "path": image_path.relative_to(root).as_posix(),
                    "subject_id": "unknown_fer2013",
                    "label": label,
                    "split": "test",
                }
            )
            counts["test"] += 1
    return counts


def _collect(split_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for source_label, target_label in FER2013_LABEL_MAP.items():
        label_dir = split_root / source_label
        if not label_dir.is_dir():
            continue
        samples.extend(
            (image_path, target_label)
            for image_path in sorted(label_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        )
    return samples


def _stratified_validation_paths(samples: list[tuple[Path, str]], seed: int) -> set[Path]:
    by_label: dict[str, list[Path]] = defaultdict(list)
    for image_path, label in samples:
        by_label[label].append(image_path)
    generator = random.Random(seed)
    validation: set[Path] = set()
    for label in EMOTION_LABELS:
        paths = by_label[label]
        if len(paths) < 2:
            raise ValueError(f"FER2013 training split needs at least two '{label}' images.")
        generator.shuffle(paths)
        validation.update(paths[: max(1, round(len(paths) * 0.15))])
    return validation
