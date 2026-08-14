from pathlib import Path

import cv2
import numpy as np

from emotians.data.catalog import EMOTION_LABELS, create_split_manifest, load_manifest, scan_dataset


def _make_dataset(root: Path) -> None:
    for subject in ("person_01", "person_02", "person_03", "person_04"):
        for label in EMOTION_LABELS:
            image = root / subject / label / "sample.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))


def test_creates_subject_disjoint_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "expressions"
    _make_dataset(dataset_root)
    manifest = tmp_path / "splits.csv"

    create_split_manifest(dataset_root, manifest, seed=7)
    train_subjects = {sample.subject_id for sample in load_manifest(dataset_root, manifest, "train")}
    validation_subjects = {sample.subject_id for sample in load_manifest(dataset_root, manifest, "validation")}
    test_subjects = {sample.subject_id for sample in load_manifest(dataset_root, manifest, "test")}

    assert train_subjects.isdisjoint(validation_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert validation_subjects.isdisjoint(test_subjects)


def test_scan_ignores_unknown_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "expressions"
    _make_dataset(dataset_root)
    unknown = dataset_root / "person_01" / "neutral" / "sample.jpg"
    unknown.parent.mkdir(parents=True)
    assert cv2.imwrite(str(unknown), np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(scan_dataset(dataset_root)) == 16
