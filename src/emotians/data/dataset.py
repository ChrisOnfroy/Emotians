"""PyTorch dataset backed by OpenCV image decoding."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from emotians.data.catalog import EMOTION_LABELS, Sample
from emotians.data.image_io import read_image


class ExpressionDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load face crops as normalized 48x48 grayscale tensors."""

    def __init__(self, samples: list[Sample], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment
        self.label_to_index = {label: index for index, label in enumerate(EMOTION_LABELS)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = read_image(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {sample.path}")
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        if self.augment:
            image = _augment(image)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, self.label_to_index[sample.label]


def _augment(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12, 12)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image
