"""Small facial-expression CNN initialized without external weights."""

from __future__ import annotations

import torch
from torch import nn


class EmotionCNN(nn.Module):
    """A compact four-class CNN for 48x48 grayscale face crops.

    Instantiating this class uses PyTorch's random initialization. No checkpoint
    is loaded by the model definition or by the training path.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 32),
            nn.MaxPool2d(2),
            _block(32, 64),
            nn.MaxPool2d(2),
            _block(64, 128),
            nn.MaxPool2d(2),
            _block(128, 192),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, num_classes))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(batch))


def _block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )
