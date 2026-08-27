"""Small single-class face detector initialized from scratch."""

from __future__ import annotations

import torch
from torch import nn


class TinyFaceDetector(nn.Module):
    """Predict face objectness and one normalized ``cx, cy, w, h`` box."""

    def __init__(self, input_size: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            _block(3, 24),
            nn.MaxPool2d(2),
            _block(24, 48),
            nn.MaxPool2d(2),
            _block(48, 96),
            nn.MaxPool2d(2),
            _block(96, 160),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(160, 96), nn.ReLU())
        self.objectness = nn.Linear(96, 1)
        self.box = nn.Sequential(nn.Linear(96, 4), nn.Sigmoid())

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.head(self.features(batch))
        return self.objectness(features).squeeze(1), self.box(features)


def _block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )
