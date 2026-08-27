"""Reproducible training, checkpointing and evaluation for EmotionCNN."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from emotians.data.catalog import EMOTION_LABELS
from emotians.data.dataset import ExpressionDataset
from emotians.models import EmotionCNN

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "auto"


def train(
    train_data: ExpressionDataset,
    validation_data: ExpressionDataset,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train a randomly initialized CNN and save the best validation checkpoint."""

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False)
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    class_weights = _class_weights(train_data).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    best_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        validation_metrics = evaluate_model(model, validation_loader, device)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(seen, 1),
            **validation_metrics,
        }
        history.append(epoch_metrics)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | validation_accuracy=%.2f%%",
            epoch,
            config.epochs,
            epoch_metrics["train_loss"],
            validation_metrics["accuracy"] * 100,
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": EMOTION_LABELS,
                    "training_config": asdict(config),
                    "metrics": validation_metrics,
                },
                destination / "best_expression_cnn.pt",
            )

    summary = {"best_validation_accuracy": best_accuracy, "epochs": config.epochs}
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoint(
    test_data: ExpressionDataset, checkpoint_path: str | Path, batch_size: int, device_name: str
) -> dict[str, object]:
    """Evaluate a saved project checkpoint on a held-out test split."""

    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if tuple(checkpoint.get("labels", ())) != EMOTION_LABELS:
        raise ValueError("Checkpoint labels do not match the EMOTIANS label contract.")
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return evaluate_model(model, loader, device, include_per_class=True)


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, include_per_class: bool = False
) -> dict[str, object]:
    """Calculate accuracy and, when requested, per-class precision/recall/F1."""

    model.eval()
    total = correct = 0
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=int)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            total += len(labels)
            correct += int((predictions == labels).sum())
            for truth, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[truth, predicted] += 1
    metrics: dict[str, object] = {"accuracy": correct / max(total, 1), "samples": total}
    if include_per_class:
        per_class: dict[str, dict[str, float]] = {}
        for index, label in enumerate(EMOTION_LABELS):
            tp = matrix[index, index]
            precision = tp / max(int(matrix[:, index].sum()), 1)
            recall = tp / max(int(matrix[index, :].sum()), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        metrics["per_class"] = per_class
        metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def _class_weights(dataset: ExpressionDataset) -> torch.Tensor:
    """Give underrepresented emotion classes a proportionally larger loss."""

    counts = Counter(sample.label for sample in dataset.samples)
    weights = [len(dataset) / (len(EMOTION_LABELS) * counts[label]) for label in EMOTION_LABELS]
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this PyTorch installation.")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
