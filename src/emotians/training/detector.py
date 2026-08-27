"""Training and evaluation loop for the Stage 3 face detector."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from emotians.data.face_boxes import FaceBoxDataset
from emotians.models import TinyFaceDetector

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001
    input_size: int = 128
    seed: int = 42
    device: str = "auto"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5


def train_detector(
    train_data: FaceBoxDataset,
    validation_data: FaceBoxDataset,
    output_dir: str | Path,
    config: DetectorTrainingConfig,
) -> dict[str, object]:
    """Train a randomly initialized single-class detector and save the best checkpoint."""

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False)
    model = TinyFaceDetector(input_size=config.input_size).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    object_loss = nn.BCEWithLogitsLoss()
    box_loss = nn.SmoothL1Loss()
    best_recall = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"Detector epoch {epoch}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for images, targets, boxes in progress:
            images = images.to(device)
            targets = targets.to(device)
            boxes = boxes.to(device)

            optimizer.zero_grad()
            logits, predicted_boxes = model(images)
            loss = object_loss(logits, targets)
            positive_mask = targets > 0.5
            if positive_mask.any():
                loss = loss + 5.0 * box_loss(predicted_boxes[positive_mask], boxes[positive_mask])
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            seen += len(targets)
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        metrics = evaluate_detector_model(
            model,
            validation_loader,
            device,
            config.confidence_threshold,
            config.iou_threshold,
        )
        epoch_report = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), **metrics}
        history.append(epoch_report)
        LOGGER.info(
            "Detector epoch %d/%d | train_loss=%.4f | precision=%.3f | recall=%.3f",
            epoch,
            config.epochs,
            epoch_report["train_loss"],
            metrics["precision"],
            metrics["recall"],
        )
        if metrics["recall"] > best_recall:
            best_recall = float(metrics["recall"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": config.input_size,
                    "training_config": asdict(config),
                    "metrics": metrics,
                    "target_encoding": "normalized cxcywh",
                    "detector_type": "single_box_face_detector",
                },
                destination / "best_face_detector.pt",
            )

    summary = {"best_validation_recall": best_recall, "epochs": config.epochs}
    (destination / "detector_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_detector_checkpoint(
    test_data: FaceBoxDataset,
    checkpoint_path: str | Path,
    batch_size: int,
    device_name: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    input_size = int(checkpoint.get("input_size", test_data.input_size))
    model = TinyFaceDetector(input_size=input_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    metrics = evaluate_detector_model(model, loader, device, confidence_threshold, iou_threshold)
    metrics["checkpoint_metrics"] = checkpoint.get("metrics", {})
    return metrics


def evaluate_detector_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    model.eval()
    true_positive = false_positive = false_negative = true_negative = 0
    ious: list[float] = []
    with torch.inference_mode():
        for images, targets, boxes in loader:
            logits, predicted_boxes = model(images.to(device))
            scores = torch.sigmoid(logits).cpu()
            predicted_boxes = predicted_boxes.cpu()
            for score, truth_present, predicted_box, truth_box in zip(
                scores, targets, predicted_boxes, boxes
            ):
                predicted_present = float(score) >= confidence_threshold
                actual_present = float(truth_present) > 0.5
                if predicted_present and actual_present:
                    iou = _box_iou(predicted_box.tolist(), truth_box.tolist())
                    ious.append(iou)
                    if iou >= iou_threshold:
                        true_positive += 1
                    else:
                        false_positive += 1
                        false_negative += 1
                elif predicted_present and not actual_present:
                    false_positive += 1
                elif not predicted_present and actual_present:
                    false_negative += 1
                else:
                    true_negative += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }


def _box_iou(first: list[float], second: list[float]) -> float:
    left_a, top_a, right_a, bottom_a = _cxcywh_to_xyxy(first)
    left_b, top_b, right_b, bottom_b = _cxcywh_to_xyxy(second)
    intersection_left = max(left_a, left_b)
    intersection_top = max(top_a, top_b)
    intersection_right = min(right_a, right_b)
    intersection_bottom = min(bottom_a, bottom_b)
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    area_a = max(0.0, right_a - left_a) * max(0.0, bottom_a - top_a)
    area_b = max(0.0, right_b - left_b) * max(0.0, bottom_b - top_b)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _cxcywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = box
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


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
