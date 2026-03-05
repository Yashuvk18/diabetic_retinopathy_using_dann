from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DR_STAGE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR (Life-threatening)",
]

DR_TRIAGE_NAMES = [
    "Class 0 (No DR): Patient is healthy. See you in a year for your annual checkup.",
    "Class 1 (Mild/Moderate): Early signs of disease detected. Schedule a routine appointment with an eye doctor to monitor.",
    "Class 2 (Severe/Proliferative): Vision-threatening disease detected. Urgent medical intervention required to prevent blindness.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: dict, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_class_names(num_classes: int, class_names: list[str] | None = None) -> list[str]:
    if class_names:
        if len(class_names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class names, but got {len(class_names)}: {class_names}"
            )
        return class_names

    if num_classes == 5:
        return DR_STAGE_NAMES.copy()
    if num_classes == 3:
        return DR_TRIAGE_NAMES.copy()

    return [f"Class {idx}" for idx in range(num_classes)]


def create_classification_report(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
) -> tuple[dict, str]:
    class_ids = list(range(len(class_names)))
    report_dict = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    report_text = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=class_ids).tolist()

    payload = {
        "class_names": class_names,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }
    return payload, report_text


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    from_dann: bool = False,
    return_predictions: bool = False,
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if from_dann:
            logits = model.classify(images)
        else:
            logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    metrics: dict[str, float | list[int]] = {
        "loss": avg_loss,
        "accuracy": float(acc),
        "macro_f1": float(f1),
    }
    if return_predictions:
        metrics["labels"] = all_labels
        metrics["preds"] = all_preds

    return metrics  # type: ignore[return-value]
