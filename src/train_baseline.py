from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.data import RetinopathyDataset, stratified_split_indices
from src.models import BaselineClassifier
from src.utils import evaluate_classifier, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source-only baseline on APTOS and evaluate on IDRiD."
    )

    parser.add_argument("--source-csv", type=str, required=True)
    parser.add_argument("--source-img-dir", type=str, required=True)
    parser.add_argument("--source-image-col", type=str, default="image_id")
    parser.add_argument("--source-label-col", type=str, default="diagnosis")
    parser.add_argument("--source-default-ext", type=str, default=".png")
    parser.add_argument("--source-val-csv", type=str, default="")
    parser.add_argument("--source-val-img-dir", type=str, default="")
    parser.add_argument("--source-val-split", type=float, default=0.2)

    parser.add_argument("--target-csv", type=str, required=True)
    parser.add_argument("--target-img-dir", type=str, required=True)
    parser.add_argument("--target-image-col", type=str, default="image")
    parser.add_argument("--target-label-col", type=str, default="label")
    parser.add_argument("--target-default-ext", type=str, default=".jpg")

    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    return parser.parse_args()


def build_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_tf, eval_tf = build_transforms(args.img_size)

    if args.source_val_csv:
        source_train_ds = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=train_tf,
            default_ext=args.source_default_ext,
        )
        source_val_ds = RetinopathyDataset(
            csv_path=args.source_val_csv,
            image_dir=args.source_val_img_dir or args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=eval_tf,
            default_ext=args.source_default_ext,
        )
    else:
        source_train_base = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=train_tf,
            default_ext=args.source_default_ext,
        )
        source_val_base = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=eval_tf,
            default_ext=args.source_default_ext,
        )
        train_idx, val_idx = stratified_split_indices(
            source_train_base.labels, val_split=args.source_val_split, seed=args.seed
        )
        source_train_ds = Subset(source_train_base, train_idx)
        source_val_ds = Subset(source_val_base, val_idx)

    target_test_ds = RetinopathyDataset(
        csv_path=args.target_csv,
        image_dir=args.target_img_dir,
        image_col=args.target_image_col,
        label_col=args.target_label_col,
        transform=eval_tf,
        default_ext=args.target_default_ext,
    )

    source_train_loader = DataLoader(
        source_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    source_val_loader = DataLoader(
        source_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    target_test_loader = DataLoader(
        target_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = BaselineClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_source_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=source_train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        source_val_metrics = evaluate_classifier(model, source_val_loader, device)
        target_metrics = evaluate_classifier(model, target_test_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "source_val_loss": source_val_metrics["loss"],
            "source_val_acc": source_val_metrics["accuracy"],
            "source_val_macro_f1": source_val_metrics["macro_f1"],
            "target_loss": target_metrics["loss"],
            "target_acc": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"source_val_acc={row['source_val_acc']:.4f} | "
            f"target_acc={row['target_acc']:.4f}"
        )

        if source_val_metrics["accuracy"] > best_source_val_acc:
            best_source_val_acc = source_val_metrics["accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_baseline.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_source_val = evaluate_classifier(model, source_val_loader, device)
    final_target = evaluate_classifier(model, target_test_loader, device)
    final_metrics = {
        "best_source_val_acc": best_source_val_acc,
        "final_source_val": final_source_val,
        "final_target": final_target,
        "history": history,
        "args": vars(args),
    }
    save_json(final_metrics, output_dir / "metrics.json")
    print("Saved:", output_dir / "best_baseline.pt")
    print("Saved:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()

