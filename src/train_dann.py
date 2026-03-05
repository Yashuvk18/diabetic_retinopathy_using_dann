from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.data import RetinopathyDataset, stratified_split_indices
from src.models import DANN
from src.utils import (
    create_classification_report,
    evaluate_classifier,
    get_class_names,
    get_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DANN for APTOS (source) -> IDRiD (target) domain adaptation."
    )

    parser.add_argument("--source-csv", type=str, required=True)
    parser.add_argument("--source-img-dir", type=str, required=True)
    parser.add_argument("--source-image-col", type=str, default="image_id")
    parser.add_argument("--source-label-col", type=str, default="diagnosis")
    parser.add_argument("--source-default-ext", type=str, default=".png")
    parser.add_argument("--source-val-csv", type=str, default="")
    parser.add_argument("--source-val-img-dir", type=str, default="")
    parser.add_argument("--source-val-split", type=float, default=0.2)

    parser.add_argument("--target-train-csv", type=str, required=True)
    parser.add_argument("--target-train-img-dir", type=str, required=True)
    parser.add_argument("--target-train-image-col", type=str, default="image")
    parser.add_argument("--target-train-label-col", type=str, default="")
    parser.add_argument("--target-train-default-ext", type=str, default=".jpg")

    parser.add_argument("--target-test-csv", type=str, required=True)
    parser.add_argument("--target-test-img-dir", type=str, required=True)
    parser.add_argument("--target-test-image-col", type=str, default="image")
    parser.add_argument("--target-test-label-col", type=str, default="label")
    parser.add_argument("--target-test-default-ext", type=str, default=".jpg")

    parser.add_argument("--label-scheme", type=str, default="three", choices=["three", "five"])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--class-names", type=str, default="")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--feature-lr-mult", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--domain-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-grl-lambda", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--class-head-lr-mult", type=float, default=1.0)
    parser.add_argument("--domain-lr-mult", type=float, default=0.25)

    parser.add_argument("--class-balanced-loss", dest="class_balanced_loss", action="store_true")
    parser.add_argument("--no-class-balanced-loss", dest="class_balanced_loss", action="store_false")
    parser.add_argument("--balance-source-sampler", dest="balance_source_sampler", action="store_true")
    parser.add_argument("--no-balance-source-sampler", dest="balance_source_sampler", action="store_false")
    parser.set_defaults(class_balanced_loss=True, balance_source_sampler=False)

    parser.add_argument(
        "--model-selection",
        type=str,
        default="source_val_macro_f1",
        choices=["source_val_macro_f1", "source_val_acc", "target_acc"],
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b2", "resnet18"],
    )
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--output-dir", type=str, default="outputs/dann")
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


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def parse_class_names(raw: str, num_classes: int) -> list[str]:
    if raw.strip():
        names = [part.strip() for part in raw.split(",") if part.strip()]
        return get_class_names(num_classes, names)
    return get_class_names(num_classes)


def get_label_map(label_scheme: str) -> dict[int, int] | None:
    if label_scheme == "three":
        # 0->No DR, (1,2)->Mild/Moderate, (3,4)->Severe/Proliferative
        return {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    return None


def expected_num_classes(label_scheme: str) -> int:
    return 3 if label_scheme == "three" else 5


def extract_labels(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if not hasattr(parent, "labels"):
            raise ValueError("Subset parent dataset does not expose labels.")
        parent_labels = list(parent.labels)
        return [int(parent_labels[idx]) for idx in dataset.indices]

    if hasattr(dataset, "labels"):
        return [int(v) for v in dataset.labels]

    raise ValueError("Dataset does not expose labels for class balancing.")


def compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int).tolist()


def build_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    per_class = 1.0 / counts
    sample_weights = [per_class[y] for y in labels]
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(sample_weights), replacement=True)


def schedule_adversarial_terms(
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    max_grl_lambda: float,
    domain_loss_weight: float,
) -> tuple[float, float]:
    if global_step < warmup_steps:
        return 0.0, 0.0

    effective_total = max(1, total_steps - warmup_steps)
    p = (global_step - warmup_steps) / effective_total
    p = min(max(p, 0.0), 1.0)

    lambda_grl = max_grl_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
    # Keep adversarial weight non-vanishing after warmup; GRL already provides progression.
    current_domain_weight = domain_loss_weight
    return float(lambda_grl), float(current_domain_weight)


def selection_score(
    model_selection: str,
    source_val_metrics: dict[str, float],
    target_metrics: dict[str, float],
) -> float:
    if model_selection == "source_val_acc":
        return float(source_val_metrics["accuracy"])
    if model_selection == "target_acc":
        return float(target_metrics["accuracy"])
    return float(source_val_metrics["macro_f1"])


def train_one_epoch_dann(
    model: DANN,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    max_grl_lambda: float,
    domain_loss_weight: float,
    entropy_weight: float,
    grad_clip_norm: float,
):
    model.train()
    steps_per_epoch = max(len(source_loader), len(target_loader))
    total_steps = max(1, total_epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    total_loss = 0.0
    total_class_loss = 0.0
    total_domain_loss = 0.0
    total_entropy_loss = 0.0

    total_correct = 0
    total_seen = 0

    total_domain_correct = 0
    total_domain_seen = 0

    lambda_sum = 0.0
    domain_weight_sum = 0.0

    pbar = tqdm(range(steps_per_epoch), desc="train_dann", leave=False)
    for step in pbar:
        (x_s, y_s), source_iter = _next_batch(source_iter, source_loader)
        (x_t, _), target_iter = _next_batch(target_iter, target_loader)

        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        global_step = epoch * steps_per_epoch + step
        lambda_grl, current_domain_weight = schedule_adversarial_terms(
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            max_grl_lambda=max_grl_lambda,
            domain_loss_weight=domain_loss_weight,
        )

        x_cat = torch.cat([x_s, x_t], dim=0)
        class_logits, domain_logits = model(x_cat, lambda_grl=lambda_grl)

        class_logits_s = class_logits[: x_s.size(0)]
        class_loss = class_criterion(class_logits_s, y_s)

        domain_labels = torch.cat(
            [
                torch.zeros(x_s.size(0), dtype=torch.long, device=device),
                torch.ones(x_t.size(0), dtype=torch.long, device=device),
            ],
            dim=0,
        )
        domain_loss = domain_criterion(domain_logits, domain_labels)

        entropy_loss = torch.tensor(0.0, device=device)
        if entropy_weight > 0.0:
            target_logits = class_logits[x_s.size(0) :]
            target_probs = torch.softmax(target_logits, dim=1)
            entropy_loss = -(target_probs * torch.log(target_probs + 1e-8)).sum(dim=1).mean()

        loss = class_loss + current_domain_weight * domain_loss + entropy_weight * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        cls_preds = torch.argmax(class_logits_s, dim=1)
        total_correct += (cls_preds == y_s).sum().item()
        total_seen += y_s.size(0)

        dom_preds = torch.argmax(domain_logits, dim=1)
        total_domain_correct += (dom_preds == domain_labels).sum().item()
        total_domain_seen += domain_labels.size(0)

        total_loss += loss.item() * y_s.size(0)
        total_class_loss += class_loss.item() * y_s.size(0)
        total_domain_loss += domain_loss.item() * domain_labels.size(0)
        total_entropy_loss += entropy_loss.item() * x_t.size(0)

        lambda_sum += lambda_grl
        domain_weight_sum += current_domain_weight

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{class_loss.item():.4f}",
            dom=f"{domain_loss.item():.4f}",
            d_w=f"{current_domain_weight:.3f}",
            lmbd=f"{lambda_grl:.3f}",
        )

    return {
        "loss": total_loss / max(1, total_seen),
        "class_loss": total_class_loss / max(1, total_seen),
        "domain_loss": total_domain_loss / max(1, total_domain_seen),
        "entropy_loss": total_entropy_loss / max(1, steps_per_epoch * target_loader.batch_size),
        "source_train_acc": total_correct / max(1, total_seen),
        "domain_acc": total_domain_correct / max(1, total_domain_seen),
        "avg_lambda": lambda_sum / max(1, steps_per_epoch),
        "avg_domain_weight": domain_weight_sum / max(1, steps_per_epoch),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_classes = expected_num_classes(args.label_scheme)
    if args.num_classes != expected_classes:
        print(
            f"Overriding num_classes from {args.num_classes} to {expected_classes} "
            f"for label scheme '{args.label_scheme}'."
        )
    args.num_classes = expected_classes
    label_map = get_label_map(args.label_scheme)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Label scheme: {args.label_scheme}")

    class_names = parse_class_names(args.class_names, args.num_classes)
    print("Classification stages:", class_names)

    train_tf, eval_tf = build_transforms(args.img_size)

    if args.source_val_csv:
        source_train_ds = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=train_tf,
            default_ext=args.source_default_ext,
            label_map=label_map,
        )
        source_val_ds = RetinopathyDataset(
            csv_path=args.source_val_csv,
            image_dir=args.source_val_img_dir or args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=eval_tf,
            default_ext=args.source_default_ext,
            label_map=label_map,
        )
    else:
        source_train_base = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=train_tf,
            default_ext=args.source_default_ext,
            label_map=label_map,
        )
        source_val_base = RetinopathyDataset(
            csv_path=args.source_csv,
            image_dir=args.source_img_dir,
            image_col=args.source_image_col,
            label_col=args.source_label_col,
            transform=eval_tf,
            default_ext=args.source_default_ext,
            label_map=label_map,
        )
        train_idx, val_idx = stratified_split_indices(
            source_train_base.labels, val_split=args.source_val_split, seed=args.seed
        )
        source_train_ds = Subset(source_train_base, train_idx)
        source_val_ds = Subset(source_val_base, val_idx)

    target_train_label_col = args.target_train_label_col or None
    target_train_ds = RetinopathyDataset(
        csv_path=args.target_train_csv,
        image_dir=args.target_train_img_dir,
        image_col=args.target_train_image_col,
        label_col=target_train_label_col,
        transform=train_tf,
        default_ext=args.target_train_default_ext,
        label_map=label_map,
    )
    target_test_ds = RetinopathyDataset(
        csv_path=args.target_test_csv,
        image_dir=args.target_test_img_dir,
        image_col=args.target_test_image_col,
        label_col=args.target_test_label_col,
        transform=eval_tf,
        default_ext=args.target_test_default_ext,
        label_map=label_map,
    )

    source_train_labels = extract_labels(source_train_ds)
    source_class_weights = None
    source_class_counts = np.bincount(np.asarray(source_train_labels), minlength=args.num_classes)
    print("Source train class counts:", source_class_counts.tolist())

    if args.class_balanced_loss:
        source_class_weights, _ = compute_class_weights(source_train_labels, args.num_classes, device)
        print("Using class-balanced loss weights:", [round(float(v), 4) for v in source_class_weights.cpu().tolist()])

    source_sampler = None
    if args.balance_source_sampler:
        source_sampler = build_weighted_sampler(source_train_labels, args.num_classes)
        print("Using weighted source sampler.")

    source_train_loader = DataLoader(
        source_train_ds,
        batch_size=args.batch_size,
        shuffle=(source_sampler is None),
        sampler=source_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    source_val_loader = DataLoader(
        source_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    target_train_loader = DataLoader(
        target_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    target_test_loader = DataLoader(
        target_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DANN(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)

    feature_params = list(model.feature_extractor.parameters())
    class_head_params = list(model.class_classifier.parameters())
    domain_head_params = list(model.domain_classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": feature_params, "lr": args.lr * args.feature_lr_mult},
            {"params": class_head_params, "lr": args.lr * args.class_head_lr_mult},
            {"params": domain_head_params, "lr": args.lr * args.domain_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )

    class_criterion = nn.CrossEntropyLoss(
        weight=source_class_weights,
        label_smoothing=args.label_smoothing,
    )
    domain_criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_selection_score = -1e18
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_dann(
            model=model,
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            optimizer=optimizer,
            class_criterion=class_criterion,
            domain_criterion=domain_criterion,
            device=device,
            epoch=epoch - 1,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            max_grl_lambda=args.max_grl_lambda,
            domain_loss_weight=args.domain_loss_weight,
            entropy_weight=args.entropy_weight,
            grad_clip_norm=args.grad_clip_norm,
        )

        source_val_metrics = evaluate_classifier(model, source_val_loader, device, from_dann=True)
        target_metrics = evaluate_classifier(model, target_test_loader, device, from_dann=True)
        scheduler.step()

        current_score = selection_score(args.model_selection, source_val_metrics, target_metrics)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_class_loss": train_metrics["class_loss"],
            "train_domain_loss": train_metrics["domain_loss"],
            "train_entropy_loss": train_metrics["entropy_loss"],
            "source_train_acc": train_metrics["source_train_acc"],
            "domain_acc": train_metrics["domain_acc"],
            "avg_lambda": train_metrics["avg_lambda"],
            "avg_domain_weight": train_metrics["avg_domain_weight"],
            "source_val_loss": source_val_metrics["loss"],
            "source_val_acc": source_val_metrics["accuracy"],
            "source_val_macro_f1": source_val_metrics["macro_f1"],
            "target_loss": target_metrics["loss"],
            "target_acc": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
            "selection_score": current_score,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"src_train_acc={row['source_train_acc']:.4f} | "
            f"src_val_f1={row['source_val_macro_f1']:.4f} | "
            f"target_acc={row['target_acc']:.4f} | "
            f"dom_acc={row['domain_acc']:.4f}"
        )

        if current_score > best_selection_score:
            best_selection_score = current_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_dann.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_source_val = evaluate_classifier(model, source_val_loader, device, from_dann=True)
    final_target = evaluate_classifier(
        model,
        target_test_loader,
        device,
        from_dann=True,
        return_predictions=True,
    )
    target_labels = [int(v) for v in final_target.pop("labels", [])]
    target_preds = [int(v) for v in final_target.pop("preds", [])]

    target_report_payload, target_report_text = create_classification_report(
        target_labels,
        target_preds,
        class_names,
    )

    save_json(target_report_payload, output_dir / "target_classification_report.json")
    with (output_dir / "target_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(target_report_text)

    final_metrics = {
        "best_selection_score": best_selection_score,
        "model_selection": args.model_selection,
        "final_source_val": final_source_val,
        "final_target": final_target,
        "source_class_counts": source_class_counts.astype(int).tolist(),
        "target_class_names": class_names,
        "history": history,
        "args": vars(args),
    }
    save_json(final_metrics, output_dir / "metrics.json")

    print("Saved:", output_dir / "best_dann.pt")
    print("Saved:", output_dir / "metrics.json")
    print("Saved:", output_dir / "target_classification_report.json")
    print("Saved:", output_dir / "target_classification_report.txt")


if __name__ == "__main__":
    main()
