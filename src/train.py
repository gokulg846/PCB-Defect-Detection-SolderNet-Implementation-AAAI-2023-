from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CLASS_NAMES, CLASS_TO_IDX, DeepPCBCropDataset, get_eval_transforms, get_train_transforms
from src.model import create_soldernet_model, unfreeze_all


def compute_class_weights(train_csv: Path, device: torch.device) -> torch.Tensor:
    labels = pd.read_csv(train_csv)["label"].map(CLASS_TO_IDX).to_numpy()
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in tqdm(dataloader, leave=False, desc="train" if is_train else "eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = running_loss / max(1, len(dataloader.dataset))
    accuracy = accuracy_score(targets, predictions) if targets else 0.0
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0) if targets else 0.0
    return average_loss, accuracy, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SolderNet-style ConvNeXt-Tiny on DeepPCB crops.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-csv", type=Path, default=Path("results/training_log.csv"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DeepPCBCropDataset(args.train_csv, transform=get_train_transforms(args.image_size))
    val_dataset = DeepPCBCropDataset(args.val_csv, transform=get_eval_transforms(args.image_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, target_layer_name = create_soldernet_model(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
        freeze_backbone_stages=True,
    )
    model.to(device)

    class_weights = compute_class_weights(args.train_csv, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    with args.log_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_epochs + 1:
                unfreeze_all(model)

            train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_accuracy, val_f1 = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val_accuracy:.6f}",
                    "val_f1": f"{val_f1:.6f}",
                    "lr": f"{lr:.8f}",
                }
            )
            handle.flush()

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "val_f1": val_f1,
                        "target_layer_name": target_layer_name,
                    },
                    args.checkpoint_dir / "best_model.pth",
                )

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}"
            )


if __name__ == "__main__":
    main()
