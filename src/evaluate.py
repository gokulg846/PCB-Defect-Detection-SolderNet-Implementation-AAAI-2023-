from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CLASS_NAMES, DeepPCBCropDataset, get_eval_transforms
from src.model import create_soldernet_model, load_checkpoint


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return targets, predictions


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="DeepPCB Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SolderNet-style model on DeepPCB test crops.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--metrics-json", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--confusion-matrix", type=Path, default=Path("results/confusion_matrix.png"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeepPCBCropDataset(args.test_csv, transform=get_eval_transforms(args.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, _ = create_soldernet_model(num_classes=len(CLASS_NAMES), pretrained=False, freeze_backbone_stages=False)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    targets, predictions = collect_predictions(model, dataloader, device)
    labels = list(range(len(CLASS_NAMES)))
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    save_confusion_matrix(matrix, CLASS_NAMES, args.confusion_matrix)

    per_class = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        total = int(matrix[idx].sum())
        class_accuracy = float(matrix[idx, idx] / total) if total else 0.0
        per_class[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "accuracy": class_accuracy,
            "support": int(support[idx]),
        }

    metrics = {
        "overall_accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix_png": str(args.confusion_matrix),
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
