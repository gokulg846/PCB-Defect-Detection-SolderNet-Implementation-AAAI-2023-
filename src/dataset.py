from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
    "good",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DeepPCBCropDataset(Dataset):
    """Classification dataset over cropped DeepPCB defect patches."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        root_dir: str | Path = ".",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = pd.read_csv(self.csv_path)

        required_columns = {"filepath", "label"}
        missing = required_columns.difference(self.samples.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.samples.iloc[index]
        image_path = Path(row["filepath"])
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = row["label"]
        if isinstance(label, str):
            label_idx = CLASS_TO_IDX[label]
        else:
            label_idx = int(label)
        return image, label_idx


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def tensor_to_display_image(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""

    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
