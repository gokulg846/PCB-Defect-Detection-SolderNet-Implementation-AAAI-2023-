from __future__ import annotations

import argparse
import csv
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB.git"
CLASS_ID_TO_NAME = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values()) + ["good"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Annotation:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clone_deeppcb(raw_dir: Path, force: bool = False) -> Path:
    dataset_dir = raw_dir / "DeepPCB"
    if force and dataset_dir.exists():
        raise FileExistsError(
            f"{dataset_dir} already exists; remove it before using --force-download"
        )
    if dataset_dir.exists():
        return dataset_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", DEEPPCB_URL, str(dataset_dir)],
        check=True,
    )
    return dataset_dir


def parse_annotation_file(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2 = [int(float(value)) for value in parts[:4]]
            class_id = int(float(parts[4]))
        except ValueError:
            continue

        label = CLASS_ID_TO_NAME.get(class_id)
        if label is None:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        annotations.append(Annotation(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
    return annotations


def build_image_index(dataset_dir: Path) -> dict[str, list[Path]]:
    image_index: dict[str, list[Path]] = {}
    for image_path in dataset_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = image_path.stem
        keys = {
            stem,
            stem.replace("_test", ""),
            stem.replace("_temp", ""),
            stem.replace("_template", ""),
        }
        for key in keys:
            image_index.setdefault(key, []).append(image_path)
    return image_index


def find_test_image(annotation_path: Path, image_index: dict[str, list[Path]]) -> Path | None:
    candidates = image_index.get(annotation_path.stem, [])
    if not candidates:
        return None
    test_images = [
        path for path in candidates if "test" in path.stem.lower() or "test" in path.parent.name.lower()
    ]
    return sorted(test_images or candidates)[0]


def padded_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def sample_good_box(
    image_size: tuple[int, int],
    defect_boxes: list[tuple[int, int, int, int]],
    patch_size: int,
    rng: random.Random,
    max_attempts: int = 100,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    crop_width = min(patch_size, width)
    crop_height = min(patch_size, height)
    if crop_width <= 0 or crop_height <= 0:
        return None

    for _ in range(max_attempts):
        x1 = rng.randint(0, max(0, width - crop_width))
        y1 = rng.randint(0, max(0, height - crop_height))
        candidate = (x1, y1, x1 + crop_width, y1 + crop_height)
        if all(iou(candidate, defect_box) == 0.0 for defect_box in defect_boxes):
            return candidate
    return None


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(box)
    crop.save(output_path, quality=95)


def split_records(
    records: list[dict[str, str]],
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    labels = [record["label"] for record in records]
    try:
        train_records, temp_records = train_test_split(
            records,
            train_size=0.70,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [record["label"] for record in temp_records]
        val_records, test_records = train_test_split(
            temp_records,
            test_size=0.50,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError:
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        train_end = int(0.70 * len(shuffled))
        val_end = train_end + int(0.15 * len(shuffled))
        train_records = shuffled[:train_end]
        val_records = shuffled[train_end:val_end]
        test_records = shuffled[val_end:]
    return train_records, val_records, test_records


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(records)


def create_classification_dataset(
    dataset_dir: Path,
    output_dir: Path,
    good_per_image: int,
    good_patch_size: int,
    seed: int,
    padding_ratio: float,
) -> list[dict[str, str]]:
    root = project_root()
    rng = random.Random(seed)
    image_index = build_image_index(dataset_dir)
    annotation_files = sorted(dataset_dir.rglob("*.txt"))
    records: list[dict[str, str]] = []

    for annotation_path in tqdm(annotation_files, desc="Cropping DeepPCB defects"):
        annotations = parse_annotation_file(annotation_path)
        if not annotations:
            continue

        image_path = find_test_image(annotation_path, image_index)
        if image_path is None:
            continue

        image = Image.open(image_path).convert("RGB")
        defect_boxes = [annotation.box for annotation in annotations]
        sample_stem = image_path.stem

        for idx, annotation in enumerate(annotations):
            crop_box = padded_box(annotation.box, image.size, padding_ratio)
            output_path = output_dir / "crops" / annotation.label / f"{sample_stem}_{idx:03d}.jpg"
            save_crop(image, crop_box, output_path)
            records.append(
                {
                    "filepath": str(output_path.relative_to(root)),
                    "label": annotation.label,
                }
            )

        for idx in range(good_per_image):
            good_box = sample_good_box(image.size, defect_boxes, good_patch_size, rng)
            if good_box is None:
                continue
            output_path = output_dir / "crops" / "good" / f"{sample_stem}_good_{idx:03d}.jpg"
            save_crop(image, good_box, output_path)
            records.append(
                {
                    "filepath": str(output_path.relative_to(root)),
                    "label": "good",
                }
            )

    if not records:
        raise RuntimeError(
            "No samples were created. Check that the DeepPCB clone contains annotation files "
            "and matching PCB test images."
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepPCB and create crop-level CSV splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--good-per-image", type=int, default=3)
    parser.add_argument("--good-patch-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing data/raw/DeepPCB directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Fail if data/raw/DeepPCB already exists instead of reusing it.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    dataset_dir = raw_dir / "DeepPCB" if args.skip_download else clone_deeppcb(raw_dir, args.force_download)

    records = create_classification_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        good_per_image=args.good_per_image,
        good_patch_size=args.good_patch_size,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
    )
    train_records, val_records, test_records = split_records(records, args.seed)

    write_csv(output_dir / "train.csv", train_records)
    write_csv(output_dir / "val.csv", val_records)
    write_csv(output_dir / "test.csv", test_records)

    print(f"Created {len(records)} samples across {len(CLASS_NAMES)} classes")
    print(f"Train/val/test: {len(train_records)}/{len(val_records)}/{len(test_records)}")
    print(f"CSV splits written under {output_dir}")


if __name__ == "__main__":
    main()
