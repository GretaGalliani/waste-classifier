"""
Phase 1: Dataset download, exploratory data analysis, and preprocessing.

Usage:
    poetry run python phase1_dataset.py --download   # download from Kaggle
    poetry run python phase1_dataset.py --split      # split into train/val/test
    poetry run python phase1_dataset.py --eda        # generate EDA plots
    poetry run python phase1_dataset.py --all        # run all steps
"""

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATASET_SLUG = "asdasdasasdas/garbage-classification"

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "split"
EDA_DIR = ROOT_DIR / "outputs" / "eda"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset() -> None:
    """Download and extract the Garbage Classification dataset from Kaggle."""
    zip_path = RAW_DIR / "garbage-classification.zip"

    print(f"Downloading dataset '{DATASET_SLUG}' ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(RAW_DIR)],
        check=True,
    )

    print("Extracting ...")
    # Extract to a clean temp folder to inspect structure before touching RAW_DIR
    tmp_dir = RAW_DIR / "_tmp_extract"
    tmp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(tmp_dir)
    zip_path.unlink()

    # Find where the class folders actually live inside the extracted tree
    class_root = _find_class_root(tmp_dir)
    print(f"  Class folders found at: {class_root}")

    # Move each class folder to RAW_DIR
    for cls in CLASSES:
        src = class_root / cls
        dst = RAW_DIR / cls
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    # Remove temp dir entirely
    shutil.rmtree(tmp_dir)
    print(f"Dataset ready in {RAW_DIR}")


def _find_class_root(base: Path) -> Path:
    """Walk extracted tree and return the directory that contains the class folders."""
    # BFS: find first directory whose children include known class names
    queue = [base]
    while queue:
        current = queue.pop(0)
        children = {p.name for p in current.iterdir() if p.is_dir()}
        if children & set(CLASSES):
            return current
        queue.extend(p for p in current.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"Could not find class folders {CLASSES} inside extracted archive."
    )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_dataset(seed: int = RANDOM_SEED) -> None:
    """Split raw images into train / val / test folders."""
    rng = random.Random(seed)

    # Clean previous split
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for cls in CLASSES:
        src = RAW_DIR / cls
        if not src.exists():
            raise FileNotFoundError(
                f"Class folder not found: {src}\n"
                "Run --download first."
            )

        all_images = sorted(src.glob("*.*"))
        rng.shuffle(all_images)

        n_total = len(all_images)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])

        buckets = {
            "train": all_images[:n_train],
            "val": all_images[n_train : n_train + n_val],
            "test": all_images[n_train + n_val :],
        }

        for split, files in buckets.items():
            dest_dir = SPLIT_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                shutil.copy(img_path, dest_dir / img_path.name)

        print(
            f"  {cls:<10}  "
            f"train={len(buckets['train'])}  "
            f"val={len(buckets['val'])}  "
            f"test={len(buckets['test'])}"
        )

    print(f"\nSplit complete → {SPLIT_DIR}")


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def run_eda() -> None:
    """Generate and save exploratory data analysis plots."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    _plot_class_distribution()
    _plot_sample_images()
    print(f"EDA plots saved to {EDA_DIR}")


def _count_images(base_dir: Path) -> dict[str, int]:
    """Return a dict mapping each class name to its image count in base_dir."""
    return {
        cls: len(list((base_dir / cls).glob("*.*")))
        for cls in CLASSES
        if (base_dir / cls).exists()
    }


def _plot_class_distribution() -> None:
    """Bar chart of image counts per class (raw dataset)."""
    counts = _count_images(RAW_DIR)
    if not counts:
        print("No images found in RAW_DIR — skipping distribution plot.")
        return

    data = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="class", y="count", hue="class", legend=False, ax=axis)

    for patch in axis.patches:
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 5,
            str(int(patch.get_height())),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_title("Class Distribution — Raw Dataset", fontsize=14, fontweight="bold")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of images")
    axis.set_ylim(0, max(counts.values()) * 1.15)
    sns.despine()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_sample_images(n_per_class: int = 4) -> None:
    """Grid of sample images, n_per_class per class."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2.5, n_classes * 2.5))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)

    rng = random.Random(RANDOM_SEED)

    for row, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        all_images = sorted(cls_dir.glob("*.*")) if cls_dir.exists() else []
        samples = rng.sample(all_images, min(n_per_class, len(all_images)))

        for col in range(n_per_class):
            axis = axes[row][col]
            axis.axis("off")
            if col < len(samples):
                img = Image.open(samples[col]).convert("RGB")
                axis.imshow(img)
            if col == 0:
                axis.set_title(cls, fontsize=10, fontweight="bold", loc="left", pad=4)

    fig.tight_layout()
    fig.savefig(EDA_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PyTorch Dataset & DataLoaders
# ---------------------------------------------------------------------------

def get_transforms(split: str) -> transforms.Compose:
    """Return the appropriate transform pipeline for a given split."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """
    Build ImageFolder datasets and return DataLoaders for train, val, and test.

    Returns:
        dict with keys 'train', 'val', 'test'.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = SPLIT_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split folder not found: {split_dir}\n"
                "Run --split first."
            )
        dataset = ImageFolder(root=str(split_dir), transform=get_transforms(split))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 1 — dataset setup and EDA")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--eda", action="store_true", help="Generate EDA plots")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_all = args.all

    if run_all or args.download:
        print("=== Download ===")
        download_dataset()

    if run_all or args.split:
        print("\n=== Split ===")
        split_dataset()

    if run_all or args.eda:
        print("\n=== EDA ===")
        run_eda()

    if not any([run_all, args.download, args.split, args.eda]):
        print("No action specified. Use --download, --split, --eda, or --all.")


if __name__ == "__main__":
    main()
