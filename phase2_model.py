"""
Phase 2: Model definition with transfer learning.

Usage:
    poetry run python phase2_model.py --summary
    poetry run python phase2_model.py --summary --backbone efficientnet_b0
    poetry run python phase2_model.py --summary --backbone resnet50 --mode fine_tune
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from phase1_dataset import CLASSES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CLASSES = len(CLASSES)
CHECKPOINTS_DIR = Path(__file__).parent / "outputs" / "checkpoints"
DEFAULT_BACKBONE = "resnet50"
DEFAULT_DROPOUT = 0.3

SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b0", "mobilenet_v3_small"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WasteClassifier(nn.Module):
    """Pretrained CNN backbone with a custom classification head."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        backbone, in_features = _build_backbone(backbone_name)
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (feature extraction mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (full fine-tuning mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n: int) -> None:
        """Freeze all backbone layers except the last n children (partial fine-tuning)."""
        children = list(self.backbone.named_children())
        for _, layer in children[:-n]:
            for param in layer.parameters():
                param.requires_grad = False
        for _, layer in children[-n:]:
            for param in layer.parameters():
                param.requires_grad = True


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _build_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """
    Build a pretrained backbone with its head removed.

    Returns:
        (backbone, in_features) where in_features is the size of the feature
        vector fed to the custom classifier head.
    """
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif backbone_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose from: {SUPPORTED_BACKBONES}"
        )

    return backbone, in_features


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------

def print_param_summary(model: WasteClassifier) -> None:
    """Print total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"  Backbone:             {model.backbone_name}")
    print(f"  Output classes:       {model.num_classes}")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {frozen:>12,}")


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model: WasteClassifier, path: Path, **meta) -> None:
    """Save model weights and metadata to a .pt file."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": model.backbone_name,
            "num_classes": model.num_classes,
            "class_names": CLASSES,
            "state_dict": model.state_dict(),
            **meta,
        },
        path,
    )
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[WasteClassifier, dict]:
    """
    Load a WasteClassifier from a checkpoint file.

    Returns:
        (model, checkpoint_dict) — model on the requested device, full checkpoint
        dict for accessing metadata (epoch, metrics, etc.).
    """
    ckpt = torch.load(path, map_location=device)
    model = WasteClassifier(
        backbone_name=ckpt["backbone"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 2 — model definition")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print parameter summary for the selected model configuration",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        choices=SUPPORTED_BACKBONES,
        help=f"Backbone architecture (default: {DEFAULT_BACKBONE})",
    )
    parser.add_argument(
        "--mode",
        default="feature_extraction",
        choices=["feature_extraction", "fine_tune"],
        help="feature_extraction freezes the backbone; fine_tune trains all layers",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not args.summary:
        print("No action specified. Use --summary.")
        return

    print(f"\n=== Model Summary [{args.mode}] ===")
    model = WasteClassifier(backbone_name=args.backbone)

    if args.mode == "feature_extraction":
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    print_param_summary(model)
    print()


if __name__ == "__main__":
    main()
