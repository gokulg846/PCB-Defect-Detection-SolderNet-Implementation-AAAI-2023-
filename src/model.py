from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


GRADCAM_TARGET_LAYER_NAME = "features[7][2].block[0]"


def _classifier_in_features(model: nn.Module) -> int:
    classifier = getattr(model, "classifier")
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        return classifier[-1].in_features
    raise ValueError("Unexpected ConvNeXt classifier layout")


def freeze_initial_stages(model: nn.Module, num_stages: int = 2) -> None:
    """Freeze the stem and first ConvNeXt stages for warm-up training."""

    for parameter in model.parameters():
        parameter.requires_grad = True

    # torchvision ConvNeXt features are: stem, stage1, downsample, stage2, ...
    freeze_until = {1: 2, 2: 4}.get(num_stages, 0)
    for idx, module in enumerate(model.features):
        if idx < freeze_until:
            for parameter in module.parameters():
                parameter.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def create_soldernet_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone_stages: bool = True,
) -> tuple[nn.Module, str]:
    """Create the ConvNeXt-Tiny classifier and expose its Grad-CAM target layer."""

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = _classifier_in_features(model)
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone_stages:
        freeze_initial_stages(model, num_stages=2)

    return model, GRADCAM_TARGET_LAYER_NAME


def get_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve names such as features[7][2].block[0] on a PyTorch module."""

    current: Any = model
    for token in layer_name.replace("]", "").split("."):
        if "[" in token:
            attr_name, index = token.split("[", maxsplit=1)
            current = getattr(current, attr_name) if attr_name else current
            current = current[int(index)]
        else:
            current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{layer_name} did not resolve to an nn.Module")
    return current


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load checkpoints saved either as raw state_dict or trainer dictionaries."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state_dict, OrderedDict):
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state_dict}
