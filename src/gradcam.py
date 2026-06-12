from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


class GradCAM:
    """Minimal Grad-CAM implementation for convolutional feature maps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()
        if float(cam_np.max()) > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        height, width = input_tensor.shape[-2:]
        heatmap = cv2.resize(cam_np.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(heatmap, 0.0, 1.0)


def overlay_heatmap(image: Image.Image | np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a red-to-yellow heatmap over an RGB image."""

    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.asarray(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

    heatmap = np.clip(heatmap, 0.0, 1.0)
    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    color = np.zeros_like(image_np, dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = heatmap * 255.0
    color[..., 2] = 0.0

    blended = (1.0 - alpha) * image_np.astype(np.float32) + alpha * color
    return np.clip(blended, 0, 255).astype(np.uint8)
