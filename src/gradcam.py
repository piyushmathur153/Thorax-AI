"""
Explainable AI (XAI) Module: Grad-CAM for DenseNet-121
Crash-proof implementation using Tensor backward hooks to handle
torchvision DenseNet's hardcoded inplace ReLU operations.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for DenseNet-121 with frozen backbones and inplace activations.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "features") and hasattr(model.features, "norm5"):
            self.target_layer = model.features.norm5
        elif hasattr(model, "features"):
            self.target_layer = model.features[-1]
        else:
            raise AttributeError("Unable to identify target convolutional layer.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handle = None
        self._register_forward_hook()

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def _register_forward_hook(self) -> None:
        def forward_hook(module, input_tensor, output_tensor):
            # Clone activations before DenseNet applies F.relu(..., inplace=True)
            self.activations = output_tensor.clone()
            # Register backward hook directly on the output tensor (avoids BackwardHookFunction inplace conflict)
            if output_tensor.requires_grad:
                output_tensor.register_hook(self._save_gradient)

        self.hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()

        # Ensure gradient tracking through frozen feature layers
        tensor_with_grad = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(tensor_with_grad)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradient or activation extraction failed during backward pass.")

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

        # Weighted combination of activation maps
        weighted_activations = pooled_gradients * self.activations
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        heatmap = cam.squeeze().detach().cpu().numpy()

        # Safe Min-Max Normalization
        heatmap_min = float(heatmap.min())
        heatmap_max = float(heatmap.max())
        if (heatmap_max - heatmap_min) > 1e-7:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def remove_hooks(self) -> None:
        """Cleans up the forward hook handle."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def overlay_heatmap(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
    threshold: float = 0.40,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clinically calibrated overlay:
    Eliminates integer wrap-around (blue center bug) using strict clipping,
    and applies smooth linear blending strictly on pathological hotspots.
    """
    orig_h, orig_w, _ = original_img_rgb.shape

    # 1. Resize using INTER_LINEAR to prevent peak overshoot (>1.0)
    resized_heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    resized_heatmap = cv2.GaussianBlur(resized_heatmap, (15, 15), 0)

    # 2. Strict clamp between 0.0 and 1.0 (Prevents uint8 overflow)
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    # 3. Only keep areas above clinical threshold
    focal_mask = resized_heatmap >= threshold

    # 4. Normalize ONLY the focal region so hotspots get pure warm spectrum
    normalized_focal = np.zeros_like(resized_heatmap)
    if np.any(focal_mask):
        normalized_focal[focal_mask] = (resized_heatmap[focal_mask] - threshold) / (1.0 - threshold + 1e-8)

    # 5. Strict uint8 conversion safely bounded to [0, 255]
    heatmap_uint8 = np.uint8(np.clip(255.0 * normalized_focal, 0, 255))
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # 6. Apply dynamic alpha mask strictly on the focal region (zero opacity elsewhere)
    alpha_mask = np.zeros((orig_h, orig_w, 1), dtype=np.float32)
    alpha_mask[focal_mask] = (normalized_focal[focal_mask] * alpha)[:, np.newaxis]

    # 7. Blend: 100% original radiograph on healthy areas, glowing heatmap on lesions
    blended = alpha_mask * colored_rgb + (1.0 - alpha_mask) * original_img_rgb
    superimposed = np.uint8(np.clip(blended, 0, 255))

    return superimposed, colored_rgb