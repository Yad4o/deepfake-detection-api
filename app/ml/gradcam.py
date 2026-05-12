import io
import logging
import os
from pathlib import Path
from typing import Optional
import uuid
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GradCAM:
    """Grad-CAM implementation for EfficientNet-B4 via hook-based gradient capture."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "backbone.blocks.6") -> None:
        self.model = model
        self.device = next(model.parameters()).device

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target = self._find_layer(target_layer_name)
        if target is not None:
            target.register_forward_hook(self._save_activation)
            target.register_full_backward_hook(self._save_gradient)
        else:
            logger.warning("GradCAM target layer '%s' not found; heatmap will be uniform", target_layer_name)

    def _find_layer(self, name: str):
        parts = name.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, pil_img: Image.Image) -> np.ndarray:
        """Return a heatmap ndarray (H×W, float32, 0–1) for the input image."""
        self.model.eval()
        tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        output = self.model(tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        if self._activations is None or self._gradients is None:
            h, w = pil_img.height, pil_img.width
            return np.zeros((h, w), dtype=np.float32)

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.ndim == 0:
            cam = np.array([[cam.item()]])

        cam = cv2.resize(cam, (pil_img.width, pil_img.height))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-colourmap heatmap onto the original image."""
    img_rgb = np.array(pil_img.convert("RGB"))
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_rgb, 1 - alpha, jet_rgb, alpha, 0)
    return Image.fromarray(blended)


def generate_and_save_gradcam(
    model,
    pil_img: Image.Image,
    upload_dir: str,
    filename_prefix: str = "gradcam",
) -> Optional[str]:
    """Generate GradCAM heatmap, save as PNG, return relative URL path."""
    try:
        gcam = GradCAM(model)
        heatmap = gcam.generate(pil_img)
        overlay = overlay_heatmap(pil_img, heatmap)

        out_name = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        out_path = Path(upload_dir) / "gradcam" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(str(out_path))

        return f"/uploads/gradcam/{out_name}"
    except Exception as exc:
        logger.warning("GradCAM generation failed: %s", exc)
        return None
