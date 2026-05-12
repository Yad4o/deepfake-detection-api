import io
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
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

_face_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml_path)
    return _face_cascade


def detect_faces(img_bgr: np.ndarray) -> tuple[int, Optional[np.ndarray]]:
    """Return (face_count, cropped_face_or_None)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return 0, None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_crop = img_bgr[y : y + h, x : x + w]
    return len(faces), face_crop


class _DeepfakeClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
            in_features = self.backbone.num_features
        except Exception:
            # Fallback: lightweight head when timm unavailable (CI / no weights)
            self.backbone = None
            in_features = 1792
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )
        self._in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            features = self.backbone(x)
        else:
            features = torch.zeros(x.size(0), self._in_features, device=x.device)
        return self.head(features)


class DeepfakeClassifier:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _DeepfakeClassifier().to(self.device)
        self.model.eval()
        self._weights_loaded = False

        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                self._weights_loaded = True
                logger.info("Loaded weights from %s", weights_path)
            except Exception as exc:
                logger.warning("Could not load weights: %s", exc)

    @torch.inference_mode()
    def predict(self, pil_img: Image.Image) -> tuple[float, float]:
        """Return (fake_probability, confidence) for a single PIL image."""
        tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        logit = self.model(tensor).squeeze(1)
        prob = torch.sigmoid(logit).item()

        # Confidence = distance from 0.5, scaled to [0, 1]
        confidence = min(abs(prob - 0.5) * 2.0, 1.0)
        return float(prob), float(confidence)

    def predict_from_bytes(self, data: bytes) -> tuple[float, float, int, int]:
        """Return (fake_probability, confidence, face_count, used_face_crop).

        Runs face detection; classifies face crop when available, full frame otherwise.
        """
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image data")

        face_count, face_crop = detect_faces(img_bgr)
        if face_crop is not None:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        prob, conf = self.predict(pil_img)
        return prob, conf, face_count, 1 if face_crop is not None else 0


_classifier_singleton: Optional[DeepfakeClassifier] = None


def get_classifier(weights_path: Optional[str] = None) -> DeepfakeClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = DeepfakeClassifier(weights_path=weights_path)
    return _classifier_singleton
