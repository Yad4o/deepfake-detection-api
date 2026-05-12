import io
import numpy as np
import pytest
from PIL import Image

from app.ml.classifier import DeepfakeClassifier, detect_faces


def _make_rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_jpeg_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    _make_rgb_image(w, h).save(buf, format="JPEG")
    return buf.getvalue()


def test_classifier_predict_range():
    clf = DeepfakeClassifier(weights_path=None)
    img = _make_rgb_image()
    prob, conf = clf.predict(img)
    assert 0.0 <= prob <= 1.0
    assert 0.0 <= conf <= 1.0


def test_classifier_predict_from_bytes():
    clf = DeepfakeClassifier(weights_path=None)
    data = _make_jpeg_bytes()
    prob, conf, face_count, _ = clf.predict_from_bytes(data)
    assert 0.0 <= prob <= 1.0
    assert face_count >= 0


def test_classifier_invalid_bytes():
    clf = DeepfakeClassifier(weights_path=None)
    with pytest.raises(ValueError, match="Could not decode"):
        clf.predict_from_bytes(b"not_an_image")


def test_detect_faces_no_face():
    import cv2
    import numpy as np
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    count, crop = detect_faces(blank)
    assert count == 0
    assert crop is None


def test_verdict_thresholds():
    from app.services.detection_service import _determine_verdict
    from app.models.detection import Verdict

    assert _determine_verdict(0.8) == Verdict.fake
    assert _determine_verdict(0.2) == Verdict.real
    assert _determine_verdict(0.5) == Verdict.uncertain
