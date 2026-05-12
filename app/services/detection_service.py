import time
import uuid
from pathlib import Path
from typing import Optional
from PIL import Image
import io

from sqlalchemy.orm import Session

from app.config import settings
from app.ml.classifier import get_classifier
from app.ml.video_analyzer import VideoAnalyzer
from app.ml.gradcam import generate_and_save_gradcam
from app.models.detection import Detection, MediaType, Verdict


def _determine_verdict(fake_prob: float) -> Verdict:
    if fake_prob >= 0.65:
        return Verdict.fake
    if fake_prob <= 0.35:
        return Verdict.real
    return Verdict.uncertain


def analyze_image(
    file_bytes: bytes,
    original_filename: str,
    db: Session,
    generate_gradcam: bool = True,
) -> Detection:
    start_ms = time.time()
    classifier = get_classifier(settings.MODEL_WEIGHTS_PATH)

    fake_prob, confidence, face_count, _ = classifier.predict_from_bytes(file_bytes)
    verdict = _determine_verdict(fake_prob)

    # Save upload
    safe_name = f"{uuid.uuid4().hex}_{Path(original_filename).name}"
    upload_path = Path(settings.UPLOAD_DIR) / safe_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_bytes)

    # GradCAM
    gradcam_url: Optional[str] = None
    if generate_gradcam:
        pil_img = Image.open(io.BytesIO(file_bytes))
        gradcam_url = generate_and_save_gradcam(
            classifier.model,
            pil_img,
            settings.UPLOAD_DIR,
            filename_prefix=Path(original_filename).stem,
        )

    elapsed_ms = int((time.time() - start_ms) * 1000)

    record = Detection(
        filename=safe_name,
        original_filename=original_filename,
        media_type=MediaType.image,
        file_size_bytes=len(file_bytes),
        verdict=verdict,
        fake_probability=fake_prob,
        confidence=confidence,
        face_detected=face_count > 0,
        faces_count=face_count,
        gradcam_url=gradcam_url,
        model_version="efficientnet_b4_v1",
        processing_time_ms=elapsed_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def analyze_video(
    file_bytes: bytes,
    original_filename: str,
    db: Session,
) -> Detection:
    start_ms = time.time()
    classifier = get_classifier(settings.MODEL_WEIGHTS_PATH)

    # Write temp video so OpenCV/FFmpeg can open it
    safe_name = f"{uuid.uuid4().hex}_{Path(original_filename).name}"
    upload_path = Path(settings.UPLOAD_DIR) / safe_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_bytes)

    analyzer = VideoAnalyzer(
        classifier=classifier,
        num_frames=settings.VIDEO_FRAMES_TO_SAMPLE,
        ffmpeg_path=settings.FFMPEG_PATH,
        confidence_threshold=settings.FRAME_CONFIDENCE_THRESHOLD,
    )
    result = analyzer.analyze(str(upload_path))
    verdict = _determine_verdict(result["fake_probability"])
    elapsed_ms = int((time.time() - start_ms) * 1000)

    record = Detection(
        filename=safe_name,
        original_filename=original_filename,
        media_type=MediaType.video,
        file_size_bytes=len(file_bytes),
        verdict=verdict,
        fake_probability=result["fake_probability"],
        confidence=result["confidence"],
        face_detected=result["face_detected"],
        faces_count=result["faces_count"],
        total_frames_analyzed=result["total_frames_analyzed"],
        fake_frames_count=result["fake_frames_count"],
        frame_scores=result["frame_scores"],
        model_version="efficientnet_b4_v1",
        processing_time_ms=elapsed_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
