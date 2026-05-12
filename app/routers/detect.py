from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.schemas.detection import DetectionResult
from app.services.detection_service import analyze_image, analyze_video

router = APIRouter(prefix="/detect", tags=["detection"])

_ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp"}
_ALLOWED_VIDEO = {"video/mp4", "video/avi", "video/mov", "video/webm", "video/quicktime"}


def _check_size(data: bytes, max_mb: int, label: str) -> None:
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{label} exceeds {max_mb} MB limit",
        )


@router.post("/image", response_model=DetectionResult, status_code=status.HTTP_200_OK)
async def detect_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Analyze a single image for deepfake manipulation."""
    if file.content_type not in _ALLOWED_IMAGE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}. Allowed: jpeg, png, webp",
        )
    data = await file.read()
    _check_size(data, settings.MAX_IMAGE_SIZE_MB, "Image")

    record = analyze_image(data, file.filename or "upload.jpg", db)
    return record


@router.post("/video", response_model=DetectionResult, status_code=status.HTTP_200_OK)
async def detect_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Analyze a video for deepfake manipulation across sampled frames."""
    if file.content_type not in _ALLOWED_VIDEO:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video type: {file.content_type}. Allowed: mp4, avi, mov, webm",
        )
    data = await file.read()
    _check_size(data, settings.MAX_VIDEO_SIZE_MB, "Video")

    record = analyze_video(data, file.filename or "upload.mp4", db)
    return record
