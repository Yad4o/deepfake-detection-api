from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.models.detection import MediaType, Verdict


class DetectionResult(BaseModel):
    id: int
    filename: str
    original_filename: str
    media_type: MediaType
    verdict: Verdict
    fake_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    face_detected: bool
    faces_count: int

    # Video fields
    total_frames_analyzed: Optional[int] = None
    fake_frames_count: Optional[int] = None
    frame_scores: Optional[List[float]] = None

    # Explainability
    gradcam_url: Optional[str] = None

    # Metadata
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DetectionSummary(BaseModel):
    id: int
    original_filename: str
    media_type: MediaType
    verdict: Verdict
    fake_probability: float
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


class DetectionStats(BaseModel):
    total_analyzed: int
    fake_detected: int
    real_detected: int
    uncertain: int
    avg_fake_probability: float
    fake_rate: float
