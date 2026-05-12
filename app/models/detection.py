import enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class MediaType(str, enum.Enum):
    image = "image"
    video = "video"


class Verdict(str, enum.Enum):
    real = "real"
    fake = "fake"
    uncertain = "uncertain"


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    media_type = Column(Enum(MediaType), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)

    # Core result
    verdict = Column(Enum(Verdict), nullable=False)
    fake_probability = Column(Float, nullable=False)   # 0.0 – 1.0
    confidence = Column(Float, nullable=False)          # model confidence in its own prediction

    # Video-specific
    total_frames_analyzed = Column(Integer, nullable=True)
    fake_frames_count = Column(Integer, nullable=True)
    frame_scores = Column(JSON, nullable=True)          # list of per-frame fake_probability

    # Explainability
    gradcam_url = Column(String, nullable=True)         # heatmap image path
    face_detected = Column(Boolean, default=True)
    faces_count = Column(Integer, default=1)

    # Model metadata
    model_version = Column(String, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
