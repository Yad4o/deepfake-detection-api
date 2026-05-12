from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models.detection import Detection, MediaType, Verdict
from app.schemas.detection import DetectionResult, DetectionStats, DetectionSummary

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/", response_model=List[DetectionSummary])
def list_detections(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    media_type: Optional[MediaType] = None,
    verdict: Optional[Verdict] = None,
    db: Session = Depends(get_db),
):
    """Paginated list of past detections, newest first."""
    q = db.query(Detection)
    if media_type:
        q = q.filter(Detection.media_type == media_type)
    if verdict:
        q = q.filter(Detection.verdict == verdict)
    return q.order_by(Detection.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/stats", response_model=DetectionStats)
def get_stats(db: Session = Depends(get_db)):
    """Aggregate statistics across all detections."""
    total = db.query(func.count(Detection.id)).scalar() or 0
    fake = db.query(func.count(Detection.id)).filter(Detection.verdict == Verdict.fake).scalar() or 0
    real = db.query(func.count(Detection.id)).filter(Detection.verdict == Verdict.real).scalar() or 0
    uncertain = db.query(func.count(Detection.id)).filter(Detection.verdict == Verdict.uncertain).scalar() or 0
    avg_prob = db.query(func.avg(Detection.fake_probability)).scalar() or 0.0

    return DetectionStats(
        total_analyzed=total,
        fake_detected=fake,
        real_detected=real,
        uncertain=uncertain,
        avg_fake_probability=round(float(avg_prob), 4),
        fake_rate=round(fake / total, 4) if total else 0.0,
    )


@router.get("/{detection_id}", response_model=DetectionResult)
def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get full result for a single detection by ID."""
    record = db.query(Detection).filter(Detection.id == detection_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    return record


@router.delete("/{detection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_detection(detection_id: int, db: Session = Depends(get_db)):
    """Delete a detection record."""
    record = db.query(Detection).filter(Detection.id == detection_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    db.delete(record)
    db.commit()
