import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from app.ml.classifier import DeepfakeClassifier, detect_faces

logger = logging.getLogger(__name__)


def _extract_frames_ffmpeg(
    video_path: str,
    num_frames: int = 30,
    ffmpeg_path: str = "ffmpeg",
) -> list[np.ndarray]:
    """Extract `num_frames` uniformly spaced frames using FFmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-vf", f"select=not(mod(n\\,1)),fps=1/1,scale=640:-1",
            "-vframes", str(num_frames),
            "-q:v", "2",
            "-y",
            out_pattern,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — install ffmpeg and ensure it's in PATH")
        except subprocess.CalledProcessError as exc:
            logger.warning("ffmpeg stderr: %s", exc.stderr.decode(errors="replace"))
            # Fall through; we may still have some frames extracted

        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        frames = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        return frames


def _extract_frames_opencv(video_path: str, num_frames: int = 30) -> list[np.ndarray]:
    """Fallback frame extraction via OpenCV when FFmpeg is unavailable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 10000  # stream — read sequentially

    indices = np.linspace(0, max(total - 1, 0), num=min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


class VideoAnalyzer:
    def __init__(
        self,
        classifier: DeepfakeClassifier,
        num_frames: int = 30,
        ffmpeg_path: str = "ffmpeg",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.classifier = classifier
        self.num_frames = num_frames
        self.ffmpeg_path = ffmpeg_path
        self.confidence_threshold = confidence_threshold

    def analyze(self, video_path: str) -> dict:
        """Analyze a video file and return a result dict."""
        try:
            frames = _extract_frames_ffmpeg(video_path, self.num_frames, self.ffmpeg_path)
        except RuntimeError:
            logger.info("FFmpeg unavailable, falling back to OpenCV for frame extraction")
            frames = _extract_frames_opencv(video_path, self.num_frames)

        if not frames:
            raise ValueError("Could not extract any frames from video")

        frame_scores: list[float] = []
        face_counts: list[int] = []

        for frame_bgr in frames:
            face_count, face_crop = detect_faces(frame_bgr)
            face_counts.append(face_count)

            if face_crop is not None:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)
            prob, _ = self.classifier.predict(pil_img)
            frame_scores.append(prob)

        total_frames = len(frame_scores)
        fake_frames = sum(1 for s in frame_scores if s >= self.confidence_threshold)
        avg_fake_prob = float(np.mean(frame_scores))
        temporal_std = float(np.std(frame_scores))

        # Temporal inconsistency boosts confidence for fakes
        # Genuine deepfakes often show high variance across frames
        adjusted_prob = avg_fake_prob
        if temporal_std > 0.2:
            adjusted_prob = min(avg_fake_prob + temporal_std * 0.2, 1.0)

        confidence = min(abs(adjusted_prob - 0.5) * 2.0, 1.0)
        total_faces = sum(face_counts)
        faces_detected = total_faces > 0

        return {
            "fake_probability": adjusted_prob,
            "confidence": confidence,
            "face_detected": faces_detected,
            "faces_count": max(face_counts) if face_counts else 0,
            "total_frames_analyzed": total_frames,
            "fake_frames_count": fake_frames,
            "frame_scores": frame_scores,
        }
