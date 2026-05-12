from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./deepfake.db"
    MODEL_WEIGHTS_PATH: str = "./model_weights/efficientnet_deepfake.pth"

    UPLOAD_DIR: str = "./uploads"
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 200
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/avi", "video/mov", "video/webm"]

    VIDEO_FRAMES_TO_SAMPLE: int = 30
    FRAME_CONFIDENCE_THRESHOLD: float = 0.5

    API_KEY: str = ""
    SECRET_KEY: str = "change-me-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    APP_NAME: str = "Deepfake Detection API"

    FFMPEG_PATH: str = "ffmpeg"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
