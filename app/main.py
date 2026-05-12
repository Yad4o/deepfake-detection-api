from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings
from app.database import engine, Base
from app.routers import detect, history

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Deepfake Detection API",
    description="EfficientNet-B4 powered deepfake detection for images and videos with GradCAM explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files and GradCAM heatmaps
upload_dir = Path(settings.UPLOAD_DIR)
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")

app.include_router(detect.router)
app.include_router(history.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
