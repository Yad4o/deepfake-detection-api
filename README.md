# Deepfake Detection API

FastAPI backend for detecting AI-manipulated images and videos using an EfficientNet-B4 CNN classifier with GradCAM explainability.

## Features

- **Image analysis** — face-aware deepfake classification (JPEG, PNG, WebP)
- **Video analysis** — temporal frame sampling + inconsistency scoring (MP4, AVI, MOV, WebM)
- **GradCAM heatmaps** — visualize which facial regions triggered the model
- **PostgreSQL** history with pagination, filtering, and aggregate stats
- **Docker Compose** — single command to run API + database

## Architecture

```
POST /detect/image  ─►  Face Detection (OpenCV Haar)
                    ─►  EfficientNet-B4 (timm) → sigmoid → fake_probability
                    ─►  GradCAM heatmap → /uploads/gradcam/*.png
                    ─►  Persist to DB → DetectionResult

POST /detect/video  ─►  FFmpeg frame extraction (uniform sampling)
                    ─►  Per-frame face detection + classification
                    ─►  Temporal std-dev boost for high-variance sequences
                    ─►  Persist to DB → DetectionResult

GET  /history/      ─►  Paginated detection log
GET  /history/stats ─►  Aggregate stats (fake_rate, avg_fake_probability)
```

## Verdict thresholds

| fake_probability | Verdict   |
|-----------------|-----------|
| ≥ 0.65          | **FAKE**  |
| ≤ 0.35          | **REAL**  |
| 0.35 – 0.65     | UNCERTAIN |

## Quick start (Docker)

```bash
cp .env.example .env
# Add your Neon/PostgreSQL connection string to .env
docker-compose up --build
```

API available at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

## Local development

```bash
# Requires Python 3.11 + ffmpeg
./setup.sh
source ./activate
cp .env.example .env

uvicorn app.main:app --reload
```

## Model weights

The API runs without weights (random predictions) for development. For production:

1. Download a pretrained EfficientNet-B4 deepfake checkpoint (e.g. from FaceForensics++ trained models)
2. Place it at `model_weights/efficientnet_deepfake.pth`
3. Update `MODEL_WEIGHTS_PATH` in `.env`

The model head expects a binary classifier (1 output unit, sigmoid activation). Fine-tune on [FaceForensics++](https://github.com/ondyari/FaceForensics) or [DFDC](https://ai.facebook.com/datasets/dfdc/) datasets.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite local | PostgreSQL connection string |
| `MODEL_WEIGHTS_PATH` | `./model_weights/efficientnet_deepfake.pth` | Path to .pth checkpoint |
| `UPLOAD_DIR` | `./uploads` | Storage for uploaded media + GradCAM images |
| `MAX_IMAGE_SIZE_MB` | 10 | Upload size limit for images |
| `MAX_VIDEO_SIZE_MB` | 200 | Upload size limit for videos |
| `VIDEO_FRAMES_TO_SAMPLE` | 30 | Frames extracted per video |
| `FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |

## API Reference

### POST `/detect/image`
Upload a JPEG/PNG/WebP image. Returns verdict + confidence + GradCAM URL.

### POST `/detect/video`
Upload MP4/AVI/MOV/WebM. Returns per-frame scores + temporal analysis.

### GET `/history/`
List past detections. Query params: `skip`, `limit`, `media_type`, `verdict`.

### GET `/history/stats`
Returns `{ total_analyzed, fake_detected, real_detected, uncertain, avg_fake_probability, fake_rate }`.

### GET `/history/{id}`
Full result for a single detection including `frame_scores` and `gradcam_url`.

## Running tests

```bash
source ./activate
pip install pytest httpx
pytest tests/ -v
```

## Tech stack

- **FastAPI** + Uvicorn
- **PyTorch 2.3** + **timm** (EfficientNet-B4)
- **OpenCV** (face detection, frame I/O)
- **FFmpeg** (video decoding)
- **SQLAlchemy 2.0** + **Alembic**
- **PostgreSQL** (production) / SQLite (dev/test)
