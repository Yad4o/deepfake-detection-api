#!/usr/bin/env bash
set -e

if ! command -v python3.11 &>/dev/null; then
  echo "Python 3.11 required. Install it and re-run."
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Warning: FFmpeg not found. Video analysis will not work."
  echo "Install: sudo apt install ffmpeg  OR  brew install ffmpeg"
fi

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p uploads model_weights
echo ""
echo "Setup complete."
echo "1. Copy .env.example to .env and configure"
echo "2. Download model weights to model_weights/"
echo "3. Run: uvicorn app.main:app --reload"
