# ============================================================
# Vehicle ReID API - Dockerfile
# Author: Sen Han (Member D - DevOps & Testing)
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set torch hub cache directory so IBN-Net code is cached inside the image
ENV TORCH_HOME=/app/.cache/torch

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download IBN-Net architecture code during build (avoids network calls at runtime)
RUN python -c "import torch; torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=False)"

# Copy application code
COPY app.py train_kg_gnn.py ./

# Copy frontend (Member B's files - added when available)
COPY frontend/ ./frontend/

# Create runtime directories for uploads and dynamic gallery cache
RUN mkdir -p runtime/uploads runtime/gallery/current/images runtime/cache

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
