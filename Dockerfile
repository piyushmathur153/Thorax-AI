# Stage 1: Build React Frontend using Node 20 LTS
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# Cache npm layers
COPY frontend/package*.json ./
RUN npm ci || npm install

# Build production assets to frontend/dist
COPY frontend/ ./
RUN npm run build

# Stage 2: Fast Inference Python Backend (Debian Slim)
FROM python:3.11-slim
WORKDIR /app

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Essential runtime dependencies for OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code and models
COPY . .

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose Hugging Face Spaces port
EXPOSE 7860

# Launch server on 0.0.0.0:7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]