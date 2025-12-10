# Solar Panel Detection System

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout and CPU-only PyTorch
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data \
    /app/outputs \
    /app/cache/satellite_images \
    /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "app.py"]

# Usage examples:
# Build: docker build -t solar-panel-detector:v1.0 .
# Run web app: docker run -p 5000:5000 solar-panel-detector:v1.0
# Run CLI: docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs solar-panel-detector:v1.0 python cli.py batch --input /app/data/sites.xlsx --output /app/outputs/predictions.json
