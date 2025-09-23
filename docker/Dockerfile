# Multi-stage build for optimized production image
# Build arguments for flexibility
ARG PYTHON_VERSION=3.11
ARG PORT=5051

FROM python:${PYTHON_VERSION}-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building Python packages and AI/ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libgl1-mesa-dev \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:${PYTHON_VERSION}-slim as production

# Set environment variables for the application and AI/ML libraries
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=${PORT} \
    DEBUG=False \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DOCLING_CACHE_DIR=/app/model_artifacts/docling \
    HF_HOME=/app/model_artifacts/huggingface \
    TRANSFORMERS_CACHE=/app/model_artifacts/transformers \
    TOKENIZERS_CACHE=/app/model_artifacts/tokenizers

# Install runtime dependencies for the application and AI/ML libraries
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data/chroma_db /app/data/vectordb /app/uploads \
    /app/model_artifacts/docling \
    /app/model_artifacts/huggingface \
    /app/model_artifacts/transformers \
    /app/model_artifacts/tokenizers \
    /app/model_artifacts/easyocr \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Define volumes for persistent data and model artifacts
VOLUME ["/app/data", "/app/model_artifacts"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["python", "run.py"]
