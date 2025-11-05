FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build (not cache files)
# This ensures models are available immediately when container runs
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" && \
    python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/test data/bm25_cache_5k \
    data/vector_indices_5k data/reranker_tokenization_cache_5k \
    artifacts/eval_runs

# Set environment variables (defaults, can be overridden)
ENV PYTHONUNBUFFERED=1
ENV RERANKER_DEVICE=cpu

