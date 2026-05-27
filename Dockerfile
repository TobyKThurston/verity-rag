# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps first so the layer caches
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[store]"

COPY evals ./evals

# Run as a non-root user.
RUN useradd --create-home --uid 1001 verity
USER verity

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Ollama runs in its own container (docker-compose.yml); this talks to it.
CMD ["verity", "serve", "--host", "0.0.0.0", "--port", "8000"]
