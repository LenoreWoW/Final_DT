# ─────────────────────────────────────────────────────────
# Quantum Digital Twin Platform - Backend Dockerfile
# Multi-stage build for a lean production image
# ─────────────────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="Quantum Digital Twin Platform Team"
LABEL description="FastAPI backend for the Quantum Digital Twin Platform"

# Install only the runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application code
COPY backend/ ./backend/
COPY dt_project/ ./dt_project/
COPY requirements.txt .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Run with uvicorn — 4 workers for production, auto-reload disabled
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
