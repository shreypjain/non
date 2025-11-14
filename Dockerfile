# NoN Production Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (creates .venv)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    NON_ENVIRONMENT=production

# Create non-root user
RUN useradd -m -u 1000 nonuser && \
    mkdir -p /app && \
    chown -R nonuser:nonuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=nonuser:nonuser /app/.venv /app/.venv

# Copy application code
COPY --chown=nonuser:nonuser nons/ ./nons/
COPY --chown=nonuser:nonuser examples/ ./examples/
COPY --chown=nonuser:nonuser pyproject.toml ./

# Switch to non-root user
USER nonuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import nons; print('OK')" || exit 1

# Default command (override in docker-compose or k8s)
CMD ["python", "-c", "print('NoN container ready. Mount your application and run it.')"]
