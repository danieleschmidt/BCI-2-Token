# BCI-2-Token Production Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV BCI2TOKEN_ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r bci2token && useradd -r -g bci2token bci2token

# Create application directories
RUN mkdir -p /app /var/lib/bci2token /var/log/bci2token /var/cache/bci2token \
    && chown -R bci2token:bci2token /app /var/lib/bci2token /var/log/bci2token /var/cache/bci2token

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY --chown=bci2token:bci2token . .

# Install the package
RUN pip install -e .

# Switch to non-root user
USER bci2token

# Set environment variables for directories
ENV BCI2TOKEN_DATA_DIR=/var/lib/bci2token
ENV BCI2TOKEN_LOG_DIR=/var/log/bci2token
ENV BCI2TOKEN_CACHE_DIR=/var/cache/bci2token

# Expose ports
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from bci2token.health import run_comprehensive_diagnostics; \
                   results = run_comprehensive_diagnostics(); \
                   exit(0 if all(r.level.value in ['pass', 'warning'] for r in results.values()) else 1)"

# Default command
CMD ["python", "-m", "bci2token.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]