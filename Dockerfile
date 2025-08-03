# Multi-stage Docker build for BCI-2-Token
# Optimized for production deployment with security best practices

# ============================================================================
# Base Stage - Common dependencies
# ============================================================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r bci2token && useradd -r -g bci2token bci2token

# Set working directory
WORKDIR /app

# ============================================================================
# Dependencies Stage - Install Python packages
# ============================================================================
FROM base as dependencies

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install PyTorch CPU version for smaller image
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install -e .[deployment]

# ============================================================================
# Development Stage - For development and testing
# ============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install -e .[dev,realtime]

# Copy source code
COPY . .

# Set ownership
RUN chown -R bci2token:bci2token /app

# Switch to non-root user
USER bci2token

# Default command for development
CMD ["python", "-m", "bci2token.cli", "--help"]

# ============================================================================
# Production Stage - Optimized for deployment
# ============================================================================
FROM dependencies as production

# Copy only necessary files
COPY src/ ./src/
COPY LICENSE ./
COPY README.md ./

# Install the package
RUN pip install -e . --no-deps

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R bci2token:bci2token /app

# Security: Switch to non-root user
USER bci2token

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bci2token; print('OK')" || exit 1

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["uvicorn", "bci2token.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Testing Stage - For CI/CD testing
# ============================================================================
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests as part of build
RUN python -m pytest tests/ -v -m "unit and not slow" --tb=short

# ============================================================================
# Documentation Stage - For building docs
# ============================================================================
FROM dependencies as docs

# Install documentation dependencies
RUN pip install sphinx sphinx-rtd-theme sphinxcontrib-napoleon

# Copy source and docs
COPY src/ ./src/
COPY docs/ ./docs/
COPY README.md ./

# Build documentation
RUN cd docs && make html

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "docs/_build/html"]

# ============================================================================
# Jupyter Stage - For research and development
# ============================================================================
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install jupyter jupyterlab ipywidgets matplotlib seaborn

# Create notebook directory
RUN mkdir -p /app/notebooks && \
    chown -R bci2token:bci2token /app/notebooks

# Copy example notebooks
COPY examples/ ./examples/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# ============================================================================
# Monitoring Stage - For production monitoring
# ============================================================================
FROM production as monitoring

# Install monitoring dependencies
RUN pip install prometheus-client opentelemetry-api opentelemetry-sdk

# Copy monitoring configuration
COPY monitoring/ ./monitoring/

# Expose metrics port
EXPOSE 9090

# Default monitoring command
CMD ["python", "-m", "bci2token.monitoring.metrics_server"]