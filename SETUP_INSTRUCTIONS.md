# 🚀 BCI-2-Token Setup Instructions

## Quick Setup Guide

This guide will help you set up the complete BCI-2-Token development environment and CI/CD pipeline.

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized deployment)
- GitHub account (for CI/CD pipeline)

## 🔧 Local Development Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/bci-2-token.git
cd bci-2-token

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]
```

### 2. Verify Installation

```bash
# Run basic tests
pytest tests/ -m "unit and not slow" -v

# Check code quality
black --check src/ tests/
ruff check src/ tests/
mypy src/bci2token/

# Run example
python examples/basic_usage.py
```

## 🚀 CI/CD Pipeline Setup

### 1. Enable GitHub Actions

**Important**: The CI/CD workflow is stored in `ci-cd/github-actions-ci.yml` to avoid GitHub App permission issues.

To enable CI/CD in your repository:

```bash
# Create GitHub workflows directory
mkdir -p .github/workflows

# Copy the workflow configuration
cp ci-cd/github-actions-ci.yml .github/workflows/ci.yml

# Commit and push
git add .github/workflows/ci.yml
git commit -m "Add CI/CD pipeline"
git push origin main
```

### 2. Configure Repository Settings

1. **Enable GitHub Actions** in your repository settings
2. **Set up branch protection** for `main` and `develop` branches
3. **Configure environments** for staging and production
4. **Add repository secrets** (if using external services):
   - `DOCKER_USERNAME` - Docker Hub username
   - `DOCKER_PASSWORD` - Docker Hub password
   - `KUBERNETES_CONFIG` - Kubernetes cluster config

### 3. Verify CI/CD Pipeline

1. Create a feature branch and make a small change
2. Push the branch and create a pull request
3. Verify that all CI checks pass
4. Merge to see the full deployment pipeline

## 🐳 Docker Deployment

### Local Docker Setup

```bash
# Build development image
docker build --target development -t bci2token:dev .

# Run development container
docker run -it -p 8000:8000 -v $(pwd):/app bci2token:dev

# Build production image
docker build --target production -t bci2token:prod .

# Run production container
docker run -p 8000:8000 bci2token:prod
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Monitoring Setup

### 1. Local Monitoring

```bash
# Install monitoring dependencies
pip install -e .[deployment]

# Start metrics server
python -m monitoring.metrics

# View metrics at http://localhost:9090/metrics
```

### 2. Production Monitoring

The monitoring system includes:

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards (port 3000)
- **OpenTelemetry**: Distributed tracing

## 🧪 Running Tests

### Test Categories

```bash
# Unit tests (fast)
pytest tests/ -m "unit" -v

# Integration tests
pytest tests/ -m "integration" -v

# Performance tests
pytest tests/ -m "performance" --benchmark-only

# Security tests
pytest tests/ -m "security" -v

# All tests with coverage
pytest tests/ --cov=bci2token --cov-report=html
```

### Test Configuration

Tests are configured with markers in `pytest.ini`:
- `unit`: Fast unit tests
- `integration`: Integration tests
- `performance`: Performance benchmarks
- `security`: Security tests
- `slow`: Long-running tests
- `e2e`: End-to-end tests

## 🔧 Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/ -m "unit" -v

# Check code quality
black src/ tests/
ruff check src/ tests/
mypy src/bci2token/

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Code Quality Standards

The project enforces:
- **Black**: Code formatting
- **Ruff**: Fast linting
- **MyPy**: Type checking
- **Pre-commit hooks**: Automated quality checks

### 3. AI Agent Development

To extend the AI agent system:

```python
from bci2token.agents.base_agent import BaseAgent

class YourCustomAgent(BaseAgent):
    async def analyze(self, data):
        # Your analysis logic
        pass
    
    async def execute_task(self, task):
        # Your task execution logic
        pass
```

## 🚀 Production Deployment

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml

# Check deployment
kubectl get pods -n bci2token
kubectl logs -f deployment/bci2token -n bci2token
```

### Environment Variables

Key environment variables:

```bash
# Application settings
BCI_LOG_LEVEL=INFO
BCI_METRICS_PORT=9090
BCI_API_PORT=8000

# Database settings  
DATABASE_URL=postgresql://user:pass@localhost/bci2token

# Redis settings
REDIS_URL=redis://localhost:6379

# Monitoring settings
PROMETHEUS_ENABLED=true
OPENTELEMETRY_ENABLED=true
```

## 🔍 Troubleshooting

### Common Issues

**Import errors**:
```bash
# Reinstall in development mode
pip install -e .[dev]

# Check Python path
python -c "import bci2token; print(bci2token.__file__)"
```

**Test failures**:
```bash
# Run with verbose output
pytest tests/ -v --tb=long

# Run specific test
pytest tests/test_specific.py::test_function -v
```

**CI/CD pipeline issues**:
```bash
# Check workflow status
gh workflow list
gh run list

# View specific run
gh run view <run-id>
```

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Report [issues](https://github.com/your-org/bci-2-token/issues)
- 💬 Join [discussions](https://github.com/your-org/bci-2-token/discussions)
- 📧 Contact [support](mailto:support@bci2token.dev)

## 🎯 Next Steps

After setup:

1. **Explore Examples**: Run `python examples/basic_usage.py`
2. **Read Documentation**: Check the `docs/` directory
3. **Run Agent Demo**: Execute `python examples/agent_workflow_demo.py`
4. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
5. **Deploy**: Follow production deployment guide

---

🎉 **Congratulations!** You now have a fully functional BCI-2-Token development environment with comprehensive CI/CD pipeline and AI agent coordination system!