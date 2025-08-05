#!/bin/bash
set -euo pipefail

# BCI-2-Token Production Deployment Script
# Automates deployment to various environments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ENVIRONMENT="${1:-production}"
DEPLOYMENT_TYPE="${2:-docker}"
VERSION="${3:-latest}"
NAMESPACE="${4:-bci2token}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_requirements() {
    log "Checking deployment requirements..."
    
    # Check if required tools are installed
    case $DEPLOYMENT_TYPE in
        docker)
            command -v docker >/dev/null 2>&1 || error "Docker is not installed"
            command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || error "kubectl is not installed"
            command -v helm >/dev/null 2>&1 || warn "Helm is not installed (optional)"
            ;;
        systemd)
            systemctl --version >/dev/null 2>&1 || error "systemd is not available"
            ;;
    esac
    
    # Check Python version
    python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || error "Python 3.8+ is required"
    
    success "Requirements check passed"
}

run_quality_checks() {
    log "Running quality checks..."
    
    cd "$PROJECT_ROOT"
    
    # Run quality gates
    if python3 run_quality_checks.py --basic-only; then
        success "Quality checks passed"
    else
        error "Quality checks failed - deployment aborted"
    fi
}

build_application() {
    log "Building application..."
    
    cd "$PROJECT_ROOT"
    
    case $DEPLOYMENT_TYPE in
        docker)
            log "Building Docker image..."
            docker build -f deployment/docker/Dockerfile -t "bci2token:${VERSION}" .
            docker tag "bci2token:${VERSION}" "bci2token:latest"
            success "Docker image built successfully"
            ;;
        *)
            log "Installing Python dependencies..."
            python3 -m pip install -e . -r requirements-prod.txt
            success "Application installed successfully"
            ;;
    esac
}

deploy_docker() {
    log "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT/deployment/docker"
    
    # Create environment file
    cat > .env <<EOF
BCI2TOKEN_VERSION=${VERSION}
BCI2TOKEN_ENVIRONMENT=${ENVIRONMENT}
BCI2TOKEN_LOG_LEVEL=INFO
COMPOSE_PROJECT_NAME=bci2token-${ENVIRONMENT}
EOF
    
    # Deploy services
    docker-compose down --remove-orphans || true
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        success "Docker deployment completed successfully"
    else
        error "Health check failed - deployment may have issues"
    fi
}

deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT/deployment/kubernetes"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configuration
    kubectl apply -n "${NAMESPACE}" -f deployment.yaml
    
    # Wait for rollout
    kubectl rollout status deployment/bci2token-app -n "${NAMESPACE}" --timeout=300s
    
    # Verify deployment
    if kubectl get pods -n "${NAMESPACE}" -l app=bci2token | grep Running >/dev/null; then
        success "Kubernetes deployment completed successfully"
    else
        error "Kubernetes deployment failed"
    fi
}

deploy_systemd() {
    log "Deploying as systemd service..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/bci2token.service > /dev/null <<EOF
[Unit]
Description=BCI-2-Token Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=bci2token
Group=bci2token
WorkingDirectory=${PROJECT_ROOT}
Environment=BCI2TOKEN_ENVIRONMENT=${ENVIRONMENT}
Environment=BCI2TOKEN_LOG_LEVEL=INFO
ExecStart=$(which python3) -m bci2token.cli serve --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bci2token

[Install]
WantedBy=multi-user.target
EOF
    
    # Create user if it doesn't exist
    if ! id "bci2token" &>/dev/null; then
        sudo useradd -r -s /bin/false bci2token
    fi
    
    # Set permissions
    sudo chown -R bci2token:bci2token "${PROJECT_ROOT}"
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable bci2token.service
    sudo systemctl start bci2token.service
    
    # Check status
    if sudo systemctl is-active --quiet bci2token.service; then
        success "Systemd deployment completed successfully"
    else
        error "Systemd service failed to start"
    fi
}

setup_monitoring() {
    log "Setting up monitoring..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Monitoring is included in docker-compose.yml
            log "Monitoring services (Prometheus, Grafana) are included in Docker Compose"
            ;;
        kubernetes)
            # Deploy monitoring stack
            kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/monitoring.yaml" -n "${NAMESPACE}" || warn "Monitoring setup failed"
            ;;
        systemd)
            # Install node_exporter for system monitoring
            log "Consider installing node_exporter for system monitoring"
            ;;
    esac
}

post_deployment_checks() {
    log "Running post-deployment checks..."
    
    # Wait a moment for services to stabilize
    sleep 10
    
    # Basic health check
    case $DEPLOYMENT_TYPE in
        docker)
            if curl -f http://localhost:8080/health >/dev/null 2>&1; then
                success "Health check passed"
            else
                warn "Health check failed - service may need more time to start"
            fi
            ;;
        kubernetes)
            kubectl get pods -n "${NAMESPACE}" -l app=bci2token
            ;;
        systemd)
            sudo systemctl status bci2token.service --no-pager
            ;;
    esac
}

cleanup_old_deployments() {
    log "Cleaning up old deployments..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Remove old images
            docker image prune -f
            docker system prune -f --volumes
            ;;
        kubernetes)
            # Clean up old replica sets
            kubectl delete replicaset -n "${NAMESPACE}" $(kubectl get rs -n "${NAMESPACE}" -o jsonpath='{.items[?(@.status.replicas==0)].metadata.name}') 2>/dev/null || true
            ;;
    esac
}

main() {
    log "Starting BCI-2-Token deployment..."
    log "Environment: ${ENVIRONMENT}"
    log "Deployment type: ${DEPLOYMENT_TYPE}"
    log "Version: ${VERSION}"
    
    check_requirements
    run_quality_checks
    build_application
    
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        systemd)
            deploy_systemd
            ;;
        *)
            error "Unknown deployment type: ${DEPLOYMENT_TYPE}"
            ;;
    esac
    
    setup_monitoring
    post_deployment_checks
    cleanup_old_deployments
    
    success "Deployment completed successfully!"
    
    # Print access information
    case $DEPLOYMENT_TYPE in
        docker)
            log "Application is available at:"
            log "  HTTP:  http://localhost:8080"
            log "  HTTPS: https://localhost:8443"
            log "  Grafana: http://localhost:3000 (admin/admin123)"
            ;;
        kubernetes)
            log "Application is deployed to Kubernetes namespace: ${NAMESPACE}"
            log "Use 'kubectl get services -n ${NAMESPACE}' to see service endpoints"
            ;;
        systemd)
            log "Application is running as systemd service"
            log "  Status: sudo systemctl status bci2token"
            log "  Logs:   sudo journalctl -u bci2token -f"
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [environment] [deployment_type] [version] [namespace]"
    echo ""
    echo "Arguments:"
    echo "  environment      Environment (production, staging, development) [default: production]"
    echo "  deployment_type  Deployment type (docker, kubernetes, systemd) [default: docker]"
    echo "  version         Version tag [default: latest]"
    echo "  namespace       Kubernetes namespace [default: bci2token]"
    echo ""
    echo "Examples:"
    echo "  $0 production docker v1.0.0"
    echo "  $0 staging kubernetes latest staging"
    echo "  $0 development systemd"
    exit 1
fi

main "$@"