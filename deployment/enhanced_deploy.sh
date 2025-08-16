#!/bin/bash

# Enhanced BCI-2-Token Deployment Script
# Supports Generation 1-3 Features with Autonomous Capabilities

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_MODE=${1:-production}
DEPLOYMENT_TYPE=${2:-docker}
VERSION=${3:-latest}

echo -e "${BLUE}🚀 Enhanced BCI-2-Token Deployment${NC}"
echo "=================================="
echo "Mode: $DEPLOYMENT_MODE"
echo "Type: $DEPLOYMENT_TYPE"
echo "Version: $VERSION"
echo ""

# Function to log messages
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $timestamp - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $timestamp - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $timestamp - $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} $timestamp - $message"
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log "INFO" "Prerequisites check passed"
}

# Function to validate configuration
validate_configuration() {
    log "INFO" "Validating configuration..."
    
    # Check environment variables
    required_vars=(
        "DB_PASSWORD"
        "GRAFANA_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log "ERROR" "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate configuration files
    config_files=(
        "config/${DEPLOYMENT_MODE}.json"
        "deployment/docker-compose.yml"
    )
    
    for file in "${config_files[@]}"; do
        if [ ! -f "$file" ]; then
            log "ERROR" "Configuration file $file not found"
            exit 1
        fi
    done
    
    log "INFO" "Configuration validation passed"
}

# Function to run enhanced tests
run_enhanced_tests() {
    log "INFO" "Running enhanced test suite..."
    
    if python3 enhanced_test_suite.py; then
        log "INFO" "Enhanced test suite passed"
    else
        local exit_code=$?
        if [ $exit_code -eq 2 ]; then
            log "WARN" "Test suite completed with warnings"
            read -p "Continue with deployment? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "INFO" "Deployment cancelled by user"
                exit 0
            fi
        else
            log "ERROR" "Enhanced test suite failed"
            exit 1
        fi
    fi
}

# Function to setup monitoring
setup_monitoring() {
    log "INFO" "Setting up monitoring infrastructure..."
    
    # Create monitoring directories
    mkdir -p monitoring/{prometheus,grafana/dashboards,grafana/datasources,logstash/pipeline}
    
    # Generate Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'bci2token'
    static_configs:
      - targets: ['bci2token:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'security-monitor'
    static_configs:
      - targets: ['security_monitor:9091']
    metrics_path: '/security-metrics'
    scrape_interval: 10s
    
  - job_name: 'performance-monitor'
    static_configs:
      - targets: ['performance_monitor:9092']
    metrics_path: '/performance-metrics'
    scrape_interval: 5s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Generate Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Generate security dashboard
    cat > monitoring/grafana/dashboards/security.json << 'EOF'
{
  "dashboard": {
    "title": "BCI-2-Token Security Dashboard",
    "panels": [
      {
        "title": "Threat Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(bci2token_threats_detected_total[5m])"
          }
        ]
      },
      {
        "title": "Security Events",
        "type": "graph",
        "targets": [
          {
            "expr": "bci2token_security_events_total"
          }
        ]
      }
    ]
  }
}
EOF

    log "INFO" "Monitoring setup completed"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    log "INFO" "Deploying with Docker Compose..."
    
    # Export environment variables
    export DEPLOYMENT_MODE
    export VERSION
    
    # Build images
    log "INFO" "Building Docker images..."
    docker-compose -f deployment/docker-compose.yml build
    
    # Start services
    log "INFO" "Starting services..."
    docker-compose -f deployment/docker-compose.yml up -d
    
    # Wait for services to be healthy
    log "INFO" "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    local services=("bci2token" "redis" "postgres" "prometheus" "grafana")
    for service in "${services[@]}"; do
        if docker-compose -f deployment/docker-compose.yml ps "$service" | grep -q "Up"; then
            log "INFO" "Service $service is running"
        else
            log "ERROR" "Service $service failed to start"
            exit 1
        fi
    done
}

# Function to deploy with Kubernetes
deploy_kubernetes() {
    log "INFO" "Deploying with Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log "ERROR" "kubectl is required for Kubernetes deployment"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/
    
    # Wait for deployment
    kubectl rollout status deployment/bci2token
    
    log "INFO" "Kubernetes deployment completed"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    log "INFO" "Running post-deployment tests..."
    
    # Wait for services to be fully ready
    sleep 60
    
    # Test main application endpoint
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        log "INFO" "Main application health check passed"
    else
        log "ERROR" "Main application health check failed"
        exit 1
    fi
    
    # Test monitoring endpoints
    local monitoring_endpoints=(
        "http://localhost:9090"  # Prometheus
        "http://localhost:3000"  # Grafana
        "http://localhost:5601"  # Kibana
    )
    
    for endpoint in "${monitoring_endpoints[@]}"; do
        if curl -f "$endpoint" > /dev/null 2>&1; then
            log "INFO" "Monitoring endpoint $endpoint is accessible"
        else
            log "WARN" "Monitoring endpoint $endpoint is not accessible"
        fi
    done
    
    # Run performance test
    python3 << 'EOF'
import requests
import time
import json

def test_performance():
    base_url = "http://localhost:8080"
    
    # Test basic endpoint
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        print("✅ Health endpoint test passed")
    else:
        print("❌ Health endpoint test failed")
        return False
    
    # Test performance under load
    start_time = time.time()
    success_count = 0
    total_requests = 10
    
    for i in range(total_requests):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                success_count += 1
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    duration = time.time() - start_time
    success_rate = success_count / total_requests
    
    print(f"Performance test results:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average response time: {duration/total_requests:.3f}s")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    if test_performance():
        print("🎉 Post-deployment tests passed")
        exit(0)
    else:
        print("💥 Post-deployment tests failed")
        exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log "INFO" "Post-deployment tests passed"
    else
        log "ERROR" "Post-deployment tests failed"
        exit 1
    fi
}

# Function to display deployment information
display_deployment_info() {
    log "INFO" "Deployment completed successfully!"
    
    echo ""
    echo "🎯 Access Information:"
    echo "====================="
    echo "Main Application:     http://localhost:8080"
    echo "Prometheus:          http://localhost:9090"
    echo "Grafana:             http://localhost:3000"
    echo "Kibana:              http://localhost:5601"
    echo "PostgreSQL:          localhost:5432"
    echo "Redis:               localhost:6379"
    echo ""
    echo "🔐 Default Credentials:"
    echo "======================="
    echo "Grafana Admin:       admin / \$GRAFANA_PASSWORD"
    echo ""
    echo "📊 Monitoring Features:"
    echo "======================"
    echo "✅ Generation 1: Operational Resilience"
    echo "✅ Generation 2: Security Monitoring"
    echo "✅ Generation 3: Performance Optimization"
    echo "✅ Enhanced Threat Detection"
    echo "✅ Adaptive Load Balancing"
    echo "✅ Auto-scaling Capabilities"
    echo ""
    echo "🚀 Next Steps:"
    echo "============="
    echo "1. Configure Grafana dashboards"
    echo "2. Set up alerting rules"
    echo "3. Configure SSL certificates"
    echo "4. Review security logs in Kibana"
    echo ""
}

# Main deployment function
main() {
    log "INFO" "Starting enhanced BCI-2-Token deployment..."
    
    # Pre-deployment checks
    check_prerequisites
    validate_configuration
    
    # Run tests
    run_enhanced_tests
    
    # Setup monitoring
    setup_monitoring
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        "docker")
            deploy_docker_compose
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        *)
            log "ERROR" "Unsupported deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Post-deployment validation
    run_post_deployment_tests
    
    # Display information
    display_deployment_info
}

# Cleanup function for graceful shutdown
cleanup() {
    log "INFO" "Cleaning up deployment..."
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        docker-compose -f deployment/docker-compose.yml down
    elif [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        kubectl delete -f deployment/kubernetes/
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Run main function
main "$@"