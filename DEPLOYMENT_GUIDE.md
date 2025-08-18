# BCI-2-Token Production Deployment Guide

## Overview
This guide covers production deployment of the BCI-2-Token framework with enterprise-grade security, monitoring, and scalability.

## Quick Deployment

### Method 1: Docker (Recommended)
```bash
# Basic deployment
./deployment/scripts/deploy.sh production docker v1.0.0

# With custom configuration
./deployment/scripts/deploy.sh staging docker latest bci2token-staging
```

### Method 2: Kubernetes (Enterprise)
```bash
# Deploy to Kubernetes cluster
./deployment/scripts/deploy.sh production kubernetes v1.0.0 bci2token-prod

# Verify deployment
kubectl get pods -n bci2token-prod
```

### Method 3: SystemD (Simple)
```bash
# Install as system service
./deployment/scripts/deploy.sh production systemd

# Monitor service
sudo systemctl status bci2token
sudo journalctl -u bci2token -f
```

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8+ with pip
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 20GB available space
- **Network**: HTTP/HTTPS ports (80/443) for API access

### Dependencies
```bash
# Core dependencies
sudo apt update
sudo apt install python3 python3-pip git curl

# Docker deployment
sudo apt install docker.io docker-compose

# Kubernetes deployment  
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## Configuration

### Environment Variables
```bash
# Core settings
export BCI2TOKEN_ENVIRONMENT=production
export BCI2TOKEN_LOG_LEVEL=INFO
export BCI2TOKEN_HOST=0.0.0.0
export BCI2TOKEN_PORT=8080

# Security settings
export BCI2TOKEN_SECRET_KEY="your-secure-secret-key"
export BCI2TOKEN_ENABLE_AUTH=true
export BCI2TOKEN_RATE_LIMIT=100

# Performance settings
export BCI2TOKEN_MAX_WORKERS=4
export BCI2TOKEN_CACHE_SIZE=1000
export BCI2TOKEN_ENABLE_MONITORING=true
```

### Production Configuration File
Create `/etc/bci2token/config.json`:
```json
{
  "environment": "production",
  "security": {
    "enable_access_control": true,
    "session_timeout": 3600,
    "max_concurrent_sessions": 100,
    "enable_rate_limiting": true,
    "max_requests_per_minute": 60
  },
  "performance": {
    "enable_signal_cache": true,
    "cache_size": 2000,
    "max_worker_threads": 8,
    "enable_auto_scaling": true
  },
  "monitoring": {
    "enable_metrics": true,
    "log_level": "INFO",
    "health_check_interval": 30
  }
}
```

## Deployment Options

### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'
services:
  bci2token:
    image: bci2token:latest
    ports:
      - "8080:8080"
      - "8443:8443"
    environment:
      - BCI2TOKEN_ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
```

### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci2token-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bci2token
  template:
    metadata:
      labels:
        app: bci2token
    spec:
      containers:
      - name: bci2token
        image: bci2token:latest
        ports:
        - containerPort: 8080
        env:
        - name: BCI2TOKEN_ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: bci2token-service
spec:
  selector:
    app: bci2token
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure in deployment
export BCI2TOKEN_SSL_CERT=/path/to/cert.pem
export BCI2TOKEN_SSL_KEY=/path/to/key.pem
```

### Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # BCI-2-Token API
```

### Access Control
```bash
# Create dedicated user
sudo useradd -r -s /bin/false bci2token
sudo mkdir -p /opt/bci2token/{data,logs}
sudo chown -R bci2token:bci2token /opt/bci2token
```

## Monitoring & Observability

### Health Checks
```bash
# API health check
curl -f http://localhost:8080/health

# Detailed system status
curl http://localhost:8080/status
```

### Metrics Collection
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting
- **Built-in monitoring**: Real-time performance metrics

### Log Management
```bash
# View application logs
sudo journalctl -u bci2token -f

# Docker logs
docker-compose logs -f bci2token

# Kubernetes logs
kubectl logs -f deployment/bci2token-app -n bci2token-prod
```

## Performance Tuning

### Auto-Scaling Configuration
```python
# Auto-scaling settings
config = {
    "auto_scaling": {
        "min_workers": 2,
        "max_workers": 16,
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.3,
        "cooldown_period": 300
    }
}
```

### Memory Optimization
```bash
# Optimize system memory
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=5' >> /etc/sysctl.conf
sysctl -p
```

### Performance Monitoring
- Monitor CPU, memory, and network usage
- Track request latency and throughput
- Set up alerting for performance degradation

## Backup & Recovery

### Data Backup
```bash
# Backup configuration and data
tar -czf bci2token-backup-$(date +%Y%m%d).tar.gz \
    /etc/bci2token \
    /opt/bci2token/data \
    /opt/bci2token/logs
```

### Database Backup (if applicable)
```bash
# PostgreSQL backup example
pg_dump bci2token > bci2token-db-backup-$(date +%Y%m%d).sql
```

### Recovery Procedures
1. Stop BCI-2-Token service
2. Restore configuration and data files
3. Restart service and verify functionality
4. Check logs for any issues

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
sudo journalctl -u bci2token --no-pager
# Check configuration
python3 -m bci2token.cli test --config
```

#### High Memory Usage
```bash
# Monitor memory
free -h
ps aux | grep bci2token
# Tune garbage collection
export PYTHONOPTIMIZE=2
```

#### Performance Issues
```bash
# Check system resources
top -p $(pgrep -f bci2token)
# Review performance metrics
curl http://localhost:8080/metrics
```

### Debug Mode
```bash
# Enable debug logging
export BCI2TOKEN_LOG_LEVEL=DEBUG
sudo systemctl restart bci2token
```

## Production Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Configuration files prepared
- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Backup procedures tested

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Logs being collected
- [ ] Performance metrics normal
- [ ] Security scanning completed
- [ ] Documentation updated

### Ongoing Maintenance
- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Log rotation configured
- [ ] Backup verification
- [ ] Capacity planning reviews

## Support & Documentation

### Resources
- **API Documentation**: http://localhost:8080/docs
- **Health Dashboard**: http://localhost:8080/health
- **Metrics Endpoint**: http://localhost:8080/metrics
- **Configuration Reference**: `/etc/bci2token/config.json`

### Getting Help
1. Check logs for error messages
2. Review configuration settings
3. Consult troubleshooting guide
4. Contact support team

## Security Considerations

### Production Security Checklist
- [ ] Strong authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation active
- [ ] Audit logging enabled
- [ ] Regular security updates
- [ ] Network segmentation
- [ ] Data encryption at rest
- [ ] Secure communication (TLS)

### Compliance
The BCI-2-Token framework supports:
- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data security
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management

---

For additional support and advanced configuration options, consult the technical documentation or contact the development team.