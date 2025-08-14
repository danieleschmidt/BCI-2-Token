"""
Production configuration and deployment settings for BCI-2-Token.

Provides comprehensive production configuration management, environment
setup, and deployment validation.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .security import SecurityConfig
from .preprocessing import PreprocessingConfig
from .auto_scaling import AutoScaler


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "bci2token"
    username: str = "bci_user"
    password: str = ""  # Set via environment variable
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    enable_ssl: bool = True


@dataclass
class CacheConfig:
    """Cache configuration."""
    backend: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    max_size: int = 1000
    ttl: float = 3600.0
    enable_clustering: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    log_level: LogLevel = LogLevel.INFO
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    enable_profiling: bool = False
    alert_webhook_url: str = ""


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_workers: int = 16
    target_cpu_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: float = 300.0


@dataclass
class ProductionConfig:
    """Comprehensive production configuration."""
    
    # Environment
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    testing: bool = False
    
    # Application
    app_name: str = "bci2token"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    
    # Security
    security: SecurityConfig = None
    
    # Processing
    preprocessing: PreprocessingConfig = None
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    
    # Infrastructure
    database: DatabaseConfig = None
    cache: CacheConfig = None
    monitoring: MonitoringConfig = None
    scaling: ScalingConfig = None
    
    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_cores: int = 4
    max_disk_gb: int = 20
    
    # Global settings
    timezone: str = "UTC"
    locale: str = "en_US.UTF-8"
    enable_cors: bool = True
    cors_origins: List[str] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.security is None:
            self.security = self._get_default_security_config()
        
        if self.preprocessing is None:
            self.preprocessing = self._get_default_preprocessing_config()
        
        if self.database is None:
            self.database = DatabaseConfig()
        
        if self.cache is None:
            self.cache = CacheConfig()
        
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        
        if self.scaling is None:
            self.scaling = ScalingConfig()
        
        if self.cors_origins is None:
            self.cors_origins = ["*"] if self.environment != Environment.PRODUCTION else []
    
    def _get_default_security_config(self) -> SecurityConfig:
        """Get default security configuration based on environment."""
        if self.environment == Environment.PRODUCTION:
            return SecurityConfig(
                enable_access_control=True,
                require_privacy_protection=True,
                encrypt_saved_data=True,
                audit_data_access=True,
                max_requests_per_minute=100,
                enable_anomaly_detection=True,
                auto_block_suspicious_ips=True
            )
        elif self.environment == Environment.STAGING:
            return SecurityConfig(
                enable_access_control=True,
                require_privacy_protection=True,
                encrypt_saved_data=False,
                audit_data_access=True,
                max_requests_per_minute=200
            )
        else:  # DEVELOPMENT
            return SecurityConfig(
                enable_access_control=False,
                require_privacy_protection=False,
                encrypt_saved_data=False,
                audit_data_access=False,
                max_requests_per_minute=1000
            )
    
    def _get_default_preprocessing_config(self) -> PreprocessingConfig:
        """Get default preprocessing configuration."""
        if self.environment == Environment.PRODUCTION:
            return PreprocessingConfig(
                apply_ica=True,
                apply_car=True,
                standardize=True,
                window_size=2.0,
                overlap=0.5
            )
        else:
            return PreprocessingConfig(
                apply_ica=False,  # Faster for dev/staging
                apply_car=True,
                standardize=True,
                window_size=1.0,
                overlap=0.25
            )
    
    @classmethod
    def from_environment(cls, env: Environment) -> 'ProductionConfig':
        """Create configuration for specific environment."""
        config = cls(environment=env)
        
        if env == Environment.DEVELOPMENT:
            config.debug = True
            config.workers = 1
            config.monitoring.log_level = LogLevel.DEBUG
            config.cache.max_size = 100
            
        elif env == Environment.STAGING:
            config.debug = False
            config.workers = 2
            config.monitoring.log_level = LogLevel.INFO
            config.cache.max_size = 500
            
        elif env == Environment.PRODUCTION:
            config.debug = False
            config.workers = 4
            config.monitoring.log_level = LogLevel.WARNING
            config.monitoring.enable_profiling = False
            config.cache.max_size = 2000
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ProductionConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionConfig':
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        
        # Handle nested configurations
        nested_configs = ['security', 'preprocessing', 'database', 'cache', 'monitoring', 'scaling']
        for config_name in nested_configs:
            if config_name in config_dict and isinstance(config_dict[config_name], dict):
                config_class = {
                    'security': SecurityConfig,
                    'preprocessing': PreprocessingConfig,
                    'database': DatabaseConfig,
                    'cache': CacheConfig,
                    'monitoring': MonitoringConfig,
                    'scaling': ScalingConfig
                }[config_name]
                
                # Handle enum conversions in nested configs
                if config_name == 'monitoring' and 'log_level' in config_dict[config_name]:
                    config_dict[config_name]['log_level'] = LogLevel(config_dict[config_name]['log_level'])
                
                config_dict[config_name] = config_class(**config_dict[config_name])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        
        # Convert enums to strings
        config_dict['environment'] = self.environment.value
        if 'log_level' in config_dict.get('monitoring', {}):
            config_dict['monitoring']['log_level'] = self.monitoring.log_level.value
        
        return config_dict
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Port validation
        if not (1 <= self.port <= 65535):
            issues.append(f"Invalid port number: {self.port}")
        
        # Resource validation
        if self.max_memory_mb < 512:
            issues.append("Minimum memory requirement is 512MB")
        
        if self.workers < 1:
            issues.append("At least 1 worker is required")
        
        # Environment-specific validation
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                issues.append("Debug mode should be disabled in production")
            
            if not self.security.enable_access_control:
                issues.append("Access control must be enabled in production")
            
            if not self.security.encrypt_saved_data:
                issues.append("Data encryption should be enabled in production")
        
        # Database validation
        if self.database.pool_size < 1:
            issues.append("Database pool size must be at least 1")
        
        # Cache validation
        if self.cache.max_size < 10:
            issues.append("Cache size should be at least 10")
        
        # Scaling validation
        if self.scaling.min_workers > self.scaling.max_workers:
            issues.append("Minimum workers cannot exceed maximum workers")
        
        return issues
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for deployment."""
        env_vars = {
            'BCI2TOKEN_ENVIRONMENT': self.environment.value,
            'BCI2TOKEN_HOST': self.host,
            'BCI2TOKEN_PORT': str(self.port),
            'BCI2TOKEN_WORKERS': str(self.workers),
            'BCI2TOKEN_DEBUG': str(self.debug).lower(),
            'BCI2TOKEN_LOG_LEVEL': self.monitoring.log_level.value,
            'BCI2TOKEN_MAX_MEMORY_MB': str(self.max_memory_mb),
            'BCI2TOKEN_MAX_CPU_CORES': str(self.max_cpu_cores),
        }
        
        # Database environment variables
        env_vars.update({
            'DB_HOST': self.database.host,
            'DB_PORT': str(self.database.port),
            'DB_NAME': self.database.database,
            'DB_USER': self.database.username,
            'DB_POOL_SIZE': str(self.database.pool_size),
        })
        
        # Cache environment variables
        env_vars.update({
            'CACHE_BACKEND': self.cache.backend,
            'CACHE_HOST': self.cache.host,
            'CACHE_PORT': str(self.cache.port),
            'CACHE_MAX_SIZE': str(self.cache.max_size),
        })
        
        # Security environment variables
        env_vars.update({
            'SECURITY_ACCESS_CONTROL': str(self.security.enable_access_control).lower(),
            'SECURITY_PRIVACY_PROTECTION': str(self.security.require_privacy_protection).lower(),
            'SECURITY_DATA_ENCRYPTION': str(self.security.encrypt_saved_data).lower(),
        })
        
        return env_vars


class ConfigurationManager:
    """Manages configuration for different environments and deployments."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.configurations: Dict[Environment, ProductionConfig] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all environments."""
        for env in Environment:
            self.configurations[env] = ProductionConfig.from_environment(env)
    
    def get_config(self, environment: Environment) -> ProductionConfig:
        """Get configuration for specific environment."""
        return self.configurations[environment]
    
    def save_all_configs(self):
        """Save all configurations to files."""
        for env, config in self.configurations.items():
            config_file = self.config_dir / f"{env.value}.json"
            config.save_to_file(str(config_file))
    
    def load_config(self, environment: Environment, config_file: str = None) -> ProductionConfig:
        """Load configuration from file."""
        if config_file is None:
            config_file = self.config_dir / f"{environment.value}.json"
        
        if Path(config_file).exists():
            config = ProductionConfig.from_file(str(config_file))
            self.configurations[environment] = config
            return config
        else:
            return self.configurations[environment]
    
    def validate_all_configs(self) -> Dict[Environment, List[str]]:
        """Validate all configurations."""
        validation_results = {}
        
        for env, config in self.configurations.items():
            issues = config.validate()
            validation_results[env] = issues
        
        return validation_results
    
    def generate_deployment_files(self, environment: Environment, output_dir: str = "deployment"):
        """Generate deployment files for specific environment."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        config = self.configurations[environment]
        
        # Generate Docker Compose file
        self._generate_docker_compose(config, output_path)
        
        # Generate Kubernetes manifests
        self._generate_kubernetes_manifests(config, output_path)
        
        # Generate environment file
        self._generate_env_file(config, output_path)
        
        # Generate systemd service file
        self._generate_systemd_service(config, output_path)
    
    def _generate_docker_compose(self, config: ProductionConfig, output_path: Path):
        """Generate Docker Compose configuration."""
        compose_content = f"""version: '3.8'

services:
  bci2token:
    image: bci2token:{config.version}
    ports:
      - "{config.port}:{config.port}"
    environment:
      - ENVIRONMENT={config.environment.value}
      - DEBUG={str(config.debug).lower()}
      - WORKERS={config.workers}
      - LOG_LEVEL={config.monitoring.log_level.value}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config.port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: {config.max_memory_mb}M
          cpus: '{config.max_cpu_cores}'
        reservations:
          memory: {config.max_memory_mb // 2}M
          cpus: '{config.max_cpu_cores / 2}'

  redis:
    image: redis:alpine
    ports:
      - "{config.cache.port}:{config.cache.port}"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: {config.database.database}
      POSTGRES_USER: {config.database.username}
      POSTGRES_PASSWORD: ${{DB_PASSWORD}}
    ports:
      - "{config.database.port}:{config.database.port}"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
"""
        
        with open(output_path / "docker-compose.yml", 'w') as f:
            f.write(compose_content)
    
    def _generate_kubernetes_manifests(self, config: ProductionConfig, output_path: Path):
        """Generate Kubernetes deployment manifests."""
        k8s_dir = output_path / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        # Deployment manifest
        deployment_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci2token
  labels:
    app: bci2token
spec:
  replicas: {config.workers}
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
        image: bci2token:{config.version}
        ports:
        - containerPort: {config.port}
        env:
        - name: ENVIRONMENT
          value: "{config.environment.value}"
        - name: LOG_LEVEL
          value: "{config.monitoring.log_level.value}"
        resources:
          limits:
            memory: "{config.max_memory_mb}Mi"
            cpu: "{config.max_cpu_cores}"
          requests:
            memory: "{config.max_memory_mb // 2}Mi"
            cpu: "{config.max_cpu_cores / 2}"
        livenessProbe:
          httpGet:
            path: /health
            port: {config.port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {config.port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: bci2token-service
spec:
  selector:
    app: bci2token
  ports:
  - protocol: TCP
    port: 80
    targetPort: {config.port}
  type: LoadBalancer
"""
        
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            f.write(deployment_content)
    
    def _generate_env_file(self, config: ProductionConfig, output_path: Path):
        """Generate environment variables file."""
        env_vars = config.get_environment_variables()
        
        env_content = "# BCI-2-Token Environment Configuration\n"
        for key, value in env_vars.items():
            env_content += f"{key}={value}\n"
        
        # Add sensitive variables as placeholders
        env_content += "\n# Sensitive variables (set these manually)\n"
        env_content += "DB_PASSWORD=your_database_password_here\n"
        env_content += "SECRET_KEY=your_secret_key_here\n"
        env_content += "API_KEY=your_api_key_here\n"
        
        with open(output_path / f".env.{config.environment.value}", 'w') as f:
            f.write(env_content)
    
    def _generate_systemd_service(self, config: ProductionConfig, output_path: Path):
        """Generate systemd service file."""
        service_content = f"""[Unit]
Description=BCI-2-Token Service
After=network.target

[Service]
Type=forking
User=bci2token
Group=bci2token
WorkingDirectory=/opt/bci2token
Environment=ENVIRONMENT={config.environment.value}
Environment=LOG_LEVEL={config.monitoring.log_level.value}
ExecStart=/opt/bci2token/venv/bin/python -m bci2token.cli serve --host {config.host} --port {config.port}
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        with open(output_path / "bci2token.service", 'w') as f:
            f.write(service_content)


def create_production_deployment(environment: Environment = Environment.PRODUCTION):
    """Create complete production deployment configuration."""
    print(f"🚀 Creating {environment.value} deployment configuration...")
    
    manager = ConfigurationManager()
    config = manager.get_config(environment)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"  • {issue}")
        print()
    
    # Save configuration
    manager.save_all_configs()
    
    # Generate deployment files
    manager.generate_deployment_files(environment)
    
    print(f"✅ {environment.value} deployment configuration created!")
    print(f"   Configuration: config/{environment.value}.json")
    print(f"   Deployment files: deployment/")
    print(f"   Environment file: deployment/.env.{environment.value}")
    
    return config


if __name__ == '__main__':
    # Create production deployment configuration
    config = create_production_deployment(Environment.PRODUCTION)
    
    print(f"\n📋 Configuration Summary:")
    print(f"Environment: {config.environment.value}")
    print(f"Host: {config.host}:{config.port}")
    print(f"Workers: {config.workers}")
    print(f"Security: {'Enabled' if config.security.enable_access_control else 'Disabled'}")
    print(f"Monitoring: {'Enabled' if config.monitoring.enable_metrics else 'Disabled'}")
    print(f"Auto-scaling: {'Enabled' if config.scaling.enable_auto_scaling else 'Disabled'}")
    
    print(f"\n🔧 Next steps:")
    print(f"1. Review and customize config/production.json")
    print(f"2. Set sensitive environment variables in deployment/.env.production")
    print(f"3. Deploy using docker-compose up -d or kubectl apply -f deployment/kubernetes/")
    print(f"4. Monitor deployment health at http://{config.host}:{config.port}/health")