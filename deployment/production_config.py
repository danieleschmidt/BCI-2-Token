"""
Production configuration for BCI-2-Token framework.

Defines secure, optimized configurations for production deployment.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Security
    enable_security: bool = True
    session_timeout: float = 3600.0  # 1 hour
    max_concurrent_sessions: int = 100
    require_https: bool = True
    enable_cors: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    
    # Privacy
    require_privacy_protection: bool = True
    min_privacy_epsilon: float = 0.5
    max_privacy_epsilon: float = 5.0
    audit_privacy_usage: bool = True
    
    # Performance
    enable_caching: bool = True
    cache_size: int = 10000
    cache_ttl: float = 3600.0  # 1 hour
    max_worker_threads: int = 8
    max_worker_processes: int = 4
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_instances: int = 2
    max_instances: int = 20
    
    # Rate limiting
    max_requests_per_minute: int = 1000
    max_decode_operations_per_hour: int = 10000
    
    # Data protection
    encrypt_saved_data: bool = True
    secure_delete: bool = True
    audit_data_access: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_retention_days: int = 30
    health_check_interval: float = 30.0
    
    # Reliability
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_self_healing: bool = True
    
    # Storage
    data_directory: Path = field(default_factory=lambda: Path("/var/lib/bci2token"))
    log_directory: Path = field(default_factory=lambda: Path("/var/log/bci2token"))
    cache_directory: Path = field(default_factory=lambda: Path("/var/cache/bci2token"))
    
    # Database (if used)
    database_url: Optional[str] = None
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # External services
    model_repository_url: str = "https://models.bci2token.com"
    telemetry_endpoint: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'ProductionConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        config.debug = os.getenv('BCI2TOKEN_DEBUG', 'false').lower() == 'true'
        config.log_level = os.getenv('BCI2TOKEN_LOG_LEVEL', 'INFO')
        
        # Security settings
        config.session_timeout = float(os.getenv('BCI2TOKEN_SESSION_TIMEOUT', '3600'))
        config.max_concurrent_sessions = int(os.getenv('BCI2TOKEN_MAX_SESSIONS', '100'))
        config.require_https = os.getenv('BCI2TOKEN_REQUIRE_HTTPS', 'true').lower() == 'true'
        
        # Performance settings
        config.cache_size = int(os.getenv('BCI2TOKEN_CACHE_SIZE', '10000'))
        config.max_worker_threads = int(os.getenv('BCI2TOKEN_MAX_THREADS', '8'))
        config.max_worker_processes = int(os.getenv('BCI2TOKEN_MAX_PROCESSES', '4'))
        
        # Auto-scaling
        config.min_instances = int(os.getenv('BCI2TOKEN_MIN_INSTANCES', '2'))
        config.max_instances = int(os.getenv('BCI2TOKEN_MAX_INSTANCES', '20'))
        
        # Directories
        if data_dir := os.getenv('BCI2TOKEN_DATA_DIR'):
            config.data_directory = Path(data_dir)
        if log_dir := os.getenv('BCI2TOKEN_LOG_DIR'):
            config.log_directory = Path(log_dir)
        if cache_dir := os.getenv('BCI2TOKEN_CACHE_DIR'):
            config.cache_directory = Path(cache_dir)
            
        # Database
        config.database_url = os.getenv('BCI2TOKEN_DATABASE_URL')
        
        # External services
        if model_repo := os.getenv('BCI2TOKEN_MODEL_REPOSITORY'):
            config.model_repository_url = model_repo
        config.telemetry_endpoint = os.getenv('BCI2TOKEN_TELEMETRY_ENDPOINT')
        
        return config
        
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check required directories exist or can be created
        for dir_name, directory in [
            ('data', self.data_directory),
            ('log', self.log_directory),
            ('cache', self.cache_directory)
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    issues.append(f"{dir_name} directory is not writable: {directory}")
            except Exception as e:
                issues.append(f"Cannot create {dir_name} directory {directory}: {e}")
                
        # Validate numeric ranges
        if self.session_timeout <= 0:
            issues.append("session_timeout must be positive")
        if self.max_concurrent_sessions <= 0:
            issues.append("max_concurrent_sessions must be positive")
        if self.cache_size <= 0:
            issues.append("cache_size must be positive")
        if self.min_instances > self.max_instances:
            issues.append("min_instances cannot exceed max_instances")
            
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            issues.append(f"log_level must be one of {valid_levels}")
            
        return issues
        
    def setup_logging(self):
        """Setup production logging configuration."""
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_directory / 'bci2token.log'),
                logging.StreamHandler() if self.debug else logging.NullHandler()
            ]
        )
        
        # Set up log rotation
        try:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.log_directory / 'bci2token.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            
            # Replace file handler
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)
            root_logger.addHandler(file_handler)
            
        except ImportError:
            # Fallback to basic file handler
            pass
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            # Convert Path objects to strings
            if isinstance(value, Path):
                config_dict[field_name] = str(value)
            elif isinstance(value, list):
                config_dict[field_name] = list(value)
            else:
                config_dict[field_name] = value
                
        return config_dict


@dataclass
class DeploymentManifest:
    """Deployment manifest with system requirements."""
    
    # System requirements
    min_python_version: str = "3.8"
    required_packages: List[str] = field(default_factory=lambda: [
        "numpy>=1.20.0",
        "scipy>=1.7.0"
    ])
    optional_packages: List[str] = field(default_factory=lambda: [
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "mne>=0.24.0",
        "cryptography>=3.0.0",
        "pyserial>=3.5"
    ])
    
    # System resources
    min_memory_gb: float = 4.0
    recommended_memory_gb: float = 16.0
    min_disk_gb: float = 10.0
    recommended_disk_gb: float = 100.0
    min_cpu_cores: int = 2
    recommended_cpu_cores: int = 8
    
    # Network requirements
    required_ports: List[int] = field(default_factory=lambda: [8080, 8443])
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "web_interface": True,
        "api_server": True,
        "real_time_processing": True,
        "batch_processing": True,
        "model_training": False,  # Disabled in production by default
        "telemetry": True
    })
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check if system meets requirements."""
        import sys
        import psutil
        import shutil
        
        results = {
            'python_version': {
                'required': self.min_python_version,
                'current': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'satisfied': sys.version_info >= tuple(map(int, self.min_python_version.split('.')))
            },
            'memory': {
                'required_gb': self.min_memory_gb,
                'recommended_gb': self.recommended_memory_gb,
                'available_gb': psutil.virtual_memory().total / (1024**3),
                'satisfied': psutil.virtual_memory().total / (1024**3) >= self.min_memory_gb
            },
            'disk': {
                'required_gb': self.min_disk_gb,
                'recommended_gb': self.recommended_disk_gb,
                'available_gb': shutil.disk_usage('/').free / (1024**3),
                'satisfied': shutil.disk_usage('/').free / (1024**3) >= self.min_disk_gb
            },
            'cpu': {
                'required_cores': self.min_cpu_cores,
                'recommended_cores': self.recommended_cpu_cores,
                'available_cores': psutil.cpu_count(),
                'satisfied': psutil.cpu_count() >= self.min_cpu_cores
            }
        }
        
        return results


def create_production_config() -> ProductionConfig:
    """Create and validate production configuration."""
    config = ProductionConfig.from_environment()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        raise ValueError(f"Configuration validation failed: {issues}")
        
    # Setup logging
    config.setup_logging()
    
    return config


def create_deployment_manifest() -> DeploymentManifest:
    """Create deployment manifest."""
    return DeploymentManifest()


if __name__ == '__main__':
    # Test configuration creation
    print("Testing production configuration...")
    
    try:
        config = create_production_config()
        print("✓ Production configuration created successfully")
        print(f"  Environment: {config.environment}")
        print(f"  Security enabled: {config.enable_security}")
        print(f"  Cache size: {config.cache_size}")
        print(f"  Data directory: {config.data_directory}")
        
        # Test deployment manifest
        manifest = create_deployment_manifest()
        requirements = manifest.check_system_requirements()
        
        print("\n✓ System requirements check:")
        for component, details in requirements.items():
            status = "✓" if details['satisfied'] else "✗"
            print(f"  {status} {component}: {details}")
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        sys.exit(1)