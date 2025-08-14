"""
Health monitoring and diagnostics for BCI-2-Token framework.

Provides system health checks, diagnostics, and automated recovery
for brain-computer interface applications.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class HealthLevel(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    level: HealthLevel
    message: str
    timestamp: float
    metrics: Dict[str, float] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.suggestions is None:
            self.suggestions = []


class SystemDiagnostics:
    """Comprehensive system diagnostics."""
    
    def __init__(self):
        self.checks = {}
        self.auto_recovery_enabled = True
        self.recovery_actions = {}
        
    def add_recovery_action(self, check_name: str, action: Callable):
        """Add automatic recovery action for a specific check."""
        self.recovery_actions[check_name] = action
        
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function."""
        self.checks[name] = check_func
        
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks with auto-recovery."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result
                
                # Attempt auto-recovery for critical issues
                if (self.auto_recovery_enabled and 
                    result.level == HealthLevel.CRITICAL and 
                    name in self.recovery_actions):
                    try:
                        self.recovery_actions[name]()
                        # Re-run check after recovery attempt
                        result = check_func()
                        result.message += " (auto-recovery attempted)"
                        results[name] = result
                    except Exception as recovery_error:
                        result.message += f" (auto-recovery failed: {recovery_error})"
                        
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    level=HealthLevel.CRITICAL,
                    message=f"Health check failed: {e}",
                    timestamp=time.time()
                )
                
        return results
        
    def get_overall_health(self) -> HealthLevel:
        """Get overall system health level."""
        results = self.run_all_checks()
        
        if not results:
            return HealthLevel.UNKNOWN
            
        # Worst status wins
        levels = [result.level for result in results.values()]
        
        if HealthLevel.CRITICAL in levels:
            return HealthLevel.CRITICAL
        elif HealthLevel.WARNING in levels:
            return HealthLevel.WARNING
        elif HealthLevel.HEALTHY in levels:
            return HealthLevel.HEALTHY
        else:
            return HealthLevel.UNKNOWN


def check_dependencies() -> HealthCheck:
    """Check dependency availability."""
    critical_deps = ['numpy', 'scipy']
    optional_deps = ['torch', 'transformers', 'mne']
    
    missing_critical = []
    missing_optional = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_critical.append(dep)
            
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
            
    if missing_critical:
        return HealthCheck(
            name="dependencies",
            level=HealthLevel.CRITICAL,
            message=f"Missing critical dependencies: {missing_critical}",
            timestamp=time.time(),
            suggestions=[f"Install {dep}" for dep in missing_critical]
        )
    elif missing_optional:
        return HealthCheck(
            name="dependencies",
            level=HealthLevel.WARNING,
            message=f"Missing optional dependencies: {missing_optional}",
            timestamp=time.time(),
            suggestions=[f"Install {dep} for full functionality" for dep in missing_optional]
        )
    else:
        return HealthCheck(
            name="dependencies",
            level=HealthLevel.HEALTHY,
            message="All dependencies available",
            timestamp=time.time()
        )


def check_memory_usage() -> HealthCheck:
    """Check system memory usage."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        percent_used = memory.percent
        
        metrics = {
            'memory_percent': percent_used,
            'memory_available_gb': memory.available / (1024**3),
            'memory_total_gb': memory.total / (1024**3)
        }
        
        if percent_used > 90:
            level = HealthLevel.CRITICAL
            message = f"Critical memory usage: {percent_used:.1f}%"
            suggestions = [
                "Close unnecessary applications",
                "Reduce batch size or buffer size",
                "Consider upgrading RAM"
            ]
        elif percent_used > 75:
            level = HealthLevel.WARNING
            message = f"High memory usage: {percent_used:.1f}%"
            suggestions = ["Monitor memory usage", "Consider reducing batch size"]
        else:
            level = HealthLevel.HEALTHY
            message = f"Memory usage normal: {percent_used:.1f}%"
            suggestions = []
            
        return HealthCheck(
            name="memory",
            level=level,
            message=message,
            timestamp=time.time(),
            metrics=metrics,
            suggestions=suggestions
        )
        
    except ImportError:
        return HealthCheck(
            name="memory",
            level=HealthLevel.WARNING,
            message="Cannot check memory usage (psutil not available)",
            timestamp=time.time(),
            suggestions=["Install psutil for memory monitoring"]
        )


def check_disk_space() -> HealthCheck:
    """Check available disk space."""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage('/')
        percent_used = (used / total) * 100
        free_gb = free / (1024**3)
        
        metrics = {
            'disk_percent': percent_used,
            'disk_free_gb': free_gb,
            'disk_total_gb': total / (1024**3)
        }
        
        if percent_used > 95 or free_gb < 1:
            level = HealthLevel.CRITICAL
            message = f"Critical disk usage: {percent_used:.1f}% ({free_gb:.1f}GB free)"
            suggestions = [
                "Free up disk space immediately",
                "Remove old log files and checkpoints",
                "Move data to external storage"
            ]
        elif percent_used > 85 or free_gb < 5:
            level = HealthLevel.WARNING
            message = f"High disk usage: {percent_used:.1f}% ({free_gb:.1f}GB free)"
            suggestions = ["Clean up old files", "Monitor disk usage"]
        else:
            level = HealthLevel.HEALTHY
            message = f"Disk usage normal: {percent_used:.1f}% ({free_gb:.1f}GB free)"
            suggestions = []
            
        return HealthCheck(
            name="disk",
            level=level,
            message=message,
            timestamp=time.time(),
            metrics=metrics,
            suggestions=suggestions
        )
        
    except Exception as e:
        return HealthCheck(
            name="disk",
            level=HealthLevel.WARNING,
            message=f"Cannot check disk usage: {e}",
            timestamp=time.time()
        )


def check_gpu_availability() -> HealthCheck:
    """Check GPU availability for acceleration."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_memory = []
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                gpu_memory.append(total_memory)
                
            metrics = {
                'gpu_count': device_count,
                'total_gpu_memory_gb': sum(gpu_memory)
            }
            
            return HealthCheck(
                name="gpu",
                level=HealthLevel.HEALTHY,
                message=f"{device_count} GPU(s) available with {sum(gpu_memory):.1f}GB total memory",
                timestamp=time.time(),
                metrics=metrics
            )
        else:
            return HealthCheck(
                name="gpu",
                level=HealthLevel.WARNING,
                message="No GPU available (using CPU)",
                timestamp=time.time(),
                suggestions=["Consider GPU for faster processing"]
            )
            
    except ImportError:
        return HealthCheck(
            name="gpu",
            level=HealthLevel.WARNING,
            message="Cannot check GPU (PyTorch not available)",
            timestamp=time.time()
        )


def check_signal_processing_capabilities() -> HealthCheck:
    """Check signal processing capabilities."""
    capabilities = []
    missing = []
    
    # Check core signal processing
    if HAS_NUMPY:
        capabilities.append("NumPy array processing")
    else:
        missing.append("numpy")
        
    try:
        import scipy.signal
        capabilities.append("SciPy filtering")
    except ImportError:
        missing.append("scipy")
        
    try:
        import mne
        capabilities.append("MNE-Python EEG processing")
    except ImportError:
        missing.append("mne")
        
    if missing:
        level = HealthLevel.CRITICAL if 'numpy' in missing else HealthLevel.WARNING
        message = f"Missing signal processing dependencies: {missing}"
        suggestions = [f"Install {dep}" for dep in missing]
    else:
        level = HealthLevel.HEALTHY
        message = f"Signal processing capabilities: {len(capabilities)} available"
        suggestions = []
        
    return HealthCheck(
        name="signal_processing",
        level=level,
        message=message,
        timestamp=time.time(),
        metrics={'capabilities_count': len(capabilities)},
        suggestions=suggestions
    )


def check_model_capabilities() -> HealthCheck:
    """Check machine learning model capabilities."""
    try:
        import torch
        
        # Check if models can be created
        model_features = []
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.mm(x, x.transpose(0, 1))
        model_features.append("Basic tensor operations")
        
        # Test neural network modules
        linear = torch.nn.Linear(10, 5)
        model_features.append("Neural network layers")
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_features.append(f"Device: {device}")
        
        return HealthCheck(
            name="model_capabilities",
            level=HealthLevel.HEALTHY,
            message=f"Model capabilities available: {len(model_features)} features",
            timestamp=time.time(),
            metrics={'features_count': len(model_features)}
        )
        
    except ImportError:
        return HealthCheck(
            name="model_capabilities",
            level=HealthLevel.CRITICAL,
            message="PyTorch not available - no model capabilities",
            timestamp=time.time(),
            suggestions=["Install PyTorch for model functionality"]
        )
    except Exception as e:
        return HealthCheck(
            name="model_capabilities",
            level=HealthLevel.WARNING,
            message=f"Model capability check failed: {e}",
            timestamp=time.time()
        )


def run_comprehensive_diagnostics() -> Dict[str, HealthCheck]:
    """Run comprehensive system diagnostics."""
    diagnostics = SystemDiagnostics()
    
    # Register all health checks
    diagnostics.register_check("dependencies", check_dependencies)
    diagnostics.register_check("memory", check_memory_usage)
    diagnostics.register_check("disk", check_disk_space)
    diagnostics.register_check("gpu", check_gpu_availability)
    diagnostics.register_check("signal_processing", check_signal_processing_capabilities)
    diagnostics.register_check("model_capabilities", check_model_capabilities)
    
    return diagnostics.run_all_checks()


def print_health_report():
    """Print comprehensive health report."""
    print("BCI-2-Token Health Report")
    print("=" * 50)
    
    results = run_comprehensive_diagnostics()
    
    # Overall status
    diagnostics = SystemDiagnostics()
    diagnostics.checks = {name: lambda: result for name, result in results.items()}
    overall = diagnostics.get_overall_health()
    
    status_emoji = {
        HealthLevel.HEALTHY: "🟢",
        HealthLevel.WARNING: "🟡", 
        HealthLevel.CRITICAL: "🔴",
        HealthLevel.UNKNOWN: "⚪"
    }
    
    print(f"Overall Status: {status_emoji[overall]} {overall.value.upper()}")
    print()
    
    # Individual checks
    for name, result in results.items():
        emoji = status_emoji[result.level]
        print(f"{emoji} {name.replace('_', ' ').title()}")
        print(f"   {result.message}")
        
        if result.metrics:
            print("   Metrics:")
            for metric, value in result.metrics.items():
                print(f"     {metric}: {value}")
                
        if result.suggestions:
            print("   Suggestions:")
            for suggestion in result.suggestions:
                print(f"     • {suggestion}")
        print()
        
    # Summary recommendations
    all_suggestions = []
    for result in results.values():
        all_suggestions.extend(result.suggestions)
        
    if all_suggestions:
        print("Priority Actions:")
        for i, suggestion in enumerate(set(all_suggestions), 1):
            print(f"  {i}. {suggestion}")


if __name__ == '__main__':
    print_health_report()