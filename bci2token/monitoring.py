"""
Monitoring and logging system for BCI-2-Token framework.

Provides comprehensive logging, metrics collection, and health monitoring
for production brain-computer interface applications.
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: float
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class HealthStatus:
    """System health status."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: float
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric point."""
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.metrics[name].append(point)
            
    def get_recent_metrics(self, name: str, duration: float = 60.0) -> List[MetricPoint]:
        """Get metrics from the last N seconds."""
        with self.lock:
            if name not in self.metrics:
                return []
                
            cutoff_time = time.time() - duration
            return [
                point for point in self.metrics[name] 
                if point.timestamp > cutoff_time
            ]
            
    def get_metric_summary(self, name: str, duration: float = 60.0) -> Dict[str, float]:
        """Get statistical summary of recent metrics."""
        recent_points = self.get_recent_metrics(name, duration)
        
        if not recent_points:
            return {'count': 0}
            
        values = [point.value for point in recent_points]
        
        if HAS_NUMPY:
            return {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
        else:
            # Fallback without numpy
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
            
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics as serializable data."""
        with self.lock:
            result = {}
            for name, points in self.metrics.items():
                result[name] = [asdict(point) for point in points]
            return result


class PerformanceMonitor:
    """Monitors system performance and health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.last_health_check = 0.0
        self.health_check_interval = 30.0  # seconds
        
    def add_health_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Add a health check function."""
        self.health_checks[name] = check_func
        
    def set_alert_threshold(self, metric_name: str, thresholds: Dict[str, float]):
        """Set alert thresholds for a metric."""
        self.alert_thresholds[metric_name] = thresholds
        
    def check_health(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                status = check_func()
                results[name] = status
            except Exception as e:
                results[name] = HealthStatus(
                    component=name,
                    status='critical',
                    message=f"Health check failed: {e}",
                    timestamp=time.time()
                )
                
        self.last_health_check = time.time()
        return results
        
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for metric threshold violations."""
        alerts = []
        
        for metric_name, thresholds in self.alert_thresholds.items():
            summary = self.metrics.get_metric_summary(metric_name, duration=60.0)
            
            if summary['count'] == 0:
                continue
                
            for threshold_type, threshold_value in thresholds.items():
                current_value = summary.get(threshold_type)
                
                if current_value is None:
                    continue
                    
                if threshold_type in ['max', 'p95', 'p99', 'mean']:
                    if current_value > threshold_value:
                        alerts.append({
                            'metric': metric_name,
                            'threshold_type': threshold_type,
                            'threshold_value': threshold_value,
                            'current_value': current_value,
                            'severity': 'warning' if current_value < threshold_value * 1.5 else 'critical',
                            'timestamp': time.time()
                        })
                        
        return alerts


class BCILogger:
    """Specialized logger for BCI applications."""
    
    def __init__(self, 
                 log_file: Optional[Path] = None,
                 console_level: str = 'INFO',
                 file_level: str = 'DEBUG'):
        """
        Initialize BCI logger.
        
        Args:
            log_file: Optional file path for logging
            console_level: Console logging level
            file_level: File logging level
        """
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level
        
        # Log levels
        self.levels = {
            'DEBUG': 0,
            'INFO': 1, 
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
        
        self.lock = threading.Lock()
        
    def _should_log(self, level: str, target: str) -> bool:
        """Check if message should be logged."""
        if target == 'console':
            return self.levels[level] >= self.levels[self.console_level]
        else:  # file
            return self.levels[level] >= self.levels[self.file_level]
            
    def _format_message(self, level: str, component: str, message: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format log message."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"[{timestamp}] {level:8} | {component:15} | {message}"
        
        if metadata:
            metadata_str = json.dumps(metadata, default=str, separators=(',', ':'))
            log_entry += f" | {metadata_str}"
            
        return log_entry
        
    def log(self, level: str, component: str, message: str, 
            metadata: Optional[Dict[str, Any]] = None):
        """Log a message."""
        formatted_message = self._format_message(level, component, message, metadata)
        
        with self.lock:
            # Console logging
            if self._should_log(level, 'console'):
                if level in ['ERROR', 'CRITICAL']:
                    print(formatted_message, file=sys.stderr)
                else:
                    print(formatted_message)
                    
            # File logging
            if self.log_file and self._should_log(level, 'file'):
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(formatted_message + '\n')
                except Exception as e:
                    print(f"Failed to write to log file: {e}", file=sys.stderr)
                    
    def debug(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.log('DEBUG', component, message, metadata)
        
    def info(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.log('INFO', component, message, metadata)
        
    def warning(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.log('WARNING', component, message, metadata)
        
    def error(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.log('ERROR', component, message, metadata)
        
    def critical(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        self.log('CRITICAL', component, message, metadata)


class BCIMonitor:
    """Main monitoring system for BCI applications."""
    
    def __init__(self, 
                 log_file: Optional[Path] = None,
                 metrics_enabled: bool = True):
        """
        Initialize BCI monitor.
        
        Args:
            log_file: Optional log file path
            metrics_enabled: Whether to collect metrics
        """
        self.logger = BCILogger(log_file)
        self.metrics = MetricsCollector() if metrics_enabled else None
        self.performance = PerformanceMonitor(self.metrics) if metrics_enabled else None
        
        # Monitoring state
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Setup default health checks
        if self.performance:
            self._setup_default_health_checks()
            self._setup_default_thresholds()
            
    def _setup_default_health_checks(self):
        """Setup default health check functions."""
        def memory_check():
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent > 90:
                    status = 'critical'
                    message = f"High memory usage: {memory_percent:.1f}%"
                elif memory_percent > 75:
                    status = 'warning'
                    message = f"Elevated memory usage: {memory_percent:.1f}%"
                else:
                    status = 'healthy'
                    message = f"Memory usage normal: {memory_percent:.1f}%"
                    
                return HealthStatus(
                    component='memory',
                    status=status,
                    message=message,
                    timestamp=time.time(),
                    metrics={'memory_percent': memory_percent}
                )
            except ImportError:
                return HealthStatus(
                    component='memory',
                    status='warning',
                    message='psutil not available for memory monitoring',
                    timestamp=time.time()
                )
                
        def disk_check():
            try:
                import shutil
                total, used, free = shutil.disk_usage('/')
                usage_percent = (used / total) * 100
                
                if usage_percent > 95:
                    status = 'critical'
                    message = f"Critical disk usage: {usage_percent:.1f}%"
                elif usage_percent > 85:
                    status = 'warning'
                    message = f"High disk usage: {usage_percent:.1f}%"
                else:
                    status = 'healthy'
                    message = f"Disk usage normal: {usage_percent:.1f}%"
                    
                return HealthStatus(
                    component='disk',
                    status=status,
                    message=message,
                    timestamp=time.time(),
                    metrics={'disk_usage_percent': usage_percent}
                )
            except Exception as e:
                return HealthStatus(
                    component='disk',
                    status='warning',
                    message=f'Disk check failed: {e}',
                    timestamp=time.time()
                )
                
        # self.performance.add_health_check('memory', memory_check)
        # self.performance.add_health_check('disk', disk_check)
        
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        thresholds = {
            'decoding_latency': {'p95': 0.2, 'max': 0.5},  # 200ms p95, 500ms max
            'confidence_score': {'mean': 0.5, 'p95': 0.3},  # Mean > 0.5, p95 > 0.3
            'signal_quality': {'min': 0.6},  # Min signal quality > 0.6
            'processing_errors': {'count': 10}  # Max 10 errors per minute
        }
        
        for metric, threshold in thresholds.items():
            if self.performance:
                self.performance.set_alert_threshold(metric, threshold)
                
    def start_monitoring(self):
        """Start background monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info('Monitor', 'Background monitoring started')
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            
        self.logger.info('Monitor', 'Background monitoring stopped')
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run health checks
                if self.performance:
                    health_results = self.performance.check_health()
                    
                    for component, status in health_results.items():
                        if status.status != 'healthy':
                            self.logger.warning(
                                f'Health-{component}',
                                status.message,
                                {'status': status.status, 'metrics': status.metrics}
                            )
                            
                    # Check for alerts
                    alerts = self.performance.check_alerts()
                    for alert in alerts:
                        self.logger.warning(
                            'Alert',
                            f"Threshold violation: {alert['metric']} {alert['threshold_type']} = {alert['current_value']:.3f} > {alert['threshold_value']:.3f}",
                            alert
                        )
                        
                # Sleep before next check
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error('Monitor', f'Monitoring loop error: {e}')
                time.sleep(5.0)  # Short sleep on error
                
    def log_decoding_performance(self, 
                                latency: float, 
                                confidence: float,
                                tokens_decoded: int,
                                signal_quality: float = None):
        """Log brain decoding performance metrics."""
        if self.metrics:
            self.metrics.record_metric('decoding_latency', latency)
            self.metrics.record_metric('confidence_score', confidence)
            self.metrics.record_metric('tokens_per_decode', tokens_decoded)
            
            if signal_quality is not None:
                self.metrics.record_metric('signal_quality', signal_quality)
                
        self.logger.debug(
            'Decoder',
            f'Decoded {tokens_decoded} tokens',
            {
                'latency_ms': latency * 1000,
                'confidence': confidence,
                'signal_quality': signal_quality
            }
        )
        
    def log_streaming_performance(self,
                                 buffer_utilization: float,
                                 processing_rate: float,
                                 dropped_samples: int = 0):
        """Log streaming performance metrics."""
        if self.metrics:
            self.metrics.record_metric('buffer_utilization', buffer_utilization)
            self.metrics.record_metric('processing_rate', processing_rate)
            
            if dropped_samples > 0:
                self.metrics.record_metric('dropped_samples', dropped_samples)
                
        self.logger.debug(
            'Streaming',
            f'Buffer: {buffer_utilization:.1%}, Rate: {processing_rate:.1f} Hz',
            {
                'buffer_utilization': buffer_utilization,
                'processing_rate': processing_rate,
                'dropped_samples': dropped_samples
            }
        )
        
    def log_device_status(self,
                         device_name: str,
                         is_connected: bool,
                         signal_strength: float = None,
                         impedance_values: List[float] = None):
        """Log device status and signal quality."""
        if self.metrics:
            self.metrics.record_metric(
                'device_connected',
                1.0 if is_connected else 0.0,
                tags={'device': device_name}
            )
            
            if signal_strength is not None:
                self.metrics.record_metric(
                    'signal_strength',
                    signal_strength,
                    tags={'device': device_name}
                )
                
        metadata = {'device': device_name, 'connected': is_connected}
        if signal_strength is not None:
            metadata['signal_strength'] = signal_strength
        if impedance_values:
            metadata['impedance_mean'] = sum(impedance_values) / len(impedance_values)
            
        self.logger.info(
            f'Device-{device_name}',
            f"Status: {'Connected' if is_connected else 'Disconnected'}",
            metadata
        )
        
    def log_privacy_event(self,
                         epsilon_used: float,
                         noise_level: float,
                         signal_degradation: float = None):
        """Log differential privacy events."""
        if self.metrics:
            self.metrics.record_metric('privacy_epsilon_used', epsilon_used)
            self.metrics.record_metric('privacy_noise_level', noise_level)
            
            if signal_degradation is not None:
                self.metrics.record_metric('privacy_signal_degradation', signal_degradation)
                
        self.logger.debug(
            'Privacy',
            f'Applied DP noise: ε={epsilon_used:.3f}',
            {
                'epsilon': epsilon_used,
                'noise_level': noise_level,
                'signal_degradation': signal_degradation
            }
        )
        
    def log_error(self, component: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context."""
        if self.metrics:
            self.metrics.record_metric(
                'processing_errors',
                1.0,
                tags={'component': component, 'error_type': type(error).__name__}
            )
            
        metadata = {'error_type': type(error).__name__}
        if context:
            metadata.update(context)
            
        self.logger.error(
            component,
            f'{type(error).__name__}: {str(error)}',
            metadata
        )
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'monitoring_active': self.is_monitoring,
            'last_health_check': self.last_health_check
        }
        
        # Add health status
        if self.performance:
            health_results = self.performance.check_health()
            status['health'] = {
                name: asdict(health) for name, health in health_results.items()
            }
            
            # Add recent alerts
            alerts = self.performance.check_alerts()
            status['alerts'] = alerts
            
        # Add recent metrics summary
        if self.metrics:
            key_metrics = ['decoding_latency', 'confidence_score', 'buffer_utilization']
            status['metrics'] = {}
            
            for metric in key_metrics:
                summary = self.metrics.get_metric_summary(metric, duration=300.0)  # 5 minutes
                if summary['count'] > 0:
                    status['metrics'][metric] = summary
                    
        return status
        
    def export_metrics(self, output_file: Path, duration: float = 3600.0):
        """Export metrics to file."""
        if not self.metrics:
            raise RuntimeError("Metrics collection not enabled")
            
        # Get all metrics from the specified duration
        all_metrics = {}
        
        for metric_name in self.metrics.metrics.keys():
            recent_points = self.metrics.get_recent_metrics(metric_name, duration)
            all_metrics[metric_name] = [asdict(point) for point in recent_points]
            
        # Save to file
        export_data = {
            'export_timestamp': time.time(),
            'duration_seconds': duration,
            'metrics': all_metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        self.logger.info(
            'Monitor',
            f'Metrics exported to {output_file}',
            {'file': str(output_file), 'duration': duration}
        )


# Global monitor instance
_global_monitor: Optional[BCIMonitor] = None


def get_monitor() -> BCIMonitor:
    """Get or create global monitor instance."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = BCIMonitor()
        
    return _global_monitor


def initialize_monitoring(log_file: Optional[Path] = None,
                         enable_background: bool = True) -> BCIMonitor:
    """Initialize global monitoring system."""
    global _global_monitor
    
    _global_monitor = BCIMonitor(log_file)
    
    if enable_background:
        _global_monitor.start_monitoring()
        
    return _global_monitor


# Convenience functions for common logging
def log_info(component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
    """Log info message to global monitor."""
    get_monitor().logger.info(component, message, metadata)

def log_error(component: str, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log error to global monitor."""
    get_monitor().log_error(component, error, context)

def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record metric to global monitor."""
    monitor = get_monitor()
    if monitor.metrics:
        monitor.metrics.record_metric(name, value, tags)