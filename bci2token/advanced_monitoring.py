"""
Advanced monitoring and alerting system for BCI-2-Token.

Provides comprehensive system monitoring, alerting, and performance tracking.
"""

import time
import threading
import queue
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque

from .utils import BCIError


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert."""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['level'] = self.level.value
        return data


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a time-series metric."""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.metrics[name].append((timestamp, value))
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'time_series': {}
            }
            
            for name, values in self.metrics.items():
                if values:
                    recent_values = [v for _, v in values]
                    summary['time_series'][name] = {
                        'count': len(recent_values),
                        'latest': recent_values[-1],
                        'average': sum(recent_values) / len(recent_values),
                        'min': min(recent_values),
                        'max': max(recent_values)
                    }
            
            return summary
    
    def get_metric_history(self, name: str, since: Optional[float] = None) -> List[tuple]:
        """Get history for a specific metric."""
        with self.lock:
            values = list(self.metrics[name])
            
            if since is not None:
                values = [(t, v) for t, v in values if t >= since]
                
            return values


class AlertManager:
    """
    Manages system alerts and notifications.
    """
    
    def __init__(self, max_alerts: int = 100):
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers = []
        self.lock = threading.Lock()
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def raise_alert(self, level: AlertLevel, component: str, message: str, 
                   metrics: Optional[Dict[str, Any]] = None):
        """Raise a new alert."""
        alert = Alert(
            timestamp=time.time(),
            level=level,
            component=component,
            message=message,
            metrics=metrics
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def get_recent_alerts(self, since: Optional[float] = None, 
                         level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts with optional filtering."""
        with self.lock:
            alerts = list(self.alerts)
        
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
            
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
            
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts by level and component."""
        with self.lock:
            alerts = list(self.alerts)
        
        summary = {
            'total_alerts': len(alerts),
            'by_level': defaultdict(int),
            'by_component': defaultdict(int),
            'recent_critical': []
        }
        
        # Count by level and component
        for alert in alerts:
            summary['by_level'][alert.level.value] += 1
            summary['by_component'][alert.component] += 1
            
            # Collect recent critical alerts
            if (alert.level == AlertLevel.CRITICAL and 
                time.time() - alert.timestamp < 3600):  # Last hour
                summary['recent_critical'].append(alert.to_dict())
        
        return summary


class PerformanceMonitor:
    """
    Monitors system performance and resource usage.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 alert_manager: AlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 80.0  # %
        self.latency_threshold = 1.0  # seconds
        
    def start_monitoring(self, interval: float = 10.0):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = time.time()
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record_metric('system.cpu_percent', cpu_percent, timestamp)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.record_metric('system.memory_percent', memory.percent, timestamp)
            self.metrics.record_metric('system.memory_available_gb', 
                                     memory.available / (1024**3), timestamp)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.record_metric('system.disk_percent', 
                                     disk.percent, timestamp)
            
        except ImportError:
            # Fallback metrics without psutil
            self.metrics.record_metric('system.timestamp', timestamp, timestamp)
    
    def _check_thresholds(self):
        """Check performance thresholds and raise alerts."""
        # Get recent CPU usage
        cpu_history = self.metrics.get_metric_history('system.cpu_percent', 
                                                     time.time() - 300)  # Last 5 minutes
        if cpu_history:
            avg_cpu = sum(v for _, v in cpu_history) / len(cpu_history)
            if avg_cpu > self.cpu_threshold:
                self.alerts.raise_alert(
                    AlertLevel.WARNING,
                    'performance_monitor',
                    f'High CPU usage: {avg_cpu:.1f}%',
                    {'cpu_percent': avg_cpu}
                )
        
        # Check memory usage
        memory_history = self.metrics.get_metric_history('system.memory_percent',
                                                        time.time() - 60)  # Last minute
        if memory_history:
            latest_memory = memory_history[-1][1]
            if latest_memory > self.memory_threshold:
                self.alerts.raise_alert(
                    AlertLevel.WARNING,
                    'performance_monitor',
                    f'High memory usage: {latest_memory:.1f}%',
                    {'memory_percent': latest_memory}
                )


class SystemMonitor:
    """
    Comprehensive system monitoring coordinator.
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.performance = PerformanceMonitor(self.metrics, self.alerts)
        
        # Component monitors
        self.component_monitors = {}
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        def log_alert(alert: Alert):
            level_map = {
                AlertLevel.INFO: logging.info,
                AlertLevel.WARNING: logging.warning,
                AlertLevel.ERROR: logging.error,
                AlertLevel.CRITICAL: logging.critical
            }
            log_func = level_map.get(alert.level, logging.info)
            log_func(f"[{alert.component}] {alert.message}")
        
        self.alerts.add_alert_handler(log_alert)
    
    def register_component(self, name: str, monitor_func: Callable[[], Dict[str, Any]]):
        """Register a component for monitoring."""
        self.component_monitors[name] = monitor_func
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.performance.start_monitoring()
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.performance.stop_monitoring()
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all registered components."""
        all_metrics = {
            'timestamp': time.time(),
            'system': self.metrics.get_metrics_summary(),
            'alerts': self.alerts.get_alert_summary(),
            'components': {}
        }
        
        # Collect from registered components
        for name, monitor_func in self.component_monitors.items():
            try:
                component_metrics = monitor_func()
                all_metrics['components'][name] = component_metrics
            except Exception as e:
                self.alerts.raise_alert(
                    AlertLevel.ERROR,
                    f'component_{name}',
                    f'Failed to collect metrics: {e}'
                )
        
        return all_metrics
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard data."""
        return {
            'system_health': 'healthy',  # Would be calculated based on thresholds
            'uptime': time.time(),  # Would track actual uptime
            'metrics': self.collect_all_metrics(),
            'recent_alerts': self.alerts.get_recent_alerts(time.time() - 3600),  # Last hour
            'performance_summary': {
                'cpu_ok': True,  # Would check actual values
                'memory_ok': True,
                'disk_ok': True
            }
        }


# Global system monitor instance
_system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return _system_monitor


def record_processing_time(component: str, duration: float):
    """Convenience function to record processing time."""
    _system_monitor.metrics.record_metric(f'{component}.processing_time', duration)


def record_error(component: str, error: str):
    """Convenience function to record an error."""
    _system_monitor.metrics.increment_counter(f'{component}.error_count')
    _system_monitor.alerts.raise_alert(
        AlertLevel.ERROR,
        component,
        f'Error occurred: {error}'
    )