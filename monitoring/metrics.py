"""
Comprehensive Monitoring and Observability for BCI-2-Token

This module provides detailed metrics collection, performance monitoring,
and observability tools for the BCI system in production.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import logging
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Container for metric values with metadata"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    inference_latency_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    model_forward_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    throughput_tokens_per_second: float = 0.0
    accuracy_score: float = 0.0
    confidence_score: float = 0.0


class BCIMetricsCollector:
    """Main metrics collector for BCI-2-Token system"""
    
    def __init__(
        self, 
        enable_prometheus: bool = True, 
        enable_opentelemetry: bool = True,
        metrics_port: int = 9090
    ):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_opentelemetry = enable_opentelemetry and OPENTELEMETRY_AVAILABLE
        self.metrics_port = metrics_port
        
        # Internal metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self.active_requests = 0
        self.total_requests = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize monitoring backends
        self._setup_prometheus()
        self._setup_opentelemetry()
        
        # Start background monitoring
        self._start_system_monitoring()
        
        logger.info("BCI Metrics Collector initialized")
    
    def _setup_prometheus(self) -> None:
        """Setup Prometheus metrics"""
        if not self.enable_prometheus:
            return
        
        # Define Prometheus metrics
        self.prom_inference_duration = Histogram(
            'bci_inference_duration_seconds',
            'Time spent on neural signal inference',
            ['model_type', 'signal_type']
        )
        
        self.prom_preprocessing_duration = Histogram(
            'bci_preprocessing_duration_seconds',
            'Time spent on signal preprocessing',
            ['signal_type', 'channels']
        )
        
        self.prom_requests_total = Counter(
            'bci_requests_total',
            'Total number of BCI requests',
            ['endpoint', 'status']
        )
        
        self.prom_active_requests = Gauge(
            'bci_active_requests',
            'Number of active BCI requests'
        )
        
        self.prom_model_accuracy = Gauge(
            'bci_model_accuracy',
            'Current model accuracy score',
            ['model_type']
        )
        
        self.prom_signal_quality = Gauge(
            'bci_signal_quality',
            'Neural signal quality metrics',
            ['channel', 'metric_type']
        )
        
        self.prom_memory_usage = Gauge(
            'bci_memory_usage_bytes',
            'Memory usage in bytes',
            ['memory_type']
        )
        
        self.prom_privacy_epsilon = Gauge(
            'bci_privacy_epsilon',
            'Current differential privacy epsilon value'
        )
        
        # Start Prometheus metrics server
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing and metrics"""
        if not self.enable_opentelemetry:
            return
        
        try:
            # Setup tracing
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
            # Setup metrics
            metrics.set_meter_provider(MeterProvider())
            self.meter = metrics.get_meter(__name__)
            
            logger.info("OpenTelemetry initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def _start_system_monitoring(self) -> None:
        """Start background system monitoring thread"""
        def monitor_system():
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Update gauges
                    self.record_gauge("system_cpu_percent", cpu_percent)
                    self.record_gauge("system_memory_percent", memory.percent)
                    self.record_gauge("system_memory_used_mb", memory.used / 1024 / 1024)
                    
                    # Update Prometheus if available
                    if self.enable_prometheus:
                        self.prom_memory_usage.labels(memory_type="system").set(memory.used)
                    
                    time.sleep(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    @contextmanager
    def measure_time(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for measuring execution time"""
        start_time = time.time()
        labels = labels or {}
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_histogram(metric_name, duration, labels)
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for OpenTelemetry tracing"""
        if not self.enable_opentelemetry:
            yield
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric"""
        with self._lock:
            self.counters[name] += value
            
            metric = MetricValue(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.COUNTER
            )
            self.metrics_history[name].append(metric)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            
            metric = MetricValue(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.GAUGE
            )
            self.metrics_history[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric"""
        labels = labels or {}
        
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels,
            metric_type=MetricType.HISTOGRAM
        )
        
        with self._lock:
            self.metrics_history[name].append(metric)
        
        # Update Prometheus histogram if available
        if self.enable_prometheus and hasattr(self, f'prom_{name}'):
            prom_metric = getattr(self, f'prom_{name}')
            if labels:
                prom_metric.labels(**labels).observe(value)
            else:
                prom_metric.observe(value)
    
    def record_inference_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record comprehensive inference performance metrics"""
        timestamp = time.time()
        
        # Record individual metrics
        self.record_histogram("inference_latency_ms", metrics.inference_latency_ms)
        self.record_histogram("preprocessing_time_ms", metrics.preprocessing_time_ms)
        self.record_histogram("model_forward_time_ms", metrics.model_forward_time_ms)
        self.record_histogram("postprocessing_time_ms", metrics.postprocessing_time_ms)
        
        self.record_gauge("memory_usage_mb", metrics.total_memory_mb)
        self.record_gauge("gpu_memory_mb", metrics.gpu_memory_mb)
        self.record_gauge("cpu_usage_percent", metrics.cpu_usage_percent)
        self.record_gauge("throughput_tokens_per_second", metrics.throughput_tokens_per_second)
        self.record_gauge("accuracy_score", metrics.accuracy_score)
        self.record_gauge("confidence_score", metrics.confidence_score)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.prom_inference_duration.observe(metrics.inference_latency_ms / 1000)
            self.prom_preprocessing_duration.observe(metrics.preprocessing_time_ms / 1000)
            self.prom_model_accuracy.labels(model_type="bci_decoder").set(metrics.accuracy_score)
    
    def record_signal_quality(self, channel: int, snr: float, artifact_level: float) -> None:
        """Record neural signal quality metrics"""
        self.record_gauge("signal_snr", snr, {"channel": str(channel)})
        self.record_gauge("signal_artifacts", artifact_level, {"channel": str(channel)})
        
        if self.enable_prometheus:
            self.prom_signal_quality.labels(channel=str(channel), metric_type="snr").set(snr)
            self.prom_signal_quality.labels(channel=str(channel), metric_type="artifacts").set(artifact_level)
    
    def record_privacy_metrics(self, epsilon: float, delta: float, noise_level: float) -> None:
        """Record differential privacy metrics"""
        self.record_gauge("privacy_epsilon", epsilon)
        self.record_gauge("privacy_delta", delta)
        self.record_gauge("privacy_noise_level", noise_level)
        
        if self.enable_prometheus:
            self.prom_privacy_epsilon.set(epsilon)
    
    def increment_request_counter(self, endpoint: str, status: str) -> None:
        """Increment request counter with labels"""
        self.record_counter("requests_total", 1, {"endpoint": endpoint, "status": status})
        
        if self.enable_prometheus:
            self.prom_requests_total.labels(endpoint=endpoint, status=status).inc()
        
        with self._lock:
            self.total_requests += 1
    
    def set_active_requests(self, count: int) -> None:
        """Set number of active requests"""
        with self._lock:
            self.active_requests = count
        
        self.record_gauge("active_requests", count)
        
        if self.enable_prometheus:
            self.prom_active_requests.set(count)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                "system": {
                    "uptime_seconds": uptime,
                    "total_requests": self.total_requests,
                    "active_requests": self.active_requests,
                    "error_count": self.error_count,
                    "requests_per_second": self.total_requests / uptime if uptime > 0 else 0
                },
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "recent_metrics": {
                    name: list(deque)[-10:] if deque else []
                    for name, deque in self.metrics_history.items()
                }
            }
    
    def get_performance_report(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Generate performance report for specified time window"""
        cutoff_time = time.time() - time_window_seconds
        
        report = {
            "time_window_seconds": time_window_seconds,
            "metrics": {}
        }
        
        with self._lock:
            for metric_name, history in self.metrics_history.items():
                recent_values = [
                    m.value for m in history 
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_values:
                    report["metrics"][metric_name] = {
                        "count": len(recent_values),
                        "mean": np.mean(recent_values),
                        "median": np.median(recent_values),
                        "std": np.std(recent_values),
                        "min": np.min(recent_values),
                        "max": np.max(recent_values),
                        "p95": np.percentile(recent_values, 95),
                        "p99": np.percentile(recent_values, 99)
                    }
        
        return report
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        summary = self.get_metrics_summary()
        
        if format == "json":
            import json
            return json.dumps(summary, indent=2)
        elif format == "prometheus":
            # Generate Prometheus format
            lines = []
            for name, value in summary["gauges"].items():
                lines.append(f"bci_{name} {value}")
            for name, value in summary["counters"].items():
                lines.append(f"bci_{name}_total {value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_alert(
        self, 
        metric_name: str, 
        threshold: float, 
        comparison: str = "greater",
        callback: Optional[Callable] = None
    ) -> None:
        """Create alert for metric threshold"""
        def check_alert():
            while True:
                try:
                    current_value = self.gauges.get(metric_name, 0)
                    
                    triggered = False
                    if comparison == "greater" and current_value > threshold:
                        triggered = True
                    elif comparison == "less" and current_value < threshold:
                        triggered = True
                    elif comparison == "equal" and abs(current_value - threshold) < 1e-6:
                        triggered = True
                    
                    if triggered:
                        alert_msg = f"Alert: {metric_name} = {current_value} ({comparison} {threshold})"
                        logger.warning(alert_msg)
                        
                        if callback:
                            callback(metric_name, current_value, threshold)
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Alert check error: {e}")
                    time.sleep(60)
        
        alert_thread = threading.Thread(target=check_alert, daemon=True)
        alert_thread.start()
        
        logger.info(f"Created alert for {metric_name} {comparison} {threshold}")


# Global metrics collector instance
_metrics_collector: Optional[BCIMetricsCollector] = None


def get_metrics_collector() -> BCIMetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = BCIMetricsCollector()
    return _metrics_collector


def init_metrics(
    enable_prometheus: bool = True, 
    enable_opentelemetry: bool = True,
    metrics_port: int = 9090
) -> BCIMetricsCollector:
    """Initialize global metrics collector"""
    global _metrics_collector
    _metrics_collector = BCIMetricsCollector(
        enable_prometheus=enable_prometheus,
        enable_opentelemetry=enable_opentelemetry,
        metrics_port=metrics_port
    )
    return _metrics_collector


# Decorator for automatic metrics collection
def monitor_performance(metric_prefix: str = ""):
    """Decorator for automatic performance monitoring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            metric_name = f"{metric_prefix}_{func.__name__}" if metric_prefix else func.__name__
            
            with collector.measure_time(f"{metric_name}_duration"):
                with collector.trace_operation(func.__name__):
                    try:
                        result = func(*args, **kwargs)
                        collector.increment_request_counter(func.__name__, "success")
                        return result
                    except Exception as e:
                        collector.increment_request_counter(func.__name__, "error")
                        raise
        
        return wrapper
    return decorator