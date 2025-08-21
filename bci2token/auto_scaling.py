"""
Auto-scaling and dynamic resource management for BCI-2-Token.

Provides intelligent scaling based on workload, performance monitoring,
and resource utilization.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .utils import BCIError
from .advanced_monitoring import MetricsCollector, AlertLevel
from .performance_scaling import ProcessingPool


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    queue_length: int
    response_time: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingRule:
    """Auto-scaling rule."""
    name: str
    metric: str
    threshold: float
    direction: ScalingDirection
    cooldown: float = 300.0  # 5 minutes
    priority: int = 1


class AutoScaler:
    """
    Intelligent auto-scaler for BCI processing resources.
    """
    
    def __init__(self, min_workers: int = 2, max_workers: int = None,
                 scale_factor: float = 1.5, monitoring_interval: float = 30.0):
        self.min_workers = min_workers
        if HAS_PSUTIL:
            self.max_workers = max_workers or psutil.cpu_count() * 4
        else:
            import multiprocessing
            self.max_workers = max_workers or multiprocessing.cpu_count() * 4
        self.scale_factor = scale_factor
        self.monitoring_interval = monitoring_interval
        
        # Current state
        self.current_workers = min_workers
        self.processing_pool = ProcessingPool(max_workers=self.current_workers, auto_scale=False)
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._setup_default_rules()
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=100)
        self.last_scaling_time = 0
        self.scaling_decisions = deque(maxlen=50)
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_collector = MetricsCollector()
        
        # Thread safety
        self.lock = threading.RLock()
        
    def make_scaling_decision(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Make scaling decision based on metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Check cooling period
            if time.time() - self.last_scaling_time < 60:  # 1 minute cooldown
                return None
            
            # Apply scaling rules
            for rule in sorted(self.scaling_rules, key=lambda r: r.priority):
                metric_value = getattr(metrics, rule.metric)
                
                if rule.direction == ScalingDirection.UP:
                    if metric_value > rule.threshold and self.current_workers < self.max_workers:
                        decision = {
                            'direction': 'up',
                            'rule': rule.name,
                            'current_workers': self.current_workers,
                            'new_workers': min(int(self.current_workers * self.scale_factor), self.max_workers),
                            'metric_value': metric_value,
                            'threshold': rule.threshold
                        }
                        self.scaling_decisions.append(decision)
                        return decision
                
                elif rule.direction == ScalingDirection.DOWN:
                    if metric_value < rule.threshold and self.current_workers > self.min_workers:
                        decision = {
                            'direction': 'down',
                            'rule': rule.name,
                            'current_workers': self.current_workers,
                            'new_workers': max(int(self.current_workers / self.scale_factor), self.min_workers),
                            'metric_value': metric_value,
                            'threshold': rule.threshold
                        }
                        self.scaling_decisions.append(decision)
                        return decision
            
            return None
    
    def get_scaling_rules(self) -> List[ScalingRule]:
        """Get current scaling rules."""
        return self.scaling_rules.copy()
        
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        self.scaling_rules = [
            # Scale up rules
            ScalingRule("cpu_high", "cpu_usage", 80.0, ScalingDirection.UP, priority=1),
            ScalingRule("memory_high", "memory_usage", 80.0, ScalingDirection.UP, priority=2),
            ScalingRule("queue_long", "queue_length", 10, ScalingDirection.UP, priority=1),
            ScalingRule("slow_response", "response_time", 2.0, ScalingDirection.UP, priority=2),
            ScalingRule("high_errors", "error_rate", 0.1, ScalingDirection.UP, priority=3),
            
            # Scale down rules
            ScalingRule("cpu_low", "cpu_usage", 20.0, ScalingDirection.DOWN, priority=1),
            ScalingRule("memory_low", "memory_usage", 30.0, ScalingDirection.DOWN, priority=2),
            ScalingRule("queue_short", "queue_length", 2, ScalingDirection.DOWN, priority=3),
            ScalingRule("fast_response", "response_time", 0.5, ScalingDirection.DOWN, priority=2),
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule."""
        with self.lock:
            self.scaling_rules.append(rule)
            # Sort by priority
            self.scaling_rules.sort(key=lambda r: r.priority)
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logging.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_scaling_metrics()
                
                # Store metrics
                with self.lock:
                    self.metrics_history.append(metrics)
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                if decision != ScalingDirection.STABLE:
                    self._execute_scaling(decision)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect metrics for scaling decisions."""
        if HAS_PSUTIL:
            try:
                # System metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
            except Exception:
                # Fallback if psutil not available
                cpu_usage = 50.0
                memory_usage = 50.0
        else:
            # Fallback if psutil not available
            cpu_usage = 50.0
            memory_usage = 50.0
        
        # Processing metrics
        pool_stats = self.processing_pool.get_stats()
        queue_length = pool_stats.get('active_tasks', 0)
        response_time = pool_stats.get('avg_task_time', 0)
        error_rate = 1.0 - pool_stats.get('success_rate', 1.0)
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_length=queue_length,
            response_time=response_time,
            error_rate=error_rate,
            timestamp=time.time()
        )
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDirection:
        """Make scaling decision based on metrics and rules."""
        # Check cooldown period
        if time.time() - self.last_scaling_time < 300:  # 5 minute cooldown
            return ScalingDirection.STABLE
        
        # Evaluate rules
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.scaling_rules:
            metric_value = getattr(metrics, rule.metric, 0)
            
            if rule.direction == ScalingDirection.UP:
                if metric_value > rule.threshold:
                    scale_up_votes += rule.priority
            
            elif rule.direction == ScalingDirection.DOWN:
                if metric_value < rule.threshold:
                    scale_down_votes += rule.priority
        
        # Make decision
        if scale_up_votes > scale_down_votes and self.current_workers < self.max_workers:
            return ScalingDirection.UP
        elif scale_down_votes > scale_up_votes and self.current_workers > self.min_workers:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _execute_scaling(self, direction: ScalingDirection):
        """Execute scaling decision."""
        with self.lock:
            old_workers = self.current_workers
            
            if direction == ScalingDirection.UP:
                new_workers = min(
                    int(self.current_workers * self.scale_factor),
                    self.max_workers
                )
            else:  # DOWN
                new_workers = max(
                    int(self.current_workers / self.scale_factor),
                    self.min_workers
                )
            
            if new_workers != old_workers:
                # Update processing pool
                self.processing_pool.max_workers = new_workers
                self.processing_pool._create_executor(new_workers)
                self.current_workers = new_workers
                
                # Record scaling decision
                decision_record = {
                    'timestamp': time.time(),
                    'direction': direction.value,
                    'old_workers': old_workers,
                    'new_workers': new_workers,
                    'reason': self._get_scaling_reason()
                }
                self.scaling_decisions.append(decision_record)
                self.last_scaling_time = time.time()
                
                logging.info(f"Scaled {direction.value}: {old_workers} -> {new_workers} workers")
                
                # Record metrics
                self.metrics_collector.record_metric('autoscaler.workers', new_workers)
                self.metrics_collector.increment_counter(f'autoscaler.scale_{direction.value}')
    
    def _get_scaling_reason(self) -> str:
        """Get reason for scaling based on latest metrics."""
        if not self.metrics_history:
            return "unknown"
        
        latest = self.metrics_history[-1]
        reasons = []
        
        if latest.cpu_usage > 80:
            reasons.append("high_cpu")
        if latest.memory_usage > 80:
            reasons.append("high_memory")
        if latest.queue_length > 10:
            reasons.append("long_queue")
        if latest.response_time > 2.0:
            reasons.append("slow_response")
        
        return ",".join(reasons) if reasons else "policy_based"
    
    def manual_scale(self, target_workers: int) -> bool:
        """Manually scale to target number of workers."""
        target_workers = max(self.min_workers, min(target_workers, self.max_workers))
        
        with self.lock:
            old_workers = self.current_workers
            
            if target_workers != old_workers:
                self.processing_pool.max_workers = target_workers
                self.processing_pool._create_executor(target_workers)
                self.current_workers = target_workers
                
                logging.info(f"Manual scale: {old_workers} -> {target_workers} workers")
                
                # Record manual scaling
                decision_record = {
                    'timestamp': time.time(),
                    'direction': 'manual',
                    'old_workers': old_workers,
                    'new_workers': target_workers,
                    'reason': 'manual_override'
                }
                self.scaling_decisions.append(decision_record)
                
                return True
        
        return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self.lock:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'monitoring_active': self.monitoring_active,
                'latest_metrics': {
                    'cpu_usage': latest_metrics.cpu_usage if latest_metrics else 0,
                    'memory_usage': latest_metrics.memory_usage if latest_metrics else 0,
                    'queue_length': latest_metrics.queue_length if latest_metrics else 0,
                    'response_time': latest_metrics.response_time if latest_metrics else 0,
                    'error_rate': latest_metrics.error_rate if latest_metrics else 0,
                } if latest_metrics else {},
                'recent_decisions': list(self.scaling_decisions)[-10:],
                'scaling_rules_count': len(self.scaling_rules)
            }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about scaling performance."""
        if len(self.metrics_history) < 10:
            return {'message': 'Insufficient data for insights'}
        
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        response_trend = self._calculate_trend([m.response_time for m in recent_metrics])
        
        # Scaling effectiveness
        scale_ups = len([d for d in self.scaling_decisions if d['direction'] == 'up'])
        scale_downs = len([d for d in self.scaling_decisions if d['direction'] == 'down'])
        
        return {
            'resource_trends': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'response_time_trend': response_trend
            },
            'scaling_activity': {
                'scale_ups': scale_ups,
                'scale_downs': scale_downs,
                'total_decisions': len(self.scaling_decisions)
            },
            'current_efficiency': {
                'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                'avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            },
            'recommendations': self._generate_recommendations(recent_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = list(range(len(values)))
        y = values
        
        # Calculate slope
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_recommendations(self, recent_metrics: List[ScalingMetrics]) -> List[str]:
        """Generate scaling recommendations."""
        recommendations = []
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        
        if avg_cpu > 70:
            recommendations.append("Consider increasing max_workers or optimizing CPU-intensive operations")
        
        if avg_memory > 70:
            recommendations.append("Monitor memory usage - consider memory optimization")
        
        if avg_response > 1.0:
            recommendations.append("Response times are high - consider scaling up or optimizing algorithms")
        
        if avg_cpu < 30 and avg_memory < 30 and self.current_workers > self.min_workers:
            recommendations.append("System may be over-provisioned - consider reducing max_workers")
        
        return recommendations


class ResourceManager:
    """
    Comprehensive resource management for BCI processing.
    """
    
    def __init__(self):
        self.auto_scaler = AutoScaler()
        self.resource_limits = {
            'max_memory_mb': 1024,  # 1GB
            'max_cpu_percent': 80,
            'max_disk_usage_gb': 10
        }
        self.resource_usage = {}
        
    def start_management(self):
        """Start resource management."""
        self.auto_scaler.start_monitoring()
        logging.info("Resource management started")
    
    def stop_management(self):
        """Stop resource management."""
        self.auto_scaler.stop_monitoring()
        logging.info("Resource management stopped")
    
    def check_resource_limits(self) -> Dict[str, Any]:
        """Check if resource usage is within limits."""
        if HAS_PSUTIL:
            try:
                # Get current resource usage
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                cpu_percent = psutil.cpu_percent()
                disk_gb = psutil.disk_usage('/').used / (1024 * 1024 * 1024)
                
                self.resource_usage = {
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'disk_gb': disk_gb
                }
                
                # Check limits
                violations = {}
                
                if memory_mb > self.resource_limits['max_memory_mb']:
                    violations['memory'] = f"Usage {memory_mb:.1f}MB > limit {self.resource_limits['max_memory_mb']}MB"
                
                if cpu_percent > self.resource_limits['max_cpu_percent']:
                    violations['cpu'] = f"Usage {cpu_percent:.1f}% > limit {self.resource_limits['max_cpu_percent']}%"
                
                if disk_gb > self.resource_limits['max_disk_usage_gb']:
                    violations['disk'] = f"Usage {disk_gb:.1f}GB > limit {self.resource_limits['max_disk_usage_gb']}GB"
                
                return {
                    'within_limits': len(violations) == 0,
                    'violations': violations,
                    'current_usage': self.resource_usage
                }
                
            except Exception as e:
                return {
                    'within_limits': False,
                    'error': str(e),
                    'current_usage': {}
                }
        else:
            # Mock resource usage when psutil not available
            self.resource_usage = {
                'memory_mb': 500.0,
                'cpu_percent': 25.0,
                'disk_gb': 2.0
            }
            
            return {
                'within_limits': True,
                'violations': {},
                'current_usage': self.resource_usage
            }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        return {
            'auto_scaler': self.auto_scaler.get_scaling_status(),
            'resource_limits': self.resource_limits,
            'resource_check': self.check_resource_limits(),
            'performance_insights': self.auto_scaler.get_performance_insights()
        }


# Global resource manager
_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager."""
    return _resource_manager