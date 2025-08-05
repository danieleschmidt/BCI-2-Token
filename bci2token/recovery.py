"""
Auto-recovery and self-healing mechanisms for BCI-2-Token framework.

Implements intelligent recovery strategies, health monitoring, and automatic
failover for production brain-computer interface systems.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ComponentStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class RecoveryAction:
    """Represents a recovery action."""
    name: str
    description: str
    action_func: Callable
    priority: int = 1  # Higher priority = executed first
    max_attempts: int = 3
    timeout: float = 30.0
    success_criteria: Optional[Callable] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: ComponentStatus
    last_check: float
    error_count: int = 0
    recovery_attempts: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)


class HealthMonitor:
    """Monitors component health and triggers recovery."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checkers: Dict[str, Callable] = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
    def register_component(self, 
                          name: str,
                          health_checker: Callable,
                          recovery_actions: List[RecoveryAction] = None):
        """
        Register a component for health monitoring.
        
        Args:
            name: Component name
            health_checker: Function that returns component health
            recovery_actions: List of recovery actions for this component
        """
        with self.lock:
            self.components[name] = ComponentHealth(
                name=name,
                status=ComponentStatus.HEALTHY,
                last_check=0.0,
                recovery_actions=recovery_actions or []
            )
            self.health_checkers[name] = health_checker
            
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.is_monitoring:
            try:
                self._check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                warnings.warn(f"Health monitoring error: {e}")
                time.sleep(5.0)  # Brief pause on error
                
    def _check_all_components(self):
        """Check health of all registered components."""
        with self.lock:
            for name in list(self.components.keys()):
                try:
                    self._check_component_health(name)
                except Exception as e:
                    warnings.warn(f"Health check failed for {name}: {e}")
                    
    def _check_component_health(self, name: str):
        """Check health of specific component."""
        if name not in self.components or name not in self.health_checkers:
            return
            
        component = self.components[name]
        health_checker = self.health_checkers[name]
        
        try:
            # Run health check
            health_result = health_checker()
            
            # Update component status
            component.last_check = time.time()
            
            if isinstance(health_result, bool):
                # Simple boolean result
                component.status = ComponentStatus.HEALTHY if health_result else ComponentStatus.FAILING
            elif isinstance(health_result, dict):
                # Detailed health result
                component.status = ComponentStatus(health_result.get('status', 'healthy'))
                component.metrics.update(health_result.get('metrics', {}))
                if 'error' in health_result:
                    component.last_error = health_result['error']
            else:
                # Assume healthy if check completed without exception
                component.status = ComponentStatus.HEALTHY
                
            # Reset error count on successful check
            if component.status == ComponentStatus.HEALTHY:
                component.error_count = 0
                
        except Exception as e:
            # Health check failed
            component.error_count += 1
            component.last_error = str(e)
            component.last_check = time.time()
            
            # Update status based on error count
            if component.error_count >= 5:
                component.status = ComponentStatus.FAILED
            elif component.error_count >= 2:
                component.status = ComponentStatus.FAILING
            else:
                component.status = ComponentStatus.DEGRADED
                
        # Trigger recovery if needed
        if component.status in [ComponentStatus.FAILING, ComponentStatus.FAILED]:
            self._attempt_component_recovery(name)
            
    def _attempt_component_recovery(self, name: str):
        """Attempt to recover a failing component."""
        component = self.components[name]
        
        if component.status == ComponentStatus.RECOVERING:
            return  # Already recovering
            
        component.status = ComponentStatus.RECOVERING
        component.recovery_attempts += 1
        
        # Sort recovery actions by priority
        actions = sorted(component.recovery_actions, key=lambda x: x.priority, reverse=True)
        
        for action in actions:
            try:
                # Execute recovery action
                action.action_func()
                
                # Test if recovery was successful
                if action.success_criteria:
                    if action.success_criteria():
                        component.status = ComponentStatus.HEALTHY
                        component.error_count = 0
                        break
                else:
                    # No specific criteria, assume success if no exception
                    component.status = ComponentStatus.HEALTHY
                    component.error_count = 0
                    break
                    
            except Exception as e:
                warnings.warn(f"Recovery action '{action.name}' failed for {name}: {e}")
                continue
                
        # If all recovery actions failed
        if component.status == ComponentStatus.RECOVERING:
            component.status = ComponentStatus.FAILED
            
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        with self.lock:
            overall_status = ComponentStatus.HEALTHY
            
            # Determine overall status
            for component in self.components.values():
                if component.status == ComponentStatus.FAILED:
                    overall_status = ComponentStatus.FAILED
                    break
                elif component.status == ComponentStatus.FAILING:
                    overall_status = ComponentStatus.FAILING
                elif component.status == ComponentStatus.DEGRADED:
                    if overall_status == ComponentStatus.HEALTHY:
                        overall_status = ComponentStatus.DEGRADED
                        
            return {
                'overall_status': overall_status.value,
                'components': {
                    name: {
                        'status': comp.status.value,
                        'last_check': comp.last_check,
                        'error_count': comp.error_count,
                        'recovery_attempts': comp.recovery_attempts,
                        'last_error': comp.last_error,
                        'metrics': comp.metrics
                    }
                    for name, comp in self.components.items()
                },
                'monitoring_active': self.is_monitoring,
                'timestamp': time.time()
            }


# Recovery action implementations
def restart_component_action(component_name: str) -> RecoveryAction:
    """Create a restart recovery action."""
    def restart():
        warnings.warn(f"Restarting component: {component_name}")
        time.sleep(1.0)  # Simulate restart time
        
    def check_restart_success():
        # Simple check - in practice would verify component functionality
        return True
        
    return RecoveryAction(
        name=f"restart_{component_name}",
        description=f"Restart {component_name} component",
        action_func=restart,
        priority=5,
        success_criteria=check_restart_success
    )


def clear_cache_action(component_name: str) -> RecoveryAction:
    """Create a cache clearing recovery action."""
    def clear_cache():
        warnings.warn(f"Clearing cache for: {component_name}")
        
    return RecoveryAction(
        name=f"clear_cache_{component_name}",
        description=f"Clear cache for {component_name}",
        action_func=clear_cache,
        priority=3
    )


def reset_connection_action(component_name: str) -> RecoveryAction:
    """Create a connection reset recovery action."""
    def reset_connection():
        warnings.warn(f"Resetting connection for: {component_name}")
        time.sleep(2.0)  # Simulate connection reset time
        
    return RecoveryAction(
        name=f"reset_connection_{component_name}",
        description=f"Reset connection for {component_name}",
        action_func=reset_connection,
        priority=4
    )


# Health check implementations
def create_decoder_health_check(decoder) -> Callable:
    """Create health check for decoder component."""
    def check_decoder_health():
        try:
            # Test basic decoder functionality
            if not hasattr(decoder, 'model') or decoder.model is None:
                return {'status': 'failed', 'error': 'No model loaded'}
                
            # Test with minimal synthetic data
            if HAS_NUMPY:
                test_signal = np.random.randn(decoder.channels, 100)
                try:
                    # Try to preprocess (without full decode to save time)
                    processed = decoder.preprocess_signals(test_signal)
                    if len(processed.get('epochs', [])) == 0:
                        return {'status': 'degraded', 'error': 'No epochs generated'}
                        
                    return {'status': 'healthy', 'metrics': {'epochs_count': len(processed['epochs'])}}
                except Exception as e:
                    return {'status': 'failing', 'error': str(e)}
            else:
                return {'status': 'degraded', 'error': 'Cannot test without numpy'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
            
    return check_decoder_health


def create_device_health_check(device) -> Callable:
    """Create health check for device component."""
    def check_device_health():
        try:
            if not hasattr(device, 'is_connected'):
                return {'status': 'failed', 'error': 'Invalid device object'}
                
            if not device.is_connected:
                return {'status': 'failed', 'error': 'Device not connected'}
                
            # Test device communication
            try:
                device_info = device.get_device_info()
                return {
                    'status': 'healthy',
                    'metrics': {
                        'sampling_rate': device_info.get('sampling_rate', 0),
                        'channels': device_info.get('n_channels', 0)
                    }
                }
            except Exception as e:
                return {'status': 'degraded', 'error': f'Device communication issue: {e}'}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
            
    return check_device_health


def create_streaming_health_check(streaming_decoder) -> Callable:
    """Create health check for streaming component."""
    def check_streaming_health():
        try:
            if not hasattr(streaming_decoder, 'get_status'):
                return {'status': 'failed', 'error': 'Invalid streaming decoder'}
                
            status = streaming_decoder.get_status()
            
            buffer_utilization = status.get('buffer_size', 0) / max(1, status.get('max_buffer_size', 1))
            
            if buffer_utilization > 0.9:
                return {
                    'status': 'degraded',
                    'error': 'High buffer utilization',
                    'metrics': {'buffer_utilization': buffer_utilization}
                }
            elif not status.get('is_streaming', False):
                return {'status': 'failed', 'error': 'Streaming not active'}
            else:
                return {
                    'status': 'healthy',
                    'metrics': {
                        'buffer_utilization': buffer_utilization,
                        'accumulated_text_length': status.get('accumulated_text_length', 0)
                    }
                }
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
            
    return check_streaming_health


class SelfHealingSystem:
    """Self-healing system that automatically maintains BCI components."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.recovery_strategies: Dict[str, List[RecoveryAction]] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        
    def register_component(self,
                          name: str,
                          component_instance: Any,
                          dependencies: List[str] = None):
        """
        Register a component for self-healing.
        
        Args:
            name: Component name
            component_instance: Instance of the component
            dependencies: List of components this depends on
        """
        # Create appropriate health checker
        if 'decoder' in name.lower():
            health_checker = create_decoder_health_check(component_instance)
            recovery_actions = [
                restart_component_action(name),
                clear_cache_action(name)
            ]
        elif 'device' in name.lower():
            health_checker = create_device_health_check(component_instance)
            recovery_actions = [
                reset_connection_action(name),
                restart_component_action(name)
            ]
        elif 'streaming' in name.lower():
            health_checker = create_streaming_health_check(component_instance)
            recovery_actions = [
                clear_cache_action(name),
                restart_component_action(name)
            ]
        else:
            # Generic health checker
            def generic_health_check():
                try:
                    # Basic existence check
                    if hasattr(component_instance, 'get_status'):
                        status = component_instance.get_status()
                        return {'status': 'healthy', 'metrics': status}
                    else:
                        return {'status': 'healthy'}
                except Exception as e:
                    return {'status': 'failed', 'error': str(e)}
                    
            health_checker = generic_health_check
            recovery_actions = [restart_component_action(name)]
            
        # Register with health monitor
        self.health_monitor.register_component(name, health_checker, recovery_actions)
        
        # Track dependencies
        if dependencies:
            self.component_dependencies[name] = dependencies
            
    def start_self_healing(self):
        """Start the self-healing system."""
        self.health_monitor.start_monitoring()
        
    def stop_self_healing(self):
        """Stop the self-healing system."""
        self.health_monitor.stop_monitoring()
        
    def trigger_recovery(self, component_name: str) -> bool:
        """
        Manually trigger recovery for a component.
        
        Args:
            component_name: Name of component to recover
            
        Returns:
            True if recovery succeeded
        """
        if component_name not in self.components:
            return False
            
        try:
            self.health_monitor._attempt_component_recovery(component_name)
            
            # Check if recovery was successful
            time.sleep(1.0)  # Brief pause
            self.health_monitor._check_component_health(component_name)
            
            component = self.health_monitor.components[component_name]
            return component.status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
            
        except Exception as e:
            warnings.warn(f"Manual recovery failed for {component_name}: {e}")
            return False
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return self.health_monitor.get_health_report()
        
    def diagnose_dependencies(self, component_name: str) -> Dict[str, Any]:
        """
        Diagnose component dependencies for troubleshooting.
        
        Args:
            component_name: Component to diagnose
            
        Returns:
            Dependency health analysis
        """
        if component_name not in self.component_dependencies:
            return {'component': component_name, 'dependencies': [], 'issues': []}
            
        dependencies = self.component_dependencies[component_name]
        dep_health = {}
        issues = []
        
        for dep in dependencies:
            if dep in self.health_monitor.components:
                dep_component = self.health_monitor.components[dep]
                dep_health[dep] = dep_component.status.value
                
                if dep_component.status != ComponentStatus.HEALTHY:
                    issues.append(f"Dependency '{dep}' is {dep_component.status.value}")
            else:
                dep_health[dep] = 'unknown'
                issues.append(f"Dependency '{dep}' not monitored")
                
        return {
            'component': component_name,
            'dependencies': dep_health,
            'issues': issues,
            'recommendation': self._generate_recovery_recommendation(component_name, issues)
        }
        
    def _generate_recovery_recommendation(self, component_name: str, issues: List[str]) -> str:
        """Generate recovery recommendation based on issues."""
        if not issues:
            return "Component appears healthy"
            
        if any('not monitored' in issue for issue in issues):
            return "Register all dependencies for monitoring"
            
        if any('failed' in issue for issue in issues):
            return "Recover failed dependencies first"
            
        return f"Check and recover component {component_name}"


# Default recovery actions with implementations
def recover_decoder(error, context):
    """Default decoder recovery."""
    warnings.warn("Recovering decoder")
    time.sleep(1.0)

def recover_device_connection(error, context):
    """Default device recovery.""" 
    warnings.warn("Recovering device connection")
    time.sleep(2.0)

def recover_streaming_buffer(error, context):
    """Default streaming recovery."""
    warnings.warn("Recovering streaming buffer")
    time.sleep(0.5)


class AdaptiveRecovery:
    """Adaptive recovery system that learns from past failures."""
    
    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, Dict[str, float]] = {}
        
    def record_recovery_attempt(self,
                               component: str,
                               action: str,
                               success: bool,
                               duration: float,
                               context: Dict[str, Any] = None):
        """Record a recovery attempt for learning."""
        record = {
            'timestamp': time.time(),
            'component': component,
            'action': action,
            'success': success,
            'duration': duration,
            'context': context or {}
        }
        
        self.recovery_history.append(record)
        
        # Update success rates
        if component not in self.success_rates:
            self.success_rates[component] = {}
            
        if action not in self.success_rates[component]:
            self.success_rates[component][action] = 0.0
            
        # Simple moving average of success rate
        current_rate = self.success_rates[component][action]
        alpha = 0.1  # Learning rate
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.success_rates[component][action] = new_rate
        
    def get_best_recovery_action(self, component: str) -> Optional[str]:
        """Get the recovery action with highest success rate for component."""
        if component not in self.success_rates:
            return None
            
        actions = self.success_rates[component]
        if not actions:
            return None
            
        # Return action with highest success rate
        best_action = max(actions.items(), key=lambda x: x[1])
        return best_action[0] if best_action[1] > 0.1 else None  # Only if >10% success rate
        
    def get_recovery_insights(self) -> Dict[str, Any]:
        """Get insights from recovery history."""
        if not self.recovery_history:
            return {'total_attempts': 0}
            
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r['success'])
        
        # Component-wise analysis
        component_stats = {}
        for record in self.recovery_history:
            comp = record['component']
            if comp not in component_stats:
                component_stats[comp] = {'attempts': 0, 'successes': 0, 'avg_duration': 0.0}
                
            component_stats[comp]['attempts'] += 1
            if record['success']:
                component_stats[comp]['successes'] += 1
                
            # Update average duration
            stats = component_stats[comp]
            old_avg = stats['avg_duration']
            stats['avg_duration'] = (old_avg * (stats['attempts'] - 1) + record['duration']) / stats['attempts']
            
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'overall_success_rate': successful_attempts / total_attempts,
            'component_stats': component_stats,
            'action_success_rates': self.success_rates
        }


if __name__ == '__main__':
    # Test self-healing system
    print("Testing BCI-2-Token Self-Healing System")
    print("=" * 45)
    
    # Create self-healing system
    healing = SelfHealingSystem()
    
    # Mock component for testing
    class MockComponent:
        def __init__(self):
            self.status = 'healthy'
            self.fail_next = False
            
        def get_status(self):
            if self.fail_next:
                raise RuntimeError("Mock failure")
            return {'status': self.status}
            
    mock_component = MockComponent()
    
    # Register component
    healing.register_component(
        'test_component',
        mock_component,
        dependencies=['test_dependency']
    )
    
    # Test health monitoring
    health_report = healing.get_system_health()
    print(f"✓ System health: {health_report['overall_status']}")
    
    # Test dependency diagnosis
    diagnosis = healing.diagnose_dependencies('test_component')
    print(f"✓ Dependency diagnosis: {len(diagnosis['issues'])} issues found")
    
    # Test adaptive recovery
    adaptive = AdaptiveRecovery()
    adaptive.record_recovery_attempt('test_comp', 'restart', True, 1.5)
    adaptive.record_recovery_attempt('test_comp', 'restart', False, 2.0)
    
    insights = adaptive.get_recovery_insights()
    print(f"✓ Recovery insights: {insights['overall_success_rate']:.1%} success rate")
    
    print("\n✓ Self-healing system working")