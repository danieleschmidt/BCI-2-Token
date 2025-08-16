"""
Robust error handling and recovery mechanisms for BCI-2-Token.

Provides circuit breakers, retry logic, graceful degradation,
and comprehensive error reporting.
"""

import time
import functools
import logging
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import threading
import random

from .utils import BCIError


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    
    
class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascading failures by temporarily blocking calls
    to failing services.
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    self.failed_calls += 1
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Try half-open state
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.successful_calls += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'failure_count': self.failure_count,
            'success_rate': self.successful_calls / max(self.total_calls, 1),
            'last_failure_time': self.last_failure_time
        }


class CircuitBreakerError(BCIError):
    """Circuit breaker is open."""
    pass


def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions."""
    if config is None:
        config = CircuitBreakerConfig()
        
    breaker = CircuitBreaker(config, name)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker for monitoring
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, max_attempts: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter


def with_retry(config: Optional[RetryConfig] = None,
               exceptions: tuple = (Exception,)):
    """Decorator to add retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt, re-raise
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.backoff_multiplier ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    time.sleep(delay)
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


class GracefulDegradation:
    """
    Implements graceful degradation patterns.
    
    Provides fallback mechanisms when primary functionality fails.
    """
    
    def __init__(self):
        self.fallback_functions = {}
        self.degradation_active = set()
        
    def register_fallback(self, primary_name: str, fallback_func: Callable):
        """Register a fallback function for a primary operation."""
        self.fallback_functions[primary_name] = fallback_func
        
    def call_with_fallback(self, primary_name: str, primary_func: Callable,
                          *args, **kwargs) -> Any:
        """
        Call primary function with fallback if it fails.
        """
        try:
            result = primary_func(*args, **kwargs)
            
            # Remove from degradation if it was there
            self.degradation_active.discard(primary_name)
            return result
            
        except Exception as e:
            # Mark as degraded
            self.degradation_active.add(primary_name)
            
            # Try fallback
            if primary_name in self.fallback_functions:
                try:
                    logging.warning(f"Primary function '{primary_name}' failed, using fallback: {e}")
                    return self.fallback_functions[primary_name](*args, **kwargs)
                except Exception as fallback_error:
                    logging.error(f"Fallback for '{primary_name}' also failed: {fallback_error}")
                    raise e
            else:
                raise e
    
    def is_degraded(self, name: str) -> bool:
        """Check if a service is in degraded mode."""
        return name in self.degradation_active
    
    def get_degradation_status(self) -> Dict[str, bool]:
        """Get status of all registered services."""
        status = {}
        for name in self.fallback_functions:
            status[name] = name in self.degradation_active
        return status


class HealthAwareComponent:
    """
    Base class for components with health monitoring and error handling.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(), 
            name=f"{name}_breaker"
        )
        self.degradation = GracefulDegradation()
        self.error_count = 0
        self.last_error = None
        self.last_error_time = None
        
    def execute_safely(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        try:
            return self.circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.last_error_time = time.time()
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return {
            'name': self.name,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time,
            'circuit_breaker': self.circuit_breaker.get_stats(),
            'degradation_status': self.degradation.get_degradation_status()
        }


# Global circuit breaker registry for monitoring
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def register_circuit_breaker(name: str, breaker: CircuitBreaker):
    """Register circuit breaker for global monitoring."""
    _circuit_breakers[name] = breaker


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all registered circuit breakers."""
    return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}


def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state."""
    for breaker in _circuit_breakers.values():
        with breaker.lock:
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0


class EnhancedErrorRecovery:
    """Generation 2 Enhanced Error Recovery Framework with adaptive capabilities."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_patterns = {}
        self.recovery_success_rates = {}
        self.adaptive_thresholds = {}
        self.lock = threading.Lock()
        
    def register_recovery_strategy(self, error_type: type, strategy: Callable, 
                                 success_threshold: float = 0.7):
        """Register adaptive recovery strategy for specific error types."""
        with self.lock:
            self.recovery_strategies[error_type] = {
                'strategy': strategy,
                'success_threshold': success_threshold,
                'attempts': 0,
                'successes': 0
            }
            
    def execute_with_enhanced_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with enhanced error recovery and pattern analysis."""
        attempt = 0
        max_attempts = 3
        
        while attempt < max_attempts:
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record successful execution time for pattern analysis
                execution_time = time.time() - start_time
                self._record_success_pattern(func.__name__, execution_time, attempt)
                
                return result
                
            except Exception as e:
                attempt += 1
                error_type = type(e)
                
                # Record error pattern
                self._record_error_pattern(func.__name__, error_type, attempt)
                
                # Try recovery strategy if available
                if error_type in self.recovery_strategies and attempt < max_attempts:
                    recovery_info = self.recovery_strategies[error_type]
                    
                    try:
                        # Adaptive recovery based on historical success rates
                        if self._should_attempt_recovery(error_type):
                            recovered_result = recovery_info['strategy'](func, e, *args, **kwargs)
                            
                            # Update recovery success rate
                            self._update_recovery_success(error_type, True)
                            
                            return recovered_result
                            
                    except Exception as recovery_error:
                        self._update_recovery_success(error_type, False)
                        # Continue to next attempt or final failure
                        if attempt == max_attempts:
                            raise RecoveryError(f"Function failed and recovery failed: {e}, Recovery error: {recovery_error}")
                
                # Apply exponential backoff for retries
                if attempt < max_attempts:
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(min(backoff_time, 10))  # Cap at 10 seconds
                    
        # All attempts failed
        raise RecoveryError(f"Function failed after {max_attempts} attempts: {e}")
        
    def _should_attempt_recovery(self, error_type: type) -> bool:
        """Determine if recovery should be attempted based on historical success."""
        if error_type not in self.recovery_strategies:
            return False
            
        recovery_info = self.recovery_strategies[error_type]
        attempts = recovery_info['attempts']
        successes = recovery_info['successes']
        
        if attempts < 5:  # Always try recovery for first few attempts
            return True
            
        success_rate = successes / attempts if attempts > 0 else 0
        return success_rate >= recovery_info['success_threshold']
        
    def _update_recovery_success(self, error_type: type, success: bool):
        """Update recovery success statistics."""
        with self.lock:
            if error_type in self.recovery_strategies:
                recovery_info = self.recovery_strategies[error_type]
                recovery_info['attempts'] += 1
                if success:
                    recovery_info['successes'] += 1
                    
    def _record_error_pattern(self, func_name: str, error_type: type, attempt: int):
        """Record error patterns for analysis."""
        with self.lock:
            if func_name not in self.error_patterns:
                self.error_patterns[func_name] = {}
                
            error_key = error_type.__name__
            if error_key not in self.error_patterns[func_name]:
                self.error_patterns[func_name][error_key] = []
                
            self.error_patterns[func_name][error_key].append({
                'timestamp': time.time(),
                'attempt': attempt
            })
            
            # Keep only recent patterns (last 1000 entries)
            if len(self.error_patterns[func_name][error_key]) > 1000:
                self.error_patterns[func_name][error_key] = \
                    self.error_patterns[func_name][error_key][-1000:]
                    
    def _record_success_pattern(self, func_name: str, execution_time: float, retry_count: int):
        """Record successful execution patterns."""
        # This could be used for performance optimization in Generation 3
        pass
        
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get comprehensive error analysis and recovery statistics."""
        with self.lock:
            analysis = {
                'recovery_strategies': {},
                'error_patterns': {},
                'recommendations': []
            }
            
            # Analyze recovery strategy effectiveness
            for error_type, info in self.recovery_strategies.items():
                attempts = info['attempts']
                successes = info['successes']
                success_rate = successes / attempts if attempts > 0 else 0
                
                analysis['recovery_strategies'][error_type.__name__] = {
                    'attempts': attempts,
                    'successes': successes,
                    'success_rate': success_rate,
                    'threshold': info['success_threshold'],
                    'effective': success_rate >= info['success_threshold']
                }
                
                # Generate recommendations
                if attempts > 10 and success_rate < 0.3:
                    analysis['recommendations'].append(
                        f"Consider revising recovery strategy for {error_type.__name__} (low success rate: {success_rate:.2f})"
                    )
                    
            # Analyze error patterns
            current_time = time.time()
            for func_name, errors in self.error_patterns.items():
                recent_errors = 0
                error_types = []
                
                for error_type, occurrences in errors.items():
                    # Count errors in last hour
                    recent_count = sum(1 for occurrence in occurrences 
                                     if current_time - occurrence['timestamp'] < 3600)
                    recent_errors += recent_count
                    
                    if recent_count > 0:
                        error_types.append(error_type)
                        
                if recent_errors > 0:
                    analysis['error_patterns'][func_name] = {
                        'recent_errors': recent_errors,
                        'error_types': error_types
                    }
                    
                    if recent_errors > 10:
                        analysis['recommendations'].append(
                            f"High error rate for {func_name}: {recent_errors} errors in last hour"
                        )
                        
            return analysis
            
    def optimize_recovery_strategies(self):
        """Optimize recovery strategies based on historical performance."""
        with self.lock:
            for error_type, info in self.recovery_strategies.items():
                attempts = info['attempts']
                successes = info['successes']
                
                if attempts > 20:  # Enough data for optimization
                    success_rate = successes / attempts
                    
                    # Adjust threshold based on actual performance
                    if success_rate > 0.8:
                        # Very successful, can be more aggressive
                        info['success_threshold'] = max(0.6, info['success_threshold'] - 0.1)
                    elif success_rate < 0.3:
                        # Poor performance, be more conservative
                        info['success_threshold'] = min(0.9, info['success_threshold'] + 0.1)


class AdaptiveRetryConfig:
    """Adaptive retry configuration that learns from patterns."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.success_delays = []
        self.failure_delays = []
        
    def get_adaptive_delay(self, attempt: int) -> float:
        """Get adaptive delay based on historical patterns."""
        # Start with exponential backoff
        base_delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        # Adjust based on historical success patterns
        if len(self.success_delays) > 5:
            avg_success_delay = sum(self.success_delays[-10:]) / len(self.success_delays[-10:])
            # If successful operations typically happen after longer delays, adjust accordingly
            if avg_success_delay > base_delay:
                base_delay = min(avg_success_delay * 0.8, self.max_delay)
                
        return base_delay
        
    def record_outcome(self, delay_used: float, success: bool):
        """Record the outcome of a retry attempt."""
        if success:
            self.success_delays.append(delay_used)
            # Keep only recent history
            if len(self.success_delays) > 100:
                self.success_delays = self.success_delays[-100:]
        else:
            self.failure_delays.append(delay_used)
            if len(self.failure_delays) > 100:
                self.failure_delays = self.failure_delays[-100:]


class RecoveryError(BCIError):
    """Error indicating recovery attempts have failed."""
    pass