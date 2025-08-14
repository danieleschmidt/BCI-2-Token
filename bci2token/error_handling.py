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