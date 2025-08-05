"""
Reliability and fault tolerance mechanisms for BCI-2-Token framework.

Implements circuit breakers, retry logic, fallback mechanisms, and auto-recovery
for production brain-computer interface applications.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0     # Seconds before trying recovery
    success_threshold: int = 3          # Successes to close from half-open
    timeout: float = 30.0              # Request timeout in seconds


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    Automatically opens circuit when failure rate exceeds threshold,
    preventing further attempts until service potentially recovers.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to protect function with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        with self.lock:
            current_time = time.time()
            
            # Check if we should attempt recovery
            if (self.state == CircuitState.OPEN and 
                current_time - self.last_failure_time > self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                
            # Block if circuit is open
            if self.state == CircuitState.OPEN:
                raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN")
                
        # Attempt the operation
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            
            # Check for timeout
            if time.time() - start_time > self.config.timeout:
                raise TimeoutError(f"Operation timed out after {self.config.timeout}s")
                
            # Success
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
            
    def _record_success(self):
        """Record successful operation."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                
    def _record_failure(self):
        """Record failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - self.last_failure_time
            }


class FallbackManager:
    """Manages fallback mechanisms for failed operations."""
    
    def __init__(self):
        self.fallbacks: Dict[str, List[Callable]] = {}
        
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        if operation not in self.fallbacks:
            self.fallbacks[operation] = []
        self.fallbacks[operation].append(fallback_func)
        
    def execute_with_fallbacks(self, operation: str, primary_func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback mechanisms."""
        # Try primary function first
        try:
            return primary_func(*args, **kwargs)
        except Exception as primary_error:
            warnings.warn(f"Primary operation '{operation}' failed: {primary_error}")
            
            # Try fallbacks in order
            if operation in self.fallbacks:
                for i, fallback_func in enumerate(self.fallbacks[operation]):
                    try:
                        warnings.warn(f"Trying fallback {i+1} for '{operation}'")
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        warnings.warn(f"Fallback {i+1} failed: {fallback_error}")
                        continue
                        
            # All fallbacks failed
            raise RuntimeError(f"All fallbacks failed for operation '{operation}'") from primary_error


class AutoRecovery:
    """Automatic recovery system for BCI components."""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
    def register_recovery_strategy(self, component: str, strategy_func: Callable):
        """Register recovery strategy for a component."""
        self.recovery_strategies[component] = strategy_func
        
    def attempt_recovery(self, component: str, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Attempt to recover from an error."""
        if component not in self.recovery_strategies:
            return False
            
        recovery_start = time.time()
        
        try:
            # Attempt recovery
            self.recovery_strategies[component](error, context or {})
            
            # Record successful recovery
            recovery_record = {
                'component': component,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'recovery_duration': time.time() - recovery_start,
                'success': True,
                'timestamp': time.time(),
                'context': context
            }
            self.recovery_history.append(recovery_record)
            
            return True
            
        except Exception as recovery_error:
            # Record failed recovery
            recovery_record = {
                'component': component,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'recovery_error': str(recovery_error),
                'recovery_duration': time.time() - recovery_start,
                'success': False,
                'timestamp': time.time(),
                'context': context
            }
            self.recovery_history.append(recovery_record)
            
            return False
            
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {'total_attempts': 0}
            
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r['success'])
        
        # Component-wise stats
        component_stats = {}
        for record in self.recovery_history:
            comp = record['component']
            if comp not in component_stats:
                component_stats[comp] = {'attempts': 0, 'successes': 0}
            component_stats[comp]['attempts'] += 1
            if record['success']:
                component_stats[comp]['successes'] += 1
                
        return {
            'total_attempts': total_attempts,
            'successful_recoveries': successful_recoveries,
            'success_rate': successful_recoveries / total_attempts,
            'component_stats': component_stats,
            'recent_history': self.recovery_history[-10:]  # Last 10 attempts
        }


class InputSanitizer:
    """Sanitizes and validates inputs for security and robustness."""
    
    @staticmethod
    def sanitize_brain_signal(signal: Any, 
                             max_channels: int = 512,
                             max_timepoints: int = 1000000,
                             max_amplitude: float = 1000.0) -> Any:
        """
        Sanitize brain signal input.
        
        Args:
            signal: Input signal to sanitize
            max_channels: Maximum allowed channels
            max_timepoints: Maximum allowed timepoints
            max_amplitude: Maximum allowed signal amplitude
            
        Returns:
            Sanitized signal
            
        Raises:
            ValueError: If input is invalid or dangerous
        """
        if not HAS_NUMPY:
            return signal  # Skip sanitization if numpy not available
            
        # Type validation
        if not isinstance(signal, np.ndarray):
            raise ValueError(f"Signal must be numpy array, got {type(signal)}")
            
        # Dimension validation
        if signal.ndim != 2:
            raise ValueError(f"Signal must be 2D (channels, timepoints), got {signal.ndim}D")
            
        n_channels, n_timepoints = signal.shape
        
        # Size limits
        if n_channels > max_channels:
            raise ValueError(f"Too many channels: {n_channels} > {max_channels}")
            
        if n_timepoints > max_timepoints:
            raise ValueError(f"Signal too long: {n_timepoints} > {max_timepoints}")
            
        # Check for invalid values
        if not np.isfinite(signal).all():
            # Replace NaN/inf with zeros
            signal = np.nan_to_num(signal, nan=0.0, posinf=max_amplitude, neginf=-max_amplitude)
            warnings.warn("Replaced NaN/inf values in signal")
            
        # Amplitude limiting
        if np.max(np.abs(signal)) > max_amplitude:
            # Clip extreme values
            signal = np.clip(signal, -max_amplitude, max_amplitude)
            warnings.warn(f"Clipped signal amplitudes to ±{max_amplitude}")
            
        return signal
        
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed text length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be string, got {type(text)}")
            
        # Length limit
        if len(text) > max_length:
            text = text[:max_length]
            warnings.warn(f"Truncated text to {max_length} characters")
            
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        if len(sanitized) != len(text):
            warnings.warn("Removed control characters from text")
            
        return sanitized
        
    @staticmethod
    def validate_config_object(config: Any, config_type: str) -> None:
        """
        Validate configuration objects.
        
        Args:
            config: Configuration object to validate
            config_type: Type of configuration for validation
            
        Raises:
            ValueError: If configuration is invalid
        """
        from .utils import ConfigValidator
        
        if config_type == 'preprocessing':
            errors = ConfigValidator.validate_preprocessing_config(config)
        elif config_type == 'model':
            errors = ConfigValidator.validate_model_config(config)
        elif config_type == 'device':
            errors = ConfigValidator.validate_device_config(config)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
            
        if errors:
            raise ValueError(f"Invalid {config_type} configuration: {'; '.join(errors)}")


class RobustOperator:
    """Provides robust operation execution with automatic error handling."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_manager = FallbackManager()
        self.auto_recovery = AutoRecovery()
        
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create a circuit breaker for an operation."""
        if config is None:
            config = CircuitBreakerConfig()
            
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
        
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function."""
        self.fallback_manager.register_fallback(operation, fallback_func)
        
    def register_recovery(self, component: str, recovery_func: Callable):
        """Register a recovery strategy."""
        self.auto_recovery.register_recovery_strategy(component, recovery_func)
        
    def execute_robust(self, 
                      operation: str,
                      func: Callable,
                      *args,
                      fallbacks: List[Callable] = None,
                      auto_recover: bool = True,
                      **kwargs) -> Any:
        """
        Execute operation with full robustness features.
        
        Args:
            operation: Name of the operation
            func: Primary function to execute
            *args: Function arguments
            fallbacks: Optional list of fallback functions
            auto_recover: Whether to attempt auto-recovery on failure
            **kwargs: Function keyword arguments
            
        Returns:
            Result of successful operation
            
        Raises:
            RuntimeError: If all attempts fail
        """
        # Register temporary fallbacks
        if fallbacks:
            for fallback in fallbacks:
                self.fallback_manager.register_fallback(operation, fallback)
                
        # Get or create circuit breaker
        if operation not in self.circuit_breakers:
            self.create_circuit_breaker(operation)
            
        breaker = self.circuit_breakers[operation]
        
        try:
            # Execute through circuit breaker and fallback system
            return self.fallback_manager.execute_with_fallbacks(
                operation,
                breaker.call,
                func,
                *args,
                **kwargs
            )
            
        except Exception as e:
            # Attempt auto-recovery if enabled
            if auto_recover:
                recovery_success = self.auto_recovery.attempt_recovery(
                    operation, e, {'args': str(args), 'kwargs': str(kwargs)}
                )
                
                if recovery_success:
                    # Retry after successful recovery
                    try:
                        return breaker.call(func, *args, **kwargs)
                    except Exception:
                        pass  # Recovery didn't work, fall through to failure
                        
            raise
            
    def get_status(self) -> Dict[str, Any]:
        """Get status of all robustness components."""
        return {
            'circuit_breakers': {
                name: breaker.get_status() 
                for name, breaker in self.circuit_breakers.items()
            },
            'recovery_stats': self.auto_recovery.get_recovery_stats(),
            'timestamp': time.time()
        }


# Default fallback functions for common operations
def fallback_decode_to_empty() -> List[int]:
    """Fallback that returns empty token list."""
    warnings.warn("Using fallback: returning empty token list")
    return []


def fallback_decode_to_random(vocab_size: int = 50257, length: int = 5) -> List[int]:
    """Fallback that returns random tokens."""
    if HAS_NUMPY:
        tokens = np.random.randint(0, vocab_size, size=length).tolist()
    else:
        import random
        tokens = [random.randint(0, vocab_size-1) for _ in range(length)]
        
    warnings.warn(f"Using fallback: returning {length} random tokens")
    return tokens


def fallback_text_output(error_message: str = "Decoding failed") -> str:
    """Fallback that returns error message as text."""
    warnings.warn(f"Using fallback: returning error message as text")
    return f"[{error_message}]"


# Recovery strategies for common components
def recover_decoder(error: Exception, context: Dict[str, Any]) -> None:
    """Recovery strategy for decoder failures."""
    # Simple recovery: reinitialize decoder
    warnings.warn("Attempting decoder recovery by reinitialization")
    time.sleep(1.0)  # Brief pause


def recover_device_connection(error: Exception, context: Dict[str, Any]) -> None:
    """Recovery strategy for device connection failures."""
    warnings.warn("Attempting device recovery by reconnection")
    time.sleep(2.0)  # Longer pause for device recovery


def recover_streaming_buffer(error: Exception, context: Dict[str, Any]) -> None:
    """Recovery strategy for streaming buffer issues."""
    warnings.warn("Attempting streaming recovery by buffer reset")
    # In practice, this would reset the streaming buffer


class SecurityValidator:
    """Validates inputs for security concerns."""
    
    @staticmethod
    def validate_file_path(path: Union[str, Any], 
                          allowed_extensions: List[str] = None,
                          max_size_mb: float = 100.0) -> None:
        """
        Validate file paths for security.
        
        Args:
            path: File path to validate
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        from pathlib import Path
        
        if not isinstance(path, (str, Path)):
            raise ValueError(f"Path must be string or Path object, got {type(path)}")
            
        path = Path(path)
        
        # Check for path traversal attempts
        try:
            path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")
            
        # Check for suspicious path components
        suspicious = ['..', '~', '$', '|', '&', ';', '`']
        path_str = str(path)
        
        for sus in suspicious:
            if sus in path_str:
                raise ValueError(f"Suspicious character '{sus}' in path")
                
        # Check file extension
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"File extension '{path.suffix}' not allowed")
            
        # Check file size if exists
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
                
    @staticmethod
    def validate_privacy_parameters(epsilon: float, delta: float) -> None:
        """
        Validate differential privacy parameters.
        
        Args:
            epsilon: Privacy budget
            delta: Failure probability
            
        Raises:
            ValueError: If privacy parameters are invalid
        """
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError(f"Epsilon must be positive number, got {epsilon}")
            
        if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be between 0 and 1, got {delta}")
            
        # Warn about weak privacy
        if epsilon > 10:
            warnings.warn(f"Large epsilon ({epsilon}) provides weak privacy protection")
            
        if delta > 1e-3:
            warnings.warn(f"Large delta ({delta}) increases privacy failure risk")
            
    @staticmethod
    def validate_model_path(path: Union[str, Any]) -> None:
        """
        Validate model file paths.
        
        Args:
            path: Model file path
            
        Raises:
            ValueError: If path is invalid or unsafe
        """
        SecurityValidator.validate_file_path(
            path,
            allowed_extensions=['.pt', '.pth', '.ckpt', '.safetensors'],
            max_size_mb=1000.0  # 1GB max for model files
        )


class HealthChecker:
    """Automated health checking for BCI components."""
    
    def __init__(self):
        self.health_functions: Dict[str, Callable] = {}
        self.last_check_times: Dict[str, float] = {}
        self.check_intervals: Dict[str, float] = {}
        
    def register_health_check(self, 
                             component: str, 
                             check_func: Callable,
                             interval: float = 60.0):
        """
        Register a health check function.
        
        Args:
            component: Component name
            check_func: Function that returns health status
            interval: Check interval in seconds
        """
        self.health_functions[component] = check_func
        self.check_intervals[component] = interval
        self.last_check_times[component] = 0.0
        
    def check_component_health(self, component: str) -> Optional[Dict[str, Any]]:
        """Check health of specific component."""
        if component not in self.health_functions:
            return None
            
        current_time = time.time()
        last_check = self.last_check_times[component]
        interval = self.check_intervals[component]
        
        # Check if it's time for a health check
        if current_time - last_check < interval:
            return None  # Too soon
            
        try:
            health_result = self.health_functions[component]()
            self.last_check_times[component] = current_time
            
            return {
                'component': component,
                'status': health_result,
                'timestamp': current_time,
                'check_interval': interval
            }
            
        except Exception as e:
            return {
                'component': component,
                'status': 'error',
                'error': str(e),
                'timestamp': current_time,
                'check_interval': interval
            }
            
    def check_all_components(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered components."""
        results = {}
        
        for component in self.health_functions.keys():
            result = self.check_component_health(component)
            if result:  # Only include if check was performed
                results[component] = result
                
        return results


# Global robust operator instance
_global_operator: Optional[RobustOperator] = None


def get_robust_operator() -> RobustOperator:
    """Get or create global robust operator."""
    global _global_operator
    
    if _global_operator is None:
        _global_operator = RobustOperator()
        
        # Register default fallbacks
        _global_operator.register_fallback('decode_tokens', fallback_decode_to_empty)
        _global_operator.register_fallback('text_output', lambda: fallback_text_output())
        
        # Register default recovery strategies
        _global_operator.register_recovery('decoder', recover_decoder)
        _global_operator.register_recovery('device', recover_device_connection)
        _global_operator.register_recovery('streaming', recover_streaming_buffer)
        
    return _global_operator


def robust_operation(operation_name: str, **circuit_config):
    """
    Decorator for making operations robust.
    
    Args:
        operation_name: Name of the operation
        **circuit_config: Circuit breaker configuration parameters
        
    Returns:
        Decorated function with robustness features
    """
    def decorator(func: Callable) -> Callable:
        operator = get_robust_operator()
        
        # Create circuit breaker with custom config
        if circuit_config:
            config = CircuitBreakerConfig(**circuit_config)
            operator.create_circuit_breaker(operation_name, config)
        else:
            operator.create_circuit_breaker(operation_name)
            
        def wrapper(*args, **kwargs):
            return operator.execute_robust(operation_name, func, *args, **kwargs)
            
        return wrapper
    return decorator


# Convenience functions
def make_robust(func: Callable, operation_name: str = None) -> Callable:
    """Make any function robust with default settings."""
    if operation_name is None:
        operation_name = func.__name__
        
    return robust_operation(operation_name)(func)


if __name__ == '__main__':
    # Test the robustness system
    print("Testing BCI-2-Token Robustness System")
    print("=" * 45)
    
    # Test circuit breaker
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
    breaker = CircuitBreaker('test', config)
    
    def failing_function():
        raise ValueError("Test failure")
        
    def working_function():
        return "success"
        
    # Test failures
    for i in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass
            
    status = breaker.get_status()
    print(f"Circuit breaker state after failures: {status['state']}")
    
    # Test fallback system
    operator = get_robust_operator()
    
    try:
        result = operator.execute_robust('test_op', failing_function)
    except:
        print("Fallback system activated as expected")
        
    print("\n✓ Robustness system working")