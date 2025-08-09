"""
Comprehensive Validation and Error Handling for BCI-2-Token

This module provides advanced validation, error handling, and input sanitization
to ensure robust operation in production environments.
"""

import numpy as np
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import time


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = None
    suggestions: List[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


class SignalValidator:
    """Comprehensive signal validation"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
    
    def validate_eeg_signal(self, signals: np.ndarray, 
                          sampling_rate: float = 256.0,
                          expected_channels: Optional[int] = None) -> ValidationResult:
        """Validate EEG signal data"""
        
        # Basic shape validation
        if signals.ndim < 2:
            return ValidationResult(
                is_valid=False,
                error_message="EEG signals must have at least 2 dimensions (channels, timepoints)",
                severity=ErrorSeverity.CRITICAL
            )
        
        warnings = []
        suggestions = []
        
        # Get signal dimensions
        if signals.ndim == 2:
            n_channels, n_timepoints = signals.shape
            n_epochs = 1
        elif signals.ndim == 3:
            n_epochs, n_channels, n_timepoints = signals.shape
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported signal shape: {signals.shape}. Expected 2D or 3D array.",
                severity=ErrorSeverity.CRITICAL
            )
        
        # Channel count validation
        if expected_channels and n_channels != expected_channels:
            if self.validation_level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Channel mismatch: expected {expected_channels}, got {n_channels}",
                    severity=ErrorSeverity.HIGH
                )
            else:
                warnings.append(f"Channel count mismatch: expected {expected_channels}, got {n_channels}")
        
        # Common EEG channel counts
        common_channel_counts = [1, 8, 16, 19, 32, 64, 128, 256]
        if n_channels not in common_channel_counts:
            warnings.append(f"Unusual channel count: {n_channels}. Common counts: {common_channel_counts}")
        
        # Time points validation
        min_timepoints = int(sampling_rate * 0.1)  # 100ms minimum
        max_timepoints = int(sampling_rate * 60)   # 60s maximum
        
        if n_timepoints < min_timepoints:
            return ValidationResult(
                is_valid=False,
                error_message=f"Too few timepoints: {n_timepoints}. Minimum: {min_timepoints}",
                severity=ErrorSeverity.HIGH
            )
        
        if n_timepoints > max_timepoints:
            warnings.append(f"Very long signal: {n_timepoints} timepoints ({n_timepoints/sampling_rate:.1f}s)")
            suggestions.append("Consider segmenting long signals for better processing")
        
        # Data type and range validation
        if not np.issubdtype(signals.dtype, np.floating):
            warnings.append(f"Non-float data type: {signals.dtype}. Consider converting to float32/float64")
        
        # Check for common EEG voltage ranges (microvolts)
        signal_range = np.max(signals) - np.min(signals)
        if signal_range > 1000:  # > 1000 µV
            warnings.append(f"Unusually large signal range: {signal_range:.1f}. Check units (should be µV)")
        elif signal_range < 0.1:  # < 0.1 µV
            warnings.append(f"Unusually small signal range: {signal_range:.3f}. Check scaling")
        
        # Check for infinite or NaN values
        if not np.isfinite(signals).all():
            nan_count = np.sum(np.isnan(signals))
            inf_count = np.sum(np.isinf(signals))
            
            if self.validation_level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid values found: {nan_count} NaN, {inf_count} Inf",
                    severity=ErrorSeverity.CRITICAL
                )
            else:
                warnings.append(f"Invalid values found: {nan_count} NaN, {inf_count} Inf")
                suggestions.append("Consider preprocessing to handle invalid values")
        
        # Signal quality checks
        quality_warnings = self._check_signal_quality(signals, sampling_rate)
        warnings.extend(quality_warnings)
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            suggestions=suggestions,
            severity=ErrorSeverity.LOW if not warnings else ErrorSeverity.MEDIUM
        )
    
    def _check_signal_quality(self, signals: np.ndarray, sampling_rate: float) -> List[str]:
        """Check various signal quality metrics"""
        warnings = []
        
        # Check for flat channels (no variation)
        if signals.ndim == 3:
            signals_2d = signals.reshape(-1, signals.shape[-1])
        else:
            signals_2d = signals
        
        channel_variances = np.var(signals_2d, axis=-1)
        flat_channels = np.sum(channel_variances < 1e-10)
        
        if flat_channels > 0:
            warnings.append(f"{flat_channels} channels appear flat (no variation)")
        
        # Check for saturation (constant values at extremes)
        for ch_idx in range(signals_2d.shape[0]):
            channel_data = signals_2d[ch_idx]
            unique_values = len(np.unique(channel_data))
            
            if unique_values < 10:
                warnings.append(f"Channel {ch_idx} has very few unique values ({unique_values})")
        
        # Power spectrum analysis for artifacts
        try:
            # Simple frequency domain check
            fft_data = np.fft.fft(signals_2d, axis=-1)
            power_spectrum = np.abs(fft_data) ** 2
            freqs = np.fft.fftfreq(signals_2d.shape[-1], 1/sampling_rate)
            
            # Check for line noise (50/60 Hz)
            line_noise_freqs = [50, 60]  # European/American line noise
            for freq in line_noise_freqs:
                freq_idx = np.argmin(np.abs(freqs - freq))
                if freq_idx < len(freqs) // 2:  # Only check positive frequencies
                    line_power = np.mean(power_spectrum[:, freq_idx])
                    total_power = np.mean(power_spectrum[:, :len(freqs)//2])
                    
                    if line_power > 0.1 * total_power:
                        warnings.append(f"Potential {freq}Hz line noise detected")
            
        except Exception:
            # Skip frequency analysis if it fails
            pass
        
        return warnings
    
    def validate_model_input(self, features: np.ndarray, 
                           expected_shape: Optional[Tuple[int, ...]] = None) -> ValidationResult:
        """Validate model input features"""
        
        if features.size == 0:
            return ValidationResult(
                is_valid=False,
                error_message="Empty feature array",
                severity=ErrorSeverity.CRITICAL
            )
        
        warnings = []
        suggestions = []
        
        # Shape validation
        if expected_shape and features.shape != expected_shape:
            if self.validation_level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Shape mismatch: expected {expected_shape}, got {features.shape}",
                    severity=ErrorSeverity.HIGH
                )
            else:
                warnings.append(f"Shape mismatch: expected {expected_shape}, got {features.shape}")
        
        # Check for valid numeric data
        if not np.isfinite(features).all():
            nan_count = np.sum(np.isnan(features))
            inf_count = np.sum(np.isinf(features))
            
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid numeric values: {nan_count} NaN, {inf_count} Inf",
                severity=ErrorSeverity.CRITICAL
            )
        
        # Check feature range
        feature_range = np.max(features) - np.min(features)
        if feature_range == 0:
            warnings.append("All features have identical values")
            suggestions.append("Check feature extraction process")
        
        # Check for extreme values
        feature_std = np.std(features)
        feature_mean = np.mean(features)
        extreme_threshold = 5 * feature_std
        
        if np.any(np.abs(features - feature_mean) > extreme_threshold):
            extreme_count = np.sum(np.abs(features - feature_mean) > extreme_threshold)
            warnings.append(f"{extreme_count} extreme outlier values detected")
            suggestions.append("Consider outlier removal or clipping")
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            suggestions=suggestions
        )


class RobustErrorHandler:
    """Robust error handling with recovery mechanisms"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
    
    def with_retry(self, operation_name: str = "operation"):
        """Decorator for operations that should be retried on failure"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        last_exception = e
                        self._log_error(operation_name, e, attempt)
                        
                        if attempt < self.max_retries:
                            self.logger.info(f"Retrying {operation_name} (attempt {attempt + 2}/{self.max_retries + 1})")
                            time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            self._increment_error_count(operation_name)
                            self.logger.error(f"Operation {operation_name} failed after {self.max_retries + 1} attempts")
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, fallback_value: Any = None, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """Safely execute a function with error handling"""
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            self._log_error(func.__name__, e, 0)
            return fallback_value, e
    
    def _log_error(self, operation: str, error: Exception, attempt: int):
        """Log error details"""
        self.logger.error(f"Error in {operation} (attempt {attempt + 1}): {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def _increment_error_count(self, operation: str):
        """Track error counts for monitoring"""
        if operation not in self.error_counts:
            self.error_counts[operation] = 0
        self.error_counts[operation] += 1
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error count statistics"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error count statistics"""
        self.error_counts.clear()


class InputSanitizer:
    """Input sanitization for security and robustness"""
    
    @staticmethod
    def sanitize_signal_data(signals: np.ndarray, 
                           clip_range: Optional[Tuple[float, float]] = None,
                           remove_outliers: bool = True,
                           outlier_threshold: float = 5.0) -> np.ndarray:
        """Sanitize signal data by clipping and removing outliers"""
        
        # Convert to float for processing
        sanitized = signals.astype(np.float64)
        
        # Remove NaN and Inf values
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip to specified range
        if clip_range:
            sanitized = np.clip(sanitized, clip_range[0], clip_range[1])
        
        # Remove statistical outliers
        if remove_outliers:
            mean_val = np.mean(sanitized)
            std_val = np.std(sanitized)
            
            lower_bound = mean_val - outlier_threshold * std_val
            upper_bound = mean_val + outlier_threshold * std_val
            
            sanitized = np.clip(sanitized, lower_bound, upper_bound)
        
        return sanitized.astype(signals.dtype)
    
    @staticmethod
    def validate_parameter_range(value: Union[int, float], 
                                param_name: str,
                                min_val: Optional[float] = None,
                                max_val: Optional[float] = None,
                                allowed_values: Optional[List[Union[int, float]]] = None) -> Union[int, float]:
        """Validate and sanitize parameter values"""
        
        if allowed_values and value not in allowed_values:
            raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def sanitize_string_input(text: str, max_length: int = 1000, 
                            allowed_chars: Optional[str] = None) -> str:
        """Sanitize string input"""
        
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Limit length
        sanitized = text[:max_length]
        
        # Filter allowed characters
        if allowed_chars:
            sanitized = ''.join(c for c in sanitized if c in allowed_chars)
        
        return sanitized


# CircuitBreaker moved to reliability.py for consistency
from .reliability import CircuitBreaker, CircuitBreakerConfig


def validate_bci_pipeline_input(signals: np.ndarray, 
                               config: Dict[str, Any],
                               validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Comprehensive validation for BCI pipeline input"""
    
    validator = SignalValidator(validation_level)
    
    # Validate signals
    signal_result = validator.validate_eeg_signal(
        signals, 
        sampling_rate=config.get('sampling_rate', 256.0),
        expected_channels=config.get('expected_channels')
    )
    
    if not signal_result.is_valid:
        return signal_result
    
    # Additional pipeline-specific validations
    warnings = signal_result.warnings.copy()
    suggestions = signal_result.suggestions.copy()
    
    # Validate configuration
    required_config_keys = ['sampling_rate', 'preprocessing']
    for key in required_config_keys:
        if key not in config:
            warnings.append(f"Missing required config key: {key}")
    
    # Validate preprocessing config
    if 'preprocessing' in config:
        preproc_config = config['preprocessing']
        
        if not isinstance(preproc_config, dict):
            return ValidationResult(
                is_valid=False,
                error_message="Preprocessing config must be a dictionary",
                severity=ErrorSeverity.HIGH
            )
        
        # Check for reasonable preprocessing parameters
        if 'lowpass' in preproc_config:
            lowpass = preproc_config['lowpass']
            if lowpass > config.get('sampling_rate', 256) / 2:
                warnings.append("Lowpass frequency exceeds Nyquist rate")
    
    return ValidationResult(
        is_valid=True,
        warnings=warnings,
        suggestions=suggestions,
        severity=ErrorSeverity.LOW if not warnings else ErrorSeverity.MEDIUM
    )


# Example usage demonstration
def demo_validation_system():
    """Demonstrate comprehensive validation system"""
    print("=== BCI-2-Token Validation System Demo ===\n")
    
    # 1. Signal Validation Demo
    print("1. Signal Validation")
    
    # Generate test signals
    np.random.seed(42)
    good_signals = np.random.randn(64, 512) * 10  # Normal EEG-like signals
    bad_signals = np.full((64, 512), np.nan)      # Signals with NaN values
    
    validator = SignalValidator(ValidationLevel.MODERATE)
    
    # Test good signals
    result_good = validator.validate_eeg_signal(good_signals, sampling_rate=256)
    print(f"  Good signals: Valid={result_good.is_valid}, Warnings={len(result_good.warnings)}")
    
    # Test bad signals
    result_bad = validator.validate_eeg_signal(bad_signals, sampling_rate=256)
    print(f"  Bad signals: Valid={result_bad.is_valid}, Error: {result_bad.error_message}")
    
    # 2. Error Handling Demo
    print("\n2. Robust Error Handling")
    
    error_handler = RobustErrorHandler(max_retries=2, retry_delay=0.1)
    
    # Function that sometimes fails
    def unreliable_function(success_rate=0.3):
        if np.random.random() > success_rate:
            raise RuntimeError("Simulated processing failure")
        return "Success!"
    
    # Test with retry decorator
    @error_handler.with_retry("processing")
    def robust_processing():
        return unreliable_function(success_rate=0.8)
    
    try:
        result = robust_processing()
        print(f"  Robust processing: {result}")
    except Exception as e:
        print(f"  Robust processing failed: {e}")
    
    # 3. Input Sanitization Demo
    print("\n3. Input Sanitization")
    
    # Create signals with outliers and NaN values
    dirty_signals = np.random.randn(8, 128)
    dirty_signals[0, 50] = 1000  # Outlier
    dirty_signals[1, 60] = np.nan  # NaN value
    dirty_signals[2, 70] = np.inf  # Inf value
    
    clean_signals = InputSanitizer.sanitize_signal_data(
        dirty_signals, 
        clip_range=(-100, 100),
        remove_outliers=True
    )
    
    print(f"  Before sanitization: NaN count = {np.sum(np.isnan(dirty_signals))}")
    print(f"  After sanitization: NaN count = {np.sum(np.isnan(clean_signals))}")
    print(f"  Signal range: {np.min(clean_signals):.2f} to {np.max(clean_signals):.2f}")
    
    # 4. Circuit Breaker Demo
    print("\n4. Circuit Breaker Pattern")
    
    from .reliability import CircuitBreakerConfig
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
    circuit_breaker = CircuitBreaker("test_service", config)
    
    def failing_service():
        if np.random.random() > 0.7:  # 30% success rate
            return "Service response"
        raise RuntimeError("Service failure")
    
    # Test circuit breaker
    for i in range(8):
        try:
            result = circuit_breaker.call(failing_service)
            print(f"  Call {i+1}: Success - {result}")
        except RuntimeError as e:
            state = circuit_breaker.get_state()
            print(f"  Call {i+1}: Failed - {e} (Circuit: {state['state']})")
        except Exception as e:
            print(f"  Call {i+1}: Error - {e}")
    
    print("\n=== Validation System Demo Complete ===")


if __name__ == "__main__":
    demo_validation_system()