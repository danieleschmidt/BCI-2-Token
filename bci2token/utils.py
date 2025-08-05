"""
Utility functions and error handling for BCI-2-Token framework.

Provides common utilities, validation functions, and error handling
for brain-computer interface applications.
"""

import functools
import time
import warnings
import sys
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class BCIError(Exception):
    """Base exception for BCI-2-Token framework."""
    pass


class SignalProcessingError(BCIError):
    """Error in signal processing pipeline."""
    pass


class ModelLoadError(BCIError):
    """Error loading or initializing models."""
    pass


class DeviceError(BCIError):
    """Error with device communication."""
    pass


class PrivacyError(BCIError):
    """Error in privacy protection mechanisms."""
    pass


def validate_signal_shape(signal: Any, 
                         expected_channels: int,
                         min_timepoints: int = 1) -> None:
    """
    Validate brain signal array shape.
    
    Args:
        signal: Signal array to validate
        expected_channels: Expected number of channels
        min_timepoints: Minimum number of timepoints
        
    Raises:
        SignalProcessingError: If signal shape is invalid
    """
    if not HAS_NUMPY:
        return  # Skip validation if numpy not available
        
    if not isinstance(signal, np.ndarray):
        raise SignalProcessingError(f"Signal must be numpy array, got {type(signal)}")
        
    if signal.ndim != 2:
        raise SignalProcessingError(f"Signal must be 2D (channels, timepoints), got {signal.ndim}D")
        
    n_channels, n_timepoints = signal.shape
    
    if n_channels != expected_channels:
        raise SignalProcessingError(
            f"Expected {expected_channels} channels, got {n_channels}"
        )
        
    if n_timepoints < min_timepoints:
        raise SignalProcessingError(
            f"Signal too short: {n_timepoints} timepoints < {min_timepoints} minimum"
        )


def validate_sampling_rate(sampling_rate: float) -> None:
    """
    Validate sampling rate.
    
    Args:
        sampling_rate: Sampling rate in Hz
        
    Raises:
        SignalProcessingError: If sampling rate is invalid
    """
    if not isinstance(sampling_rate, (int, float)):
        raise SignalProcessingError(f"Sampling rate must be numeric, got {type(sampling_rate)}")
        
    if sampling_rate <= 0:
        raise SignalProcessingError(f"Sampling rate must be positive, got {sampling_rate}")
        
    if sampling_rate < 50:
        warnings.warn(f"Low sampling rate ({sampling_rate} Hz) may affect decoding quality")
        
    if sampling_rate > 10000:
        warnings.warn(f"Very high sampling rate ({sampling_rate} Hz) may be unnecessary")


def validate_frequency_bands(lowpass: float, 
                           highpass: float, 
                           sampling_rate: float) -> None:
    """
    Validate frequency band parameters.
    
    Args:
        lowpass: Lowpass frequency in Hz
        highpass: Highpass frequency in Hz  
        sampling_rate: Sampling rate in Hz
        
    Raises:
        SignalProcessingError: If frequency parameters are invalid
    """
    if highpass <= 0:
        raise SignalProcessingError(f"Highpass frequency must be positive, got {highpass}")
        
    if lowpass <= highpass:
        raise SignalProcessingError(
            f"Lowpass frequency ({lowpass}) must be greater than highpass ({highpass})"
        )
        
    nyquist = sampling_rate / 2
    if lowpass >= nyquist:
        raise SignalProcessingError(
            f"Lowpass frequency ({lowpass}) must be less than Nyquist frequency ({nyquist})"
        )


def retry_on_error(max_attempts: int = 3, 
                  delay: float = 1.0,
                  exponential_backoff: bool = True,
                  exceptions: Tuple[Exception, ...] = (Exception,)):
    """
    Decorator for retrying functions on error.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between attempts
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # Don't sleep on last attempt
                        time.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                            
            # All attempts failed
            raise last_exception
            
        return wrapper
    return decorator


def timeout_after(seconds: float):
    """
    Decorator to add timeout to functions.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
                
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        return wrapper
    return decorator


def measure_performance(metric_name: str, monitor=None):
    """
    Decorator to measure function performance.
    
    Args:
        metric_name: Name of the metric to record
        monitor: Optional monitor instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = e
                
            end_time = time.time()
            duration = end_time - start_time
            
            # Record metrics
            if monitor and hasattr(monitor, 'metrics') and monitor.metrics:
                monitor.metrics.record_metric(f'{metric_name}_duration', duration)
                monitor.metrics.record_metric(
                    f'{metric_name}_success',
                    1.0 if success else 0.0
                )
                
            # Log performance
            if monitor and hasattr(monitor, 'logger'):
                if success:
                    monitor.logger.debug(
                        'Performance',
                        f'{func.__name__} completed in {duration:.3f}s',
                        {'function': func.__name__, 'duration': duration}
                    )
                else:
                    monitor.logger.error(
                        'Performance',
                        f'{func.__name__} failed after {duration:.3f}s: {error}',
                        {'function': func.__name__, 'duration': duration, 'error': str(error)}
                    )
                    
            if not success:
                raise error
                
            return result
            
        return wrapper
    return decorator


def validate_config_dict(config_dict: Dict[str, Any], 
                        required_keys: List[str],
                        optional_keys: List[str] = None,
                        value_validators: Dict[str, Callable] = None) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config_dict: Configuration to validate
        required_keys: List of required keys
        optional_keys: List of optional keys  
        value_validators: Dict mapping keys to validation functions
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
    # Check for unknown keys
    all_valid_keys = set(required_keys)
    if optional_keys:
        all_valid_keys.update(optional_keys)
        
    unknown_keys = [key for key in config_dict.keys() if key not in all_valid_keys]
    if unknown_keys:
        warnings.warn(f"Unknown configuration keys (will be ignored): {unknown_keys}")
        
    # Validate values
    if value_validators:
        for key, validator in value_validators.items():
            if key in config_dict:
                try:
                    validator(config_dict[key])
                except Exception as e:
                    raise ValueError(f"Invalid value for '{key}': {e}")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if b == 0:
            return default
        return a / b
    except (TypeError, ValueError):
        return default


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file with error handling."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a JSON object")
            
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_json_config(config: Dict[str, Any], 
                    config_path: Union[str, Path],
                    indent: int = 2) -> None:
    """Save configuration to JSON file."""
    config_path = Path(config_path)
    
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=indent, default=str)
    except Exception as e:
        raise IOError(f"Failed to save configuration: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_count: int) -> str:
    """Format byte count in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}PB"


def calculate_signal_quality(signal: Any) -> float:
    """
    Calculate basic signal quality score.
    
    Args:
        signal: Brain signal array
        
    Returns:
        Quality score between 0 and 1
    """
    if not HAS_NUMPY:
        return 0.5  # Default score if numpy not available
        
    try:
        if not isinstance(signal, np.ndarray):
            return 0.0
            
        # Check for obvious issues
        if signal.size == 0:
            return 0.0
            
        # Check for NaN or infinite values
        if not np.isfinite(signal).all():
            return 0.0
            
        # Calculate signal-to-noise ratio estimate
        signal_power = np.var(signal)
        
        # Estimate noise as high-frequency component
        if signal.shape[-1] > 10:
            # Simple high-frequency noise estimate
            diff_signal = np.diff(signal, axis=-1)
            noise_power = np.var(diff_signal)
            
            snr = safe_divide(signal_power, noise_power, default=1.0)
            
            # Convert SNR to quality score (0-1)
            quality = min(1.0, snr / 10.0)  # SNR of 10 = quality 1.0
        else:
            quality = 0.5  # Default for short signals
            
        return max(0.0, min(1.0, quality))
        
    except Exception:
        return 0.0


def detect_signal_artifacts(signal: Any) -> Dict[str, bool]:
    """
    Detect common artifacts in brain signals.
    
    Args:
        signal: Brain signal array (channels, timepoints)
        
    Returns:
        Dictionary with artifact detection results
    """
    artifacts = {
        'flat_channels': False,
        'high_amplitude': False,
        'line_noise': False,
        'muscle_artifacts': False
    }
    
    if not HAS_NUMPY or not isinstance(signal, np.ndarray):
        return artifacts
        
    try:
        # Check for flat channels (no variation)
        channel_stds = np.std(signal, axis=1)
        artifacts['flat_channels'] = np.any(channel_stds < 1e-6)
        
        # Check for high amplitude (> 200 µV)
        max_amplitude = np.max(np.abs(signal))
        artifacts['high_amplitude'] = max_amplitude > 200.0
        
        # Simple line noise detection (60 Hz and harmonics)
        # This is a very basic implementation
        if signal.shape[1] > 128:  # Need enough samples for FFT
            from scipy import signal as sp_signal
            
            # Estimate sampling rate from signal length (rough)
            estimated_fs = 256  # Default assumption
            
            freqs, psd = sp_signal.welch(signal[0], fs=estimated_fs, nperseg=128)
            
            # Check for peaks at 60 Hz and 120 Hz
            freq_60_idx = np.argmin(np.abs(freqs - 60))
            freq_120_idx = np.argmin(np.abs(freqs - 120))
            
            if len(psd) > max(freq_60_idx, freq_120_idx):
                line_noise_power = psd[freq_60_idx] + psd[freq_120_idx]
                total_power = np.mean(psd)
                artifacts['line_noise'] = line_noise_power > 0.1 * total_power
                
        # Simple muscle artifact detection (high frequency content)
        if signal.shape[1] > 10:
            # High frequency variance as proxy for muscle artifacts
            diff_signal = np.diff(signal, axis=1)
            high_freq_var = np.var(diff_signal, axis=1)
            signal_var = np.var(signal, axis=1)
            
            # If high frequency variance is large relative to signal
            relative_high_freq = np.mean(high_freq_var / (signal_var + 1e-6))
            artifacts['muscle_artifacts'] = relative_high_freq > 0.5
            
    except Exception:
        # If any detection fails, return defaults
        pass
        
    return artifacts


def create_signal_summary(signal: Any) -> Dict[str, Any]:
    """
    Create comprehensive summary of brain signal.
    
    Args:
        signal: Brain signal array
        
    Returns:
        Dictionary with signal statistics and properties
    """
    summary = {
        'shape': None,
        'dtype': None,
        'quality_score': 0.0,
        'artifacts': {},
        'statistics': {}
    }
    
    if not HAS_NUMPY:
        return summary
        
    try:
        if isinstance(signal, np.ndarray):
            summary['shape'] = signal.shape
            summary['dtype'] = str(signal.dtype)
            
            # Basic statistics
            summary['statistics'] = {
                'mean': float(np.mean(signal)),
                'std': float(np.std(signal)),
                'min': float(np.min(signal)),
                'max': float(np.max(signal)),
                'range': float(np.ptp(signal))  # peak-to-peak
            }
            
            # Quality assessment
            summary['quality_score'] = calculate_signal_quality(signal)
            
            # Artifact detection
            summary['artifacts'] = detect_signal_artifacts(signal)
            
            # Channel-wise statistics
            if signal.ndim == 2:
                channel_stats = []
                for ch in range(signal.shape[0]):
                    ch_stats = {
                        'mean': float(np.mean(signal[ch])),
                        'std': float(np.std(signal[ch])),
                        'quality': calculate_signal_quality(signal[ch:ch+1])
                    }
                    channel_stats.append(ch_stats)
                summary['channel_statistics'] = channel_stats
                
    except Exception as e:
        summary['error'] = str(e)
        
    return summary


class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, name: str, monitor=None):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.monitor:
            # Record to monitor
            if hasattr(self.monitor, 'metrics') and self.monitor.metrics:
                self.monitor.metrics.record_metric(f'{self.name}_duration', duration)
                
            if hasattr(self.monitor, 'logger'):
                self.monitor.logger.debug(
                    'Performance',
                    f'{self.name} took {duration:.3f}s',
                    {'operation': self.name, 'duration': duration}
                )
                
    @property
    def duration(self) -> Optional[float]:
        """Get measured duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ConfigValidator:
    """Validates configuration objects."""
    
    @staticmethod
    def validate_preprocessing_config(config) -> List[str]:
        """Validate preprocessing configuration."""
        errors = []
        
        try:
            # Check sampling rate
            if hasattr(config, 'sampling_rate'):
                validate_sampling_rate(config.sampling_rate)
                
            # Check frequency bands
            if all(hasattr(config, attr) for attr in ['lowpass_freq', 'highpass_freq', 'sampling_rate']):
                validate_frequency_bands(
                    config.lowpass_freq,
                    config.highpass_freq, 
                    config.sampling_rate
                )
                
            # Check window parameters
            if hasattr(config, 'window_size') and config.window_size <= 0:
                errors.append("Window size must be positive")
                
            if hasattr(config, 'overlap'):
                if not (0 <= config.overlap < 1):
                    errors.append("Overlap must be between 0 and 1")
                    
        except Exception as e:
            errors.append(str(e))
            
        return errors
        
    @staticmethod
    def validate_model_config(config) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        try:
            # Check dimensions
            if hasattr(config, 'n_channels') and config.n_channels <= 0:
                errors.append("Number of channels must be positive")
                
            if hasattr(config, 'vocab_size') and config.vocab_size <= 0:
                errors.append("Vocabulary size must be positive")
                
            if hasattr(config, 'd_model') and config.d_model <= 0:
                errors.append("Model dimension must be positive")
                
            # Check reasonable ranges
            if hasattr(config, 'dropout') and not (0 <= config.dropout <= 1):
                errors.append("Dropout must be between 0 and 1")
                
            if hasattr(config, 'n_heads') and config.n_heads <= 0:
                errors.append("Number of attention heads must be positive")
                
        except Exception as e:
            errors.append(str(e))
            
        return errors
        
    @staticmethod
    def validate_device_config(config) -> List[str]:
        """Validate device configuration."""
        errors = []
        
        try:
            # Check device type
            if hasattr(config, 'device_type'):
                valid_types = ['openbci', 'emotiv', 'lsl', 'simulated']
                if config.device_type not in valid_types:
                    errors.append(f"Invalid device type. Must be one of: {valid_types}")
                    
            # Check sampling rate
            if hasattr(config, 'sampling_rate'):
                validate_sampling_rate(config.sampling_rate)
                
            # Check channels
            if hasattr(config, 'n_channels') and config.n_channels <= 0:
                errors.append("Number of channels must be positive")
                
        except Exception as e:
            errors.append(str(e))
            
        return errors


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and capabilities."""
    requirements = {
        'python_version': sys.version_info[:2],
        'dependencies': {},
        'capabilities': {},
        'recommendations': []
    }
    
    # Check Python version
    if sys.version_info < (3, 9):
        requirements['recommendations'].append("Python 3.9+ recommended")
        
    # Check dependencies
    deps_to_check = [
        'numpy', 'scipy', 'torch', 'transformers', 'tiktoken',
        'mne', 'opacus', 'pyserial', 'pylsl'
    ]
    
    for dep in deps_to_check:
        try:
            __import__(dep)
            requirements['dependencies'][dep] = True
        except ImportError:
            requirements['dependencies'][dep] = False
            
    # Check capabilities
    core_available = all(requirements['dependencies'][dep] for dep in ['numpy', 'scipy'])
    ml_available = requirements['dependencies']['torch']
    signal_available = requirements['dependencies']['mne']
    devices_available = any(requirements['dependencies'][dep] for dep in ['pyserial', 'pylsl'])
    
    requirements['capabilities'] = {
        'core_processing': core_available,
        'machine_learning': ml_available,
        'signal_processing': signal_available,
        'device_support': devices_available,
        'full_functionality': all([core_available, ml_available, signal_available])
    }
    
    # Generate recommendations
    if not ml_available:
        requirements['recommendations'].append("Install PyTorch for full ML functionality")
    if not signal_available:
        requirements['recommendations'].append("Install MNE-Python for advanced signal processing")
    if not devices_available:
        requirements['recommendations'].append("Install pyserial/pylsl for device support")
        
    return requirements


def print_system_status():
    """Print comprehensive system status."""
    print("BCI-2-Token System Status")
    print("=" * 40)
    
    reqs = check_system_requirements()
    
    print(f"Python: {'.'.join(map(str, reqs['python_version']))}")
    
    print("\nDependencies:")
    for dep, available in reqs['dependencies'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
        
    print("\nCapabilities:")
    for cap, available in reqs['capabilities'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {cap.replace('_', ' ').title()}")
        
    if reqs['recommendations']:
        print("\nRecommendations:")
        for rec in reqs['recommendations']:
            print(f"  • {rec}")
            
    print()


if __name__ == '__main__':
    print_system_status()