"""
Advanced input validation and sanitization for BCI-2-Token.

Provides comprehensive validation, sanitization, and anomaly detection
for brain signal data and system inputs.
"""

import numpy as np
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from .utils import BCIError, validate_signal_shape


class ValidationError(BCIError):
    """Input validation error."""
    pass


class SanitizationError(BCIError):
    """Input sanitization error."""
    pass


class AnomalyLevel(Enum):
    """Anomaly detection levels."""
    NORMAL = 1
    SUSPICIOUS = 2
    ANOMALOUS = 3
    CRITICAL = 4
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    anomaly_level: AnomalyLevel
    issues: List[str]
    sanitized_data: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SignalConstraints:
    """Constraints for brain signal validation."""
    min_channels: int = 1
    max_channels: int = 256
    min_sampling_rate: float = 128.0
    max_sampling_rate: float = 10000.0
    min_duration: float = 0.1  # seconds
    max_duration: float = 300.0  # 5 minutes
    max_amplitude: float = 1000.0  # µV
    min_amplitude: float = -1000.0  # µV
    max_noise_level: float = 100.0  # µV RMS


class SignalValidator:
    """
    Validates brain signal data for safety and quality.
    """
    
    def __init__(self, constraints: Optional[SignalConstraints] = None):
        self.constraints = constraints or SignalConstraints()
        
    def validate_signal(self, signal: np.ndarray, 
                       sampling_rate: float,
                       channel_names: Optional[List[str]] = None) -> ValidationResult:
        """
        Comprehensive signal validation.
        
        Args:
            signal: Brain signal data (channels x timepoints)
            sampling_rate: Sampling rate in Hz
            channel_names: Optional channel names
            
        Returns:
            Validation result with any issues found
        """
        issues = []
        anomaly_level = AnomalyLevel.NORMAL
        metadata = {}
        
        # Basic shape validation
        if signal.ndim != 2:
            issues.append(f"Signal must be 2D (channels x timepoints), got {signal.ndim}D")
            return ValidationResult(False, AnomalyLevel.CRITICAL, issues)
        
        n_channels, n_timepoints = signal.shape
        duration = n_timepoints / sampling_rate
        
        # Channel count validation
        if n_channels < self.constraints.min_channels:
            issues.append(f"Too few channels: {n_channels} < {self.constraints.min_channels}")
        elif n_channels > self.constraints.max_channels:
            issues.append(f"Too many channels: {n_channels} > {self.constraints.max_channels}")
            anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Sampling rate validation
        if sampling_rate < self.constraints.min_sampling_rate:
            issues.append(f"Sampling rate too low: {sampling_rate} < {self.constraints.min_sampling_rate}")
        elif sampling_rate > self.constraints.max_sampling_rate:
            issues.append(f"Sampling rate too high: {sampling_rate} > {self.constraints.max_sampling_rate}")
            anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Duration validation
        if duration < self.constraints.min_duration:
            issues.append(f"Signal too short: {duration:.3f}s < {self.constraints.min_duration}s")
        elif duration > self.constraints.max_duration:
            issues.append(f"Signal too long: {duration:.3f}s > {self.constraints.max_duration}s")
            anomaly_level = max(anomaly_level, AnomalyLevel.ANOMALOUS)
        
        # Amplitude validation
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        if signal_min < self.constraints.min_amplitude:
            issues.append(f"Signal amplitude too low: {signal_min:.1f} < {self.constraints.min_amplitude}")
            anomaly_level = max(anomaly_level, AnomalyLevel.ANOMALOUS)
        
        if signal_max > self.constraints.max_amplitude:
            issues.append(f"Signal amplitude too high: {signal_max:.1f} > {self.constraints.max_amplitude}")
            anomaly_level = max(anomaly_level, AnomalyLevel.CRITICAL)
        
        # Check for NaN/inf values
        if np.any(np.isnan(signal)):
            issues.append("Signal contains NaN values")
            anomaly_level = AnomalyLevel.CRITICAL
        
        if np.any(np.isinf(signal)):
            issues.append("Signal contains infinite values")
            anomaly_level = AnomalyLevel.CRITICAL
        
        # Noise level assessment
        noise_rms = np.sqrt(np.mean(signal**2, axis=1))
        max_noise = np.max(noise_rms)
        
        if max_noise > self.constraints.max_noise_level:
            issues.append(f"Excessive noise level: {max_noise:.1f} > {self.constraints.max_noise_level}")
            anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Statistical anomalies
        channel_variances = np.var(signal, axis=1)
        mean_variance = np.mean(channel_variances)
        
        # Check for dead channels (very low variance)
        dead_channels = np.where(channel_variances < mean_variance * 0.01)[0]
        if len(dead_channels) > 0:
            issues.append(f"Potential dead channels detected: {dead_channels.tolist()}")
            anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Check for saturated channels (clipped values)
        for ch in range(n_channels):
            ch_signal = signal[ch]
            # Check for repeated values (potential saturation)
            unique_values = len(np.unique(ch_signal))
            if unique_values < n_timepoints * 0.1:  # Less than 10% unique values
                issues.append(f"Channel {ch} may be saturated (low value diversity)")
                anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Frequency domain analysis
        try:
            from scipy import signal as scipy_signal
            
            # Check for obvious artifacts (50/60 Hz line noise)
            freqs, psd = scipy_signal.welch(signal, fs=sampling_rate, axis=1)
            
            # Look for excessive line noise
            line_freq_50 = np.argmin(np.abs(freqs - 50))
            line_freq_60 = np.argmin(np.abs(freqs - 60))
            
            avg_psd = np.mean(psd, axis=0)
            median_psd = np.median(avg_psd)
            
            if avg_psd[line_freq_50] > median_psd * 10:
                issues.append("Excessive 50Hz line noise detected")
                anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
                
            if avg_psd[line_freq_60] > median_psd * 10:
                issues.append("Excessive 60Hz line noise detected")
                anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
                
        except ImportError:
            pass  # Skip frequency analysis if scipy not available
        
        # Metadata
        metadata = {
            'n_channels': n_channels,
            'n_timepoints': n_timepoints,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'amplitude_range': [float(signal_min), float(signal_max)],
            'noise_rms': noise_rms.tolist(),
            'dead_channels': dead_channels.tolist() if 'dead_channels' in locals() else [],
            'validation_timestamp': time.time()
        }
        
        # Determine if valid
        is_valid = (anomaly_level != AnomalyLevel.CRITICAL and 
                   len([issue for issue in issues if 'too' in issue.lower()]) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            anomaly_level=anomaly_level,
            issues=issues,
            metadata=metadata
        )


class InputSanitizer:
    """
    Sanitizes inputs to prevent injection attacks and ensure data safety.
    """
    
    @staticmethod
    def sanitize_brain_signal(signal: np.ndarray,
                            max_amplitude: float = 1000.0,
                            remove_outliers: bool = True) -> np.ndarray:
        """
        Sanitize brain signal data.
        
        Args:
            signal: Input signal
            max_amplitude: Maximum allowed amplitude
            remove_outliers: Whether to remove statistical outliers
            
        Returns:
            Sanitized signal
        """
        if not isinstance(signal, np.ndarray):
            try:
                signal = np.array(signal, dtype=np.float64)
            except Exception as e:
                raise SanitizationError(f"Cannot convert input to numpy array: {e}")
        
        # Ensure float type
        if not np.issubdtype(signal.dtype, np.floating):
            signal = signal.astype(np.float64)
        
        # Remove NaN and inf values
        if np.any(np.isnan(signal)):
            signal = np.nan_to_num(signal, nan=0.0)
            warnings.warn("NaN values replaced with zeros")
            
        if np.any(np.isinf(signal)):
            signal = np.nan_to_num(signal, posinf=max_amplitude, neginf=-max_amplitude)
            warnings.warn("Infinite values clipped")
        
        # Clip extreme amplitudes
        signal = np.clip(signal, -max_amplitude, max_amplitude)
        
        # Remove statistical outliers if requested
        if remove_outliers:
            for ch in range(signal.shape[0]):
                ch_signal = signal[ch]
                q1, q3 = np.percentile(ch_signal, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Clip outliers
                outlier_mask = (ch_signal < lower_bound) | (ch_signal > upper_bound)
                if np.any(outlier_mask):
                    ch_signal[outlier_mask] = np.median(ch_signal)
                    signal[ch] = ch_signal
        
        return signal
    
    @staticmethod
    def sanitize_string_input(input_str: str, 
                            max_length: int = 1000,
                            allowed_chars: Optional[str] = None) -> str:
        """
        Sanitize string input.
        
        Args:
            input_str: Input string
            max_length: Maximum allowed length
            allowed_chars: Regex pattern for allowed characters
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Remove control characters
        input_str = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
            warnings.warn(f"String truncated to {max_length} characters")
        
        # Filter allowed characters
        if allowed_chars:
            input_str = re.sub(f'[^{allowed_chars}]', '', input_str)
        
        return input_str
    
    @staticmethod
    def sanitize_numeric_input(value: Union[int, float],
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None) -> float:
        """
        Sanitize numeric input.
        
        Args:
            value: Input value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Sanitized numeric value
        """
        try:
            value = float(value)
        except (ValueError, TypeError) as e:
            raise SanitizationError(f"Cannot convert to numeric: {e}")
        
        # Check for NaN/inf
        if np.isnan(value):
            raise SanitizationError("Value is NaN")
        if np.isinf(value):
            raise SanitizationError("Value is infinite")
        
        # Apply bounds
        if min_value is not None and value < min_value:
            value = min_value
            warnings.warn(f"Value clipped to minimum: {min_value}")
            
        if max_value is not None and value > max_value:
            value = max_value
            warnings.warn(f"Value clipped to maximum: {max_value}")
        
        return value


class AnomalyDetector:
    """
    Detects anomalies in brain signal data and system behavior.
    """
    
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.baseline_stats = {}
        self.recent_signals = []
        self.max_history = 100
        
    def update_baseline(self, signal: np.ndarray, sampling_rate: float):
        """Update baseline statistics with new signal."""
        stats = {
            'mean_amplitude': np.mean(np.abs(signal)),
            'rms_power': np.sqrt(np.mean(signal**2)),
            'peak_amplitude': np.max(np.abs(signal)),
            'signal_variance': np.var(signal),
            'sampling_rate': sampling_rate,
            'timestamp': time.time()
        }
        
        self.recent_signals.append(stats)
        if len(self.recent_signals) > self.max_history:
            self.recent_signals.pop(0)
        
        # Update baseline statistics
        if len(self.recent_signals) >= 10:  # Minimum samples for baseline
            recent_stats = self.recent_signals[-10:]  # Use last 10 signals
            
            self.baseline_stats = {
                'mean_amplitude': {
                    'mean': np.mean([s['mean_amplitude'] for s in recent_stats]),
                    'std': np.std([s['mean_amplitude'] for s in recent_stats])
                },
                'rms_power': {
                    'mean': np.mean([s['rms_power'] for s in recent_stats]),
                    'std': np.std([s['rms_power'] for s in recent_stats])
                },
                'peak_amplitude': {
                    'mean': np.mean([s['peak_amplitude'] for s in recent_stats]),
                    'std': np.std([s['peak_amplitude'] for s in recent_stats])
                },
                'signal_variance': {
                    'mean': np.mean([s['signal_variance'] for s in recent_stats]),
                    'std': np.std([s['signal_variance'] for s in recent_stats])
                }
            }
    
    def detect_anomalies(self, signal: np.ndarray, 
                        sampling_rate: float) -> Tuple[AnomalyLevel, List[str]]:
        """
        Detect anomalies in brain signal.
        
        Args:
            signal: Brain signal to analyze
            sampling_rate: Sampling rate
            
        Returns:
            Tuple of (anomaly_level, list_of_issues)
        """
        issues = []
        anomaly_level = AnomalyLevel.NORMAL
        
        if not self.baseline_stats:
            # No baseline yet, update and return normal
            self.update_baseline(signal, sampling_rate)
            return AnomalyLevel.NORMAL, []
        
        # Calculate signal statistics
        mean_amplitude = np.mean(np.abs(signal))
        rms_power = np.sqrt(np.mean(signal**2))
        peak_amplitude = np.max(np.abs(signal))
        signal_variance = np.var(signal)
        
        # Check against baseline with z-score
        threshold = 2.0 * self.sensitivity  # Adjust threshold based on sensitivity
        
        # Mean amplitude anomaly
        if 'mean_amplitude' in self.baseline_stats:
            baseline = self.baseline_stats['mean_amplitude']
            z_score = abs(mean_amplitude - baseline['mean']) / (baseline['std'] + 1e-8)
            if z_score > threshold:
                issues.append(f"Anomalous mean amplitude (z-score: {z_score:.2f})")
                anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
                if z_score > threshold * 1.5:
                    anomaly_level = AnomalyLevel.ANOMALOUS
        
        # RMS power anomaly
        if 'rms_power' in self.baseline_stats:
            baseline = self.baseline_stats['rms_power']
            z_score = abs(rms_power - baseline['mean']) / (baseline['std'] + 1e-8)
            if z_score > threshold:
                issues.append(f"Anomalous RMS power (z-score: {z_score:.2f})")
                anomaly_level = max(anomaly_level, AnomalyLevel.SUSPICIOUS)
        
        # Peak amplitude anomaly
        if 'peak_amplitude' in self.baseline_stats:
            baseline = self.baseline_stats['peak_amplitude']
            z_score = abs(peak_amplitude - baseline['mean']) / (baseline['std'] + 1e-8)
            if z_score > threshold * 2:  # Higher threshold for peaks
                issues.append(f"Anomalous peak amplitude (z-score: {z_score:.2f})")
                anomaly_level = max(anomaly_level, AnomalyLevel.ANOMALOUS)
                if z_score > threshold * 3:
                    anomaly_level = AnomalyLevel.CRITICAL
        
        # Update baseline with current signal
        self.update_baseline(signal, sampling_rate)
        
        return anomaly_level, issues


def create_robust_validator(constraints: Optional[SignalConstraints] = None) -> SignalValidator:
    """Create a robust signal validator with default settings."""
    if constraints is None:
        constraints = SignalConstraints(
            max_amplitude=500.0,  # Conservative limit
            max_duration=60.0,    # 1 minute max
            max_noise_level=50.0  # Lower noise threshold
        )
    
    return SignalValidator(constraints)