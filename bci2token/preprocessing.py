"""
Signal preprocessing pipeline for EEG/ECoG brain data.

Handles filtering, artifact removal, feature extraction, and preparation
for neural decoding models.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

try:
    import mne
    from scipy import signal
    from scipy.stats import zscore
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE-Python not available. Some preprocessing features will be limited.")


@dataclass
class PreprocessingConfig:
    """Configuration for signal preprocessing pipeline."""
    
    sampling_rate: int = 256
    lowpass_freq: float = 40.0
    highpass_freq: float = 0.5
    notch_freq: float = 60.0  # Power line frequency
    apply_ica: bool = True
    ica_n_components: Optional[int] = None
    apply_car: bool = True  # Common Average Reference
    standardize: bool = True
    window_size: float = 2.0  # seconds
    overlap: float = 0.5  # fraction


class SignalPreprocessor:
    """
    Preprocesses brain signals for neural decoding.
    
    Supports EEG, ECoG, and other electrophysiological data formats.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate preprocessing configuration."""
        if not HAS_MNE:
            if self.config.apply_ica:
                warnings.warn("ICA requested but MNE not available. Disabling ICA.")
                self.config.apply_ica = False
                
        if self.config.lowpass_freq >= self.config.sampling_rate / 2:
            raise ValueError("Lowpass frequency must be less than Nyquist frequency")
            
    def preprocess(self, 
                   raw_data: np.ndarray, 
                   channel_names: Optional[list] = None) -> Dict[str, Any]:
        """
        Main preprocessing pipeline.
        
        Args:
            raw_data: Shape (n_channels, n_timepoints)
            channel_names: Optional channel names
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        if raw_data.ndim != 2:
            raise ValueError("Input data must be 2D (channels x timepoints)")
            
        # Create channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i+1}" for i in range(raw_data.shape[0])]
            
        processed_data = raw_data.copy()
        
        # Apply bandpass filter
        processed_data = self._apply_bandpass_filter(processed_data)
        
        # Apply notch filter for power line interference
        processed_data = self._apply_notch_filter(processed_data)
        
        # Apply Common Average Reference
        if self.config.apply_car:
            processed_data = self._apply_car(processed_data)
            
        # Apply ICA for artifact removal
        if self.config.apply_ica and HAS_MNE:
            processed_data = self._apply_ica(processed_data, channel_names)
            
        # Standardize signals
        if self.config.standardize:
            processed_data = self._standardize(processed_data)
            
        # Create windowed epochs
        epochs = self._create_epochs(processed_data)
        
        # Extract features
        features = self._extract_features(processed_data)
        
        return {
            'preprocessed_data': processed_data,
            'epochs': epochs,
            'features': features,
            'sampling_rate': self.config.sampling_rate,
            'channel_names': channel_names,
            'preprocessing_config': self.config
        }
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove noise."""
        try:
            from scipy import signal
            nyquist = self.config.sampling_rate / 2
            low = self.config.highpass_freq / nyquist
            high = self.config.lowpass_freq / nyquist
            
            # Design Butterworth filter
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data, axis=1)
            return filtered_data
        except ImportError:
            warnings.warn("SciPy not available. Skipping bandpass filter.")
            return data
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line interference."""
        try:
            from scipy import signal
            nyquist = self.config.sampling_rate / 2
            freq = self.config.notch_freq / nyquist
            
            # Design notch filter
            b, a = signal.iirnotch(freq, Q=30)
            filtered_data = signal.filtfilt(b, a, data, axis=1)
            return filtered_data
        except ImportError:
            warnings.warn("SciPy not available. Skipping notch filter.")
            return data
    
    def _apply_car(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference."""
        car_data = data - np.mean(data, axis=0, keepdims=True)
        return car_data
    
    def _apply_ica(self, data: np.ndarray, channel_names: list) -> np.ndarray:
        """Apply Independent Component Analysis for artifact removal."""
        if not HAS_MNE:
            return data
            
        try:
            # Create MNE info structure
            info = mne.create_info(
                ch_names=channel_names,
                sfreq=self.config.sampling_rate,
                ch_types='eeg'
            )
            
            # Create raw object
            raw = mne.io.RawArray(data, info)
            
            # Apply ICA
            ica = mne.preprocessing.ICA(
                n_components=self.config.ica_n_components,
                random_state=42
            )
            ica.fit(raw)
            
            # Remove components (would typically use automated detection)
            ica.apply(raw)
            
            return raw.get_data()
        except Exception as e:
            warnings.warn(f"ICA failed: {e}. Using original data.")
            return data
    
    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize signals to zero mean and unit variance."""
        try:
            from scipy.stats import zscore
            return zscore(data, axis=1)
        except ImportError:
            # Manual standardization
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            return (data - mean) / (std + 1e-8)
    
    def _create_epochs(self, data: np.ndarray) -> np.ndarray:
        """Create windowed epochs from continuous data."""
        n_channels, n_timepoints = data.shape
        window_samples = int(self.config.window_size * self.config.sampling_rate)
        step_samples = int(window_samples * (1 - self.config.overlap))
        
        if window_samples > n_timepoints:
            return data.reshape(1, n_channels, n_timepoints)
        
        n_epochs = (n_timepoints - window_samples) // step_samples + 1
        epochs = np.zeros((n_epochs, n_channels, window_samples))
        
        for i in range(n_epochs):
            start = i * step_samples
            end = start + window_samples
            epochs[i] = data[:, start:end]
        
        return epochs
    
    def _extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-domain and frequency-domain features."""
        features = {}
        
        # Time-domain features
        features['mean'] = np.mean(data, axis=1)
        features['std'] = np.std(data, axis=1)
        features['variance'] = np.var(data, axis=1)
        features['skewness'] = self._calculate_skewness(data)
        features['kurtosis'] = self._calculate_kurtosis(data)
        
        # Frequency-domain features
        try:
            from scipy import signal
            freqs, psd = signal.welch(
                data, 
                fs=self.config.sampling_rate,
                axis=1
            )
            
            # Band power features
            features['alpha_power'] = self._band_power(freqs, psd, 8, 12)
            features['beta_power'] = self._band_power(freqs, psd, 13, 30)
            features['gamma_power'] = self._band_power(freqs, psd, 31, 100)
            features['total_power'] = np.sum(psd, axis=1)
            
        except ImportError:
            warnings.warn("SciPy not available. Skipping frequency features.")
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each channel."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        return np.mean(normalized**3, axis=1)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each channel."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        return np.mean(normalized**4, axis=1) - 3
    
    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                   low_freq: float, high_freq: float) -> np.ndarray:
        """Calculate power in specific frequency band."""
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(psd[:, freq_mask], axis=1)
