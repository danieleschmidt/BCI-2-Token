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
        
        return {
            'processed_data': processed_data,
            'epochs': epochs,
            'sampling_rate': self.config.sampling_rate,
            'channel_names': channel_names,
            'preprocessing_config': self.config
        }
        
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove noise outside frequency band of interest."""
        sos = signal.butter(
            4, 
            [self.config.highpass_freq, self.config.lowpass_freq],
            btype='band',
            fs=self.config.sampling_rate,
            output='sos'
        )
        return signal.sosfiltfilt(sos, data, axis=1)
        
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line interference."""
        sos = signal.iirnotch(
            self.config.notch_freq,
            Q=30,
            fs=self.config.sampling_rate
        )
        return signal.sosfiltfilt(sos, data, axis=1)
        
    def _apply_car(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference to reduce common-mode noise."""
        car = np.mean(data, axis=0, keepdims=True)
        return data - car
        
    def _apply_ica(self, data: np.ndarray, channel_names: list) -> np.ndarray:
        """Apply Independent Component Analysis for artifact removal."""
        if not HAS_MNE:
            return data
            
        # Create MNE Info object
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.config.sampling_rate,
            ch_types='eeg'
        )
        
        # Create Raw object
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply ICA
        n_components = self.config.ica_n_components or min(len(channel_names), 20)
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='infomax',
            random_state=42,
            verbose=False
        )
        
        ica.fit(raw)
        
        # Auto-detect and exclude eye blink/muscle artifacts
        # This is a simplified approach - in practice, you'd want more sophisticated artifact detection
        exclude_idx = []
        
        # Simple heuristic: exclude components with high variance in frontal channels
        if any('Fp' in ch or 'AF' in ch for ch in channel_names):
            frontal_channels = [i for i, ch in enumerate(channel_names) 
                              if 'Fp' in ch or 'AF' in ch]
            if frontal_channels:
                # Exclude components with high correlation to frontal channels
                for i in range(n_components):
                    if np.max(np.abs(ica.mixing_matrix_[frontal_channels, i])) > 0.7:
                        exclude_idx.append(i)
                        
        ica.exclude = exclude_idx
        raw_corrected = ica.apply(raw, verbose=False)
        
        return raw_corrected.get_data()
        
    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize signals to zero mean and unit variance."""
        return zscore(data, axis=1)
        
    def _create_epochs(self, data: np.ndarray) -> np.ndarray:
        """Create overlapping windows/epochs from continuous data."""
        window_samples = int(self.config.window_size * self.config.sampling_rate)
        step_samples = int(window_samples * (1 - self.config.overlap))
        
        epochs = []
        start = 0
        
        while start + window_samples <= data.shape[1]:
            epoch = data[:, start:start + window_samples]
            epochs.append(epoch)
            start += step_samples
            
        return np.array(epochs)  # Shape: (n_epochs, n_channels, n_timepoints)
        
    def extract_features(self, epochs: np.ndarray) -> np.ndarray:
        """
        Extract features from epoched data for neural decoding.
        
        Args:
            epochs: Shape (n_epochs, n_channels, n_timepoints)
            
        Returns:
            Feature array of shape (n_epochs, n_features)
        """
        n_epochs, n_channels, n_timepoints = epochs.shape
        features = []
        
        for epoch in epochs:
            epoch_features = []
            
            # Time domain features
            epoch_features.extend(np.mean(epoch, axis=1))  # Channel means
            epoch_features.extend(np.std(epoch, axis=1))   # Channel standard deviations
            epoch_features.extend(np.max(epoch, axis=1))   # Channel maxima
            epoch_features.extend(np.min(epoch, axis=1))   # Channel minima
            
            # Frequency domain features
            freqs, psd = signal.welch(
                epoch, 
                fs=self.config.sampling_rate, 
                nperseg=min(256, n_timepoints)
            )
            
            # Band power features
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8), 
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)
            }
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.mean(psd[:, band_mask], axis=1)
                    epoch_features.extend(band_power)
                    
            features.append(epoch_features)
            
        return np.array(features)