"""
Tests for signal preprocessing functionality.
"""

import unittest
import numpy as np
import warnings
from bci2token.preprocessing import SignalPreprocessor, PreprocessingConfig


class TestSignalPreprocessing(unittest.TestCase):
    """Test signal preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PreprocessingConfig(
            sampling_rate=256,
            lowpass_freq=40.0,
            highpass_freq=0.5,
            apply_ica=False,  # Disable ICA for testing
            apply_car=True,
            standardize=True
        )
        self.preprocessor = SignalPreprocessor(self.config)
        
        # Generate test data
        self.n_channels = 8
        self.n_timepoints = 512
        self.test_data = self._generate_test_signal()
        
    def _generate_test_signal(self):
        """Generate synthetic test signal."""
        t = np.linspace(0, 2, self.n_timepoints)  # 2 seconds
        signal = np.zeros((self.n_channels, self.n_timepoints))
        
        for ch in range(self.n_channels):
            # Mix of frequencies
            freq1 = 10 + ch  # Alpha range
            freq2 = 25 + ch  # Beta range
            
            signal[ch] = (np.sin(2 * np.pi * freq1 * t) + 
                         0.5 * np.sin(2 * np.pi * freq2 * t) +
                         0.1 * np.random.randn(len(t)))
                         
        return signal
        
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        result = self.preprocessor.preprocess(self.test_data)
        
        # Check that all expected keys are present
        expected_keys = ['processed_data', 'epochs', 'sampling_rate', 
                        'channel_names', 'preprocessing_config']
        for key in expected_keys:
            self.assertIn(key, result)
            
        # Check data shapes
        self.assertEqual(result['processed_data'].shape[0], self.n_channels)
        self.assertTrue(len(result['epochs']) > 0)
        self.assertEqual(result['epochs'].shape[1], self.n_channels)
        
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        filtered_data = self.preprocessor._apply_bandpass_filter(self.test_data)
        
        # Should have same shape
        self.assertEqual(filtered_data.shape, self.test_data.shape)
        
        # Should reduce high frequency content
        original_power = np.mean(self.test_data**2)
        filtered_power = np.mean(filtered_data**2)
        self.assertLess(filtered_power, original_power * 1.1)  # Some reduction expected
        
    def test_common_average_reference(self):
        """Test common average reference."""
        car_data = self.preprocessor._apply_car(self.test_data)
        
        # Should have same shape
        self.assertEqual(car_data.shape, self.test_data.shape)
        
        # Mean across channels should be close to zero at each timepoint
        channel_mean = np.mean(car_data, axis=0)
        self.assertTrue(np.allclose(channel_mean, 0, atol=1e-10))
        
    def test_standardization(self):
        """Test signal standardization."""
        std_data = self.preprocessor._standardize(self.test_data)
        
        # Should have same shape
        self.assertEqual(std_data.shape, self.test_data.shape)
        
        # Each channel should have approximately zero mean and unit variance
        for ch in range(self.n_channels):
            channel_data = std_data[ch]
            self.assertAlmostEqual(np.mean(channel_data), 0, places=5)
            self.assertAlmostEqual(np.std(channel_data), 1, places=5)
            
    def test_epoch_creation(self):
        """Test epoch creation."""
        epochs = self.preprocessor._create_epochs(self.test_data)
        
        # Should be 3D: (n_epochs, n_channels, n_timepoints_per_epoch)
        self.assertEqual(epochs.ndim, 3)
        self.assertEqual(epochs.shape[1], self.n_channels)
        
        # Number of epochs should be reasonable
        expected_window_samples = int(self.config.window_size * self.config.sampling_rate)
        self.assertEqual(epochs.shape[2], expected_window_samples)
        
    def test_feature_extraction(self):
        """Test feature extraction."""
        # Create epochs first
        epochs = self.preprocessor._create_epochs(self.test_data)
        
        # Extract features
        features = self.preprocessor.extract_features(epochs)
        
        # Should be 2D: (n_epochs, n_features)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[0], epochs.shape[0])
        
        # Should have multiple features per channel
        expected_min_features = self.n_channels * 4  # At least time domain features
        self.assertGreater(features.shape[1], expected_min_features)
        
    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        # 1D input should raise error
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess(np.random.randn(100))
            
        # 3D input should raise error
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess(np.random.randn(5, 8, 100))
            
    def test_config_validation(self):
        """Test configuration validation."""
        # Lowpass frequency too high should raise error
        invalid_config = PreprocessingConfig(
            sampling_rate=256,
            lowpass_freq=200  # Above Nyquist
        )
        
        with self.assertRaises(ValueError):
            SignalPreprocessor(invalid_config)


if __name__ == '__main__':
    unittest.main()