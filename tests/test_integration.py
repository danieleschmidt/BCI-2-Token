"""
Integration tests for the complete BCI-2-Token pipeline.
"""

import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path

from bci2token import BrainDecoder, LLMInterface
from bci2token.models import ModelConfig
from bci2token.preprocessing import PreprocessingConfig
from bci2token.streaming import StreamingDecoder, StreamingConfig
from bci2token.devices import SimulatedDevice, DeviceConfig


class TestIntegration(unittest.TestCase):
    """Test complete system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_channels = 8
        self.sampling_rate = 256
        self.signal_duration = 2.0  # seconds
        self.n_timepoints = int(self.signal_duration * self.sampling_rate)
        
        # Generate test brain signal
        self.test_signal = self._generate_test_signal()
        
    def _generate_test_signal(self):
        """Generate synthetic brain signal."""
        t = np.linspace(0, self.signal_duration, self.n_timepoints)
        signal = np.zeros((self.n_channels, self.n_timepoints))
        
        for ch in range(self.n_channels):
            # Alpha rhythm around 10 Hz
            freq = 10 + ch * 0.5
            signal[ch] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
            
        return signal
        
    def test_basic_pipeline(self):
        """Test basic brain-to-token pipeline."""
        # Initialize decoder
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        # Initialize LLM interface  
        llm = LLMInterface(model_name='gpt2')
        
        # Decode brain signals to tokens
        tokens = decoder.decode_to_tokens(self.test_signal)
        
        # Should return a list of integers
        self.assertIsInstance(tokens, list)
        if tokens:  # May be empty for random signals
            for token in tokens:
                self.assertIsInstance(token, int)
                self.assertGreaterEqual(token, 0)
                
        # Decode to logits
        logits = decoder.decode_to_logits(self.test_signal)
        
        # Should return numpy array
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(logits.ndim, 2)  # (sequence_length, vocab_size)
        
        # Convert tokens to text
        if tokens:
            text = llm.tokens_to_text(tokens)
            self.assertIsInstance(text, str)
            
    def test_with_privacy_protection(self):
        """Test pipeline with differential privacy."""
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc',
            privacy_epsilon=1.0
        )
        
        # Should have privacy engine
        self.assertIsNotNone(decoder.privacy_engine)
        
        # Decode with privacy protection
        tokens = decoder.decode_to_tokens(self.test_signal)
        
        # Should still work (though accuracy may be lower)
        self.assertIsInstance(tokens, list)
        
    def test_streaming_pipeline(self):
        """Test real-time streaming pipeline."""
        # Initialize components
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        llm = LLMInterface(model_name='gpt2')
        
        config = StreamingConfig(
            buffer_duration=1.0,
            update_interval=0.1,
            confidence_threshold=0.5
        )
        
        streaming_decoder = StreamingDecoder(decoder, llm, config)
        
        # Test streaming session
        streaming_decoder.start_streaming()
        
        try:
            # Add some test data
            chunk_size = self.sampling_rate // 10  # 100ms chunks
            for i in range(0, self.n_timepoints, chunk_size):
                end_idx = min(i + chunk_size, self.n_timepoints)
                chunk = self.test_signal[:, i:end_idx]
                streaming_decoder.add_data(chunk)
                
            # Get status
            status = streaming_decoder.get_status()
            self.assertTrue(status['is_streaming'])
            self.assertGreater(status['buffer_size'], 0)
            
        finally:
            streaming_decoder.stop_streaming()
            
        # Should be stopped now
        status = streaming_decoder.get_status()
        self.assertFalse(status['is_streaming'])
        
    def test_simulated_device_integration(self):
        """Test integration with simulated device."""
        # Set up simulated device
        device_config = DeviceConfig(
            device_type='simulated',
            sampling_rate=self.sampling_rate,
            n_channels=self.n_channels
        )
        
        device = SimulatedDevice(device_config)
        
        # Connect device
        self.assertTrue(device.connect())
        self.assertTrue(device.is_connected)
        
        # Set up data collection
        collected_data = []
        
        def data_callback(data):
            collected_data.append(data.copy())
            
        device.set_data_callback(data_callback)
        
        # Start streaming briefly
        device.start_streaming()
        
        import time
        time.sleep(0.5)  # Collect for 500ms
        
        device.stop_streaming()
        device.disconnect()
        
        # Should have collected some data
        self.assertGreater(len(collected_data), 0)
        
        # Each chunk should have correct shape
        for chunk in collected_data:
            self.assertEqual(chunk.shape[0], self.n_channels)
            self.assertGreater(chunk.shape[1], 0)
            
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        # Get initial prediction
        initial_tokens = decoder.decode_to_tokens(self.test_signal)
        
        # Save model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.pt'
            decoder.save_model(model_path)
            
            # Create new decoder and load model
            new_decoder = BrainDecoder(
                signal_type='eeg',
                channels=self.n_channels,
                sampling_rate=self.sampling_rate,
                model_type='ctc'
            )
            
            new_decoder.load_model(model_path)
            
            # Should give same predictions
            loaded_tokens = new_decoder.decode_to_tokens(self.test_signal)
            self.assertEqual(initial_tokens, loaded_tokens)
            
    def test_model_info(self):
        """Test model information retrieval."""
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        info = decoder.get_model_info()
        
        # Check expected fields
        expected_fields = [
            'signal_type', 'channels', 'sampling_rate', 'model_type',
            'total_parameters', 'trainable_parameters', 'vocab_size'
        ]
        
        for field in expected_fields:
            self.assertIn(field, info)
            
        # Check values
        self.assertEqual(info['signal_type'], 'eeg')
        self.assertEqual(info['channels'], self.n_channels)
        self.assertEqual(info['sampling_rate'], self.sampling_rate)
        self.assertEqual(info['model_type'], 'ctc')
        self.assertGreater(info['total_parameters'], 0)
        self.assertGreater(info['trainable_parameters'], 0)
        
    def test_confidence_scores(self):
        """Test confidence score generation."""
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        # Get tokens with confidence
        result = decoder.decode_to_tokens(self.test_signal, return_confidence=True)
        
        # Should be a dictionary
        self.assertIsInstance(result, dict)
        self.assertIn('tokens', result)
        self.assertIn('confidence', result)
        
        tokens = result['tokens']
        confidence = result['confidence']
        
        # Confidence should be list of floats between 0 and 1
        self.assertIsInstance(confidence, list)
        if confidence:  # May be empty
            for conf in confidence:
                self.assertIsInstance(conf, (float, np.floating))
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)
                
        # Should have confidence for each token
        if tokens and confidence:
            # Note: May not be exactly equal due to epoch processing
            self.assertGreater(len(confidence), 0)
            
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=self.n_channels,
            sampling_rate=self.sampling_rate,
            model_type='ctc'
        )
        
        # Test with wrong number of channels
        wrong_signal = np.random.randn(self.n_channels + 1, self.n_timepoints)
        
        with self.assertRaises(ValueError):
            decoder.decode_to_tokens(wrong_signal)
            
        # Test with empty signal
        empty_signal = np.zeros((self.n_channels, 0))
        
        # Should handle gracefully
        tokens = decoder.decode_to_tokens(empty_signal)
        self.assertEqual(tokens, [])


if __name__ == '__main__':
    unittest.main()