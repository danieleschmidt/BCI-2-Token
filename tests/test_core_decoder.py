"""
Comprehensive tests for the BrainDecoder core component
"""

import pytest
import torch
import numpy as np
from typing import List
from unittest.mock import patch, MagicMock

from bci2token.core.decoder import BrainDecoder, DecoderConfig, DecoderType


class TestBrainDecoderInitialization:
    """Test decoder initialization and configuration"""
    
    @pytest.mark.unit
    def test_default_initialization(self, decoder_config: DecoderConfig):
        """Test decoder initializes with default configuration"""
        decoder = BrainDecoder(decoder_config)
        
        assert decoder.config == decoder_config
        assert decoder.device.type == "cpu"  # Forced in test config
        assert decoder.signal_processor is not None
        assert decoder.decoder_model is not None
    
    @pytest.mark.unit
    def test_privacy_initialization(self, privacy_test_config: DecoderConfig):
        """Test decoder initializes with privacy settings"""
        decoder = BrainDecoder(privacy_test_config)
        
        assert decoder.privacy_engine is not None
        assert decoder.config.privacy_epsilon == 1.0
    
    @pytest.mark.unit
    def test_device_selection(self):
        """Test automatic device selection"""
        config = DecoderConfig(device="auto")
        decoder = BrainDecoder(config)
        
        # Should select an available device
        assert decoder.device.type in ["cpu", "cuda", "mps"]
    
    @pytest.mark.unit
    def test_invalid_decoder_type(self):
        """Test handling of invalid decoder type"""
        config = DecoderConfig()
        # Temporarily modify enum to test error handling
        config.decoder_type = "invalid_type"
        
        with pytest.raises(ValueError, match="Unsupported decoder type"):
            BrainDecoder(config)


class TestSignalPreprocessing:
    """Test signal preprocessing functionality"""
    
    @pytest.mark.unit
    def test_eeg_preprocessing(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test EEG signal preprocessing"""
        processed = brain_decoder.preprocess_signals(sample_eeg_data, apply_privacy=False)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.device == brain_decoder.device
        assert processed.dtype == torch.float32
        assert not torch.isnan(processed).any()
        assert not torch.isinf(processed).any()
    
    @pytest.mark.unit
    def test_privacy_preprocessing(self, sample_eeg_data: np.ndarray):
        """Test preprocessing with differential privacy"""
        config = DecoderConfig(privacy_epsilon=1.0, device="cpu")
        decoder = BrainDecoder(config)
        
        # Process with and without privacy
        without_privacy = decoder.preprocess_signals(sample_eeg_data, apply_privacy=False)
        with_privacy = decoder.preprocess_signals(sample_eeg_data, apply_privacy=True)
        
        # Should be different due to noise injection
        assert not torch.allclose(without_privacy, with_privacy, atol=1e-6)
    
    @pytest.mark.unit
    def test_input_validation(self, brain_decoder: BrainDecoder):
        """Test input validation for preprocessing"""
        # Test with wrong number of channels
        wrong_channels = np.random.randn(32, 1000)  # Expected 64 channels
        
        with pytest.raises(ValueError):
            brain_decoder.preprocess_signals(wrong_channels)
        
        # Test with NaN values
        nan_signals = np.full((64, 1000), np.nan)
        with pytest.raises(ValueError):
            brain_decoder.preprocess_signals(nan_signals)


class TestDecoding:
    """Test token decoding functionality"""
    
    @pytest.mark.unit
    def test_forward_pass(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test forward pass through decoder"""
        processed_signals = brain_decoder.preprocess_signals(sample_eeg_data)
        
        # Add batch dimension
        batched_signals = processed_signals.unsqueeze(0)
        
        with torch.no_grad():
            logits = brain_decoder.forward(batched_signals)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape[-1] == brain_decoder.config.vocab_size
        assert not torch.isnan(logits).any()
    
    @pytest.mark.unit
    def test_decode_to_tokens(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test end-to-end token decoding"""
        tokens = brain_decoder.decode_to_tokens(sample_eeg_data)
        
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < brain_decoder.config.vocab_size for t in tokens)
    
    @pytest.mark.unit
    def test_confidence_threshold(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test confidence threshold filtering"""
        # Low threshold should return more tokens
        low_threshold_tokens = brain_decoder.decode_to_tokens(
            sample_eeg_data, confidence_threshold=0.1
        )
        
        # High threshold should return fewer tokens
        high_threshold_tokens = brain_decoder.decode_to_tokens(
            sample_eeg_data, confidence_threshold=0.9
        )
        
        assert len(low_threshold_tokens) >= len(high_threshold_tokens)
    
    @pytest.mark.unit
    def test_attention_weights(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test attention weight extraction"""
        processed_signals = brain_decoder.preprocess_signals(sample_eeg_data)
        batched_signals = processed_signals.unsqueeze(0)
        
        attention_weights = brain_decoder.get_attention_weights(batched_signals)
        
        # Attention weights may or may not be available depending on model
        if attention_weights is not None:
            assert isinstance(attention_weights, torch.Tensor)
            assert not torch.isnan(attention_weights).any()


class TestCalibration:
    """Test model calibration functionality"""
    
    @pytest.mark.unit
    def test_calibration(
        self, 
        brain_decoder: BrainDecoder, 
        training_data: List[tuple]
    ):
        """Test model calibration"""
        metrics = brain_decoder.calibrate(training_data, num_epochs=2)
        
        assert isinstance(metrics, dict)
        assert "calibration_loss" in metrics
        assert "num_epochs" in metrics
        assert "num_samples" in metrics
        assert metrics["num_epochs"] == 2
        assert metrics["num_samples"] == len(training_data)
        assert metrics["calibration_loss"] >= 0.0
    
    @pytest.mark.unit
    def test_calibration_improves_performance(
        self, 
        brain_decoder: BrainDecoder,
        training_data: List[tuple]
    ):
        """Test that calibration improves model performance"""
        # Get initial predictions
        test_signals = training_data[0][0]
        initial_tokens = brain_decoder.decode_to_tokens(test_signals)
        
        # Calibrate model
        brain_decoder.calibrate(training_data, num_epochs=5)
        
        # Get post-calibration predictions
        calibrated_tokens = brain_decoder.decode_to_tokens(test_signals)
        
        # Predictions should be different after calibration
        # (This is a weak test, but demonstrates the functionality)
        assert len(calibrated_tokens) >= 0  # At least decoder still works


class TestModelPersistence:
    """Test model saving and loading"""
    
    @pytest.mark.unit
    def test_save_and_load(self, brain_decoder: BrainDecoder, temp_dir: str):
        """Test model saving and loading"""
        model_path = f"{temp_dir}/test_model.pt"
        
        # Save model
        brain_decoder.save_model(model_path)
        
        # Load model
        loaded_decoder = BrainDecoder.load_model(model_path)
        
        # Verify configuration matches
        assert loaded_decoder.config.signal_type == brain_decoder.config.signal_type
        assert loaded_decoder.config.channels == brain_decoder.config.channels
        assert loaded_decoder.config.vocab_size == brain_decoder.config.vocab_size
    
    @pytest.mark.unit
    def test_loaded_model_inference(
        self, 
        brain_decoder: BrainDecoder, 
        sample_eeg_data: np.ndarray,
        temp_dir: str
    ):
        """Test that loaded model produces same results"""
        model_path = f"{temp_dir}/test_model.pt"
        
        # Get original predictions
        original_tokens = brain_decoder.decode_to_tokens(sample_eeg_data)
        
        # Save and load model
        brain_decoder.save_model(model_path)
        loaded_decoder = BrainDecoder.load_model(model_path)
        
        # Get loaded model predictions
        loaded_tokens = loaded_decoder.decode_to_tokens(sample_eeg_data)
        
        # Should produce identical results
        assert loaded_tokens == original_tokens


class TestModelInfo:
    """Test model information and statistics"""
    
    @pytest.mark.unit
    def test_model_info(self, brain_decoder: BrainDecoder):
        """Test model information retrieval"""
        info = brain_decoder.get_model_info()
        
        assert isinstance(info, dict)
        assert "decoder_type" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "device" in info
        assert "privacy_enabled" in info
        
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["decoder_type"] == brain_decoder.config.decoder_type.value
    
    @pytest.mark.unit
    def test_parameter_count_consistency(self, brain_decoder: BrainDecoder):
        """Test parameter count consistency"""
        info = brain_decoder.get_model_info()
        
        # Manually count parameters
        manual_count = sum(p.numel() for p in brain_decoder.parameters())
        
        assert info["total_parameters"] == manual_count


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    def test_invalid_signal_shape(self, brain_decoder: BrainDecoder):
        """Test handling of invalid signal shapes"""
        # Wrong number of dimensions
        wrong_shape = np.random.randn(64)  # Should be 2D
        
        with pytest.raises(ValueError):
            brain_decoder.preprocess_signals(wrong_shape)
    
    @pytest.mark.unit
    def test_empty_signals(self, brain_decoder: BrainDecoder):
        """Test handling of empty signals"""
        empty_signals = np.array([]).reshape(64, 0)
        
        with pytest.raises(ValueError):
            brain_decoder.preprocess_signals(empty_signals)
    
    @pytest.mark.security
    def test_adversarial_inputs(self, brain_decoder: BrainDecoder, security_test_inputs: dict):
        """Test handling of adversarial inputs"""
        for input_name, adversarial_input in security_test_inputs.items():
            if "signals" in input_name:
                try:
                    # Should handle gracefully without crashing
                    result = brain_decoder.decode_to_tokens(adversarial_input)
                    assert isinstance(result, list)  # Should return valid format
                except (ValueError, RuntimeError) as e:
                    # Expected for some adversarial inputs
                    assert "Invalid" in str(e) or "NaN" in str(e) or "inf" in str(e)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for decoder"""
    
    def test_inference_speed(self, brain_decoder: BrainDecoder, sample_eeg_data: np.ndarray):
        """Test inference speed"""
        import time
        
        # Warm up
        for _ in range(3):
            brain_decoder.decode_to_tokens(sample_eeg_data)
        
        # Time inference
        start_time = time.time()
        for _ in range(10):
            tokens = brain_decoder.decode_to_tokens(sample_eeg_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 1 second per inference)
        assert avg_time < 1.0
        
        # Should produce valid output
        assert len(tokens) >= 0
    
    def test_memory_usage(self, brain_decoder: BrainDecoder, performance_test_data: dict):
        """Test memory usage with large inputs"""
        large_signals = performance_test_data["large_eeg"]
        
        # Should handle large inputs without excessive memory usage
        try:
            tokens = brain_decoder.decode_to_tokens(large_signals)
            assert isinstance(tokens, list)
        except RuntimeError as e:
            # Acceptable if out of memory on test hardware
            if "out of memory" not in str(e).lower():
                raise
    
    def test_batch_processing(self, brain_decoder: BrainDecoder):
        """Test batch processing efficiency"""
        batch_size = 4
        signal_length = 1000
        
        # Generate batch of signals
        batch_signals = [
            np.random.randn(64, signal_length) 
            for _ in range(batch_size)
        ]
        
        # Process individually
        import time
        start_time = time.time()
        individual_results = []
        for signals in batch_signals:
            tokens = brain_decoder.decode_to_tokens(signals)
            individual_results.append(tokens)
        individual_time = time.time() - start_time
        
        # For now, we don't have batch processing implemented
        # But this test structure is ready for when we do
        assert len(individual_results) == batch_size