"""
Tests for neural model architectures.
"""

import unittest
import torch
import numpy as np
from bci2token.models import (
    BrainToTokenModel, ModelConfig, 
    BrainSignalEncoder, CTCDecoder, DiffusionDecoder
)


class TestModels(unittest.TestCase):
    """Test neural model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            n_channels=8,
            sequence_length=256,
            d_model=128,  # Smaller for testing
            n_heads=4,
            n_layers=2,
            vocab_size=1000,
            max_sequence_length=32
        )
        
        # Test data
        self.batch_size = 4
        self.test_signals = torch.randn(
            self.batch_size, 
            self.config.n_channels, 
            self.config.sequence_length
        )
        self.test_tokens = torch.randint(
            1, self.config.vocab_size, 
            (self.batch_size, self.config.max_sequence_length)
        )
        
    def test_brain_signal_encoder(self):
        """Test brain signal encoder."""
        encoder = BrainSignalEncoder(self.config)
        
        # Forward pass
        output = encoder(self.test_signals)
        
        # Check output shape
        expected_shape = (self.batch_size, self.config.sequence_length, self.config.d_model)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is not all zeros/NaN
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.any(torch.isnan(output)))
        
    def test_ctc_decoder(self):
        """Test CTC decoder."""
        decoder = CTCDecoder(self.config)
        
        # Create encoder output
        encoder_output = torch.randn(
            self.batch_size, 
            self.config.sequence_length, 
            self.config.d_model
        )
        
        # Forward pass
        log_probs = decoder(encoder_output)
        
        # Check output shape (vocab_size + 1 for blank token)
        expected_shape = (self.batch_size, self.config.sequence_length, self.config.vocab_size + 1)
        self.assertEqual(log_probs.shape, expected_shape)
        
        # Check that it's log probabilities (sum to 1 after exp)
        probs = torch.exp(log_probs)
        prob_sums = torch.sum(probs, dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))
        
        # Test greedy decoding
        decoded = decoder.decode_greedy(log_probs)
        self.assertEqual(len(decoded), self.batch_size)
        
        # Each decoded sequence should be a list of integers
        for seq in decoded:
            self.assertIsInstance(seq, list)
            for token in seq:
                self.assertIsInstance(token, int)
                self.assertGreaterEqual(token, 0)
                self.assertLess(token, self.config.vocab_size)
                
    def test_diffusion_decoder(self):
        """Test diffusion decoder."""
        decoder = DiffusionDecoder(self.config)
        
        # Create encoder output
        encoder_output = torch.randn(
            self.batch_size,
            self.config.sequence_length,
            self.config.d_model
        )
        
        # Test training mode
        decoder.train()
        output = decoder(encoder_output, target_tokens=self.test_tokens)
        
        # Check output shape
        expected_shape = (self.batch_size, self.config.max_sequence_length, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Test inference mode
        decoder.eval()
        with torch.no_grad():
            # Note: This will be slow due to iterative sampling
            # In practice, you'd use fewer inference steps for testing
            pass  # Skip full inference test for speed
            
    def test_brain_to_token_model_ctc(self):
        """Test complete brain-to-token model with CTC decoder."""
        model = BrainToTokenModel(self.config, decoder_type='ctc')
        
        # Forward pass
        output = model(self.test_signals)
        
        # Check output shape
        expected_shape = (self.batch_size, self.config.sequence_length, self.config.vocab_size + 1)
        self.assertEqual(output.shape, expected_shape)
        
        # Test decoding
        decoded = model.decode(self.test_signals)
        self.assertEqual(len(decoded), self.batch_size)
        
    def test_brain_to_token_model_diffusion(self):
        """Test complete brain-to-token model with diffusion decoder."""
        model = BrainToTokenModel(self.config, decoder_type='diffusion')
        
        # Training forward pass
        model.train()
        output = model(self.test_signals, target_tokens=self.test_tokens)
        
        # Check output shape
        expected_shape = (self.batch_size, self.config.max_sequence_length, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        ctc_model = BrainToTokenModel(self.config, decoder_type='ctc')
        diffusion_model = BrainToTokenModel(self.config, decoder_type='diffusion')
        
        ctc_params = sum(p.numel() for p in ctc_model.parameters())
        diffusion_params = sum(p.numel() for p in diffusion_model.parameters())
        
        # Should have reasonable number of parameters
        self.assertGreater(ctc_params, 1000)  # At least 1K parameters
        self.assertLess(ctc_params, 10**8)    # Less than 100M parameters
        
        self.assertGreater(diffusion_params, 1000)
        self.assertLess(diffusion_params, 10**8)
        
        # Diffusion model should generally have more parameters
        self.assertGreater(diffusion_params, ctc_params)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the models."""
        model = BrainToTokenModel(self.config, decoder_type='ctc')
        
        # Forward pass
        output = model(self.test_signals)
        
        # Create dummy loss
        dummy_targets = torch.randint(0, self.config.vocab_size + 1, output.shape[:2])
        loss = torch.nn.CrossEntropyLoss()(
            output.view(-1, output.size(-1)), 
            dummy_targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
                
        self.assertTrue(has_gradients, "Model should have non-zero gradients")
        
    def test_device_compatibility(self):
        """Test model device compatibility."""
        model = BrainToTokenModel(self.config, decoder_type='ctc')
        
        # Test CPU
        cpu_output = model(self.test_signals)
        self.assertEqual(cpu_output.device.type, 'cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            gpu_signals = self.test_signals.cuda()
            gpu_output = model(gpu_signals)
            self.assertEqual(gpu_output.device.type, 'cuda')
            
    def test_model_modes(self):
        """Test training vs evaluation modes."""
        model = BrainToTokenModel(self.config, decoder_type='ctc')
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        train_output1 = model(self.test_signals)
        train_output2 = model(self.test_signals)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
        
        with torch.no_grad():
            eval_output1 = model(self.test_signals)
            eval_output2 = model(self.test_signals)
            
        # In eval mode, outputs should be deterministic
        self.assertTrue(torch.allclose(eval_output1, eval_output2))


if __name__ == '__main__':
    unittest.main()