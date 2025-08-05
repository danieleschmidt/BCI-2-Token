#!/usr/bin/env python3
"""
Basic usage example for BCI-2-Token framework.

Demonstrates simple brain signal decoding to text using synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bci2token import BrainDecoder, LLMInterface


def generate_synthetic_eeg(n_channels=8, duration=2.0, sampling_rate=256):
    """Generate synthetic EEG-like signal for demonstration."""
    n_timepoints = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_timepoints)
    
    # Create multi-channel signal with different frequencies
    signal = np.zeros((n_channels, n_timepoints))
    
    for ch in range(n_channels):
        # Alpha rhythm around 10 Hz with some variation
        freq = 10 + ch * 0.5
        phase = ch * np.pi / 4
        
        # Add some higher frequency components
        alpha_wave = np.sin(2 * np.pi * freq * t + phase)
        beta_wave = 0.3 * np.sin(2 * np.pi * (20 + ch) * t)
        
        # Add noise
        noise = 0.1 * np.random.randn(n_timepoints)
        
        signal[ch] = alpha_wave + beta_wave + noise
        
    # Scale to typical EEG amplitude range (microvolts)
    signal *= 50.0  # 50 µV amplitude
    
    return signal


def main():
    print("BCI-2-Token Basic Usage Example")
    print("=" * 40)
    
    # Generate synthetic brain signal
    print("1. Generating synthetic EEG signal...")
    brain_signal = generate_synthetic_eeg(n_channels=8, duration=2.0)
    print(f"   Signal shape: {brain_signal.shape} (channels, timepoints)")
    print(f"   Signal amplitude range: {brain_signal.min():.1f} to {brain_signal.max():.1f} µV")
    
    # Initialize brain decoder
    print("\n2. Initializing brain decoder...")
    decoder = BrainDecoder(
        signal_type='eeg',
        channels=8,
        sampling_rate=256,
        model_type='ctc'
    )
    
    # Print model information
    model_info = decoder.get_model_info()
    print(f"   Model type: {model_info['model_type']}")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Vocabulary size: {model_info['vocab_size']:,}")
    
    # Initialize LLM interface
    print("\n3. Initializing LLM interface...")
    llm = LLMInterface(model_name='gpt2')
    
    tokenizer_info = llm.get_tokenizer_info()
    print(f"   LLM model: {tokenizer_info['model_name']}")
    print(f"   Tokenizer: {tokenizer_info['tokenizer_type']}")
    print(f"   Vocabulary size: {tokenizer_info['vocab_size']:,}")
    
    # Decode brain signals
    print("\n4. Decoding brain signals to tokens...")
    result = decoder.decode_to_tokens(brain_signal, return_confidence=True)
    
    tokens = result['tokens']
    confidence = result.get('confidence', [])
    
    print(f"   Decoded {len(tokens)} tokens")
    if confidence:
        avg_confidence = np.mean(confidence)
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Convert tokens to text
    if tokens:
        print("\n5. Converting tokens to text...")
        text = llm.tokens_to_text(tokens)
        print(f"   Decoded text: '{text}'")
        print(f"   Token IDs: {tokens[:10]}..." if len(tokens) > 10 else f"   Token IDs: {tokens}")
    else:
        print("\n5. No tokens decoded (this is normal for random signals)")
        
    # Demonstrate logits generation
    print("\n6. Generating token logits for LLM integration...")
    logits = decoder.decode_to_logits(brain_signal)
    print(f"   Logits shape: {logits.shape} (sequence_length, vocab_size)")
    print(f"   Logits range: {logits.min():.2f} to {logits.max():.2f}")
    
    # Test privacy protection
    print("\n7. Testing with privacy protection...")
    private_decoder = BrainDecoder(
        signal_type='eeg',
        channels=8,
        sampling_rate=256,
        model_type='ctc',
        privacy_epsilon=1.0  # Differential privacy
    )
    
    private_tokens = private_decoder.decode_to_tokens(brain_signal)
    print(f"   Private decoder tokens: {len(private_tokens)}")
    
    if private_decoder.privacy_engine:
        privacy_report = private_decoder.privacy_engine.generate_privacy_report()
        print(f"   Privacy level: {privacy_report['privacy_parameters']['epsilon']}")
        
    print("\n✓ Basic usage example completed successfully!")
    

if __name__ == '__main__':
    main()