#!/usr/bin/env python3
"""
Simple test runner for BCI-2-Token framework without requiring pytest.

Runs basic functionality tests to validate the implementation.
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from bci2token import BrainDecoder, LLMInterface
        from bci2token.streaming import StreamingDecoder
        from bci2token.devices import create_device, DeviceConfig
        from bci2token.privacy import PrivacyEngine
        print("   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_initialization():
    """Test decoder initialization."""
    print("Testing decoder initialization...")
    
    try:
        from bci2token import BrainDecoder
        
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=8,
            sampling_rate=256,
            model_type='ctc'
        )
        
        info = decoder.get_model_info()
        assert info['channels'] == 8
        assert info['sampling_rate'] == 256
        assert info['model_type'] == 'ctc'
        
        print("   ✓ Decoder initialization successful")
        return True
    except Exception as e:
        print(f"   ✗ Decoder initialization failed: {e}")
        traceback.print_exc()
        return False


def test_synthetic_decoding():
    """Test decoding with synthetic data."""
    print("Testing synthetic signal decoding...")
    
    try:
        from bci2token import BrainDecoder, LLMInterface
        
        # Create decoder
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=8,
            sampling_rate=256,
            model_type='ctc'
        )
        
        # Create LLM interface
        llm = LLMInterface(model_name='gpt2')
        
        # Generate synthetic signal
        signal = np.random.randn(8, 512)  # 8 channels, 2 seconds
        
        # Decode to tokens
        tokens = decoder.decode_to_tokens(signal)
        
        # Should return a list
        assert isinstance(tokens, list)
        
        # Test with confidence
        result = decoder.decode_to_tokens(signal, return_confidence=True)
        assert isinstance(result, dict)
        assert 'tokens' in result
        assert 'confidence' in result
        
        print(f"   ✓ Decoded {len(tokens)} tokens successfully")
        return True
    except Exception as e:
        print(f"   ✗ Synthetic decoding failed: {e}")
        traceback.print_exc()
        return False


def test_privacy_protection():
    """Test differential privacy functionality."""
    print("Testing privacy protection...")
    
    try:
        from bci2token import BrainDecoder
        from bci2token.privacy import PrivacyEngine
        
        # Create decoder with privacy
        decoder = BrainDecoder(
            signal_type='eeg',
            channels=8,
            sampling_rate=256,
            privacy_epsilon=1.0
        )
        
        # Should have privacy engine
        assert decoder.privacy_engine is not None
        
        # Test noise addition
        test_data = np.random.randn(8, 256)
        noisy_data = decoder.privacy_engine.add_noise(test_data)
        
        # Data should be different after noise addition
        assert not np.array_equal(test_data, noisy_data)
        
        # Generate privacy report
        report = decoder.privacy_engine.generate_privacy_report()
        assert 'privacy_parameters' in report
        
        print("   ✓ Privacy protection working")
        return True
    except Exception as e:
        print(f"   ✗ Privacy protection failed: {e}")
        traceback.print_exc()
        return False


def test_device_simulation():
    """Test simulated device functionality."""
    print("Testing simulated device...")
    
    try:
        from bci2token.devices import create_device, DeviceConfig
        
        # Create simulated device
        config = DeviceConfig(
            device_type='simulated',
            sampling_rate=256,
            n_channels=8
        )
        
        device = create_device('simulated', config)
        
        # Test connection
        assert device.connect()
        assert device.is_connected
        
        # Test data generation
        data = device.read_data()
        if data is not None:
            assert data.shape[0] == 8  # 8 channels
            
        # Test streaming setup
        device.start_streaming()
        assert device.is_streaming
        
        device.stop_streaming()
        assert not device.is_streaming
        
        device.disconnect()
        assert not device.is_connected
        
        print("   ✓ Simulated device working")
        return True
    except Exception as e:
        print(f"   ✗ Simulated device failed: {e}")
        traceback.print_exc()
        return False


def test_cli_basic():
    """Test CLI functionality."""
    print("Testing CLI module...")
    
    try:
        from bci2token.cli import create_parser, cmd_test
        
        # Test parser creation
        parser = create_parser()
        assert parser is not None
        
        # Test argument parsing
        args = parser.parse_args(['test', '--quick'])
        assert args.command == 'test'
        assert args.quick == True
        
        print("   ✓ CLI module working")
        return True
    except Exception as e:
        print(f"   ✗ CLI test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("BCI-2-Token Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_decoder_initialization,
        test_synthetic_decoding,
        test_privacy_protection,
        test_device_simulation,
        test_cli_basic
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ✗ Test {test_func.__name__} crashed: {e}")
            failed += 1
        print()
        
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())