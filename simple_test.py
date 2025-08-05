#!/usr/bin/env python3
"""
Simple structure test for BCI-2-Token framework.
Tests basic imports and structure without PyTorch dependencies.
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_structure():
    """Test basic module structure."""
    print("Testing module structure...")
    
    try:
        # Test basic imports that don't require PyTorch
        import bci2token
        print(f"   ✓ Main package version: {bci2token.__version__}")
        
        # Test individual modules exist
        modules = [
            'bci2token.cli',
            'bci2token.preprocessing', 
            'bci2token.privacy',
            'bci2token.devices'
        ]
        
        for module in modules:
            try:
                __import__(module)
                print(f"   ✓ {module} imported")
            except ImportError as e:
                print(f"   ✗ {module} failed: {e}")
                
        return True
    except Exception as e:
        print(f"   ✗ Structure test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_parser():
    """Test CLI parser creation."""
    print("Testing CLI parser...")
    
    try:
        from bci2token.cli import create_parser
        
        parser = create_parser()
        
        # Test help doesn't crash
        help_text = parser.format_help()
        assert 'decode' in help_text
        assert 'stream' in help_text
        assert 'train' in help_text
        
        print("   ✓ CLI parser working")
        return True
    except Exception as e:
        print(f"   ✗ CLI parser failed: {e}")
        traceback.print_exc()
        return False


def test_preprocessing_config():
    """Test preprocessing configuration."""
    print("Testing preprocessing config...")
    
    try:
        from bci2token.preprocessing import PreprocessingConfig
        
        config = PreprocessingConfig(
            sampling_rate=256,
            lowpass_freq=40.0,
            highpass_freq=0.5
        )
        
        assert config.sampling_rate == 256
        assert config.lowpass_freq == 40.0
        
        print("   ✓ Preprocessing config working")
        return True
    except Exception as e:
        print(f"   ✗ Preprocessing config failed: {e}")
        traceback.print_exc()
        return False


def test_device_config():
    """Test device configuration."""
    print("Testing device config...")
    
    try:
        from bci2token.devices import DeviceConfig
        
        config = DeviceConfig(
            device_type='simulated',
            sampling_rate=256,
            n_channels=8
        )
        
        assert config.device_type == 'simulated'
        assert config.sampling_rate == 256
        assert config.n_channels == 8
        
        print("   ✓ Device config working")
        return True
    except Exception as e:
        print(f"   ✗ Device config failed: {e}")
        traceback.print_exc()
        return False


def test_privacy_config():
    """Test privacy configuration."""
    print("Testing privacy config...")
    
    try:
        from bci2token.privacy import PrivacyConfig
        
        config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            mechanism='gaussian'
        )
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.mechanism == 'gaussian'
        
        print("   ✓ Privacy config working")
        return True
    except Exception as e:
        print(f"   ✗ Privacy config failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run simple structure tests."""
    print("BCI-2-Token Simple Structure Test")
    print("=" * 40)
    
    tests = [
        test_structure,
        test_cli_parser,
        test_preprocessing_config,
        test_device_config,
        test_privacy_config
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
        
    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 Structure tests passed!")
        return 0
    else:
        print("❌ Some structure tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())