#!/usr/bin/env python3
"""
Basic test suite for BCI-2-Token framework that works without PyTorch.

Tests core functionality that doesn't require machine learning dependencies.
"""

import sys
from pathlib import Path
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that individual modules can be imported."""
    print("Testing module imports...")
    
    tests = [
        ('preprocessing config', 'from bci2token.preprocessing import PreprocessingConfig'),
        ('device config', 'from bci2token.devices import DeviceConfig'),
        ('utility functions', 'from bci2token.utils import validate_sampling_rate'),
        ('health checks', 'from bci2token.health import run_comprehensive_diagnostics'),
        ('monitoring', 'from bci2token.monitoring import BCILogger')
    ]
    
    passed = 0
    for test_name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"   ✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"   ✗ {test_name}: {e}")
            
    return passed, len(tests) - passed


def test_configurations():
    """Test configuration objects."""
    print("Testing configuration objects...")
    
    try:
        from bci2token.preprocessing import PreprocessingConfig
        from bci2token.devices import DeviceConfig
        
        # Test preprocessing config
        prep_config = PreprocessingConfig(
            sampling_rate=256,
            lowpass_freq=40.0,
            highpass_freq=0.5
        )
        assert prep_config.sampling_rate == 256
        print("   ✓ PreprocessingConfig")
        
        # Test device config
        dev_config = DeviceConfig(
            device_type='simulated',
            sampling_rate=256,
            n_channels=8
        )
        assert dev_config.device_type == 'simulated'
        print("   ✓ DeviceConfig")
        
        return 2, 0
        
    except Exception as e:
        print(f"   ✗ Configuration test failed: {e}")
        return 0, 2


def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")
    
    try:
        from bci2token.utils import (
            validate_sampling_rate, 
            validate_frequency_bands,
            format_duration,
            format_bytes,
            safe_divide
        )
        
        # Test validation functions
        validate_sampling_rate(256)
        validate_frequency_bands(40.0, 0.5, 256)
        print("   ✓ Validation functions")
        
        # Test formatting functions
        assert format_duration(0.5) == "500.0ms"
        assert format_duration(65) == "1.1m"
        assert format_bytes(1024) == "1.0KB"
        print("   ✓ Formatting functions")
        
        # Test safe math
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0, default=99) == 99
        print("   ✓ Safe math functions")
        
        return 3, 0
        
    except Exception as e:
        print(f"   ✗ Utility test failed: {e}")
        traceback.print_exc()
        return 0, 3


def test_health_system():
    """Test health monitoring system."""
    print("Testing health monitoring...")
    
    try:
        from bci2token.health import run_comprehensive_diagnostics, HealthLevel
        
        # Run diagnostics
        results = run_comprehensive_diagnostics()
        
        # Should return dictionary
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that results have expected structure
        for name, result in results.items():
            assert hasattr(result, 'level')
            assert hasattr(result, 'message')
            assert hasattr(result, 'timestamp')
            
        print(f"   ✓ Health diagnostics ({len(results)} checks)")
        
        return 1, 0
        
    except Exception as e:
        print(f"   ✗ Health system test failed: {e}")
        traceback.print_exc()
        return 0, 1


def test_monitoring():
    """Test monitoring and logging system."""
    print("Testing monitoring system...")
    
    try:
        from bci2token.monitoring import BCILogger, MetricsCollector
        
        # Test logger
        logger = BCILogger()
        logger.info('Test', 'Test message')
        print("   ✓ Logger creation and basic logging")
        
        # Test metrics collector
        metrics = MetricsCollector()
        metrics.record_metric('test_metric', 1.0)
        
        summary = metrics.get_metric_summary('test_metric')
        assert summary['count'] == 1
        print("   ✓ Metrics collection")
        
        return 2, 0
        
    except Exception as e:
        print(f"   ✗ Monitoring test failed: {e}")
        traceback.print_exc()
        return 0, 2


def test_signal_processing():
    """Test signal processing without ML dependencies."""
    print("Testing signal processing...")
    
    try:
        import numpy as np
        from bci2token.preprocessing import PreprocessingConfig, SignalPreprocessor
        from bci2token.utils import calculate_signal_quality, detect_signal_artifacts
        
        # Create test signal
        signal = np.random.randn(8, 512)  # 8 channels, 512 timepoints
        
        # Test signal quality calculation
        quality = calculate_signal_quality(signal)
        assert 0 <= quality <= 1
        print(f"   ✓ Signal quality calculation: {quality:.3f}")
        
        # Test artifact detection
        artifacts = detect_signal_artifacts(signal)
        assert isinstance(artifacts, dict)
        print("   ✓ Artifact detection")
        
        # Test preprocessing config
        config = PreprocessingConfig(sampling_rate=256)
        preprocessor = SignalPreprocessor(config)
        print("   ✓ Preprocessor creation")
        
        return 3, 0
        
    except Exception as e:
        print(f"   ✗ Signal processing test failed: {e}")
        traceback.print_exc()
        return 0, 3


def main():
    """Run all basic tests."""
    print("BCI-2-Token Basic Test Suite")
    print("=" * 50)
    print("Note: Testing core functionality without ML dependencies")
    print()
    
    test_functions = [
        test_imports,
        test_configurations,
        test_utilities,
        test_health_system,
        test_monitoring,
        test_signal_processing
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_func in test_functions:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"   ✗ Test {test_func.__name__} crashed: {e}")
            total_failed += 1
        print()
        
    print("=" * 50)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Install transformers: pip install transformers")
        print("3. Run full test suite: python -m pytest tests/")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())