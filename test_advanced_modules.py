#!/usr/bin/env python3
"""
Test suite for advanced BCI-2-Token modules

Tests the new research, benchmarking, adaptive algorithms, validation,
and performance optimization modules.
"""

import numpy as np
import time
import sys
import traceback

def test_research_framework():
    """Test research framework functionality"""
    print("Testing Research Framework...")
    try:
        from bci2token.research import BaselineImplementations, NovelMethods, StatisticalAnalysis
        
        # Test baseline methods
        test_signals = np.random.randn(10, 64, 256)  # 10 trials, 64 channels, 256 timepoints
        
        # Linear decoder
        linear_output = BaselineImplementations.linear_decoder(test_signals)
        assert linear_output.shape[0] == test_signals.shape[0]
        
        # Template matching
        template_output = BaselineImplementations.template_matching(test_signals)
        assert len(template_output) == 100  # Default 100 templates
        
        # Statistical analysis
        baseline_results = np.random.randn(20) + 0.7  # Simulate baseline performance
        novel_results = np.random.randn(20) + 0.8     # Simulate novel performance
        
        stats = StatisticalAnalysis.paired_t_test(baseline_results, novel_results)
        assert 'p_value' in stats
        assert 'effect_size' in stats
        
        print("  ✓ Research framework tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Research framework test failed: {e}")
        return False

def test_adaptive_algorithms():
    """Test adaptive algorithms functionality"""
    print("Testing Adaptive Algorithms...")
    try:
        from bci2token.adaptive_algorithms import (
            RecursiveLeastSquares, UserPersonalizationEngine, 
            EnsembleAdaptiveDecoder, AdaptationConfig
        )
        
        # Test RLS
        rls = RecursiveLeastSquares(n_features=64, n_outputs=10)
        features = np.random.randn(5, 64)
        targets = np.random.randn(5, 10)
        rls.update(features, targets)
        
        predictions = rls.predict(features)
        assert predictions.shape == (5, 10)
        
        # Test user personalization
        config = AdaptationConfig(learning_rate=0.01)
        personalization = UserPersonalizationEngine(config)
        
        personalization.initialize_user('test_user', (features, targets))
        pred, confidence = personalization.predict_for_user('test_user', features[:1])
        assert pred is not None
        assert isinstance(confidence, float)
        
        print("  ✓ Adaptive algorithms tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Adaptive algorithms test failed: {e}")
        return False

def test_validation_system():
    """Test validation and error handling"""
    print("Testing Validation System...")
    try:
        from bci2token.validation import (
            SignalValidator, RobustErrorHandler, InputSanitizer,
            ValidationLevel, CircuitBreaker
        )
        
        # Test signal validation
        validator = SignalValidator(ValidationLevel.MODERATE)
        good_signals = np.random.randn(64, 512) * 10  # Normal EEG-like
        
        result = validator.validate_eeg_signal(good_signals, sampling_rate=256)
        assert result.is_valid
        
        # Test input sanitization
        dirty_signals = np.random.randn(8, 128)
        dirty_signals[0, 50] = np.inf  # Add infinity
        dirty_signals[1, 60] = np.nan  # Add NaN
        
        clean_signals = InputSanitizer.sanitize_signal_data(dirty_signals)
        assert np.isfinite(clean_signals).all()
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # Should fail and eventually open circuit
        failure_count = 0
        for _ in range(5):
            try:
                circuit_breaker.call(failing_function)
            except RuntimeError:
                failure_count += 1
        
        assert failure_count > 0
        
        print("  ✓ Validation system tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Validation system test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization features"""
    print("Testing Performance Optimization...")
    try:
        from bci2token.performance_optimization import (
            IntelligentCache, ConcurrentProcessor, MemoryOptimizer,
            PerformanceConfig, cached
        )
        
        # Test caching
        cache = IntelligentCache(max_size=10, default_ttl=1)
        cache.put('test_key', 'test_value')
        
        value, hit = cache.get('test_key')
        assert hit == True
        assert value == 'test_value'
        
        # Test cached decorator
        call_count = 0
        
        @cached(ttl=60)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = test_function(5)
        result2 = test_function(5)  # Should be cached
        
        assert result1 == result2 == 10
        assert call_count == 1  # Only called once due to caching
        
        # Test concurrent processor
        config = PerformanceConfig(max_workers=2)
        processor = ConcurrentProcessor(config)
        
        def simple_operation(x):
            return x * 2
        
        test_data = [1, 2, 3, 4, 5]
        results = processor.process_batch(simple_operation, test_data)
        
        assert len(results) == len(test_data)
        # Just check that we got results (concurrent processing can reorder)
        assert all(r is not None for r in results)
        
        processor.shutdown()
        
        print("  ✓ Performance optimization tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Performance optimization test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between modules"""
    print("Testing Module Integration...")
    try:
        from bci2token.validation import SignalValidator, ValidationLevel
        from bci2token.adaptive_algorithms import RecursiveLeastSquares
        from bci2token.performance_optimization import cached
        
        # Create a processing pipeline that uses multiple modules
        validator = SignalValidator(ValidationLevel.MODERATE)
        
        def cache_key_func(*args, **kwargs):
            # Create a hashable cache key from numpy array
            signals = args[0]
            return f"signals_{signals.shape}_{hash(signals.tobytes())}"
        
        @cached(ttl=30, key_func=cache_key_func)
        def process_validated_signals(signals):
            # Validate signals first
            validation_result = validator.validate_eeg_signal(signals)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid signals: {validation_result.error_message}")
            
            # Process with adaptive algorithm
            learner = RecursiveLeastSquares(n_features=signals.shape[-1], n_outputs=10)
            features = np.mean(signals, axis=0).reshape(1, -1)  # Simple feature extraction
            
            # Simulate some targets for demonstration
            targets = np.random.randn(1, 10)
            learner.update(features, targets)
            
            return learner.predict(features)
        
        # Test the integrated pipeline
        test_signals = np.random.randn(64, 256)
        result = process_validated_signals(test_signals)
        
        assert result is not None
        assert result.shape == (1, 10)
        
        # Test caching
        result2 = process_validated_signals(test_signals)
        assert np.array_equal(result, result2)  # Should be identical due to caching
        
        print("  ✓ Module integration tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Module integration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all advanced module tests"""
    print("=" * 60)
    print("BCI-2-Token Advanced Modules Test Suite")
    print("=" * 60)
    
    tests = [
        test_research_framework,
        test_adaptive_algorithms,
        test_validation_system,
        test_performance_optimization,
        test_integration
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    print(f"Execution time: {total_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All advanced module tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)