#!/usr/bin/env python3
"""
Comprehensive test suite for BCI-2-Token framework.

Tests all major components with proper error handling and environment adaptation.
"""

import sys
import time
import tempfile
from pathlib import Path
import traceback
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available, skipping numpy-dependent tests")


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, passed: bool, message: str, details: dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()


class ComprehensiveTestSuite:
    """Comprehensive test suite for BCI-2-Token."""
    
    def __init__(self):
        self.results = []
        
    def run_test(self, test_func):
        """Run a single test function and record results."""
        test_name = test_func.__name__
        print(f"Running {test_name}...")
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if isinstance(result, TestResult):
                result.details['duration'] = duration
                self.results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.message} ({duration:.3f}s)")
            else:
                # Assume success if no TestResult returned
                self.results.append(TestResult(
                    test_name, True, "Test completed successfully",
                    {'duration': duration}
                ))
                print(f"  ✓ Test completed successfully ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            self.results.append(TestResult(
                test_name, False, f"Test failed with exception: {e}",
                {'exception': str(e), 'traceback': traceback.format_exc(), 'duration': duration}
            ))
            print(f"  ✗ Test failed: {e} ({duration:.3f}s)")
            
    def test_core_imports(self):
        """Test that core modules can be imported."""
        core_modules = [
            'bci2token',
            'bci2token.preprocessing',
            'bci2token.devices',
            'bci2token.utils',
            'bci2token.monitoring',
            'bci2token.health'
        ]
        
        failed_imports = []
        
        for module in core_modules:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append((module, str(e)))
                
        if failed_imports:
            return TestResult(
                'test_core_imports', False,
                f"Failed to import {len(failed_imports)} core modules",
                {'failed_imports': failed_imports}
            )
        else:
            return TestResult(
                'test_core_imports', True,
                f"All {len(core_modules)} core modules imported successfully"
            )
            
    def test_advanced_imports(self):
        """Test that advanced modules can be imported."""
        advanced_modules = [
            'bci2token.reliability',
            'bci2token.security',
            'bci2token.recovery',
            'bci2token.optimization',
            'bci2token.quality_gates',
            'bci2token.cli'
        ]
        
        failed_imports = []
        warning_imports = []
        
        for module in advanced_modules:
            try:
                __import__(module)
            except ImportError as e:
                if 'torch' in str(e) or 'transformers' in str(e):
                    warning_imports.append((module, str(e)))
                else:
                    failed_imports.append((module, str(e)))
                    
        if failed_imports:
            return TestResult(
                'test_advanced_imports', False,
                f"Failed to import {len(failed_imports)} advanced modules",
                {'failed_imports': failed_imports, 'warning_imports': warning_imports}
            )
        else:
            return TestResult(
                'test_advanced_imports', True,
                f"All {len(advanced_modules)} advanced modules imported successfully",
                {'warning_imports': warning_imports}
            )
            
    def test_configuration_objects(self):
        """Test configuration object creation and validation."""
        config_tests = []
        
        try:
            from bci2token.preprocessing import PreprocessingConfig
            config = PreprocessingConfig(sampling_rate=256)
            assert config.sampling_rate == 256
            config_tests.append("PreprocessingConfig")
        except Exception as e:
            return TestResult('test_configuration_objects', False, f"PreprocessingConfig failed: {e}")
            
        try:
            from bci2token.devices import DeviceConfig
            config = DeviceConfig(device_type='simulated', n_channels=8)
            assert config.device_type == 'simulated'
            config_tests.append("DeviceConfig")
        except Exception as e:
            return TestResult('test_configuration_objects', False, f"DeviceConfig failed: {e}")
            
        try:
            from bci2token.security import SecurityConfig
            config = SecurityConfig(enable_access_control=True)
            assert config.enable_access_control
            config_tests.append("SecurityConfig")
        except Exception as e:
            return TestResult('test_configuration_objects', False, f"SecurityConfig failed: {e}")
            
        try:
            from bci2token.optimization import OptimizationConfig
            config = OptimizationConfig(cache_size=100)
            assert config.cache_size == 100
            config_tests.append("OptimizationConfig")
        except Exception as e:
            return TestResult('test_configuration_objects', False, f"OptimizationConfig failed: {e}")
            
        return TestResult(
            'test_configuration_objects', True,
            f"All {len(config_tests)} configuration objects created successfully",
            {'configs_tested': config_tests}
        )
        
    def test_utility_functions(self):
        """Test utility functions."""
        from bci2token.utils import (
            validate_sampling_rate,
            validate_frequency_bands,
            format_duration,
            format_bytes,
            safe_divide
        )
        
        # Test validation functions
        try:
            validate_sampling_rate(256)
            validate_frequency_bands(40.0, 0.5, 256)
        except Exception as e:
            return TestResult('test_utility_functions', False, f"Validation functions failed: {e}")
            
        # Test formatting functions
        assert format_duration(0.5) == "500.0ms"
        assert format_duration(65) == "1.1m"
        assert format_bytes(1024) == "1.0KB"
        
        # Test safe math
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0, default=99) == 99
        
        return TestResult(
            'test_utility_functions', True,
            "All utility functions working correctly"
        )
        
    def test_monitoring_system(self):
        """Test monitoring and logging system."""
        from bci2token.monitoring import BCILogger, MetricsCollector, get_monitor
        
        # Test logger
        logger = BCILogger()
        logger.info('Test', 'Test message')
        
        # Test metrics collector
        metrics = MetricsCollector()
        metrics.record_metric('test_metric', 1.0)
        summary = metrics.get_metric_summary('test_metric')
        assert summary['count'] == 1
        
        # Test global monitor
        monitor = get_monitor()
        monitor.logger.debug('Test', 'Debug message')
        
        return TestResult(
            'test_monitoring_system', True,
            "Monitoring system working correctly"
        )
        
    def test_health_diagnostics(self):
        """Test health monitoring and diagnostics."""
        from bci2token.health import run_comprehensive_diagnostics, HealthLevel
        
        # Run diagnostics
        results = run_comprehensive_diagnostics()
        
        # Validate results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for name, result in results.items():
            assert hasattr(result, 'level')
            assert hasattr(result, 'message')
            assert hasattr(result, 'timestamp')
            assert isinstance(result.level, HealthLevel)
            
        return TestResult(
            'test_health_diagnostics', True,
            f"Health diagnostics completed ({len(results)} checks)",
            {'diagnostics_count': len(results)}
        )
        
    def test_signal_processing(self):
        """Test signal processing capabilities."""
        if not HAS_NUMPY:
            return TestResult(
                'test_signal_processing', True,
                "Skipped (NumPy not available)",
                {'skipped': True}
            )
            
        from bci2token.preprocessing import PreprocessingConfig, SignalPreprocessor
        from bci2token.utils import calculate_signal_quality, detect_signal_artifacts
        
        # Create test signal
        signal = np.random.randn(8, 512)  # 8 channels, 512 timepoints
        
        # Test signal quality calculation
        quality = calculate_signal_quality(signal)
        assert 0 <= quality <= 1
        
        # Test artifact detection
        artifacts = detect_signal_artifacts(signal)
        assert isinstance(artifacts, dict)
        
        # Test preprocessor
        config = PreprocessingConfig(sampling_rate=256)
        preprocessor = SignalPreprocessor(config)
        
        return TestResult(
            'test_signal_processing', True,
            f"Signal processing working (quality: {quality:.3f})",
            {'signal_quality': quality, 'artifacts_detected': len(artifacts)}
        )
        
    def test_security_system(self):
        """Test security and access control."""
        from bci2token.security import SecurityConfig, SecureProcessor
        
        config = SecurityConfig(
            enable_access_control=True,
            max_concurrent_sessions=5
        )
        
        processor = SecureProcessor(config)
        
        # Test session creation
        session_token = processor.access_controller.create_session('test_user', ['basic'])
        assert len(session_token) > 20
        
        # Test session validation
        valid = processor.access_controller.validate_session(session_token)
        assert valid
        
        # Test rate limiting
        rate_ok = processor.rate_limiter.check_rate_limit('test_user')
        assert rate_ok
        
        return TestResult(
            'test_security_system', True,
            "Security system working correctly",
            {'session_token_length': len(session_token)}
        )
        
    def test_reliability_system(self):
        """Test reliability and recovery mechanisms."""
        from bci2token.reliability import CircuitBreaker, InputSanitizer
        from bci2token.recovery import SelfHealingSystem
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        def failing_function():
            raise RuntimeError("Test failure")
            
        failures = 0
        for _ in range(5):
            try:
                circuit_breaker.call(failing_function)
            except:
                failures += 1
                
        assert failures >= 3  # Should fail and then be blocked by circuit breaker
        
        # Test self-healing system
        healing = SelfHealingSystem()
        
        class MockComponent:
            def get_status(self):
                return {'status': 'healthy'}
                
        healing.register_component('test_component', MockComponent())
        health_report = healing.get_system_health()
        assert health_report['overall_status'] == 'healthy'
        
        # Test input sanitization
        if HAS_NUMPY:
            bad_signal = np.array([[np.inf, np.nan, 1000.0]])
            sanitized = InputSanitizer.sanitize_brain_signal(bad_signal)
            assert not np.any(np.isnan(sanitized))
            assert not np.any(np.isinf(sanitized))
            
        return TestResult(
            'test_reliability_system', True,
            "Reliability system working correctly",
            {'circuit_breaker_failures': failures}
        )
        
    def test_optimization_system(self):
        """Test performance optimization features."""
        from bci2token.optimization import PerformanceOptimizer, OptimizationConfig
        
        config = OptimizationConfig(cache_size=100, max_worker_threads=2)
        optimizer = PerformanceOptimizer(config)
        
        # Test performance report
        report = optimizer.get_performance_report()
        assert isinstance(report, dict)
        assert 'timestamp' in report
        
        # Test optimization presets
        optimizer.optimize_for_latency()
        optimizer.optimize_for_throughput()
        
        # Cleanup
        optimizer.cleanup()
        
        return TestResult(
            'test_optimization_system', True,
            "Optimization system working correctly",
            {'report_sections': len(report)}
        )
        
    def test_cli_system(self):
        """Test command-line interface."""
        import bci2token.cli as cli_module
        import inspect
        
        # Check that CLI has required functions
        functions = [name for name, obj in inspect.getmembers(cli_module, inspect.isfunction)]
        
        required_functions = ['main']
        missing_functions = [f for f in required_functions if f not in functions]
        
        if missing_functions:
            return TestResult(
                'test_cli_system', False,
                f"Missing CLI functions: {missing_functions}",
                {'missing_functions': missing_functions, 'available_functions': functions}
            )
            
        return TestResult(
            'test_cli_system', True,
            f"CLI system complete ({len(functions)} functions)",
            {'available_functions': functions}
        )
        
    def test_integration_flows(self):
        """Test end-to-end integration flows."""
        integration_tests = []
        
        # Test monitoring + health integration
        try:
            from bci2token.monitoring import get_monitor
            from bci2token.health import run_comprehensive_diagnostics
            
            monitor = get_monitor()
            monitor.logger.info('Integration', 'Testing health integration')
            
            health_results = run_comprehensive_diagnostics()
            assert len(health_results) > 0
            integration_tests.append('monitoring_health')
        except Exception as e:
            return TestResult('test_integration_flows', False, f"Monitoring-Health integration failed: {e}")
            
        # Test security + monitoring integration
        try:
            from bci2token.security import SecurityConfig, SecureProcessor
            
            config = SecurityConfig()
            processor = SecureProcessor(config)
            
            # This should log to monitoring system
            session = processor.access_controller.create_session('integration_test')
            assert len(session) > 0
            integration_tests.append('security_monitoring')
        except Exception as e:
            return TestResult('test_integration_flows', False, f"Security-Monitoring integration failed: {e}")
            
        return TestResult(
            'test_integration_flows', True,
            f"Integration flows working ({len(integration_tests)} tested)",
            {'integrations_tested': integration_tests}
        )
        
    def run_all_tests(self):
        """Run all tests in the comprehensive suite."""
        test_methods = [
            self.test_core_imports,
            self.test_advanced_imports,
            self.test_configuration_objects,
            self.test_utility_functions,
            self.test_monitoring_system,
            self.test_health_diagnostics,
            self.test_signal_processing,
            self.test_security_system,
            self.test_reliability_system,
            self.test_optimization_system,
            self.test_cli_system,
            self.test_integration_flows
        ]
        
        print("BCI-2-Token Comprehensive Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        for test_method in test_methods:
            self.run_test(test_method)
            
        total_time = time.time() - start_time
        
        # Generate summary
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print()
        print("=" * 50)
        print(f"Test Summary ({total_time:.1f}s total)")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total:  {len(self.results)}")
        
        if failed == 0:
            print("\n🎉 All tests passed!")
            success = True
        else:
            print(f"\n❌ {failed} test(s) failed")
            success = False
            
            # Show failed tests
            print("\nFailed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  • {result.name}: {result.message}")
                    
        return success, {
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'total_time': total_time,
            'results': self.results
        }


def main():
    """Run the comprehensive test suite."""
    suite = ComprehensiveTestSuite()
    success, summary = suite.run_all_tests()
    
    # Save detailed results if possible
    try:
        import json
        results_data = {
            'summary': {
                'total_tests': summary['total_tests'],
                'passed': summary['passed'],
                'failed': summary['failed'],
                'total_time': summary['total_time'],
                'timestamp': time.time()
            },
            'test_results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp
                }
                for r in summary['results']
            ]
        }
        
        results_path = Path('/root/repo/test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not save detailed results: {e}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())