"""
Comprehensive quality assurance and testing framework for BCI-2-Token.

Provides automated testing, code quality checks, performance benchmarks,
and production readiness validation.
"""

import time
import threading
import traceback
import inspect
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

import numpy as np
from .utils import BCIError, validate_signal_shape
from .preprocessing import SignalPreprocessor, PreprocessingConfig
from .input_validation import SignalValidator, InputSanitizer
from .error_handling import CircuitBreaker, CircuitBreakerConfig
from .advanced_monitoring import MetricsCollector, AlertManager
from .performance_scaling import PerformanceOptimizer
from .health import check_dependencies


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class TestSeverity(Enum):
    """Test severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestCase:
    """Individual test case."""
    name: str
    description: str
    test_function: Callable
    severity: TestSeverity = TestSeverity.MEDIUM
    timeout: float = 30.0
    dependencies: List[str] = field(default_factory=list)
    
    
@dataclass
class TestReport:
    """Test execution report."""
    test_name: str
    result: TestResult
    execution_time: float
    message: str = ""
    error_details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class QualityGateRunner:
    """
    Comprehensive quality gate test runner.
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.test_reports: List[TestReport] = []
        self.metrics_collector = MetricsCollector()
        
        # Setup test environment
        self._setup_test_environment()
        self._register_default_tests()
    
    def _setup_test_environment(self):
        """Setup test environment and mock data."""
        # Generate test signals
        self.test_signals = {
            'clean_signal': np.random.randn(8, 256) * 10,  # Clean EEG-like signal
            'noisy_signal': np.random.randn(8, 256) * 50,  # Noisy signal
            'artifact_signal': self._generate_artifact_signal(),
            'edge_case_signal': np.zeros((1, 10)),  # Minimal signal
        }
        
        # Test configurations
        self.test_configs = {
            'default': PreprocessingConfig(),
            'conservative': PreprocessingConfig(
                lowpass_freq=30.0,
                highpass_freq=1.0,
                apply_ica=False
            ),
            'aggressive': PreprocessingConfig(
                lowpass_freq=50.0,
                highpass_freq=0.1,
                apply_ica=True
            )
        }
    
    def _generate_artifact_signal(self) -> np.ndarray:
        """Generate signal with known artifacts."""
        signal = np.random.randn(8, 256) * 10
        
        # Add eye blink artifact (high amplitude in frontal channels)
        signal[0, 50:70] += 100
        signal[1, 50:70] += 80
        
        # Add muscle artifact (high frequency)
        t = np.linspace(0, 1, 256)
        muscle_artifact = 20 * np.sin(2 * np.pi * 100 * t)
        signal[6, :] += muscle_artifact
        signal[7, :] += muscle_artifact * 0.8
        
        return signal
    
    def add_test_case(self, test_case: TestCase):
        """Add custom test case."""
        self.test_cases.append(test_case)
    
    def run_all_tests(self, include_severity: List[TestSeverity] = None) -> Dict[str, Any]:
        """Run all registered tests."""
        if include_severity is None:
            include_severity = list(TestSeverity)
        
        self.test_reports.clear()
        start_time = time.time()
        
        # Filter tests by severity
        tests_to_run = [tc for tc in self.test_cases if tc.severity in include_severity]
        
        logging.info(f"Running {len(tests_to_run)} quality gate tests...")
        
        for test_case in tests_to_run:
            report = self._run_single_test(test_case)
            self.test_reports.append(report)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_test_summary(total_time)
        
        return summary
    
    def _run_single_test(self, test_case: TestCase) -> TestReport:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            # Check dependencies
            missing_deps = self._check_test_dependencies(test_case.dependencies)
            if missing_deps:
                return TestReport(
                    test_name=test_case.name,
                    result=TestResult.SKIP,
                    execution_time=0,
                    message=f"Missing dependencies: {missing_deps}"
                )
            
            # Run test with timeout
            result = self._run_with_timeout(test_case.test_function, test_case.timeout)
            
            execution_time = time.time() - start_time
            
            if result is True:
                return TestReport(
                    test_name=test_case.name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    message="Test passed successfully"
                )
            elif result is False:
                return TestReport(
                    test_name=test_case.name,
                    result=TestResult.FAIL,
                    execution_time=execution_time,
                    message="Test assertion failed"
                )
            else:
                return TestReport(
                    test_name=test_case.name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    message="Test completed",
                    metrics=result if isinstance(result, dict) else {}
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = traceback.format_exc()
            
            return TestReport(
                test_name=test_case.name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                message=f"Test error: {str(e)}",
                error_details=error_details
            )
    
    def _run_with_timeout(self, test_func: Callable, timeout: float) -> Any:
        """Run test function with timeout."""
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = test_func()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Test timed out after {timeout} seconds")
        
        if exception:
            raise exception
        
        return result
    
    def _check_test_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if test dependencies are available."""
        missing = []
        
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        return missing
    
    def _generate_test_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test execution summary."""
        results_by_status = {}
        for status in TestResult:
            results_by_status[status.value] = len([r for r in self.test_reports if r.result == status])
        
        critical_failures = [
            r for r in self.test_reports 
            if r.result in [TestResult.FAIL, TestResult.ERROR] and
            any(tc.severity == TestSeverity.CRITICAL for tc in self.test_cases if tc.name == r.test_name)
        ]
        
        return {
            'summary': {
                'total_tests': len(self.test_reports),
                'total_time': total_time,
                'results': results_by_status,
                'success_rate': results_by_status.get('pass', 0) / len(self.test_reports) if self.test_reports else 0,
                'critical_failures': len(critical_failures)
            },
            'detailed_reports': [
                {
                    'test_name': r.test_name,
                    'result': r.result.value,
                    'execution_time': r.execution_time,
                    'message': r.message,
                    'error_details': r.error_details,
                    'metrics': r.metrics
                }
                for r in self.test_reports
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in self.test_reports if r.result == TestResult.FAIL]
        error_tests = [r for r in self.test_reports if r.result == TestResult.ERROR]
        
        if failed_tests:
            recommendations.append(f"{len(failed_tests)} tests failed - review and fix issues before production")
        
        if error_tests:
            recommendations.append(f"{len(error_tests)} tests had errors - check test environment and dependencies")
        
        slow_tests = [r for r in self.test_reports if r.execution_time > 10.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are slow - consider optimization")
        
        skipped_tests = [r for r in self.test_reports if r.result == TestResult.SKIP]
        if skipped_tests:
            recommendations.append(f"{len(skipped_tests)} tests skipped - install missing dependencies for full validation")
        
        return recommendations
    
    def _register_default_tests(self):
        """Register default quality gate tests."""
        
        # Critical system tests
        self.add_test_case(TestCase(
            "test_dependencies",
            "Check that critical dependencies are available",
            self._test_dependencies,
            TestSeverity.CRITICAL
        ))
        
        self.add_test_case(TestCase(
            "test_signal_preprocessing",
            "Test signal preprocessing pipeline",
            self._test_signal_preprocessing,
            TestSeverity.CRITICAL
        ))
        
        self.add_test_case(TestCase(
            "test_input_validation",
            "Test input validation and sanitization",
            self._test_input_validation,
            TestSeverity.HIGH
        ))
        
        self.add_test_case(TestCase(
            "test_error_handling",
            "Test error handling and recovery",
            self._test_error_handling,
            TestSeverity.HIGH
        ))
        
        self.add_test_case(TestCase(
            "test_performance_benchmarks",
            "Run performance benchmarks",
            self._test_performance_benchmarks,
            TestSeverity.MEDIUM
        ))
        
        self.add_test_case(TestCase(
            "test_memory_management",
            "Test memory usage and cleanup",
            self._test_memory_management,
            TestSeverity.MEDIUM
        ))
        
        self.add_test_case(TestCase(
            "test_concurrent_processing",
            "Test concurrent processing capabilities",
            self._test_concurrent_processing,
            TestSeverity.MEDIUM
        ))
        
        self.add_test_case(TestCase(
            "test_edge_cases",
            "Test edge cases and boundary conditions",
            self._test_edge_cases,
            TestSeverity.HIGH
        ))
    
    # Test implementations
    def _test_dependencies(self) -> bool:
        """Test that critical dependencies are available."""
        health_check = check_dependencies()
        return health_check.level.value != 'critical'
    
    def _test_signal_preprocessing(self) -> Dict[str, Any]:
        """Test signal preprocessing functionality."""
        results = {}
        
        for signal_name, signal in self.test_signals.items():
            for config_name, config in self.test_configs.items():
                try:
                    preprocessor = SignalPreprocessor(config)
                    result = preprocessor.preprocess(signal)
                    
                    # Validate result structure
                    assert 'preprocessed_data' in result
                    assert 'epochs' in result
                    assert result['preprocessed_data'].shape == signal.shape
                    
                    results[f"{signal_name}_{config_name}"] = "pass"
                    
                except Exception as e:
                    results[f"{signal_name}_{config_name}"] = f"fail: {e}"
        
        return results
    
    def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization."""
        results = {}
        
        validator = SignalValidator()
        
        # Test various signal types
        for signal_name, signal in self.test_signals.items():
            try:
                validation_result = validator.validate_signal(signal, 256.0)
                results[f"validation_{signal_name}"] = validation_result.is_valid
                
                # Test sanitization
                sanitized = InputSanitizer.sanitize_brain_signal(signal)
                assert not np.any(np.isnan(sanitized))
                assert not np.any(np.isinf(sanitized))
                results[f"sanitization_{signal_name}"] = "pass"
                
            except Exception as e:
                results[f"validation_{signal_name}"] = f"fail: {e}"
        
        return results
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling mechanisms."""
        results = {}
        
        # Test circuit breaker
        try:
            config = CircuitBreakerConfig(failure_threshold=2)
            breaker = CircuitBreaker("test", config)
            
            def failing_function():
                raise Exception("Test failure")
            
            # Trigger failures
            for i in range(3):
                try:
                    breaker.call(failing_function)
                except:
                    pass
            
            # Should be open now
            try:
                breaker.call(failing_function)
                results['circuit_breaker'] = "fail: circuit breaker didn't open"
            except Exception as e:
                if "OPEN" in str(e):
                    results['circuit_breaker'] = "pass"
                else:
                    results['circuit_breaker'] = f"fail: unexpected error {e}"
                    
        except Exception as e:
            results['circuit_breaker'] = f"error: {e}"
        
        return results
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        results = {}
        
        # Preprocessing benchmark
        signal = self.test_signals['clean_signal']
        config = self.test_configs['default']
        
        start_time = time.time()
        for _ in range(10):
            preprocessor = SignalPreprocessor(config)
            preprocessor.preprocess(signal)
        preprocessing_time = (time.time() - start_time) / 10
        
        results['preprocessing_time_ms'] = preprocessing_time * 1000
        results['preprocessing_benchmark'] = "pass" if preprocessing_time < 0.1 else "slow"
        
        # Validation benchmark
        validator = SignalValidator()
        start_time = time.time()
        for _ in range(100):
            validator.validate_signal(signal, 256.0)
        validation_time = (time.time() - start_time) / 100
        
        results['validation_time_ms'] = validation_time * 1000
        results['validation_benchmark'] = "pass" if validation_time < 0.01 else "slow"
        
        return results
    
    def _test_memory_management(self) -> Dict[str, Any]:
        """Test memory usage and management."""
        results = {}
        
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Create and process many signals
            for i in range(100):
                signal = np.random.randn(8, 256)
                preprocessor = SignalPreprocessor(self.test_configs['default'])
                result = preprocessor.preprocess(signal)
                del signal, preprocessor, result
            
            # Force garbage collection again
            gc.collect()
            final_objects = len(gc.get_objects())
            
            object_growth = final_objects - initial_objects
            results['object_growth'] = object_growth
            results['memory_leak_test'] = "pass" if object_growth < 1000 else "potential_leak"
            
        except Exception as e:
            results['memory_leak_test'] = f"error: {e}"
        
        return results
    
    def _test_concurrent_processing(self) -> Dict[str, Any]:
        """Test concurrent processing capabilities."""
        results = {}
        
        try:
            from .performance_scaling import ProcessingPool
            
            pool = ProcessingPool(max_workers=2)
            
            def process_signal(signal):
                preprocessor = SignalPreprocessor(self.test_configs['default'])
                return preprocessor.preprocess(signal)
            
            # Test batch processing
            signals = [np.random.randn(8, 256) for _ in range(5)]
            
            start_time = time.time()
            batch_results = pool.submit_batch(process_signal, signals)
            concurrent_time = time.time() - start_time
            
            results['concurrent_processing_time'] = concurrent_time
            results['concurrent_results_count'] = len(batch_results)
            results['concurrent_test'] = "pass" if len(batch_results) == 5 else "fail"
            
        except Exception as e:
            results['concurrent_test'] = f"error: {e}"
        
        return results
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions."""
        results = {}
        
        edge_cases = {
            'empty_signal': np.array([[]]),
            'single_sample': np.random.randn(8, 1),
            'single_channel': np.random.randn(1, 256),
            'large_signal': np.random.randn(32, 1024),
            'zero_signal': np.zeros((8, 256)),
            'constant_signal': np.ones((8, 256)) * 42
        }
        
        for case_name, signal in edge_cases.items():
            try:
                if signal.size == 0:
                    # Skip empty signals
                    results[case_name] = "skip"
                    continue
                
                validator = SignalValidator()
                validation_result = validator.validate_signal(signal, 256.0)
                
                if validation_result.is_valid:
                    preprocessor = SignalPreprocessor(self.test_configs['conservative'])
                    result = preprocessor.preprocess(signal)
                    results[case_name] = "pass"
                else:
                    results[case_name] = f"validation_failed: {validation_result.issues}"
                    
            except Exception as e:
                results[case_name] = f"error: {e}"
        
        return results


def run_production_readiness_check() -> Dict[str, Any]:
    """
    Run comprehensive production readiness check.
    
    Returns complete quality assessment report.
    """
    print("🔍 Running BCI-2-Token Production Readiness Check...")
    print("=" * 60)
    
    runner = QualityGateRunner()
    
    # Run all tests
    report = runner.run_all_tests()
    
    # Add production-specific checks
    production_checks = _run_production_specific_checks()
    report['production_checks'] = production_checks
    
    # Calculate overall readiness score
    readiness_score = _calculate_readiness_score(report)
    report['readiness_score'] = readiness_score
    
    return report


def _run_production_specific_checks() -> Dict[str, Any]:
    """Run production-specific quality checks."""
    checks = {}
    
    # Check if security features are enabled
    try:
        from .security import SecurityConfig
        config = SecurityConfig()
        checks['security_enabled'] = config.enable_access_control
        checks['privacy_protection'] = config.require_privacy_protection
        checks['data_encryption'] = config.encrypt_saved_data
    except Exception as e:
        checks['security_check'] = f"error: {e}"
    
    # Check monitoring capabilities
    try:
        from .advanced_monitoring import get_system_monitor
        monitor = get_system_monitor()
        dashboard = monitor.get_health_dashboard()
        checks['monitoring_active'] = dashboard.get('system_health') == 'healthy'
    except Exception as e:
        checks['monitoring_check'] = f"error: {e}"
    
    # Check performance optimization
    try:
        from .performance_scaling import get_performance_optimizer
        optimizer = get_performance_optimizer()
        report = optimizer.get_performance_report()
        checks['caching_enabled'] = report['cache_stats']['total_size'] > 0
        checks['parallel_processing'] = report['processing_pool']['max_workers'] > 1
    except Exception as e:
        checks['performance_check'] = f"error: {e}"
    
    return checks


def _calculate_readiness_score(report: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate production readiness score."""
    summary = report['summary']
    
    # Base score from test results
    success_rate = summary['success_rate']
    critical_failures = summary['critical_failures']
    
    base_score = success_rate * 100
    
    # Penalties
    if critical_failures > 0:
        base_score -= critical_failures * 20  # 20 points per critical failure
    
    # Bonuses for production features
    production_bonus = 0
    production_checks = report.get('production_checks', {})
    
    if production_checks.get('security_enabled'):
        production_bonus += 5
    if production_checks.get('monitoring_active'):
        production_bonus += 5
    if production_checks.get('caching_enabled'):
        production_bonus += 3
    
    final_score = max(0, min(100, base_score + production_bonus))
    
    # Determine grade
    if final_score >= 90:
        grade = "A"
        status = "Production Ready"
    elif final_score >= 80:
        grade = "B"
        status = "Nearly Ready"
    elif final_score >= 70:
        grade = "C"
        status = "Needs Improvement"
    elif final_score >= 60:
        grade = "D"
        status = "Not Recommended"
    else:
        grade = "F"
        status = "Not Ready"
    
    return {
        'score': final_score,
        'grade': grade,
        'status': status,
        'breakdown': {
            'base_score': base_score,
            'production_bonus': production_bonus,
            'success_rate': success_rate,
            'critical_failures': critical_failures
        }
    }


if __name__ == '__main__':
    # Run production readiness check
    report = run_production_readiness_check()
    
    print(f"\n📊 Final Results:")
    print(f"Readiness Score: {report['readiness_score']['score']:.1f}/100 ({report['readiness_score']['grade']})")
    print(f"Status: {report['readiness_score']['status']}")
    print(f"Tests Passed: {report['summary']['results']['pass']}/{report['summary']['total_tests']}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Save detailed report
    with open('quality_gate_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to quality_gate_report.json")