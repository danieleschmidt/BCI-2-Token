"""
Quality gates and comprehensive testing framework for BCI-2-Token.

Implements automated quality checks, performance benchmarks, security validation,
and comprehensive testing to ensure production readiness.
"""

import time
import threading
import subprocess
import sys
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings
import tempfile
import shutil

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class QualityLevel(Enum):
    """Quality check result levels."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class QualityResult:
    """Result of a quality check."""
    name: str
    level: QualityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    
    def is_passing(self) -> bool:
        """Check if result is passing (not fail or critical)."""
        return self.level in [QualityLevel.PASS, QualityLevel.WARNING]


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str, timeout: float = 30.0):
        self.name = name
        self.description = description
        self.timeout = timeout
        
    def run(self) -> QualityResult:
        """Run the quality check."""
        start_time = time.time()
        
        try:
            result = self._execute()
            result.duration = time.time() - start_time
            return result
        except Exception as e:
            return QualityResult(
                name=self.name,
                level=QualityLevel.CRITICAL,
                message=f"Quality gate failed with exception: {e}",
                details={'exception': str(e), 'type': type(e).__name__},
                duration=time.time() - start_time
            )
            
    def _execute(self) -> QualityResult:
        """Override this method in subclasses."""
        raise NotImplementedError


class ImportQualityGate(QualityGate):
    """Validates that all modules can be imported correctly."""
    
    def __init__(self):
        super().__init__(
            "import_validation",
            "Validate all modules can be imported without errors"
        )
        
    def _execute(self) -> QualityResult:
        """Test all module imports."""
        modules_to_test = [
            'bci2token',
            'bci2token.preprocessing',
            'bci2token.devices',
            'bci2token.utils',
            'bci2token.monitoring',
            'bci2token.health',
            'bci2token.reliability',
            'bci2token.security',
            'bci2token.recovery',
            'bci2token.optimization',
            'bci2token.cli'
        ]
        
        failed_imports = []
        warning_imports = []
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                if 'torch' in str(e) or 'transformers' in str(e):
                    warning_imports.append((module_name, str(e)))
                else:
                    failed_imports.append((module_name, str(e)))
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                
        if failed_imports:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Critical import failures: {len(failed_imports)} modules",
                details={
                    'failed_imports': failed_imports,
                    'warning_imports': warning_imports,
                    'total_tested': len(modules_to_test)
                }
            )
        elif warning_imports:
            return QualityResult(
                name=self.name,
                level=QualityLevel.WARNING,
                message=f"Optional dependencies missing: {len(warning_imports)} modules",
                details={
                    'warning_imports': warning_imports,
                    'total_tested': len(modules_to_test)
                }
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message=f"All {len(modules_to_test)} modules imported successfully",
                details={'total_tested': len(modules_to_test)}
            )


class UnitTestQualityGate(QualityGate):
    """Runs unit tests and validates coverage."""
    
    def __init__(self):
        super().__init__(
            "unit_tests",
            "Run unit tests and validate test coverage",
            timeout=120.0
        )
        
    def _execute(self) -> QualityResult:
        """Run basic unit tests."""
        try:
            # Run our basic test suite
            result = subprocess.run([
                sys.executable, 'test_basic.py'
            ], capture_output=True, text=True, timeout=self.timeout, cwd='/root/repo')
            
            if result.returncode == 0:
                return QualityResult(
                    name=self.name,
                    level=QualityLevel.PASS,
                    message="All unit tests passed",
                    details={
                        'exit_code': result.returncode,
                        'stdout': result.stdout[-500:] if result.stdout else '',
                        'test_runner': 'test_basic.py'
                    }
                )
            else:
                return QualityResult(
                    name=self.name,
                    level=QualityLevel.FAIL,
                    message="Unit tests failed",
                    details={
                        'exit_code': result.returncode,
                        'stdout': result.stdout[-500:] if result.stdout else '',
                        'stderr': result.stderr[-500:] if result.stderr else ''
                    }
                )
                
        except subprocess.TimeoutExpired:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Unit tests timed out after {self.timeout}s",
                details={'timeout': self.timeout}
            )
        except Exception as e:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Failed to run unit tests: {e}",
                details={'exception': str(e)}
            )


class SecurityQualityGate(QualityGate):
    """Validates security measures and configurations."""
    
    def __init__(self):
        super().__init__(
            "security_validation",
            "Validate security measures and configurations"
        )
        
    def _execute(self) -> QualityResult:
        """Test security components."""
        issues = []
        warnings_found = []
        
        try:
            from bci2token.security import SecurityConfig, SecureProcessor
            
            # Test security configuration
            config = SecurityConfig()
            if not config.enable_access_control:
                issues.append("Access control disabled")
            if not config.require_privacy_protection:
                warnings_found.append("Privacy protection not required")
            if not config.encrypt_saved_data:
                warnings_found.append("Data encryption disabled")
                
            # Test secure processor creation
            processor = SecureProcessor(config)
            
            # Test session creation
            session_token = processor.access_controller.create_session('test_user')
            if len(session_token) < 32:
                issues.append("Session tokens too short")
                
            # Test rate limiting
            for _ in range(5):
                if not processor.rate_limiter.check_rate_limit('test_user'):
                    break
            else:
                warnings_found.append("Rate limiting may be too permissive")
                
        except Exception as e:
            issues.append(f"Security system error: {e}")
            
        if issues:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Security validation failed: {len(issues)} critical issues",
                details={'issues': issues, 'warnings': warnings_found}
            )
        elif warnings_found:
            return QualityResult(
                name=self.name,
                level=QualityLevel.WARNING,
                message=f"Security warnings found: {len(warnings_found)} issues",
                details={'warnings': warnings_found}
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message="Security validation passed",
                details={'checks_performed': 'access_control,privacy,encryption,sessions,rate_limiting'}
            )


class PerformanceQualityGate(QualityGate):
    """Validates performance benchmarks and optimization."""
    
    def __init__(self):
        super().__init__(
            "performance_benchmarks",
            "Run performance benchmarks and validate optimization",
            timeout=60.0
        )
        
    def _execute(self) -> QualityResult:
        """Run performance benchmarks."""
        benchmarks = {}
        issues = []
        
        try:
            from bci2token.optimization import PerformanceOptimizer, OptimizationConfig
            
            # Test cache performance
            start_time = time.time()
            config = OptimizationConfig(cache_size=100)
            optimizer = PerformanceOptimizer(config)
            
            def mock_decode(signal):
                time.sleep(0.01)  # Simulate processing
                return {'tokens': [1, 2, 3], 'confidence': 0.8}
                
            optimized_decode = optimizer.optimize_decode_operation(mock_decode)
            
            # Test caching benefit
            if HAS_NUMPY:
                test_signal = np.random.randn(8, 256)
                
                # First call (uncached)
                cache_start = time.time()
                result1 = optimized_decode(test_signal)
                first_call_time = time.time() - cache_start
                
                # Second call (cached)
                cache_start = time.time()
                result2 = optimized_decode(test_signal)
                second_call_time = time.time() - cache_start
                
                if first_call_time > 0 and second_call_time > 0:
                    speedup = first_call_time / second_call_time
                    benchmarks['cache_speedup'] = speedup
                    
                    if speedup < 2.0:
                        issues.append(f"Cache speedup too low: {speedup:.1f}x")
                        
            # Test optimization config
            setup_time = time.time() - start_time
            benchmarks['setup_time'] = setup_time
            
            if setup_time > 1.0:
                issues.append(f"Optimization setup too slow: {setup_time:.2f}s")
                
            # Test performance report generation
            report_start = time.time()
            report = optimizer.get_performance_report()
            report_time = time.time() - report_start
            benchmarks['report_generation_time'] = report_time
            
            if report_time > 0.5:
                issues.append(f"Performance reporting too slow: {report_time:.2f}s")
                
            # Cleanup
            optimizer.cleanup()
            
        except Exception as e:
            issues.append(f"Performance testing error: {e}")
            
        if issues:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Performance benchmarks failed: {len(issues)} issues",
                details={'issues': issues, 'benchmarks': benchmarks}
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message="Performance benchmarks passed",
                details={'benchmarks': benchmarks}
            )


class ReliabilityQualityGate(QualityGate):
    """Validates reliability and recovery mechanisms."""
    
    def __init__(self):
        super().__init__(
            "reliability_validation",
            "Validate reliability and recovery mechanisms"
        )
        
    def _execute(self) -> QualityResult:
        """Test reliability components."""
        issues = []
        checks_passed = 0
        
        try:
            # Test circuit breaker
            from bci2token.reliability import CircuitBreaker
            
            circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
            
            # Test circuit breaker functionality
            def failing_function():
                raise RuntimeError("Test failure")
                
            # Should fail and eventually open circuit
            failures = 0
            for _ in range(5):
                try:
                    circuit_breaker.call(failing_function)
                except:
                    failures += 1
                    
            if failures < 3:
                issues.append("Circuit breaker not triggering properly")
            else:
                checks_passed += 1
                
            # Test recovery system
            from bci2token.recovery import SelfHealingSystem
            
            healing = SelfHealingSystem()
            
            class MockComponent:
                def get_status(self):
                    return {'status': 'healthy'}
                    
            healing.register_component('test_component', MockComponent())
            health_report = healing.get_system_health()
            
            if health_report['overall_status'] != 'healthy':
                issues.append("Self-healing system not reporting correctly")
            else:
                checks_passed += 1
                
            # Test input sanitization
            from bci2token.reliability import InputSanitizer
            
            if HAS_NUMPY:
                # Test with malformed input
                bad_signal = np.array([[np.inf, np.nan, 1000.0]])
                try:
                    sanitized = InputSanitizer.sanitize_brain_signal(bad_signal)
                    if np.any(np.isnan(sanitized)) or np.any(np.isinf(sanitized)):
                        issues.append("Input sanitization not working properly")
                    else:
                        checks_passed += 1
                except Exception as e:
                    issues.append(f"Input sanitization failed: {e}")
            else:
                checks_passed += 1  # Skip numpy-dependent test
                
        except Exception as e:
            issues.append(f"Reliability testing error: {e}")
            
        if issues:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Reliability validation failed: {len(issues)} issues",
                details={'issues': issues, 'checks_passed': checks_passed}
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message=f"Reliability validation passed ({checks_passed} checks)",
                details={'checks_passed': checks_passed}
            )


class ConfigurationQualityGate(QualityGate):
    """Validates configuration management and defaults."""
    
    def __init__(self):
        super().__init__(
            "configuration_validation",
            "Validate configuration management and defaults"
        )
        
    def _execute(self) -> QualityResult:
        """Test configuration systems."""
        issues = []
        configs_tested = 0
        
        try:
            # Test preprocessing configuration
            from bci2token.preprocessing import PreprocessingConfig
            
            config = PreprocessingConfig()
            if config.sampling_rate <= 0:
                issues.append("Invalid default sampling rate")
            if config.lowpass_freq <= config.highpass_freq:
                issues.append("Invalid frequency band configuration")
            configs_tested += 1
            
            # Test device configuration
            from bci2token.devices import DeviceConfig
            
            dev_config = DeviceConfig()
            if dev_config.n_channels <= 0:
                issues.append("Invalid default channel count")
            configs_tested += 1
            
            # Test security configuration
            from bci2token.security import SecurityConfig
            
            sec_config = SecurityConfig()
            if sec_config.session_timeout <= 0:
                issues.append("Invalid session timeout")
            if sec_config.max_concurrent_sessions <= 0:
                issues.append("Invalid concurrent session limit")
            configs_tested += 1
            
            # Test optimization configuration
            from bci2token.optimization import OptimizationConfig
            
            opt_config = OptimizationConfig()
            if opt_config.cache_size <= 0:
                issues.append("Invalid cache size")
            if opt_config.max_worker_threads <= 0:
                issues.append("Invalid worker thread count")
            configs_tested += 1
            
        except Exception as e:
            issues.append(f"Configuration testing error: {e}")
            
        if issues:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Configuration validation failed: {len(issues)} issues",
                details={'issues': issues, 'configs_tested': configs_tested}
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message=f"Configuration validation passed ({configs_tested} configs)",
                details={'configs_tested': configs_tested}
            )


class IntegrationQualityGate(QualityGate):
    """Validates integration between components."""
    
    def __init__(self):
        super().__init__(
            "integration_validation",
            "Validate integration between system components",
            timeout=45.0
        )
        
    def _execute(self) -> QualityResult:
        """Test component integration."""
        issues = []
        integrations_tested = 0
        
        try:
            # Test monitoring integration
            from bci2token.monitoring import get_monitor
            
            monitor = get_monitor()
            monitor.logger.info('Integration', 'Testing integration')
            integrations_tested += 1
            
            # Test health system integration
            from bci2token.health import run_comprehensive_diagnostics
            
            health_results = run_comprehensive_diagnostics()
            if not isinstance(health_results, dict) or len(health_results) == 0:
                issues.append("Health diagnostics integration failed")
            else:
                integrations_tested += 1
                
            # Test utility integration
            from bci2token.utils import validate_sampling_rate, calculate_signal_quality
            
            try:
                validate_sampling_rate(256)
                if HAS_NUMPY:
                    test_signal = np.random.randn(8, 256)
                    quality = calculate_signal_quality(test_signal)
                    if not (0.0 <= quality <= 1.0):
                        issues.append("Signal quality calculation out of range")
                integrations_tested += 1
            except Exception as e:
                issues.append(f"Utility integration failed: {e}")
                
            # Test CLI integration
            from bci2token.cli import main as cli_main
            
            # Test that CLI can be imported and has expected structure
            import inspect
            cli_functions = [name for name, obj in inspect.getmembers(
                sys.modules['bci2token.cli'], inspect.isfunction
            )]
            
            if len(cli_functions) < 3:  # Should have main, cmd_* functions
                issues.append("CLI integration incomplete")
            else:
                integrations_tested += 1
                
        except Exception as e:
            issues.append(f"Integration testing error: {e}")
            
        if issues:
            return QualityResult(
                name=self.name,
                level=QualityLevel.FAIL,
                message=f"Integration validation failed: {len(issues)} issues",
                details={'issues': issues, 'integrations_tested': integrations_tested}
            )
        else:
            return QualityResult(
                name=self.name,
                level=QualityLevel.PASS,
                message=f"Integration validation passed ({integrations_tested} integrations)",
                details={'integrations_tested': integrations_tested}
            )


class QualityGateRunner:
    """Runs all quality gates and generates comprehensive reports."""
    
    def __init__(self):
        self.gates = [
            ImportQualityGate(),
            UnitTestQualityGate(),
            SecurityQualityGate(),
            PerformanceQualityGate(),
            ReliabilityQualityGate(),
            ConfigurationQualityGate(),
            IntegrationQualityGate()
        ]
        
    def run_all_gates(self, fail_fast: bool = False) -> Dict[str, QualityResult]:
        """
        Run all quality gates.
        
        Args:
            fail_fast: Stop on first critical failure
            
        Returns:
            Dictionary mapping gate names to results
        """
        results = {}
        
        for gate in self.gates:
            print(f"Running quality gate: {gate.name}")
            
            try:
                result = gate.run()
                results[gate.name] = result
                
                # Print immediate feedback
                status_symbol = "✓" if result.is_passing() else "✗"
                print(f"  {status_symbol} {result.message} ({result.duration:.2f}s)")
                
                if fail_fast and result.level == QualityLevel.CRITICAL:
                    print(f"  CRITICAL failure in {gate.name}, stopping execution")
                    break
                    
            except Exception as e:
                result = QualityResult(
                    name=gate.name,
                    level=QualityLevel.CRITICAL,
                    message=f"Quality gate crashed: {e}",
                    details={'exception': str(e)}
                )
                results[gate.name] = result
                
                print(f"  ✗ Quality gate crashed: {e}")
                
                if fail_fast:
                    break
                    
        return results
        
    def generate_report(self, results: Dict[str, QualityResult]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        total_gates = len(results)
        passed = sum(1 for r in results.values() if r.level == QualityLevel.PASS)
        warnings = sum(1 for r in results.values() if r.level == QualityLevel.WARNING)
        failed = sum(1 for r in results.values() if r.level == QualityLevel.FAIL)
        critical = sum(1 for r in results.values() if r.level == QualityLevel.CRITICAL)
        
        overall_status = "PASS"
        if critical > 0:
            overall_status = "CRITICAL"
        elif failed > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "WARNING"
            
        total_duration = sum(r.duration for r in results.values())
        
        return {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'summary': {
                'total_gates': total_gates,
                'passed': passed,
                'warnings': warnings,
                'failed': failed,
                'critical': critical,
                'total_duration': total_duration
            },
            'gate_results': {
                name: {
                    'level': result.level.value,
                    'message': result.message,
                    'duration': result.duration,
                    'details': result.details
                }
                for name, result in results.items()
            },
            'recommendations': self._generate_recommendations(results)
        }
        
    def _generate_recommendations(self, results: Dict[str, QualityResult]) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for name, result in results.items():
            if result.level == QualityLevel.CRITICAL:
                recommendations.append(f"URGENT: Fix critical issue in {name} - {result.message}")
            elif result.level == QualityLevel.FAIL:
                recommendations.append(f"HIGH: Address failure in {name} - {result.message}")
            elif result.level == QualityLevel.WARNING:
                recommendations.append(f"MEDIUM: Review warning in {name} - {result.message}")
                
        # Add general recommendations
        failed_count = sum(1 for r in results.values() if r.level in [QualityLevel.FAIL, QualityLevel.CRITICAL])
        
        if failed_count == 0:
            recommendations.append("System is ready for production deployment")
        elif failed_count <= 2:
            recommendations.append("Address remaining issues before production deployment")
        else:
            recommendations.append("Significant quality issues detected - not ready for production")
            
        return recommendations


def run_quality_gates(fail_fast: bool = False, 
                     generate_report_file: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all quality gates and return success status with report.
    
    Args:
        fail_fast: Stop on first critical failure
        generate_report_file: Save report to file
        
    Returns:
        Tuple of (success, report_dict)
    """
    runner = QualityGateRunner()
    
    print("BCI-2-Token Quality Gate Validation")
    print("=" * 50)
    print()
    
    # Run all gates
    start_time = time.time()
    results = runner.run_all_gates(fail_fast=fail_fast)
    total_time = time.time() - start_time
    
    # Generate report
    report = runner.generate_report(results)
    
    print()
    print("=" * 50)
    print(f"Quality Gate Summary ({total_time:.1f}s total)")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Results: {report['summary']['passed']} passed, "
          f"{report['summary']['warnings']} warnings, "
          f"{report['summary']['failed']} failed, "
          f"{report['summary']['critical']} critical")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations'][:5]:  # Show top 5
            print(f"  • {rec}")
            
    # Save report to file if requested
    if generate_report_file:
        try:
            import json
            report_path = Path('/root/repo/quality_gate_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"\nWarning: Could not save report file: {e}")
    
    # Return success status
    success = report['overall_status'] in ['PASS', 'WARNING']
    return success, report


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    fail_fast = '--fail-fast' in sys.argv
    no_report = '--no-report' in sys.argv
    
    success, report = run_quality_gates(
        fail_fast=fail_fast,
        generate_report_file=not no_report
    )
    
    sys.exit(0 if success else 1)