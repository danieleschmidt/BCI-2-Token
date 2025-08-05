#!/usr/bin/env python3
"""
Quality check runner for BCI-2-Token framework.

Orchestrates comprehensive testing, quality gates, and validation.
"""

import sys
import time
import subprocess
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_basic_tests():
    """Run basic test suite."""
    print("Running basic tests...")
    
    result = subprocess.run([
        sys.executable, 'test_basic.py'
    ], cwd=Path(__file__).parent)
    
    return result.returncode == 0


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("Running comprehensive tests...")
    
    result = subprocess.run([
        sys.executable, 'tests/test_comprehensive.py'
    ], cwd=Path(__file__).parent)
    
    return result.returncode == 0


def run_quality_gates(fail_fast=False):
    """Run quality gates."""
    print("Running quality gates...")
    
    try:
        from bci2token.quality_gates import run_quality_gates
        success, report = run_quality_gates(fail_fast=fail_fast)
        return success
    except Exception as e:
        print(f"Quality gates failed to run: {e}")
        return False


def check_code_style():
    """Check code style and formatting."""
    print("Checking code style...")
    
    # Simple code style checks
    issues = []
    
    # Check for Python files with obvious issues
    for py_file in Path('.').rglob('*.py'):
        if py_file.name.startswith('.'):
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Check for basic style issues
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Check line length (relaxed for this framework)
                if len(line) > 120:
                    issues.append(f"{py_file}:{i} Line too long ({len(line)} chars)")
                    
                # Check for trailing whitespace
                if line.endswith(' ') or line.endswith('\t'):
                    issues.append(f"{py_file}:{i} Trailing whitespace")
                    
        except Exception as e:
            issues.append(f"{py_file}: Could not check file - {e}")
            
    if issues:
        print(f"Code style issues found: {len(issues)}")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print("Code style checks passed")
        return True


def check_dependencies():
    """Check dependency status."""
    print("Checking dependencies...")
    
    core_deps = ['numpy']
    optional_deps = ['torch', 'transformers', 'cryptography']
    
    missing_core = []
    missing_optional = []
    
    for dep in core_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_core.append(dep)
            
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
            
    if missing_core:
        print(f"Missing core dependencies: {missing_core}")
        return False
    else:
        print("Core dependencies satisfied")
        if missing_optional:
            print(f"Missing optional dependencies: {missing_optional}")
        return True


def generate_quality_report():
    """Generate comprehensive quality report."""
    print("Generating quality report...")
    
    report = {
        'timestamp': time.time(),
        'checks': {}
    }
    
    # Run all checks
    checks = [
        ('dependencies', check_dependencies),
        ('code_style', check_code_style),
        ('basic_tests', run_basic_tests),
        ('comprehensive_tests', run_comprehensive_tests),
        ('quality_gates', lambda: run_quality_gates(fail_fast=False))
    ]
    
    overall_success = True
    
    for check_name, check_func in checks:
        print(f"\n{'-' * 20}")
        start_time = time.time()
        
        try:
            success = check_func()
            duration = time.time() - start_time
            
            report['checks'][check_name] = {
                'success': success,
                'duration': duration,
                'timestamp': start_time
            }
            
            status = "✓" if success else "✗"
            print(f"{status} {check_name} ({'PASS' if success else 'FAIL'}) ({duration:.1f}s)")
            
            if not success:
                overall_success = False
                
        except Exception as e:
            duration = time.time() - start_time
            report['checks'][check_name] = {
                'success': False,
                'duration': duration,
                'error': str(e),
                'timestamp': start_time
            }
            
            print(f"✗ {check_name} FAILED with exception: {e} ({duration:.1f}s)")
            overall_success = False
    
    # Summary
    print(f"\n{'=' * 50}")
    print("QUALITY CHECK SUMMARY")
    print(f"{'=' * 50}")
    
    passed = sum(1 for c in report['checks'].values() if c['success'])
    total = len(report['checks'])
    
    print(f"Overall Status: {'PASS' if overall_success else 'FAIL'}")
    print(f"Checks Passed: {passed}/{total}")
    
    total_duration = sum(c['duration'] for c in report['checks'].values())
    print(f"Total Time: {total_duration:.1f}s")
    
    # Save report
    try:
        import json
        report_path = Path('quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {report_path}")
    except Exception as e:
        print(f"\nWarning: Could not save report: {e}")
    
    return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BCI-2-Token Quality Check Runner')
    parser.add_argument('--basic-only', action='store_true', 
                       help='Run only basic tests')
    parser.add_argument('--comprehensive-only', action='store_true',
                       help='Run only comprehensive tests')
    parser.add_argument('--quality-gates-only', action='store_true',
                       help='Run only quality gates')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating detailed report')
    
    args = parser.parse_args()
    
    print("BCI-2-Token Quality Check Runner")
    print("=" * 50)
    
    success = True
    
    if args.basic_only:
        success = run_basic_tests()
    elif args.comprehensive_only:
        success = run_comprehensive_tests()
    elif args.quality_gates_only:
        success = run_quality_gates(fail_fast=args.fail_fast)
    elif args.no_report:
        # Run all checks without generating report
        success = (
            check_dependencies() and
            check_code_style() and
            run_basic_tests() and
            run_comprehensive_tests() and
            run_quality_gates(fail_fast=args.fail_fast)
        )
    else:
        # Full quality report
        success = generate_quality_report()
    
    if success:
        print("\n🎉 All quality checks passed!")
        print("\nSystem is ready for production deployment.")
    else:
        print("\n❌ Quality checks failed.")
        print("\nPlease address the issues before deployment.")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())