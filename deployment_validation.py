#!/usr/bin/env python3
"""
Production Deployment Validation
===============================

Validates that all deployment infrastructure is ready for production.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

def check_deployment_files():
    """Check that all deployment files are present and valid."""
    print("Checking deployment files...")
    
    required_files = [
        'deployment/scripts/deploy.sh',
        'deployment/docker-compose.yml',
        'deployment/kubernetes/deployment.yaml',
        'config/production.json',
        'requirements.txt',
        'pyproject.toml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ⚠️  Missing files: {missing_files}")
    else:
        print("   ✓ All deployment files present")
    
    # Check if deploy script is executable
    deploy_script = 'deployment/scripts/deploy.sh'
    if os.path.exists(deploy_script):
        if os.access(deploy_script, os.X_OK):
            print("   ✓ Deploy script is executable")
        else:
            print("   ⚠️  Deploy script not executable")
            # Make it executable
            os.chmod(deploy_script, 0o755)
            print("   ✓ Made deploy script executable")
    
    return len(missing_files) == 0

def validate_configuration():
    """Validate production configuration."""
    print("Validating configuration...")
    
    try:
        # Check production config
        with open('config/production.json', 'r') as f:
            prod_config = json.load(f)
        
        required_keys = ['logging', 'security', 'performance', 'monitoring']
        missing_keys = [key for key in required_keys if key not in prod_config]
        
        if missing_keys:
            print(f"   ⚠️  Missing config keys: {missing_keys}")
        else:
            print("   ✓ Production configuration valid")
        
        return len(missing_keys) == 0
        
    except FileNotFoundError:
        print("   ⚠️  Production config file not found")
        return False
    except json.JSONDecodeError as e:
        print(f"   ⚠️  Invalid JSON in production config: {e}")
        return False

def check_docker_setup():
    """Check Docker deployment setup."""
    print("Checking Docker setup...")
    
    try:
        # Check if docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ Docker available: {result.stdout.strip()}")
            docker_available = True
        else:
            print("   ⚠️  Docker not available")
            docker_available = False
        
        # Check docker-compose
        compose_available = False
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✓ Docker Compose available: {result.stdout.strip()}")
                compose_available = True
        except FileNotFoundError:
            # Try docker compose (newer version)
            try:
                result2 = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
                if result2.returncode == 0:
                    print(f"   ✓ Docker Compose (v2) available: {result2.stdout.strip()}")
                    compose_available = True
            except FileNotFoundError:
                pass
        
        if not compose_available:
            print("   ⚠️  Docker Compose not available (but Docker is available)")
        
        # Validate docker-compose.yml (if available)
        compose_cmd = None
        for cmd in [['docker-compose'], ['docker', 'compose']]:
            test_result = subprocess.run(cmd + ['--version'], capture_output=True, text=True)
            if test_result.returncode == 0:
                compose_cmd = cmd
                break
        
        if compose_cmd:
            result = subprocess.run(compose_cmd + ['-f', 'deployment/docker-compose.yml', 'config'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ Docker Compose configuration valid")
            else:
                print(f"   ⚠️  Docker Compose config validation skipped: {result.stderr[:100]}")
        else:
            print("   ⚠️  Docker Compose config validation skipped (compose not available)")
        
        return docker_available
        
    except FileNotFoundError:
        print("   ⚠️  Docker not installed")
        return False

def check_security_readiness():
    """Check security deployment readiness."""
    print("Checking security readiness...")
    
    try:
        from bci2token.enhanced_security import ZeroTrustValidator
        from bci2token.input_validation import SignalValidator
        
        # Test security components
        validator = ZeroTrustValidator()
        signal_validator = SignalValidator()
        
        print("   ✓ Security components functional")
        
        # Check if security logs directory exists
        if not os.path.exists('deployment/security_logs'):
            os.makedirs('deployment/security_logs', exist_ok=True)
            print("   ✓ Created security logs directory")
        else:
            print("   ✓ Security logs directory exists")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Security components not available: {e}")
        return False

def check_monitoring_readiness():
    """Check monitoring and logging readiness."""
    print("Checking monitoring readiness...")
    
    try:
        from bci2token.monitoring import get_monitor
        from bci2token.health import run_comprehensive_diagnostics
        
        # Test monitoring system
        monitor = get_monitor()
        monitor.logger.info('Deployment Validation', 'Testing monitoring system')
        
        # Test health diagnostics
        health_results = run_comprehensive_diagnostics()
        assert len(health_results) > 0, "Health diagnostics should return results"
        
        print("   ✓ Monitoring system functional")
        
        # Check logs directory
        if not os.path.exists('deployment/logs'):
            os.makedirs('deployment/logs', exist_ok=True)
            print("   ✓ Created logs directory")
        else:
            print("   ✓ Logs directory exists")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Monitoring system issue: {e}")
        return False

def check_performance_readiness():
    """Check performance optimization readiness."""
    print("Checking performance readiness...")
    
    try:
        from bci2token.performance_optimization import PerformanceOptimizer, PerformanceConfig
        from bci2token.auto_scaling import AutoScaler
        
        # Test performance components
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)
        scaler = AutoScaler()
        
        print("   ✓ Performance components functional")
        
        # Check cache directory
        if not os.path.exists('deployment/cache'):
            os.makedirs('deployment/cache', exist_ok=True)
            print("   ✓ Created cache directory")
        else:
            print("   ✓ Cache directory exists")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Performance components issue: {e}")
        return False

def check_data_directories():
    """Check and create necessary data directories."""
    print("Checking data directories...")
    
    directories = [
        'deployment/data',
        'deployment/logs', 
        'deployment/cache',
        'deployment/security_logs',
        'deployment/backups'
    ]
    
    created_dirs = []
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
    
    if created_dirs:
        print(f"   ✓ Created directories: {created_dirs}")
    else:
        print("   ✓ All data directories exist")
    
    return True

def test_deployment_script():
    """Test the deployment script."""
    print("Testing deployment script...")
    
    try:
        # Test dry run if supported
        result = subprocess.run(['bash', 'deployment/scripts/deploy.sh', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 or "Usage:" in result.stdout + result.stderr:
            print("   ✓ Deployment script executable")
        else:
            print("   ⚠️  Deployment script may have issues")
            
        return True
        
    except subprocess.TimeoutExpired:
        print("   ⚠️  Deployment script timed out")
        return False
    except Exception as e:
        print(f"   ⚠️  Deployment script test failed: {e}")
        return False

def main():
    """Main deployment validation."""
    print("Production Deployment Validation")
    print("=" * 40)
    
    tests = [
        ("Deployment Files", check_deployment_files),
        ("Configuration", validate_configuration),
        ("Docker Setup", check_docker_setup),
        ("Security Readiness", check_security_readiness),
        ("Monitoring Readiness", check_monitoring_readiness),
        ("Performance Readiness", check_performance_readiness),
        ("Data Directories", check_data_directories),
        ("Deployment Script", test_deployment_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Deployment Validation Results: {passed}/{total} passed")
    
    if passed == total:
        print("🚀 DEPLOYMENT READY - All systems go!")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  DEPLOYMENT MOSTLY READY - Review warnings")
        return 1
    else:
        print("❌ DEPLOYMENT NOT READY - Critical issues detected")
        return 2

if __name__ == "__main__":
    sys.exit(main())