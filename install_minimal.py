#!/usr/bin/env python3
"""
Minimal installation helper for BCI-2-Token in constrained environments.
"""

import sys
import subprocess
import importlib.util

def check_module(module_name):
    """Check if a module is available."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def install_system_packages():
    """Install available system packages."""
    packages = [
        'python3-numpy',
        'python3-scipy', 
    ]
    
    print("Installing system packages...")
    for package in packages:
        try:
            result = subprocess.run(['apt', 'list', '--installed', package], 
                                  capture_output=True, text=True)
            if package in result.stdout:
                print(f"   ✓ {package} already installed")
            else:
                print(f"   Installing {package}...")
                subprocess.run(['apt', 'install', '-y', package], check=True)
        except subprocess.CalledProcessError:
            print(f"   ✗ Failed to install {package}")

def create_mock_modules():
    """Create mock modules for missing dependencies."""
    import sys
    from pathlib import Path
    
    # Import our mock torch
    mock_torch_path = Path(__file__).parent / 'mock_torch.py'
    if mock_torch_path.exists():
        spec = importlib.util.spec_from_file_location("mock_torch", mock_torch_path)
        mock_torch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mock_torch)
        
        # Register mock modules
        sys.modules['torch'] = mock_torch.torch
        sys.modules['torch.nn'] = mock_torch.torch.nn
        sys.modules['torch.nn.functional'] = mock_torch.F
        
        print("   ✓ Mock PyTorch modules created")

def run_tests():
    """Run basic tests to verify installation."""
    print("\nRunning basic tests...")
    
    # Test imports
    modules_to_test = [
        'numpy',
        'scipy', 
        'bci2token.preprocessing',
        'bci2token.devices'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ✗ {module}: {e}")
            
    # Test basic functionality
    try:
        import numpy as np
        from bci2token.preprocessing import PreprocessingConfig
        from bci2token.devices import DeviceConfig
        
        # Create configs
        prep_config = PreprocessingConfig(sampling_rate=256)
        dev_config = DeviceConfig(device_type='simulated')
        
        print("   ✓ Configuration objects created")
        
    except Exception as e:
        print(f"   ✗ Basic functionality test failed: {e}")

def main():
    print("BCI-2-Token Minimal Installation")
    print("=" * 40)
    
    # Check current environment
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Install system packages
    install_system_packages()
    
    # Create mock modules
    create_mock_modules()
    
    # Run tests
    run_tests()
    
    print("\n✓ Minimal installation completed")
    print("\nNote: This is a minimal installation for testing.")
    print("For full functionality, install PyTorch and other dependencies.")

if __name__ == '__main__':
    main()