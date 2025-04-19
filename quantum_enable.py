#!/usr/bin/env python3
"""
Script to enable quantum features in the digital twin.

This script modifies the configuration to ensure quantum components are enabled
and properly initialized with the required packages installed in the virtual environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Enabling quantum features...")

# Ensure packages are installed (will be skipped if already installed in the virtual environment)
try:
    import qiskit
    import pennylane
    import numpy
    import scipy
    import sklearn
    print("Required quantum packages are already installed")
except ImportError as e:
    print(f"Package not found: {e}")
    print("Please run this script in the virtual environment with the required packages installed:")
    print("  python3 -m venv quantum_env")
    print("  source quantum_env/bin/activate")
    print("  pip install qiskit pennylane scikit-learn")
    sys.exit(1)

# Check and update the configuration file
config_path = Path('config/config.json')
if not config_path.exists():
    print(f"Configuration file not found at {config_path}")
    sys.exit(1)

try:
    # Load the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Ensure quantum is enabled in all environments
    for env in ['default', 'development', 'testing', 'production']:
        if env in config:
            if 'quantum' not in config[env]:
                config[env]['quantum'] = {}
            config[env]['quantum']['enabled'] = True
            print(f"Enabled quantum in {env} environment")
    
    # Save the updated configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Configuration updated successfully. Quantum features are now enabled.")
    
except Exception as e:
    print(f"Error updating configuration: {e}")
    sys.exit(1)

# Try to import the ConfigManager and create a configured instance
try:
    from dt_project.config import ConfigManager
    
    # Force reload the config
    config = ConfigManager(config_path=str(config_path.resolve()))
    config_dict = config.get_all()
    
    quantum_enabled = config.get("quantum.enabled", False)
    print(f"Quantum enabled in current configuration: {quantum_enabled}")
    
    if not quantum_enabled:
        print("Warning: Quantum is still not enabled in the configuration.")
        print("You may need to set the FLASK_ENV environment variable to 'development' or manually modify the code.")
        print("export FLASK_ENV=development")
except Exception as e:
    print(f"Error checking configuration: {e}")

print("\nTo run the reproducibility validator with quantum features:")
print("1. Activate the virtual environment:")
print("   source quantum_env/bin/activate")
print("2. Run the validator:")
print("   python3 scripts/reproducibility_validator.py") 