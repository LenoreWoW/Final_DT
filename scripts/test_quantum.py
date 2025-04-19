#!/usr/bin/env python3
"""
Basic test script for quantum components
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create necessary output directories
os.makedirs("results/test", exist_ok=True)

print("Initializing quantum components test...")

try:
    from dt_project.config import ConfigManager
    
    print("Successfully imported ConfigManager")
    
    # Create configuration with quantum enabled
    config = ConfigManager()
    config_dict = config.get_all()
    print(f"Configuration loaded: {config_dict.get('quantum', {}).get('enabled', False)}")
    
    # Override configuration by directly modifying the config dictionary
    # Since there's no update method, we'll modify the config directly
    if 'quantum' not in config_dict:
        config_dict['quantum'] = {}
    config_dict['quantum']['enabled'] = True
    
    # Use the config dictionary directly for the rest of the script
    print(f"Quantum enabled (manually set): {config_dict['quantum']['enabled']}")
    
    try:
        # Import numpy and print version
        print(f"NumPy version: {np.__version__}")
        
        # Try to import qiskit
        try:
            import qiskit
            print(f"Qiskit version: {qiskit.__version__}")
            print("Qiskit is available")
        except ImportError:
            print("Qiskit is not available")
        
        # Try to import pennylane
        try:
            import pennylane as qml
            print(f"PennyLane version: {qml.version()}")
            print("PennyLane is available")
        except ImportError:
            print("PennyLane is not available")
        
        # Initialize quantum components
        try:
            from dt_project.quantum import initialize_quantum_components
            
            print("Initializing quantum components...")
            # Create a new ConfigManager with our modifications
            modified_config = ConfigManager()
            # We'll need to modify the internal config dictionary
            modified_config.config = config_dict
            
            quantum_components = initialize_quantum_components(modified_config)
            
            print(f"Quantum components initialized:")
            print(f"- QMC available: {quantum_components['qmc_available']}")
            print(f"- QML available: {quantum_components['qml_available']}")
            
            # Run a simple test if components are available
            if quantum_components['qmc_available']:
                qmc = quantum_components['monte_carlo']
                
                print("\nRunning a simple Monte Carlo test...")
                try:
                    test_result = qmc.run_classical_monte_carlo(
                        param_ranges={'x': (-1.0, 1.0), 'y': (-1.0, 1.0)},
                        iterations=1000,
                        target_function=lambda x, y: x**2 + y**2
                    )
                    
                    print(f"Test result mean: {test_result['mean']:.6f}")
                    print(f"Test result std: {test_result['std']:.6f}")
                    print("Monte Carlo test completed successfully")
                except Exception as e:
                    print(f"Error running Monte Carlo test: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"Error initializing quantum components: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in quantum test: {str(e)}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Error importing ConfigManager: {str(e)}")
    import traceback
    traceback.print_exc()

print("Quantum test completed.") 