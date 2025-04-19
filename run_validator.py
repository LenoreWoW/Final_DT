#!/usr/bin/env python3
"""
Run the reproducibility validator script to test quantum and classical methods.
"""

import os
import sys
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override environment variables if necessary
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

print("Starting reproducibility validation...")
start_time = time.time()

try:
    # Import and run the validator
    from scripts.reproducibility_validator import main, test_monte_carlo_reproducibility, test_qml_reproducibility
    
    # Option 1: Run individual tests
    print("\nRunning Monte Carlo reproducibility test...")
    mc_results = test_monte_carlo_reproducibility(n_runs=4)  # Use fewer runs for quicker testing
    
    print("\nRunning QML reproducibility test...")
    qml_results = test_qml_reproducibility(n_runs=3)  # Use fewer runs for quicker testing
    
    # Option 2: Run full validation
    print("\nRunning full validation...")
    main()
    
    print(f"\nValidation completed in {time.time() - start_time:.2f} seconds")
    print("Check results in the 'results/reproducibility' directory")
    
except Exception as e:
    print(f"\nError running validator: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 