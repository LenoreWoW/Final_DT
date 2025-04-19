#!/usr/bin/env python3
"""
Test only the Monte Carlo reproducibility function.
"""

import os
import sys
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override environment variables if necessary
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

print("Starting Monte Carlo reproducibility test...")
start_time = time.time()

try:
    # Import just the Monte Carlo test
    from scripts.reproducibility_validator import test_monte_carlo_reproducibility
    
    # Run with fewer iterations for quicker testing
    mc_results = test_monte_carlo_reproducibility(n_runs=3)
    
    print(f"\nTest completed in {time.time() - start_time:.2f} seconds")
    print("Check results in the 'results/reproducibility' directory")
    
except Exception as e:
    print(f"\nError running test: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 