#!/bin/bash

echo "========================="
echo "Quantum Reproducibility Test with Fixes"
echo "========================="

# Make scripts executable
chmod +x install_quantum_deps.sh

# Install dependencies
./install_quantum_deps.sh

# Activate virtual environment
source quantum_env/bin/activate

# Run test with our fixes
echo "Running test with quantum features..."
python run_simple_test.py

echo "Test complete!"
echo "Results saved in results/reproducibility/" 