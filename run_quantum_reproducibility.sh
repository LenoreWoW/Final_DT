#!/bin/bash
# Script to run reproducibility validator with quantum features enabled

set -e  # Exit on error

echo "========================="
echo "Quantum Reproducibility Test"
echo "========================="

# Check if virtual environment exists
if [ -d "quantum_env" ]; then
    echo "Using existing quantum_env..."
    source quantum_env/bin/activate
else
    echo "Creating new virtual environment..."
    python -m venv quantum_env
    source quantum_env/bin/activate
    
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install qiskit qiskit_aer pennylane scikit-learn numpy matplotlib
fi

# Make the Python script executable
chmod +x run_quantum_reproduction.py

# Run the validator
echo "Running the Quantum Reproducibility Validator..."
python run_quantum_reproduction.py

echo ""
echo "If the test was successful, results are saved in results/reproducibility/"
echo "To view the summary: cat results/reproducibility/summary.txt" 