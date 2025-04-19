#!/bin/bash

# Activate virtual environment if it exists, create it if not
if [ -d "quantum_env" ]; then
    echo "Activating existing quantum_env..."
    source quantum_env/bin/activate
else
    echo "Creating new virtual environment..."
    python -m venv quantum_env
    source quantum_env/bin/activate
fi

echo "Installing/upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install qiskit qiskit_aer pennylane scikit-learn numpy matplotlib

echo "Installation complete. Use 'source quantum_env/bin/activate' to activate the environment." 