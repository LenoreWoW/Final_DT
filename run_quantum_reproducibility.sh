#!/bin/bash
# Script to run reproducibility validator with quantum features enabled

set -e  # Exit on error

echo "========================================"
echo "Quantum Reproducibility Setup and Runner"
echo "========================================"
echo

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Not currently in a virtual environment."
    
    # Check if quantum_env exists
    if [ -d "quantum_env" ]; then
        echo "Activating existing quantum_env virtual environment..."
        source quantum_env/bin/activate
    else
        echo "Creating new quantum_env virtual environment..."
        python3 -m venv quantum_env
        source quantum_env/bin/activate
        
        echo "Installing required packages..."
        pip install qiskit pennylane scikit-learn
    fi
else
    echo "Already in virtual environment: $VIRTUAL_ENV"
fi

# Try to import required packages
echo "Checking for required packages..."
python3 -c "import qiskit; import pennylane; import sklearn; print('All required packages are installed.')" || {
    echo "Installing missing packages..."
    pip install qiskit pennylane scikit-learn
}

# Update config to enable quantum features
echo "Enabling quantum features in configuration..."
python3 quantum_enable.py

# Patch the reproducibility validator
echo "Patching the reproducibility validator..."
python3 quantum_validator_patch.py

# Set environment variable for development environment
export FLASK_ENV=development

# Run the reproducibility validator
echo
echo "========================================"
echo "Running Reproducibility Validator"
echo "========================================"
python3 scripts/reproducibility_validator.py

echo
echo "Results are saved in the results/reproducibility directory."
echo "You can view the summary at results/reproducibility/summary.txt" 