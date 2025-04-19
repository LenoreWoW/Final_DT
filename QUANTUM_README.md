# Quantum Modules for Digital Twin

This component of the Digital Twin project provides quantum computing enhancements for simulations and machine learning. The quantum modules offer improved performance and more accurate results for suitable problems.

## Overview

The quantum module consists of two main components:

1. **Quantum Monte Carlo (QMC)**: Enhances Monte Carlo simulations with quantum algorithms.
2. **Quantum Machine Learning (QML)**: Implements quantum-enhanced machine learning models.

Both components provide classical fallbacks when quantum computing resources are unavailable.

## Requirements

To use the quantum features, you need either:

- **PennyLane**: For quantum machine learning and circuit simulation
- **Qiskit**: For quantum Monte Carlo and hardware access

Install these dependencies with:

```bash
pip install pennylane qiskit
```

For full functionality, including IBM Quantum hardware access:

```bash
pip install pennylane pennylane-qiskit qiskit qiskit-ibmq-provider
```

## Configuration

Configure quantum computing settings in your configuration file:

```json
{
  "quantum": {
    "enabled": true,
    "backend": "simulator",
    "shots": 1024,
    "max_circuit_depth": 20,
    "optimization_level": 1,
    "error_mitigation": true,
    "use_amplitude_amplification": false,
    "ml": {
      "learning_rate": 0.01,
      "n_layers": 2,
      "max_iterations": 100,
      "batch_size": 10,
      "early_stopping": true,
      "feature_map": "zz",
      "ansatz_type": "strongly_entangling"
    }
  }
}
```

To use IBM Quantum hardware, set `backend` to `ibmq_auto` for automatic selection of the least busy device, or specify a specific backend name like `ibmq_manila`. You'll also need to provide your IBM Quantum API token:

```bash
export IBMQ_TOKEN="your_ibm_quantum_token"
```

## Usage Examples

### Initializing Quantum Components

```python
from dt_project.quantum import initialize_quantum_components

# Initialize all quantum components
quantum_components = initialize_quantum_components()

# Access components
qmc = quantum_components["monte_carlo"]
qml = quantum_components["machine_learning"]

# Check availability
is_qmc_available = quantum_components["qmc_available"]
is_qml_available = quantum_components["qml_available"]
```

### Quantum Monte Carlo

```python
from dt_project.quantum import QuantumMonteCarlo

# Initialize QMC
qmc = QuantumMonteCarlo()

# Define a target function
def target_function(x, y):
    return x**2 + y**2 + 0.5 * x * y

# Define parameter ranges
param_ranges = {
    'x': (-2.0, 2.0),
    'y': (-2.0, 2.0)
}

# Run quantum Monte Carlo simulation
result = qmc.run_quantum_monte_carlo(
    param_ranges, 
    iterations=1000, 
    target_function=lambda x, y: target_function(x, y),
    distribution_type="normal"  # Options: uniform, normal, exponential, beta
)

# Access results
mean = result['mean']
std_dev = result['std']
samples = result['param_samples']

# Compare with classical Monte Carlo
comparison = qmc.compare_with_classical(
    param_ranges,
    lambda x, y: target_function(x, y),
    iterations=1000
)

print(f"Quantum mean: {comparison['quantum']['mean']}")
print(f"Classical mean: {comparison['classical']['mean']}")
print(f"Speedup factor: {comparison['speedup']}x")
```

### Quantum Machine Learning

```python
from dt_project.quantum import QuantumML
import numpy as np

# Initialize QML
qml = QuantumML()

# Prepare data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * np.sin(X[:, 0] * 10)

# Compare different encoding strategies
encoding_comparison = qml.compare_encoding_strategies(X, y)
best_encoding = encoding_comparison["best_encoding"]

# Configure QML
qml.feature_map = best_encoding
qml.n_layers = 2
qml.ansatz_type = "strongly_entangling"  # Options: strongly_entangling, basic, hardware_efficient, real_amplitudes

# Train quantum model
result = qml.train_model(X, y, test_size=0.2, verbose=True)

# Make predictions
X_new = np.random.rand(10, 3)
predictions = qml.predict(X_new)

# Get training history
history = qml.get_training_history()
```

## Advanced Features

### Probability Distribution Analysis

Analyze data distributions to determine if they are suitable for quantum advantage:

```python
from dt_project.quantum import QuantumMonteCarlo
import numpy as np

qmc = QuantumMonteCarlo()

# Sample data to analyze
data = {
    'x_values': np.random.normal(0, 1, 1000),
    'y_values': np.random.uniform(-1, 1, 1000)
}

# Analyze distributions
analysis = qmc.analyze_distributions(data)

# Results tell you which distributions could benefit from quantum computing
for name, result in analysis.items():
    print(f"{name}: best fit is {result['best_fit']} distribution")
    print(f"Potential quantum advantage: {result['potential_quantum_advantage']}")
```

### Error Mitigation

Enable error mitigation to improve results on noisy quantum hardware:

```python
# Enable error mitigation in config
config.update("quantum.error_mitigation", True)

# Or enable for a specific run
result = qmc.run_quantum_monte_carlo(
    param_ranges, 
    target_function=target_function,
    error_mitigation=True
)
```

### Variational Quantum Circuits

Experiment with different ansatz types for quantum machine learning:

```python
qml = QuantumML()

# Try different ansatz types
for ansatz in ["strongly_entangling", "hardware_efficient", "real_amplitudes", "basic"]:
    qml.ansatz_type = ansatz
    result = qml.train_model(X, y)
    print(f"Ansatz: {ansatz}, Test Loss: {result['final_test_loss']}")
```

## Demo Script

A demonstration script is provided to showcase the quantum features:

```bash
python demo_quantum_features.py
```

This script demonstrates:
- Quantum Monte Carlo with different distribution types
- Comparison with classical Monte Carlo
- Quantum ML with different encoding strategies
- Comparison with classical ML models

## Limitations and Fallbacks

- When quantum backends are unavailable, the system falls back to classical implementations
- Real quantum hardware may have limitations on circuit depth and qubits
- Error mitigation is essential for real hardware but may increase runtime

## Performance Considerations

- Quantum simulations can be slower than classical for simple problems
- The advantage becomes more apparent for complex probability distributions
- Quantum ML provides better results for specific types of problems

## For More Information

- [PennyLane Documentation](https://pennylane.ai/qml/)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [IBM Quantum](https://quantum-computing.ibm.com/) 