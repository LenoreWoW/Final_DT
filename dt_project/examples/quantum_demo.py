#!/usr/bin/env python3
"""
Demo script to showcase quantum computing capabilities in the Digital Twin.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dt_project.config import ConfigManager
from dt_project.quantum import QuantumMonteCarlo, QuantumML

def demo_quantum_monte_carlo():
    """Demonstrate Quantum Monte Carlo simulation."""
    print("\n=== Quantum Monte Carlo Demonstration ===")
    
    # Initialize QMC with quantum enabled
    config = ConfigManager()
    quantum_config = config.get("quantum", {})
    quantum_config["enabled"] = True
    config.config["quantum"] = quantum_config
    
    qmc = QuantumMonteCarlo(config)
    
    # Check if quantum is available
    if not qmc.is_available():
        print("Quantum computing is not available. Using classical simulation instead.")
    else:
        print("Quantum computing is available.")
    
    # Define parameter ranges for the simulation
    param_ranges = {
        "temperature": (10.0, 30.0),
        "humidity": (30.0, 80.0),
        "wind_speed": (0.0, 20.0)
    }
    
    # Define a target function (athlete performance model)
    def athlete_performance(temperature, humidity, wind_speed):
        # Simple model: Performance drops with higher temperature, humidity, and wind speed
        base_performance = 100.0
        temp_factor = 1.0 - 0.01 * max(0, temperature - 20)  # Optimal at 20°C
        humidity_factor = 1.0 - 0.005 * max(0, humidity - 50)  # Optimal at 50% 
        wind_factor = 1.0 - 0.01 * wind_speed  # Lower is better
        
        performance = base_performance * temp_factor * humidity_factor * wind_factor
        return performance
    
    # Run comparison between quantum and classical Monte Carlo
    print("\nRunning comparison between quantum and classical Monte Carlo...")
    start_time = time.time()
    comparison = qmc.compare_with_classical(param_ranges, athlete_performance, iterations=500)
    print(f"Comparison completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    print("\nResults:")
    print(f"  Quantum Mean: {comparison['quantum']['mean']:.2f}")
    print(f"  Classical Mean: {comparison['classical']['mean']:.2f}")
    print(f"  Quantum StdDev: {comparison['quantum']['std']:.2f}")
    print(f"  Classical StdDev: {comparison['classical']['std']:.2f}")
    print(f"  Speedup: {comparison['speedup']:.2f}x")
    print(f"  StdDev Ratio: {comparison['std_ratio']:.2f}")
    
    # Plot histograms of parameter samples
    plot_parameter_histograms(comparison)
    
    return comparison

def demo_quantum_ml():
    """Demonstrate Quantum Machine Learning."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("scikit-learn is not available. Skipping quantum ML demo.")
        return
    
    print("\n=== Quantum Machine Learning Demonstration ===")
    
    # Initialize QML with quantum enabled
    config = ConfigManager()
    quantum_config = config.get("quantum", {})
    quantum_config["enabled"] = True
    config.config["quantum"] = quantum_config
    
    qml = QuantumML(config)
    
    # Check if quantum ML is available
    if not qml.is_available():
        print("Quantum ML is not available. Please install PennyLane and scikit-learn.")
        return
    
    print("Quantum ML is available.")
    
    # Generate synthetic dataset for athlete performance prediction
    np.random.seed(42)
    n_samples = 100
    
    # Features: temperature, humidity, wind_speed, athlete_strength, athlete_endurance
    X = np.random.rand(n_samples, 5)
    X[:, 0] = X[:, 0] * 20 + 10  # temperature: 10-30°C
    X[:, 1] = X[:, 1] * 50 + 30  # humidity: 30-80%
    X[:, 2] = X[:, 2] * 20       # wind_speed: 0-20 km/h
    X[:, 3] = X[:, 3] * 0.5 + 0.3  # strength: 0.3-0.8
    X[:, 4] = X[:, 4] * 0.5 + 0.3  # endurance: 0.3-0.8
    
    # Target: athlete performance
    y = np.zeros(n_samples)
    for i in range(n_samples):
        temp, humidity, wind, strength, endurance = X[i]
        # Base performance affected by environmental factors
        temp_factor = 1.0 - 0.01 * max(0, temp - 20)
        humidity_factor = 1.0 - 0.005 * max(0, humidity - 50)
        wind_factor = 1.0 - 0.01 * wind
        # Athlete factors provide base performance and resilience
        base = 60 + 40 * (strength * 0.5 + endurance * 0.5)
        resilience = 0.5 + 0.5 * endurance  # Higher endurance = more resilient
        
        # Environmental impact reduced by resilience
        env_impact = (temp_factor * humidity_factor * wind_factor - 1) * (1 - resilience) + 1
        
        performance = base * env_impact
        # Add some noise
        y[i] = performance + np.random.normal(0, 5)
    
    print(f"\nGenerated synthetic dataset with {n_samples} samples.")
    print("Features: temperature, humidity, wind_speed, athlete_strength, athlete_endurance")
    print("Target: athlete performance (0-100)")
    
    # Set up classical model for comparison
    classical_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Run quantum ML with reduced iterations for demo
    qml.max_iterations = 50  # Reduce iterations for demo
    print("\nComparing quantum and classical ML models...")
    start_time = time.time()
    comparison = qml.compare_with_classical(X, y, classical_model, test_size=0.3, verbose=True)
    print(f"Comparison completed in {time.time() - start_time:.2f} seconds")
    
    # Print results
    print("\nResults:")
    print(f"  Quantum MSE: {comparison['quantum_mse']:.2f}")
    print(f"  Classical MSE: {comparison['classical_mse']:.2f}")
    print(f"  Relative improvement: {comparison['relative_improvement']*100:.1f}%")
    print(f"  Training time - Quantum: {comparison['quantum_training_time']:.2f}s, Classical: {comparison['classical_training_time']:.2f}s")
    
    # Plot training history
    plot_qml_training_history(qml.get_training_history())
    
    return comparison

def plot_parameter_histograms(comparison: Dict[str, Any]):
    """Plot histograms of parameter samples from Monte Carlo."""
    quantum_samples = comparison['quantum']['param_samples']
    classical_samples = comparison['classical']['param_samples']
    
    fig, axes = plt.subplots(len(quantum_samples), 1, figsize=(10, 3*len(quantum_samples)))
    
    for i, param_name in enumerate(quantum_samples.keys()):
        ax = axes[i] if len(quantum_samples) > 1 else axes
        ax.hist(quantum_samples[param_name], bins=20, alpha=0.5, label="Quantum")
        ax.hist(classical_samples[param_name], bins=20, alpha=0.5, label="Classical")
        ax.set_title(f"Distribution of {param_name}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("quantum_vs_classical_mc.png")
    print("Histogram plot saved as 'quantum_vs_classical_mc.png'")

def plot_qml_training_history(history):
    """Plot training history of quantum ML model."""
    iterations = [entry["iteration"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    test_loss = [entry["test_loss"] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss, label="Training Loss")
    plt.plot(iterations, test_loss, label="Testing Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Quantum ML Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("quantum_ml_training.png")
    print("Training history plot saved as 'quantum_ml_training.png'")

def main():
    """Main function to run demonstrations."""
    print("=== Digital Twin Quantum Capabilities Demo ===")
    
    print("\nThis demo showcases the quantum computing capabilities of the Digital Twin,")
    print("including Quantum Monte Carlo simulation and Quantum Machine Learning.")
    
    # Run QMC demo
    qmc_results = demo_quantum_monte_carlo()
    
    # Run QML demo
    qml_results = demo_quantum_ml()
    
    print("\n=== Demo Complete ===")
    print("The quantum capabilities enhance the Digital Twin by providing:")
    print("1. More efficient Monte Carlo simulations for environmental prediction")
    print("2. Higher accuracy performance predictions with quantum machine learning")
    print("3. Ability to leverage quantum hardware when available")

if __name__ == "__main__":
    main() 