#!/usr/bin/env python3
"""
Quantum Features Demo
Demonstrates the quantum-enhanced functionality in the Digital Twin project.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
from sklearn.linear_model import LinearRegression
from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def demonstrate_monte_carlo():
    """Demonstrate quantum Monte Carlo simulation."""
    print_section("QUANTUM MONTE CARLO DEMONSTRATION")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        print("Quantum processing not available. This demo will run using classical simulation.")
    
    # Define a target function (a simple quadratic function)
    def target_function(x, y):
        return x**2 + y**2 + 0.5 * x * y
    
    # Define parameter ranges
    param_ranges = {
        'x': (-2.0, 2.0),
        'y': (-2.0, 2.0)
    }
    
    # Run Monte Carlo with different distribution types
    print("Comparing different distribution types...")
    
    distributions = ['uniform', 'normal', 'exponential', 'beta']
    results = {}
    
    for dist in distributions:
        print(f"Running with {dist} distribution...")
        try:
            start_time = time.time()
            result = qmc.run_quantum_monte_carlo(
                param_ranges, 
                iterations=1000, 
                target_function=lambda x, y: target_function(x, y),
                distribution_type=dist
            )
            duration = time.time() - start_time
            
            # Display stats
            print(f"  Mean: {result['mean']:.4f}")
            print(f"  Std dev: {result['std']:.4f}")
            print(f"  Min: {result['min']:.4f}")
            print(f"  Max: {result['max']:.4f}")
            print(f"  Time: {duration:.2f} seconds")
            
            results[dist] = result
        except Exception as e:
            print(f"Error with {dist} distribution: {str(e)}")
    
    # Compare with classical Monte Carlo
    print("\nComparing with classical Monte Carlo...")
    try:
        comparison = qmc.compare_with_classical(
            param_ranges,
            lambda x, y: target_function(x, y),
            iterations=1000
        )
        
        quantum_mean = comparison['quantum']['mean']
        classical_mean = comparison['classical']['mean']
        
        print(f"Quantum mean: {quantum_mean:.4f}")
        print(f"Classical mean: {classical_mean:.4f}")
        print(f"Difference: {abs(quantum_mean - classical_mean):.4f}")
        print(f"Speedup factor: {comparison['speedup']:.2f}x")
        
        # Plot comparison
        if results:
            # Create a scatter plot of samples
            plt.figure(figsize=(12, 8))
            
            for i, dist in enumerate(results.keys()):
                result = results[dist]
                x_samples = result['param_samples']['x'][:500]  # Limit to 500 points for clarity
                y_samples = result['param_samples']['y'][:500]
                
                plt.subplot(2, 2, i+1)
                plt.scatter(x_samples, y_samples, alpha=0.5, s=10)
                plt.title(f"{dist.title()} Distribution")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            plt.savefig("output/quantum_monte_carlo_comparison.png")
            print("Plot saved to output/quantum_monte_carlo_comparison.png")
            
    except Exception as e:
        print(f"Error in comparison: {str(e)}")

def demonstrate_quantum_ml():
    """Demonstrate quantum machine learning."""
    print_section("QUANTUM MACHINE LEARNING DEMONSTRATION")
    
    # Initialize quantum components
    config = ConfigManager()
    qml = QuantumML(config)
    
    if not qml.is_available():
        print("Quantum ML not available. This demo will show what would happen with limited functionality.")
    
    # Generate synthetic data
    print("Generating synthetic dataset...")
    np.random.seed(42)
    x = np.random.rand(100, 3)  # 100 samples, 3 features
    y = 0.5 * x[:, 0] + 0.3 * x[:, 1] + 0.2 * x[:, 2] + 0.1 * np.sin(x[:, 0] * 10) + 0.05 * np.random.rand(100)
    
    # Compare different encoding strategies
    print("\nComparing data encoding strategies...")
    try:
        encoding_comparison = qml.compare_encoding_strategies(x, y, test_size=0.2, iterations=15)
        
        if "best_encoding" in encoding_comparison:
            best_encoding = encoding_comparison["best_encoding"]
            best_mse = encoding_comparison["best_mse"]
            print(f"Best encoding strategy: {best_encoding} (MSE: {best_mse:.4f})")
            
            # Plot comparison
            details = encoding_comparison.get("details", {})
            if details:
                plt.figure(figsize=(12, 8))
                
                valid_encodings = [k for k, v in details.items() if "error" not in v]
                for i, encoding in enumerate(valid_encodings):
                    detail = details[encoding]
                    train_costs = detail.get("train_costs", [])
                    test_costs = detail.get("test_costs", [])
                    
                    plt.subplot(2, 2, i+1)
                    plt.plot(train_costs, label='Training Loss')
                    plt.plot(test_costs, label='Testing Loss')
                    plt.title(f"{encoding.title()} Encoding")
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    
                plt.tight_layout()
                plt.savefig("output/quantum_encoding_comparison.png")
                print("Encoding comparison plot saved to output/quantum_encoding_comparison.png")
        else:
            print("Could not determine best encoding strategy.")
            
    except Exception as e:
        print(f"Error comparing encoding strategies: {str(e)}")
    
    # Train and compare with classical model
    print("\nTraining quantum ML model and comparing with classical...")
    try:
        # Train classical model for comparison
        classical_model = LinearRegression()
        
        # Train quantum model
        qml.feature_map = qml.encoding_comparison.get("best_encoding", "zz")
        qml.n_layers = 2  # Keep it simple for the demo
        qml.max_iterations = 50
        
        training_result = qml.train_model(x, y, test_size=0.2, verbose=True)
        
        if training_result.get("success", False):
            print(f"Quantum model trained successfully in {training_result['training_time']:.2f} seconds")
            print(f"Final test loss: {training_result['final_test_loss']:.4f}")
            
            # Compare with classical model
            comparison = qml.compare_with_classical(x, y, classical_model, test_size=0.2)
            
            print(f"Quantum MSE: {comparison['quantum_mse']:.4f}")
            print(f"Classical MSE: {comparison['classical_mse']:.4f}")
            print(f"Relative improvement: {comparison['relative_improvement']*100:.1f}%")
            
            # Plot training history
            history = qml.get_training_history()
            plt.figure(figsize=(10, 6))
            train_loss = [entry['train_loss'] for entry in history]
            test_loss = [entry['test_loss'] for entry in history]
            plt.plot(train_loss, label='Training Loss')
            plt.plot(test_loss, label='Testing Loss')
            plt.title('Quantum ML Training Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig("output/quantum_ml_training.png")
            print("Training history plot saved to output/quantum_ml_training.png")
        else:
            print(f"Quantum model training failed: {training_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error training quantum ML model: {str(e)}")

def main():
    """Run the demonstration."""
    print_section("QUANTUM FEATURES DEMONSTRATION")
    
    # Initialize quantum components
    components = initialize_quantum_components()
    
    print(f"Quantum Monte Carlo available: {components['qmc_available']}")
    print(f"Quantum Machine Learning available: {components['qml_available']}")
    
    # Demonstrate Monte Carlo simulation
    demonstrate_monte_carlo()
    
    # Demonstrate Quantum ML
    demonstrate_quantum_ml()
    
    print_section("DEMONSTRATION COMPLETE")
    print("Check the output directory for generated visualizations.")

if __name__ == "__main__":
    main() 