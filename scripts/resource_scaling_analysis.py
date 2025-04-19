#!/usr/bin/env python3
"""
Resource Scaling Analysis

Analyzes how performance scales with problem complexity for both classical and quantum approaches.
This script measures execution time, memory usage, and accuracy as problem size increases.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/resource_scaling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def measure_memory():
    """Measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def analyze_monte_carlo_scaling(param_dimensions=[1, 2, 3, 5, 8], iterations=2000):
    """
    Analyze how resource usage scales with increasing problem dimensions for Monte Carlo.
    
    Args:
        param_dimensions: List of parameter dimensions to test
        iterations: Number of iterations for Monte Carlo
        
    Returns:
        Dictionary with scaling results
    """
    print_section("MONTE CARLO RESOURCE SCALING ANALYSIS")
    
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Running in simulation mode.")
    
    # Store results
    results = {
        "dimensions": param_dimensions,
        "quantum_time": [],
        "classical_time": [],
        "quantum_memory": [],
        "classical_memory": [],
        "quantum_std": [],
        "classical_std": []
    }
    
    # For each dimension, create parameter ranges and run comparison
    for dim in param_dimensions:
        print(f"Testing with {dim} dimensions...")
        
        # Create parameter ranges with increasing dimensions
        param_ranges = {}
        for i in range(dim):
            param_ranges[f'x{i}'] = (-1.0, 1.0)
        
        # Define n-dimensional target function - simple quadratic
        def target_function(*args):
            return sum(x**2 for x in args) + 0.1 * sum(args[i] * args[j] for i in range(len(args)) for j in range(i+1, len(args)))
        
        # Measure classical performance
        start_memory = measure_memory()
        start_time = time.time()
        
        classical_result = qmc.run_classical_monte_carlo(
            param_ranges, 
            iterations=iterations,
            target_function=target_function
        )
        
        classical_time = time.time() - start_time
        classical_memory = measure_memory() - start_memory
        
        # Measure quantum performance
        start_memory = measure_memory()
        start_time = time.time()
        
        quantum_result = qmc.run_quantum_monte_carlo(
            param_ranges, 
            iterations=iterations,
            target_function=target_function
        )
        
        quantum_time = time.time() - start_time
        quantum_memory = measure_memory() - start_memory
        
        # Store results
        results["quantum_time"].append(quantum_time)
        results["classical_time"].append(classical_time)
        results["quantum_memory"].append(quantum_memory)
        results["classical_memory"].append(classical_memory)
        results["quantum_std"].append(quantum_result["std"])
        results["classical_std"].append(classical_result["std"])
        
        print(f"  Dimension: {dim}")
        print(f"  Classical: Time={classical_time:.2f}s, Memory={classical_memory:.2f}MB")
        print(f"  Quantum:   Time={quantum_time:.2f}s, Memory={quantum_memory:.2f}MB")
        print(f"  Speedup:   {classical_time/quantum_time:.2f}x")
    
    # Generate plots
    plot_scaling_results(results, "monte_carlo")
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_scaling.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_ml_scaling(feature_dimensions=[2, 4, 8, 16, 32], dataset_size=200):
    """
    Analyze how resource usage scales with increasing feature dimensions for quantum ML.
    
    Args:
        feature_dimensions: List of feature dimensions to test
        dataset_size: Size of the synthetic dataset
        
    Returns:
        Dictionary with scaling results
    """
    print_section("MACHINE LEARNING RESOURCE SCALING ANALYSIS")
    
    config = ConfigManager()
    qml = QuantumML(config)
    
    if not qml.is_available():
        logger.warning("Quantum ML not available. Running in simulation mode.")
        
    # Basic QML configuration
    qml.feature_map = "zz"
    qml.n_layers = 2
    qml.ansatz_type = "strongly_entangling"
    qml.max_iterations = 30
    
    from sklearn.linear_model import Ridge
    classical_model = Ridge(alpha=0.1)
    
    # Store results
    results = {
        "dimensions": feature_dimensions,
        "quantum_time": [],
        "classical_time": [],
        "quantum_memory": [],
        "classical_memory": [],
        "quantum_mse": [],
        "classical_mse": []
    }
    
    # For each dimension, create dataset and run comparison
    for dim in feature_dimensions:
        print(f"Testing with {dim} features...")
        
        # Generate synthetic dataset with the current dimension
        np.random.seed(42)
        X = np.random.rand(dataset_size, dim)
        
        # Create target values with some nonlinearity
        weights = np.random.rand(dim) * 2 - 1  # Random weights between -1 and 1
        
        y = np.dot(X, weights)  # Linear component
        
        # Add nonlinear components for first few dimensions
        for i in range(min(3, dim)):
            for j in range(i+1, min(5, dim)):
                y += 0.2 * X[:, i] * X[:, j]  # Interaction terms
        
        # Add some noise
        y += 0.05 * np.random.randn(dataset_size)
        
        # Measure classical performance
        start_memory = measure_memory()
        start_time = time.time()
        
        classical_model.fit(X, y)
        classical_pred = classical_model.predict(X)
        classical_mse = np.mean((classical_pred - y)**2)
        
        classical_time = time.time() - start_time
        classical_memory = measure_memory() - start_memory
        
        # Measure quantum performance
        try:
            start_memory = measure_memory()
            start_time = time.time()
            
            quantum_result = qml.train_model(X, y, test_size=0.0, verbose=False)  # Use all data for training
            quantum_pred = qml.predict(X)
            quantum_mse = np.mean((quantum_pred - y)**2)
            
            quantum_time = time.time() - start_time
            quantum_memory = measure_memory() - start_memory
            
        except Exception as e:
            logger.error(f"Error in quantum ML for dim={dim}: {str(e)}")
            quantum_time = float('nan')
            quantum_memory = float('nan')
            quantum_mse = float('nan')
        
        # Store results
        results["quantum_time"].append(quantum_time)
        results["classical_time"].append(classical_time)
        results["quantum_memory"].append(quantum_memory)
        results["classical_memory"].append(classical_memory)
        results["quantum_mse"].append(quantum_mse)
        results["classical_mse"].append(classical_mse)
        
        print(f"  Features: {dim}")
        print(f"  Classical: Time={classical_time:.2f}s, Memory={classical_memory:.2f}MB, MSE={classical_mse:.4f}")
        
        if not np.isnan(quantum_time):
            print(f"  Quantum:   Time={quantum_time:.2f}s, Memory={quantum_memory:.2f}MB, MSE={quantum_mse:.4f}")
            print(f"  Speedup:   {classical_time/quantum_time:.2f}x")
        else:
            print("  Quantum:   Failed to complete")
    
    # Generate plots
    plot_scaling_results(results, "machine_learning")
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_scaling.json'), 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in results.items()}, f, indent=2)
    
    return results

def plot_scaling_results(results, analysis_type):
    """
    Generate plots for scaling analysis results.
    
    Args:
        results: Dictionary with scaling results
        analysis_type: Type of analysis ('monte_carlo' or 'machine_learning')
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    dimensions = results["dimensions"]
    
    # Plot execution time
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, results["quantum_time"], 'o-', label='Quantum')
    plt.plot(dimensions, results["classical_time"], 's-', label='Classical')
    plt.xlabel('Problem Dimension')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time Scaling ({analysis_type.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{analysis_type}_time_scaling.png'))
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, results["quantum_memory"], 'o-', label='Quantum')
    plt.plot(dimensions, results["classical_memory"], 's-', label='Classical')
    plt.xlabel('Problem Dimension')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage Scaling ({analysis_type.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{analysis_type}_memory_scaling.png'))
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    speedup = [c/q if q > 0 else float('nan') for c, q in zip(results["classical_time"], results["quantum_time"])]
    plt.plot(dimensions, speedup, 'o-')
    plt.xlabel('Problem Dimension')
    plt.ylabel('Speedup Factor (Classical/Quantum)')
    plt.title(f'Quantum Speedup ({analysis_type.replace("_", " ").title()})')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{analysis_type}_speedup.png'))
    
    # Plot accuracy metrics (different for MC and ML)
    plt.figure(figsize=(10, 6))
    
    if analysis_type == "monte_carlo":
        plt.plot(dimensions, results["quantum_std"], 'o-', label='Quantum')
        plt.plot(dimensions, results["classical_std"], 's-', label='Classical')
        plt.ylabel('Standard Deviation')
        plt.title('Monte Carlo Precision')
    else:  # machine_learning
        plt.plot(dimensions, results["quantum_mse"], 'o-', label='Quantum')
        plt.plot(dimensions, results["classical_mse"], 's-', label='Classical')
        plt.ylabel('Mean Squared Error')
        plt.title('Machine Learning Accuracy')
    
    plt.xlabel('Problem Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{analysis_type}_accuracy.png'))

def main():
    """Run resource scaling analysis."""
    print_section("RESOURCE SCALING ANALYSIS")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Run Monte Carlo scaling analysis
    mc_results = analyze_monte_carlo_scaling(
        param_dimensions=[1, 2, 3, 5, 8],
        iterations=2000
    )
    
    # Run Machine Learning scaling analysis
    ml_results = analyze_ml_scaling(
        feature_dimensions=[2, 4, 8, 16, 32],
        dataset_size=200
    )
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("RESOURCE SCALING ANALYSIS SUMMARY\n")
        f.write("================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Scaling Summary:\n")
        f.write("--------------------------\n")
        f.write("Time complexity analysis:\n")
        
        # Calculate time complexity exponent through linear regression
        log_dims = np.log(mc_results['dimensions'])
        log_q_time = np.log(mc_results['quantum_time'])
        log_c_time = np.log(mc_results['classical_time'])
        
        q_exponent = np.polyfit(log_dims, log_q_time, 1)[0]
        c_exponent = np.polyfit(log_dims, log_c_time, 1)[0]
        
        f.write(f"  Classical time complexity: O(n^{c_exponent:.2f})\n")
        f.write(f"  Quantum time complexity: O(n^{q_exponent:.2f})\n\n")
        
        # Machine Learning summary
        f.write("Machine Learning Scaling Summary:\n")
        f.write("-------------------------------\n")
        
        # Dimension threshold where quantum becomes advantageous
        advantage_dims = []
        for i in range(1, len(ml_results['dimensions'])):
            if i < len(ml_results['quantum_time']) and i < len(ml_results['classical_time']):
                if ml_results['quantum_time'][i-1] > ml_results['classical_time'][i-1] and \
                   ml_results['quantum_time'][i] <= ml_results['classical_time'][i]:
                    advantage_dims.append(ml_results['dimensions'][i])
        
        if advantage_dims:
            f.write(f"Quantum advantage appears at dimension: {advantage_dims[0]}\n\n")
        else:
            f.write("No clear quantum advantage threshold detected in the tested dimensions.\n\n")
        
        # Overall findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        f.write("1. Problem dimensionality has a significant impact on relative performance.\n")
        
        max_mc_speedup = max([c/q if q > 0 else 0 for c, q in zip(mc_results["classical_time"], mc_results["quantum_time"])])
        f.write(f"2. Maximum observed Monte Carlo speedup: {max_mc_speedup:.2f}x\n")
        
        valid_ml_speedups = [c/q for c, q in zip(ml_results["classical_time"], ml_results["quantum_time"]) if q > 0 and not np.isnan(q)]
        if valid_ml_speedups:
            max_ml_speedup = max(valid_ml_speedups)
            f.write(f"3. Maximum observed Machine Learning speedup: {max_ml_speedup:.2f}x\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nAnalysis completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 