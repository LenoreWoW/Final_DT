#!/usr/bin/env python3
"""
Compare Classical vs Quantum Simulation Approaches

This script provides a direct comparison between classical and quantum simulation
approaches across different problem types and complexity levels. It measures performance,
accuracy, and resource utilization to quantify potential quantum advantage.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/classical_vs_quantum"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def compare_monte_carlo(iterations_list=[500, 1000, 2000, 5000], repeats=5):
    """
    Compare classical and quantum Monte Carlo approaches.
    
    Args:
        iterations_list: List of simulation iteration counts to test
        repeats: Number of repeats for statistical reliability
    
    Returns:
        Dictionary of comparative results
    """
    print_section("MONTE CARLO SIMULATION COMPARISON")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Running in classical simulation mode.")
    
    # Target function with varying complexity
    def target_function(x, y, z=0):
        # Basic quadratic function
        return x**2 + y**2 + 0.5*x*y + z**2
    
    # Parameter ranges
    param_ranges = {
        'x': (-2.0, 2.0),
        'y': (-2.0, 2.0),
    }
    
    distributions = ['uniform', 'normal', 'exponential']
    
    # Store results
    results = {
        "iterations": iterations_list,
        "distributions": distributions,
        "quantum_mean": {dist: [] for dist in distributions},
        "classical_mean": {dist: [] for dist in distributions},
        "quantum_std": {dist: [] for dist in distributions}, 
        "classical_std": {dist: [] for dist in distributions},
        "quantum_time": {dist: [] for dist in distributions},
        "classical_time": {dist: [] for dist in distributions},
        "speedup": {dist: [] for dist in distributions},
        "mean_diff": {dist: [] for dist in distributions},
        "std_ratio": {dist: [] for dist in distributions}
    }
    
    # Run comparisons
    for dist in distributions:
        print(f"\nTesting {dist} distribution:")
        
        for iterations in iterations_list:
            print(f"  Running with {iterations} iterations...")
            quantum_means = []
            classical_means = []
            quantum_stds = []
            classical_stds = []
            quantum_times = []
            classical_times = []
            speedups = []
            
            for r in range(repeats):
                try:
                    # Compare quantum vs classical
                    comparison = qmc.compare_with_classical(
                        param_ranges,
                        lambda x, y: target_function(x, y),
                        iterations=iterations
                    )
                    
                    quantum_means.append(comparison['quantum']['mean'])
                    classical_means.append(comparison['classical']['mean'])
                    quantum_stds.append(comparison['quantum']['std'])
                    classical_stds.append(comparison['classical']['std'])
                    quantum_times.append(comparison['quantum']['execution_time'])
                    classical_times.append(comparison['classical']['execution_time'])
                    speedups.append(comparison['speedup'])
                    
                except Exception as e:
                    logger.error(f"Error in comparison (dist={dist}, iter={iterations}, repeat={r}): {str(e)}")
            
            # Average the results
            q_mean = np.mean(quantum_means) if quantum_means else 0
            c_mean = np.mean(classical_means) if classical_means else 0
            q_std = np.mean(quantum_stds) if quantum_stds else 0
            c_std = np.mean(classical_stds) if classical_stds else 0
            q_time = np.mean(quantum_times) if quantum_times else 0
            c_time = np.mean(classical_times) if classical_times else 0
            speedup = np.mean(speedups) if speedups else 1.0
            
            # Store in results
            results["quantum_mean"][dist].append(q_mean)
            results["classical_mean"][dist].append(c_mean)
            results["quantum_std"][dist].append(q_std)
            results["classical_std"][dist].append(c_std)
            results["quantum_time"][dist].append(q_time)
            results["classical_time"][dist].append(c_time)
            results["speedup"][dist].append(speedup)
            results["mean_diff"][dist].append(abs(q_mean - c_mean))
            results["std_ratio"][dist].append(q_std / c_std if c_std > 0 else 1.0)
            
            print(f"    Quantum Mean: {q_mean:.4f}, Classical Mean: {c_mean:.4f}")
            print(f"    Quantum Time: {q_time:.2f}s, Classical Time: {c_time:.2f}s")
            print(f"    Speedup: {speedup:.2f}x")
    
    # Generate plots
    plot_monte_carlo_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_monte_carlo_results(results):
    """
    Generate plots for Monte Carlo comparison results.
    
    Args:
        results: Dictionary of results from compare_monte_carlo
    """
    # Create plot directory
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    iterations = results["iterations"]
    distributions = results["distributions"]
    
    # 1. Execution Time Comparison
    plt.figure(figsize=(12, 8))
    for i, dist in enumerate(distributions):
        plt.subplot(2, 2, i+1)
        plt.plot(iterations, results["quantum_time"][dist], 'o-', label='Quantum')
        plt.plot(iterations, results["classical_time"][dist], 's-', label='Classical')
        plt.xscale('log')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Execution Time (s)')
        plt.title(f'{dist.capitalize()} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'execution_time_comparison.png'))
    
    # 2. Speedup Factor
    plt.figure(figsize=(10, 6))
    for dist in distributions:
        plt.plot(iterations, results["speedup"][dist], 'o-', label=f'{dist.capitalize()}')
    
    plt.xscale('log')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Speedup Factor (Classical/Quantum)')
    plt.title('Quantum vs Classical Speedup')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'speedup_factor.png'))
    
    # 3. Accuracy Comparison (Mean Difference)
    plt.figure(figsize=(10, 6))
    for dist in distributions:
        plt.plot(iterations, results["mean_diff"][dist], 'o-', label=f'{dist.capitalize()}')
    
    plt.xscale('log')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Absolute Difference in Mean')
    plt.title('Accuracy Comparison: |Quantum Mean - Classical Mean|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_comparison.png'))
    
    # 4. Combined Summary Plot
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    for dist in distributions:
        plt.plot(iterations, results["speedup"][dist], 'o-', label=f'{dist.capitalize()}')
    plt.xscale('log')
    plt.ylabel('Speedup Factor')
    plt.title('Performance Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for dist in distributions:
        plt.plot(iterations, results["mean_diff"][dist], 'o-', label=f'{dist.capitalize()}')
    plt.xscale('log')
    plt.ylabel('Mean Difference')
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    bar_width = 0.35
    index = np.arange(len(distributions))
    
    # Use last iteration for summary
    q_times = [results["quantum_time"][dist][-1] for dist in distributions]
    c_times = [results["classical_time"][dist][-1] for dist in distributions]
    
    plt.bar(index, c_times, bar_width, label='Classical')
    plt.bar(index + bar_width, q_times, bar_width, label='Quantum')
    plt.xlabel('Distribution')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time ({iterations[-1]} Iterations)')
    plt.xticks(index + bar_width/2, [d.capitalize() for d in distributions])
    plt.legend()
    
    plt.subplot(2, 2, 4)
    speedups = [results["speedup"][dist][-1] for dist in distributions]
    plt.bar(index, speedups, color='green')
    plt.xlabel('Distribution')
    plt.ylabel('Speedup Factor')
    plt.title(f'Speedup Factor ({iterations[-1]} Iterations)')
    plt.xticks(index, [d.capitalize() for d in distributions])
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monte_carlo_summary.png'))

def compare_machine_learning(dataset_sizes=[50, 100, 200], feature_dims=[2, 4, 8], repeats=3):
    """
    Compare classical and quantum machine learning approaches.
    
    Args:
        dataset_sizes: List of dataset sizes to test
        feature_dims: List of feature dimensions to test
        repeats: Number of repeats for statistical reliability
    
    Returns:
        Dictionary of comparative results
    """
    print_section("MACHINE LEARNING COMPARISON")
    
    # Initialize quantum components
    config = ConfigManager()
    qml = QuantumML(config)
    
    if not qml.is_available():
        logger.warning("Quantum ML not available. Running in classical simulation mode.")
    
    # Store results
    results = {
        "dataset_sizes": dataset_sizes,
        "feature_dims": feature_dims,
        "quantum_mse": {f"{dim}d": {str(size): [] for size in dataset_sizes} for dim in feature_dims},
        "classical_mse": {f"{dim}d": {str(size): [] for size in dataset_sizes} for dim in feature_dims},
        "quantum_time": {f"{dim}d": {str(size): [] for size in dataset_sizes} for dim in feature_dims},
        "classical_time": {f"{dim}d": {str(size): [] for size in dataset_sizes} for dim in feature_dims},
        "improvement": {f"{dim}d": {str(size): [] for size in dataset_sizes} for dim in feature_dims}
    }
    
    # Run comparisons across dataset sizes and feature dimensions
    for dim in feature_dims:
        print(f"\nTesting with {dim} features:")
        
        for size in dataset_sizes:
            print(f"  Dataset size: {size}")
            
            for r in range(repeats):
                print(f"    Repeat {r+1}/{repeats}...")
                try:
                    # Generate synthetic dataset with increasing complexity
                    np.random.seed(42 + r)  # Vary seed slightly for statistical variation
                    X = np.random.rand(size, dim)
                    
                    # Target function complexity increases with dimension
                    if dim <= 2:
                        y = 0.5 * X[:, 0] + 0.5 * X[:, 1] ** 2
                    elif dim <= 4:
                        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2
                    else:
                        # More complex relationships for higher dimensions
                        y = 0.2 * X[:, 0] + 0.1 * X[:, 1] * X[:, 2] + 0.3 * np.sin(5 * X[:, 3]) + 0.4 * X[:, 4] ** 2
                        if dim > 5:
                            y += 0.1 * np.cos(3 * X[:, 5]) * X[:, 6]
                            if dim > 7:
                                y += 0.05 * np.exp(X[:, 7]) * 0.5
                    
                    # Add noise
                    y += 0.05 * np.random.randn(size)
                    
                    # Configure quantum ML
                    qml.feature_map = "zz"  # Use consistent encoding for comparison
                    qml.n_layers = min(3, dim)  # Scale circuit depth with dimension
                    qml.ansatz_type = "strongly_entangling"
                    qml.max_iterations = 50  # Limit iterations for benchmark
                    
                    # Train quantum model and compare with classical
                    from sklearn.linear_model import Ridge
                    classical_model = Ridge(alpha=0.1)
                    
                    comparison = qml.compare_with_classical(X, y, classical_model, test_size=0.3)
                    
                    # Extract results
                    quantum_mse = comparison.get("quantum_mse", 0)
                    classical_mse = comparison.get("classical_mse", 0)
                    quantum_time = comparison.get("quantum_training_time", 0)
                    classical_time = comparison.get("classical_training_time", 0)
                    improvement = comparison.get("relative_improvement", 0)
                    
                    # Store in results
                    dim_key = f"{dim}d"
                    size_key = str(size)
                    results["quantum_mse"][dim_key][size_key].append(quantum_mse)
                    results["classical_mse"][dim_key][size_key].append(classical_mse)
                    results["quantum_time"][dim_key][size_key].append(quantum_time)
                    results["classical_time"][dim_key][size_key].append(classical_time)
                    results["improvement"][dim_key][size_key].append(improvement)
                    
                    print(f"      Quantum MSE: {quantum_mse:.4f}, Classical MSE: {classical_mse:.4f}")
                    print(f"      Improvement: {improvement*100:.1f}%")
                
                except Exception as e:
                    logger.error(f"Error in ML comparison (dim={dim}, size={size}, repeat={r}): {str(e)}")
                    
    # Average results across repeats
    for dim in feature_dims:
        dim_key = f"{dim}d"
        for size in dataset_sizes:
            size_key = str(size)
            for metric in ["quantum_mse", "classical_mse", "quantum_time", "classical_time", "improvement"]:
                values = results[metric][dim_key][size_key]
                if values:
                    results[metric][dim_key][size_key] = float(np.mean(values))
                else:
                    results[metric][dim_key][size_key] = None
    
    # Generate plots
    plot_ml_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_ml_results(results):
    """
    Generate plots for machine learning comparison results.
    
    Args:
        results: Dictionary of results from compare_machine_learning
    """
    # Create plot directory
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    dataset_sizes = results["dataset_sizes"]
    feature_dims = results["feature_dims"]
    
    # 1. MSE Comparison across feature dimensions
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(feature_dims):
        dim_key = f"{dim}d"
        plt.subplot(2, len(feature_dims)//2 + len(feature_dims)%2, i+1)
        
        q_mse = [results["quantum_mse"][dim_key][str(size)] for size in dataset_sizes]
        c_mse = [results["classical_mse"][dim_key][str(size)] for size in dataset_sizes]
        
        plt.plot(dataset_sizes, q_mse, 'o-', label='Quantum')
        plt.plot(dataset_sizes, c_mse, 's-', label='Classical')
        plt.xlabel('Dataset Size')
        plt.ylabel('Mean Squared Error')
        plt.title(f'{dim} Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_mse_comparison.png'))
    
    # 2. Improvement percentage across dimensions and dataset sizes
    plt.figure(figsize=(10, 6))
    for dim in feature_dims:
        dim_key = f"{dim}d"
        improvements = [results["improvement"][dim_key][str(size)] * 100 for size in dataset_sizes]
        plt.plot(dataset_sizes, improvements, 'o-', label=f'{dim} Features')
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Improvement (%)')
    plt.title('Quantum vs Classical Improvement')
    plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_improvement.png'))
    
    # 3. Training Time Comparison
    plt.figure(figsize=(10, 6))
    for dim in feature_dims:
        dim_key = f"{dim}d"
        q_times = [results["quantum_time"][dim_key][str(size)] for size in dataset_sizes]
        plt.plot(dataset_sizes, q_times, 'o-', label=f'{dim} Features')
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (s)')
    plt.title('Quantum Training Time by Feature Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_training_time.png'))
    
    # 4. Feature Dimension Impact on Performance
    plt.figure(figsize=(12, 8))
    
    # For the largest dataset size
    largest_size = str(dataset_sizes[-1])
    
    # MSE by dimension
    plt.subplot(2, 2, 1)
    q_mse = [results["quantum_mse"][f"{dim}d"][largest_size] for dim in feature_dims]
    c_mse = [results["classical_mse"][f"{dim}d"][largest_size] for dim in feature_dims]
    
    plt.plot(feature_dims, q_mse, 'o-', label='Quantum')
    plt.plot(feature_dims, c_mse, 's-', label='Classical')
    plt.xlabel('Feature Dimension')
    plt.ylabel('MSE')
    plt.title(f'Error by Dimension (n={largest_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improvement by dimension
    plt.subplot(2, 2, 2)
    improvements = [results["improvement"][f"{dim}d"][largest_size] * 100 for dim in feature_dims]
    
    plt.plot(feature_dims, improvements, 'o-', color='green')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Improvement (%)')
    plt.title(f'Quantum Advantage by Dimension (n={largest_size})')
    plt.axhline(y=0.0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    # Time by dimension
    plt.subplot(2, 2, 3)
    q_times = [results["quantum_time"][f"{dim}d"][largest_size] for dim in feature_dims]
    c_times = [results["classical_time"][f"{dim}d"][largest_size] for dim in feature_dims]
    
    plt.plot(feature_dims, q_times, 'o-', label='Quantum')
    plt.plot(feature_dims, c_times, 's-', label='Classical')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Training Time (s)')
    plt.title(f'Training Time by Dimension (n={largest_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time ratio by dimension
    plt.subplot(2, 2, 4)
    time_ratios = [q_times[i]/max(c_times[i], 0.001) for i in range(len(feature_dims))]
    
    plt.plot(feature_dims, time_ratios, 'o-', color='purple')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Time Ratio (Q/C)')
    plt.title(f'Training Time Ratio by Dimension (n={largest_size})')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_dimension_impact.png'))

def main():
    """Run classical vs quantum comparison benchmarks."""
    print_section("CLASSICAL VS QUANTUM COMPARISON")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Run Monte Carlo comparisons
    mc_results = compare_monte_carlo(iterations_list=[500, 1000, 2000, 4000], repeats=3)
    
    # Run Machine Learning comparisons
    ml_results = compare_machine_learning(dataset_sizes=[50, 100, 200], feature_dims=[2, 4, 6], repeats=2)
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("CLASSICAL VS QUANTUM COMPARISON SUMMARY\n")
        f.write("======================================\n\n")
        f.write(f"Comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Simulation Summary:\n")
        f.write("------------------------------\n")
        for dist in mc_results['distributions']:
            last_iter_idx = -1
            speedup = mc_results['speedup'][dist][last_iter_idx]
            q_mean = mc_results['quantum_mean'][dist][last_iter_idx]
            c_mean = mc_results['classical_mean'][dist][last_iter_idx]
            mean_diff = mc_results['mean_diff'][dist][last_iter_idx]
            
            f.write(f"  {dist.capitalize()} distribution:\n")
            f.write(f"    - Speedup: {speedup:.2f}x\n")
            f.write(f"    - Quantum mean: {q_mean:.4f}\n")
            f.write(f"    - Classical mean: {c_mean:.4f}\n")
            f.write(f"    - Mean difference: {mean_diff:.4f}\n\n")
        
        # Machine Learning summary
        f.write("Machine Learning Summary:\n")
        f.write("------------------------\n")
        for dim in ml_results['feature_dims']:
            dim_key = f"{dim}d"
            largest_size = str(ml_results['dataset_sizes'][-1])
            
            q_mse = ml_results['quantum_mse'][dim_key][largest_size]
            c_mse = ml_results['classical_mse'][dim_key][largest_size]
            improvement = ml_results['improvement'][dim_key][largest_size]
            
            f.write(f"  {dim} features (n={largest_size}):\n")
            f.write(f"    - Quantum MSE: {q_mse:.4f}\n")
            f.write(f"    - Classical MSE: {c_mse:.4f}\n")
            f.write(f"    - Improvement: {improvement*100:.1f}%\n\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nComparison completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 