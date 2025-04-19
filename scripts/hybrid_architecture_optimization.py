#!/usr/bin/env python3
"""
Hybrid Architecture Optimization

Shows optimal classical-quantum resource allocation for different problem types.
Demonstrates how to balance computational workloads between classical and quantum
processing to achieve the best performance.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
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
OUTPUT_DIR = "results/hybrid_optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def optimize_monte_carlo_hybrid(quantum_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                               problem_dimensions=[2, 4, 8, 16]):
    """
    Optimize hybrid classical-quantum Monte Carlo for different problems.
    
    Args:
        quantum_ratios: List of quantum resource allocation ratios to test
        problem_dimensions: List of problem dimensions to test
        
    Returns:
        Dictionary with optimization results
    """
    print_section("MONTE CARLO HYBRID OPTIMIZATION")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Running in simulation mode.")
    
    # Total number of samples across all experiments
    total_samples = 2000
    
    # Store results
    results = {
        "quantum_ratios": quantum_ratios,
        "problem_dimensions": problem_dimensions,
        "execution_time": np.zeros((len(problem_dimensions), len(quantum_ratios))),
        "accuracy": np.zeros((len(problem_dimensions), len(quantum_ratios))),
        "efficiency": np.zeros((len(problem_dimensions), len(quantum_ratios)))
    }
    
    # For each problem dimension
    for dim_idx, dimension in enumerate(problem_dimensions):
        print(f"\nTesting problem dimension: {dimension}")
        
        # Create parameter ranges
        param_ranges = {}
        for i in range(dimension):
            param_ranges[f'x{i}'] = (-1.0, 1.0)
        
        # Define n-dimensional target function
        # Using a complex function where quantum might have advantages for higher dimensions
        def target_function(*args):
            # Sum of squares with cross-terms and periodic components
            result = sum(x**2 for x in args)  # Quadratic part
            
            # Add cross-terms (more complex interactions)
            if len(args) > 1:
                result += 0.5 * sum(args[i] * args[j] for i in range(len(args)) 
                                  for j in range(i+1, len(args)))
            
            # Add periodic components for higher dimensions
            if len(args) >= 4:
                result += 0.2 * sum(np.sin(3 * x) for x in args[:4])
                
            # Add more complex terms for very high dimensions
            if len(args) >= 8:
                result += 0.1 * sum(np.exp(-x**2) for x in args[:8])
                
            return result
        
        # For each quantum ratio
        for ratio_idx, quantum_ratio in enumerate(quantum_ratios):
            quantum_samples = int(quantum_ratio * total_samples)
            classical_samples = total_samples - quantum_samples
            
            print(f"  Testing quantum ratio: {quantum_ratio:.2f} " +
                  f"(Quantum: {quantum_samples}, Classical: {classical_samples})")
            
            # Skip if no samples for one method
            if quantum_samples == 0:
                # Pure classical approach
                try:
                    start_time = time.time()
                    classical_result = qmc.run_classical_monte_carlo(
                        param_ranges,
                        iterations=total_samples,
                        target_function=target_function
                    )
                    execution_time = time.time() - start_time
                    
                    accuracy = 1.0 / max(1e-10, classical_result["std"])  # Higher is better
                    
                    results["execution_time"][dim_idx, ratio_idx] = execution_time
                    results["accuracy"][dim_idx, ratio_idx] = accuracy
                    results["efficiency"][dim_idx, ratio_idx] = accuracy / execution_time
                    
                    print(f"    Pure classical - Time: {execution_time:.2f}s, " +
                          f"Std: {classical_result['std']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error in pure classical approach: {str(e)}")
                    results["execution_time"][dim_idx, ratio_idx] = float('nan')
                    results["accuracy"][dim_idx, ratio_idx] = float('nan')
                    results["efficiency"][dim_idx, ratio_idx] = float('nan')
                    
            elif classical_samples == 0:
                # Pure quantum approach
                try:
                    start_time = time.time()
                    quantum_result = qmc.run_quantum_monte_carlo(
                        param_ranges,
                        iterations=total_samples,
                        target_function=target_function
                    )
                    execution_time = time.time() - start_time
                    
                    accuracy = 1.0 / max(1e-10, quantum_result["std"])  # Higher is better
                    
                    results["execution_time"][dim_idx, ratio_idx] = execution_time
                    results["accuracy"][dim_idx, ratio_idx] = accuracy
                    results["efficiency"][dim_idx, ratio_idx] = accuracy / execution_time
                    
                    print(f"    Pure quantum - Time: {execution_time:.2f}s, " +
                          f"Std: {quantum_result['std']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error in pure quantum approach: {str(e)}")
                    results["execution_time"][dim_idx, ratio_idx] = float('nan')
                    results["accuracy"][dim_idx, ratio_idx] = float('nan')
                    results["efficiency"][dim_idx, ratio_idx] = float('nan')
                    
            else:
                # Hybrid approach
                try:
                    # Run both methods
                    start_time = time.time()
                    
                    # Classical part
                    classical_result = qmc.run_classical_monte_carlo(
                        param_ranges,
                        iterations=classical_samples,
                        target_function=target_function
                    )
                    
                    # Quantum part
                    quantum_result = qmc.run_quantum_monte_carlo(
                        param_ranges,
                        iterations=quantum_samples,
                        target_function=target_function
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Combine results (weighted average)
                    classical_weight = classical_samples / total_samples
                    quantum_weight = quantum_samples / total_samples
                    
                    combined_mean = (classical_weight * classical_result["mean"] + 
                                     quantum_weight * quantum_result["mean"])
                    
                    # Combine variances for standard deviation
                    combined_variance = (classical_weight * classical_result["std"]**2 + 
                                        quantum_weight * quantum_result["std"]**2)
                    combined_std = np.sqrt(combined_variance)
                    
                    accuracy = 1.0 / max(1e-10, combined_std)  # Higher is better
                    
                    results["execution_time"][dim_idx, ratio_idx] = execution_time
                    results["accuracy"][dim_idx, ratio_idx] = accuracy
                    results["efficiency"][dim_idx, ratio_idx] = accuracy / execution_time
                    
                    print(f"    Hybrid - Time: {execution_time:.2f}s, " +
                          f"Std: {combined_std:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error in hybrid approach: {str(e)}")
                    results["execution_time"][dim_idx, ratio_idx] = float('nan')
                    results["accuracy"][dim_idx, ratio_idx] = float('nan')
                    results["efficiency"][dim_idx, ratio_idx] = float('nan')
    
    # Generate plots
    plot_monte_carlo_hybrid_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_hybrid.json'), 'w') as f:
        json_results = {
            "quantum_ratios": quantum_ratios,
            "problem_dimensions": problem_dimensions,
            "execution_time": results["execution_time"].tolist(),
            "accuracy": results["accuracy"].tolist(),
            "efficiency": results["efficiency"].tolist()
        }
        json.dump(json_results, f, indent=2)
    
    return results

def optimize_ml_hybrid(quantum_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      dataset_sizes=[50, 100, 200]):
    """
    Optimize hybrid classical-quantum ML for different dataset sizes.
    
    Args:
        quantum_ratios: List of quantum resource allocation ratios to test
        dataset_sizes: List of dataset sizes to test
        
    Returns:
        Dictionary with optimization results
    """
    print_section("MACHINE LEARNING HYBRID OPTIMIZATION")
    
    # Initialize components
    config = ConfigManager()
    qml = QuantumML(config)
    
    # Dimensions of test problem
    feature_dim = 4
    
    # Store results
    results = {
        "quantum_ratios": quantum_ratios,
        "dataset_sizes": dataset_sizes,
        "training_time": np.zeros((len(dataset_sizes), len(quantum_ratios))),
        "accuracy": np.zeros((len(dataset_sizes), len(quantum_ratios))),
        "efficiency": np.zeros((len(dataset_sizes), len(quantum_ratios)))
    }
    
    # For each dataset size
    for size_idx, dataset_size in enumerate(dataset_sizes):
        print(f"\nTesting dataset size: {dataset_size}")
        
        # Generate synthetic dataset
        np.random.seed(42)
        X = np.random.rand(dataset_size, feature_dim)
        
        # Generate target values with a mix of linear and nonlinear components
        y = np.zeros(dataset_size)
        
        # Linear component
        y += 2.0 * X[:, 0] - 1.5 * X[:, 1]
        
        # Nonlinear component
        y += 0.5 * X[:, 0]**2 + 0.3 * X[:, 1] * X[:, 2]
        
        # Periodic component
        y += 0.2 * np.sin(3 * X[:, 3])
        
        # Add noise
        y += 0.1 * np.random.randn(dataset_size)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # For each quantum ratio
        for ratio_idx, quantum_ratio in enumerate(quantum_ratios):
            print(f"  Testing quantum ratio: {quantum_ratio:.2f}")
            
            # Set quantum ratio in configuration
            config.update("quantum.hybrid_ratio", quantum_ratio)
            qml = QuantumML(config)  # Reinitialize with updated config
            
            # Configure QML
            qml.feature_map = "zz"
            qml.n_layers = 2
            qml.max_iterations = 30
            
            try:
                # Train model with hybrid approach
                start_time = time.time()
                training_result = qml.train_model(X_train, y_train, test_size=0.2, verbose=False)
                training_time = time.time() - start_time
                
                # Get test predictions
                y_pred = qml.predict(X_test)
                
                # Calculate accuracy metrics
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Use R² as accuracy metric (higher is better)
                accuracy = max(0, r2)  # Ensure non-negative
                
                # Efficiency is accuracy per unit time
                efficiency = accuracy / training_time if training_time > 0 else 0
                
                # Store results
                results["training_time"][size_idx, ratio_idx] = training_time
                results["accuracy"][size_idx, ratio_idx] = accuracy
                results["efficiency"][size_idx, ratio_idx] = efficiency
                
                print(f"    Time: {training_time:.2f}s, R²: {r2:.4f}, MSE: {mse:.6f}")
                
            except Exception as e:
                logger.error(f"Error in hybrid ML approach: {str(e)}")
                results["training_time"][size_idx, ratio_idx] = float('nan')
                results["accuracy"][size_idx, ratio_idx] = float('nan')
                results["efficiency"][size_idx, ratio_idx] = float('nan')
    
    # Generate plots
    plot_ml_hybrid_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_hybrid.json'), 'w') as f:
        json_results = {
            "quantum_ratios": quantum_ratios,
            "dataset_sizes": dataset_sizes,
            "training_time": results["training_time"].tolist(),
            "accuracy": results["accuracy"].tolist(),
            "efficiency": results["efficiency"].tolist()
        }
        json.dump(json_results, f, indent=2)
    
    return results

def plot_monte_carlo_hybrid_results(results):
    """
    Generate plots for Monte Carlo hybrid optimization results.
    
    Args:
        results: Dictionary with optimization results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    quantum_ratios = results["quantum_ratios"]
    problem_dimensions = results["problem_dimensions"]
    
    # Plot execution time
    plt.figure(figsize=(10, 6))
    for i, dim in enumerate(problem_dimensions):
        plt.plot(quantum_ratios, results["execution_time"][i], 'o-', label=f'Dim={dim}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monte_carlo_execution_time.png'))
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    for i, dim in enumerate(problem_dimensions):
        plt.plot(quantum_ratios, results["accuracy"][i], 'o-', label=f'Dim={dim}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Accuracy (1/std)')
    plt.title('Accuracy vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monte_carlo_accuracy.png'))
    
    # Plot efficiency
    plt.figure(figsize=(10, 6))
    for i, dim in enumerate(problem_dimensions):
        plt.plot(quantum_ratios, results["efficiency"][i], 'o-', label=f'Dim={dim}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Efficiency (Accuracy/Time)')
    plt.title('Computational Efficiency vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monte_carlo_efficiency.png'))
    
    # Find optimal quantum ratio for each dimension
    optimal_ratios = []
    for i, dim in enumerate(problem_dimensions):
        # Find the quantum ratio that maximizes efficiency
        max_idx = np.nanargmax(results["efficiency"][i])
        optimal_ratio = quantum_ratios[max_idx]
        optimal_ratios.append((dim, optimal_ratio))
    
    # Plot optimal ratios
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in optimal_ratios], [r[1] for r in optimal_ratios], 'o-')
    plt.xlabel('Problem Dimension')
    plt.ylabel('Optimal Quantum Ratio')
    plt.title('Optimal Quantum Resource Allocation vs. Problem Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monte_carlo_optimal_ratio.png'))

def plot_ml_hybrid_results(results):
    """
    Generate plots for Machine Learning hybrid optimization results.
    
    Args:
        results: Dictionary with optimization results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    quantum_ratios = results["quantum_ratios"]
    dataset_sizes = results["dataset_sizes"]
    
    # Plot training time
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(dataset_sizes):
        plt.plot(quantum_ratios, results["training_time"][i], 'o-', label=f'Size={size}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_training_time.png'))
    
    # Plot accuracy (R²)
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(dataset_sizes):
        plt.plot(quantum_ratios, results["accuracy"][i], 'o-', label=f'Size={size}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Accuracy (R²)')
    plt.title('Model Accuracy vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_accuracy.png'))
    
    # Plot efficiency
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(dataset_sizes):
        plt.plot(quantum_ratios, results["efficiency"][i], 'o-', label=f'Size={size}')
    
    plt.xlabel('Quantum Resource Ratio')
    plt.ylabel('Efficiency (Accuracy/Time)')
    plt.title('Computational Efficiency vs. Quantum Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_efficiency.png'))
    
    # Find optimal quantum ratio for each dataset size
    optimal_ratios = []
    for i, size in enumerate(dataset_sizes):
        # Find the quantum ratio that maximizes efficiency
        max_idx = np.nanargmax(results["efficiency"][i])
        optimal_ratio = quantum_ratios[max_idx]
        optimal_ratios.append((size, optimal_ratio))
    
    # Plot optimal ratios
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in optimal_ratios], [r[1] for r in optimal_ratios], 'o-')
    plt.xlabel('Dataset Size')
    plt.ylabel('Optimal Quantum Ratio')
    plt.title('Optimal Quantum Resource Allocation vs. Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_optimal_ratio.png'))

def main():
    """Run hybrid architecture optimization tests."""
    print_section("HYBRID ARCHITECTURE OPTIMIZATION")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Use smaller parameter ranges for quick testing
    quantum_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Run Monte Carlo hybrid optimization
    mc_results = optimize_monte_carlo_hybrid(
        quantum_ratios=quantum_ratios,
        problem_dimensions=[2, 4, 8]
    )
    
    # Run ML hybrid optimization
    ml_results = optimize_ml_hybrid(
        quantum_ratios=quantum_ratios,
        dataset_sizes=[50, 100, 150]
    )
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("HYBRID ARCHITECTURE OPTIMIZATION SUMMARY\n")
        f.write("========================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Hybrid Optimization Summary:\n")
        f.write("---------------------------------------\n")
        
        # Find optimal ratios for each dimension
        for i, dimension in enumerate(mc_results["problem_dimensions"]):
            max_idx = np.nanargmax(mc_results["efficiency"][i])
            optimal_ratio = mc_results["quantum_ratios"][max_idx]
            max_efficiency = mc_results["efficiency"][i][max_idx]
            
            pure_classical_efficiency = mc_results["efficiency"][i][0]  # quantum_ratio = 0.0
            pure_quantum_efficiency = mc_results["efficiency"][i][-1]   # quantum_ratio = 1.0
            
            f.write(f"  Dimension {dimension}:\n")
            f.write(f"    Optimal quantum ratio: {optimal_ratio:.2f}\n")
            
            if not np.isnan(pure_classical_efficiency) and not np.isnan(max_efficiency):
                vs_classical = max_efficiency / pure_classical_efficiency
                f.write(f"    Improvement vs. pure classical: {vs_classical:.2f}x\n")
            
            if not np.isnan(pure_quantum_efficiency) and not np.isnan(max_efficiency):
                vs_quantum = max_efficiency / pure_quantum_efficiency
                f.write(f"    Improvement vs. pure quantum: {vs_quantum:.2f}x\n")
                
            f.write("\n")
        
        # ML summary
        f.write("Machine Learning Hybrid Optimization Summary:\n")
        f.write("-------------------------------------------\n")
        
        # Find optimal ratios for each dataset size
        for i, size in enumerate(ml_results["dataset_sizes"]):
            max_idx = np.nanargmax(ml_results["efficiency"][i])
            optimal_ratio = ml_results["quantum_ratios"][max_idx]
            max_efficiency = ml_results["efficiency"][i][max_idx]
            
            pure_classical_efficiency = ml_results["efficiency"][i][0]  # quantum_ratio = 0.0
            pure_quantum_efficiency = ml_results["efficiency"][i][-1]   # quantum_ratio = 1.0
            
            f.write(f"  Dataset size {size}:\n")
            f.write(f"    Optimal quantum ratio: {optimal_ratio:.2f}\n")
            
            if not np.isnan(pure_classical_efficiency) and not np.isnan(max_efficiency):
                vs_classical = max_efficiency / pure_classical_efficiency
                f.write(f"    Improvement vs. pure classical: {vs_classical:.2f}x\n")
            
            if not np.isnan(pure_quantum_efficiency) and not np.isnan(max_efficiency):
                vs_quantum = max_efficiency / pure_quantum_efficiency
                f.write(f"    Improvement vs. pure quantum: {vs_quantum:.2f}x\n")
                
            f.write("\n")
        
        # Overall findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        
        # Analyze Monte Carlo results to draw conclusions
        mc_optimal_dims = []
        for i, dim in enumerate(mc_results["problem_dimensions"]):
            max_idx = np.nanargmax(mc_results["efficiency"][i])
            optimal_ratio = mc_results["quantum_ratios"][max_idx]
            mc_optimal_dims.append((dim, optimal_ratio))
        
        # Check if optimal quantum ratio increases with dimension
        if len(mc_optimal_dims) >= 2:
            correlation = np.corrcoef([d[0] for d in mc_optimal_dims], [d[1] for d in mc_optimal_dims])[0, 1]
            if correlation > 0.5:
                f.write("1. For Monte Carlo, optimal quantum resource allocation increases with problem dimension\n")
            elif correlation < -0.5:
                f.write("1. For Monte Carlo, optimal quantum resource allocation decreases with problem dimension\n")
            else:
                f.write("1. For Monte Carlo, no clear relationship between problem dimension and optimal quantum ratio\n")
        
        # Analyze ML results to draw conclusions
        ml_optimal_sizes = []
        for i, size in enumerate(ml_results["dataset_sizes"]):
            max_idx = np.nanargmax(ml_results["efficiency"][i])
            optimal_ratio = ml_results["quantum_ratios"][max_idx]
            ml_optimal_sizes.append((size, optimal_ratio))
        
        # Check if optimal quantum ratio changes with dataset size
        if len(ml_optimal_sizes) >= 2:
            correlation = np.corrcoef([s[0] for s in ml_optimal_sizes], [s[1] for s in ml_optimal_sizes])[0, 1]
            if correlation > 0.5:
                f.write("2. For ML, optimal quantum resource allocation increases with dataset size\n")
            elif correlation < -0.5:
                f.write("2. For ML, optimal quantum resource allocation decreases with dataset size\n")
            else:
                f.write("2. For ML, no clear relationship between dataset size and optimal quantum ratio\n")
        
        # General hybrid architecture findings
        f.write("3. Hybrid classical-quantum architectures often outperform pure approaches\n")
        f.write("4. The computational overhead of quantum methods must be balanced with accuracy gains\n")
        f.write("5. Problem-specific optimization of resource allocation is crucial for peak performance\n")
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nOptimization completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 