#!/usr/bin/env python3
"""
Error Mitigation Demonstration
Shows improvements from various error mitigation techniques in quantum computing.
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
OUTPUT_DIR = "results/error_mitigation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_mitigation_monte_carlo(noise_levels=[0.01, 0.05, 0.1], iterations=1000, repeats=3):
    """
    Test error mitigation techniques on Monte Carlo simulation.
    
    Args:
        noise_levels: List of noise levels to test
        iterations: Number of iterations for Monte Carlo
        repeats: Number of repeats for statistical reliability
    
    Returns:
        Dictionary with test results
    """
    print_section("ERROR MITIGATION FOR MONTE CARLO")
    
    # Initialize quantum components
    config = ConfigManager()
    config.update("quantum.error_mitigation", False)  # Start with mitigation off
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Some features may be limited.")
    
    # Target function with varying complexity
    def target_function(x, y):
        return x**2 + y**2 + 0.5 * x * y
    
    # Parameter ranges
    param_ranges = {
        'x': (-2.0, 2.0),
        'y': (-2.0, 2.0),
    }
    
    # Mitigation techniques to test
    mitigation_techniques = ["none", "measurement", "zne", "combined"]
    
    # Store results
    results = {
        "noise_levels": noise_levels,
        "techniques": mitigation_techniques,
        "mean_error": {tech: [] for tech in mitigation_techniques},
        "std_dev": {tech: [] for tech in mitigation_techniques},
        "execution_time": {tech: [] for tech in mitigation_techniques}
    }
    
    # Get exact reference result from classical computation
    exact_result = 0
    n_samples = 1000000  # Large number for accurate reference
    x_samples = np.random.uniform(param_ranges['x'][0], param_ranges['x'][1], n_samples)
    y_samples = np.random.uniform(param_ranges['y'][0], param_ranges['y'][1], n_samples)
    values = [target_function(x, y) for x, y in zip(x_samples, y_samples)]
    exact_result = np.mean(values)
    
    logger.info(f"Exact result (reference): {exact_result:.6f}")
    
    # Run tests at different noise levels
    for noise_level in noise_levels:
        print(f"\nTesting with noise level: {noise_level}")
        
        # Configure noise level in config
        config.update("quantum.noise_level", noise_level)
        
        for technique in mitigation_techniques:
            print(f"  Testing {technique} mitigation...")
            
            # Configure mitigation technique
            if technique == "none":
                config.update("quantum.error_mitigation", False)
                config.update("quantum.measurement_mitigation", False)
                config.update("quantum.zero_noise_extrapolation", False)
            elif technique == "measurement":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", True)
                config.update("quantum.zero_noise_extrapolation", False)
            elif technique == "zne":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", False)
                config.update("quantum.zero_noise_extrapolation", True)
            elif technique == "combined":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", True)
                config.update("quantum.zero_noise_extrapolation", True)
            
            # Reload QMC with new config
            qmc = QuantumMonteCarlo(config)
            
            mean_errors = []
            stds = []
            times = []
            
            for r in range(repeats):
                try:
                    start_time = time.time()
                    
                    # Run Monte Carlo simulation
                    result = qmc.run_quantum_monte_carlo(
                        param_ranges, 
                        iterations=iterations,
                        target_function=target_function,
                        distribution_type="uniform"
                    )
                    
                    end_time = time.time()
                    
                    # Get error relative to exact result
                    mean_error = abs(result['mean'] - exact_result) / abs(exact_result)
                    execution_time = end_time - start_time
                    
                    mean_errors.append(mean_error)
                    stds.append(result['std'])
                    times.append(execution_time)
                    
                except Exception as e:
                    logger.error(f"Error in test (noise={noise_level}, technique={technique}, repeat={r}): {str(e)}")
            
            # Average results across repeats
            avg_mean_error = np.mean(mean_errors) if mean_errors else 1.0
            avg_std = np.mean(stds) if stds else 0.0
            avg_time = np.mean(times) if times else 0.0
            
            # Store results
            results["mean_error"][technique].append(avg_mean_error)
            results["std_dev"][technique].append(avg_std)
            results["execution_time"][technique].append(avg_time)
            
            print(f"    Relative Error: {avg_mean_error:.4%}")
            print(f"    Execution Time: {avg_time:.2f}s")
    
    # Generate plots
    plot_monte_carlo_mitigation_results(results, exact_result)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_mitigation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_monte_carlo_mitigation_results(results, exact_result):
    """
    Generate plots for Monte Carlo error mitigation results.
    
    Args:
        results: Dictionary with test results
        exact_result: Reference exact result
    """
    # Create plot directory
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    noise_levels = results["noise_levels"]
    techniques = results["techniques"]
    
    # 1. Error comparison across noise levels
    plt.figure(figsize=(10, 6))
    
    for technique in techniques:
        plt.plot(noise_levels, results["mean_error"][technique], 'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.title('Error Mitigation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_error_comparison.png'))
    
    # 2. Error reduction factor
    plt.figure(figsize=(10, 6))
    
    for technique in techniques[1:]:  # Skip "none"
        error_reduction = [results["mean_error"]["none"][i] / max(results["mean_error"][technique][i], 1e-10) 
                          for i in range(len(noise_levels))]
        plt.plot(noise_levels, error_reduction, 'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Error Reduction Factor')
    plt.title('Error Reduction from Mitigation')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_error_reduction.png'))
    
    # 3. Execution time comparison
    plt.figure(figsize=(10, 6))
    
    for technique in techniques:
        plt.plot(noise_levels, results["execution_time"][technique], 'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Execution Time (s)')
    plt.title('Computational Cost of Error Mitigation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_time_comparison.png'))
    
    # 4. Combined summary at highest noise level
    plt.figure(figsize=(12, 9))
    
    # Error bar plot
    plt.subplot(2, 2, 1)
    highest_noise_idx = -1
    errors = [results["mean_error"][tech][highest_noise_idx] for tech in techniques]
    x_pos = np.arange(len(techniques))
    
    plt.bar(x_pos, errors)
    plt.xticks(x_pos, [t.capitalize() for t in techniques])
    plt.ylabel('Relative Error')
    plt.title(f'Error at Noise Level = {noise_levels[highest_noise_idx]}')
    plt.yscale('log')
    
    # Time bar plot
    plt.subplot(2, 2, 2)
    times = [results["execution_time"][tech][highest_noise_idx] for tech in techniques]
    
    plt.bar(x_pos, times)
    plt.xticks(x_pos, [t.capitalize() for t in techniques])
    plt.ylabel('Execution Time (s)')
    plt.title('Computational Cost')
    
    # Error vs time scatter plot
    plt.subplot(2, 2, 3)
    
    plt.scatter(times, errors)
    for i, technique in enumerate(techniques):
        plt.annotate(technique.capitalize(), (times[i], errors[i]))
    
    plt.xlabel('Execution Time (s)')
    plt.ylabel('Relative Error')
    plt.title('Error vs. Computational Cost')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Error reduction per additional time
    plt.subplot(2, 2, 4)
    
    base_error = errors[0]  # "None" technique
    base_time = times[0]    # "None" technique
    
    error_reduction_per_time = [(base_error - errors[i]) / max((times[i] - base_time), 0.001) 
                              for i in range(1, len(techniques))]
    
    plt.bar(np.arange(len(techniques)-1), error_reduction_per_time)
    plt.xticks(np.arange(len(techniques)-1), [t.capitalize() for t in techniques[1:]])
    plt.ylabel('Error Reduction / Additional Time')
    plt.title('Efficiency of Mitigation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_mitigation_summary.png'))

def test_mitigation_machine_learning(noise_levels=[0.01, 0.05, 0.1], dataset_size=100, feature_dim=4, repeats=3):
    """
    Test error mitigation techniques on quantum machine learning.
    
    Args:
        noise_levels: List of noise levels to test
        dataset_size: Size of the synthetic dataset
        feature_dim: Number of features in the dataset
        repeats: Number of repeats for statistical reliability
    
    Returns:
        Dictionary with test results
    """
    print_section("ERROR MITIGATION FOR MACHINE LEARNING")
    
    # Initialize quantum components
    config = ConfigManager()
    config.update("quantum.error_mitigation", False)  # Start with mitigation off
    qml = QuantumML(config)
    
    if not qml.is_available():
        logger.warning("Quantum ML not available. Some features may be limited.")
    
    # Mitigation techniques to test
    mitigation_techniques = ["none", "measurement", "zne", "combined"]
    
    # Store results
    results = {
        "noise_levels": noise_levels,
        "techniques": mitigation_techniques,
        "mse": {tech: [] for tech in mitigation_techniques},
        "training_time": {tech: [] for tech in mitigation_techniques},
        "improvement": {tech: [] for tech in mitigation_techniques}
    }
    
    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.rand(dataset_size, feature_dim)
    
    # Target function with nonlinear components
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2 + 0.05 * np.random.randn(dataset_size)
    
    # Run tests at different noise levels
    for noise_level in noise_levels:
        print(f"\nTesting with noise level: {noise_level}")
        
        # Configure noise level in config
        config.update("quantum.noise_level", noise_level)
        
        for technique in mitigation_techniques:
            print(f"  Testing {technique} mitigation...")
            
            # Configure mitigation technique
            if technique == "none":
                config.update("quantum.error_mitigation", False)
                config.update("quantum.measurement_mitigation", False)
                config.update("quantum.zero_noise_extrapolation", False)
            elif technique == "measurement":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", True)
                config.update("quantum.zero_noise_extrapolation", False)
            elif technique == "zne":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", False)
                config.update("quantum.zero_noise_extrapolation", True)
            elif technique == "combined":
                config.update("quantum.error_mitigation", True)
                config.update("quantum.measurement_mitigation", True)
                config.update("quantum.zero_noise_extrapolation", True)
            
            # Reload QML with new config
            qml = QuantumML(config)
            
            # Basic QML configuration
            qml.feature_map = "zz"
            qml.n_layers = 2
            qml.ansatz_type = "strongly_entangling"
            qml.max_iterations = 30
            
            mses = []
            times = []
            improvements = []
            
            for r in range(repeats):
                try:
                    # Train quantum model and compare with classical
                    from sklearn.linear_model import Ridge
                    classical_model = Ridge(alpha=0.1)
                    
                    # First train quantum model
                    start_time = time.time()
                    quantum_result = qml.train_model(X, y, test_size=0.3, verbose=False)
                    training_time = time.time() - start_time
                    
                    # Compare with classical
                    comparison = qml.compare_with_classical(X, y, classical_model, test_size=0.3)
                    
                    quantum_mse = comparison["quantum_mse"]
                    classical_mse = comparison["classical_mse"]
                    improvement = comparison["relative_improvement"]
                    
                    mses.append(quantum_mse)
                    times.append(training_time)
                    improvements.append(improvement)
                    
                except Exception as e:
                    logger.error(f"Error in ML test (noise={noise_level}, technique={technique}, repeat={r}): {str(e)}")
            
            # Average results across repeats
            avg_mse = np.mean(mses) if mses else 1.0
            avg_time = np.mean(times) if times else 0.0
            avg_improvement = np.mean(improvements) if improvements else 0.0
            
            # Store results
            results["mse"][technique].append(avg_mse)
            results["training_time"][technique].append(avg_time)
            results["improvement"][technique].append(avg_improvement)
            
            print(f"    MSE: {avg_mse:.4f}")
            print(f"    Training Time: {avg_time:.2f}s")
            print(f"    Improvement vs Classical: {avg_improvement*100:.1f}%")
    
    # Generate plots
    plot_machine_learning_mitigation_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_mitigation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_machine_learning_mitigation_results(results):
    """
    Generate plots for machine learning error mitigation results.
    
    Args:
        results: Dictionary with test results
    """
    # Create plot directory
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    noise_levels = results["noise_levels"]
    techniques = results["techniques"]
    
    # 1. MSE comparison across noise levels
    plt.figure(figsize=(10, 6))
    
    for technique in techniques:
        plt.plot(noise_levels, results["mse"][technique], 'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Mitigation Impact on MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_mse_comparison.png'))
    
    # 2. Training time comparison
    plt.figure(figsize=(10, 6))
    
    for technique in techniques:
        plt.plot(noise_levels, results["training_time"][technique], 'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Training Time (s)')
    plt.title('Computational Cost of Error Mitigation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_time_comparison.png'))
    
    # 3. Improvement vs classical comparison
    plt.figure(figsize=(10, 6))
    
    for technique in techniques:
        plt.plot(noise_levels, [100 * imp for imp in results["improvement"][technique]], 
                'o-', label=technique.capitalize())
    
    plt.xlabel('Noise Level')
    plt.ylabel('Improvement vs Classical (%)')
    plt.title('Impact of Error Mitigation on Quantum Advantage')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_improvement_comparison.png'))
    
    # 4. Combined summary at highest noise level
    plt.figure(figsize=(12, 9))
    
    # MSE bar plot
    plt.subplot(2, 2, 1)
    highest_noise_idx = -1
    mses = [results["mse"][tech][highest_noise_idx] for tech in techniques]
    x_pos = np.arange(len(techniques))
    
    plt.bar(x_pos, mses)
    plt.xticks(x_pos, [t.capitalize() for t in techniques])
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE at Noise Level = {noise_levels[highest_noise_idx]}')
    
    # Time bar plot
    plt.subplot(2, 2, 2)
    times = [results["training_time"][tech][highest_noise_idx] for tech in techniques]
    
    plt.bar(x_pos, times)
    plt.xticks(x_pos, [t.capitalize() for t in techniques])
    plt.ylabel('Training Time (s)')
    plt.title('Computational Cost')
    
    # Improvement bar plot
    plt.subplot(2, 2, 3)
    improvements = [100 * results["improvement"][tech][highest_noise_idx] for tech in techniques]
    
    plt.bar(x_pos, improvements)
    plt.xticks(x_pos, [t.capitalize() for t in techniques])
    plt.ylabel('Improvement vs Classical (%)')
    plt.title('Quantum Advantage')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # MSE reduction per additional time
    plt.subplot(2, 2, 4)
    
    base_mse = mses[0]  # "None" technique
    base_time = times[0]    # "None" technique
    
    mse_reduction_per_time = [(base_mse - mses[i]) / max((times[i] - base_time), 0.001) 
                            for i in range(1, len(techniques))]
    
    plt.bar(np.arange(len(techniques)-1), mse_reduction_per_time)
    plt.xticks(np.arange(len(techniques)-1), [t.capitalize() for t in techniques[1:]])
    plt.ylabel('MSE Reduction / Additional Time')
    plt.title('Efficiency of Mitigation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_mitigation_summary.png'))

def main():
    """Run error mitigation demonstration."""
    print_section("ERROR MITIGATION DEMONSTRATION")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Test error mitigation on Monte Carlo
    mc_results = test_mitigation_monte_carlo(noise_levels=[0.01, 0.05, 0.1], iterations=1000, repeats=3)
    
    # Test error mitigation on Machine Learning
    ml_results = test_mitigation_machine_learning(noise_levels=[0.01, 0.05, 0.1], dataset_size=100, feature_dim=4, repeats=2)
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("ERROR MITIGATION DEMONSTRATION SUMMARY\n")
        f.write("====================================\n\n")
        f.write(f"Demonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Simulation Summary:\n")
        f.write("------------------------------\n")
        highest_noise_idx = -1
        noise_level = mc_results['noise_levels'][highest_noise_idx]
        
        f.write(f"Results at noise level {noise_level}:\n")
        for technique in mc_results['techniques']:
            error = mc_results['mean_error'][technique][highest_noise_idx]
            time_taken = mc_results['execution_time'][technique][highest_noise_idx]
            
            f.write(f"  {technique.capitalize()}:\n")
            f.write(f"    - Error: {error:.4%}\n")
            f.write(f"    - Time: {time_taken:.2f}s\n")
            
            if technique != "none":
                error_reduction = mc_results['mean_error']["none"][highest_noise_idx] / max(error, 1e-10)
                f.write(f"    - Error reduction factor: {error_reduction:.2f}x\n")
            
            f.write("\n")
        
        # Machine Learning summary
        f.write("Machine Learning Summary:\n")
        f.write("------------------------\n")
        highest_noise_idx = -1
        noise_level = ml_results['noise_levels'][highest_noise_idx]
        
        f.write(f"Results at noise level {noise_level}:\n")
        for technique in ml_results['techniques']:
            mse = ml_results['mse'][technique][highest_noise_idx]
            time_taken = ml_results['training_time'][technique][highest_noise_idx]
            improvement = ml_results['improvement'][technique][highest_noise_idx]
            
            f.write(f"  {technique.capitalize()}:\n")
            f.write(f"    - MSE: {mse:.4f}\n")
            f.write(f"    - Training time: {time_taken:.2f}s\n")
            f.write(f"    - Improvement vs classical: {improvement*100:.1f}%\n")
            
            if technique != "none":
                mse_reduction = ml_results['mse']["none"][highest_noise_idx] / max(mse, 1e-10)
                f.write(f"    - MSE reduction factor: {mse_reduction:.2f}x\n")
            
            f.write("\n")
        
        # Key findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        
        # Monte Carlo
        best_mc_technique = min([(technique, mc_results['mean_error'][technique][highest_noise_idx]) 
                               for technique in mc_results['techniques']], key=lambda x: x[1])[0]
        best_mc_reduction = mc_results['mean_error']["none"][highest_noise_idx] / max(mc_results['mean_error'][best_mc_technique][highest_noise_idx], 1e-10)
        
        f.write(f"1. For Monte Carlo simulations, {best_mc_technique.capitalize()} mitigation provides the best error reduction ({best_mc_reduction:.2f}x).\n")
        
        # Machine Learning
        best_ml_technique = min([(technique, ml_results['mse'][technique][highest_noise_idx]) 
                              for technique in ml_results['techniques']], key=lambda x: x[1])[0]
        best_ml_reduction = ml_results['mse']["none"][highest_noise_idx] / max(ml_results['mse'][best_ml_technique][highest_noise_idx], 1e-10)
        
        f.write(f"2. For Machine Learning, {best_ml_technique.capitalize()} mitigation provides the best MSE reduction ({best_ml_reduction:.2f}x).\n")
        
        # Overall recommendation
        f.write(f"3. At higher noise levels ({noise_level}), error mitigation becomes increasingly crucial for maintaining quantum advantage.\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nDemonstration completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 