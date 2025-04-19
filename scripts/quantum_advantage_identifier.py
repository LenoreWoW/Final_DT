#!/usr/bin/env python3
"""
Quantum Advantage Identifier

Pinpoints specific areas where quantum methods excel over classical approaches
by systematically varying problem parameters and measuring performance differences.
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
OUTPUT_DIR = "results/quantum_advantage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def identify_monte_carlo_advantage():
    """
    Identify quantum advantage regions for Monte Carlo simulations.
    Tests combinations of problem dimensions, distribution types, and iterations.
    
    Returns:
        Dictionary with advantage regions identified
    """
    print_section("MONTE CARLO ADVANTAGE IDENTIFICATION")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Running in simulation mode.")
    
    # Problem dimensions to test
    dimensions = [1, 2, 3, 5, 8]
    
    # Distribution types to test
    distributions = ["uniform", "normal", "exponential"]
    
    # Iteration counts to test
    iterations_list = [500, 1000, 2000, 5000]
    
    # Target functions with different complexity levels
    target_functions = {
        "linear": lambda *args: sum(args),
        "quadratic": lambda *args: sum(x**2 for x in args) + 0.5 * sum(args[i] * args[j] for i in range(len(args)) for j in range(i+1, len(args))),
        "periodic": lambda *args: sum(np.sin(3 * x) for x in args),
        "mixed": lambda *args: sum(x**2 for x in args) + sum(np.sin(3 * x) for x in args)
    }
    
    # Store results
    results = {
        "dimensions": dimensions,
        "distributions": distributions,
        "iterations": iterations_list,
        "functions": list(target_functions.keys()),
        "data": {}
    }
    
    # Run tests for each function type
    for func_name, func in target_functions.items():
        print(f"\nTesting {func_name} function...")
        results["data"][func_name] = {}
        
        for dist in distributions:
            print(f"  Testing {dist} distribution...")
            results["data"][func_name][dist] = {}
            
            for dim in dimensions:
                print(f"    Testing {dim} dimensions...")
                results["data"][func_name][dist][dim] = {
                    "quantum_time": [],
                    "classical_time": [],
                    "speedup": [],
                    "quantum_accuracy": [],
                    "classical_accuracy": []
                }
                
                # Create parameter ranges
                param_ranges = {}
                for i in range(dim):
                    param_ranges[f'x{i}'] = (-2.0, 2.0)
                
                # Create wrapped function to handle variable arguments
                def target_function(*args):
                    return func(*args)
                
                for iterations in iterations_list:
                    try:
                        # Run comparison between quantum and classical
                        comparison = qmc.compare_with_classical(
                            param_ranges,
                            target_function,
                            iterations=iterations,
                            distribution_type=dist
                        )
                        
                        # Extract results
                        quantum_time = comparison["quantum"]["execution_time"]
                        classical_time = comparison["classical"]["execution_time"]
                        speedup = comparison["speedup"]
                        
                        # For this demonstration, we'll use standard deviation as a proxy for accuracy
                        # (lower std dev indicates more precise sampling)
                        quantum_accuracy = 1.0 / (comparison["quantum"]["std"] + 1e-10)
                        classical_accuracy = 1.0 / (comparison["classical"]["std"] + 1e-10)
                        
                        # Store results
                        results["data"][func_name][dist][dim]["quantum_time"].append(quantum_time)
                        results["data"][func_name][dist][dim]["classical_time"].append(classical_time)
                        results["data"][func_name][dist][dim]["speedup"].append(speedup)
                        results["data"][func_name][dist][dim]["quantum_accuracy"].append(quantum_accuracy)
                        results["data"][func_name][dist][dim]["classical_accuracy"].append(classical_accuracy)
                        
                        print(f"      Iterations: {iterations}, Speedup: {speedup:.2f}x")
                        
                    except Exception as e:
                        logger.error(f"Error in MC advantage test: {str(e)}")
                        # Fill with NaN for failed tests
                        results["data"][func_name][dist][dim]["quantum_time"].append(float('nan'))
                        results["data"][func_name][dist][dim]["classical_time"].append(float('nan'))
                        results["data"][func_name][dist][dim]["speedup"].append(float('nan'))
                        results["data"][func_name][dist][dim]["quantum_accuracy"].append(float('nan'))
                        results["data"][func_name][dist][dim]["classical_accuracy"].append(float('nan'))
    
    # Identify regions of advantage
    advantage_regions = identify_advantage_regions(results)
    
    # Generate plots
    plot_monte_carlo_advantage(results, advantage_regions)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_advantage.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_advantage_regions.json'), 'w') as f:
        json.dump(advantage_regions, f, indent=2)
    
    return advantage_regions

def identify_ml_advantage():
    """
    Identify quantum advantage regions for machine learning applications.
    Tests combinations of dataset sizes, feature dimensions, and dataset complexity.
    
    Returns:
        Dictionary with advantage regions identified
    """
    print_section("MACHINE LEARNING ADVANTAGE IDENTIFICATION")
    
    # Initialize quantum ML
    config = ConfigManager()
    qml = QuantumML(config)
    
    if not qml.is_available():
        logger.warning("Quantum ML not available. Running in simulation mode.")
    
    # Basic QML configuration
    qml.feature_map = "zz"
    qml.n_layers = 2
    qml.ansatz_type = "strongly_entangling"
    qml.max_iterations = 25
    
    # Dataset sizes to test
    dataset_sizes = [50, 100, 150, 200]
    
    # Feature dimensions to test
    feature_dims = [2, 4, 6, 8]
    
    # Dataset complexity levels
    complexity_levels = ["linear", "nonlinear", "highly_nonlinear"]
    
    # Store results
    results = {
        "dataset_sizes": dataset_sizes,
        "feature_dims": feature_dims,
        "complexity_levels": complexity_levels,
        "data": {}
    }
    
    # Classical models to compare against
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    classical_models = {
        "linear": Ridge(alpha=0.1),
        "nonlinear": RandomForestRegressor(n_estimators=10)
    }
    
    # Run tests for each complexity level
    for complexity in complexity_levels:
        print(f"\nTesting {complexity} complexity level...")
        results["data"][complexity] = {}
        
        for feature_dim in feature_dims:
            print(f"  Testing with {feature_dim} features...")
            results["data"][complexity][feature_dim] = {}
            
            for dataset_size in dataset_sizes:
                print(f"    Testing with {dataset_size} samples...")
                results["data"][complexity][feature_dim][dataset_size] = {
                    "quantum_time": 0.0,
                    "classical_time": 0.0,
                    "quantum_mse": 0.0,
                    "classical_mse": 0.0,
                    "improvement": 0.0
                }
                
                try:
                    # Generate dataset with appropriate complexity
                    X, y = generate_ml_dataset(dataset_size, feature_dim, complexity)
                    
                    # Choose classical model based on complexity
                    classical_model = classical_models["linear"] if complexity == "linear" else classical_models["nonlinear"]
                    
                    # Compare quantum vs classical
                    comparison = qml.compare_with_classical(X, y, classical_model, test_size=0.3)
                    
                    # Store results
                    results["data"][complexity][feature_dim][dataset_size] = {
                        "quantum_time": comparison["quantum_training_time"],
                        "classical_time": comparison["classical_training_time"],
                        "quantum_mse": comparison["quantum_mse"],
                        "classical_mse": comparison["classical_mse"],
                        "improvement": comparison["relative_improvement"]
                    }
                    
                    print(f"      Quantum MSE: {comparison['quantum_mse']:.4f}, Classical MSE: {comparison['classical_mse']:.4f}")
                    print(f"      Improvement: {comparison['relative_improvement']*100:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Error in ML advantage test: {str(e)}")
                    # Keep default values for failed tests
    
    # Identify regions of advantage
    advantage_regions = identify_ml_advantage_regions(results)
    
    # Generate plots
    plot_ml_advantage(results, advantage_regions)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_advantage.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'ml_advantage_regions.json'), 'w') as f:
        json.dump(advantage_regions, f, indent=2)
    
    return advantage_regions

def generate_ml_dataset(n_samples, n_features, complexity):
    """Generate synthetic dataset with specified complexity."""
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    
    # Generate target values based on complexity
    if complexity == "linear":
        # Simple linear relationship
        weights = np.random.rand(n_features) * 2 - 1
        y = np.dot(X, weights) + 0.1 * np.random.randn(n_samples)
        
    elif complexity == "nonlinear":
        # Nonlinear with interactions and quadratic terms
        y = np.zeros(n_samples)
        
        # Linear terms
        for i in range(n_features):
            y += (i+1)/n_features * X[:, i]
        
        # Quadratic terms for first few features
        for i in range(min(3, n_features)):
            y += 0.5 * X[:, i]**2
        
        # Interaction terms
        for i in range(min(3, n_features)):
            for j in range(i+1, min(5, n_features)):
                y += 0.3 * X[:, i] * X[:, j]
                
        # Add noise
        y += 0.2 * np.random.randn(n_samples)
        
    else:  # highly_nonlinear
        # Complex nonlinear relationships
        y = np.zeros(n_samples)
        
        # Sinusoidal components
        for i in range(n_features):
            y += np.sin(3 * X[:, i] + (i/n_features))
        
        # Exponential terms
        for i in range(min(2, n_features)):
            y += 0.5 * np.exp(-3 * ((X[:, i] - 0.5)**2))
        
        # Higher-order interactions
        for i in range(min(3, n_features)):
            for j in range(i+1, min(4, n_features)):
                for k in range(j+1, min(5, n_features)):
                    y += 0.1 * X[:, i] * X[:, j] * X[:, k]
                    
        # Add noise
        y += 0.3 * np.random.randn(n_samples)
    
    return X, y

def identify_advantage_regions(results):
    """
    Identify regions of quantum advantage from Monte Carlo results.
    
    Args:
        results: Dictionary with test results
        
    Returns:
        Dictionary with identified advantage regions
    """
    advantage_regions = {
        "speedup": {},
        "accuracy": {},
        "overall": {}
    }
    
    # For each function type and distribution
    for func_name in results["functions"]:
        advantage_regions["speedup"][func_name] = {}
        advantage_regions["accuracy"][func_name] = {}
        advantage_regions["overall"][func_name] = {}
        
        for dist in results["distributions"]:
            advantage_regions["speedup"][func_name][dist] = []
            advantage_regions["accuracy"][func_name][dist] = []
            advantage_regions["overall"][func_name][dist] = []
            
            # Check each dimension for advantage
            for i, dim in enumerate(results["dimensions"]):
                # Check if we have speedup advantage
                mean_speedup = np.nanmean(results["data"][func_name][dist][dim]["speedup"])
                has_speedup = mean_speedup > 1.0
                
                # Check if we have accuracy advantage
                accuracy_ratios = [q/c for q, c in zip(
                    results["data"][func_name][dist][dim]["quantum_accuracy"],
                    results["data"][func_name][dist][dim]["classical_accuracy"]
                ) if not np.isnan(q) and not np.isnan(c)]
                
                mean_accuracy_ratio = np.mean(accuracy_ratios) if accuracy_ratios else 0
                has_accuracy = mean_accuracy_ratio > 1.0
                
                # Combined advantage
                has_overall = has_speedup and has_accuracy
                
                # Record advantages
                if has_speedup:
                    advantage_regions["speedup"][func_name][dist].append({
                        "dimension": dim,
                        "speedup": mean_speedup
                    })
                    
                if has_accuracy:
                    advantage_regions["accuracy"][func_name][dist].append({
                        "dimension": dim,
                        "accuracy_ratio": mean_accuracy_ratio
                    })
                    
                if has_overall:
                    advantage_regions["overall"][func_name][dist].append({
                        "dimension": dim,
                        "speedup": mean_speedup,
                        "accuracy_ratio": mean_accuracy_ratio
                    })
    
    return advantage_regions

def identify_ml_advantage_regions(results):
    """
    Identify regions of quantum advantage from ML results.
    
    Args:
        results: Dictionary with test results
        
    Returns:
        Dictionary with identified advantage regions
    """
    advantage_regions = {
        "speedup": {},
        "accuracy": {},
        "overall": {}
    }
    
    # For each complexity level
    for complexity in results["complexity_levels"]:
        advantage_regions["speedup"][complexity] = []
        advantage_regions["accuracy"][complexity] = []
        advantage_regions["overall"][complexity] = []
        
        # Check each feature dimension
        for feature_dim in results["feature_dims"]:
            # Check each dataset size
            for dataset_size in results["dataset_sizes"]:
                data = results["data"][complexity][feature_dim][dataset_size]
                
                # Check for speedup advantage
                speedup = data["classical_time"] / data["quantum_time"] if data["quantum_time"] > 0 else 0
                has_speedup = speedup > 1.0
                
                # Check for accuracy advantage
                accuracy_ratio = data["classical_mse"] / data["quantum_mse"] if data["quantum_mse"] > 0 else 0
                has_accuracy = accuracy_ratio > 1.0
                
                # Combined advantage
                has_overall = has_speedup and has_accuracy
                
                # Record advantages
                if has_speedup:
                    advantage_regions["speedup"][complexity].append({
                        "feature_dim": feature_dim,
                        "dataset_size": dataset_size,
                        "speedup": speedup
                    })
                    
                if has_accuracy:
                    advantage_regions["accuracy"][complexity].append({
                        "feature_dim": feature_dim,
                        "dataset_size": dataset_size,
                        "accuracy_ratio": accuracy_ratio
                    })
                    
                if has_overall:
                    advantage_regions["overall"][complexity].append({
                        "feature_dim": feature_dim,
                        "dataset_size": dataset_size,
                        "speedup": speedup,
                        "accuracy_ratio": accuracy_ratio
                    })
    
    return advantage_regions

def plot_monte_carlo_advantage(results, advantage_regions):
    """
    Generate plots for Monte Carlo quantum advantage.
    
    Args:
        results: Dictionary with test results
        advantage_regions: Dictionary with identified advantage regions
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Speedup heatmaps for each function type and distribution
    for func_name in results["functions"]:
        plt.figure(figsize=(15, 10))
        
        for j, dist in enumerate(results["distributions"]):
            plt.subplot(2, 2, j+1)
            
            # Create heatmap data
            heatmap_data = np.zeros((len(results["dimensions"]), len(results["iterations"])))
            
            for i, dim in enumerate(results["dimensions"]):
                for k, _ in enumerate(results["iterations"]):
                    if k < len(results["data"][func_name][dist][dim]["speedup"]):
                        heatmap_data[i, k] = results["data"][func_name][dist][dim]["speedup"][k]
            
            # Plot heatmap
            plt.imshow(heatmap_data, cmap='cool', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Speedup Factor')
            
            # Set labels
            plt.title(f'{dist.capitalize()} Distribution')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Dimension')
            
            plt.xticks(np.arange(len(results["iterations"])), results["iterations"])
            plt.yticks(np.arange(len(results["dimensions"])), results["dimensions"])
            
            # Highlight advantage regions
            for region in advantage_regions["overall"][func_name][dist]:
                dim_idx = results["dimensions"].index(region["dimension"])
                plt.axhline(y=dim_idx, color='white', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Quantum Advantage Map - {func_name.capitalize()} Function', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(plots_dir, f'mc_advantage_map_{func_name}.png'))
    
    # Summary bar chart of overall advantage
    plt.figure(figsize=(12, 8))
    
    # Count advantage regions for each function type
    advantage_counts = {func: sum(len(advantage_regions["overall"][func][dist]) for dist in results["distributions"]) 
                      for func in results["functions"]}
    
    # Plot
    plt.bar(advantage_counts.keys(), advantage_counts.values())
    plt.xlabel('Function Type')
    plt.ylabel('Number of Advantage Regions')
    plt.title('Quantum Advantage by Function Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_advantage_summary.png'))

def plot_ml_advantage(results, advantage_regions):
    """
    Generate plots for Machine Learning quantum advantage.
    
    Args:
        results: Dictionary with test results
        advantage_regions: Dictionary with identified advantage regions
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Improvement heatmaps for each complexity level
    for complexity in results["complexity_levels"]:
        plt.figure(figsize=(10, 8))
        
        # Create heatmap data for improvement percentage
        heatmap_data = np.zeros((len(results["feature_dims"]), len(results["dataset_sizes"])))
        
        for i, feature_dim in enumerate(results["feature_dims"]):
            for j, dataset_size in enumerate(results["dataset_sizes"]):
                improvement = results["data"][complexity][feature_dim][dataset_size]["improvement"]
                heatmap_data[i, j] = improvement * 100  # Convert to percentage
        
        # Plot heatmap
        plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest', vmin=-50, vmax=50)
        plt.colorbar(label='Improvement %')
        
        # Set labels
        plt.title(f'Quantum Advantage - {complexity.replace("_", " ").title()} Complexity')
        plt.xlabel('Dataset Size')
        plt.ylabel('Feature Dimension')
        
        plt.xticks(np.arange(len(results["dataset_sizes"])), results["dataset_sizes"])
        plt.yticks(np.arange(len(results["feature_dims"])), results["feature_dims"])
        
        # Highlight advantage regions
        for region in advantage_regions["overall"][complexity]:
            feature_idx = results["feature_dims"].index(region["feature_dim"])
            dataset_idx = results["dataset_sizes"].index(region["dataset_size"])
            plt.plot(dataset_idx, feature_idx, 'o', color='white', markersize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'ml_advantage_map_{complexity}.png'))
    
    # Summary of advantage counts by complexity
    advantage_counts = {complexity: len(advantage_regions["overall"][complexity]) 
                      for complexity in results["complexity_levels"]}
    
    plt.figure(figsize=(10, 6))
    plt.bar(advantage_counts.keys(), advantage_counts.values())
    plt.xlabel('Dataset Complexity')
    plt.ylabel('Number of Advantage Regions')
    plt.title('Quantum Advantage by Dataset Complexity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_advantage_summary.png'))
    
    # Feature dimension impact on advantage
    feature_advantage = {feature_dim: 0 for feature_dim in results["feature_dims"]}
    
    for complexity in results["complexity_levels"]:
        for region in advantage_regions["overall"][complexity]:
            feature_advantage[region["feature_dim"]] += 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_advantage.keys(), feature_advantage.values())
    plt.xlabel('Feature Dimension')
    plt.ylabel('Advantage Count')
    plt.title('Feature Dimension Impact on Quantum Advantage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_feature_impact.png'))

def main():
    """Run quantum advantage identification."""
    print_section("QUANTUM ADVANTAGE IDENTIFIER")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Identify Monte Carlo advantage regions
    mc_advantage = identify_monte_carlo_advantage()
    
    # Identify Machine Learning advantage regions
    ml_advantage = identify_ml_advantage()
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("QUANTUM ADVANTAGE IDENTIFICATION SUMMARY\n")
        f.write("=======================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Advantage Summary:\n")
        f.write("-----------------------------\n")
        
        # Count advantage regions by function type
        for func_name in mc_advantage["overall"]:
            total_regions = sum(len(regions) for regions in mc_advantage["overall"][func_name].values())
            f.write(f"  {func_name.capitalize()} function: {total_regions} advantage regions\n")
            
            # List best regions
            best_regions = []
            for dist in mc_advantage["overall"][func_name]:
                for region in mc_advantage["overall"][func_name][dist]:
                    best_regions.append((dist, region))
            
            # Sort by combined metric (speedup * accuracy_ratio)
            best_regions.sort(key=lambda x: x[1]["speedup"] * x[1]["accuracy_ratio"], reverse=True)
            
            if best_regions:
                best = best_regions[0]
                f.write(f"  Best region: {best[0]} distribution, {best[1]['dimension']} dimensions\n")
                f.write(f"    Speedup: {best[1]['speedup']:.2f}x, Accuracy ratio: {best[1]['accuracy_ratio']:.2f}\n\n")
            else:
                f.write("  No clear advantage regions identified\n\n")
        
        # Machine Learning summary
        f.write("Machine Learning Advantage Summary:\n")
        f.write("---------------------------------\n")
        
        # Count advantage regions by complexity
        for complexity in ml_advantage["overall"]:
            regions = ml_advantage["overall"][complexity]
            f.write(f"  {complexity.replace('_', ' ').title()} complexity: {len(regions)} advantage regions\n")
            
            if regions:
                # Sort by combined metric
                regions.sort(key=lambda x: x["speedup"] * x["accuracy_ratio"], reverse=True)
                best = regions[0]
                f.write(f"  Best region: Features={best['feature_dim']}, Samples={best['dataset_size']}\n")
                f.write(f"    Speedup: {best['speedup']:.2f}x, Accuracy ratio: {best['accuracy_ratio']:.2f}\n\n")
            else:
                f.write("  No clear advantage regions identified\n\n")
        
        # Overall findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        
        # Monte Carlo findings
        best_mc_func = max(
            [(func, sum(len(regions) for regions in mc_advantage["overall"][func].values())) 
             for func in mc_advantage["overall"]], 
            key=lambda x: x[1]
        )[0]
        
        f.write(f"1. Quantum Monte Carlo shows most advantage for {best_mc_func} functions\n")
        
        # ML findings
        advantage_by_complexity = {complexity: len(ml_advantage["overall"][complexity]) 
                                for complexity in ml_advantage["overall"]}
        best_ml_complexity = max(advantage_by_complexity.items(), key=lambda x: x[1], default=("none", 0))[0]
        
        if advantage_by_complexity[best_ml_complexity] > 0:
            f.write(f"2. Quantum ML shows most advantage for {best_ml_complexity.replace('_', ' ')} problems\n")
        else:
            f.write("2. No clear quantum ML advantage detected in tested scenarios\n")
        
        # Feature dimensionality findings
        feature_counts = {dim: 0 for dim in results["feature_dims"]}
        total_regions = 0
        
        for complexity in ml_advantage["overall"]:
            for region in ml_advantage["overall"][complexity]:
                feature_counts[region["feature_dim"]] += 1
                total_regions += 1
        
        if total_regions > 0:
            best_feature_dim = max(feature_counts.items(), key=lambda x: x[1])[0]
            f.write(f"3. Optimal feature dimensionality for quantum advantage: {best_feature_dim}\n")
            
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nAnalysis completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 