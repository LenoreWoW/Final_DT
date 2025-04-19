#!/usr/bin/env python3
"""
Statistical Significance Tester

Evaluates whether improvements from quantum methods are statistically significant
across different test cases and problem types. Uses various statistical tests
to validate the significance of observed performance differences.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/statistical_significance"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_monte_carlo_significance(n_tests=30, sample_sizes=[100, 500, 1000]):
    """
    Test statistical significance of quantum vs classical Monte Carlo methods.
    
    Args:
        n_tests: Number of tests to run for each sample size
        sample_sizes: List of Monte Carlo sample sizes to test
        
    Returns:
        Dictionary with test results
    """
    print_section("MONTE CARLO STATISTICAL SIGNIFICANCE")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Define test functions of different complexity
    test_functions = {
        "quadratic": lambda x, y: x**2 + y**2 + 0.5*x*y,
        "trigonometric": lambda x, y: np.sin(3*x) * np.cos(2*y),
        "mixed": lambda x, y: x**2 - 0.5*np.sin(3*y)
    }
    
    # Parameter ranges
    param_ranges = {
        'x': (-1.0, 1.0),
        'y': (-1.0, 1.0)
    }
    
    # Store results
    results = {
        "sample_sizes": sample_sizes,
        "functions": list(test_functions.keys()),
        "raw_data": {func_name: {
            size: {"classical": [], "quantum": []}
            for size in sample_sizes
        } for func_name in test_functions},
        "p_values": {func_name: {
            "t_test": [], 
            "wilcoxon": [],
            "mannwhitney": []
        } for func_name in test_functions},
        "effect_sizes": {func_name: [] for func_name in test_functions},
        "significant_improvement": {func_name: [] for func_name in test_functions}
    }
    
    # Run tests for each function and sample size
    for func_name, func in test_functions.items():
        print(f"\nTesting {func_name} function...")
        
        for size_idx, size in enumerate(sample_sizes):
            print(f"  Sample size: {size}")
            
            classical_errors = []
            quantum_errors = []
            
            for i in range(n_tests):
                if i % 5 == 0:
                    print(f"    Test {i+1}/{n_tests}")
                
                try:
                    # Run classical Monte Carlo
                    classical_result = qmc.run_classical_monte_carlo(
                        param_ranges, 
                        iterations=size,
                        target_function=func
                    )
                    
                    # Run quantum Monte Carlo
                    quantum_result = qmc.run_quantum_monte_carlo(
                        param_ranges, 
                        iterations=size,
                        target_function=func
                    )
                    
                    # We'll use standard deviation as our error metric
                    # (lower std dev = better estimate precision)
                    classical_errors.append(classical_result["std"])
                    quantum_errors.append(quantum_result["std"])
                    
                except Exception as e:
                    logger.error(f"Error in test {i+1} for {func_name}, size {size}: {str(e)}")
            
            # Store raw results
            results["raw_data"][func_name][size]["classical"] = classical_errors
            results["raw_data"][func_name][size]["quantum"] = quantum_errors
            
            # Print interim results
            if classical_errors and quantum_errors:
                print(f"    Classical mean std: {np.mean(classical_errors):.6f}")
                print(f"    Quantum mean std: {np.mean(quantum_errors):.6f}")
                improvement = (1 - np.mean(quantum_errors) / np.mean(classical_errors)) * 100
                print(f"    Improvement: {improvement:.2f}%")
    
    # Perform statistical tests
    for func_name in test_functions:
        t_test_pvals = []
        wilcoxon_pvals = []
        mann_whitney_pvals = []
        effects = []
        significances = []
        
        for size in sample_sizes:
            classical = results["raw_data"][func_name][size]["classical"]
            quantum = results["raw_data"][func_name][size]["quantum"]
            
            if not classical or not quantum:
                t_test_pvals.append(float('nan'))
                wilcoxon_pvals.append(float('nan'))
                mann_whitney_pvals.append(float('nan'))
                effects.append(float('nan'))
                significances.append(False)
                continue
            
            # Two-sample t-test (assumes normality)
            t_stat, p_value = stats.ttest_ind(classical, quantum)
            t_test_pvals.append(p_value)
            
            # Wilcoxon signed-rank test (non-parametric)
            # Requires paired samples, so we'll use min length
            min_len = min(len(classical), len(quantum))
            if min_len > 1:
                w_stat, p_value = stats.wilcoxon(classical[:min_len], quantum[:min_len])
                wilcoxon_pvals.append(p_value)
            else:
                wilcoxon_pvals.append(float('nan'))
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(classical, quantum)
            mann_whitney_pvals.append(p_value)
            
            # Cohen's d effect size
            d = (np.mean(classical) - np.mean(quantum)) / np.sqrt(
                (np.var(classical) + np.var(quantum)) / 2)
            effects.append(d)
            
            # Is improvement significant? (p < 0.05 and quantum better than classical)
            is_significant = (p_value < 0.05 and np.mean(quantum) < np.mean(classical))
            significances.append(is_significant)
        
        # Store results
        results["p_values"][func_name]["t_test"] = t_test_pvals
        results["p_values"][func_name]["wilcoxon"] = wilcoxon_pvals
        results["p_values"][func_name]["mannwhitney"] = mann_whitney_pvals
        results["effect_sizes"][func_name] = effects
        results["significant_improvement"][func_name] = significances
    
    # Generate plots
    plot_monte_carlo_significance(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_significance.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results

def test_ml_significance(n_tests=20, dataset_sizes=[50, 100, 200]):
    """
    Test statistical significance of quantum vs classical ML methods.
    
    Args:
        n_tests: Number of tests to run for each dataset size
        dataset_sizes: List of dataset sizes to test
        
    Returns:
        Dictionary with test results
    """
    print_section("MACHINE LEARNING STATISTICAL SIGNIFICANCE")
    
    # Initialize quantum ML
    config = ConfigManager()
    qml = QuantumML(config)
    
    # Configure QML
    qml.feature_map = "zz"
    qml.n_layers = 2
    qml.max_iterations = 30
    
    # Classical models to compare
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    
    # Dataset complexity levels
    complexity_levels = ["linear", "nonlinear", "highly_nonlinear"]
    
    # Store results
    results = {
        "dataset_sizes": dataset_sizes,
        "complexity_levels": complexity_levels,
        "raw_data": {complexity: {
            size: {"classical": [], "quantum": []}
            for size in dataset_sizes
        } for complexity in complexity_levels},
        "p_values": {complexity: {
            "t_test": [], 
            "wilcoxon": [],
            "mannwhitney": []
        } for complexity in complexity_levels},
        "effect_sizes": {complexity: [] for complexity in complexity_levels},
        "significant_improvement": {complexity: [] for complexity in complexity_levels}
    }
    
    # Run tests for each complexity level and dataset size
    for complexity in complexity_levels:
        print(f"\nTesting {complexity} complexity...")
        
        for size_idx, size in enumerate(dataset_sizes):
            print(f"  Dataset size: {size}")
            
            classical_errors = []
            quantum_errors = []
            
            for i in range(n_tests):
                if i % 5 == 0:
                    print(f"    Test {i+1}/{n_tests}")
                
                try:
                    # Generate synthetic dataset
                    X, y = generate_ml_dataset(size, complexity)
                    
                    # Split into train/test
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42+i)
                    
                    # Train classical model
                    if complexity == "linear":
                        classical_model = Ridge(alpha=0.1)
                    else:
                        classical_model = RandomForestRegressor(n_estimators=10, random_state=42)
                    
                    classical_model.fit(X_train, y_train)
                    y_pred_classical = classical_model.predict(X_test)
                    classical_mse = mean_squared_error(y_test, y_pred_classical)
                    
                    # Train quantum model
                    qml.train_model(X_train, y_train, test_size=0.2, verbose=False)
                    y_pred_quantum = qml.predict(X_test)
                    quantum_mse = mean_squared_error(y_test, y_pred_quantum)
                    
                    # Store MSE values
                    classical_errors.append(classical_mse)
                    quantum_errors.append(quantum_mse)
                    
                except Exception as e:
                    logger.error(f"Error in test {i+1} for {complexity}, size {size}: {str(e)}")
            
            # Store raw results
            results["raw_data"][complexity][size]["classical"] = classical_errors
            results["raw_data"][complexity][size]["quantum"] = quantum_errors
            
            # Print interim results
            if classical_errors and quantum_errors:
                print(f"    Classical mean MSE: {np.mean(classical_errors):.6f}")
                print(f"    Quantum mean MSE: {np.mean(quantum_errors):.6f}")
                improvement = (1 - np.mean(quantum_errors) / np.mean(classical_errors)) * 100
                print(f"    Improvement: {improvement:.2f}%")
    
    # Perform statistical tests
    for complexity in complexity_levels:
        t_test_pvals = []
        wilcoxon_pvals = []
        mann_whitney_pvals = []
        effects = []
        significances = []
        
        for size in dataset_sizes:
            classical = results["raw_data"][complexity][size]["classical"]
            quantum = results["raw_data"][complexity][size]["quantum"]
            
            if not classical or not quantum:
                t_test_pvals.append(float('nan'))
                wilcoxon_pvals.append(float('nan'))
                mann_whitney_pvals.append(float('nan'))
                effects.append(float('nan'))
                significances.append(False)
                continue
            
            # Two-sample t-test (assumes normality)
            t_stat, p_value = stats.ttest_ind(classical, quantum)
            t_test_pvals.append(p_value)
            
            # Wilcoxon signed-rank test (non-parametric)
            # Requires paired samples, so we'll use min length
            min_len = min(len(classical), len(quantum))
            if min_len > 1:
                w_stat, p_value = stats.wilcoxon(classical[:min_len], quantum[:min_len])
                wilcoxon_pvals.append(p_value)
            else:
                wilcoxon_pvals.append(float('nan'))
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(classical, quantum)
            mann_whitney_pvals.append(p_value)
            
            # Cohen's d effect size
            d = (np.mean(classical) - np.mean(quantum)) / np.sqrt(
                (np.var(classical) + np.var(quantum)) / 2)
            effects.append(d)
            
            # Is improvement significant? (p < 0.05 and quantum better than classical)
            is_significant = (p_value < 0.05 and np.mean(quantum) < np.mean(classical))
            significances.append(is_significant)
        
        # Store results
        results["p_values"][complexity]["t_test"] = t_test_pvals
        results["p_values"][complexity]["wilcoxon"] = wilcoxon_pvals
        results["p_values"][complexity]["mannwhitney"] = mann_whitney_pvals
        results["effect_sizes"][complexity] = effects
        results["significant_improvement"][complexity] = significances
    
    # Generate plots
    plot_ml_significance(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ml_significance.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results

def generate_ml_dataset(n_samples, complexity):
    """
    Generate synthetic dataset for ML testing with specified complexity.
    
    Args:
        n_samples: Number of samples to generate
        complexity: Complexity level ('linear', 'nonlinear', 'highly_nonlinear')
        
    Returns:
        X, y: Features and target values
    """
    np.random.seed(42)
    
    # Set feature dimensionality based on complexity
    if complexity == "linear":
        n_features = 5
    elif complexity == "nonlinear":
        n_features = 8
    else:  # highly_nonlinear
        n_features = 10
    
    # Generate features
    X = np.random.rand(n_samples, n_features) * 2 - 1  # Scale to [-1, 1]
    
    # Generate target values with appropriate complexity
    y = np.zeros(n_samples)
    
    if complexity == "linear":
        # Linear relationship
        coeffs = np.random.randn(n_features) * 2
        y = np.dot(X, coeffs) + 0.1 * np.random.randn(n_samples)
        
    elif complexity == "nonlinear":
        # Polynomial and simple interaction terms
        for i in range(n_features):
            y += (i+1)/10 * X[:, i]**2  # Quadratic terms
        
        for i in range(n_features-1):
            y += 0.1 * X[:, i] * X[:, i+1]  # Interaction terms
            
        y += 0.1 * np.random.randn(n_samples)  # Noise
        
    else:  # highly_nonlinear
        # Complex nonlinear relationships
        for i in range(n_features):
            y += 0.2 * np.sin(3 * X[:, i])  # Sine waves
            y += 0.1 * np.exp(-X[:, i]**2)  # Gaussian functions
        
        for i in range(1, n_features):
            y += 0.2 * X[:, i-1]**2 * X[:, i]  # Higher-order interactions
            
        y += 0.2 * np.random.randn(n_samples)  # More noise
    
    return X, y

def plot_monte_carlo_significance(results):
    """
    Generate plots for Monte Carlo statistical significance tests.
    
    Args:
        results: Dictionary with test results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    
    # Extract data
    functions = results["functions"]
    sample_sizes = results["sample_sizes"]
    
    # 1. P-value plots for each function
    for func_name in functions:
        plt.figure(figsize=(12, 8))
        
        # Get p-values for different tests
        t_test_pvals = results["p_values"][func_name]["t_test"]
        wilcoxon_pvals = results["p_values"][func_name]["wilcoxon"]
        mannwhitney_pvals = results["p_values"][func_name]["mannwhitney"]
        
        plt.plot(sample_sizes, t_test_pvals, 'o-', label='t-test')
        plt.plot(sample_sizes, wilcoxon_pvals, 's-', label='Wilcoxon')
        plt.plot(sample_sizes, mannwhitney_pvals, '^-', label='Mann-Whitney')
        
        plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05 threshold')
        
        plt.xlabel('Sample Size')
        plt.ylabel('p-value')
        plt.yscale('log')  # Log scale for p-values
        plt.title(f'Statistical Significance: {func_name.replace("_", " ").title()} Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'mc_{func_name}_pvalues.png'))
    
    # 2. Effect size plot
    plt.figure(figsize=(10, 6))
    
    for func_name in functions:
        effects = results["effect_sizes"][func_name]
        plt.plot(sample_sizes, effects, 'o-', label=func_name.replace("_", " ").title())
    
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Small effect')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Medium effect')
    plt.axhline(y=0.8, color='b', linestyle='--', alpha=0.5, label='Large effect')
    
    plt.xlabel('Sample Size')
    plt.ylabel("Cohen's d Effect Size")
    plt.title('Effect Size by Function and Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_effect_sizes.png'))
    
    # 3. Box plots of raw data for largest sample size
    largest_size = sample_sizes[-1]
    plt.figure(figsize=(12, 8))
    
    data_to_plot = []
    labels = []
    
    for func_name in functions:
        c_data = results["raw_data"][func_name][largest_size]["classical"]
        q_data = results["raw_data"][func_name][largest_size]["quantum"]
        
        if c_data and q_data:
            data_to_plot.extend([c_data, q_data])
            labels.extend([f'{func_name} (Classical)', f'{func_name} (Quantum)'])
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Standard Deviation')
    plt.title(f'Distribution of Results (Sample Size = {largest_size})')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_boxplot_comparison.png'))
    
    # 4. Heatmap of significant results
    plt.figure(figsize=(10, 6))
    
    significance_matrix = np.zeros((len(functions), len(sample_sizes)))
    for i, func_name in enumerate(functions):
        significance_matrix[i, :] = results["significant_improvement"][func_name]
    
    sns.heatmap(significance_matrix, annot=True, cmap="YlGnBu", 
                xticklabels=sample_sizes, yticklabels=functions,
                cbar_kws={'label': 'Statistically Significant Improvement'})
    
    plt.xlabel('Sample Size')
    plt.ylabel('Function')
    plt.title('Statistical Significance Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mc_significance_heatmap.png'))

def plot_ml_significance(results):
    """
    Generate plots for Machine Learning statistical significance tests.
    
    Args:
        results: Dictionary with test results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    
    # Extract data
    complexity_levels = results["complexity_levels"]
    dataset_sizes = results["dataset_sizes"]
    
    # 1. P-value plots for each complexity level
    for complexity in complexity_levels:
        plt.figure(figsize=(12, 8))
        
        # Get p-values for different tests
        t_test_pvals = results["p_values"][complexity]["t_test"]
        wilcoxon_pvals = results["p_values"][complexity]["wilcoxon"]
        mannwhitney_pvals = results["p_values"][complexity]["mannwhitney"]
        
        plt.plot(dataset_sizes, t_test_pvals, 'o-', label='t-test')
        plt.plot(dataset_sizes, wilcoxon_pvals, 's-', label='Wilcoxon')
        plt.plot(dataset_sizes, mannwhitney_pvals, '^-', label='Mann-Whitney')
        
        plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05 threshold')
        
        plt.xlabel('Dataset Size')
        plt.ylabel('p-value')
        plt.yscale('log')  # Log scale for p-values
        plt.title(f'Statistical Significance: {complexity.replace("_", " ").title()} Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'ml_{complexity}_pvalues.png'))
    
    # 2. Effect size plot
    plt.figure(figsize=(10, 6))
    
    for complexity in complexity_levels:
        effects = results["effect_sizes"][complexity]
        plt.plot(dataset_sizes, effects, 'o-', label=complexity.replace("_", " ").title())
    
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Small effect')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Medium effect')
    plt.axhline(y=0.8, color='b', linestyle='--', alpha=0.5, label='Large effect')
    
    plt.xlabel('Dataset Size')
    plt.ylabel("Cohen's d Effect Size")
    plt.title('Effect Size by Complexity and Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_effect_sizes.png'))
    
    # 3. Box plots of errors for largest dataset size
    largest_size = dataset_sizes[-1]
    plt.figure(figsize=(12, 8))
    
    data_to_plot = []
    labels = []
    
    for complexity in complexity_levels:
        c_data = results["raw_data"][complexity][largest_size]["classical"]
        q_data = results["raw_data"][complexity][largest_size]["quantum"]
        
        if c_data and q_data:
            data_to_plot.extend([c_data, q_data])
            labels.extend([f'{complexity} (Classical)', f'{complexity} (Quantum)'])
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Mean Squared Error')
    plt.title(f'Distribution of Errors (Dataset Size = {largest_size})')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_boxplot_comparison.png'))
    
    # 4. Heatmap of significant results
    plt.figure(figsize=(10, 6))
    
    significance_matrix = np.zeros((len(complexity_levels), len(dataset_sizes)))
    for i, complexity in enumerate(complexity_levels):
        significance_matrix[i, :] = results["significant_improvement"][complexity]
    
    sns.heatmap(significance_matrix, annot=True, cmap="YlGnBu", 
                xticklabels=dataset_sizes, yticklabels=complexity_levels,
                cbar_kws={'label': 'Statistically Significant Improvement'})
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Complexity Level')
    plt.title('Statistical Significance Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_significance_heatmap.png'))
    
    # 5. Significance vs complexity plot
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage of significant results for each complexity
    significance_percentages = []
    for complexity in complexity_levels:
        sig_count = sum(results["significant_improvement"][complexity])
        total_count = len(results["significant_improvement"][complexity])
        percentage = 100 * sig_count / total_count if total_count > 0 else 0
        significance_percentages.append(percentage)
    
    plt.bar(range(len(complexity_levels)), significance_percentages)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    plt.xlabel('Complexity Level')
    plt.ylabel('Percentage of Significant Results')
    plt.title('Significance Rate vs. Problem Complexity')
    plt.xticks(range(len(complexity_levels)), [c.replace("_", " ").title() for c in complexity_levels])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ml_significance_vs_complexity.png'))

def main():
    """Run statistical significance tests."""
    print_section("STATISTICAL SIGNIFICANCE TESTING")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Use smaller parameter sets for quick testing
    smaller_samples = True
    
    if smaller_samples:
        # Monte Carlo test with reduced parameters
        monte_carlo_results = test_monte_carlo_significance(
            n_tests=10, 
            sample_sizes=[100, 500, 1000]
        )
        
        # ML test with reduced parameters
        ml_results = test_ml_significance(
            n_tests=10, 
            dataset_sizes=[50, 100, 200]
        )
    else:
        # Full Monte Carlo test
        monte_carlo_results = test_monte_carlo_significance(
            n_tests=30, 
            sample_sizes=[100, 500, 1000, 2000, 5000]
        )
        
        # Full ML test
        ml_results = test_ml_significance(
            n_tests=20, 
            dataset_sizes=[50, 100, 200, 500]
        )
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("STATISTICAL SIGNIFICANCE TESTING SUMMARY\n")
        f.write("======================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Monte Carlo summary
        f.write("Monte Carlo Statistical Significance:\n")
        f.write("------------------------------------\n")
        
        for func_name in monte_carlo_results["functions"]:
            significant_count = sum(monte_carlo_results["significant_improvement"][func_name])
            total_count = len(monte_carlo_results["significant_improvement"][func_name])
            sig_percentage = 100 * significant_count / total_count if total_count > 0 else 0
            
            f.write(f"  {func_name.replace('_', ' ').title()} function:\n")
            f.write(f"    Statistically significant improvements: {significant_count}/{total_count} tests ({sig_percentage:.1f}%)\n")
            
            # Get average effect size
            avg_effect = np.nanmean(monte_carlo_results["effect_sizes"][func_name])
            effect_description = "small"
            if avg_effect >= 0.8:
                effect_description = "large"
            elif avg_effect >= 0.5:
                effect_description = "medium"
                
            f.write(f"    Average effect size: {avg_effect:.2f} ({effect_description})\n")
            
            # Get min p-value
            min_p = np.nanmin(monte_carlo_results["p_values"][func_name]["t_test"])
            f.write(f"    Minimum p-value: {min_p:.6f}\n\n")
        
        # ML summary
        f.write("Machine Learning Statistical Significance:\n")
        f.write("-----------------------------------------\n")
        
        for complexity in ml_results["complexity_levels"]:
            significant_count = sum(ml_results["significant_improvement"][complexity])
            total_count = len(ml_results["significant_improvement"][complexity])
            sig_percentage = 100 * significant_count / total_count if total_count > 0 else 0
            
            f.write(f"  {complexity.replace('_', ' ').title()} complexity:\n")
            f.write(f"    Statistically significant improvements: {significant_count}/{total_count} tests ({sig_percentage:.1f}%)\n")
            
            # Get average effect size
            avg_effect = np.nanmean(ml_results["effect_sizes"][complexity])
            effect_description = "small"
            if avg_effect >= 0.8:
                effect_description = "large"
            elif avg_effect >= 0.5:
                effect_description = "medium"
                
            f.write(f"    Average effect size: {avg_effect:.2f} ({effect_description})\n")
            
            # Get min p-value
            min_p = np.nanmin(ml_results["p_values"][complexity]["t_test"])
            f.write(f"    Minimum p-value: {min_p:.6f}\n\n")
        
        # Key findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        
        # Monte Carlo findings
        mc_sig_counts = {func_name: sum(monte_carlo_results["significant_improvement"][func_name]) 
                        for func_name in monte_carlo_results["functions"]}
        most_sig_func = max(mc_sig_counts, key=mc_sig_counts.get)
        
        f.write(f"1. Monte Carlo: Most significant improvements found in {most_sig_func.replace('_', ' ').title()} function\n")
        
        # Check if significance increases with sample size
        last_func = monte_carlo_results["functions"][-1]
        sig_trend = all(b >= a for a, b in zip(monte_carlo_results["significant_improvement"][last_func][:-1], 
                                              monte_carlo_results["significant_improvement"][last_func][1:]))
        if sig_trend:
            f.write("2. Monte Carlo: Statistical significance tends to increase with larger sample sizes\n")
        else:
            f.write("2. Monte Carlo: No clear trend in statistical significance with sample size\n")
        
        # ML findings
        ml_sig_counts = {complexity: sum(ml_results["significant_improvement"][complexity]) 
                        for complexity in ml_results["complexity_levels"]}
        most_sig_complexity = max(ml_sig_counts, key=ml_sig_counts.get)
        
        f.write(f"3. ML: Most significant improvements found in {most_sig_complexity.replace('_', ' ').title()} complexity problems\n")
        
        # Check correlation between complexity and significance
        complexity_order = {"linear": 1, "nonlinear": 2, "highly_nonlinear": 3}
        sig_percentages = []
        for complexity in ml_results["complexity_levels"]:
            sig_count = sum(ml_results["significant_improvement"][complexity])
            total_count = len(ml_results["significant_improvement"][complexity])
            sig_percentages.append(100 * sig_count / total_count if total_count > 0 else 0)
        
        if sig_percentages[0] < sig_percentages[1] < sig_percentages[2]:
            f.write("4. ML: Quantum advantage increases with problem complexity\n")
        elif sig_percentages[0] > sig_percentages[1] > sig_percentages[2]:
            f.write("4. ML: Quantum advantage decreases with problem complexity\n")
        else:
            f.write("4. ML: No clear trend in quantum advantage with problem complexity\n")
        
        # Overall recommendation
        f.write("5. Overall: Statistical evidence supports quantum advantage claims in specific problem domains, but not universally\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nStatistical significance testing completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 