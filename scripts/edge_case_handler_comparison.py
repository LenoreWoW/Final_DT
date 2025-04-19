#!/usr/bin/env python3
"""
Edge Case Handler Comparison

Tests how classical and quantum methods handle boundary conditions and edge cases,
demonstrating the robustness of each approach under extreme or unusual inputs.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/edge_case_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_extreme_parameter_ranges():
    """
    Test how methods handle extreme parameter ranges.
    
    Returns:
        Dictionary with test results
    """
    print_section("TESTING EXTREME PARAMETER RANGES")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Define test function
    def target_function(x, y):
        return x**2 + y**2 + 0.5*x*y
    
    # Define parameter range scales to test
    range_scales = [
        ("tiny", 1e-6, 1e-5),     # Very small range
        ("small", 0.01, 0.1),     # Small range
        ("normal", 1.0, 10.0),    # Normal range
        ("large", 1e3, 1e4),      # Large range
        ("huge", 1e6, 1e7)        # Very large range
    ]
    
    # Number of iterations
    iterations = 1000
    
    # Store results
    results = {
        "range_scales": [s[0] for s in range_scales],
        "classical": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "error_rate": []
        },
        "quantum": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "error_rate": []
        }
    }
    
    # Test each range scale
    for scale_name, min_val, max_val in range_scales:
        print(f"Testing {scale_name} range: [{min_val}, {max_val}]")
        
        # Define parameter ranges
        param_ranges = {
            'x': (min_val, max_val),
            'y': (min_val, max_val)
        }
        
        # Run classical Monte Carlo
        try:
            start_time = time.time()
            classical_result = qmc.run_classical_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=target_function
            )
            execution_time = time.time() - start_time
            
            results["classical"]["mean"].append(classical_result["mean"])
            results["classical"]["std"].append(classical_result["std"])
            results["classical"]["min"].append(classical_result["min"])
            results["classical"]["execution_time"].append(execution_time)
            results["classical"]["error_rate"].append(0.0)  # No error
            
            print(f"  Classical mean: {classical_result['mean']:.6e}, std: {classical_result['std']:.6e}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in classical method for {scale_name} range: {str(e)}")
            results["classical"]["mean"].append(float('nan'))
            results["classical"]["std"].append(float('nan'))
            results["classical"]["min"].append(float('nan'))
            results["classical"]["execution_time"].append(float('nan'))
            results["classical"]["error_rate"].append(1.0)  # Error occurred
            
            print(f"  Classical method failed: {str(e)}")
        
        # Run quantum Monte Carlo
        try:
            start_time = time.time()
            quantum_result = qmc.run_quantum_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=target_function
            )
            execution_time = time.time() - start_time
            
            results["quantum"]["mean"].append(quantum_result["mean"])
            results["quantum"]["std"].append(quantum_result["std"])
            results["quantum"]["min"].append(quantum_result["min"])
            results["quantum"]["execution_time"].append(execution_time)
            results["quantum"]["error_rate"].append(0.0)  # No error
            
            print(f"  Quantum mean: {quantum_result['mean']:.6e}, std: {quantum_result['std']:.6e}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in quantum method for {scale_name} range: {str(e)}")
            results["quantum"]["mean"].append(float('nan'))
            results["quantum"]["std"].append(float('nan'))
            results["quantum"]["min"].append(float('nan'))
            results["quantum"]["execution_time"].append(float('nan'))
            results["quantum"]["error_rate"].append(1.0)  # Error occurred
            
            print(f"  Quantum method failed: {str(e)}")
    
    # Plot results
    plot_extreme_range_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'extreme_parameter_ranges.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results

def test_ill_conditioned_problems():
    """
    Test how methods handle ill-conditioned optimization problems.
    
    Returns:
        Dictionary with test results
    """
    print_section("TESTING ILL-CONDITIONED PROBLEMS")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Define test functions with varying condition numbers
    test_functions = {
        "well_conditioned": lambda x, y: x**2 + y**2,
        "medium_conditioned": lambda x, y: x**2 + 0.1*y**2,
        "ill_conditioned": lambda x, y: x**2 + 0.01*y**2,
        "very_ill_conditioned": lambda x, y: x**2 + 0.0001*y**2,
        "extremely_ill_conditioned": lambda x, y: x**2 + 1e-6*y**2
    }
    
    # Parameter ranges
    param_ranges = {
        'x': (-10.0, 10.0),
        'y': (-10.0, 10.0)
    }
    
    # Number of iterations
    iterations = 1000
    
    # Store results
    results = {
        "condition_levels": list(test_functions.keys()),
        "classical": {
            "mean": [],
            "std": [],
            "min": [],
            "min_x": [],
            "min_y": [],
            "distance_from_true": [],
            "execution_time": []
        },
        "quantum": {
            "mean": [],
            "std": [],
            "min": [],
            "min_x": [],
            "min_y": [],
            "distance_from_true": [],
            "execution_time": []
        }
    }
    
    # Test each function
    for func_name, func in test_functions.items():
        print(f"Testing {func_name} function")
        
        # Run classical Monte Carlo
        try:
            start_time = time.time()
            classical_result = qmc.run_classical_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=func,
                return_all_samples=True
            )
            execution_time = time.time() - start_time
            
            # Find minimum point
            classical_samples = classical_result.get("samples", [])
            if classical_samples:
                sample_values = [func(x, y) for x, y in classical_samples]
                min_index = np.argmin(sample_values)
                min_x, min_y = classical_samples[min_index]
                min_val = sample_values[min_index]
            else:
                min_x, min_y = float('nan'), float('nan')
                min_val = classical_result["min"]
            
            # True minimum is at (0, 0)
            distance_from_true = np.sqrt(min_x**2 + min_y**2)
            
            results["classical"]["mean"].append(classical_result["mean"])
            results["classical"]["std"].append(classical_result["std"])
            results["classical"]["min"].append(min_val)
            results["classical"]["min_x"].append(min_x)
            results["classical"]["min_y"].append(min_y)
            results["classical"]["distance_from_true"].append(distance_from_true)
            results["classical"]["execution_time"].append(execution_time)
            
            print(f"  Classical minimum: {min_val:.6f} at ({min_x:.6f}, {min_y:.6f})")
            print(f"  Distance from true minimum: {distance_from_true:.6f}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in classical method for {func_name}: {str(e)}")
            results["classical"]["mean"].append(float('nan'))
            results["classical"]["std"].append(float('nan'))
            results["classical"]["min"].append(float('nan'))
            results["classical"]["min_x"].append(float('nan'))
            results["classical"]["min_y"].append(float('nan'))
            results["classical"]["distance_from_true"].append(float('nan'))
            results["classical"]["execution_time"].append(float('nan'))
            
            print(f"  Classical method failed: {str(e)}")
        
        # Run quantum Monte Carlo
        try:
            start_time = time.time()
            quantum_result = qmc.run_quantum_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=func,
                return_all_samples=True
            )
            execution_time = time.time() - start_time
            
            # Find minimum point
            quantum_samples = quantum_result.get("samples", [])
            if quantum_samples:
                sample_values = [func(x, y) for x, y in quantum_samples]
                min_index = np.argmin(sample_values)
                min_x, min_y = quantum_samples[min_index]
                min_val = sample_values[min_index]
            else:
                min_x, min_y = float('nan'), float('nan')
                min_val = quantum_result["min"]
            
            # True minimum is at (0, 0)
            distance_from_true = np.sqrt(min_x**2 + min_y**2)
            
            results["quantum"]["mean"].append(quantum_result["mean"])
            results["quantum"]["std"].append(quantum_result["std"])
            results["quantum"]["min"].append(min_val)
            results["quantum"]["min_x"].append(min_x)
            results["quantum"]["min_y"].append(min_y)
            results["quantum"]["distance_from_true"].append(distance_from_true)
            results["quantum"]["execution_time"].append(execution_time)
            
            print(f"  Quantum minimum: {min_val:.6f} at ({min_x:.6f}, {min_y:.6f})")
            print(f"  Distance from true minimum: {distance_from_true:.6f}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in quantum method for {func_name}: {str(e)}")
            results["quantum"]["mean"].append(float('nan'))
            results["quantum"]["std"].append(float('nan'))
            results["quantum"]["min"].append(float('nan'))
            results["quantum"]["min_x"].append(float('nan'))
            results["quantum"]["min_y"].append(float('nan'))
            results["quantum"]["distance_from_true"].append(float('nan'))
            results["quantum"]["execution_time"].append(float('nan'))
            
            print(f"  Quantum method failed: {str(e)}")
    
    # Plot results
    plot_ill_conditioned_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'ill_conditioned_problems.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results 

def test_discontinuous_functions():
    """
    Test how methods handle functions with discontinuities.
    
    Returns:
        Dictionary with test results
    """
    print_section("TESTING DISCONTINUOUS FUNCTIONS")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Define test functions with discontinuities
    test_functions = {
        "continuous": lambda x, y: x**2 + y**2,
        "step": lambda x, y: x**2 + y**2 + (1.0 if x > 0 else 0.0),
        "barrier": lambda x, y: x**2 + y**2 + (1e6 if -0.1 < x < 0.1 else 0.0),
        "divide_by_zero": lambda x, y: x**2 + y**2 + (0.1 / (abs(x - 0.5) + 1e-10)),
        "oscillatory": lambda x, y: x**2 + y**2 + np.sin(1.0 / (abs(x) + 1e-10))
    }
    
    # Parameter ranges
    param_ranges = {
        'x': (-1.0, 1.0),
        'y': (-1.0, 1.0)
    }
    
    # Number of iterations
    iterations = 1000
    
    # Store results
    results = {
        "function_types": list(test_functions.keys()),
        "classical": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "error_rate": []
        },
        "quantum": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "error_rate": []
        }
    }
    
    # Test each function
    for func_name, func in test_functions.items():
        print(f"Testing {func_name} function")
        
        # Wrap the function to catch any errors
        def safe_func(x, y):
            try:
                return func(x, y)
            except:
                return np.nan
        
        # Run classical Monte Carlo
        classical_errors = 0
        try:
            start_time = time.time()
            classical_result = qmc.run_classical_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=safe_func
            )
            execution_time = time.time() - start_time
            
            # Count NaN values as errors if they exist in result
            if "nan_count" in classical_result:
                classical_errors = classical_result["nan_count"]
            
            results["classical"]["mean"].append(classical_result["mean"])
            results["classical"]["std"].append(classical_result["std"])
            results["classical"]["min"].append(classical_result["min"])
            results["classical"]["execution_time"].append(execution_time)
            results["classical"]["error_rate"].append(classical_errors / iterations if iterations > 0 else 0.0)
            
            print(f"  Classical mean: {classical_result['mean']:.6f}, std: {classical_result['std']:.6f}")
            print(f"  Error rate: {classical_errors / iterations * 100:.2f}%")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in classical method for {func_name}: {str(e)}")
            results["classical"]["mean"].append(float('nan'))
            results["classical"]["std"].append(float('nan'))
            results["classical"]["min"].append(float('nan'))
            results["classical"]["execution_time"].append(float('nan'))
            results["classical"]["error_rate"].append(1.0)  # Complete failure
            
            print(f"  Classical method failed: {str(e)}")
        
        # Run quantum Monte Carlo
        quantum_errors = 0
        try:
            start_time = time.time()
            quantum_result = qmc.run_quantum_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=safe_func
            )
            execution_time = time.time() - start_time
            
            # Count NaN values as errors if they exist in result
            if "nan_count" in quantum_result:
                quantum_errors = quantum_result["nan_count"]
            
            results["quantum"]["mean"].append(quantum_result["mean"])
            results["quantum"]["std"].append(quantum_result["std"])
            results["quantum"]["min"].append(quantum_result["min"])
            results["quantum"]["execution_time"].append(execution_time)
            results["quantum"]["error_rate"].append(quantum_errors / iterations if iterations > 0 else 0.0)
            
            print(f"  Quantum mean: {quantum_result['mean']:.6f}, std: {quantum_result['std']:.6f}")
            print(f"  Error rate: {quantum_errors / iterations * 100:.2f}%")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in quantum method for {func_name}: {str(e)}")
            results["quantum"]["mean"].append(float('nan'))
            results["quantum"]["std"].append(float('nan'))
            results["quantum"]["min"].append(float('nan'))
            results["quantum"]["execution_time"].append(float('nan'))
            results["quantum"]["error_rate"].append(1.0)  # Complete failure
            
            print(f"  Quantum method failed: {str(e)}")
    
    # Plot results
    plot_discontinuous_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'discontinuous_functions.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results

def test_high_dimensionality():
    """
    Test how methods handle high-dimensional optimization problems.
    
    Returns:
        Dictionary with test results
    """
    print_section("TESTING HIGH DIMENSIONALITY")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Dimensions to test
    dimensions = [2, 3, 5, 8, 10, 15, 20]
    
    # Function factory for n-dimensional problems
    def create_func(dim):
        # Simple quadratic function in n dimensions
        def func(*args):
            result = 0
            for i, x in enumerate(args):
                # Add quadratic term
                result += x**2
                
                # Add some cross-terms to create dependencies
                if i < dim - 1:
                    result += 0.1 * x * args[i+1]
            return result
        return func
    
    # Number of iterations
    iterations = 1000
    
    # Store results
    results = {
        "dimensions": dimensions,
        "classical": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "success": []
        },
        "quantum": {
            "mean": [],
            "std": [],
            "min": [],
            "execution_time": [],
            "success": []
        }
    }
    
    # Test each dimension
    for dim in dimensions:
        print(f"Testing {dim}-dimensional problem")
        
        # Create function for this dimension
        func = create_func(dim)
        
        # Create parameter ranges
        param_ranges = {}
        for i in range(dim):
            param_ranges[f'x{i}'] = (-1.0, 1.0)
        
        # Run classical Monte Carlo
        try:
            start_time = time.time()
            classical_result = qmc.run_classical_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=func
            )
            execution_time = time.time() - start_time
            
            results["classical"]["mean"].append(classical_result["mean"])
            results["classical"]["std"].append(classical_result["std"])
            results["classical"]["min"].append(classical_result["min"])
            results["classical"]["execution_time"].append(execution_time)
            results["classical"]["success"].append(1)  # Success
            
            print(f"  Classical mean: {classical_result['mean']:.6f}, std: {classical_result['std']:.6f}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in classical method for {dim} dimensions: {str(e)}")
            results["classical"]["mean"].append(float('nan'))
            results["classical"]["std"].append(float('nan'))
            results["classical"]["min"].append(float('nan'))
            results["classical"]["execution_time"].append(float('nan'))
            results["classical"]["success"].append(0)  # Failure
            
            print(f"  Classical method failed: {str(e)}")
        
        # Run quantum Monte Carlo
        try:
            start_time = time.time()
            quantum_result = qmc.run_quantum_monte_carlo(
                param_ranges, 
                iterations=iterations,
                target_function=func
            )
            execution_time = time.time() - start_time
            
            results["quantum"]["mean"].append(quantum_result["mean"])
            results["quantum"]["std"].append(quantum_result["std"])
            results["quantum"]["min"].append(quantum_result["min"])
            results["quantum"]["execution_time"].append(execution_time)
            results["quantum"]["success"].append(1)  # Success
            
            print(f"  Quantum mean: {quantum_result['mean']:.6f}, std: {quantum_result['std']:.6f}")
            print(f"  Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in quantum method for {dim} dimensions: {str(e)}")
            results["quantum"]["mean"].append(float('nan'))
            results["quantum"]["std"].append(float('nan'))
            results["quantum"]["min"].append(float('nan'))
            results["quantum"]["execution_time"].append(float('nan'))
            results["quantum"]["success"].append(0)  # Failure
            
            print(f"  Quantum method failed: {str(e)}")
    
    # Plot results
    plot_high_dimension_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'high_dimensionality.json'), 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        results_serializable = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.float32) or isinstance(o, np.float64) else o))
        json.dump(results_serializable, f, indent=2)
    
    return results 

def plot_discontinuous_results(results):
    """
    Plot results from discontinuous function tests.
    
    Args:
        results: Dictionary with test results
    """
    plt.figure(figsize=(15, 10))
    
    # Error rate comparison
    plt.subplot(2, 2, 1)
    x = np.arange(len(results["function_types"]))
    width = 0.35
    
    plt.bar(x - width/2, results["classical"]["error_rate"], width, label='Classical')
    plt.bar(x + width/2, results["quantum"]["error_rate"], width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Error Rate')
    plt.title('Error Rate Comparison by Function Type')
    plt.xticks(x, results["function_types"], rotation=45)
    plt.legend()
    
    # Execution time comparison
    plt.subplot(2, 2, 2)
    
    plt.bar(x - width/2, results["classical"]["execution_time"], width, label='Classical')
    plt.bar(x + width/2, results["quantum"]["execution_time"], width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison by Function Type')
    plt.xticks(x, results["function_types"], rotation=45)
    plt.legend()
    
    # Mean value comparison
    plt.subplot(2, 2, 3)
    
    # Check for NaN values and replace with zeros for plotting
    classical_means = [val if not np.isnan(val) else 0 for val in results["classical"]["mean"]]
    quantum_means = [val if not np.isnan(val) else 0 for val in results["quantum"]["mean"]]
    
    plt.bar(x - width/2, classical_means, width, label='Classical')
    plt.bar(x + width/2, quantum_means, width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Mean Value')
    plt.title('Mean Value Comparison by Function Type')
    plt.xticks(x, results["function_types"], rotation=45)
    plt.legend()
    
    # Min value comparison
    plt.subplot(2, 2, 4)
    
    # Check for NaN values and replace with zeros for plotting
    classical_mins = [val if not np.isnan(val) else 0 for val in results["classical"]["min"]]
    quantum_mins = [val if not np.isnan(val) else 0 for val in results["quantum"]["min"]]
    
    plt.bar(x - width/2, classical_mins, width, label='Classical')
    plt.bar(x + width/2, quantum_mins, width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Minimum Value')
    plt.title('Minimum Value Comparison by Function Type')
    plt.xticks(x, results["function_types"], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'discontinuous_functions.png'))
    plt.close()

def plot_high_dimension_results(results):
    """
    Plot results from high-dimensional optimization tests.
    
    Args:
        results: Dictionary with test results
    """
    plt.figure(figsize=(15, 10))
    
    # Filter out NaN values for plotting
    valid_indices = []
    dimensions = []
    for i, dim in enumerate(results["dimensions"]):
        if (not np.isnan(results["classical"]["execution_time"][i]) and 
            not np.isnan(results["quantum"]["execution_time"][i])):
            valid_indices.append(i)
            dimensions.append(dim)
    
    # Execution time comparison
    plt.subplot(2, 2, 1)
    
    classical_times = [results["classical"]["execution_time"][i] for i in valid_indices]
    quantum_times = [results["quantum"]["execution_time"][i] for i in valid_indices]
    
    plt.plot(dimensions, classical_times, 'o-', label='Classical')
    plt.plot(dimensions, quantum_times, 'o-', label='Quantum')
    
    plt.xlabel('Dimensionality')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Dimensionality')
    plt.legend()
    plt.grid(True)
    
    # Execution time ratio
    plt.subplot(2, 2, 2)
    
    time_ratio = [q/c if c > 0 else 0 for q, c in zip(quantum_times, classical_times)]
    
    plt.plot(dimensions, time_ratio, 'o-')
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    plt.xlabel('Dimensionality')
    plt.ylabel('Time Ratio (Quantum/Classical)')
    plt.title('Quantum/Classical Time Ratio vs Dimensionality')
    plt.grid(True)
    
    # Mean comparison
    plt.subplot(2, 2, 3)
    
    classical_means = [results["classical"]["mean"][i] for i in valid_indices]
    quantum_means = [results["quantum"]["mean"][i] for i in valid_indices]
    
    plt.plot(dimensions, classical_means, 'o-', label='Classical')
    plt.plot(dimensions, quantum_means, 'o-', label='Quantum')
    
    plt.xlabel('Dimensionality')
    plt.ylabel('Mean Value')
    plt.title('Mean Value vs Dimensionality')
    plt.legend()
    plt.grid(True)
    
    # Success rate
    plt.subplot(2, 2, 4)
    
    classical_success = [results["classical"]["success"][i] for i in range(len(results["dimensions"]))]
    quantum_success = [results["quantum"]["success"][i] for i in range(len(results["dimensions"]))]
    
    x = np.arange(len(results["dimensions"]))
    width = 0.35
    
    plt.bar(x - width/2, classical_success, width, label='Classical')
    plt.bar(x + width/2, quantum_success, width, label='Quantum')
    
    plt.xlabel('Dimensionality')
    plt.ylabel('Success (1 = Success, 0 = Failure)')
    plt.title('Success Rate by Dimensionality')
    plt.xticks(x, results["dimensions"])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'high_dimensionality.png'))
    plt.close()

def generate_summary_report(discontinuous_results, high_dim_results):
    """
    Generate a summary report of all test findings.
    
    Args:
        discontinuous_results: Results from discontinuous function tests
        high_dim_results: Results from high dimensionality tests
    
    Returns:
        str: Summary report text
    """
    report = []
    report.append("EDGE CASE HANDLER COMPARISON SUMMARY REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Discontinuous Functions Summary
    report.append("1. DISCONTINUOUS FUNCTIONS ANALYSIS")
    report.append("-" * 40)
    
    # Overall performance comparison
    classical_avg_error = np.mean(discontinuous_results["classical"]["error_rate"])
    quantum_avg_error = np.mean(discontinuous_results["quantum"]["error_rate"])
    
    report.append(f"Average Error Rate: Classical = {classical_avg_error:.4f}, Quantum = {quantum_avg_error:.4f}")
    
    if quantum_avg_error < classical_avg_error:
        improvement = ((classical_avg_error - quantum_avg_error) / classical_avg_error) * 100
        report.append(f"Quantum shows {improvement:.2f}% lower error rate than classical methods")
    else:
        report.append("Classical methods show lower error rates for discontinuous functions")
    
    # Function-specific findings
    report.append("\nFunction-specific findings:")
    for i, func_type in enumerate(discontinuous_results["function_types"]):
        c_error = discontinuous_results["classical"]["error_rate"][i]
        q_error = discontinuous_results["quantum"]["error_rate"][i]
        c_time = discontinuous_results["classical"]["execution_time"][i]
        q_time = discontinuous_results["quantum"]["execution_time"][i]
        
        report.append(f"  - {func_type}:")
        report.append(f"    Error rates: Classical = {c_error:.4f}, Quantum = {q_error:.4f}")
        report.append(f"    Execution time: Classical = {c_time:.4f}s, Quantum = {q_time:.4f}s")
        
        if q_error < c_error:
            report.append(f"    Quantum performs better with {((c_error - q_error) / c_error) * 100:.2f}% lower error")
        else:
            report.append(f"    Classical performs better for this function type")
    
    report.append("")
    
    # High Dimensionality Summary
    report.append("2. HIGH DIMENSIONALITY ANALYSIS")
    report.append("-" * 40)
    
    # Calculate where quantum begins to outperform classical
    crossover_dim = None
    for i, dim in enumerate(high_dim_results["dimensions"]):
        if i > 0:
            c_time_prev = high_dim_results["classical"]["execution_time"][i-1]
            q_time_prev = high_dim_results["quantum"]["execution_time"][i-1]
            c_time = high_dim_results["classical"]["execution_time"][i]
            q_time = high_dim_results["quantum"]["execution_time"][i]
            
            if (c_time_prev < q_time_prev) and (c_time > q_time):
                crossover_dim = dim
                break
    
    if crossover_dim:
        report.append(f"Quantum begins to outperform classical at dimensionality: {crossover_dim}")
    else:
        report.append("No clear dimensionality crossover point observed in the tests")
    
    # Success rate analysis
    c_success = np.mean(high_dim_results["classical"]["success"])
    q_success = np.mean(high_dim_results["quantum"]["success"])
    
    report.append(f"\nOverall success rate: Classical = {c_success:.2f}, Quantum = {q_success:.2f}")
    
    # Performance by dimension
    report.append("\nPerformance by dimension:")
    for i, dim in enumerate(high_dim_results["dimensions"]):
        c_time = high_dim_results["classical"]["execution_time"][i]
        q_time = high_dim_results["quantum"]["execution_time"][i]
        c_succ = high_dim_results["classical"]["success"][i]
        q_succ = high_dim_results["quantum"]["success"][i]
        
        report.append(f"  - {dim} dimensions:")
        report.append(f"    Execution time: Classical = {c_time:.4f}s, Quantum = {q_time:.4f}s")
        report.append(f"    Success: Classical = {c_succ}, Quantum = {q_succ}")
        
        if q_time < c_time:
            speedup = (c_time / q_time) if q_time > 0 else float('inf')
            report.append(f"    Quantum is {speedup:.2f}x faster")
        else:
            slowdown = (q_time / c_time) if c_time > 0 else float('inf')
            report.append(f"    Classical is {slowdown:.2f}x faster")
    
    report.append("")
    report.append("3. OVERALL CONCLUSIONS")
    report.append("-" * 40)
    
    # Generate overall conclusions based on the results
    if quantum_avg_error < classical_avg_error and q_success > c_success:
        report.append("Quantum methods demonstrate superior performance overall, with lower error rates")
        report.append("for discontinuous functions and better success rates for high-dimensional problems.")
    elif quantum_avg_error < classical_avg_error:
        report.append("Quantum methods show advantages for discontinuous functions but less consistent")
        report.append("benefits for high-dimensional problems.")
    elif q_success > c_success:
        report.append("Quantum methods demonstrate superior performance for high-dimensional problems")
        report.append("but show less advantage for discontinuous functions.")
    else:
        report.append("Classical methods generally outperform quantum methods in the tested scenarios,")
        report.append("though specific use cases may still benefit from quantum approaches.")
    
    report.append("")
    report.append("=" * 50)
    
    return "\n".join(report)

def main():
    """
    Main function to run all edge case tests and generate reports.
    """
    # Create output directory structure
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
    
    # Run tests
    logger.info("Testing discontinuous functions")
    discontinuous_results = test_discontinuous_functions()
    
    logger.info("Testing high-dimensionality problems")
    high_dim_results = test_high_dimensionality()
    
    # Generate plots
    logger.info("Generating plots")
    plot_discontinuous_results(discontinuous_results)
    plot_high_dimension_results(high_dim_results)
    
    # Generate summary report
    logger.info("Generating summary report")
    summary_report = generate_summary_report(discontinuous_results, high_dim_results)
    
    with open(os.path.join(OUTPUT_DIR, 'summary_report.txt'), 'w') as f:
        f.write(summary_report)
    
    logger.info(f"Summary report saved to {os.path.join(OUTPUT_DIR, 'summary_report.txt')}")
    logger.info("Edge case comparison completed")

if __name__ == "__main__":
    main() 