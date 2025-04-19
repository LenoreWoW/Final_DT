#!/usr/bin/env python3
"""
Quantum Monte Carlo Convergence

Compares convergence rates of classical vs. quantum Monte Carlo methods
across different problem types and parameter spaces, demonstrating where
quantum approaches offer speed advantages.
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
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/monte_carlo_convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_convergence_rate(target_functions, iterations_list, repeats=3):
    """
    Test convergence rate of quantum vs classical Monte Carlo.
    
    Args:
        target_functions: Dictionary of test functions
        iterations_list: List of iteration counts to test
        repeats: Number of repeats for each test
        
    Returns:
        Dictionary with convergence results
    """
    print_section("MONTE CARLO CONVERGENCE TEST")
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    if not qmc.is_available():
        logger.warning("Quantum processing not available. Running in simulation mode.")
    
    # Parameter ranges - same for all functions for consistency
    param_ranges = {
        'x': (-1.0, 1.0),
        'y': (-1.0, 1.0),
        'z': (-1.0, 1.0)
    }
    
    # Known analytical solutions for each function
    # For our test functions, we use simple functions with known exact values
    analytical_solutions = {
        "quadratic": 0.0,  # Minimum at (0,0,0)
        "trigonometric": -3.0,  # Sum of -sin(0)=-0, -sin(0)=-0, -sin(0)=-0
        "mixed": -1.5,  # x^2 + y^2 + z^2 - sin(x) - sin(y) - sin(z) at (0,0,0)
        "exponential": -3.0  # -e^(-(0^2)) - e^(-(0^2)) - e^(-(0^2))
    }
    
    # Store results
    results = {
        "functions": list(target_functions.keys()),
        "iterations": iterations_list,
        "quantum": {
            func_name: {
                "mean": [[] for _ in iterations_list],
                "std": [[] for _ in iterations_list],
                "error": [[] for _ in iterations_list],
                "time": [[] for _ in iterations_list]
            } for func_name in target_functions
        },
        "classical": {
            func_name: {
                "mean": [[] for _ in iterations_list],
                "std": [[] for _ in iterations_list], 
                "error": [[] for _ in iterations_list],
                "time": [[] for _ in iterations_list]
            } for func_name in target_functions
        }
    }
    
    # Run tests for each function type
    for func_name, func in target_functions.items():
        print(f"\nTesting {func_name} function:")
        true_value = analytical_solutions[func_name]
        
        # Test with different iteration counts
        for i, iterations in enumerate(iterations_list):
            print(f"  Iterations: {iterations}")
            
            # Repeat tests for statistical significance
            for r in range(repeats):
                print(f"    Repeat {r+1}/{repeats}")
                
                # Run classical Monte Carlo
                try:
                    start_time = time.time()
                    classical_result = qmc.run_classical_monte_carlo(
                        param_ranges,
                        iterations=iterations,
                        target_function=func
                    )
                    classical_time = time.time() - start_time
                    
                    classical_mean = classical_result["mean"]
                    classical_std = classical_result["std"]
                    classical_error = abs(classical_mean - true_value)
                    
                    results["classical"][func_name]["mean"][i].append(classical_mean)
                    results["classical"][func_name]["std"][i].append(classical_std)
                    results["classical"][func_name]["error"][i].append(classical_error)
                    results["classical"][func_name]["time"][i].append(classical_time)
                    
                    print(f"      Classical: Mean={classical_mean:.6f}, Error={classical_error:.6f}, Time={classical_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error in classical MC for {func_name}, iterations={iterations}, repeat={r}: {str(e)}")
                    results["classical"][func_name]["mean"][i].append(float('nan'))
                    results["classical"][func_name]["std"][i].append(float('nan'))
                    results["classical"][func_name]["error"][i].append(float('nan'))
                    results["classical"][func_name]["time"][i].append(float('nan'))
                
                # Run quantum Monte Carlo
                try:
                    start_time = time.time()
                    quantum_result = qmc.run_quantum_monte_carlo(
                        param_ranges,
                        iterations=iterations,
                        target_function=func
                    )
                    quantum_time = time.time() - start_time
                    
                    quantum_mean = quantum_result["mean"]
                    quantum_std = quantum_result["std"]
                    quantum_error = abs(quantum_mean - true_value)
                    
                    results["quantum"][func_name]["mean"][i].append(quantum_mean)
                    results["quantum"][func_name]["std"][i].append(quantum_std)
                    results["quantum"][func_name]["error"][i].append(quantum_error)
                    results["quantum"][func_name]["time"][i].append(quantum_time)
                    
                    print(f"      Quantum: Mean={quantum_mean:.6f}, Error={quantum_error:.6f}, Time={quantum_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error in quantum MC for {func_name}, iterations={iterations}, repeat={r}: {str(e)}")
                    results["quantum"][func_name]["mean"][i].append(float('nan'))
                    results["quantum"][func_name]["std"][i].append(float('nan'))
                    results["quantum"][func_name]["error"][i].append(float('nan'))
                    results["quantum"][func_name]["time"][i].append(float('nan'))
    
    # Process results - average over repeats
    for func_name in target_functions:
        for i in range(len(iterations_list)):
            # Classical
            valid_indices = [j for j, val in enumerate(results["classical"][func_name]["error"][i]) if not np.isnan(val)]
            if valid_indices:
                results["classical"][func_name]["mean"][i] = np.mean([results["classical"][func_name]["mean"][i][j] for j in valid_indices])
                results["classical"][func_name]["std"][i] = np.mean([results["classical"][func_name]["std"][i][j] for j in valid_indices])
                results["classical"][func_name]["error"][i] = np.mean([results["classical"][func_name]["error"][i][j] for j in valid_indices])
                results["classical"][func_name]["time"][i] = np.mean([results["classical"][func_name]["time"][i][j] for j in valid_indices])
            else:
                results["classical"][func_name]["mean"][i] = float('nan')
                results["classical"][func_name]["std"][i] = float('nan')
                results["classical"][func_name]["error"][i] = float('nan')
                results["classical"][func_name]["time"][i] = float('nan')
            
            # Quantum
            valid_indices = [j for j, val in enumerate(results["quantum"][func_name]["error"][i]) if not np.isnan(val)]
            if valid_indices:
                results["quantum"][func_name]["mean"][i] = np.mean([results["quantum"][func_name]["mean"][i][j] for j in valid_indices])
                results["quantum"][func_name]["std"][i] = np.mean([results["quantum"][func_name]["std"][i][j] for j in valid_indices])
                results["quantum"][func_name]["error"][i] = np.mean([results["quantum"][func_name]["error"][i][j] for j in valid_indices])
                results["quantum"][func_name]["time"][i] = np.mean([results["quantum"][func_name]["time"][i][j] for j in valid_indices])
            else:
                results["quantum"][func_name]["mean"][i] = float('nan')
                results["quantum"][func_name]["std"][i] = float('nan')
                results["quantum"][func_name]["error"][i] = float('nan')
                results["quantum"][func_name]["time"][i] = float('nan')
    
    # Generate plots
    plot_convergence_results(results, analytical_solutions)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'convergence_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_convergence_results(results, analytical_solutions):
    """
    Generate plots for convergence test results.
    
    Args:
        results: Dictionary with convergence results
        analytical_solutions: Dictionary with true values for each function
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    iterations = results["iterations"]
    functions = results["functions"]
    
    # Error convergence plot for each function
    for func_name in functions:
        plt.figure(figsize=(10, 6))
        
        classical_errors = results["classical"][func_name]["error"]
        quantum_errors = results["quantum"][func_name]["error"]
        
        plt.loglog(iterations, classical_errors, 'o-', label='Classical')
        plt.loglog(iterations, quantum_errors, 's-', label='Quantum')
        
        # Add reference lines for common convergence rates
        n_ref = np.array(iterations)
        
        # 1/sqrt(n) - typical Monte Carlo
        plt.loglog(n_ref, n_ref[0]**0.5 * classical_errors[0] / n_ref**0.5, 'k--', alpha=0.5, label='O(1/√n)')
        
        # 1/n - faster convergence (if quantum shows this)
        plt.loglog(n_ref, n_ref[0] * quantum_errors[0] / n_ref, 'r--', alpha=0.5, label='O(1/n)')
        
        plt.xlabel('Number of Iterations')
        plt.ylabel('Absolute Error')
        plt.title(f'Error Convergence: {func_name.capitalize()} Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{func_name}_error_convergence.png'))
    
    # Combined error convergence plot
    plt.figure(figsize=(12, 8))
    
    for i, func_name in enumerate(functions):
        classical_errors = results["classical"][func_name]["error"]
        quantum_errors = results["quantum"][func_name]["error"]
        
        plt.subplot(2, 2, i+1)
        plt.loglog(iterations, classical_errors, 'o-', label='Classical')
        plt.loglog(iterations, quantum_errors, 's-', label='Quantum')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Absolute Error')
        plt.title(f'{func_name.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_error_convergence.png'))
    
    # Convergence rate comparison
    plt.figure(figsize=(10, 6))
    
    convergence_rates = []
    for func_name in functions:
        # Calculate empirical convergence rates
        # For classical: error ~ n^(-a)
        # For quantum: error ~ n^(-b)
        # Log-log regression to find a and b
        try:
            # Classical
            log_iterations = np.log(iterations)
            log_classical_errors = np.log(results["classical"][func_name]["error"])
            
            valid_indices = ~np.isnan(log_classical_errors)
            if sum(valid_indices) >= 2:
                classical_fit = np.polyfit(log_iterations[valid_indices], log_classical_errors[valid_indices], 1)
                classical_rate = -classical_fit[0]  # Negative because we want positive rate for faster convergence
            else:
                classical_rate = float('nan')
            
            # Quantum
            log_quantum_errors = np.log(results["quantum"][func_name]["error"])
            
            valid_indices = ~np.isnan(log_quantum_errors)
            if sum(valid_indices) >= 2:
                quantum_fit = np.polyfit(log_iterations[valid_indices], log_quantum_errors[valid_indices], 1)
                quantum_rate = -quantum_fit[0]  # Negative because we want positive rate for faster convergence
            else:
                quantum_rate = float('nan')
            
            convergence_rates.append((func_name, classical_rate, quantum_rate))
        except Exception as e:
            logger.error(f"Error calculating convergence rate for {func_name}: {str(e)}")
            convergence_rates.append((func_name, float('nan'), float('nan')))
    
    # Plot convergence rates
    x = np.arange(len(functions))
    width = 0.35
    
    plt.bar(x - width/2, [r[1] for r in convergence_rates], width, label='Classical')
    plt.bar(x + width/2, [r[2] for r in convergence_rates], width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Convergence Rate')
    plt.title('Empirical Convergence Rates')
    plt.xticks(x, [f.capitalize() for f in functions])
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='O(1/√n) reference')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='O(1/n) reference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_rates.png'))
    
    # Convergence efficiency (error reduction per unit time)
    plt.figure(figsize=(10, 6))
    
    efficiency = []
    for func_name in functions:
        classical_efficiency = []
        quantum_efficiency = []
        
        for i in range(len(iterations)):
            # Calculate error reduction per second
            if (not np.isnan(results["classical"][func_name]["error"][i]) and 
                not np.isnan(results["classical"][func_name]["time"][i]) and
                results["classical"][func_name]["time"][i] > 0):
                
                classical_eff = 1.0 / (results["classical"][func_name]["error"][i] * 
                                      results["classical"][func_name]["time"][i])
                classical_efficiency.append(classical_eff)
            else:
                classical_efficiency.append(float('nan'))
            
            if (not np.isnan(results["quantum"][func_name]["error"][i]) and 
                not np.isnan(results["quantum"][func_name]["time"][i]) and
                results["quantum"][func_name]["time"][i] > 0):
                
                quantum_eff = 1.0 / (results["quantum"][func_name]["error"][i] * 
                                    results["quantum"][func_name]["time"][i])
                quantum_efficiency.append(quantum_eff)
            else:
                quantum_efficiency.append(float('nan'))
        
        # Average efficiency across iterations
        classical_avg = np.nanmean(classical_efficiency)
        quantum_avg = np.nanmean(quantum_efficiency)
        
        efficiency.append((func_name, classical_avg, quantum_avg))
    
    # Plot efficiency
    x = np.arange(len(functions))
    width = 0.35
    
    plt.bar(x - width/2, [e[1] for e in efficiency], width, label='Classical')
    plt.bar(x + width/2, [e[2] for e in efficiency], width, label='Quantum')
    
    plt.xlabel('Function Type')
    plt.ylabel('Efficiency (1/(error*time))')
    plt.title('Convergence Efficiency')
    plt.xticks(x, [f.capitalize() for f in functions])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_efficiency.png'))

def main():
    """Run Monte Carlo convergence tests."""
    print_section("QUANTUM MONTE CARLO CONVERGENCE")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Define test functions with different characteristics
    target_functions = {
        "quadratic": lambda x, y, z: x**2 + y**2 + z**2,
        "trigonometric": lambda x, y, z: -np.sin(x) - np.sin(y) - np.sin(z),
        "mixed": lambda x, y, z: x**2 + y**2 + z**2 - np.sin(x) - np.sin(y) - np.sin(z),
        "exponential": lambda x, y, z: -np.exp(-x**2) - np.exp(-y**2) - np.exp(-z**2)
    }
    
    # Define iteration counts to test convergence
    iterations_list = [100, 250, 500, 1000, 2000]
    
    # Run convergence test
    results = test_convergence_rate(target_functions, iterations_list, repeats=3)
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("QUANTUM MONTE CARLO CONVERGENCE SUMMARY\n")
        f.write("======================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Calculate convergence rates for each function
        f.write("Convergence Rates (higher is better):\n")
        f.write("----------------------------------\n")
        
        for func_name in results["functions"]:
            try:
                # Log-log regression to find convergence rates
                log_iterations = np.log(results["iterations"])
                
                # Classical
                log_classical_errors = np.log(results["classical"][func_name]["error"])
                valid_indices = ~np.isnan(log_classical_errors)
                if sum(valid_indices) >= 2:
                    classical_fit = np.polyfit(log_iterations[valid_indices], log_classical_errors[valid_indices], 1)
                    classical_rate = -classical_fit[0]
                else:
                    classical_rate = float('nan')
                
                # Quantum
                log_quantum_errors = np.log(results["quantum"][func_name]["error"])
                valid_indices = ~np.isnan(log_quantum_errors)
                if sum(valid_indices) >= 2:
                    quantum_fit = np.polyfit(log_iterations[valid_indices], log_quantum_errors[valid_indices], 1)
                    quantum_rate = -quantum_fit[0]
                else:
                    quantum_rate = float('nan')
                
                f.write(f"  {func_name.capitalize()} function:\n")
                f.write(f"    Classical: O(1/n^{classical_rate:.3f})\n")
                f.write(f"    Quantum:   O(1/n^{quantum_rate:.3f})\n")
                
                if not np.isnan(quantum_rate) and not np.isnan(classical_rate):
                    f.write(f"    Improvement: {quantum_rate/classical_rate:.2f}x faster convergence\n\n")
                else:
                    f.write("    Improvement: Could not calculate\n\n")
                
            except Exception as e:
                logger.error(f"Error calculating convergence rate summary for {func_name}: {str(e)}")
                f.write(f"  {func_name.capitalize()} function: Error calculating convergence rates\n\n")
        
        # Overall findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        f.write("1. Quantum Monte Carlo methods show different convergence characteristics than classical approaches\n")
        
        # Determine which functions showed better quantum convergence
        better_quantum = []
        for func_name in results["functions"]:
            try:
                # Get errors at highest iteration count
                highest_idx = len(results["iterations"]) - 1
                c_error = results["classical"][func_name]["error"][highest_idx]
                q_error = results["quantum"][func_name]["error"][highest_idx]
                
                if not np.isnan(c_error) and not np.isnan(q_error) and q_error < c_error:
                    better_quantum.append(func_name)
            except:
                pass
        
        if better_quantum:
            f.write(f"2. Quantum methods achieved better final accuracy for: {', '.join(f.capitalize() for f in better_quantum)}\n")
        else:
            f.write("2. Classical methods generally achieved better final accuracy in these tests\n")
        
        # Check if quantum shows better than 1/sqrt(n) convergence
        faster_convergence = []
        for func_name in results["functions"]:
            try:
                log_iterations = np.log(results["iterations"])
                log_quantum_errors = np.log(results["quantum"][func_name]["error"])
                valid_indices = ~np.isnan(log_quantum_errors)
                if sum(valid_indices) >= 2:
                    quantum_fit = np.polyfit(log_iterations[valid_indices], log_quantum_errors[valid_indices], 1)
                    quantum_rate = -quantum_fit[0]
                    if quantum_rate > 0.6:  # Significantly better than 0.5 (sqrt(n))
                        faster_convergence.append(func_name)
            except:
                pass
        
        if faster_convergence:
            f.write(f"3. Quantum methods showed faster than O(1/√n) convergence for: {', '.join(f.capitalize() for f in faster_convergence)}\n")
        else:
            f.write("3. Quantum methods generally showed O(1/√n) convergence similar to classical Monte Carlo\n")
        
        f.write("4. The computational overhead of quantum methods may offset convergence advantages for simple problems\n")
        
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nConvergence tests completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 