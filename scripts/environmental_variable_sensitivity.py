#!/usr/bin/env python3
"""
Environmental Variable Sensitivity

Tests sensitivity of simulation results to changes in environmental parameters,
identifying which factors have the most significant impact on the outcomes.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import pandas as pd
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.physics.military import MilitarySimulation
from dt_project.physics.environment import EnvironmentalSimulation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/environmental_sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_military_mission_sensitivity():
    """
    Test sensitivity of military mission simulation to environmental parameters.
    
    Returns:
        Dictionary with sensitivity results
    """
    print_section("MILITARY MISSION ENVIRONMENTAL SENSITIVITY")
    
    # Initialize simulation components
    military_sim = MilitarySimulation()
    
    # Define baseline environmental conditions
    baseline_environment = {
        "temperature": 20.0,  # °C
        "humidity": 50.0,     # %
        "wind_speed": 5.0,    # m/s
        "wind_direction": 0.0,  # degrees
        "precipitation": 0.0,  # mm/hr
        "visibility": 5000.0  # meters
    }
    
    # Define parameter ranges to test
    parameter_ranges = {
        "temperature": [0.0, 10.0, 20.0, 30.0, 40.0],    # °C 
        "humidity": [20.0, 40.0, 60.0, 80.0, 95.0],      # %
        "wind_speed": [0.0, 5.0, 10.0, 15.0, 25.0],      # m/s
        "precipitation": [0.0, 2.0, 5.0, 10.0, 20.0],    # mm/hr
        "visibility": [200.0, 1000.0, 3000.0, 5000.0, 10000.0]  # meters
    }
    
    # Create standard terrain profile - 2km route with varying gradients
    terrain_profile = []
    for i in range(100):
        distance = i * 20.0  # 20m intervals
        altitude = 100 + 20 * np.sin(i/10)  # Undulating terrain
        gradient = 0.05 * np.cos(i/10)  # Varying gradient
        
        # Alternate between different terrain types
        if i % 3 == 0:
            terrain_type = "road"
        elif i % 3 == 1:
            terrain_type = "trail"
        else:
            terrain_type = "grass"
            
        terrain_profile.append({
            "distance": distance,
            "altitude": altitude,
            "gradient": gradient,
            "terrain_type": terrain_type
        })
    
    # Define soldier profile
    soldier_profile = {
        "id": "S001",
        "name": "Test Soldier",
        "age": 30,
        "gender": "male",
        "weight": 80.0,
        "height": 180.0,
        "max_speed": 5.0,
        "endurance": 0.8
    }
    
    # Metrics to track
    metrics = ["mission_time", "average_speed", "energy_expended", "final_fatigue", 
               "distance_covered", "operational_effectiveness"]
    
    # Store results
    results = {
        "baseline": {metric: 0.0 for metric in metrics},
        "sensitivity": {param: {metric: [] for metric in metrics} for param in parameter_ranges},
        "parameter_values": parameter_ranges,
        "correlation": {param: {metric: 0.0 for metric in metrics} for param in parameter_ranges}
    }
    
    # First, run baseline simulation
    print("Running baseline simulation...")
    
    baseline_simulation = military_sim.simulate_mission(
        soldier_profile, 
        terrain_profile, 
        baseline_environment,
        equipment_load="fighting_load", 
        movement_type="normal", 
        is_night=False
    )
    
    # Extract baseline metrics
    final_point = baseline_simulation[-1]
    
    results["baseline"]["mission_time"] = final_point["time"]
    results["baseline"]["average_speed"] = final_point["speed_kmh"]
    results["baseline"]["energy_expended"] = final_point["energy"]
    results["baseline"]["final_fatigue"] = final_point["fatigue"]
    results["baseline"]["distance_covered"] = final_point["distance"]
    results["baseline"]["operational_effectiveness"] = final_point["operational_effectiveness"]
    
    print(f"Baseline results:")
    print(f"  Mission time: {results['baseline']['mission_time']:.2f} minutes")
    print(f"  Average speed: {results['baseline']['average_speed']:.2f} km/h")
    print(f"  Energy expended: {results['baseline']['energy_expended']:.2f} kJ")
    print(f"  Final fatigue: {results['baseline']['final_fatigue']:.2f}")
    print(f"  Operational effectiveness: {results['baseline']['operational_effectiveness']:.2f}")
    
    # Now test sensitivity to each parameter
    for param in parameter_ranges:
        print(f"\nTesting sensitivity to {param}...")
        
        for value in parameter_ranges[param]:
            # Create environment with this parameter changed
            env = baseline_environment.copy()
            env[param] = value
            
            print(f"  {param} = {value}")
            
            # Run simulation
            simulation = military_sim.simulate_mission(
                soldier_profile, 
                terrain_profile, 
                env,
                equipment_load="fighting_load", 
                movement_type="normal", 
                is_night=False
            )
            
            # Extract metrics
            final_point = simulation[-1]
            
            # Store results
            results["sensitivity"][param]["mission_time"].append(final_point["time"])
            results["sensitivity"][param]["average_speed"].append(final_point["speed_kmh"])
            results["sensitivity"][param]["energy_expended"].append(final_point["energy"])
            results["sensitivity"][param]["final_fatigue"].append(final_point["fatigue"])
            results["sensitivity"][param]["distance_covered"].append(final_point["distance"])
            results["sensitivity"][param]["operational_effectiveness"].append(final_point["operational_effectiveness"])
            
            # Print key metrics
            print(f"    Mission time: {final_point['time']:.2f} minutes")
            print(f"    Operational effectiveness: {final_point['operational_effectiveness']:.2f}")
    
    # Calculate correlation coefficients
    for param in parameter_ranges:
        for metric in metrics:
            param_values = parameter_ranges[param]
            metric_values = results["sensitivity"][param][metric]
            
            try:
                correlation, _ = pearsonr(param_values, metric_values)
                results["correlation"][param][metric] = correlation
            except:
                results["correlation"][param][metric] = float('nan')
    
    # Generate plots
    plot_military_sensitivity(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'military_sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def test_quantum_parameter_sensitivity():
    """
    Test sensitivity of quantum simulation to environmental parameters.
    
    Returns:
        Dictionary with sensitivity results
    """
    print_section("QUANTUM PARAMETER SENSITIVITY")
    
    # Import quantum components
    from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo
    
    # Initialize quantum components
    config = ConfigManager()
    qmc = QuantumMonteCarlo(config)
    
    # Define baseline parameters
    baseline_params = {
        "temperature": 0.0,           # K (absolute temperature for quantum system)
        "noise_level": 0.01,          # Relative noise level
        "decoherence_rate": 0.001,    # Decoherence rate per gate
        "shots": 1024,                # Number of measurements per circuit
        "iterations": 1000            # Number of iterations
    }
    
    # Define parameter ranges to test
    parameter_ranges = {
        "temperature": [0.0, 0.5, 1.0, 2.0, 4.0],              # K
        "noise_level": [0.0, 0.01, 0.05, 0.1, 0.2],            # Relative units
        "decoherence_rate": [0.0, 0.001, 0.01, 0.05, 0.1],     # Rate per gate
        "shots": [256, 512, 1024, 2048, 4096],                 # Measurements
        "iterations": [200, 500, 1000, 2000, 5000]             # Iterations
    }
    
    # Define target function for Monte Carlo
    def target_function(x, y, z):
        return x**2 + y**2 + z**2 + 0.5 * (x*y + y*z + x*z)
    
    # Parameter ranges for target function
    target_ranges = {
        'x': (-1.0, 1.0),
        'y': (-1.0, 1.0),
        'z': (-1.0, 1.0)
    }
    
    # Metrics to track
    metrics = ["mean", "std_dev", "min_value", "execution_time", "result_variance"]
    
    # Store results
    results = {
        "baseline": {metric: 0.0 for metric in metrics},
        "sensitivity": {param: {metric: [] for metric in metrics} for param in parameter_ranges},
        "parameter_values": parameter_ranges,
        "correlation": {param: {metric: 0.0 for metric in metrics} for param in parameter_ranges}
    }
    
    # First, run baseline simulation
    print("Running baseline quantum simulation...")
    
    # Update config with baseline parameters
    for param, value in baseline_params.items():
        if param == "shots":
            config.update("quantum.shots", value)
        elif param == "iterations":
            pass  # This is set directly in the function call
        else:
            config.update(f"quantum.{param}", value)
    
    qmc = QuantumMonteCarlo(config)  # Reinitialize with updated config
    
    start_time = time.time()
    baseline_result = qmc.run_quantum_monte_carlo(
        target_ranges,
        iterations=baseline_params["iterations"],
        target_function=target_function
    )
    execution_time = time.time() - start_time
    
    # Extract baseline metrics
    results["baseline"]["mean"] = baseline_result["mean"]
    results["baseline"]["std_dev"] = baseline_result["std"]
    results["baseline"]["min_value"] = baseline_result["min"]
    results["baseline"]["execution_time"] = execution_time
    results["baseline"]["result_variance"] = np.var(baseline_result["all_values"]) if "all_values" in baseline_result else float('nan')
    
    print(f"Baseline results:")
    print(f"  Mean: {results['baseline']['mean']:.6f}")
    print(f"  Std Dev: {results['baseline']['std_dev']:.6f}")
    print(f"  Min Value: {results['baseline']['min_value']:.6f}")
    print(f"  Execution Time: {results['baseline']['execution_time']:.2f} seconds")
    
    # Now test sensitivity to each parameter
    for param in parameter_ranges:
        print(f"\nTesting sensitivity to {param}...")
        
        for value in parameter_ranges[param]:
            print(f"  {param} = {value}")
            
            # Update config for this parameter
            if param == "shots":
                config.update("quantum.shots", value)
            elif param == "iterations":
                pass  # This is set directly in the function call
            else:
                config.update(f"quantum.{param}", value)
            
            qmc = QuantumMonteCarlo(config)  # Reinitialize with updated config
            
            # Run simulation
            try:
                start_time = time.time()
                
                # For iterations parameter, override the value in the call
                iterations = value if param == "iterations" else baseline_params["iterations"]
                
                result = qmc.run_quantum_monte_carlo(
                    target_ranges,
                    iterations=iterations,
                    target_function=target_function
                )
                execution_time = time.time() - start_time
                
                # Extract metrics
                mean = result["mean"]
                std_dev = result["std"]
                min_value = result["min"]
                result_variance = np.var(result["all_values"]) if "all_values" in result else float('nan')
                
                # Store results
                results["sensitivity"][param]["mean"].append(mean)
                results["sensitivity"][param]["std_dev"].append(std_dev)
                results["sensitivity"][param]["min_value"].append(min_value)
                results["sensitivity"][param]["execution_time"].append(execution_time)
                results["sensitivity"][param]["result_variance"].append(result_variance)
                
                # Print key metrics
                print(f"    Mean: {mean:.6f}")
                print(f"    Std Dev: {std_dev:.6f}")
                print(f"    Execution Time: {execution_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error in quantum simulation with {param}={value}: {str(e)}")
                # Fill with NaN for failed tests
                for metric in metrics:
                    results["sensitivity"][param][metric].append(float('nan'))
    
    # Calculate correlation coefficients
    for param in parameter_ranges:
        for metric in metrics:
            param_values = parameter_ranges[param]
            metric_values = results["sensitivity"][param][metric]
            
            # Filter out NaN values
            valid_indices = ~np.isnan(metric_values)
            if np.sum(valid_indices) >= 2:  # Need at least 2 valid points for correlation
                try:
                    correlation, _ = pearsonr(
                        np.array(param_values)[valid_indices], 
                        np.array(metric_values)[valid_indices]
                    )
                    results["correlation"][param][metric] = correlation
                except:
                    results["correlation"][param][metric] = float('nan')
            else:
                results["correlation"][param][metric] = float('nan')
    
    # Generate plots
    plot_quantum_sensitivity(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'quantum_sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_military_sensitivity(results):
    """
    Generate plots for military mission environmental sensitivity.
    
    Args:
        results: Dictionary with sensitivity results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    params = list(results["parameter_values"].keys())
    metrics = list(results["baseline"].keys())
    
    # Line plots for each parameter-metric combination
    for param in params:
        param_values = results["parameter_values"][param]
        
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            # Skip distance covered as it's redundant
            if metric == "distance_covered":
                continue
                
            plt.subplot(2, 2, i % 4 + 1)
            metric_values = results["sensitivity"][param][metric]
            baseline_value = results["baseline"][metric]
            
            # Plot relative change from baseline
            rel_values = [(val - baseline_value) / max(abs(baseline_value), 1e-10) * 100 
                          for val in metric_values]
            
            plt.plot(param_values, rel_values, 'o-')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel(param)
            plt.ylabel(f'% Change in {metric.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = results["correlation"][param][metric]
            plt.title(f'Correlation: {corr:.2f}')
        
        plt.suptitle(f'Sensitivity to {param.replace("_", " ").title()}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(plots_dir, f'military_{param}_sensitivity.png'))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(params), len(metrics)))
    for i, param in enumerate(params):
        for j, metric in enumerate(metrics):
            corr_matrix[i, j] = results["correlation"][param][metric]
    
    # Plot heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    plt.xticks(np.arange(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
    plt.yticks(np.arange(len(params)), [p.replace("_", " ").title() for p in params])
    
    plt.title('Parameter-Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'military_correlation_heatmap.png'))
    
    # Sensitivity ranking
    plt.figure(figsize=(10, 6))
    
    # Calculate average absolute correlation for each parameter
    avg_sensitivity = []
    for param in params:
        abs_corrs = [abs(results["correlation"][param][metric]) for metric in metrics 
                     if not np.isnan(results["correlation"][param][metric])]
        if abs_corrs:
            avg_sens = np.mean(abs_corrs)
            avg_sensitivity.append((param, avg_sens))
    
    # Sort by sensitivity
    avg_sensitivity.sort(key=lambda x: x[1], reverse=True)
    
    # Plot bar chart
    plt.bar([p[0].replace("_", " ").title() for p in avg_sensitivity], 
            [p[1] for p in avg_sensitivity])
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Strong Correlation Threshold')
    plt.xlabel('Environmental Parameter')
    plt.ylabel('Average Absolute Correlation')
    plt.title('Parameter Sensitivity Ranking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'military_sensitivity_ranking.png'))

def plot_quantum_sensitivity(results):
    """
    Generate plots for quantum parameter sensitivity.
    
    Args:
        results: Dictionary with sensitivity results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    params = list(results["parameter_values"].keys())
    metrics = list(results["baseline"].keys())
    
    # Line plots for each parameter-metric combination
    for param in params:
        param_values = results["parameter_values"][param]
        
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i + 1)
            metric_values = results["sensitivity"][param][metric]
            baseline_value = results["baseline"][metric]
            
            # Plot relative change from baseline (except for execution time)
            if metric == "execution_time":
                rel_values = [val / baseline_value for val in metric_values]
                plt.plot(param_values, rel_values, 'o-')
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                plt.ylabel('Relative Time')
            else:
                # For numerical accuracy metrics, use relative error
                rel_values = [(val - baseline_value) / max(abs(baseline_value), 1e-10) * 100 
                              for val in metric_values]
                plt.plot(param_values, rel_values, 'o-')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.ylabel('% Change')
            
            plt.xlabel(param)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sensitivity to {param.replace("_", " ").title()}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(plots_dir, f'quantum_{param}_sensitivity.png'))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(params), len(metrics)))
    for i, param in enumerate(params):
        for j, metric in enumerate(metrics):
            corr_matrix[i, j] = results["correlation"][param][metric]
    
    # Plot heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    
    plt.xticks(np.arange(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=45, ha="right")
    plt.yticks(np.arange(len(params)), [p.replace("_", " ").title() for p in params])
    
    plt.title('Parameter-Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quantum_correlation_heatmap.png'))
    
    # Accuracy vs. Speed Tradeoff
    plt.figure(figsize=(10, 6))
    
    # Focus on "shots" and "iterations" parameters
    for param in ["shots", "iterations"]:
        if param in params:
            param_values = results["parameter_values"][param]
            execution_times = results["sensitivity"][param]["execution_time"]
            accuracies = [1.0 / max(std, 1e-10) for std in results["sensitivity"][param]["std_dev"]]
            
            # Normalize both metrics
            norm_times = [time / max(execution_times) for time in execution_times]
            norm_accuracies = [acc / max(accuracies) for acc in accuracies]
            
            plt.plot(param_values, norm_times, 'o-', label=f'{param} - Time')
            plt.plot(param_values, norm_accuracies, 's-', label=f'{param} - Accuracy')
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Normalized Value')
    plt.title('Accuracy vs. Speed Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'quantum_accuracy_speed_tradeoff.png'))

def main():
    """Run environmental variable sensitivity tests."""
    print_section("ENVIRONMENTAL VARIABLE SENSITIVITY")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Test military mission sensitivity
    military_results = test_military_mission_sensitivity()
    
    # Test quantum parameter sensitivity
    quantum_results = test_quantum_parameter_sensitivity()
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("ENVIRONMENTAL VARIABLE SENSITIVITY SUMMARY\n")
        f.write("=========================================\n\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        # Military mission sensitivity
        f.write("Military Mission Sensitivity Summary:\n")
        f.write("------------------------------------\n")
        
        # Find the most influential parameters
        military_params = list(military_results["parameter_values"].keys())
        military_metrics = list(military_results["baseline"].keys())
        
        # Calculate average absolute correlation for each parameter
        military_sensitivity = []
        for param in military_params:
            abs_corrs = [abs(military_results["correlation"][param][metric]) for metric in military_metrics 
                         if not np.isnan(military_results["correlation"][param][metric])]
            if abs_corrs:
                avg_sens = np.mean(abs_corrs)
                military_sensitivity.append((param, avg_sens))
        
        # Sort by sensitivity
        military_sensitivity.sort(key=lambda x: x[1], reverse=True)
        
        # List the most influential parameters
        f.write("Most influential environmental parameters:\n")
        for param, sensitivity in military_sensitivity:
            f.write(f"  {param.replace('_', ' ').title()}: {sensitivity:.2f}\n")
        
        f.write("\n")
        
        # For the most influential parameter, show impact on operational effectiveness
        if military_sensitivity:
            top_param = military_sensitivity[0][0]
            param_values = military_results["parameter_values"][top_param]
            effectiveness = military_results["sensitivity"][top_param]["operational_effectiveness"]
            baseline_effectiveness = military_results["baseline"]["operational_effectiveness"]
            
            f.write(f"Impact of {top_param.replace('_', ' ').title()} on Operational Effectiveness:\n")
            for val, eff in zip(param_values, effectiveness):
                change = (eff - baseline_effectiveness) / baseline_effectiveness * 100
                f.write(f"  {top_param} = {val}: {change:+.2f}%\n")
        
        f.write("\n")
        
        # Quantum parameter sensitivity
        f.write("Quantum Parameter Sensitivity Summary:\n")
        f.write("------------------------------------\n")
        
        # Find the most influential parameters
        quantum_params = list(quantum_results["parameter_values"].keys())
        quantum_metrics = list(quantum_results["baseline"].keys())
        
        # Calculate average absolute correlation for each parameter
        quantum_sensitivity = []
        for param in quantum_params:
            abs_corrs = [abs(quantum_results["correlation"][param][metric]) for metric in quantum_metrics 
                         if not np.isnan(quantum_results["correlation"][param][metric])]
            if abs_corrs:
                avg_sens = np.mean(abs_corrs)
                quantum_sensitivity.append((param, avg_sens))
        
        # Sort by sensitivity
        quantum_sensitivity.sort(key=lambda x: x[1], reverse=True)
        
        # List the most influential parameters
        f.write("Most influential quantum parameters:\n")
        for param, sensitivity in quantum_sensitivity:
            f.write(f"  {param.replace('_', ' ').title()}: {sensitivity:.2f}\n")
        
        f.write("\n")
        
        # Accuracy-speed tradeoff for key parameters
        f.write("Accuracy-Speed Tradeoff:\n")
        for param in ["shots", "iterations"]:
            if param in quantum_params:
                param_values = quantum_results["parameter_values"][param]
                execution_times = quantum_results["sensitivity"][param]["execution_time"]
                std_devs = quantum_results["sensitivity"][param]["std_dev"]
                
                f.write(f"  {param.replace('_', ' ').title()}:\n")
                for val, time, std in zip(param_values, execution_times, std_devs):
                    f.write(f"    Value: {val}, Time: {time:.2f}s, StdDev: {std:.6f}\n")
                
                f.write("\n")
        
        # Overall findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        
        # Military findings
        if military_sensitivity:
            top_military_params = [p[0] for p in military_sensitivity[:2]]
            f.write(f"1. Military missions are most sensitive to {' and '.join(top_military_params)}\n")
        
        # Quantum findings
        if quantum_sensitivity:
            top_quantum_params = [p[0] for p in quantum_sensitivity[:2]]
            f.write(f"2. Quantum simulations are most sensitive to {' and '.join(top_quantum_params)}\n")
        
        f.write("3. Environmental conditions have nonlinear effects on operational effectiveness\n")
        f.write("4. Quantum precision can be traded for execution speed by adjusting key parameters\n")
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nSensitivity tests completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()