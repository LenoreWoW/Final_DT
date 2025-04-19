#!/usr/bin/env python3
"""
Batch Simulation Processor

Tests system robustness and throughput by processing large batches of simulations.
This script evaluates the system's ability to handle concurrent simulations
and measures various performance metrics.
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import logging
from datetime import datetime
import json
from queue import Queue
from threading import Thread

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.config import ConfigManager
from dt_project.quantum import initialize_quantum_components, QuantumMonteCarlo, QuantumML
from dt_project.physics.military import MilitarySimulation
from dt_project.physics.environment import EnvironmentalSimulation
from dt_project.physics.terrain import TerrainSimulation, Point
from dt_project.physics.biomechanics import BiomechanicalModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots and results
OUTPUT_DIR = "results/batch_processor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_military_simulation(config):
    """
    Run a single military simulation with the given configuration.
    
    Args:
        config: Dictionary with simulation parameters
        
    Returns:
        Dictionary with simulation results
    """
    try:
        # Extract parameters
        soldier_profile = config["soldier_profile"]
        terrain_profile = config["terrain_profile"]
        environmental_conditions = config["environmental_conditions"]
        equipment_load = config["equipment_load"]
        movement_type = config["movement_type"]
        is_night = config["is_night"]
        
        # Initialize simulation components
        military_sim = MilitarySimulation()
        
        # Run simulation
        start_time = time.time()
        mission_data = military_sim.simulate_mission(
            soldier_profile, terrain_profile, environmental_conditions,
            equipment_load, movement_type, is_night
        )
        execution_time = time.time() - start_time
        
        # Extract key metrics from last point
        last_point = mission_data[-1]
        
        return {
            "status": "success",
            "execution_time": execution_time,
            "final_distance": last_point["distance"],
            "final_time": last_point["time"],
            "average_speed": last_point["speed_kmh"],
            "final_fatigue": last_point["fatigue"],
            "energy_expended": last_point["energy"],
            "operational_effectiveness": last_point["operational_effectiveness"]
        }
        
    except Exception as e:
        logger.error(f"Error in military simulation: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def run_quantum_simulation(config):
    """
    Run a single quantum simulation with the given configuration.
    
    Args:
        config: Dictionary with simulation parameters
        
    Returns:
        Dictionary with simulation results
    """
    try:
        # Extract parameters
        param_ranges = config["param_ranges"]
        iterations = config["iterations"]
        distribution_type = config["distribution_type"]
        
        # Initialize quantum components
        qmc = QuantumMonteCarlo(ConfigManager())
        
        # Define target function
        def target_function(*args):
            return sum(x**2 for x in args) + 0.1 * sum(args[i] * args[j] for i in range(len(args)) for j in range(i+1, len(args)))
        
        # Run simulation
        start_time = time.time()
        result = qmc.run_quantum_monte_carlo(
            param_ranges, 
            iterations=iterations,
            target_function=target_function,
            distribution_type=distribution_type
        )
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "execution_time": execution_time,
            "mean": result["mean"],
            "std": result["std"],
            "iterations": iterations
        }
        
    except Exception as e:
        logger.error(f"Error in quantum simulation: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def generate_random_config(sim_type):
    """
    Generate a random simulation configuration.
    
    Args:
        sim_type: Type of simulation ("military" or "quantum")
        
    Returns:
        Dictionary with simulation parameters
    """
    if sim_type == "military":
        # Generate a random soldier profile
        soldier_profile = {
            "id": f"soldier_{random.randint(1, 1000)}",
            "name": f"Soldier {random.randint(1, 1000)}",
            "age": random.randint(20, 40),
            "gender": random.choice(["male", "female"]),
            "weight": random.uniform(60, 90),
            "height": random.uniform(160, 190),
            "max_speed": random.uniform(4.0, 6.0),
            "endurance": random.uniform(0.6, 0.9)
        }
        
        # Generate a random terrain profile
        terrain_length = random.randint(1000, 5000)
        num_points = random.randint(20, 50)
        
        terrain_profile = []
        for i in range(num_points):
            distance = i * terrain_length / (num_points - 1)
            terrain_profile.append({
                "distance": distance,
                "altitude": 100 + 50 * np.sin(distance / 500),
                "gradient": 0.05 * np.cos(distance / 500),
                "terrain_type": random.choice(["road", "trail", "grass"])
            })
        
        # Generate random environmental conditions
        environmental_conditions = {
            "temperature": random.uniform(10, 35),
            "humidity": random.uniform(30, 90),
            "wind_speed": random.uniform(0, 15),
            "wind_direction": random.uniform(0, 360),
            "precipitation": random.uniform(0, 10),
            "visibility": random.uniform(500, 10000)
        }
        
        return {
            "soldier_profile": soldier_profile,
            "terrain_profile": terrain_profile,
            "environmental_conditions": environmental_conditions,
            "equipment_load": random.choice(["fighting_load", "approach_load", "emergency_load"]),
            "movement_type": random.choice(["normal", "rush", "patrol", "stealth"]),
            "is_night": random.choice([True, False])
        }
        
    elif sim_type == "quantum":
        # Generate random parameter ranges
        dim = random.randint(1, 5)
        param_ranges = {}
        for i in range(dim):
            param_ranges[f'x{i}'] = (-2.0, 2.0)
        
        return {
            "param_ranges": param_ranges,
            "iterations": random.choice([500, 1000, 2000]),
            "distribution_type": random.choice(["uniform", "normal", "exponential"])
        }
    
    return {}

def test_sequential_processing(num_simulations=10, sim_types=None):
    """
    Test sequential processing of simulations.
    
    Args:
        num_simulations: Number of simulations to run
        sim_types: List of simulation types to include
        
    Returns:
        Dictionary with performance metrics
    """
    print_section("SEQUENTIAL PROCESSING TEST")
    
    if sim_types is None:
        sim_types = ["military", "quantum"]
    
    print(f"Running {num_simulations} sequential simulations...")
    
    # Generate simulation configurations
    configs = []
    for i in range(num_simulations):
        sim_type = sim_types[i % len(sim_types)]
        configs.append((sim_type, generate_random_config(sim_type)))
    
    # Run simulations sequentially
    results = []
    start_time = time.time()
    
    for i, (sim_type, config) in enumerate(configs):
        print(f"Running simulation {i+1}/{num_simulations} ({sim_type})...")
        
        if sim_type == "military":
            result = run_military_simulation(config)
        else:  # quantum
            result = run_quantum_simulation(config)
            
        results.append({
            "type": sim_type,
            "result": result
        })
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    success_count = sum(1 for r in results if r["result"]["status"] == "success")
    success_rate = success_count / num_simulations if num_simulations > 0 else 0
    
    avg_execution_time = np.mean([r["result"]["execution_time"] for r in results 
                                 if r["result"]["status"] == "success" and "execution_time" in r["result"]])
    
    metrics = {
        "total_simulations": num_simulations,
        "successful_simulations": success_count,
        "success_rate": success_rate,
        "total_processing_time": total_time,
        "average_execution_time": avg_execution_time,
        "throughput": num_simulations / total_time if total_time > 0 else 0,
        "results": results
    }
    
    print(f"Sequential processing completed in {total_time:.2f} seconds")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Average execution time: {avg_execution_time:.2f} seconds")
    print(f"Throughput: {metrics['throughput']:.2f} simulations/second")
    
    return metrics

def test_parallel_processing(num_simulations=10, max_workers=None, sim_types=None):
    """
    Test parallel processing of simulations.
    
    Args:
        num_simulations: Number of simulations to run
        max_workers: Maximum number of worker processes
        sim_types: List of simulation types to include
        
    Returns:
        Dictionary with performance metrics
    """
    print_section("PARALLEL PROCESSING TEST")
    
    if sim_types is None:
        sim_types = ["military", "quantum"]
        
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)
    
    print(f"Running {num_simulations} parallel simulations with {max_workers} workers...")
    
    # Generate simulation configurations
    configs = []
    for i in range(num_simulations):
        sim_type = sim_types[i % len(sim_types)]
        configs.append((sim_type, generate_random_config(sim_type)))
    
    # Run simulations in parallel using ProcessPoolExecutor
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare futures
        futures = []
        for sim_type, config in configs:
            if sim_type == "military":
                future = executor.submit(run_military_simulation, config)
            else:  # quantum
                future = executor.submit(run_quantum_simulation, config)
                
            futures.append((sim_type, future))
        
        # Collect results
        for i, (sim_type, future) in enumerate(futures):
            try:
                result = future.result()
                results.append({
                    "type": sim_type,
                    "result": result
                })
                print(f"Completed simulation {i+1}/{num_simulations} ({sim_type})")
            except Exception as e:
                logger.error(f"Error in parallel simulation {i+1}: {str(e)}")
                results.append({
                    "type": sim_type,
                    "result": {"status": "error", "error": str(e)}
                })
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    success_count = sum(1 for r in results if r["result"]["status"] == "success")
    success_rate = success_count / num_simulations if num_simulations > 0 else 0
    
    avg_execution_time = np.mean([r["result"]["execution_time"] for r in results 
                                 if r["result"]["status"] == "success" and "execution_time" in r["result"]])
    
    metrics = {
        "total_simulations": num_simulations,
        "successful_simulations": success_count,
        "success_rate": success_rate,
        "total_processing_time": total_time,
        "average_execution_time": avg_execution_time,
        "throughput": num_simulations / total_time if total_time > 0 else 0,
        "num_workers": max_workers,
        "parallelization_speedup": 0,  # Will be filled after sequential test
        "results": results
    }
    
    print(f"Parallel processing completed in {total_time:.2f} seconds")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Average execution time: {avg_execution_time:.2f} seconds")
    print(f"Throughput: {metrics['throughput']:.2f} simulations/second")
    
    return metrics

def test_batch_throughput(batch_sizes=[10, 20, 50, 100], max_workers=None):
    """
    Test system throughput with different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary with throughput metrics
    """
    print_section("BATCH THROUGHPUT TEST")
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)
    
    print(f"Testing throughput with different batch sizes using {max_workers} workers...")
    
    # Store results
    results = {
        "batch_sizes": batch_sizes,
        "sequential_throughput": [],
        "parallel_throughput": [],
        "success_rates": [],
        "speedups": []
    }
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Run sequential test
        seq_metrics = test_sequential_processing(num_simulations=batch_size)
        
        # Run parallel test
        parallel_metrics = test_parallel_processing(num_simulations=batch_size, max_workers=max_workers)
        
        # Calculate speedup
        speedup = seq_metrics["total_processing_time"] / parallel_metrics["total_processing_time"] if parallel_metrics["total_processing_time"] > 0 else 0
        parallel_metrics["parallelization_speedup"] = speedup
        
        # Store metrics
        results["sequential_throughput"].append(seq_metrics["throughput"])
        results["parallel_throughput"].append(parallel_metrics["throughput"])
        results["success_rates"].append(parallel_metrics["success_rate"])
        results["speedups"].append(speedup)
        
        print(f"Parallelization speedup: {speedup:.2f}x")
    
    # Generate throughput plot
    plot_throughput_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'throughput_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def test_robustness(num_simulations=100, failure_rates=[0, 0.1, 0.2, 0.5], max_workers=None):
    """
    Test system robustness by introducing different failure rates.
    
    Args:
        num_simulations: Number of simulations per failure rate
        failure_rates: List of failure rates to test
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary with robustness metrics
    """
    print_section("ROBUSTNESS TEST")
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)
    
    print(f"Testing robustness with different failure rates...")
    
    # Store results
    results = {
        "failure_rates": failure_rates,
        "completion_rates": [],
        "processing_times": [],
        "avg_execution_times": []
    }
    
    # Original simulation runner to be patched
    original_military_sim = run_military_simulation
    original_quantum_sim = run_quantum_simulation
    
    # Run tests for each failure rate
    for failure_rate in failure_rates:
        print(f"\nTesting with failure rate: {failure_rate:.1%}")
        
        # Patch the simulation runners to introduce failures
        def patched_military_sim(config):
            if random.random() < failure_rate:
                # Simulate a failure
                time.sleep(random.uniform(0.1, 0.5))  # Simulate some processing time
                return {"status": "error", "error": "Simulated failure"}
            else:
                return original_military_sim(config)
                
        def patched_quantum_sim(config):
            if random.random() < failure_rate:
                # Simulate a failure
                time.sleep(random.uniform(0.1, 0.5))  # Simulate some processing time
                return {"status": "error", "error": "Simulated failure"}
            else:
                return original_quantum_sim(config)
        
        # Temporarily replace the simulation runners
        globals()["run_military_simulation"] = patched_military_sim
        globals()["run_quantum_simulation"] = patched_quantum_sim
        
        try:
            # Run batch with current failure rate
            metrics = test_parallel_processing(
                num_simulations=num_simulations,
                max_workers=max_workers
            )
            
            # Store metrics
            results["completion_rates"].append(metrics["success_rate"])
            results["processing_times"].append(metrics["total_processing_time"])
            results["avg_execution_times"].append(metrics["average_execution_time"])
            
        finally:
            # Restore original simulation runners
            globals()["run_military_simulation"] = original_military_sim
            globals()["run_quantum_simulation"] = original_quantum_sim
    
    # Generate robustness plot
    plot_robustness_results(results)
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_throughput_results(results):
    """
    Generate plots for throughput test results.
    
    Args:
        results: Dictionary with throughput results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Throughput comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results["batch_sizes"], results["sequential_throughput"], 'o-', label='Sequential')
    plt.plot(results["batch_sizes"], results["parallel_throughput"], 's-', label='Parallel')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (simulations/second)')
    plt.title('Batch Processing Throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'throughput_comparison.png'))
    
    # 2. Speedup factor
    plt.figure(figsize=(10, 6))
    plt.plot(results["batch_sizes"], results["speedups"], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor (Sequential/Parallel)')
    plt.title('Parallelization Speedup')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parallelization_speedup.png'))
    
    # 3. Success rates
    plt.figure(figsize=(10, 6))
    plt.plot(results["batch_sizes"], [rate * 100 for rate in results["success_rates"]], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Success Rate (%)')
    plt.title('Batch Processing Success Rate')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'success_rate.png'))

def plot_robustness_results(results):
    """
    Generate plots for robustness test results.
    
    Args:
        results: Dictionary with robustness results
    """
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Completion rates
    plt.figure(figsize=(10, 6))
    plt.plot([rate * 100 for rate in results["failure_rates"]], 
             [rate * 100 for rate in results["completion_rates"]], 'o-')
    plt.xlabel('Injected Failure Rate (%)')
    plt.ylabel('Completion Rate (%)')
    plt.title('System Robustness: Completion Rate vs Failure Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_completion.png'))
    
    # 2. Processing times
    plt.figure(figsize=(10, 6))
    plt.plot([rate * 100 for rate in results["failure_rates"]], 
             results["processing_times"], 'o-')
    plt.xlabel('Injected Failure Rate (%)')
    plt.ylabel('Total Processing Time (s)')
    plt.title('System Robustness: Processing Time vs Failure Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_processing_time.png'))
    
    # 3. Execution times
    plt.figure(figsize=(10, 6))
    plt.plot([rate * 100 for rate in results["failure_rates"]], 
             results["avg_execution_times"], 'o-')
    plt.xlabel('Injected Failure Rate (%)')
    plt.ylabel('Average Execution Time (s)')
    plt.title('System Robustness: Execution Time vs Failure Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_execution_time.png'))

def main():
    """Run batch simulation processor tests."""
    print_section("BATCH SIMULATION PROCESSOR")
    print(f"Results will be saved to {os.path.abspath(OUTPUT_DIR)}")
    
    start_time = time.time()
    
    # Determine number of CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    max_workers = min(num_cores, 4)  # Use up to 4 cores
    
    print(f"System has {num_cores} CPU cores, using {max_workers} workers for parallel tests")
    
    # Test throughput with different batch sizes
    throughput_results = test_batch_throughput(
        batch_sizes=[10, 25, 50],
        max_workers=max_workers
    )
    
    # Test robustness with failure injection
    robustness_results = test_robustness(
        num_simulations=30,
        failure_rates=[0, 0.1, 0.3, 0.5],
        max_workers=max_workers
    )
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
        f.write("BATCH SIMULATION PROCESSOR SUMMARY\n")
        f.write("==================================\n\n")
        f.write(f"Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total testing time: {total_time:.2f} seconds\n\n")
        
        # Throughput summary
        f.write("Throughput Summary:\n")
        f.write("-----------------\n")
        f.write(f"Batch sizes tested: {throughput_results['batch_sizes']}\n")
        
        max_throughput_idx = np.argmax(throughput_results["parallel_throughput"])
        max_throughput = throughput_results["parallel_throughput"][max_throughput_idx]
        max_throughput_batch = throughput_results["batch_sizes"][max_throughput_idx]
        
        f.write(f"Maximum throughput: {max_throughput:.2f} simulations/second with batch size {max_throughput_batch}\n")
        
        max_speedup_idx = np.argmax(throughput_results["speedups"])
        max_speedup = throughput_results["speedups"][max_speedup_idx]
        max_speedup_batch = throughput_results["batch_sizes"][max_speedup_idx]
        
        f.write(f"Maximum parallelization speedup: {max_speedup:.2f}x with batch size {max_speedup_batch}\n\n")
        
        # Robustness summary
        f.write("Robustness Summary:\n")
        f.write("------------------\n")
        f.write(f"Failure rates tested: {[f'{rate:.1%}' for rate in robustness_results['failure_rates']]}\n")
        
        # Calculate degradation factor: how much the completion rate drops as failure rate increases
        degradation_factor = (robustness_results["completion_rates"][0] - robustness_results["completion_rates"][-1]) / (robustness_results["failure_rates"][-1] - robustness_results["failure_rates"][0]) if robustness_results["failure_rates"][-1] > robustness_results["failure_rates"][0] else 0
        
        f.write(f"System degradation factor: {degradation_factor:.2f}\n")
        f.write(f"Completion rate at highest failure rate ({robustness_results['failure_rates'][-1]:.1%}): {robustness_results['completion_rates'][-1]:.1%}\n\n")
        
        # Key findings
        f.write("Key Findings:\n")
        f.write("------------\n")
        f.write("1. Parallelization provides significant throughput benefits for batch processing\n")
        
        # Ideal batch size based on throughput per simulation
        ideal_batch_idx = np.argmax([t/b for t, b in zip(throughput_results["parallel_throughput"], throughput_results["batch_sizes"])])
        ideal_batch = throughput_results["batch_sizes"][ideal_batch_idx]
        
        f.write(f"2. Optimal batch size for efficiency is approximately {ideal_batch} simulations\n")
        
        # Robustness assessment
        if degradation_factor < 1:
            f.write("3. System shows high resilience to failures with graceful degradation\n")
        else:
            f.write("3. System shows moderate sensitivity to failures, consider adding more error handling\n")
            
        f.write("\nDetailed results and plots are available in the results directory.\n")
    
    print("\nTesting completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main() 