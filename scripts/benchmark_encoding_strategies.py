#!/usr/bin/env python3
"""
Benchmark and compare different quantum data encoding strategies for the Digital Twin project.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_project.quantum.encoders import AmplitudeEncoder, AngleEncoder, BasisEncoder
from dt_project.quantum.simulator import QuantumSimulator
from dt_project.utils.logging_utils import setup_logger

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger("encoding_benchmark", log_dir)

# Create output directory for results
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "results", "encoding_benchmarks")
os.makedirs(output_dir, exist_ok=True)

def generate_test_data(n_samples, dimensions):
    """Generate synthetic test data for benchmarking encoders."""
    return np.random.random((n_samples, dimensions))

def benchmark_encoder(encoder, data, n_circuits=10):
    """Benchmark an encoder with given data."""
    start_time = time.time()
    circuits = []
    
    # Encode data points into circuits
    for i in range(min(len(data), n_circuits)):
        circuit = encoder.encode(data[i])
        circuits.append(circuit)
    
    encoding_time = time.time() - start_time
    
    # Calculate encoding density (qubits used vs. classical data dimensions)
    if hasattr(encoder, 'n_qubits'):
        encoding_density = data.shape[1] / encoder.n_qubits
    else:
        encoding_density = 1.0  # Default if encoder doesn't specify qubit count
        
    return {
        'encoding_time': encoding_time,
        'avg_time_per_sample': encoding_time / min(len(data), n_circuits),
        'encoding_density': encoding_density,
        'n_qubits': getattr(encoder, 'n_qubits', 0)
    }

def run_encoding_benchmark():
    """Run benchmarks on different encoding strategies."""
    dimensions_to_test = [2, 4, 8, 16, 32]
    n_samples = 100
    
    encoders = {
        'Amplitude': AmplitudeEncoder,
        'Angle': AngleEncoder,
        'Basis': BasisEncoder,
    }
    
    results = {name: [] for name in encoders}
    
    for dim in dimensions_to_test:
        logger.info(f"Testing with {dim} dimensions")
        test_data = generate_test_data(n_samples, dim)
        
        for name, encoder_class in encoders.items():
            try:
                n_qubits = dim if name != 'Amplitude' else int(np.ceil(np.log2(dim)))
                encoder = encoder_class(n_qubits=n_qubits)
                
                logger.info(f"Benchmarking {name} encoder")
                benchmark_result = benchmark_encoder(encoder, test_data)
                benchmark_result['dimensions'] = dim
                results[name].append(benchmark_result)
                
                logger.info(f"  Time: {benchmark_result['encoding_time']:.4f}s, "
                           f"Density: {benchmark_result['encoding_density']:.4f}")
            except Exception as e:
                logger.error(f"Error benchmarking {name} encoder: {str(e)}")
    
    return results, dimensions_to_test

def plot_benchmark_results(results, dimensions):
    """Plot the benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Encoding time vs dimensions
    plt.subplot(2, 2, 1)
    for name, res_list in results.items():
        times = [r['encoding_time'] for r in res_list]
        plt.plot(dimensions, times, 'o-', label=name)
    
    plt.xlabel('Data Dimensions')
    plt.ylabel('Encoding Time (s)')
    plt.title('Encoding Time vs. Data Dimensions')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Encoding density vs dimensions
    plt.subplot(2, 2, 2)
    for name, res_list in results.items():
        density = [r['encoding_density'] for r in res_list]
        plt.plot(dimensions, density, 'o-', label=name)
    
    plt.xlabel('Data Dimensions')
    plt.ylabel('Encoding Density (dims/qubit)')
    plt.title('Encoding Efficiency vs. Data Dimensions')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Time per sample vs dimensions
    plt.subplot(2, 2, 3)
    for name, res_list in results.items():
        time_per_sample = [r['avg_time_per_sample'] for r in res_list]
        plt.plot(dimensions, time_per_sample, 'o-', label=name)
    
    plt.xlabel('Data Dimensions')
    plt.ylabel('Time per Sample (s)')
    plt.title('Time per Sample vs. Data Dimensions')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Number of qubits vs dimensions
    plt.subplot(2, 2, 4)
    for name, res_list in results.items():
        qubits = [r['n_qubits'] for r in res_list]
        plt.plot(dimensions, qubits, 'o-', label=name)
    
    plt.xlabel('Data Dimensions')
    plt.ylabel('Number of Qubits')
    plt.title('Qubit Requirements vs. Data Dimensions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"encoding_benchmark_{timestamp}.png"), dpi=300)
    logger.info(f"Saved benchmark plot to {os.path.join(output_dir, f'encoding_benchmark_{timestamp}.png')}")

def save_results_to_file(results, dimensions):
    """Save benchmark results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"encoding_benchmark_results_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("QUANTUM DATA ENCODING BENCHMARK RESULTS\n")
        f.write("======================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, res_list in results.items():
            f.write(f"{name} Encoder:\n")
            f.write("-" * (len(name) + 9) + "\n")
            
            for i, res in enumerate(res_list):
                f.write(f"  Dimensions: {dimensions[i]}\n")
                f.write(f"  Encoding Time: {res['encoding_time']:.6f} seconds\n")
                f.write(f"  Time per Sample: {res['avg_time_per_sample']:.6f} seconds\n")
                f.write(f"  Encoding Density: {res['encoding_density']:.4f} dims/qubit\n")
                f.write(f"  Qubits Required: {res['n_qubits']}\n\n")
            
        f.write("\nSUMMARY\n")
        f.write("=======\n\n")
        
        # Find best encoder for different metrics
        for dim_idx, dim in enumerate(dimensions):
            f.write(f"For {dim} dimensions:\n")
            
            # Fastest encoder
            fastest = min([(name, res_list[dim_idx]['encoding_time']) 
                          for name, res_list in results.items()], key=lambda x: x[1])
            f.write(f"  Fastest: {fastest[0]} ({fastest[1]:.6f}s)\n")
            
            # Most efficient encoder (highest encoding density)
            most_efficient = max([(name, res_list[dim_idx]['encoding_density']) 
                                 for name, res_list in results.items()], key=lambda x: x[1])
            f.write(f"  Most Efficient: {most_efficient[0]} ({most_efficient[1]:.4f} dims/qubit)\n")
            
            # Fewest qubits
            fewest_qubits = min([(name, res_list[dim_idx]['n_qubits']) 
                                for name, res_list in results.items()], key=lambda x: x[1])
            f.write(f"  Fewest Qubits: {fewest_qubits[0]} ({fewest_qubits[1]} qubits)\n\n")
    
    logger.info(f"Saved benchmark results to {output_file}")
    return output_file

def main():
    logger.info("Starting quantum data encoding benchmark")
    
    try:
        results, dimensions = run_encoding_benchmark()
        plot_benchmark_results(results, dimensions)
        output_file = save_results_to_file(results, dimensions)
        
        logger.info("Benchmark completed successfully")
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 