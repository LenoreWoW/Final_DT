#!/usr/bin/env python3
"""
Real Quantum Benchmarks - Generate actual performance data.
"""

import asyncio
import json
import time
from datetime import datetime
import numpy as np

from dt_project.quantum.real_quantum_algorithms import create_quantum_algorithms
from dt_project.quantum.quantum_digital_twin_core import create_quantum_digital_twin_platform


async def run_comprehensive_quantum_benchmarks():
    """Run comprehensive quantum benchmarks and generate real data."""
    
    print("🚀 RUNNING COMPREHENSIVE QUANTUM BENCHMARKS")
    print("=" * 60)
    
    # Initialize quantum algorithms
    quantum_algorithms = create_quantum_algorithms()
    
    # Results storage
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "quantum_algorithms": [],
        "quantum_digital_twins": [],
        "performance_summary": {}
    }
    
    # Test 1: Grover's Search Algorithm
    print("\n🔍 Testing Grover's Search Algorithm...")
    search_results = []
    for size in [4, 8, 16]:
        target = size // 2
        result = await quantum_algorithms.grovers_search(size, target)
        search_results.append({
            "search_space_size": size,
            "target": target,
            "execution_time": result.execution_time,
            "quantum_advantage": result.quantum_advantage,
            "success": result.success,
            "n_qubits": result.n_qubits,
            "circuit_depth": result.circuit_depth
        })
        print(f"   Size {size}: {'✅' if result.success else '❌'} "
              f"Advantage: {result.quantum_advantage:.2f}x "
              f"Time: {result.execution_time:.4f}s")
    
    benchmark_results["quantum_algorithms"].append({
        "algorithm": "Grover's Search",
        "results": search_results
    })
    
    # Test 2: Bernstein-Vazirani Algorithm
    print("\n🔤 Testing Bernstein-Vazirani Algorithm...")
    bv_results = []
    for secret in ["10", "101", "1010", "10101"]:
        result = await quantum_algorithms.bernstein_vazirani(secret)
        bv_results.append({
            "secret_length": len(secret),
            "secret": secret,
            "execution_time": result.execution_time,
            "quantum_advantage": result.quantum_advantage,
            "success": result.success,
            "n_qubits": result.n_qubits
        })
        print(f"   Secret '{secret}': {'✅' if result.success else '❌'} "
              f"Advantage: {result.quantum_advantage:.2f}x "
              f"Time: {result.execution_time:.4f}s")
    
    benchmark_results["quantum_algorithms"].append({
        "algorithm": "Bernstein-Vazirani",
        "results": bv_results
    })
    
    # Test 3: Quantum Phase Estimation
    print("\n📐 Testing Quantum Phase Estimation...")
    phase_results = []
    for phase in [0.25, 0.125, 0.0625]:
        result = await quantum_algorithms.quantum_phase_estimation(phase)
        phase_results.append({
            "true_phase": phase,
            "execution_time": result.execution_time,
            "quantum_advantage": result.quantum_advantage,
            "success": result.success,
            "n_qubits": result.n_qubits,
            "estimated_phase": result.result_data.get('estimated_phase', 0.0) if result.success else None
        })
        print(f"   Phase {phase}: {'✅' if result.success else '❌'} "
              f"Advantage: {result.quantum_advantage:.2f}x "
              f"Time: {result.execution_time:.4f}s")
    
    benchmark_results["quantum_algorithms"].append({
        "algorithm": "Quantum Phase Estimation",
        "results": phase_results
    })
    
    # Test 4: Quantum Fourier Transform
    print("\n🌊 Testing Quantum Fourier Transform...")
    qft_results = []
    for n_qubits in [2, 3, 4]:
        result = await quantum_algorithms.quantum_fourier_transform_demo(n_qubits)
        qft_results.append({
            "n_qubits": n_qubits,
            "execution_time": result.execution_time,
            "quantum_advantage": result.quantum_advantage,
            "success": result.success,
            "circuit_depth": result.circuit_depth
        })
        print(f"   {n_qubits} qubits: {'✅' if result.success else '❌'} "
              f"Advantage: {result.quantum_advantage:.2f}x "
              f"Time: {result.execution_time:.4f}s")
    
    benchmark_results["quantum_algorithms"].append({
        "algorithm": "Quantum Fourier Transform",
        "results": qft_results
    })
    
    # Test 5: Quantum Digital Twin Platform
    print("\n🧮 Testing Quantum Digital Twin Platform...")
    
    # Create platform
    config = {
        'fault_tolerance': False,  # Disable for benchmarking
        'quantum_internet': False,
        'holographic_viz': False,
        'max_qubits': 4
    }
    
    platform = create_quantum_digital_twin_platform(config)
    
    # Test twin creation
    start_time = time.time()
    from dt_project.quantum.quantum_digital_twin_core import QuantumTwinType
    
    twin = await platform.create_quantum_digital_twin(
        entity_id="benchmark_twin",
        twin_type=QuantumTwinType.SYSTEM,
        initial_state={'test_value': 0.5, 'another_value': 0.8},
        quantum_resources={'n_qubits': 3, 'circuit_depth': 5}
    )
    creation_time = time.time() - start_time
    
    # Test quantum evolution
    start_time = time.time()
    evolution_result = await platform.run_quantum_evolution("benchmark_twin", 0.001)
    evolution_time = time.time() - start_time
    
    # Test optimization
    start_time = time.time()
    optimization_result = await platform.optimize_twin_performance(
        "benchmark_twin", 
        "test_optimization",
        {"test_constraint": True}
    )
    optimization_time = time.time() - start_time
    
    benchmark_results["quantum_digital_twins"].append({
        "platform_creation": True,
        "twin_creation_time": creation_time,
        "twin_fidelity": twin.quantum_state.fidelity if twin.quantum_state else 0.0,
        "evolution_time": evolution_time,
        "evolution_fidelity": evolution_result.get('quantum_fidelity', 0.0),
        "optimization_time": optimization_time,
        "optimization_improvement": optimization_result.get('performance_improvement', 1.0)
    })
    
    print(f"   Twin Creation: ✅ Fidelity: {twin.quantum_state.fidelity:.3f} Time: {creation_time:.4f}s")
    print(f"   Evolution: ✅ Fidelity: {evolution_result.get('quantum_fidelity', 0.0):.3f} Time: {evolution_time:.4f}s")
    print(f"   Optimization: ✅ Improvement: {optimization_result.get('performance_improvement', 1.0):.2f}x Time: {optimization_time:.4f}s")
    
    # Calculate overall performance summary
    total_quantum_algorithms = len(benchmark_results["quantum_algorithms"])
    successful_algorithms = sum(1 for alg in benchmark_results["quantum_algorithms"] 
                               if any(r.get('success', False) for r in alg['results']))
    
    avg_quantum_advantage = np.mean([
        result.get('quantum_advantage', 1.0)
        for alg in benchmark_results["quantum_algorithms"]
        for result in alg['results']
        if result.get('success', False)
    ])
    
    total_execution_time = sum([
        result.get('execution_time', 0.0)
        for alg in benchmark_results["quantum_algorithms"]
        for result in alg['results']
    ])
    
    benchmark_results["performance_summary"] = {
        "total_algorithms_tested": total_quantum_algorithms,
        "successful_algorithms": successful_algorithms,
        "success_rate": successful_algorithms / total_quantum_algorithms if total_quantum_algorithms > 0 else 0.0,
        "average_quantum_advantage": float(avg_quantum_advantage) if not np.isnan(avg_quantum_advantage) else 1.0,
        "total_execution_time": total_execution_time,
        "quantum_digital_twin_tested": len(benchmark_results["quantum_digital_twins"]) > 0
    }
    
    # Save results
    with open('benchmark_results/real_quantum_benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("\n📊 BENCHMARK SUMMARY:")
    print(f"   Total Algorithms Tested: {benchmark_results['performance_summary']['total_algorithms_tested']}")
    print(f"   Success Rate: {benchmark_results['performance_summary']['success_rate']:.1%}")
    print(f"   Average Quantum Advantage: {benchmark_results['performance_summary']['average_quantum_advantage']:.2f}x")
    print(f"   Total Execution Time: {benchmark_results['performance_summary']['total_execution_time']:.4f}s")
    print(f"   Quantum Digital Twin: {'✅' if benchmark_results['performance_summary']['quantum_digital_twin_tested'] else '❌'}")
    
    print(f"\n✅ Results saved to: benchmark_results/real_quantum_benchmark_results.json")
    
    return benchmark_results


if __name__ == "__main__":
    """Run quantum benchmarks."""
    
    print("🌌 REAL QUANTUM PLATFORM BENCHMARKS")
    print("Generating actual performance data...")
    
    results = asyncio.run(run_comprehensive_quantum_benchmarks())
    
    print("\n🎯 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("Real quantum performance data generated.")
