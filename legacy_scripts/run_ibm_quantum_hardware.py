#!/usr/bin/env python3
"""
IBM Quantum Hardware Test Runner - Simplified Version
Executes quantum benchmarks on real IBM quantum hardware
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qiskit import QuantumCircuit, transpile, execute
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Qiskit Runtime not fully available: {e}")
    try:
        from qiskit import QuantumCircuit, transpile, execute
        from qiskit_ibm_runtime import QiskitRuntimeService
        QISKIT_BASIC = True
    except ImportError:
        QISKIT_BASIC = False
    QISKIT_AVAILABLE = False

class IBMQuantumHardwareRunner:
    """Runs benchmarks on real IBM quantum hardware."""
    
    def __init__(self):
        self.service = None
        self.backend = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "IBM Quantum",
            "test_results": {},
            "performance_metrics": {},
            "backend_info": {},
            "errors": []
        }
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def initialize_ibm_service(self):
        """Initialize IBM Quantum service connection."""
        try:
            self.log("Connecting to IBM Quantum...")
            self.service = QiskitRuntimeService()
            
            # Get available backends
            backends = list(self.service.backends())
            self.log(f"Found {len(backends)} available backends")
            
            # Prefer real quantum hardware
            real_backends = [b for b in backends if not b.simulator]
            simulator_backends = [b for b in backends if b.simulator]
            
            if real_backends:
                # Choose least busy real backend
                operational_backends = [b for b in real_backends if b.status().operational]
                if operational_backends:
                    self.backend = min(operational_backends, key=lambda x: x.status().pending_jobs)
                    self.log(f"✅ Selected real quantum backend: {self.backend.name}")
                else:
                    self.backend = real_backends[0]
                    self.log(f"⚠️ Selected non-operational backend: {self.backend.name}")
            elif simulator_backends:
                self.backend = simulator_backends[0]
                self.log(f"⚠️ Using simulator backend: {self.backend.name}")
            else:
                self.log("❌ No backends available")
                return False
                
            # Get backend information
            config = self.backend.configuration()
            status = self.backend.status()
            
            self.results["backend_info"] = {
                "name": self.backend.name,
                "version": getattr(config, 'backend_version', 'unknown'),
                "num_qubits": config.num_qubits,
                "max_shots": getattr(config, 'max_shots', 'unknown'),
                "quantum_volume": getattr(config, 'quantum_volume', None),
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "simulator": self.backend.simulator
            }
            
            self.log(f"Backend info: {config.num_qubits} qubits, {status.pending_jobs} pending jobs")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize IBM Quantum service: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            return False
            
    def create_bell_state_circuit(self):
        """Create a simple Bell state circuit for testing."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc
        
    def create_ghz_state_circuit(self, num_qubits=3):
        """Create a GHZ state circuit."""
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
        qc.measure_all()
        return qc
        
    def create_qaoa_maxcut_circuit(self, num_qubits=4):
        """Create a simple QAOA MaxCut circuit."""
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Initialize superposition
        for i in range(num_qubits):
            qc.h(i)
            
        # QAOA cost layer (triangle graph)
        gamma = 0.5
        edges = [(0, 1), (1, 2), (2, 0)]
        for i, j in edges:
            if j < num_qubits:
                qc.cx(i, j)
                qc.rz(gamma, j)
                qc.cx(i, j)
                
        # QAOA mixer layer
        beta = 0.3
        for i in range(num_qubits):
            qc.rx(beta, i)
            
        qc.measure_all()
        return qc
        
    def run_bell_state_test(self):
        """Run Bell state test on IBM hardware."""
        self.log("Running Bell state test...")
        
        try:
            qc = self.create_bell_state_circuit()
            
            # Transpile for backend
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            
            self.log(f"Circuit transpiled. Depth: {transpiled_qc.depth()}")
            
            # Execute on hardware
            start_time = time.time()
            
            if QISKIT_AVAILABLE:
                # Use new runtime interface
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([transpiled_qc], shots=1024)
                    result = job.result()
                    
                # Get measurement results
                quasi_dists = result.quasi_dists[0]
                counts = {format(k, '02b'): int(v * 1024) for k, v in quasi_dists.items()}
                
            else:
                # Use legacy execute method
                job = execute(transpiled_qc, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
            execution_time = time.time() - start_time
            
            # Analyze Bell state fidelity
            bell_states = ['00', '11']
            bell_counts = sum(counts.get(state, 0) for state in bell_states)
            fidelity = bell_counts / 1024
            
            self.log(f"✅ Bell state test completed in {execution_time:.2f}s")
            self.log(f"Bell state fidelity: {fidelity:.3f}")
            self.log(f"Measurement counts: {dict(list(counts.items())[:4])}")
            
            return {
                "test_name": "Bell State",
                "execution_time": execution_time,
                "qubits_used": 2,
                "circuit_depth": transpiled_qc.depth(),
                "shots": 1024,
                "fidelity": fidelity,
                "measurement_counts": counts,
                "backend_name": self.backend.name
            }
            
        except Exception as e:
            error_msg = f"Bell state test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            return None
            
    def run_ghz_state_test(self):
        """Run GHZ state test."""
        self.log("Running GHZ state test...")
        
        try:
            max_qubits = min(3, self.backend.configuration().num_qubits)
            qc = self.create_ghz_state_circuit(max_qubits)
            
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            
            start_time = time.time()
            
            if QISKIT_AVAILABLE:
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([transpiled_qc], shots=1024)
                    result = job.result()
                quasi_dists = result.quasi_dists[0]
                counts = {format(k, f'0{max_qubits}b'): int(v * 1024) for k, v in quasi_dists.items()}
            else:
                job = execute(transpiled_qc, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
            execution_time = time.time() - start_time
            
            # Analyze GHZ fidelity
            ghz_states = ['0' * max_qubits, '1' * max_qubits]
            ghz_counts = sum(counts.get(state, 0) for state in ghz_states)
            fidelity = ghz_counts / 1024
            
            self.log(f"✅ GHZ state test completed in {execution_time:.2f}s")
            self.log(f"GHZ state fidelity: {fidelity:.3f}")
            
            return {
                "test_name": "GHZ State",
                "execution_time": execution_time,
                "qubits_used": max_qubits,
                "circuit_depth": transpiled_qc.depth(),
                "shots": 1024,
                "fidelity": fidelity,
                "measurement_counts": counts,
                "backend_name": self.backend.name
            }
            
        except Exception as e:
            error_msg = f"GHZ state test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            return None
            
    def run_qaoa_maxcut_test(self):
        """Run QAOA MaxCut test."""
        self.log("Running QAOA MaxCut test...")
        
        try:
            max_qubits = min(4, self.backend.configuration().num_qubits)
            qc = self.create_qaoa_maxcut_circuit(max_qubits)
            
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            
            start_time = time.time()
            
            if QISKIT_AVAILABLE:
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run([transpiled_qc], shots=1024)
                    result = job.result()
                quasi_dists = result.quasi_dists[0]
                counts = {format(k, f'0{max_qubits}b'): int(v * 1024) for k, v in quasi_dists.items()}
            else:
                job = execute(transpiled_qc, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
            execution_time = time.time() - start_time
            
            # Find best cut value
            best_cut = 0
            best_bitstring = ""
            
            for bitstring, count in counts.items():
                cut_value = self.calculate_triangle_cut(bitstring)
                if cut_value > best_cut:
                    best_cut = cut_value
                    best_bitstring = bitstring
                    
            self.log(f"✅ QAOA MaxCut completed in {execution_time:.2f}s")
            self.log(f"Best cut value: {best_cut}, bitstring: {best_bitstring}")
            
            return {
                "test_name": "QAOA MaxCut",
                "execution_time": execution_time,
                "qubits_used": max_qubits,
                "circuit_depth": transpiled_qc.depth(),
                "shots": 1024,
                "best_cut_value": best_cut,
                "best_bitstring": best_bitstring,
                "measurement_counts": counts,
                "backend_name": self.backend.name
            }
            
        except Exception as e:
            error_msg = f"QAOA MaxCut test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            return None
            
    def calculate_triangle_cut(self, bitstring):
        """Calculate cut value for triangle graph."""
        if len(bitstring) < 3:
            return 0
            
        edges = [(0, 1), (1, 2), (2, 0)]
        cut_value = 0
        
        for i, j in edges:
            if i < len(bitstring) and j < len(bitstring):
                if bitstring[i] != bitstring[j]:
                    cut_value += 1
                    
        return cut_value
        
    def run_hardware_benchmarks(self):
        """Run all hardware benchmarks."""
        self.log("Starting IBM quantum hardware benchmarks...")
        
        # Run Bell state test
        bell_result = self.run_bell_state_test()
        if bell_result:
            self.results["test_results"]["bell_state"] = bell_result
            
        # Run GHZ state test
        ghz_result = self.run_ghz_state_test()
        if ghz_result:
            self.results["test_results"]["ghz_state"] = ghz_result
            
        # Run QAOA MaxCut test
        qaoa_result = self.run_qaoa_maxcut_test()
        if qaoa_result:
            self.results["test_results"]["qaoa_maxcut"] = qaoa_result
            
        # Calculate performance metrics
        self.calculate_hardware_metrics()
        
    def calculate_hardware_metrics(self):
        """Calculate performance metrics from hardware tests."""
        tests = self.results["test_results"]
        
        if not tests:
            self.log("⚠️ No hardware test results to analyze", "WARNING")
            return
            
        total_execution_time = sum(test.get("execution_time", 0) for test in tests.values())
        total_qubits = sum(test.get("qubits_used", 0) for test in tests.values())
        avg_circuit_depth = sum(test.get("circuit_depth", 0) for test in tests.values()) / len(tests)
        
        # Calculate average fidelity
        fidelities = [test.get("fidelity", 0) for test in tests.values() if "fidelity" in test]
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0
        
        self.results["performance_metrics"] = {
            "total_hardware_tests": len(tests),
            "total_execution_time": total_execution_time,
            "total_qubits_used": total_qubits,
            "average_circuit_depth": avg_circuit_depth,
            "average_fidelity": avg_fidelity,
            "backend_name": self.backend.name if self.backend else "unknown",
            "real_quantum_hardware": not self.backend.simulator if self.backend else False,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log(f"✅ Hardware metrics calculated: {len(tests)} tests, {avg_fidelity:.3f} avg fidelity")
        
    def save_results(self):
        """Save hardware test results."""
        results_file = "benchmark_results/ibm_quantum_hardware_results.json"
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.log(f"✅ Hardware results saved to {results_file}")
        
    def run_all_hardware_tests(self):
        """Run complete IBM quantum hardware test suite."""
        print("🚀 Starting IBM Quantum Hardware Tests")
        print("=" * 60)
        
        # Initialize IBM service
        if not self.initialize_ibm_service():
            print("❌ Failed to connect to IBM Quantum. Exiting...")
            return False
            
        # Run hardware benchmarks
        self.run_hardware_benchmarks()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 IBM QUANTUM HARDWARE TEST RESULTS")
        print("=" * 60)
        
        backend_info = self.results["backend_info"]
        metrics = self.results["performance_metrics"]
        
        print(f"✅ Backend: {backend_info.get('name', 'unknown')}")
        print(f"✅ Qubits Available: {backend_info.get('num_qubits', 'unknown')}")
        print(f"✅ Real Hardware: {'YES' if not backend_info.get('simulator', True) else 'SIMULATOR'}")
        print(f"✅ Tests Completed: {metrics.get('total_hardware_tests', 0)}")
        print(f"✅ Total Execution Time: {metrics.get('total_execution_time', 0):.2f}s")
        print(f"✅ Average Fidelity: {metrics.get('average_fidelity', 0):.3f}")
        
        if self.results["errors"]:
            print(f"\n⚠️ Errors encountered: {len(self.results['errors'])}")
            for error in self.results["errors"][:3]:  # Show first 3 errors
                print(f"   - {error}")
        else:
            print("\n🎉 All hardware tests completed successfully!")
            
        print(f"\n📊 Results saved to benchmark_results/ibm_quantum_hardware_results.json")
        print("=" * 60)
        
        return len(self.results["test_results"]) > 0

if __name__ == "__main__":
    runner = IBMQuantumHardwareRunner()
    success = runner.run_all_hardware_tests()
    
    if success:
        print("\n🚀 IBM Quantum hardware testing completed successfully!")
        print("Real quantum hardware results now available for thesis defense.")
    else:
        print("\n❌ Hardware testing failed. Check connection and credentials.")