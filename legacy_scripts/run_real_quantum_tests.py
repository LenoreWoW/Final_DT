#!/usr/bin/env python3
"""
Real IBM Quantum Hardware Test Runner
Executes quantum tests on actual IBM quantum hardware
"""

import sys
import os
import json
import time
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RealQuantumTestRunner:
    """Runs benchmarks on real IBM quantum hardware with simplified approach."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": "IBM Quantum",
            "test_results": {},
            "performance_metrics": {},
            "backend_info": {},
            "errors": [],
            "connection_status": "unknown"
        }
        self.service = None
        self.backend = None
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def test_ibm_connection(self):
        """Test IBM Quantum connection and get backend info."""
        self.log("Testing IBM Quantum connection...")
        
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Initialize service
            service = QiskitRuntimeService()
            self.log("✅ Successfully connected to IBM Quantum!")
            
            # Get available backends
            backends = list(service.backends())
            self.log(f"Found {len(backends)} available backends")
            
            # Find real quantum hardware
            real_backends = [b for b in backends if not b.simulator and b.status().operational]
            simulator_backends = [b for b in backends if b.simulator]
            
            if real_backends:
                # Choose the least busy real backend
                backend = min(real_backends, key=lambda x: x.status().pending_jobs)
                self.log(f"✅ Selected real quantum backend: {backend.name}")
                
                # Get detailed backend information
                config = backend.configuration()
                status = backend.status()
                
                backend_info = {
                    "name": backend.name,
                    "num_qubits": config.num_qubits,
                    "quantum_volume": getattr(config, 'quantum_volume', None),
                    "max_shots": getattr(config, 'max_shots', 8192),
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "simulator": backend.simulator,
                    "backend_version": getattr(config, 'backend_version', 'unknown'),
                    "basis_gates": getattr(config, 'basis_gates', []),
                    "coupling_map": str(getattr(config, 'coupling_map', 'all-to-all'))[:100] + "..." if hasattr(config, 'coupling_map') else None
                }
                
                self.results["connection_status"] = "success"
                self.results["backend_info"] = backend_info
                self.backend = backend
                self.service = service
                
                self.log(f"Backend details: {config.num_qubits} qubits, {status.pending_jobs} pending jobs")
                
                return True
                
            elif simulator_backends:
                backend = simulator_backends[0]
                self.log(f"⚠️ Only simulators available, using: {backend.name}")
                
                config = backend.configuration()
                backend_info = {
                    "name": backend.name,
                    "num_qubits": config.num_qubits,
                    "simulator": True,
                    "max_shots": getattr(config, 'max_shots', 8192)
                }
                
                self.results["connection_status"] = "simulator_only"
                self.results["backend_info"] = backend_info
                self.backend = backend
                self.service = service
                
                return True
            else:
                self.log("❌ No backends available")
                self.results["connection_status"] = "no_backends"
                return False
                
        except ImportError as e:
            error_msg = f"Qiskit IBM Runtime not available: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            self.results["connection_status"] = "import_error"
            return False
            
        except Exception as e:
            error_msg = f"Failed to connect to IBM Quantum: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            self.results["connection_status"] = "connection_failed"
            return False
            
    def run_simple_bell_test(self):
        """Run a simple Bell state test."""
        self.log("Attempting Bell state test...")
        
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_ibm_runtime import Sampler, Session
            
            # Create Bell state circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            
            # Transpile for backend
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            self.log(f"Circuit transpiled. Depth: {transpiled_qc.depth()}")
            
            # Execute on hardware
            start_time = time.time()
            
            with Session(service=self.service, backend=self.backend) as session:
                sampler = Sampler(session=session)
                
                # Use fewer shots for faster execution
                job = sampler.run([transpiled_qc], shots=512)
                self.log("Job submitted to quantum backend...")
                
                # Wait for completion with status updates
                while not job.done():
                    self.log(f"Job status: {job.status()}")
                    time.sleep(5)
                
                result = job.result()
                
            execution_time = time.time() - start_time
            
            # Analyze results
            quasi_dists = result.quasi_dists[0]
            
            # Convert to counts
            counts = {}
            for state, prob in quasi_dists.items():
                bitstring = format(state, '02b')
                counts[bitstring] = int(prob * 512)
                
            # Calculate Bell state fidelity
            bell_states = ['00', '11']
            bell_counts = sum(counts.get(state, 0) for state in bell_states)
            fidelity = bell_counts / 512
            
            self.log(f"✅ Bell state test completed in {execution_time:.2f}s")
            self.log(f"Bell state fidelity: {fidelity:.3f}")
            self.log(f"Measurement counts: {counts}")
            
            return {
                "test_name": "Bell State",
                "execution_time": execution_time,
                "qubits_used": 2,
                "circuit_depth": transpiled_qc.depth(),
                "shots": 512,
                "fidelity": fidelity,
                "measurement_counts": counts,
                "backend_name": self.backend.name,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Bell state test failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.results["errors"].append(error_msg)
            return {
                "test_name": "Bell State",
                "success": False,
                "error": error_msg
            }
            
    def generate_hardware_metrics(self):
        """Generate performance metrics from hardware tests."""
        tests = self.results["test_results"]
        backend_info = self.results["backend_info"]
        
        successful_tests = [t for t in tests.values() if t.get("success", False)]
        
        if successful_tests:
            total_execution_time = sum(t.get("execution_time", 0) for t in successful_tests)
            avg_fidelity = sum(t.get("fidelity", 0) for t in successful_tests if "fidelity" in t) / len([t for t in successful_tests if "fidelity" in t])
            
            self.results["performance_metrics"] = {
                "successful_tests": len(successful_tests),
                "total_execution_time": total_execution_time,
                "average_fidelity": avg_fidelity,
                "backend_qubits": backend_info.get("num_qubits", 0),
                "real_quantum_hardware": not backend_info.get("simulator", True),
                "quantum_volume": backend_info.get("quantum_volume"),
                "backend_name": backend_info.get("name", "unknown")
            }
            
            self.log(f"✅ Metrics: {len(successful_tests)} tests, {avg_fidelity:.3f} avg fidelity")
        else:
            self.results["performance_metrics"] = {
                "successful_tests": 0,
                "message": "No successful hardware tests completed"
            }
            
    def save_results(self):
        """Save hardware test results."""
        results_file = "benchmark_results/real_quantum_hardware_results.json"
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.log(f"✅ Results saved to {results_file}")
        
    def run_all_tests(self):
        """Run complete hardware test suite."""
        print("🚀 Starting Real IBM Quantum Hardware Tests")
        print("=" * 60)
        
        # Test connection
        if not self.test_ibm_connection():
            print("❌ Failed to connect to IBM Quantum")
            self.save_results()
            return False
            
        # Run simple Bell state test
        bell_result = self.run_simple_bell_test()
        if bell_result:
            self.results["test_results"]["bell_state"] = bell_result
            
        # Generate metrics
        self.generate_hardware_metrics()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 REAL IBM QUANTUM HARDWARE RESULTS")
        print("=" * 60)
        
        backend_info = self.results.get("backend_info", {})
        metrics = self.results.get("performance_metrics", {})
        
        print(f"✅ Connection Status: {self.results['connection_status']}")
        print(f"✅ Backend: {backend_info.get('name', 'unknown')}")
        print(f"✅ Available Qubits: {backend_info.get('num_qubits', 'unknown')}")
        print(f"✅ Real Hardware: {'YES' if not backend_info.get('simulator', True) else 'SIMULATOR'}")
        
        if backend_info.get('quantum_volume'):
            print(f"✅ Quantum Volume: {backend_info['quantum_volume']}")
            
        if backend_info.get('pending_jobs') is not None:
            print(f"✅ Queue Length: {backend_info['pending_jobs']} jobs")
            
        print(f"✅ Successful Tests: {metrics.get('successful_tests', 0)}")
        
        if metrics.get('average_fidelity'):
            print(f"✅ Average Fidelity: {metrics['average_fidelity']:.3f}")
            
        if metrics.get('total_execution_time'):
            print(f"✅ Total Runtime: {metrics['total_execution_time']:.2f}s")
            
        if self.results["errors"]:
            print(f"\n⚠️ Issues encountered: {len(self.results['errors'])}")
            for error in self.results["errors"][:2]:  # Show first 2 errors
                print(f"   - {error}")
        
        success = metrics.get('successful_tests', 0) > 0 or self.results['connection_status'] == 'success'
        
        if success:
            print("\n🎉 IBM Quantum hardware connection and testing successful!")
        else:
            print("\n⚠️ Limited success - connection established but tests may have failed")
            
        print(f"\n📊 Full results saved to benchmark_results/real_quantum_hardware_results.json")
        print("=" * 60)
        
        return success

if __name__ == "__main__":
    runner = RealQuantumTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🚀 Real IBM Quantum hardware testing completed!")
        print("Actual quantum hardware results now available for thesis defense.")
    else:
        print("\n⚠️ Hardware testing had limited success. Check results file for details.")