#!/usr/bin/env python3
"""
Framework Comparison Example: Qiskit vs PennyLane
==================================================

This demonstrates the same quantum algorithm implemented in both Qiskit and PennyLane
for direct performance and usability comparison.

For Independent Study: "Comparative Analysis of Quantum Computing Frameworks"
"""

import time
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Framework imports
try:
    import qiskit
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available")
except ImportError:
    QISKIT_AVAILABLE = False
    print("❌ Qiskit not available")

try:
    import pennylane as qml
    import pennylane.numpy as pnp
    PENNYLANE_AVAILABLE = True
    print("✅ PennyLane available")
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("❌ PennyLane not available")

@dataclass
class FrameworkResult:
    """Results from running algorithm in a specific framework"""
    framework: str
    algorithm: str
    execution_time: float
    result: Any
    circuit_depth: int
    lines_of_code: int
    success: bool
    error_message: str = None

class QuantumFrameworkComparator:
    """Compare quantum algorithms across Qiskit and PennyLane"""
    
    def __init__(self):
        self.qiskit_available = QISKIT_AVAILABLE
        self.pennylane_available = PENNYLANE_AVAILABLE
        self.results = {}
        
    def bell_state_qiskit(self) -> FrameworkResult:
        """Create Bell state using Qiskit"""
        if not self.qiskit_available:
            return FrameworkResult("qiskit", "bell_state", 0, None, 0, 0, False, "Qiskit not available")
            
        start_time = time.time()
        
        try:
            # Create quantum circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)              # Hadamard on qubit 0
            qc.cx(0, 1)          # CNOT between qubits 0 and 1
            qc.measure_all()     # Measure all qubits
            
            # Execute on simulator
            simulator = AerSimulator()
            sampler = Sampler()
            job = sampler.run(qc, shots=1024)
            result = job.result()
            
            execution_time = time.time() - start_time
            
            return FrameworkResult(
                framework="qiskit",
                algorithm="bell_state",
                execution_time=execution_time,
                result=result,
                circuit_depth=qc.depth(),
                lines_of_code=8,  # Approximate lines of algorithm code
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return FrameworkResult(
                framework="qiskit",
                algorithm="bell_state", 
                execution_time=execution_time,
                result=None,
                circuit_depth=0,
                lines_of_code=8,
                success=False,
                error_message=str(e)
            )
    
    def bell_state_pennylane(self) -> FrameworkResult:
        """Create Bell state using PennyLane"""
        if not self.pennylane_available:
            return FrameworkResult("pennylane", "bell_state", 0, None, 0, 0, False, "PennyLane not available")
            
        start_time = time.time()
        
        try:
            # Create device
            dev = qml.device('default.qubit', wires=2, shots=1024)
            
            @qml.qnode(dev)
            def bell_circuit():
                qml.Hadamard(wires=0)       # Hadamard on qubit 0
                qml.CNOT(wires=[0, 1])      # CNOT between qubits 0 and 1
                return qml.sample(wires=[0, 1])
            
            # Execute circuit
            result = bell_circuit()
            
            execution_time = time.time() - start_time
            
            return FrameworkResult(
                framework="pennylane",
                algorithm="bell_state",
                execution_time=execution_time,
                result=result,
                circuit_depth=2,  # H + CNOT
                lines_of_code=5,  # Approximate lines of algorithm code
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return FrameworkResult(
                framework="pennylane",
                algorithm="bell_state",
                execution_time=execution_time,
                result=None,
                circuit_depth=2,
                lines_of_code=5,
                success=False,
                error_message=str(e)
            )
    
    def grover_search_qiskit(self, search_space_size: int, target: int) -> FrameworkResult:
        """Grover's algorithm using Qiskit"""
        if not self.qiskit_available:
            return FrameworkResult("qiskit", "grover", 0, None, 0, 0, False, "Qiskit not available")
            
        start_time = time.time()
        
        try:
            n_qubits = int(np.ceil(np.log2(search_space_size)))
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Initialize superposition
            qc.h(range(n_qubits))
            
            # Oracle (flip target state)
            if target < 2**n_qubits:
                # Simple oracle - flip target state
                binary_target = format(target, f'0{n_qubits}b')
                for i, bit in enumerate(binary_target):
                    if bit == '0':
                        qc.x(i)
                qc.mcz(list(range(n_qubits-1)), n_qubits-1)
                for i, bit in enumerate(binary_target):
                    if bit == '0':
                        qc.x(i)
            
            # Diffusion operator
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            qc.mcz(list(range(n_qubits-1)), n_qubits-1)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))
            
            qc.measure_all()
            
            # Execute
            simulator = AerSimulator()
            sampler = Sampler()
            job = sampler.run(qc, shots=1024)
            result = job.result()
            
            execution_time = time.time() - start_time
            
            return FrameworkResult(
                framework="qiskit",
                algorithm="grover",
                execution_time=execution_time,
                result=result,
                circuit_depth=qc.depth(),
                lines_of_code=25,  # Approximate
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return FrameworkResult(
                framework="qiskit",
                algorithm="grover",
                execution_time=execution_time,
                result=None,
                circuit_depth=0,
                lines_of_code=25,
                success=False,
                error_message=str(e)
            )
    
    def grover_search_pennylane(self, search_space_size: int, target: int) -> FrameworkResult:
        """Grover's algorithm using PennyLane"""
        if not self.pennylane_available:
            return FrameworkResult("pennylane", "grover", 0, None, 0, 0, False, "PennyLane not available")
            
        start_time = time.time()
        
        try:
            n_qubits = int(np.ceil(np.log2(search_space_size)))
            dev = qml.device('default.qubit', wires=n_qubits, shots=1024)
            
            def oracle(target_state):
                """Oracle that flips the target state"""
                binary_target = format(target_state, f'0{n_qubits}b')
                for i, bit in enumerate(binary_target):
                    if bit == '0':
                        qml.PauliX(wires=i)
                qml.MultiControlledX(wires=list(range(n_qubits)), control_values=[1]*n_qubits)
                for i, bit in enumerate(binary_target):
                    if bit == '0':
                        qml.PauliX(wires=i)
            
            def diffusion():
                """Diffusion operator"""
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.PauliX(wires=i)
                qml.MultiControlledX(wires=list(range(n_qubits)), control_values=[1]*n_qubits)
                for i in range(n_qubits):
                    qml.PauliX(wires=i)
                    qml.Hadamard(wires=i)
            
            @qml.qnode(dev)
            def grover_circuit():
                # Initialize superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Apply oracle and diffusion
                oracle(target)
                diffusion()
                
                return qml.sample(wires=list(range(n_qubits)))
            
            result = grover_circuit()
            execution_time = time.time() - start_time
            
            return FrameworkResult(
                framework="pennylane",
                algorithm="grover",
                execution_time=execution_time,
                result=result,
                circuit_depth=10,  # Approximate
                lines_of_code=20,  # Approximate
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return FrameworkResult(
                framework="pennylane",
                algorithm="grover",
                execution_time=execution_time,
                result=None,
                circuit_depth=10,
                lines_of_code=20,
                success=False,
                error_message=str(e)
            )
    
    def compare_frameworks(self) -> Dict[str, Any]:
        """Run comprehensive framework comparison"""
        
        print("\n🔬 Quantum Framework Comparison Study")
        print("=" * 50)
        
        # Test Bell State creation
        print("\n1. Bell State Creation:")
        qiskit_bell = self.bell_state_qiskit()
        pennylane_bell = self.bell_state_pennylane()
        
        print(f"   Qiskit:    {qiskit_bell.execution_time:.4f}s - {'✅ Success' if qiskit_bell.success else '❌ Failed'}")
        print(f"   PennyLane: {pennylane_bell.execution_time:.4f}s - {'✅ Success' if pennylane_bell.success else '❌ Failed'}")
        
        # Test Grover's Algorithm
        print("\n2. Grover's Search (4-item space, target=2):")
        qiskit_grover = self.grover_search_qiskit(4, 2)
        pennylane_grover = self.grover_search_pennylane(4, 2)
        
        print(f"   Qiskit:    {qiskit_grover.execution_time:.4f}s - {'✅ Success' if qiskit_grover.success else '❌ Failed'}")
        print(f"   PennyLane: {pennylane_grover.execution_time:.4f}s - {'✅ Success' if pennylane_grover.success else '❌ Failed'}")
        
        # Summary comparison
        print("\n📊 Framework Comparison Summary:")
        print("-" * 40)
        
        if qiskit_bell.success and pennylane_bell.success:
            speed_ratio = qiskit_bell.execution_time / pennylane_bell.execution_time
            print(f"Bell State Speed Ratio (Qiskit/PennyLane): {speed_ratio:.2f}x")
        
        if qiskit_grover.success and pennylane_grover.success:
            speed_ratio = qiskit_grover.execution_time / pennylane_grover.execution_time
            print(f"Grover Speed Ratio (Qiskit/PennyLane): {speed_ratio:.2f}x")
        
        # Code complexity comparison
        print(f"\nCode Complexity (Lines of Code):")
        print(f"   Bell State  - Qiskit: {qiskit_bell.lines_of_code}, PennyLane: {pennylane_bell.lines_of_code}")
        print(f"   Grover      - Qiskit: {qiskit_grover.lines_of_code}, PennyLane: {pennylane_grover.lines_of_code}")
        
        return {
            'bell_state': {
                'qiskit': qiskit_bell,
                'pennylane': pennylane_bell
            },
            'grover': {
                'qiskit': qiskit_grover,
                'pennylane': pennylane_grover
            }
        }

def main():
    """Run framework comparison demonstration"""
    print("🌌 Quantum Framework Comparison for Independent Study")
    print("Focus: Qiskit vs PennyLane Performance and Usability Analysis")
    print("=" * 60)
    
    comparator = QuantumFrameworkComparator()
    results = comparator.compare_frameworks()
    
    print("\n✅ Framework comparison complete!")
    print("\nNext steps for Independent Study:")
    print("1. Implement all 4 of your quantum algorithms in both frameworks")
    print("2. Create statistical analysis of performance differences")
    print("3. Analyze developer experience and API usability")
    print("4. Write research paper on framework comparison findings")

if __name__ == "__main__":
    main()
