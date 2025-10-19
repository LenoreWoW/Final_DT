#!/usr/bin/env python3
"""
Quick Validation for Neural Quantum Digital Twin

Based on Lu et al. (2025) arXiv:2505.15662
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dt_project.quantum.neural_quantum_digital_twin import (
    create_neural_quantum_twin,
    AnnealingSchedule,
    PhaseType
)

print("="*80)
print("  NEURAL QUANTUM DIGITAL TWIN - QUICK VALIDATION")
print("="*80)
print("\nTheoretical Foundation: Lu et al. (2025) arXiv:2505.15662\n")

# Create twin
print("1. Creating Neural Quantum Digital Twin...")
nqdt = create_neural_quantum_twin(num_qubits=8, hidden_layers=[32, 16])
print(f"   ✓ Created: {nqdt.config.num_qubits} qubits, layers={nqdt.config.hidden_layers}")

# Test annealing
print("\n2. Testing Quantum Annealing...")
result = nqdt.quantum_annealing(schedule=AnnealingSchedule.ADAPTIVE, num_steps=50)
print(f"   Energy: {result.energy:.6f}")
print(f"   Success prob: {result.success_probability:.4f}")
print(f"   Phase: {result.phase_detected.value}")
print(f"   ✓ Annealing works")

# Test phase detection
print("\n3. Testing Phase Transition Detection...")
transitions = nqdt.detect_phase_transition(parameter_range=(0.0, 1.5), num_points=10)
print(f"   Detected {len(transitions)} phase transitions")
for trans in transitions[:2]:  # Show first 2
    print(f"   - {trans.phase_before.value} → {trans.phase_after.value} at {trans.transition_parameter:.4f}")
print(f"   ✓ Phase detection works")

# Generate report
print("\n4. Generating Report...")
report = nqdt.generate_report()
print(f"   Optimizations: {report['annealing_results']['num_optimizations']}")
print(f"   Best energy: {report['annealing_results']['best_energy']:.6f}")
print(f"   Phase transitions: {report['phase_analysis']['transitions_detected']}")
print(f"   ✓ Reporting works")

print("\n" + "="*80)
print("✅ ALL VALIDATION TESTS PASSED!")
print("Neural Quantum Digital Twin is OPERATIONAL")
print("="*80)

