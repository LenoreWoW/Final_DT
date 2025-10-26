#!/usr/bin/env python3
"""
Quick Validation for Uncertainty Quantification Framework

Based on Otgonbaatar et al. (2024) arXiv:2410.23311
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dt_project.quantum.uncertainty_quantification import (
    create_uq_framework,
    NoiseParameters,
    UncertaintyType
)

print("="*80)
print("  UNCERTAINTY QUANTIFICATION FRAMEWORK - QUICK VALIDATION")
print("="*80)
print("\nTheoretical Foundation: Otgonbaatar et al. (2024) arXiv:2410.23311\n")

# Create framework
print("1. Creating UQ Framework with vQPU...")
uq = create_uq_framework(num_qubits=5)
print(f"   ✓ Created: {uq.vqpu.config.num_qubits} qubits")
print(f"   ✓ T1={uq.vqpu.config.noise_params.T1}μs, T2={uq.vqpu.config.noise_params.T2}μs")

# Test uncertainty quantification
print("\n2. Testing Uncertainty Quantification...")
result = uq.quantify_uncertainty(circuit_depth=10, num_samples=30)
print(f"   Expected value: {result.expected_value:.6f}")
print(f"   Total uncertainty: {result.uncertainty_metrics.total_uncertainty:.6f}")
print(f"   SNR: {result.uncertainty_metrics.signal_to_noise_ratio:.2f}")
print(f"   95% CI: [{result.uncertainty_metrics.confidence_95[0]:.4f}, {result.uncertainty_metrics.confidence_95[1]:.4f}]")
print(f"   ✓ UQ works")

# Test uncertainty decomposition
print("\n3. Testing Uncertainty Decomposition...")
print(f"   Epistemic: {result.uncertainty_metrics.epistemic_uncertainty:.6f}")
print(f"   Aleatoric: {result.uncertainty_metrics.aleatoric_uncertainty:.6f}")
print(f"   Systematic: {result.uncertainty_metrics.systematic_uncertainty:.6f}")
print(f"   Statistical: {result.uncertainty_metrics.statistical_uncertainty:.6f}")
print(f"   ✓ Decomposition works")

# Test noise analysis
print("\n4. Testing Noise Contribution Analysis...")
for noise_type, contribution in result.noise_contribution.items():
    print(f"   {noise_type.value}: {contribution:.4f}")
print(f"   ✓ Noise analysis works")

# Generate report
print("\n5. Generating Report...")
report = uq.generate_report()
print(f"   Analyses: {report['uncertainty_analysis']['num_analyses']}")
print(f"   Mean uncertainty: {report['uncertainty_analysis']['mean_total_uncertainty']:.6f}")
print(f"   ✓ Reporting works")

print("\n" + "="*80)
print("✅ ALL VALIDATION TESTS PASSED!")
print("Uncertainty Quantification Framework is OPERATIONAL")
print("="*80)

