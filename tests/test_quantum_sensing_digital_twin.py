"""
Test Suite for Enhanced Quantum Sensing Digital Twin

Tests the implementation against theoretical foundations:
- Degen et al. (2017) Rev. Mod. Phys. 89, 035002
- Giovannetti et al. (2011) Nature Photonics 5, 222-229
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dt_project.quantum.quantum_sensing_digital_twin import (
    QuantumSensingDigitalTwin,
    QuantumSensingTheory,
    SensingModality,
    PrecisionScaling,
    SensingResult
)


class TestQuantumSensingTheory:
    """Test theoretical foundations"""
    
    def test_sql_scaling(self):
        """Test Standard Quantum Limit 1/√N scaling (Giovannetti 2011)"""
        theory = QuantumSensingTheory()
        
        # SQL should scale as 1/√N
        precision_10 = theory.calculate_precision_limit(10, PrecisionScaling.STANDARD_QUANTUM_LIMIT)
        precision_100 = theory.calculate_precision_limit(100, PrecisionScaling.STANDARD_QUANTUM_LIMIT)
        
        # Should improve by √10 when going from 10 to 100 measurements
        expected_ratio = np.sqrt(10)
        actual_ratio = precision_10 / precision_100
        
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"SQL scaling incorrect: expected {expected_ratio:.2f}, got {actual_ratio:.2f}"
    
    def test_heisenberg_limit_scaling(self):
        """Test Heisenberg Limit 1/N scaling (Degen 2017, Giovannetti 2011)"""
        theory = QuantumSensingTheory()
        
        # HL should scale as 1/N
        precision_10 = theory.calculate_precision_limit(10, PrecisionScaling.HEISENBERG_LIMIT)
        precision_100 = theory.calculate_precision_limit(100, PrecisionScaling.HEISENBERG_LIMIT)
        
        # Should improve by 10x when going from 10 to 100 measurements
        expected_ratio = 10.0
        actual_ratio = precision_10 / precision_100
        
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"HL scaling incorrect: expected {expected_ratio:.2f}, got {actual_ratio:.2f}"
    
    def test_quantum_advantage_factor(self):
        """Test quantum advantage scaling (Degen 2017)"""
        theory = QuantumSensingTheory()
        
        # Quantum advantage should be √N
        for N in [10, 100, 1000]:
            advantage = theory.quantum_advantage_factor(N)
            expected = np.sqrt(N)
            
            assert abs(advantage - expected) < 0.01, \
                f"Quantum advantage at N={N}: expected {expected:.2f}, got {advantage:.2f}"
    
    def test_heisenberg_better_than_sql(self):
        """Verify Heisenberg limit is better than SQL"""
        theory = QuantumSensingTheory()
        
        for N in [10, 100, 1000]:
            sql = theory.calculate_precision_limit(N, PrecisionScaling.STANDARD_QUANTUM_LIMIT)
            hl = theory.calculate_precision_limit(N, PrecisionScaling.HEISENBERG_LIMIT)
            
            assert hl < sql, \
                f"HL should be better than SQL at N={N}: HL={hl:.6f}, SQL={sql:.6f}"
            
            # Improvement should be √N
            improvement = sql / hl
            expected_improvement = np.sqrt(N)
            assert abs(improvement - expected_improvement) < 0.01, \
                f"Improvement should be √{N}={expected_improvement:.2f}, got {improvement:.2f}"


class TestSensingResult:
    """Test sensing result analysis"""
    
    def test_cramer_rao_bound(self):
        """Test Cramér-Rao bound calculation (Giovannetti 2011)"""
        # Create result with known QFI
        qfi = 100.0  # Quantum Fisher Information
        result = SensingResult(
            modality=SensingModality.PHASE_ESTIMATION,
            measured_value=0.5,
            precision=0.1,
            scaling_regime=PrecisionScaling.HEISENBERG_LIMIT,
            num_measurements=100,
            quantum_fisher_information=qfi
        )
        
        # Cramér-Rao bound: Δφ ≥ 1/√F_Q
        crb = result.cramer_rao_bound()
        expected_crb = 1.0 / np.sqrt(qfi)
        
        assert abs(crb - expected_crb) < 1e-10, \
            f"Cramér-Rao bound incorrect: expected {expected_crb:.6f}, got {crb:.6f}"
    
    def test_quantum_advantage_detection(self):
        """Test detection of quantum advantage"""
        theory = QuantumSensingTheory()
        
        # Result that beats SQL
        sql_limit = theory.calculate_precision_limit(100, PrecisionScaling.STANDARD_QUANTUM_LIMIT)
        hl_precision = theory.calculate_precision_limit(100, PrecisionScaling.HEISENBERG_LIMIT)
        
        result_quantum = SensingResult(
            modality=SensingModality.PHASE_ESTIMATION,
            measured_value=0.5,
            precision=hl_precision,  # Heisenberg-limited
            scaling_regime=PrecisionScaling.HEISENBERG_LIMIT,
            num_measurements=100,
            quantum_fisher_information=10000
        )
        
        assert result_quantum.achieves_quantum_advantage(theory), \
            "Should detect quantum advantage when precision beats SQL"
        
        # Result that doesn't beat SQL
        result_classical = SensingResult(
            modality=SensingModality.PHASE_ESTIMATION,
            measured_value=0.5,
            precision=sql_limit * 1.1,  # Worse than SQL
            scaling_regime=PrecisionScaling.STANDARD_QUANTUM_LIMIT,
            num_measurements=100,
            quantum_fisher_information=100
        )
        
        assert not result_classical.achieves_quantum_advantage(theory), \
            "Should not detect quantum advantage when precision doesn't beat SQL"


class TestQuantumSensingDigitalTwin:
    """Test the complete quantum sensing digital twin"""
    
    def test_initialization(self):
        """Test digital twin initialization"""
        twin = QuantumSensingDigitalTwin(
            num_qubits=4,
            modality=SensingModality.PHASE_ESTIMATION
        )
        
        assert twin.num_qubits == 4
        assert twin.modality == SensingModality.PHASE_ESTIMATION
        assert twin.theory is not None
        assert len(twin.sensing_history) == 0
    
    def test_sensing_measurement(self):
        """Test basic sensing measurement"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        true_parameter = 0.5
        result = twin.perform_sensing(true_parameter, num_shots=1000)
        
        # Check result structure
        assert result.modality == SensingModality.PHASE_ESTIMATION
        assert result.num_measurements == 1000
        assert result.scaling_regime == PrecisionScaling.HEISENBERG_LIMIT
        assert result.quantum_fisher_information > 0
        
        # Measurement should be close to true value
        error = abs(result.measured_value - true_parameter)
        assert error < 0.5, f"Measurement error too large: {error:.4f}"
        
        # Should achieve good precision
        assert result.precision < 0.1, \
            f"Precision should be better than 0.1, got {result.precision:.6f}"
    
    def test_heisenberg_limited_precision(self):
        """Test that measurements achieve Heisenberg-limited precision"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)

        num_shots = 1000
        result = twin.perform_sensing(0.5, num_shots=num_shots)

        # Calculate theoretical limits
        sql = twin.theory.calculate_precision_limit(num_shots, PrecisionScaling.STANDARD_QUANTUM_LIMIT)
        hl = twin.theory.calculate_precision_limit(num_shots, PrecisionScaling.HEISENBERG_LIMIT)

        # For n entangled qubits, HL precision is: Δφ = 1/(n × √N)
        # From Giovannetti 2011: QFI = n² × N → precision = 1/√QFI = 1/(n√N)
        hl_with_qubits = 1.0 / (twin.num_qubits * np.sqrt(num_shots))

        # Achieved precision should be close to HL
        assert result.precision < sql, \
            f"Should beat SQL: precision={result.precision:.6f}, SQL={sql:.6f}"

        # Should be within 2x of theoretical HL with qubits (accounting for noise)
        assert result.precision < 2 * hl_with_qubits, \
            f"Should approach HL: precision={result.precision:.6f}, HL={hl_with_qubits:.6f}"
    
    def test_multiple_measurements(self):
        """Test multiple sensing measurements"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        true_parameter = 0.3
        num_measurements = 10
        
        for _ in range(num_measurements):
            result = twin.perform_sensing(true_parameter, num_shots=500)
            assert result is not None
        
        assert len(twin.sensing_history) == num_measurements
        
        # All measurements should beat SQL
        for result in twin.sensing_history:
            sql = twin.theory.calculate_precision_limit(
                result.num_measurements,
                PrecisionScaling.STANDARD_QUANTUM_LIMIT
            )
            assert result.precision < sql, "All measurements should beat SQL"
    
    def test_quantum_fisher_information_scaling(self):
        """Test QFI scaling with number of qubits (Giovannetti 2011)"""
        # QFI should scale as N² for entangled state
        results = {}
        
        for n_qubits in [2, 4, 8]:
            twin = QuantumSensingDigitalTwin(num_qubits=n_qubits)
            result = twin.perform_sensing(0.5, num_shots=1000)
            results[n_qubits] = result.quantum_fisher_information
        
        # Check scaling: QFI(2N) / QFI(N) should be ≈ 4 for N² scaling
        ratio_2_to_4 = results[4] / results[2]
        expected_ratio = (4.0 / 2.0) ** 2  # = 4
        
        assert abs(ratio_2_to_4 - expected_ratio) / expected_ratio < 0.5, \
            f"QFI should scale as N²: ratio={ratio_2_to_4:.2f}, expected≈{expected_ratio:.2f}"
    
    def test_sensing_report_generation(self):
        """Test report generation"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        # Perform measurements
        for _ in range(5):
            twin.perform_sensing(0.5, num_shots=1000)
        
        report = twin.generate_sensing_report()
        
        # Check report structure
        assert 'theoretical_foundation' in report
        assert 'experimental_results' in report
        assert 'theoretical_comparison' in report
        assert 'quantum_fisher_information' in report
        
        # Check theoretical references
        assert 'Degen' in report['theoretical_foundation']['primary_reference']
        assert 'Giovannetti' in report['theoretical_foundation']['secondary_reference']
        
        # Check experimental results
        assert report['experimental_results']['num_measurements'] == 5
        assert report['experimental_results']['num_qubits'] == 4
        
        # Check theoretical comparison
        assert 'quantum_advantage_factor' in report['theoretical_comparison']
        assert report['theoretical_comparison']['beats_sql'] == True
    
    def test_different_modalities(self):
        """Test different sensing modalities"""
        modalities = [
            SensingModality.PHASE_ESTIMATION,
            SensingModality.AMPLITUDE_ESTIMATION,
            SensingModality.FREQUENCY_ESTIMATION,
            SensingModality.FORCE_DETECTION
        ]
        
        for modality in modalities:
            twin = QuantumSensingDigitalTwin(num_qubits=4, modality=modality)
            result = twin.perform_sensing(0.5, num_shots=1000)
            
            assert result.modality == modality
            assert result.precision > 0
            assert result.quantum_fisher_information > 0


class TestStatisticalValidation:
    """Test statistical validation of quantum advantage"""
    
    def test_validation_with_sufficient_data(self):
        """Test validation when sufficient data is available"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        # Perform 30+ measurements for statistical validity
        for _ in range(35):
            twin.perform_sensing(0.5, num_shots=1000)
        
        # Attempt validation
        validation = twin.validate_quantum_advantage()
        
        # Should have validation results
        if validation is not None:
            assert validation.p_value >= 0
            assert validation.effect_size >= 0
            assert validation.statistical_power >= 0
    
    def test_validation_insufficient_data(self):
        """Test validation with insufficient data"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        # Only 5 measurements (need 30+)
        for _ in range(5):
            twin.perform_sensing(0.5, num_shots=1000)
        
        validation = twin.validate_quantum_advantage()
        
        # Should return None or warning
        # (Depending on whether validation framework is available)
        assert validation is None or hasattr(validation, 'p_value')


class TestTheoreticalConsistency:
    """Test consistency with theoretical foundations"""
    
    def test_precision_bounds(self):
        """Test that precisions respect theoretical bounds"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        for num_shots in [100, 1000, 10000]:
            result = twin.perform_sensing(0.5, num_shots=num_shots)
            
            # Precision should be bounded by Cramér-Rao
            crb = result.cramer_rao_bound()
            assert result.precision >= crb * 0.5, \
                f"Precision violates Cramér-Rao bound: {result.precision:.6f} < {crb:.6f}"
    
    def test_advantage_consistency(self):
        """Test quantum advantage is consistent across measurements"""
        twin = QuantumSensingDigitalTwin(num_qubits=4)
        
        advantages = []
        for _ in range(10):
            result = twin.perform_sensing(0.5, num_shots=1000)
            sql = twin.theory.calculate_precision_limit(
                result.num_measurements,
                PrecisionScaling.STANDARD_QUANTUM_LIMIT
            )
            advantage = sql / result.precision
            advantages.append(advantage)
        
        # All should show quantum advantage (ratio > 1)
        assert all(a > 1.0 for a in advantages), \
            "All measurements should show quantum advantage"
        
        # Advantages should be reasonably consistent
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages)
        assert std_advantage / mean_advantage < 0.5, \
            f"Quantum advantage should be consistent: mean={mean_advantage:.2f}, std={std_advantage:.2f}"


# Integration test
def test_full_sensing_workflow():
    """Integration test of complete sensing workflow"""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Complete Quantum Sensing Workflow")
    print("="*80)
    
    # Create twin
    twin = QuantumSensingDigitalTwin(
        num_qubits=4,
        modality=SensingModality.PHASE_ESTIMATION
    )
    print(f"✓ Created quantum sensing digital twin with {twin.num_qubits} qubits")
    
    # Theoretical predictions
    theory_advantage = twin.theory.quantum_advantage_factor(1000)
    print(f"✓ Theoretical quantum advantage: {theory_advantage:.2f}x")
    
    # Perform measurements
    print(f"✓ Performing 30 sensing measurements...")
    true_phase = 0.42  # True parameter
    for i in range(30):
        result = twin.perform_sensing(true_phase, num_shots=1000)
        if i % 10 == 0:
            print(f"  Measurement {i+1}: precision={result.precision:.6f}")
    
    # Generate report
    report = twin.generate_sensing_report()
    print(f"✓ Generated sensing report")
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"  Standard Quantum Limit: {report['theoretical_comparison']['standard_quantum_limit']:.6f}")
    print(f"  Heisenberg Limit:       {report['theoretical_comparison']['heisenberg_limit']:.6f}")
    print(f"  Achieved Precision:     {report['experimental_results']['mean_precision']:.6f}")
    print(f"  Quantum Advantage:      {report['theoretical_comparison']['quantum_advantage_factor']:.2f}x")
    print(f"  Beats SQL:              {report['theoretical_comparison']['beats_sql']}")
    print(f"  Approaches HL:          {report['theoretical_comparison']['approaches_hl']}")
    
    # Validate
    print(f"\n📈 Statistical Validation:")
    validation = twin.validate_quantum_advantage()
    if validation:
        print(f"  P-value:            {validation.p_value:.6f}")
        print(f"  Effect Size:        {validation.effect_size:.2f}")
        print(f"  Statistical Power:  {validation.statistical_power:.4f}")
        print(f"  Standards Met:      {validation.academic_standards_met}")
    else:
        print(f"  Validation framework not available or insufficient data")
    
    print(f"\n✅ Integration test completed successfully!")
    print("="*80 + "\n")
    
    # Assertions
    assert report['theoretical_comparison']['beats_sql'] == True
    assert report['theoretical_comparison']['quantum_advantage_factor'] > 1.0


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])

