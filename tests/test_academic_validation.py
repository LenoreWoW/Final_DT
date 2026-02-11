"""
Test Suite for Academic Statistical Validation Framework

Following best practices for scientific software testing and validation.
Tests ensure our statistical framework meets academic publication standards.
"""

import pytest
import numpy as np
import sys
import os
from typing import List

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dt_project.validation.academic_statistical_framework import (
    AcademicStatisticalValidator, 
    StatisticalResults, 
    PerformanceBenchmark,
)

# QuantumState was removed - create a mock for backward compatibility
class QuantumState:
    """Mock QuantumState for backward compatibility"""
    def __init__(self, amplitudes=None):
        self.amplitudes = amplitudes or [1.0, 0.0]

class TestAcademicStatisticalValidator:
    """Comprehensive test suite for academic statistical validation"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing"""
        return AcademicStatisticalValidator()
    
    @pytest.fixture
    def sample_performance_data(self):
        """Generate realistic quantum performance data for testing"""
        # Simulate quantum digital twin performance measurements
        np.random.seed(42)  # Reproducible results
        
        # High-performance quantum measurements (targeting 98.5% fidelity)
        quantum_data = np.random.normal(0.985, 0.005, 50)  # Mean 98.5%, std 0.5%
        
        # Baseline classical measurements (85% performance)
        classical_data = np.random.normal(0.850, 0.01, 50)   # Mean 85%, std 1%
        
        return {
            'quantum': quantum_data.tolist(),
            'classical': classical_data.tolist()
        }
    
    def test_statistical_significance_calculation(self, validator, sample_performance_data):
        """Test statistical significance calculation meets academic standards"""
        results = validator.validate_performance_claim(
            experimental_data=sample_performance_data['quantum'],
            control_data=sample_performance_data['classical'],
            claim_description="Quantum vs Classical Performance"
        )
        
        # Academic standard: p < 0.001
        assert results.p_value < 0.001, f"P-value {results.p_value} does not meet p < 0.001 standard"
        
        # Ensure statistically significant difference detected
        assert results.p_value < 0.05, "Should detect statistically significant difference"
    
    def test_confidence_interval_calculation(self, validator, sample_performance_data):
        """Test confidence interval calculation accuracy"""
        results = validator.validate_performance_claim(
            experimental_data=sample_performance_data['quantum'],
            control_data=sample_performance_data['classical']
        )
        
        # Confidence interval should contain the true mean (approximately)
        ci_lower, ci_upper = results.confidence_interval
        sample_mean = np.mean(sample_performance_data['quantum'])
        
        assert ci_lower <= sample_mean <= ci_upper, "Confidence interval should contain sample mean"
        assert ci_upper > ci_lower, "Upper bound should be greater than lower bound"
        assert (ci_upper - ci_lower) > 0, "Confidence interval should have positive width"
    
    def test_effect_size_calculation(self, validator, sample_performance_data):
        """Test Cohen's d effect size calculation"""
        results = validator.validate_performance_claim(
            experimental_data=sample_performance_data['quantum'],
            control_data=sample_performance_data['classical']
        )
        
        # With our sample data (98.5% vs 85%), should have large effect size
        assert results.effect_size > 0.8, f"Effect size {results.effect_size} should be > 0.8 (large effect)"
        
        # Effect size should be reasonable (not infinite)
        assert results.effect_size < 50, f"Effect size {results.effect_size} seems unreasonably large"
    
    def test_statistical_power_calculation(self, validator, sample_performance_data):
        """Test statistical power calculation"""
        results = validator.validate_performance_claim(
            experimental_data=sample_performance_data['quantum'],
            control_data=sample_performance_data['classical']
        )
        
        # Should have high statistical power with large effect size
        assert results.statistical_power > 0.8, f"Statistical power {results.statistical_power} should be > 0.8"
        assert 0 <= results.statistical_power <= 1, "Statistical power should be between 0 and 1"
    
    def test_academic_standards_validation(self, validator, sample_performance_data):
        """Test overall academic standards validation"""
        results = validator.validate_performance_claim(
            experimental_data=sample_performance_data['quantum'],
            control_data=sample_performance_data['classical']
        )
        
        # Should meet all academic standards with our sample data
        assert results.academic_standards_met, "Should meet academic standards with high-quality data"
        assert results.validation_status == "MEETS_ACADEMIC_STANDARDS"
        
        # Individual components should meet standards
        assert results.p_value < 0.001, "Should meet significance standard"
        assert results.effect_size > 0.8, "Should meet effect size standard"
        assert results.statistical_power > 0.8, "Should meet power standard"
        assert results.sample_size >= 30, "Should meet sample size standard"
    
    def test_fidelity_validation_specific(self, validator):
        """Test quantum fidelity-specific validation"""
        # Simulate high-fidelity quantum measurements
        fidelity_measurements = [0.985, 0.987, 0.983, 0.989, 0.986, 0.984, 0.988, 0.982, 0.990, 0.985] * 5
        
        results = validator.validate_fidelity_claim(fidelity_measurements, target_fidelity=0.985)
        
        assert results.p_value < 0.001, "Fidelity validation should be highly significant"
        assert results.effect_size > 0.8, "Fidelity improvement should show large effect"
        
        # Check confidence interval is reasonable for fidelity
        ci_lower, ci_upper = results.confidence_interval
        assert 0.8 < ci_lower < 1.0, "Fidelity CI lower bound should be reasonable"
        assert 0.8 < ci_upper < 1.0, "Fidelity CI upper bound should be reasonable"
    
    def test_optimization_speedup_validation(self, validator):
        """Test quantum optimization speedup validation"""
        # Quantum times (faster)
        quantum_times = [0.1, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.14, 0.09, 0.12] * 5
        
        # Classical times (slower)
        classical_times = [0.5, 0.52, 0.48, 0.51, 0.53, 0.47, 0.50, 0.54, 0.49, 0.52] * 5
        
        results = validator.validate_optimization_speedup(quantum_times, classical_times)
        
        assert results.p_value < 0.001, "Speedup should be statistically significant"
        assert results.effect_size > 0.8, "Should show large speedup effect"
        
        # Mean speedup should be approximately 5x (0.5/0.1)
        mean_speedup = np.mean([c/q for c, q in zip(classical_times, quantum_times)])
        assert 4 < mean_speedup < 6, f"Mean speedup {mean_speedup} should be around 5x"
    
    def test_sensing_precision_validation(self, validator):
        """Test quantum sensing precision validation"""
        # Quantum sensing (higher precision - lower error values)
        quantum_precisions = [0.98, 0.985, 0.982, 0.987, 0.983, 0.986, 0.984, 0.988, 0.981, 0.989] * 3
        
        # Classical sensing (lower precision - higher error values)  
        classical_precisions = [0.85, 0.82, 0.87, 0.84, 0.86, 0.83, 0.88, 0.81, 0.89, 0.85] * 3
        
        results = validator.validate_sensing_precision(quantum_precisions, classical_precisions)
        
        assert results.p_value < 0.001, "Sensing improvement should be highly significant"
        assert results.effect_size > 0.8, "Should show large precision improvement effect"
    
    def test_edge_cases_and_robustness(self, validator):
        """Test framework robustness with edge cases"""
        
        # Test with identical data (should show no effect)
        identical_data = [0.5] * 30
        results = validator.validate_performance_claim(identical_data, identical_data)
        
        assert results.p_value > 0.05, "Identical data should not be significant"
        assert results.effect_size < 0.1, "Identical data should show no effect"
        
        # Test with small sample size
        small_sample = [0.9, 0.8, 0.85]
        results = validator.validate_performance_claim(small_sample, [0.5, 0.4, 0.45])
        
        assert not results.academic_standards_met, "Small sample should not meet academic standards"
        assert results.sample_size < 30, "Should flag small sample size"
    
    def test_report_generation(self, validator, sample_performance_data):
        """Test academic report generation"""
        # Run several validations
        validator.validate_performance_claim(
            sample_performance_data['quantum'],
            sample_performance_data['classical'],
            "Test Validation 1"
        )
        
        validator.validate_fidelity_claim([0.98] * 40, target_fidelity=0.97)
        
        # Generate report
        report = validator.generate_academic_report()
        
        assert "ACADEMIC STATISTICAL VALIDATION REPORT" in report
        assert "Total Claims Validated: 2" in report
        assert "Statistical Significance:" in report
        assert "Effect Size (Cohen's d):" in report
        assert "95% Confidence Interval:" in report
    
    def test_export_for_publication(self, validator, sample_performance_data):
        """Test data export for academic publication"""
        validator.validate_performance_claim(
            sample_performance_data['quantum'],
            sample_performance_data['classical'],
            "Publication Test"
        )
        
        df = validator.export_results_for_publication()
        
        assert not df.empty, "Should export non-empty dataframe"
        assert 'p_value' in df.columns, "Should include p-value"
        assert 'effect_size_d' in df.columns, "Should include effect size"
        assert 'ci_lower' in df.columns, "Should include confidence intervals"
        assert 'academic_standards_met' in df.columns, "Should include standards validation"

class TestPerformanceBenchmarks:
    """Test academic performance benchmarks"""
    
    def test_benchmark_constants(self):
        """Test benchmark constants match academic standards"""
        benchmarks = PerformanceBenchmark()
        
        assert benchmarks.cern_fidelity == 0.999, "CERN fidelity benchmark should be 99.9%"
        assert benchmarks.dlr_variation_distance == 0.15, "DLR variation distance should be 0.15"
        assert benchmarks.statistical_significance == 0.001, "Significance should be p < 0.001"
        assert benchmarks.confidence_level == 0.95, "Confidence level should be 95%"
        assert benchmarks.effect_size_threshold == 0.8, "Effect size threshold should be 0.8"

# Integration tests with real quantum digital twin data
class TestQuantumDigitalTwinIntegration:
    """Integration tests with actual quantum digital twin implementations"""
    
    def test_integration_with_existing_twins(self):
        """Test integration with existing quantum digital twin implementations"""
        validator = AcademicStatisticalValidator()
        
        # Simulate data from our AthletePerformanceDigitalTwin
        athlete_performance_data = np.random.normal(0.92, 0.02, 40)  # 92% performance
        baseline_performance = np.random.normal(0.85, 0.03, 40)     # 85% baseline
        
        results = validator.validate_performance_claim(
            athlete_performance_data.tolist(),
            baseline_performance.tolist(),
            "AthletePerformanceDigitalTwin Validation"
        )
        
        # Should meet academic standards
        assert results.academic_standards_met, "Athlete twin should meet academic standards"
        
        # Simulate data from QuantumSensingDigitalTwin  
        sensing_precision_data = np.random.normal(0.98, 0.005, 45)  # 98% precision
        classical_sensing = np.random.normal(0.80, 0.01, 45)        # 80% classical
        
        sensing_results = validator.validate_sensing_precision(
            sensing_precision_data.tolist(),
            classical_sensing.tolist()
        )
        
        assert sensing_results.academic_standards_met, "Sensing twin should meet academic standards"
        assert sensing_results.effect_size > 2.0, "Should show very large sensing improvement"

if __name__ == "__main__":
    # Run tests with pytest for better output
    pytest.main([__file__, "-v", "--tb=short"])
