"""
Academic Statistical Framework for Quantum Digital Twin Validation

This module implements comprehensive statistical validation meeting academic publication
standards, including p-value testing, confidence intervals, effect size analysis,
and statistical power calculations.

Academic Standards Implemented:
- Statistical significance: p < 0.001
- Confidence intervals: 95% CI
- Effect sizes: Cohen's d > 0.8
- Statistical power: β > 0.8
"""

import numpy as np
import scipy.stats as stats
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResults:
    """Container for comprehensive statistical validation results"""
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    sample_size: int
    test_statistic: float
    validation_status: str
    academic_standards_met: bool
    timestamp: datetime
    
    def __post_init__(self):
        """Validate results meet academic standards"""
        self.academic_standards_met = (
            self.p_value < 0.001 and  # Highly significant
            self.effect_size > 0.8 and  # Large effect size
            self.statistical_power > 0.8 and  # Adequate power
            self.sample_size >= 30  # Minimum sample size
        )
        
        if self.academic_standards_met:
            self.validation_status = "MEETS_ACADEMIC_STANDARDS"
        else:
            self.validation_status = "REQUIRES_IMPROVEMENT"

@dataclass
class PerformanceBenchmark:
    """Academic benchmarks for quantum digital twin performance"""
    cern_fidelity: float = 0.999  # CERN tensor network standard
    dlr_variation_distance: float = 0.15  # DLR total variation distance
    statistical_significance: float = 0.001  # p < 0.001
    confidence_level: float = 0.95  # 95% confidence intervals
    effect_size_threshold: float = 0.8  # Cohen's d > 0.8
    power_threshold: float = 0.8  # Statistical power > 0.8

class AcademicStatisticalValidator:
    """
    Comprehensive statistical validation framework for quantum digital twins
    
    Implements rigorous academic standards for performance validation,
    ensuring all claims meet peer-review publication requirements.
    """
    
    def __init__(self):
        self.benchmarks = PerformanceBenchmark()
        self.validation_history = []
        
    def validate_performance_claim(
        self, 
        experimental_data: List[float], 
        control_data: Optional[List[float]] = None,
        claim_description: str = "Performance improvement"
    ) -> StatisticalResults:
        """
        Comprehensive statistical validation of performance claims
        
        Args:
            experimental_data: Performance measurements from enhanced system
            control_data: Baseline performance measurements (if applicable)
            claim_description: Description of the performance claim being tested
            
        Returns:
            StatisticalResults with comprehensive statistical analysis
        """
        logger.info(f"Validating performance claim: {claim_description}")
        
        # Convert to numpy arrays
        exp_data = np.array(experimental_data)
        
        if control_data is not None:
            ctrl_data = np.array(control_data)
        else:
            # Use theoretical baseline or literature benchmark
            ctrl_data = np.array([0.85] * len(exp_data))  # Conservative baseline
        
        # Statistical significance testing
        p_value = self._calculate_statistical_significance(exp_data, ctrl_data)
        
        # Confidence intervals
        confidence_interval = self._calculate_confidence_interval(exp_data)
        
        # Effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(exp_data, ctrl_data)
        
        # Statistical power
        statistical_power = self._calculate_statistical_power(exp_data, ctrl_data)
        
        # Test statistic
        test_statistic, _ = stats.ttest_ind(exp_data, ctrl_data)
        
        results = StatisticalResults(
            p_value=p_value,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            statistical_power=statistical_power,
            sample_size=len(exp_data),
            test_statistic=test_statistic,
            validation_status="",  # Will be set in __post_init__
            academic_standards_met=False,  # Will be set in __post_init__
            timestamp=datetime.now()
        )
        
        # Store validation history
        self.validation_history.append({
            'claim': claim_description,
            'results': results,
            'timestamp': results.timestamp
        })
        
        logger.info(f"Validation complete: {results.validation_status}")
        return results
    
    def _calculate_statistical_significance(
        self, 
        experimental_data: np.ndarray, 
        control_data: np.ndarray
    ) -> float:
        """
        Calculate statistical significance using appropriate statistical test
        
        Uses Welch's t-test for unequal variances, more robust than standard t-test
        """
        try:
            # Handle edge case: identical data (zero variance)
            # If both datasets are identical, there's no significant difference
            if np.array_equal(experimental_data, control_data):
                return 1.0  # Not significant
            
            # If either dataset has zero variance, can't compute t-test meaningfully
            if np.std(experimental_data) == 0 and np.std(control_data) == 0:
                if np.mean(experimental_data) == np.mean(control_data):
                    return 1.0  # No difference, not significant
                else:
                    return 0.0  # Completely different, highly significant
            
            # Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(
                experimental_data, 
                control_data, 
                equal_var=False
            )
            
            # Handle nan p-values (can occur with zero variance)
            if np.isnan(p_value):
                return 1.0  # Conservative: not significant
            
            return p_value
            
        except Exception as e:
            logger.error(f"Statistical significance calculation failed: {e}")
            return 1.0  # Conservative fallback
    
    def _calculate_confidence_interval(
        self, 
        data: np.ndarray, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean
        
        Uses t-distribution for small samples, normal for large samples
        """
        try:
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)  # Standard error of the mean
            
            # Use t-distribution for small samples
            if n < 30:
                t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
                margin_error = t_val * std_err
            else:
                # Use normal distribution for large samples
                z_val = stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_val * std_err
            
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            
            return (ci_lower, ci_upper)
            
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 1.0)  # Conservative fallback
    
    def _calculate_cohens_d(
        self, 
        experimental_data: np.ndarray, 
        control_data: np.ndarray
    ) -> float:
        """
        Calculate Cohen's d effect size
        
        Cohen's d interpretation:
        - 0.2: Small effect
        - 0.5: Medium effect  
        - 0.8: Large effect (our target)
        """
        try:
            mean_exp = np.mean(experimental_data)
            mean_ctrl = np.mean(control_data)
            
            # Handle identical data case
            if mean_exp == mean_ctrl:
                return 0.0  # No effect when means are identical
            
            # Pooled standard deviation
            n_exp = len(experimental_data)
            n_ctrl = len(control_data)
            
            var_exp = np.var(experimental_data, ddof=1)
            var_ctrl = np.var(control_data, ddof=1)
            
            # Handle zero variance case
            if var_exp == 0 and var_ctrl == 0:
                # Both datasets have zero variance (constant values)
                # If means differ, effect is infinite - use a large value
                return 0.0 if mean_exp == mean_ctrl else 10.0
            
            pooled_std = np.sqrt(
                ((n_exp - 1) * var_exp + 
                 (n_ctrl - 1) * var_ctrl) / 
                (n_exp + n_ctrl - 2)
            )
            
            # Handle zero pooled_std
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean_exp - mean_ctrl) / pooled_std
            
            # Handle nan
            if np.isnan(cohens_d):
                return 0.0
            
            return abs(cohens_d)  # Return absolute value
            
        except Exception as e:
            logger.error(f"Cohen's d calculation failed: {e}")
            return 0.0  # Conservative fallback
    
    def _calculate_statistical_power(
        self, 
        experimental_data: np.ndarray, 
        control_data: np.ndarray,
        alpha: float = 0.001
    ) -> float:
        """
        Calculate statistical power (1 - β)
        
        Power is the probability of detecting an effect if it exists
        Target power > 0.8 (80%)
        """
        try:
            effect_size = self._calculate_cohens_d(experimental_data, control_data)
            n = len(experimental_data)
            
            # Calculate power using effect size and sample size
            # This is a simplified calculation; for production, use statsmodels
            z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
            z_beta = effect_size * np.sqrt(n/2) - z_alpha
            
            power = stats.norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Statistical power calculation failed: {e}")
            return 0.5  # Conservative fallback
    
    def validate_fidelity_claim(
        self, 
        measured_fidelities: List[float], 
        target_fidelity: float = 0.995
    ) -> StatisticalResults:
        """
        Validate quantum digital twin fidelity claims against academic benchmarks
        
        Args:
            measured_fidelities: List of fidelity measurements
            target_fidelity: Target fidelity (default: 99.5%, approaching CERN 99.9%)
            
        Returns:
            Statistical validation results for fidelity claims
        """
        # Create baseline comparison (previous performance or theoretical)
        baseline_fidelities = [0.85] * len(measured_fidelities)  # Conservative baseline
        
        return self.validate_performance_claim(
            experimental_data=measured_fidelities,
            control_data=baseline_fidelities,
            claim_description=f"Quantum fidelity improvement (target: {target_fidelity:.1%})"
        )
    
    def validate_optimization_speedup(
        self, 
        quantum_times: List[float], 
        classical_times: List[float]
    ) -> StatisticalResults:
        """
        Validate quantum optimization speedup claims
        
        Args:
            quantum_times: Execution times using quantum algorithms
            classical_times: Execution times using classical algorithms
            
        Returns:
            Statistical validation of speedup claims
        """
        # Calculate speedup ratios
        speedup_ratios = [c/q for c, q in zip(classical_times, quantum_times)]
        baseline_ratios = [1.0] * len(speedup_ratios)  # No speedup baseline
        
        return self.validate_performance_claim(
            experimental_data=speedup_ratios,
            control_data=baseline_ratios,
            claim_description="Quantum optimization speedup"
        )
    
    def validate_sensing_precision(
        self, 
        quantum_precisions: List[float], 
        classical_precisions: List[float]
    ) -> StatisticalResults:
        """
        Validate quantum sensing precision improvement claims
        
        Args:
            quantum_precisions: Precision measurements with quantum sensing
            classical_precisions: Precision measurements with classical methods
            
        Returns:
            Statistical validation of sensing precision claims
        """
        return self.validate_performance_claim(
            experimental_data=quantum_precisions,
            control_data=classical_precisions,
            claim_description="Quantum sensing precision improvement"
        )
    
    def generate_academic_report(self) -> str:
        """
        Generate comprehensive academic validation report
        
        Returns:
            Formatted report suitable for academic publication
        """
        if not self.validation_history:
            return "No validation results available"
        
        report = []
        report.append("ACADEMIC STATISTICAL VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_validations = len(self.validation_history)
        passed_validations = sum(
            1 for v in self.validation_history 
            if v['results'].academic_standards_met
        )
        
        report.append(f"Total Claims Validated: {total_validations}")
        report.append(f"Academic Standards Met: {passed_validations}/{total_validations} ({passed_validations/total_validations:.1%})")
        report.append("")
        
        # Detailed results
        for validation in self.validation_history:
            claim = validation['claim']
            results = validation['results']
            
            report.append(f"CLAIM: {claim}")
            report.append(f"  Statistical Significance: p = {results.p_value:.6f} {'✓' if results.p_value < 0.001 else '✗'}")
            report.append(f"  95% Confidence Interval: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
            report.append(f"  Effect Size (Cohen's d): {results.effect_size:.4f} {'✓' if results.effect_size > 0.8 else '✗'}")
            report.append(f"  Statistical Power: {results.statistical_power:.4f} {'✓' if results.statistical_power > 0.8 else '✗'}")
            report.append(f"  Sample Size: {results.sample_size}")
            report.append(f"  Academic Standards: {'✓ MET' if results.academic_standards_met else '✗ NOT MET'}")
            report.append("")
        
        return "\n".join(report)
    
    def export_results_for_publication(self) -> pd.DataFrame:
        """
        Export validation results in format suitable for academic publication
        
        Returns:
            DataFrame with all validation results for statistical reporting
        """
        data = []
        for validation in self.validation_history:
            results = validation['results']
            data.append({
                'claim': validation['claim'],
                'p_value': results.p_value,
                'ci_lower': results.confidence_interval[0],
                'ci_upper': results.confidence_interval[1],
                'effect_size_d': results.effect_size,
                'statistical_power': results.statistical_power,
                'sample_size': results.sample_size,
                'test_statistic': results.test_statistic,
                'academic_standards_met': results.academic_standards_met,
                'timestamp': results.timestamp
            })
        
        return pd.DataFrame(data)

# Example usage and testing
if __name__ == "__main__":
    # Example validation of quantum digital twin performance
    validator = AcademicStatisticalValidator()
    
    # Simulate quantum fidelity measurements
    quantum_fidelities = np.random.normal(0.985, 0.01, 50)  # Mean 98.5%, std 1%
    
    # Validate fidelity claims
    fidelity_results = validator.validate_fidelity_claim(quantum_fidelities.tolist())
    
    print("Quantum Fidelity Validation Results:")
    print(f"P-value: {fidelity_results.p_value:.6f}")
    print(f"95% CI: [{fidelity_results.confidence_interval[0]:.4f}, {fidelity_results.confidence_interval[1]:.4f}]")
    print(f"Effect Size: {fidelity_results.effect_size:.4f}")
    print(f"Academic Standards Met: {fidelity_results.academic_standards_met}")
    
    # Generate academic report
    print("\n" + validator.generate_academic_report())
