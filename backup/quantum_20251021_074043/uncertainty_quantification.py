"""
Uncertainty Quantification Framework for Quantum Digital Twins

Theoretical Foundation:
- Otgonbaatar et al. (2024) "Uncertainty Quantification..." arXiv:2410.23311

This implementation provides uncertainty quantification for quantum systems:
- Virtual Quantum Processing Units (vQPUs)
- Noise characterization and analysis
- Distributed quantum computation modeling
- Uncertainty propagation in quantum circuits

Key Features:
- vQPU simulation with realistic noise
- Uncertainty metrics (epistemic & aleatoric)
- Noise model calibration
- Distributed system uncertainty
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Qiskit for quantum operations
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available for quantum operations")


class NoiseType(Enum):
    """Types of quantum noise"""
    DEPOLARIZING = "depolarizing"
    THERMAL = "thermal_relaxation"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    READOUT = "readout_error"


class UncertaintyType(Enum):
    """Types of uncertainty in quantum systems"""
    EPISTEMIC = "epistemic"  # Knowledge-based (reducible)
    ALEATORIC = "aleatoric"  # Inherent randomness (irreducible)
    SYSTEMATIC = "systematic"  # Device/calibration errors
    STATISTICAL = "statistical"  # Sampling uncertainty


@dataclass
class NoiseParameters:
    """
    Noise parameters for virtual QPU
    
    Based on Otgonbaatar et al. (2024) for realistic noise modeling
    """
    # Gate errors
    single_qubit_error: float = 0.001  # 0.1% error rate
    two_qubit_error: float = 0.01  # 1% error rate
    
    # Decoherence times (in microseconds)
    T1: float = 50.0  # Relaxation time
    T2: float = 70.0  # Dephasing time
    
    # Readout errors
    readout_error: float = 0.01  # 1% readout error
    
    # Temperature
    temperature: float = 0.015  # Kelvin (typical)
    
    def __post_init__(self):
        """Validate noise parameters"""
        if self.T2 > 2 * self.T1:
            logger.warning(f"T2 ({self.T2}) > 2*T1 ({2*self.T1}), adjusting T2")
            self.T2 = 2 * self.T1


@dataclass
class VirtualQPUConfig:
    """Configuration for Virtual Quantum Processing Unit"""
    num_qubits: int = 5
    noise_params: NoiseParameters = field(default_factory=NoiseParameters)
    backend_name: str = "virtual_qpu"
    num_shots: int = 1024
    
    def __post_init__(self):
        if self.num_qubits < 2:
            raise ValueError("Need at least 2 qubits")


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics for quantum computation"""
    # Total uncertainty
    total_uncertainty: float
    
    # Breakdown by type
    epistemic_uncertainty: float  # Reducible (model/calibration)
    aleatoric_uncertainty: float  # Irreducible (quantum randomness)
    systematic_uncertainty: float  # Device errors
    statistical_uncertainty: float  # Sampling uncertainty
    
    # Confidence intervals
    confidence_95: Tuple[float, float]
    confidence_99: Tuple[float, float]
    
    # Additional metrics
    signal_to_noise_ratio: float
    fidelity_uncertainty: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def total_variance(self) -> float:
        """Calculate total variance from uncertainty components"""
        return (self.epistemic_uncertainty**2 + 
                self.aleatoric_uncertainty**2 + 
                self.systematic_uncertainty**2 + 
                self.statistical_uncertainty**2)


@dataclass
class UQResult:
    """Result from uncertainty quantification analysis"""
    circuit_depth: int
    num_qubits: int
    expected_value: float
    uncertainty_metrics: UncertaintyMetrics
    noise_contribution: Dict[NoiseType, float]
    num_samples: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV = σ/μ)"""
        if abs(self.expected_value) < 1e-10:
            return float('inf')
        return self.uncertainty_metrics.total_uncertainty / abs(self.expected_value)


class VirtualQPU:
    """
    Virtual Quantum Processing Unit with Realistic Noise
    
    From Otgonbaatar et al. (2024):
    Models realistic quantum hardware with calibrated noise
    """
    
    def __init__(self, config: VirtualQPUConfig):
        """
        Initialize virtual QPU
        
        Args:
            config: vQPU configuration
        """
        self.config = config
        self.noise_model = None
        
        # Build noise model
        if QISKIT_AVAILABLE:
            self._build_noise_model()
        
        # Calibration data
        self.calibration_data: Dict[str, Any] = {}
        
        logger.info(f"Virtual QPU initialized: {config.num_qubits} qubits")
        logger.info(f"  T1: {config.noise_params.T1}μs, T2: {config.noise_params.T2}μs")
        logger.info(f"  Single-qubit error: {config.noise_params.single_qubit_error:.4f}")
        logger.info(f"  Two-qubit error: {config.noise_params.two_qubit_error:.4f}")
    
    def _build_noise_model(self):
        """Build realistic noise model for vQPU"""
        if not QISKIT_AVAILABLE:
            return
        
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
            
            noise_model = NoiseModel()
            
            # Single-qubit gates
            error_1q = depolarizing_error(self.config.noise_params.single_qubit_error, 1)
            noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
            
            # Two-qubit gates
            error_2q = depolarizing_error(self.config.noise_params.two_qubit_error, 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
            
            # Thermal relaxation (T1, T2)
            for qubit in range(self.config.num_qubits):
                thermal_error = thermal_relaxation_error(
                    self.config.noise_params.T1,
                    self.config.noise_params.T2,
                    time=100  # Gate time in nanoseconds
                )
                noise_model.add_quantum_error(thermal_error, ['u1', 'u2', 'u3'], [qubit])
            
            self.noise_model = noise_model
            logger.info("Noise model built successfully")
            
        except Exception as e:
            logger.warning(f"Could not build full noise model: {e}")
    
    def execute_with_noise(self, circuit: Any, num_shots: int = None) -> Dict[str, int]:
        """
        Execute quantum circuit with noise
        
        Args:
            circuit: Quantum circuit to execute
            num_shots: Number of measurement shots
            
        Returns:
            Measurement counts
        """
        if num_shots is None:
            num_shots = self.config.num_shots
        
        if QISKIT_AVAILABLE and self.noise_model:
            # Execute with Qiskit noise
            simulator = AerSimulator(noise_model=self.noise_model)
            job = simulator.run(circuit, shots=num_shots)
            result = job.result()
            counts = result.get_counts()
        else:
            # Simplified noise simulation
            counts = self._simulate_noisy_measurement(num_shots)
        
        return counts
    
    def _simulate_noisy_measurement(self, num_shots: int) -> Dict[str, int]:
        """Simulate noisy quantum measurement (simplified)"""
        # Generate random measurement outcomes with noise
        num_outcomes = 2 ** self.config.num_qubits
        
        # Ideal distribution (uniform for simplicity)
        ideal_probs = np.ones(num_outcomes) / num_outcomes
        
        # Add noise
        noise_level = self.config.noise_params.single_qubit_error
        noisy_probs = ideal_probs * (1 - noise_level) + noise_level / num_outcomes
        noisy_probs = noisy_probs / np.sum(noisy_probs)
        
        # Sample
        outcomes = np.random.choice(num_outcomes, size=num_shots, p=noisy_probs)
        
        # Convert to bitstrings
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.config.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts


class UncertaintyQuantificationFramework:
    """
    Uncertainty Quantification Framework for Quantum Digital Twins
    
    Theoretical Foundation:
    =======================
    
    Otgonbaatar et al. (2024) - "Uncertainty Quantification..."
    arXiv:2410.23311
    
    Key Concepts:
    - Virtual quantum processing units (vQPUs)
    - Uncertainty quantification in quantum systems
    - Noise characterization
    - Distributed quantum computation
    
    Implementation:
    ===============
    
    This framework provides:
    1. vQPU simulation with realistic noise
    2. Comprehensive uncertainty metrics
    3. Noise source decomposition
    4. Uncertainty propagation analysis
    
    Applications (from Otgonbaatar 2024):
    =====================================
    
    - Quantum algorithm validation
    - Hardware calibration
    - Error budgeting
    - Distributed quantum systems
    """
    
    def __init__(self, vqpu_config: Optional[VirtualQPUConfig] = None):
        """
        Initialize uncertainty quantification framework
        
        Args:
            vqpu_config: Configuration for virtual QPU
        """
        if vqpu_config is None:
            vqpu_config = VirtualQPUConfig()
        
        self.vqpu = VirtualQPU(vqpu_config)
        self.uq_history: List[UQResult] = []
        
        logger.info("Uncertainty Quantification Framework initialized")
    
    def quantify_uncertainty(self,
                            circuit: Optional[Any] = None,
                            circuit_depth: int = 10,
                            num_samples: int = 100) -> UQResult:
        """
        Perform comprehensive uncertainty quantification
        
        This is the primary method from Otgonbaatar et al. (2024):
        Quantifying uncertainty in quantum computations.
        
        Args:
            circuit: Quantum circuit to analyze
            circuit_depth: Depth of circuit (if generating)
            num_samples: Number of Monte Carlo samples
            
        Returns:
            UQResult with comprehensive uncertainty analysis
        """
        logger.info(f"Quantifying uncertainty: depth={circuit_depth}, samples={num_samples}")
        
        # Generate or use provided circuit
        if circuit is None:
            circuit = self._generate_test_circuit(circuit_depth)
        
        # Collect samples
        samples = []
        for i in range(num_samples):
            result = self._execute_and_measure(circuit)
            samples.append(result)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"  Collected {i+1}/{num_samples} samples")
        
        # Calculate expected value
        expected_value = np.mean(samples)
        
        # Decompose uncertainty by type
        epistemic = self._estimate_epistemic_uncertainty(samples)
        aleatoric = self._estimate_aleatoric_uncertainty(samples)
        systematic = self._estimate_systematic_uncertainty()
        statistical = self._estimate_statistical_uncertainty(samples)
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2 + 
                                    systematic**2 + statistical**2)
        
        # Confidence intervals
        ci_95 = self._calculate_confidence_interval(samples, 0.95)
        ci_99 = self._calculate_confidence_interval(samples, 0.99)
        
        # Signal to noise ratio
        snr = abs(expected_value) / total_uncertainty if total_uncertainty > 0 else float('inf')
        
        # Fidelity uncertainty (estimate)
        fidelity_uncertainty = total_uncertainty / (1 + abs(expected_value))
        
        uncertainty_metrics = UncertaintyMetrics(
            total_uncertainty=total_uncertainty,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            systematic_uncertainty=systematic,
            statistical_uncertainty=statistical,
            confidence_95=ci_95,
            confidence_99=ci_99,
            signal_to_noise_ratio=snr,
            fidelity_uncertainty=fidelity_uncertainty
        )
        
        # Noise contribution analysis
        noise_contribution = self._analyze_noise_contribution()
        
        result = UQResult(
            circuit_depth=circuit_depth,
            num_qubits=self.vqpu.config.num_qubits,
            expected_value=expected_value,
            uncertainty_metrics=uncertainty_metrics,
            noise_contribution=noise_contribution,
            num_samples=num_samples
        )
        
        self.uq_history.append(result)
        
        logger.info(f"UQ complete: value={expected_value:.6f}, "
                   f"uncertainty={total_uncertainty:.6f}, SNR={snr:.2f}")
        
        return result
    
    def _generate_test_circuit(self, depth: int) -> Any:
        """Generate test quantum circuit"""
        if QISKIT_AVAILABLE:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(self.vqpu.config.num_qubits, self.vqpu.config.num_qubits)
            
            # Random circuit
            for _ in range(depth):
                for qubit in range(self.vqpu.config.num_qubits):
                    qc.h(qubit)
                for qubit in range(self.vqpu.config.num_qubits - 1):
                    qc.cx(qubit, qubit + 1)
            
            qc.measure_all()
            return qc
        else:
            # Mock circuit
            return {"depth": depth, "qubits": self.vqpu.config.num_qubits}
    
    def _execute_and_measure(self, circuit: Any) -> float:
        """Execute circuit and extract measurement value"""
        counts = self.vqpu.execute_with_noise(circuit)
        
        # Extract expectation value (simplified: parity measurement)
        total_counts = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Parity: even number of 1s -> +1, odd -> -1
            parity = (-1) ** bitstring.count('1')
            expectation += parity * count / total_counts
        
        return expectation
    
    def _estimate_epistemic_uncertainty(self, samples: List[float]) -> float:
        """
        Estimate epistemic uncertainty (knowledge-based, reducible)
        
        From Otgonbaatar 2024: Model/calibration uncertainties
        """
        # Simplified: based on calibration quality
        base_epistemic = 0.05  # 5% baseline
        
        # Reduces with more calibration data
        calibration_factor = max(0.1, 1.0 / (1 + len(self.vqpu.calibration_data)))
        
        return base_epistemic * calibration_factor
    
    def _estimate_aleatoric_uncertainty(self, samples: List[float]) -> float:
        """
        Estimate aleatoric uncertainty (inherent randomness, irreducible)
        
        From quantum measurement statistics
        """
        # Quantum shot noise
        std = np.std(samples)
        return std / np.sqrt(len(samples))
    
    def _estimate_systematic_uncertainty(self) -> float:
        """
        Estimate systematic uncertainty (device/calibration errors)
        
        From noise parameters
        """
        # Based on gate errors and decoherence
        gate_error = self.vqpu.config.noise_params.single_qubit_error
        two_qubit_error = self.vqpu.config.noise_params.two_qubit_error
        
        # Weighted combination
        systematic = np.sqrt(gate_error**2 + two_qubit_error**2)
        
        return systematic
    
    def _estimate_statistical_uncertainty(self, samples: List[float]) -> float:
        """
        Estimate statistical uncertainty (sampling)
        
        Standard error of the mean
        """
        return np.std(samples) / np.sqrt(len(samples))
    
    def _calculate_confidence_interval(self, 
                                      samples: List[float],
                                      confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        mean = np.mean(samples)
        std = np.std(samples)
        n = len(samples)
        
        # Use t-distribution for small samples
        from scipy import stats as scipy_stats
        
        if n < 30:
            t_val = scipy_stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin = t_val * std / np.sqrt(n)
        else:
            z_val = scipy_stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_val * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def _analyze_noise_contribution(self) -> Dict[NoiseType, float]:
        """Analyze contribution of different noise types"""
        contributions = {}
        
        # Simplified: estimate from noise parameters
        params = self.vqpu.config.noise_params
        
        contributions[NoiseType.DEPOLARIZING] = params.single_qubit_error * 0.4
        contributions[NoiseType.THERMAL] = (1.0 / params.T1) * 0.3
        contributions[NoiseType.PHASE_DAMPING] = (1.0 / params.T2) * 0.2
        contributions[NoiseType.READOUT] = params.readout_error * 0.1
        
        # Normalize
        total = sum(contributions.values())
        contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions
    
    def calibrate_vqpu(self, calibration_circuits: List[Any]) -> Dict[str, float]:
        """
        Calibrate virtual QPU using reference circuits
        
        From Otgonbaatar 2024: Calibration reduces epistemic uncertainty
        """
        logger.info(f"Calibrating vQPU with {len(calibration_circuits)} circuits")
        
        calibration_results = {}
        
        for i, circuit in enumerate(calibration_circuits):
            result = self._execute_and_measure(circuit)
            calibration_results[f"circuit_{i}"] = result
        
        # Store calibration data
        self.vqpu.calibration_data.update(calibration_results)
        
        logger.info(f"Calibration complete: {len(calibration_results)} measurements")
        
        return calibration_results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive uncertainty quantification report
        
        Returns:
            Report with theoretical foundation and UQ results
        """
        if not self.uq_history:
            return {"error": "No UQ data available"}
        
        total_uncertainties = [r.uncertainty_metrics.total_uncertainty for r in self.uq_history]
        snrs = [r.uncertainty_metrics.signal_to_noise_ratio for r in self.uq_history]
        
        report = {
            "theoretical_foundation": {
                "reference": "Otgonbaatar et al. (2024) arXiv:2410.23311",
                "method": "Uncertainty Quantification Framework",
                "application": "Virtual QPU with Noise Modeling"
            },
            "vqpu_configuration": {
                "num_qubits": self.vqpu.config.num_qubits,
                "T1_relaxation": self.vqpu.config.noise_params.T1,
                "T2_dephasing": self.vqpu.config.noise_params.T2,
                "single_qubit_error": self.vqpu.config.noise_params.single_qubit_error,
                "two_qubit_error": self.vqpu.config.noise_params.two_qubit_error
            },
            "uncertainty_analysis": {
                "num_analyses": len(self.uq_history),
                "mean_total_uncertainty": float(np.mean(total_uncertainties)),
                "mean_snr": float(np.mean([s for s in snrs if s != float('inf')])),
                "calibration_measurements": len(self.vqpu.calibration_data)
            },
            "uncertainty_breakdown": {
                "epistemic": float(np.mean([r.uncertainty_metrics.epistemic_uncertainty for r in self.uq_history])),
                "aleatoric": float(np.mean([r.uncertainty_metrics.aleatoric_uncertainty for r in self.uq_history])),
                "systematic": float(np.mean([r.uncertainty_metrics.systematic_uncertainty for r in self.uq_history])),
                "statistical": float(np.mean([r.uncertainty_metrics.statistical_uncertainty for r in self.uq_history]))
            }
        }
        
        return report


# Factory function
def create_uq_framework(num_qubits: int = 5,
                       noise_params: Optional[NoiseParameters] = None) -> UncertaintyQuantificationFramework:
    """
    Create uncertainty quantification framework
    
    Based on Otgonbaatar et al. (2024)
    
    Args:
        num_qubits: Number of qubits for vQPU
        noise_params: Noise parameters (uses defaults if None)
        
    Returns:
        Configured UncertaintyQuantificationFramework
    """
    if noise_params is None:
        noise_params = NoiseParameters()
    
    config = VirtualQPUConfig(
        num_qubits=num_qubits,
        noise_params=noise_params
    )
    
    return UncertaintyQuantificationFramework(config)


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║          UNCERTAINTY QUANTIFICATION FRAMEWORK                                ║
    ║          Virtual QPU with Noise Modeling                                     ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    Theoretical Foundation:
    ----------------------
    Otgonbaatar et al. (2024) arXiv:2410.23311
    
    - Virtual quantum processing units
    - Uncertainty quantification
    - Noise characterization
    - Distributed quantum computation
    """)
    
    # Create UQ framework
    uq = create_uq_framework(num_qubits=5)
    
    print(f"\n🎯 UQ Framework created:")
    print(f"   vQPU qubits: {uq.vqpu.config.num_qubits}")
    print(f"   T1: {uq.vqpu.config.noise_params.T1}μs")
    print(f"   T2: {uq.vqpu.config.noise_params.T2}μs")
    
    # Quantify uncertainty
    print(f"\n📊 Quantifying uncertainty...")
    
    for depth in [5, 10, 15]:
        result = uq.quantify_uncertainty(circuit_depth=depth, num_samples=50)
        
        print(f"\n   Depth {depth:2d}:")
        print(f"     Expected value: {result.expected_value:.6f}")
        print(f"     Total uncertainty: {result.uncertainty_metrics.total_uncertainty:.6f}")
        print(f"     SNR: {result.uncertainty_metrics.signal_to_noise_ratio:.2f}")
        print(f"     95% CI: [{result.uncertainty_metrics.confidence_95[0]:.4f}, "
              f"{result.uncertainty_metrics.confidence_95[1]:.4f}]")
    
    # Generate report
    print(f"\n📋 Generating report...")
    report = uq.generate_report()
    
    print(f"\n✅ RESULTS:")
    print(f"   Reference: {report['theoretical_foundation']['reference']}")
    print(f"   Analyses run: {report['uncertainty_analysis']['num_analyses']}")
    print(f"   Mean total uncertainty: {report['uncertainty_analysis']['mean_total_uncertainty']:.6f}")
    print(f"   Mean SNR: {report['uncertainty_analysis']['mean_snr']:.2f}")
    print(f"\n   Uncertainty breakdown:")
    print(f"     Epistemic: {report['uncertainty_breakdown']['epistemic']:.6f}")
    print(f"     Aleatoric: {report['uncertainty_breakdown']['aleatoric']:.6f}")
    print(f"     Systematic: {report['uncertainty_breakdown']['systematic']:.6f}")
    print(f"     Statistical: {report['uncertainty_breakdown']['statistical']:.6f}")

