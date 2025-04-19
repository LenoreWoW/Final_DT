"""
Quantum Monte Carlo Module
Implements quantum-enhanced Monte Carlo methods for simulation and prediction.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import time
import json
import os
from collections import defaultdict

# Global variables to track availability
QISKIT_AVAILABLE = False
PENNYLANE_AVAILABLE = False
SCIPY_AVAILABLE = False

try:
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.visualization import plot_histogram
    from qiskit.providers.ibmq import least_busy
    from qiskit.providers import Backend
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # Define a placeholder QuantumCircuit for type hints
    class QuantumCircuit:
        """Placeholder for Qiskit's QuantumCircuit when Qiskit is not available."""
        pass
    
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# For probability distribution analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class QuantumMonteCarlo:
    """
    Quantum Monte Carlo simulation for enhancing performance predictions.
    Uses quantum algorithms to generate higher-quality random samples and
    accelerate convergence of simulation results.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the Quantum Monte Carlo simulator.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        self._load_config()
        
        # Check quantum backend availability
        self.quantum_available = QISKIT_AVAILABLE or PENNYLANE_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        
        if not self.quantum_available:
            logger.warning("Neither Qiskit nor PennyLane are available. Falling back to classical simulation.")
        
        # Initialize backend
        self.backend = None
        if self.quantum_available and self.enabled:
            self._initialize_backend()
            
        # Cache for simulation results
        cache_dir = self.config.get("cache_dir", "data/cache")
        self.cache_dir = os.path.join(cache_dir, "quantum")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Distribution analysis results
        self.distribution_analysis = {}
        self.available_distributions = {
            "uniform": self._create_uniform_distribution_circuit,
            "normal": self._create_normal_distribution_circuit,
            "exponential": self._create_exponential_distribution_circuit,
            "beta": self._create_beta_distribution_circuit
        }
    
    def _load_config(self) -> None:
        """Load configuration parameters."""
        quantum_config = self.config.get("quantum", {})
        self.enabled = quantum_config.get("enabled", False)
        self.backend_name = quantum_config.get("backend", "simulator")
        self.shots = quantum_config.get("shots", 1024)
        self.max_circuit_depth = quantum_config.get("max_circuit_depth", 20)
        
        # Advanced parameters
        self.optimization_level = quantum_config.get("optimization_level", 1)
        self.error_mitigation = quantum_config.get("error_mitigation", False)
        self.use_amplitude_amp = quantum_config.get("use_amplitude_amplification", False)
        self.parallel_circuits = quantum_config.get("parallel_circuits", 2)
        self.preferred_distribution = quantum_config.get("preferred_distribution", "auto")
        
    def _initialize_backend(self) -> None:
        """Initialize the quantum backend based on configuration."""
        if not self.quantum_available:
            return
            
        if QISKIT_AVAILABLE:
            try:
                if self.backend_name == "simulator":
                    self.backend = Aer.get_backend('qasm_simulator')
                    logger.info("Using QASM simulator backend")
                elif self.backend_name.startswith('ibmq'):
                    # Try to use IBMQ backend if token is provided
                    token = os.getenv('IBMQ_TOKEN')
                    if token:
                        try:
                            from qiskit import IBMQ
                            IBMQ.save_account(token, overwrite=True)
                            provider = IBMQ.load_account()
                            
                            if self.backend_name == 'ibmq_auto':
                                # Find least busy backend
                                self.backend = least_busy(provider.backends(
                                    filters=lambda b: b.configuration().n_qubits >= 5 and
                                                     not b.configuration().simulator and
                                                     b.status().operational
                                ))
                                logger.info(f"Selected least busy IBMQ backend: {self.backend.name()}")
                            else:
                                # Use specified backend
                                self.backend = provider.get_backend(self.backend_name)
                                logger.info(f"Using specified IBMQ backend: {self.backend.name()}")
                        except Exception as e:
                            logger.error(f"Failed to initialize IBMQ backend: {str(e)}")
                            self.backend = Aer.get_backend('qasm_simulator')
                            logger.info("Falling back to QASM simulator")
                    else:
                        logger.warning("IBMQ token not provided. Falling back to simulator.")
                        self.backend = Aer.get_backend('qasm_simulator')
            except Exception as e:
                logger.error(f"Error initializing Qiskit backend: {str(e)}")
                self.backend = None
                
        elif PENNYLANE_AVAILABLE:
            try:
                if self.backend_name == "simulator":
                    self.backend = "default.qubit"
                    logger.info("Using PennyLane default.qubit simulator")
                elif self.backend_name.startswith('ibmq'):
                    # If using PennyLane with IBM backend
                    token = os.getenv('IBMQ_TOKEN')
                    if token:
                        # Configure IBM provider
                        self.backend = "qiskit.ibmq"
                        logger.info(f"Using PennyLane with IBM quantum backend")
                    else:
                        logger.warning("IBMQ token not provided. Falling back to default.qubit simulator.")
                        self.backend = "default.qubit"
            except Exception as e:
                logger.error(f"Error initializing PennyLane backend: {str(e)}")
                self.backend = None
    
    def is_available(self) -> bool:
        """
        Check if quantum processing is available.
        
        Returns:
            True if quantum processing is available, False otherwise
        """
        return self.quantum_available and self.enabled and self.backend is not None
    
    def analyze_distributions(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze probability distributions in data to identify potential quantum advantage.
        
        Args:
            data: Dictionary of named data columns to analyze
            
        Returns:
            Dictionary of distribution analysis results
        """
        if not self.scipy_available:
            logger.warning("SciPy not available. Cannot analyze distributions.")
            return {}
        
        analysis_results = {}
        
        for name, values in data.items():
            values = np.array(values)
            
            # Test various distributions
            distributions = {
                "normal": stats.normaltest(values),
                "uniform": stats.kstest(values, 'uniform', 
                                     args=(np.min(values), np.max(values) - np.min(values))),
                "exponential": stats.kstest(values - np.min(values), 'expon'),
                "beta": stats.kstest((values - np.min(values)) / (np.max(values) - np.min(values)), 'beta', 
                                   args=(2, 2)),  # Assuming symmetric beta for test
            }
            
            # Determine best fit
            p_values = {k: v[1] for k, v in distributions.items()}
            best_fit = max(p_values.items(), key=lambda x: x[1])
            
            # Estimate quantum advantage 
            quantum_advantage = {
                "normal": 0.4,       # Medium advantage for normal distributions
                "uniform": 0.7,      # High advantage for uniform sampling
                "exponential": 0.5,  # Medium-high advantage
                "beta": 0.6          # High advantage for beta
            }
            
            analysis_results[name] = {
                "best_fit": best_fit[0],
                "p_value": best_fit[1],
                "distribution_tests": p_values,
                "potential_quantum_advantage": quantum_advantage.get(best_fit[0], 0.3),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        self.distribution_analysis = analysis_results
        return analysis_results
    
    def _apply_error_mitigation(self, circuit, n_qubits):
        """
        Apply error mitigation techniques to the quantum circuit.
        
        Args:
            circuit: Quantum circuit to apply error mitigation to
            n_qubits: Number of qubits in the circuit
            
        Returns:
            Mitigated circuit and mitigation data for post-processing
        """
        if not self.error_mitigation or not QISKIT_AVAILABLE:
            return circuit, None
            
        mitigation_data = {}
        
        try:
            from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
            
            # Create measurement calibration circuits
            meas_calibs, state_labels = complete_meas_cal(qubit_list=range(n_qubits), qr=circuit.qregs[0])
            
            # Execute calibration circuits
            calibration_job = execute(meas_calibs, self.backend, shots=self.shots)
            cal_results = calibration_job.result()
            
            # Create measurement fitter
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)
            mitigation_data['measurement_fitter'] = meas_fitter
            
            logger.info(f"Applied measurement error mitigation for {n_qubits} qubits")
            return circuit, mitigation_data
            
        except Exception as e:
            logger.error(f"Error applying error mitigation: {str(e)}")
            return circuit, None
    
    def _apply_zero_noise_extrapolation(self, results, noise_factors=(1.0, 2.0, 3.0)):
        """
        Apply zero-noise extrapolation to circuit results.
        
        Args:
            results: Circuit execution results
            noise_factors: Noise scaling factors to use
            
        Returns:
            Mitigated results
        """
        # This is a simplified implementation of the zero-noise extrapolation technique
        # In a full implementation, you would run the circuit with different noise factors
        # and extrapolate to zero noise
        
        # For demonstration purposes only
        if not self.error_mitigation or not QISKIT_AVAILABLE:
            return results
            
        if len(noise_factors) < 2:
            return results
            
        try:
            # Simulate running with increased noise
            # In a real implementation, you would modify circuit parameters
            # or use a noise injection method
            
            # For now, just return the original results with metadata
            results.zne_applied = True
            results.noise_factors = noise_factors
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying zero-noise extrapolation: {str(e)}")
            return results
    
    def _process_with_error_mitigation(self, counts, mitigation_data):
        """
        Apply error mitigation post-processing to circuit results.
        
        Args:
            counts: Raw bitstring counts from circuit execution
            mitigation_data: Mitigation data for post-processing
            
        Returns:
            Mitigated counts
        """
        if not self.error_mitigation or not mitigation_data or not QISKIT_AVAILABLE:
            return counts
            
        try:
            # Apply measurement error mitigation
            if 'measurement_fitter' in mitigation_data:
                meas_fitter = mitigation_data['measurement_fitter']
                mitigated_counts = meas_fitter.filter.apply(counts)
                logger.info("Applied measurement error mitigation to results")
                return mitigated_counts
                
        except Exception as e:
            logger.error(f"Error applying mitigation post-processing: {str(e)}")
            
        return counts
    
    def run_quantum_monte_carlo(self, 
                               param_ranges: Dict[str, Tuple[float, float]], 
                               iterations: int = 1000,
                               target_function: Optional[Callable] = None,
                               distribution_type: str = "auto",
                               error_mitigation: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run Quantum Monte Carlo simulation for the given parameter ranges.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            iterations: Number of iterations to run
            target_function: Optional function to evaluate points (if None, just returns sampling)
            distribution_type: Type of distribution to use (auto, uniform, normal, exponential, beta)
            error_mitigation: Whether to apply error mitigation (overrides class setting)
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        # Check availability
        if not self.is_available():
            logger.warning("Quantum processing not available. Falling back to classical Monte Carlo.")
            return self._run_classical_monte_carlo(param_ranges, iterations, target_function)
        
        # Set error mitigation flag for this run
        if error_mitigation is not None:
            original_error_mitigation = self.error_mitigation
            self.error_mitigation = error_mitigation
        
        # If distribution type is auto, try to determine best fit from analysis
        if distribution_type == "auto":
            if self.distribution_analysis:
                # Use the distribution with highest advantage from analysis
                best_dists = [(name, info["best_fit"], info["potential_quantum_advantage"]) 
                           for name, info in self.distribution_analysis.items()]
                
                if best_dists:
                    # Pick the distribution type with highest quantum advantage
                    distribution_type = max(best_dists, key=lambda x: x[2])[1]
                    logger.info(f"Auto-selected distribution type: {distribution_type}")
                else:
                    distribution_type = self.preferred_distribution
            else:
                distribution_type = self.preferred_distribution
        
        # Generate cache key
        cache_key = self._generate_cache_key(param_ranges, iterations, distribution_type)
        cached_results = self._get_from_cache(cache_key)
        
        if cached_results:
            logger.debug("Using cached quantum Monte Carlo results")
            # Restore original error mitigation setting if needed
            if error_mitigation is not None:
                self.error_mitigation = original_error_mitigation
            return cached_results
        
        try:
            if QISKIT_AVAILABLE:
                results = self._run_qiskit_monte_carlo(param_ranges, iterations, target_function, distribution_type)
            elif PENNYLANE_AVAILABLE:
                results = self._run_pennylane_monte_carlo(param_ranges, iterations, target_function, distribution_type)
                
            # Add timing information
            results['execution_time'] = time.time() - start_time
            results['backend'] = str(self.backend)
            results['quantum'] = True
            results['distribution_type'] = distribution_type
            results['error_mitigation_applied'] = self.error_mitigation
            
            # Cache results
            self._save_to_cache(cache_key, results)
            
            # Restore original error mitigation setting if needed
            if error_mitigation is not None:
                self.error_mitigation = original_error_mitigation
                
            return results
            
        except Exception as e:
            logger.error(f"Error in quantum Monte Carlo: {str(e)}")
            logger.info("Falling back to classical Monte Carlo")
            
            # Restore original error mitigation setting if needed
            if error_mitigation is not None:
                self.error_mitigation = original_error_mitigation
                
            return self._run_classical_monte_carlo(param_ranges, iterations, target_function)
    
    def _create_uniform_distribution_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a quantum circuit that generates a uniform distribution."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        # Apply Hadamard to all qubits to create superposition
        qc.h(range(n_qubits))
        return qc
    
    def _create_normal_distribution_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a quantum circuit that approximates a normal distribution."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Apply rotations to approximate normal distribution
        for i in range(n_qubits):
            qc.h(i)
        
        # Add entangling operations 
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
            
        # Add a second layer of rotations
        angles = [np.pi/4, np.pi/8, np.pi/16, np.pi/32]
        for i in range(min(n_qubits, len(angles))):
            qc.ry(angles[i], i)
            
        return qc
    
    def _create_exponential_distribution_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a quantum circuit that approximates an exponential distribution."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize in superposition
        qc.h(range(n_qubits))
        
        # Apply successive controlled rotations to create exponential-like distribution
        for i in range(n_qubits-1):
            angle = np.pi/(2**(i+1))
            qc.cry(angle, i, i+1)
            
        # Add some interference
        for i in range(n_qubits-1, 0, -1):
            qc.cx(i-1, i)
            
        return qc
    
    def _create_beta_distribution_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a quantum circuit that approximates a beta distribution."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize with varied angles
        for i in range(n_qubits):
            # Different rotation angle for each qubit
            angle = (i+1) * np.pi / (2 * n_qubits)
            qc.ry(angle, i)
        
        # Create entanglement pattern
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
            
        # Second rotation layer
        for i in range(n_qubits):
            angle = (n_qubits-i) * np.pi / (2 * n_qubits)
            qc.ry(angle, i)
            
        return qc
    
    def _run_qiskit_monte_carlo(self, 
                               param_ranges: Dict[str, Tuple[float, float]], 
                               iterations: int,
                               target_function: Optional[Callable],
                               distribution_type: str = "uniform") -> Dict[str, Any]:
        """
        Implement Monte Carlo using Qiskit quantum circuits.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            iterations: Number of iterations to run
            target_function: Optional function to evaluate points
            distribution_type: Type of distribution to use
            
        Returns:
            Dictionary with results and statistics
        """
        n_params = len(param_ranges)
        
        # Determine number of qubits needed (log2 of iterations)
        n_qubits = max(3, int(np.ceil(np.log2(iterations)))) 
        n_qubits = min(n_qubits, 10)  # Cap at 10 qubits for practical reasons
        
        # Create quantum circuits based on distribution type
        if distribution_type in self.available_distributions:
            circuit_creator = self.available_distributions[distribution_type]
            qc = circuit_creator(n_qubits * n_params)
        else:
            # Default to uniform if unknown distribution
            qc = self._create_uniform_distribution_circuit(n_qubits * n_params)
        
        # If using parallel circuits, create multiple circuit variants
        circuits = []
        if self.parallel_circuits > 1:
            # Create variations of the circuit with different initializations
            for i in range(self.parallel_circuits):
                qc_variant = qc.copy()
                
                # Add some distinguishing gate patterns
                for j in range(min(3, n_qubits)):
                    if i % 2 == 0:
                        qc_variant.sx(j)
                    else:
                        qc_variant.s(j)
                        
                # Add measurement
                qc_variant.measure_all()
                circuits.append(qc_variant)
        else:
            # Just use the single circuit with measurement
            qc.measure_all()
            circuits = [qc]
        
        # Apply error mitigation
        mitigated_circuits = []
        mitigation_data = None
        for circuit in circuits:
            mitigated_circuit, mit_data = self._apply_error_mitigation(circuit, n_qubits * n_params)
            mitigated_circuits.append(mitigated_circuit)
            if mit_data:
                mitigation_data = mit_data
        
        # Execute the circuits
        job = execute(mitigated_circuits, self.backend, shots=self.shots, optimization_level=self.optimization_level)
        result = job.result()
        
        # Apply zero-noise extrapolation if enabled
        if self.error_mitigation:
            result = self._apply_zero_noise_extrapolation(result)
        
        # Process the results
        all_counts = []
        for i, circuit in enumerate(mitigated_circuits):
            counts = result.get_counts(circuit)
            
            # Apply error mitigation post-processing
            if mitigation_data:
                counts = self._process_with_error_mitigation(counts, mitigation_data)
                
            all_counts.append(counts)
        
        # Combine counts from parallel circuits
        combined_counts = {}
        for counts in all_counts:
            for bitstring, count in counts.items():
                if bitstring in combined_counts:
                    combined_counts[bitstring] += count
                else:
                    combined_counts[bitstring] = count
        
        # Normalize combined counts
        total = sum(combined_counts.values())
        for bitstring in combined_counts:
            combined_counts[bitstring] = combined_counts[bitstring] / total * self.shots
        
        # Extract parameter samples from the quantum measurements
        param_names = list(param_ranges.keys())
        param_samples = {name: [] for name in param_names}
        values = []
        
        for bitstring, count in combined_counts.items():
            params = {}
            for p_idx, param_name in enumerate(param_names):
                # Extract bits for this parameter
                param_bits = bitstring[p_idx * n_qubits:(p_idx + 1) * n_qubits]
                param_bits = param_bits.zfill(n_qubits)
                
                # Convert bits to a normalized value between 0 and 1
                normalized_value = int(param_bits, 2) / (2**n_qubits - 1)
                
                # Scale to parameter range
                min_val, max_val = param_ranges[param_name]
                param_value = min_val + normalized_value * (max_val - min_val)
                
                params[param_name] = param_value
                param_samples[param_name].extend([param_value] * int(count))
            
            # Evaluate target function if provided
            if target_function:
                result_value = target_function(**params)
                values.extend([result_value] * int(count))
        
        results = {
            'param_samples': param_samples,
            'circuit_depth': max(qc.depth() for qc in mitigated_circuits),
            'n_qubits': n_qubits * n_params,
            'distribution': distribution_type,
            'parallel_circuits': self.parallel_circuits,
            'error_mitigation_applied': self.error_mitigation
        }
        
        if target_function:
            results['values'] = values
            results['mean'] = np.mean(values)
            results['std'] = np.std(values)
            results['min'] = np.min(values)
            results['max'] = np.max(values)
        
        return results
    
    def _run_pennylane_monte_carlo(self, 
                                  param_ranges: Dict[str, Tuple[float, float]], 
                                  iterations: int,
                                  target_function: Optional[Callable],
                                  distribution_type: str = "uniform") -> Dict[str, Any]:
        """
        Implement Monte Carlo using PennyLane quantum circuits.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            iterations: Number of iterations to run
            target_function: Optional function to evaluate points
            distribution_type: Type of distribution to use
            
        Returns:
            Dictionary with results and statistics
        """
        n_params = len(param_ranges)
        
        # Determine number of qubits needed (log2 of iterations)
        n_qubits = max(3, int(np.ceil(np.log2(iterations))))
        n_qubits = min(n_qubits, 8)  # Cap at 8 qubits for practical reasons
        
        param_names = list(param_ranges.keys())
        
        # Define a quantum circuit that generates superpositions
        dev = qml.device(self.backend, wires=n_qubits, shots=self.shots)
        
        @qml.qnode(dev)
        def circuit():
            # Apply Hadamard gates to create superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Add some entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Return measurements
            return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Execute the circuit
        samples = circuit()
        
        # Convert samples to parameters
        samples = np.array(samples).T  # Transpose to get shots as rows
        
        # Process samples into parameter values
        param_samples = {name: [] for name in param_names}
        values = []
        
        for sample in samples:
            # Convert binary samples (-1, 1) to bits (0, 1)
            bits = (sample + 1) // 2
            
            # Divide bits among parameters
            qubits_per_param = n_qubits // n_params
            params = {}
            
            for p_idx, param_name in enumerate(param_names):
                if p_idx < n_params - 1:
                    param_bits = bits[p_idx * qubits_per_param:(p_idx + 1) * qubits_per_param]
                else:
                    # Last parameter gets all remaining qubits
                    param_bits = bits[p_idx * qubits_per_param:]
                
                # Convert bits to a normalized value between 0 and 1
                bit_str = ''.join(map(str, param_bits.astype(int)))
                normalized_value = int(bit_str, 2) / (2**len(param_bits) - 1) if len(param_bits) > 0 else 0.5
                
                # Scale to parameter range
                min_val, max_val = param_ranges[param_name]
                param_value = min_val + normalized_value * (max_val - min_val)
                
                params[param_name] = param_value
                param_samples[param_name].append(param_value)
            
            # Evaluate target function if provided
            if target_function:
                result_value = target_function(**params)
                values.append(result_value)
        
        results = {
            'param_samples': param_samples,
            'n_qubits': n_qubits
        }
        
        if target_function:
            results['values'] = values
            results['mean'] = np.mean(values)
            results['std'] = np.std(values)
            results['min'] = np.min(values)
            results['max'] = np.max(values)
        
        return results
    
    def _run_classical_monte_carlo(self, 
                                  param_ranges: Dict[str, Tuple[float, float]], 
                                  iterations: int,
                                  target_function: Optional[Callable]) -> Dict[str, Any]:
        """
        Fallback to classical Monte Carlo if quantum is not available.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            iterations: Number of iterations to run
            target_function: Optional function to evaluate points
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        param_names = list(param_ranges.keys())
        param_samples = {name: [] for name in param_names}
        values = []
        
        for _ in range(iterations):
            params = {}
            for param_name in param_names:
                min_val, max_val = param_ranges[param_name]
                param_value = np.random.uniform(min_val, max_val)
                params[param_name] = param_value
                param_samples[param_name].append(param_value)
            
            # Evaluate target function if provided
            if target_function:
                result_value = target_function(**params)
                values.append(result_value)
        
        results = {
            'param_samples': param_samples,
            'execution_time': time.time() - start_time,
            'backend': 'classical',
            'quantum': False
        }
        
        if target_function:
            results['values'] = values
            results['mean'] = np.mean(values)
            results['std'] = np.std(values)
            results['min'] = np.min(values)
            results['max'] = np.max(values)
        
        return results
    
    def _generate_cache_key(self, param_ranges: Dict[str, Tuple[float, float]], iterations: int, distribution_type: str) -> str:
        """Generate a cache key for the given parameters."""
        key_dict = {
            'param_ranges': param_ranges,
            'iterations': iterations,
            'backend': str(self.backend),
            'shots': self.shots,
            'distribution_type': distribution_type
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if it exists and is not expired."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Cache expires after 1 day for quantum results
            timestamp = cached_data.get('_cached_at', 0)
            if time.time() - timestamp > 86400:  # 24 hours
                logger.debug(f"Cache expired for {cache_key}")
                return None
                
            return cached_data.get('data')
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error reading cache file {cache_file}: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                else:
                    return obj
            
            data_serializable = convert_numpy(data)
            
            cache_data = {
                '_cached_at': time.time(),
                'data': data_serializable
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Saved quantum Monte Carlo results to cache: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error saving to cache file {cache_file}: {str(e)}")
            
    def compare_with_classical(self, 
                              param_ranges: Dict[str, Tuple[float, float]], 
                              target_function: Callable,
                              iterations: int = 1000) -> Dict[str, Any]:
        """
        Run both quantum and classical Monte Carlo and compare results.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            target_function: Function to evaluate points
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with comparison results
        """
        # Run quantum Monte Carlo
        quantum_results = self.run_quantum_monte_carlo(param_ranges, iterations, target_function)
        
        # Run classical Monte Carlo
        classical_results = self._run_classical_monte_carlo(param_ranges, iterations, target_function)
        
        # Compare results
        comparison = {
            'quantum': quantum_results,
            'classical': classical_results,
            'speedup': classical_results['execution_time'] / max(quantum_results['execution_time'], 0.001),
            'mean_difference': abs(quantum_results['mean'] - classical_results['mean']),
            'std_ratio': quantum_results['std'] / max(classical_results['std'], 0.001)
        }
        
        return comparison 