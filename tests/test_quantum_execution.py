"""
Tests for real quantum circuit execution in quantum_modules.py.

Validates that all quantum module functions execute actual quantum circuits
via Qiskit Aer / PennyLane rather than returning random/hardcoded data.
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Core algorithm tests
# ---------------------------------------------------------------------------

class TestCoreQuantumAlgorithms:
    """Test that each core algorithm executes real quantum circuits."""

    def test_qaoa_optimization(self):
        from backend.engine.quantum_modules import run_qaoa_optimization
        result = run_qaoa_optimization({"n_qubits": 4, "p_layers": 2, "max_iterations": 50})
        assert result.success is True
        assert result.used_quantum is True
        assert result.algorithm == "qaoa"
        assert result.metrics["qubits_used"] > 0
        assert result.metrics["circuit_depth"] > 0
        assert result.metrics["gate_count"] > 0
        assert result.qasm_circuit is not None
        assert "OPENQASM" in result.qasm_circuit
        assert "best_cut" in result.result
        assert "cut_value" in result.result

    def test_quantum_sensing(self):
        from backend.engine.quantum_modules import run_quantum_sensing
        result = run_quantum_sensing({"n_qubits": 4, "true_parameter": 0.5, "num_shots": 500})
        assert result.success is True
        assert result.used_quantum is True
        assert result.algorithm == "quantum_sensing"
        assert result.metrics["qubits_used"] > 0
        assert result.metrics["circuit_depth"] > 0
        assert result.qasm_circuit is not None
        assert "OPENQASM" in result.qasm_circuit
        assert "measured_value" in result.result
        assert "precision" in result.result

    def test_tensor_network(self):
        from backend.engine.quantum_modules import run_tensor_network
        result = run_tensor_network({"n_qubits": 4, "circuit_depth": 6})
        assert result.success is True
        assert result.used_quantum is True
        assert result.algorithm == "tree_tensor_network"
        assert result.metrics["qubits_used"] > 0
        assert result.metrics["circuit_depth"] > 0
        assert result.qasm_circuit is not None
        assert "OPENQASM" in result.qasm_circuit
        assert "fidelity" in result.result

    def test_neural_quantum(self):
        from backend.engine.quantum_modules import run_neural_quantum
        result = run_neural_quantum({"n_qubits": 4, "num_steps": 50})
        assert result.success is True
        assert result.used_quantum is True
        assert result.algorithm == "neural_quantum_digital_twin"
        assert result.metrics["qubits_used"] > 0
        assert result.metrics["circuit_depth"] > 0
        assert result.qasm_circuit is not None
        assert "OPENQASM" in result.qasm_circuit
        assert "solution" in result.result

    def test_pennylane_ml(self):
        from backend.engine.quantum_modules import run_pennylane_ml
        result = run_pennylane_ml({"n_qubits": 4, "n_layers": 2, "num_epochs": 5})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.metrics["circuit_depth"] > 0
        assert result.qasm_circuit is not None
        assert "OPENQASM" in result.qasm_circuit
        assert "accuracy" in result.result
        assert "final_loss" in result.result

    def test_qaoa_with_custom_edges(self):
        from backend.engine.quantum_modules import run_qaoa_optimization
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        result = run_qaoa_optimization({"n_qubits": 4, "p_layers": 1, "edges": edges})
        assert result.success is True
        assert result.used_quantum is True
        assert result.result["cut_value"] >= 0


# ---------------------------------------------------------------------------
# Healthcare wrapper tests
# ---------------------------------------------------------------------------

class TestHealthcareWrappers:
    """Test that healthcare wrappers delegate to real quantum execution."""

    def test_personalized_medicine(self):
        from backend.engine.quantum_modules import run_personalized_medicine
        result = run_personalized_medicine({"patient_data": {"age": 55}})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None

    def test_drug_discovery(self):
        from backend.engine.quantum_modules import run_drug_discovery
        result = run_drug_discovery({"num_candidates": 5})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None
        assert "top_candidates" in result.result

    def test_medical_imaging(self):
        from backend.engine.quantum_modules import run_medical_imaging
        result = run_medical_imaging({})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None

    def test_genomic_analysis(self):
        from backend.engine.quantum_modules import run_genomic_analysis
        result = run_genomic_analysis({"n_genes": 50})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None

    def test_epidemic_modeling(self):
        from backend.engine.quantum_modules import run_epidemic_modeling
        result = run_epidemic_modeling({"population": 10000})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None

    def test_hospital_operations(self):
        from backend.engine.quantum_modules import run_hospital_operations
        result = run_hospital_operations({"n_patients": 8})
        assert result.success is True
        assert result.used_quantum is True
        assert "classical_fallback" not in result.algorithm
        assert result.metrics["qubits_used"] > 0
        assert result.qasm_circuit is not None


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the QuantumModuleRegistry with real quantum execution."""

    def test_registry_modules_available(self):
        from backend.engine.quantum_modules import registry
        assert len(registry.available_modules) == 11
        assert "qaoa" in registry.available_modules
        assert "personalized_medicine" in registry.available_modules

    def test_registry_quantum_ready(self):
        from backend.engine.quantum_modules import registry
        # With Qiskit Aer installed, all should be quantum-ready
        assert len(registry.quantum_ready_modules) > 0

    def test_registry_run(self):
        from backend.engine.quantum_modules import registry
        result = registry.run("qaoa", {"n_qubits": 4, "p_layers": 1})
        assert result.success is True
        assert result.used_quantum is True


# ---------------------------------------------------------------------------
# Statistics module tests
# ---------------------------------------------------------------------------

class TestStatistics:
    """Test the statistical validation module."""

    def test_paired_comparison(self):
        from backend.engine.statistics import paired_comparison
        quantum_vals = [0.90, 0.91, 0.89, 0.92, 0.90, 0.88]
        classical_vals = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70]
        result = paired_comparison(quantum_vals, classical_vals, n_comparisons=6)
        assert result.p_value < 0.05
        assert result.cohens_d > 0
        assert result.significant
        assert result.mean_quantum > result.mean_classical
        assert result.ci_lower > 0  # quantum - classical > 0

    def test_cohens_d(self):
        from backend.engine.statistics import compute_cohens_d
        group1 = np.array([0.9, 0.91, 0.89])
        group2 = np.array([0.7, 0.71, 0.69])
        d = compute_cohens_d(group1, group2)
        assert d > 1.0  # Large effect

    def test_confidence_interval(self):
        from backend.engine.statistics import compute_confidence_interval
        data = np.array([0.9, 0.91, 0.89, 0.92, 0.90])
        ci_low, ci_high = compute_confidence_interval(data)
        assert ci_low < np.mean(data) < ci_high

    def test_statistical_result_to_dict(self):
        from backend.engine.statistics import paired_comparison
        result = paired_comparison([0.9, 0.85, 0.88], [0.7, 0.72, 0.68], n_comparisons=6)
        d = result.to_dict()
        assert "t_statistic" in d
        assert "p_value" in d
        assert "cohens_d" in d
        assert "significant" in d


# ---------------------------------------------------------------------------
# Benchmark runner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    """Test the live benchmark runner."""

    def test_run_benchmark_personalized_medicine(self):
        from backend.engine.benchmark_runner import run_benchmark
        report = run_benchmark("personalized_medicine", n_runs=3, seed=42)
        assert report.module_id == "personalized_medicine"
        assert report.n_runs == 3
        assert len(report.quantum_times) == 3
        assert len(report.classical_times) == 3
        assert len(report.quantum_accuracies) == 3
        assert len(report.classical_accuracies) == 3
        assert report.used_quantum is True
        assert report.statistical_result is not None

    def test_benchmark_report_to_dict(self):
        from backend.engine.benchmark_runner import run_benchmark
        report = run_benchmark("personalized_medicine", n_runs=3, seed=42)
        d = report.to_dict()
        assert "module_id" in d
        assert "statistical_result" in d
        assert "used_quantum" in d
        assert d["used_quantum"] is True
