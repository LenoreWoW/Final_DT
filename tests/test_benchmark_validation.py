"""
Tests for benchmark API endpoints and quantum-vs-classical validation.

Validates the /api/benchmark/ endpoints, verifies that both quantum and
classical pipelines succeed for every healthcare module, and checks
statistical consistency over repeated runs.
"""

import pytest

from backend.classical_baselines import run as classical_run
from backend.engine.quantum_modules import registry, QuantumResult


# The six healthcare modules present in the benchmark router
HEALTHCARE_MODULES = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]

# Small parameter sets per module to keep tests fast
SMALL_PARAMS = {
    "personalized_medicine": {},
    "drug_discovery": {"library_size": 10, "num_candidates": 5},
    "medical_imaging": {"num_images": 4},
    "genomic_analysis": {"n_genes": 50, "n_samples": 30},
    "epidemic_modeling": {"population_size": 50, "simulation_days": 10},
    "hospital_operations": {"n_patients": 10},
}

# Small quantum parameter overrides (capped the same way the router does)
SMALL_QUANTUM_PARAMS = {
    "personalized_medicine": {},
    "drug_discovery": {"num_candidates": 5},
    "medical_imaging": {},
    "genomic_analysis": {"n_genes": 8},
    "epidemic_modeling": {"population": 500},
    "hospital_operations": {"n_patients": 5},
}


# =============================================================================
# Benchmark API tests
# =============================================================================


class TestBenchmarkAPI:
    """Tests for the /api/benchmark/ REST endpoints."""

    def test_list_modules_returns_six_or_more(self, client):
        resp = client.get("/api/benchmark/modules")
        assert resp.status_code == 200
        data = resp.json()
        assert "modules" in data
        assert len(data["modules"]) >= 6

    def test_list_modules_structure(self, client):
        resp = client.get("/api/benchmark/modules")
        modules = resp.json()["modules"]
        for mod in modules:
            assert "id" in mod
            assert "name" in mod
            assert "description" in mod

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_get_results_by_module(self, client, module_id):
        resp = client.get(f"/api/benchmark/results/{module_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["module"] == module_id
        assert "classical_time_seconds" in data
        assert "quantum_time_seconds" in data
        assert "speedup" in data

    def test_get_results_unknown_module_404(self, client):
        resp = client.get("/api/benchmark/results/nonexistent")
        assert resp.status_code == 404

    def test_get_all_results(self, client):
        resp = client.get("/api/benchmark/results")
        assert resp.status_code == 200
        data = resp.json()
        assert "benchmarks" in data
        assert "summary" in data
        assert "total_quantum_advantage" in data
        assert len(data["benchmarks"]) >= 6

    def test_methodology_endpoint(self, client):
        resp = client.get("/api/benchmark/methodology")
        assert resp.status_code == 200
        data = resp.json()
        assert "title" in data
        assert "sections" in data
        sections = data["sections"]
        assert "hardware" in sections
        assert "fairness" in sections
        assert "metrics" in sections


# =============================================================================
# Quantum vs Classical cross-validation
# =============================================================================


class TestQuantumVsClassical:
    """
    For each healthcare module, verify that both the quantum registry
    and the classical baselines produce successful results.
    """

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_classical_baseline_succeeds(self, module_id):
        params = SMALL_PARAMS[module_id]
        out = classical_run(module_id, params)
        assert "result" in out
        assert "method" in out
        assert isinstance(out["execution_time"], float)
        assert out["execution_time"] >= 0

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_quantum_module_in_registry(self, module_id):
        """The module is registered (available_modules is a property)."""
        assert module_id in registry.available_modules

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_quantum_module_succeeds(self, module_id):
        params = SMALL_QUANTUM_PARAMS[module_id]
        qr = registry.run(module_id, params)
        assert isinstance(qr, QuantumResult)
        assert qr.success is True, f"{module_id} failed: {qr.error}"
        assert isinstance(qr.result, dict)
        assert isinstance(qr.metrics, dict)
        assert qr.algorithm is not None

    @pytest.mark.parametrize("module_id", HEALTHCARE_MODULES)
    def test_quantum_result_has_qasm(self, module_id):
        """Every quantum module should produce a QASM circuit string."""
        params = SMALL_QUANTUM_PARAMS[module_id]
        qr = registry.run(module_id, params)
        assert qr.qasm_circuit is not None
        # QASM strings start with "OPENQASM" or contain gate operations
        assert len(qr.qasm_circuit) > 0


# =============================================================================
# Statistical consistency
# =============================================================================


class TestStatisticalConsistency:
    """Run a module multiple times and verify consistent success."""

    def test_personalized_medicine_consistent(self):
        """Personalized medicine succeeds 5 out of 5 times."""
        successes = 0
        for _ in range(5):
            out = classical_run("personalized_medicine", {})
            if "result" in out and "method" in out:
                successes += 1
        assert successes == 5

    def test_hospital_operations_consistent(self):
        """Hospital operations succeeds 5 out of 5 times."""
        successes = 0
        for _ in range(5):
            out = classical_run("hospital_operations", {"n_patients": 10})
            if "result" in out and "method" in out:
                successes += 1
        assert successes == 5

    def test_quantum_registry_consistent(self):
        """A quantum module succeeds 5 out of 5 times via registry."""
        successes = 0
        for _ in range(5):
            qr = registry.run("personalized_medicine", {})
            if qr.success:
                successes += 1
        assert successes == 5
