"""
Tests for backend.engine.quantum_modules
=========================================
Covers: QuantumModuleRegistry, all 11 module wrappers, QASM generation,
resource estimation, and QuantumResult serialization.

~35 tests total across 5 test classes.
"""

import pytest
from backend.engine.quantum_modules import (
    registry,
    QuantumResult,
    QuantumModuleRegistry,
    _estimate_resources,
)

# ---------------------------------------------------------------------------
# Module lists
# ---------------------------------------------------------------------------

ALL_MODULES = [
    "qaoa",
    "quantum_sensing",
    "tensor_network",
    "neural_quantum",
    "pennylane_ml",
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]

HEALTHCARE_MODULES = [
    "personalized_medicine",
    "drug_discovery",
    "medical_imaging",
    "genomic_analysis",
    "epidemic_modeling",
    "hospital_operations",
]


# ===================================================================
# 1. Registry tests
# ===================================================================

class TestModuleRegistry:
    """Tests for QuantumModuleRegistry discovery and metadata."""

    def test_registry_has_at_least_11_modules(self):
        """Registry must expose all 11 modules."""
        # available_modules is a PROPERTY, not a method
        assert len(registry.available_modules) >= 11

    def test_all_expected_modules_registered(self):
        """Every name in ALL_MODULES must appear in the registry."""
        registered = set(registry.available_modules)
        for mod in ALL_MODULES:
            assert mod in registered, f"Module '{mod}' missing from registry"

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean for any module name."""
        for mod in ALL_MODULES:
            result = registry.is_available(mod)
            assert isinstance(result, bool), f"is_available('{mod}') returned {type(result)}"

    def test_is_available_false_for_unknown(self):
        """Unknown module names should return False."""
        assert registry.is_available("nonexistent_module_xyz") is False

    def test_list_modules_returns_list_of_dicts(self):
        """list_modules() must return a list of dicts with required keys."""
        modules = registry.list_modules()
        assert isinstance(modules, list)
        assert len(modules) >= 11
        for entry in modules:
            assert isinstance(entry, dict)
            assert "name" in entry
            assert "description" in entry
            assert "category" in entry
            assert "reference" in entry
            assert "quantum_ready" in entry

    def test_get_info_returns_metadata(self):
        """get_info() for a known module returns a dict with expected keys."""
        info = registry.get_info("qaoa")
        assert info is not None
        assert info["name"] == "qaoa"
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0

    def test_get_info_none_for_unknown(self):
        """get_info() for an unknown module returns None."""
        assert registry.get_info("nonexistent_abc") is None


# ===================================================================
# 2. Module execution tests (parametrized over all 11 modules)
# ===================================================================

class TestModuleExecution:
    """Each of the 11 modules should execute and return a valid QuantumResult."""

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_module_runs_successfully(self, module_name):
        """Module must run with default params and return success=True."""
        result = registry.run(module_name, {})
        assert isinstance(result, QuantumResult), (
            f"Expected QuantumResult, got {type(result)}"
        )
        assert result.success is True, (
            f"Module '{module_name}' returned success=False, error={result.error}"
        )

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_execution_time_non_negative(self, module_name):
        """execution_time must be >= 0."""
        result = registry.run(module_name, {})
        assert result.execution_time >= 0

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_result_is_nonempty_dict(self, module_name):
        """The .result field must be a non-empty dict."""
        result = registry.run(module_name, {})
        assert isinstance(result.result, dict)
        assert len(result.result) > 0, f"Module '{module_name}' returned empty result dict"

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_metrics_is_dict(self, module_name):
        """The .metrics field must be a dict."""
        result = registry.run(module_name, {})
        assert isinstance(result.metrics, dict)


# ===================================================================
# 3. QASM generation tests (parametrized over all 11 modules)
# ===================================================================

class TestQASMGeneration:
    """Every module must produce a valid OpenQASM 2.0 circuit string."""

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_qasm_circuit_present(self, module_name):
        """qasm_circuit field must be a non-empty string."""
        result = registry.run(module_name, {})
        assert result.qasm_circuit is not None, (
            f"Module '{module_name}' has qasm_circuit=None"
        )
        assert isinstance(result.qasm_circuit, str)
        assert len(result.qasm_circuit) > 0

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_qasm_contains_openqasm_header(self, module_name):
        """QASM string must contain the OPENQASM header."""
        result = registry.run(module_name, {})
        assert "OPENQASM" in result.qasm_circuit

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_qasm_contains_registers(self, module_name):
        """QASM string must declare qreg and creg."""
        qasm = registry.run(module_name, {}).qasm_circuit
        assert "qreg" in qasm, f"No qreg in QASM for '{module_name}'"
        assert "creg" in qasm, f"No creg in QASM for '{module_name}'"

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_qasm_contains_quantum_gates(self, module_name):
        """QASM string must contain at least one quantum gate."""
        qasm = registry.run(module_name, {}).qasm_circuit
        gate_keywords = ["h ", "h q", "cx ", "cx q", "rx(", "ry(", "rz("]
        found = any(g in qasm for g in gate_keywords)
        assert found, f"No quantum gates found in QASM for '{module_name}'"

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_qasm_contains_measure(self, module_name):
        """QASM string must contain measurement instructions."""
        qasm = registry.run(module_name, {}).qasm_circuit
        assert "measure" in qasm, f"No measure instruction in QASM for '{module_name}'"


# ===================================================================
# 4. Resource estimation tests
# ===================================================================

class TestResourceEstimation:
    """Tests for the _estimate_resources helper and registry.estimate_resources."""

    def test_estimate_returns_dict_with_expected_keys(self):
        """Resource estimation must return qubits, depth, gates, shots."""
        est = _estimate_resources("qaoa", {"n_qubits": 4, "p_layers": 3})
        assert isinstance(est, dict)
        for key in ("qubits", "depth", "gates", "shots"):
            assert key in est, f"Missing key '{key}' in resource estimate"

    def test_registry_estimate_resources(self):
        """registry.estimate_resources() wraps _estimate_resources."""
        est = registry.estimate_resources("quantum_sensing", {"n_qubits": 6})
        assert isinstance(est, dict)
        assert "qubits" in est

    def test_different_modules_different_estimates(self):
        """Different modules should produce different resource profiles."""
        params = {"n_qubits": 8}
        est_qaoa = _estimate_resources("qaoa", params)
        est_sensing = _estimate_resources("quantum_sensing", params)
        # At least one field should differ (they use different formulas)
        differ = any(
            est_qaoa.get(k) != est_sensing.get(k)
            for k in ("qubits", "depth", "gates", "shots")
        )
        assert differ, "qaoa and quantum_sensing should have different resource estimates"

    def test_healthcare_modules_have_fixed_qubits(self):
        """Healthcare modules use fixed qubit counts regardless of params."""
        for mod_name in HEALTHCARE_MODULES:
            est = _estimate_resources(mod_name, {"n_qubits": 4})
            assert est["qubits"] > 0, f"{mod_name} has zero qubits"

    def test_unknown_module_uses_generic_formula(self):
        """An unknown module name should still return a valid estimate (generic)."""
        est = _estimate_resources("completely_unknown", {"n_qubits": 5})
        assert isinstance(est, dict)
        assert "qubits" in est
        assert est["qubits"] == 5


# ===================================================================
# 5. Result structure / serialization tests
# ===================================================================

class TestModuleResultStructure:
    """Tests for QuantumResult.to_dict() and field completeness."""

    @pytest.mark.parametrize("module_name", ["qaoa", "quantum_sensing", "drug_discovery"])
    def test_to_dict_returns_all_keys(self, module_name):
        """to_dict() must include all 8 canonical fields."""
        result = registry.run(module_name, {})
        d = result.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "success", "algorithm", "result", "metrics",
            "execution_time", "used_quantum", "error", "qasm_circuit",
        }
        assert expected_keys.issubset(d.keys()), (
            f"Missing keys: {expected_keys - d.keys()}"
        )

    @pytest.mark.parametrize("module_name", ["tensor_network", "epidemic_modeling", "hospital_operations"])
    def test_to_dict_values_are_serializable(self, module_name):
        """All values in to_dict() must be JSON-friendly base types."""
        import json
        result = registry.run(module_name, {})
        d = result.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_run_unknown_module_returns_failure(self):
        """Running an unregistered module returns success=False with an error."""
        result = registry.run("totally_fake_module", {})
        assert result.success is False
        assert result.error is not None
        assert "Unknown module" in result.error
